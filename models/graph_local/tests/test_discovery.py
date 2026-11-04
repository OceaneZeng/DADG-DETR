import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.graph_local.discovery import (DiscoveryCriterion, DiscoveryDetector,
                                         DiscoveryFFN, DiscoveryPostProcess, balanced_bce)
from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from models.graph_local.discovery_memory import PrototypeMemory, initialize_from_deltas
from models.graph_local.routed_adapter import ClassAdapter, append_stage
from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.tests.test_routed_adapter import Detector, spec
from util.misc import nested_tensor_from_tensor_list


def args():
    return SimpleNamespace(owod_known_class_ids=[1], discovery_no_transfer=False,
                          discovery_transfer_threshold=.5, discovery_min_support=1.,
                          hidden_dim=4, discovery_capacity=2, graph_adapter_last_n=1,
                          graph_adapter_rank=2, discovery_warmup=0, discovery_top_k=5,
                          discovery_saliency_threshold=.65, discovery_consistency_threshold=.5)


def evidence():
    boxes = torch.tensor([[[.2, .2, .2, .2], [.7, .7, .2, .2], [.85, .15, .1, .1]]])
    features = torch.eye(4)[:3][None]
    heat = torch.zeros(1, 1, 10, 10)
    heat[:, :, 5:9, 5:9] = 1.
    logits = torch.full((1, 3, 7), -8., requires_grad=True)
    outputs = dict(pred_logits=logits, pred_boxes=boxes.clone().requires_grad_(),
                   query_features=features.clone().requires_grad_(), attention_feature=heat,
                   image_sizes=[(100, 100)], padded_size=(100, 100),
                   pred_objectness=torch.zeros(1, 3, requires_grad=True),
                   pred_novelty=torch.zeros(1, 3, requires_grad=True),
                   discovery_routes=[torch.zeros(1, 3, 3, requires_grad=True)],
                   discovery_embedding=features.clone().requires_grad_(),
                   affinities=torch.full((1, 3, 93), -1e4, requires_grad=True))
    teacher = {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in outputs.items()}
    teacher['pred_boxes'][..., 0] = 1 - teacher['pred_boxes'][..., 0]
    teacher['attention_feature'] = heat.flip(-1)
    teacher['pred_objectness'][:] = -8.
    outputs['discovery_teacher'] = teacher
    targets = [dict(labels=torch.tensor([1]), boxes=boxes[0, :1])]
    return outputs, targets


def test_objectness_prevents_background_from_becoming_unknown():
    outputs, _ = evidence()
    outputs['pred_objectness'] = torch.tensor([[-12., 12., 12.]])
    outputs['pred_novelty'] = torch.tensor([[12., 12., -12.]])
    result = DiscoveryPostProcess(known_ids=[1], max_detections=6)(outputs, torch.tensor([[100, 100]]))[0]
    unknown = result['scores'][result['labels'] == 91].sort(descending=True).values
    assert unknown[0] > .99
    assert unknown[1] < 1e-4 and unknown[2] < 1e-4


def test_focal_scoring_preserves_known_sigmoid_without_novelty_veto():
    outputs, _ = evidence()
    outputs['pred_logits'] = torch.zeros(1, 3, 7)
    outputs['pred_objectness'] = torch.zeros(1, 3)
    outputs['pred_novelty'] = torch.full((1, 3), 8.)
    result = DiscoveryPostProcess(known_ids=[1, 2], max_detections=9, scoring='focal')(
        outputs, torch.tensor([[100, 100]]))[0]
    assert torch.allclose(result['scores'][result['labels'] != 91], torch.full((6,), .25))
    assert torch.allclose(result['scores'][result['labels'] == 91], .5 * torch.sigmoid(torch.full((3,), 8.)))


def test_candidates_need_local_saliency_and_view_agreement():
    outputs, targets = evidence()
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    found = mine_candidates(outputs, targets, indices, [1], outputs['discovery_teacher'])
    assert found['pseudo'].tolist() == [[False, True, False]]
    assert found['background'].tolist() == [[False, False, True]]
    missing_teacher = mine_candidates(outputs, targets, indices, [1])
    assert not missing_teacher['pseudo'].any()
    outputs['discovery_teacher']['query_features'].zero_()
    assert not mine_candidates(outputs, targets, indices, [1], outputs['discovery_teacher'])['pseudo'].any()


def test_overconfident_known_logits_cannot_starve_unknown_or_background_supervision():
    outputs, targets = evidence()
    with torch.no_grad():
        outputs['pred_logits'][..., 1] = 12.
        outputs['discovery_teacher']['pred_logits'][..., 1] = 12.
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    found = mine_candidates(outputs, targets, indices, [1], outputs['discovery_teacher'])
    assert found['pseudo'].tolist() == [[False, True, False]]
    assert found['background'].tolist() == [[False, False, True]]
    assert 0 < found['weights'][0, 1] <= 1
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    criterion = DiscoveryCriterion(base, None, args())
    losses = criterion(outputs, targets)
    losses['loss_ce'].backward()
    assert outputs['pred_logits'].grad[0, 1, 1] > 0
    assert outputs['pred_logits'].grad[0, 2, 1] > 0


def test_hierarchical_scores_do_not_double_count_known_and_unknown_objectness():
    outputs, _ = evidence()
    with torch.no_grad():
        outputs['pred_logits'][..., 1] = 12.
        outputs['pred_objectness'][:] = 2.
        outputs['pred_novelty'][:] = 3.
    result = DiscoveryPostProcess(known_ids=[1], max_detections=6)(outputs, torch.tensor([[100, 100]]))[0]
    assert torch.allclose(result['scores'].sum(), outputs['pred_objectness'].sigmoid().sum())
    assert (result['scores'][result['labels'] == 91] > .8).all()


def test_graph_ablation_keeps_parameters_and_scoring_heads_identical():
    from models.graph_local.discovery_memory import AffinityGraph
    graph = AffinityGraph(4)
    ablation = AffinityGraph(4, use_messages=False)
    ablation.load_state_dict(graph.state_dict(), strict=True)
    assert sum(p.numel() for p in graph.parameters()) == sum(p.numel() for p in ablation.parameters())
    memory = PrototypeMemory(4, 2)
    memory.observe(torch.eye(4)[:1], torch.tensor([1]), torch.tensor([1.]))
    features = torch.randn(1, 3, 4)
    a, b = graph(features, memory), ablation(features, memory)
    assert torch.equal(a['affinities'], b['affinities'])
    assert not torch.allclose(a['pred_objectness'], b['pred_objectness'])


def test_unknown_loss_has_real_gradients_and_future_gt_is_not_used():
    outputs, targets = evidence()
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    criterion = DiscoveryCriterion(base, None, args())
    outputs['attention_feature'][:, :, :4, 6:] = .3  # Not trusted background.
    losses = criterion(outputs, targets)
    sum(losses[k] for k in ('loss_objectness', 'loss_unknown', 'loss_discovery_route')).backward()
    assert outputs['pred_novelty'].grad[0, 1] < 0  # A mined unknown is a positive.
    assert outputs['pred_objectness'].grad[0, 1] < 0
    assert criterion.evidence[1].tolist() == [1, -1]
    outputs, targets = evidence()
    criterion.eval()
    first = criterion(outputs, targets)
    extended = [{'labels': torch.tensor([1, 6]), 'boxes': torch.cat([targets[0]['boxes'], outputs['pred_boxes'][0, 1:2]])}]
    second = criterion(outputs, extended)
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert criterion.evidence is None


def test_ambiguous_queries_receive_no_known_background_gradient():
    criterion = SetCriterion(7, None, {}, ['labels'])
    criterion.active_class_ids = [1]
    logits = torch.zeros(1, 3, 7, requires_grad=True)
    outputs = {'pred_logits': logits, 'query_label_weights': torch.tensor([[1., 0., 1.]])}
    targets = [{'labels': torch.tensor([1])}]
    criterion.loss_labels(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))], 1)['loss_ce'].backward()
    assert logits.grad[0, 1].count_nonzero() == 0
    assert logits.grad[0, 2, 1] > 0


def test_discovery_preserves_dense_known_loss_and_rejects_unselected_proposals():
    outputs, targets = evidence()
    # A moderate response is neither accepted unknown nor trusted background.
    outputs['attention_feature'][:, :, :4, 6:] = .3
    outputs['discovery_teacher']['attention_feature'] = outputs['attention_feature'].flip(-1)
    outputs['aux_outputs'] = [dict(pred_logits=outputs['pred_logits'].detach().clone().requires_grad_(),
                                  pred_boxes=outputs['pred_boxes'])]
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    reference = base(outputs, targets)
    criterion = DiscoveryCriterion(base, None, args())
    losses = criterion(outputs, targets)
    assert torch.equal(losses['loss_ce'], reference['loss_ce'])
    assert torch.equal(losses['loss_ce_0'], reference['loss_ce_0'])
    (losses['loss_ce'] + losses['loss_ce_0']).backward(retain_graph=True)
    for logits in (outputs['pred_logits'], outputs['aux_outputs'][0]['pred_logits']):
        assert logits.grad[0, 2, 1] > 0  # Known-class negative still suppresses duplicates.
        assert logits.grad[..., 6].count_nonzero() == 0  # Future class is never supervised.
    losses['loss_objectness'].backward()
    assert outputs['pred_objectness'].grad[0, 2] > 0  # An unselected detection slot.
    assert losses['discovery_known_supervised_fraction'] == 1


def test_novelty_is_conditional_on_selected_foreground_and_router_rejects_other_slots():
    outputs, targets = evidence()
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    losses = DiscoveryCriterion(base, None, args())(outputs, targets)
    (losses['loss_unknown'] + losses['loss_discovery_route']).backward()
    assert outputs['pred_novelty'].grad[0, 0] > 0  # Known object.
    assert outputs['pred_novelty'].grad[0, 1] < 0  # Accepted pseudo unknown.
    assert outputs['pred_novelty'].grad[0, 2] == 0  # No object is not a known object.
    router_grad = outputs['discovery_routes'][0].grad
    assert router_grad[0, 2, 0] < 0  # Reject unselected slot via null/background route.
    assert router_grad[0, 1, 2] < 0


@pytest.mark.parametrize('all_empty', [False, True])
def test_proposal_objectness_has_negative_gradients_without_saliency_background(all_empty):
    from models.graph_local.discovery import proposal_supervision
    positive = torch.tensor([[not all_empty, False, False, False]])
    pseudo = torch.tensor([[False, not all_empty, False, False]])
    confidence = torch.tensor([[0., .4, 0., 0.]])
    foreground, novel_weights, proposal_weights = proposal_supervision(positive, pseudo, confidence)
    logits = torch.full((1, 4), 6., requires_grad=True)
    loss = balanced_bce(logits, foreground.float(), proposal_weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad[~foreground] > 0).all()
    if not all_empty:
        assert (logits.grad[foreground] < 0).all()
        assert novel_weights.tolist() == [[1., pytest.approx(.4), 0., 0.]]
    else:
        novelty = torch.zeros(1, 4, requires_grad=True)
        empty_loss = balanced_bce(novelty, pseudo.float(), novel_weights)
        empty_loss.backward()
        assert empty_loss == 0 and novelty.grad.count_nonzero() == 0


def test_prototypes_are_anonymous_and_refuse_forced_merge_when_full():
    memory = PrototypeMemory(4, capacity=1)
    first = memory.observe(torch.tensor([[1., 0, 0, 0]]), torch.tensor([-1]), torch.tensor([1.]))
    assert first == [0]
    assert memory.counts[91] == 1
    assert memory.observe(torch.tensor([[0., 1, 0, 0]]), torch.tensor([-1]), torch.tensor([1.])) == []
    assert memory.counts[91] == 1
    restored = PrototypeMemory(4, capacity=1)
    restored.load_state_dict(memory.state_dict())
    assert torch.equal(restored.centers, memory.centers)


def test_promotion_inherits_effective_mapping_and_keeps_mixed_prototype():
    model = DiscoveryDetector(Detector(), args())
    feature = torch.tensor([0., 1., 0., 0.])
    model.memory.observe(feature[None], torch.tensor([-1]), torch.tensor([3.]))
    discovery = next(m for m in model.modules() if isinstance(m, DiscoveryFFN))
    nn.init.normal_(discovery.adapter.up.weight)
    discovery.snapshot([0])
    layout = spec(classes=(3,), old=(1, 2))
    append_stage(model, layout)
    report = model.promote({3: feature}, layout)
    assert report[0]['inherited']
    target = model.detr.transformer.decoder.layers[0].graph_adapter.stages[-1].adapters[0]
    expected = discovery.snapshot_up[0] @ discovery.snapshot_down[0]
    assert torch.allclose(target.up.weight @ target.down.weight, expected, atol=1e-6)
    assert model.memory.counts[91] == 3  # No whole-cluster deletion.
    model.transfer_enabled = False
    assert not model.promote({3: feature}, layout)[0]['inherited']


def test_promotion_router_bootstrap_selects_new_class_even_with_cold_classifier():
    model = DiscoveryDetector(Detector(), args())
    feature = torch.tensor([0., 1., 0., 0.])
    layout = spec(classes=(3,), old=(1, 2))
    append_stage(model, layout)
    branch = model.detr.transformer.decoder.layers[0].graph_adapter.stages[-1]
    with torch.no_grad():
        branch.router.weight[0].zero_()
        branch.router.bias[0] = -10.
    assert branch.router(feature).argmax() == 1  # Initial null route starves the adapter.
    model.promote({3: feature}, layout, {3: [feature]})
    branch(feature)
    assert branch.last_gates[0] > .5
    assert branch.router(feature).argmax() == 0


def test_transfer_mixes_effective_deltas_not_individual_factors():
    adapter = ClassAdapter(4, rank=4)
    deltas = [torch.eye(4), torch.eye(4) * 3]
    initialize_from_deltas(adapter, deltas, [.25, .75])
    assert torch.allclose(adapter.up.weight @ adapter.down.weight, torch.eye(4) * 2.5)


def test_ema_startup_removes_random_initial_bias_and_restores_update_count():
    from models.graph_local.discovery import make_ema, update_ema
    student = torch.nn.Linear(2, 2)
    student.register_buffer('support', torch.tensor(0.))
    teacher = make_ema(student)
    with torch.no_grad():
        student.weight.fill_(2.)
        student.support.fill_(3.)
    update_ema(teacher, student)
    assert torch.equal(teacher.weight, student.weight)
    assert teacher.support == 3 and teacher.ema_update_count == 1
    restored = make_ema(student)
    restored.load_state_dict(teacher.state_dict(), strict=True)
    with torch.no_grad():
        student.weight.fill_(4.)
    update_ema(restored, student)
    assert torch.equal(restored.weight, torch.full_like(student.weight, 3.))
    assert restored.ema_update_count == 2


def test_flip_preserves_padding_and_coordinates():
    samples = nested_tensor_from_tensor_list([torch.arange(24.).reshape(3, 2, 4), torch.ones(3, 3, 6)])
    flipped = flip_samples(samples)
    assert torch.equal(flipped.tensors[0, :, :2, :4], samples.tensors[0, :, :2, :4].flip(-1))
    assert torch.equal(flipped.mask, samples.mask)
    assert torch.equal(flipped.tensors[:, :, :, 4:], samples.tensors[:, :, :, 4:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_encoder_checkpoint_preserves_dropout_outputs_and_cuda_gradients():
    from models.deformable_transformer import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
    dense = DeformableTransformerEncoder(DeformableTransformerEncoderLayer(
        d_model=32, d_ffn=64, dropout=.2, n_levels=1, n_heads=4, n_points=2), 2).cuda()
    compact = copy.deepcopy(dense)
    compact.activation_checkpointing = True
    features = torch.randn(1, 6, 32, device='cuda')
    position = torch.randn_like(features)
    target = torch.randn_like(features)
    shapes = torch.tensor([[2, 3]], device='cuda')
    starts = torch.tensor([0], device='cuda')
    ratios = torch.ones(1, 1, 2, device='cuda')
    runs = []
    for encoder in (dense, compact):
        torch.manual_seed(42)
        source = features.detach().clone().requires_grad_(True)
        output = encoder(source, shapes, starts, ratios, position)
        (output * target).sum().backward()
        runs.append((output.detach(), source.grad, [p.grad for p in encoder.parameters()]))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.allclose(runs[0][1], runs[1][1], atol=1e-5)
    for a, b in zip(runs[0][2], runs[1][2]):
        assert a is not None and b is not None
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('revision', [5, 6])
def test_real_deformable_detector_cuda_discovery_and_checkpoint(revision):
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema
    configuration = get_args_parser().parse_args([])
    configuration.discovery_version = revision
    configuration.device = 'cuda'
    configuration.no_pretrained_backbone = True
    configuration.num_classes = 91
    configuration.hidden_dim = 64
    configuration.dim_feedforward = 128
    configuration.enc_layers = 1
    configuration.dec_layers = 2
    configuration.num_queries = 8
    configuration.graph_adapter_last_n = 1
    configuration.owod_known_class_ids = [1, 2]
    configuration.owod_max_detections = 8
    configuration.discovery_warmup = 0
    configuration.graph_routed_adapter = True
    model, criterion, _ = build(configuration)
    model.cuda()
    criterion.cuda()
    teacher = make_ema(model)
    samples = nested_tensor_from_tensor_list([torch.rand(3, 128, 128, device='cuda')])
    outputs = model(samples)
    with torch.no_grad():
        outputs['discovery_teacher'] = teacher(flip_samples(samples))
    targets = [{'labels': torch.tensor([1], device='cuda'),
                'boxes': torch.tensor([[.5, .5, .3, .3]], device='cuda')}]
    losses = criterion(outputs, targets)
    total = sum(value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict)
    total.backward()
    assert torch.isfinite(total)
    discovery = next(m for m in model.modules() if isinstance(m, DiscoveryFFN))
    assert discovery.adapter.up.weight.grad.abs().sum() > 0
    model.observe_training(criterion.evidence)
    assert model.memory.counts[1] == 1
    # Exercise the actual CUDA increment path, including CPU Hungarian indices.
    layout = dict(stage=1, class_ids=[3], old_ids=[1, 2], rank=8, last_n=1,
                  risk_beta=2., risk=[[0., 0.]])
    append_stage(model, layout)
    model.promote({3: outputs['query_features'][0, 0].detach()}, layout,
                  {3: [outputs['discovery_inputs'][0][0, 0]]})
    from models.graph_local.routed_adapter import configure_trainable
    configure_trainable(model, [3], [1, 2, 3], localization='bbox-last')
    criterion.weight_dict['loss_adapter_route'] = configuration.graph_route_loss_coef
    model.train_discovery()
    model.known_ids = criterion.known_ids = criterion.base.active_class_ids = [1, 2, 3]
    criterion.old_ids = [1, 2]
    model.zero_grad(set_to_none=True)
    current = model(samples)
    with torch.no_grad():
        current['discovery_old_teacher'] = teacher(samples)
    new_targets = [{'labels': torch.tensor([3], device='cuda'), 'boxes': targets[0]['boxes']}]
    new_losses = criterion(current, new_targets)
    new_total = sum(value * criterion.weight_dict[key] for key, value in new_losses.items()
                    if key in criterion.weight_dict)
    new_total.backward()
    assert model.detr.bbox_embed[0].layers[-1].weight.grad.abs().sum() > 0
    assert model.detr.bbox_embed[0].layers[0].weight.grad is None
    branch = model.detr.transformer.decoder.layers[-1].graph_adapter.stages[-1]
    assert torch.isfinite(new_total) and branch.router.weight.grad.abs().sum() > 0
    model.eval()
    state = copy.deepcopy(model.state_dict())
    restored, _, _ = build(configuration)
    append_stage(restored, layout)
    restored.cuda().load_state_dict(state, strict=True)
    restored.eval()
    for name, value in state.items():
        assert torch.equal(restored.state_dict()[name], value), name
    with torch.no_grad():
        assert torch.allclose(model(samples)['pred_objectness'], restored(samples)['pred_objectness'], atol=1e-6)
