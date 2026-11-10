"""Numerical and state contracts for selective teachers and batched adapters."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from engine import teacher_prediction
from models.deformable_detr import DeformableDETR, SetCriterion
from models.deformable_transformer import DeformableTransformerDecoderLayer
from models.graph_local.discovery import DiscoveryDetector, DiscoveryCriterion, make_ema, update_ema
from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from models.graph_local.routed_adapter import RiskRoutedStage, append_stage, mask_inactive_adapter_gradients
from models.matcher import HungarianMatcher
from util.misc import NestedTensor
from util.runtime_profile import RuntimeProfile, scope


class SmallBackbone(nn.Module):
    num_channels = [32, 32]
    strides = [1, 2]

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 1)

    def forward(self, samples):
        features = self.conv(samples.tensors)
        values = [NestedTensor(features, samples.mask),
                  NestedTensor(features[:, :, ::2, ::2], samples.mask[:, ::2, ::2])]
        return values, [torch.zeros_like(value.tensors) for value in values]


class SmallTransformer(nn.Module):
    d_model = 32

    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.num_layers = 3
        self.decoder.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(d_model=32, d_ffn=64, dropout=0, n_heads=4)
            for _ in range(3)])

    def forward(self, srcs, masks, pos, queries, **kwargs):
        x = queries[:, :32][None] + srcs[0].mean((2, 3))[:, None]
        hidden, references = [], []
        initial = x.new_full((*x.shape[:-1], 2), .5)
        reference = initial
        for index, layer in enumerate(self.decoder.layers):
            x = layer.forward_ffn(x)
            hidden.append(x)
            # Exercise the decoder's own box-refinement calls as well.
            if self.decoder.bbox_embed is not None:
                reference = self.decoder.bbox_embed[index](x).sigmoid().detach()
            references.append(reference)
        return torch.stack(hidden), initial, torch.stack(references), None, None


def small_discovery(refine=False):
    base = DeformableDETR(SmallBackbone(), SmallTransformer(), 7, 6, 2, with_box_refine=refine)
    args = SimpleNamespace(owod_known_class_ids=[1, 2], discovery_no_transfer=False,
        discovery_transfer_threshold=.8, discovery_min_support=1., hidden_dim=32,
        discovery_capacity=2, graph_adapter_last_n=2, graph_adapter_rank=2,
        discovery_warmup=0, discovery_top_k=5, discovery_saliency_threshold=.1,
        discovery_consistency_threshold=.1)
    model = DiscoveryDetector(base, args)
    append_stage(model, dict(stage=1, class_ids=[2], old_ids=[1], rank=2, last_n=2,
                            risk_beta=2., risk=[[.2]]))
    with torch.no_grad():
        model.memory.centers.normal_()
        model.memory.counts.fill_(3)
        for name, parameter in model.named_parameters():
            if name.endswith('up.weight'):
                parameter.normal_(std=.1)
    weights = dict(loss_ce=2., loss_bbox=5., loss_giou=2., loss_adapter_route=1.)
    for index in range(2):
        weights.update({f'loss_ce_{index}': 2., f'loss_bbox_{index}': 5., f'loss_giou_{index}': 2.})
    criterion = DiscoveryCriterion(SetCriterion(7, HungarianMatcher(), weights,
                                               ['labels', 'boxes'], focal_alpha=.25), model, args)
    return model, criterion


@pytest.mark.parametrize('refine', [False, True])
def test_teachers_keep_consumed_outputs_and_skip_unused_heads(refine):
    torch.manual_seed(21)
    model, _ = small_discovery(refine)
    model.eval().requires_grad_(False)
    samples = NestedTensor(torch.randn(2, 3, 8, 10), torch.zeros(2, 8, 10, dtype=torch.bool))
    original = copy.deepcopy(model.state_dict())
    calls = []
    handle = model.detr.class_embed[-1].register_forward_hook(lambda *args: calls.append(1))
    graph_calls = []
    graph_handle = model.affinity_graph.register_forward_hook(lambda *args: graph_calls.append(1))
    with torch.no_grad():
        full = model(samples)
        full_calls = len(calls)
        calls.clear(); graph_calls.clear()
        old = teacher_prediction(model, samples, 'distillation')
        assert set(old) == {'pred_logits', 'pred_boxes'}
        assert len(calls) == 1 and not graph_calls
        assert full_calls == (1 if refine else 3)
        for key, value in old.items():
            torch.testing.assert_close(value, full[key], rtol=0, atol=0)
        ema = teacher_prediction(model, samples, 'discovery')
        assert 'discovery_embedding' in ema and len(graph_calls) == 1
        assert 'aux_outputs' not in ema and 'affinities' not in ema
        for key, value in ema.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, full[key], rtol=0, atol=0)
            else:
                assert value == full[key]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    for module in model.modules():
        if hasattr(module, 'last_inputs'):
            assert module.last_inputs is None and module.last_logits is None
    handle.remove(); graph_handle.remove()
    model.train()
    with pytest.raises(ValueError, match='eval mode'):
        teacher_prediction(model, samples, 'discovery')


def test_selective_ema_keeps_candidates_loss_gradients_optimizer_and_memory():
    torch.manual_seed(7)
    first, criterion = small_discovery()
    second = copy.deepcopy(first)
    other_criterion = copy.deepcopy(criterion)
    ema = make_ema(first)
    other_ema = copy.deepcopy(ema)
    samples = NestedTensor(torch.randn(1, 3, 8, 10), torch.zeros(1, 8, 10, dtype=torch.bool))
    targets = [dict(labels=torch.tensor([1]), boxes=torch.tensor([[.5, .5, .2, .2]]))]
    full_teacher = ema(flip_samples(samples))
    fast_teacher = teacher_prediction(other_ema, flip_samples(samples), 'discovery')
    for student, loss_fn, teacher in ((first, criterion, full_teacher), (second, other_criterion, fast_teacher)):
        outputs = student(samples)
        indices = loss_fn.matcher(outputs, targets)
        candidates = mine_candidates(outputs, targets, indices, [1, 2], teacher, saliency_threshold=.1,
                                     consistency_threshold=.1)
        if student is first:
            expected_candidates = candidates
        else:
            for key, value in candidates.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, expected_candidates[key], rtol=0, atol=0)
        outputs['discovery_teacher'] = teacher
        losses = loss_fn(outputs, targets)
        if student is first:
            expected_losses = {key: value.detach().clone() for key, value in losses.items()}
        else:
            for key, value in losses.items():
                torch.testing.assert_close(value, expected_losses[key], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        sum(value * loss_fn.weight_dict[key] for key, value in losses.items()
            if key in loss_fn.weight_dict).backward()
        mask_inactive_adapter_gradients(student)
        optimizer.step()
        student.observe_training(loss_fn.evidence)
        update_ema(ema if student is first else other_ema, student)
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for left, right in ((first, second), (ema, other_ema)):
        for name, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('routing', ['hard', 'soft'])
@pytest.mark.parametrize('null', [False, True])
def test_batched_adapter_keeps_gradients_and_adamw_activity(device, routing, null):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    config = dict(class_ids=[3, 4, 5], old_ids=[1, 2], rank=4, risk_beta=2.,
                  risk=[[.1, .8], [.4, .1], [.2, .3]], routing=routing)
    loop = RiskRoutedStage(nn.Linear(32, 7), config).to(device)
    with torch.no_grad():
        for branch in loop.adapters:
            branch.up.weight.normal_(std=.1)
        if null:
            loop.router.weight.zero_()
            loop.router.bias[:] = loop.router.bias.new_tensor([-10., -10., -10., 10.])
    batch = RiskRoutedStage(nn.Linear(32, 7), {**config, 'compute': 'batched'}).to(device)
    batch.load_state_dict(loop.state_dict(), strict=True)
    x = torch.randn(2, 9, 32, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    outputs = [loop(x), batch(y)]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1) for model in (loop, batch)]
    for model, output, optimizer in zip((loop, batch), outputs, optimizers):
        output.square().sum().backward()
        mask_inactive_adapter_gradients(model)
        optimizer.step()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)
    for (name, left), (_, right) in zip(loop.named_parameters(), batch.named_parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)


def test_zero_beta_skips_old_evidence_computation():
    stage = RiskRoutedStage(nn.Linear(4, 5), dict(class_ids=[3], old_ids=[1, 2], rank=2,
        risk_beta=0., risk=[[0., 0.]], risk_mode='disabled'))
    def reject(*args):
        raise AssertionError('Disabled risk must not evaluate old-class evidence')
    stage.old_evidence.register_forward_hook(reject)
    stage(torch.randn(1, 2, 4)).sum().backward()


def test_plain_detector_teacher_keeps_final_outputs_with_a_routed_bank():
    model = DeformableDETR(SmallBackbone(), SmallTransformer(), 7, 6, 2).eval()
    append_stage(model, dict(stage=1, class_ids=[3], old_ids=[1, 2], rank=2, last_n=1,
                            risk_beta=2., risk=[[.1, .2]], compute='batched'))
    samples = NestedTensor(torch.randn(1, 3, 8, 10), torch.zeros(1, 8, 10, dtype=torch.bool))
    with torch.no_grad():
        full = model(samples)
        fast = teacher_prediction(model, samples, 'distillation')
    assert set(fast) == {'pred_logits', 'pred_boxes'}
    for key in fast:
        torch.testing.assert_close(fast[key], full[key], rtol=0, atol=0)


def test_profiler_finishes_window_without_changing_training_or_rng(tmp_path):
    torch.manual_seed(1)
    model = nn.Linear(3, 1)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    expected = torch.optim.SGD(reference.parameters(), lr=.01)
    x = torch.ones(2, 3)
    rng_before = torch.get_rng_state().clone()
    with RuntimeProfile(tmp_path, 'cpu', warmup=1, steps=2) as profile:
        for _ in range(5):
            optimizer.zero_grad()
            with scope('student_forward'):
                loss = model(x).square().sum()
            with scope('backward'):
                loss.backward()
            optimizer.step()
            profile.step()
    for _ in range(5):
        expected.zero_grad()
        reference(x).square().sum().backward()
        expected.step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    summary = json.loads(next(tmp_path.glob('profiles/*/summary.json')).read_text())
    assert summary['regions']['owod/student_forward']['calls'] == 2
    assert summary['active_steps'] == 2
    assert profile.profiler is None
    assert next(tmp_path.glob('profiles/*/trace.json')).stat().st_size > 0


def test_interrupted_profile_does_not_claim_a_complete_window(tmp_path):
    with RuntimeProfile(tmp_path, 'cpu', warmup=1, steps=5) as profile:
        for _ in range(2):
            with scope('student_forward'):
                torch.ones(2).square()
            profile.step()
    summary = json.loads(next(tmp_path.glob('profiles/*/summary.json')).read_text())
    assert summary['active_steps'] == 1
    assert summary['requested_active_steps'] == 5
    assert summary['window_complete'] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('refine', [False, True])
def test_real_cuda_detector_selective_teachers_and_increment_update(refine):
    from main import get_args_parser
    from models.graph_local.discovery import build
    from models.graph_local.routed_adapter import configure_trainable
    from util.misc import nested_tensor_from_tensor_list
    configuration = get_args_parser().parse_args([])
    configuration.no_pretrained_backbone = True
    configuration.num_classes = 91
    configuration.hidden_dim = 64
    configuration.dim_feedforward = 128
    configuration.enc_layers = 1
    configuration.dec_layers = 2
    configuration.num_queries = 8
    configuration.graph_adapter_last_n = 1
    configuration.owod_known_class_ids = [1, 2, 3]
    configuration.owod_max_detections = 8
    configuration.discovery_warmup = 0
    configuration.with_box_refine = refine
    model, criterion, _ = build(configuration)
    model.cuda(); criterion.cuda()
    append_stage(model, dict(stage=1, class_ids=[3], old_ids=[1, 2], rank=8, last_n=1,
                            risk_beta=2., risk=[[.2, .3]], compute='batched'))
    configure_trainable(model, [3], [1, 2, 3])
    model.train_discovery()
    model.eval()
    teacher = make_ema(model)
    samples = nested_tensor_from_tensor_list([torch.rand(3, 128, 128, device='cuda')])
    with torch.no_grad():
        full = teacher(samples)
        for purpose in ('distillation', 'discovery'):
            fast = teacher_prediction(teacher, samples, purpose)
            for key, value in fast.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, full[key], rtol=0, atol=0)
    model.train()
    criterion.train()
    outputs = model(samples)
    outputs['discovery_teacher'] = teacher_prediction(teacher, flip_samples(samples), 'discovery')
    target = [dict(labels=torch.tensor([3], device='cuda'),
                   boxes=torch.tensor([[.5, .5, .3, .3]], device='cuda'))]
    losses = criterion(outputs, target)
    total = sum(value * criterion.weight_dict[key] for key, value in losses.items()
                if key in criterion.weight_dict)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    total.backward()
    mask_inactive_adapter_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    model.observe_training(criterion.evidence)
    update_ema(teacher, model)
    assert torch.isfinite(total) and teacher.ema_update_count.item() == 1
    assert model.memory.counts[3] == teacher.memory.counts[3] == 1
