"""Check actual score ordering, released-GT isolation and gradient boundaries."""
import copy

import pytest
import torch

from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.discovery import (DiscoveryCriterion, DiscoveryPostProcess,
    known_unknown_log_ratio, matched_known_ranking_loss, scale_gradient)
from models.graph_local.discovery_memory import AffinityGraph, PrototypeMemory
from models.graph_local.tests.test_discovery import args, evidence


def criterion(rank_coef=1., grad_scale=1., scoring='hierarchical'):
    configuration = args()
    configuration.owod_known_class_ids = [1, 4]
    configuration.discovery_known_rank_coef = rank_coef
    configuration.discovery_feature_grad_scale = grad_scale
    configuration.discovery_scoring = scoring
    configuration.discovery_pseudo_normalization = 'count'
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = configuration.owod_known_class_ids
    return DiscoveryCriterion(base, None, configuration)


@pytest.mark.parametrize('scoring', ['hierarchical', 'focal'])
def test_ranking_uses_actual_gt_category_scores_and_improves_their_order(scoring):
    outputs, _ = evidence()
    outputs = {k: outputs[k] for k in ('pred_logits', 'pred_novelty', 'pred_objectness', 'pred_boxes')}
    outputs['pred_logits'] = torch.tensor([[[0., -1., 0., 0., 2., 0., 0.]]], requires_grad=True)
    outputs['pred_novelty'] = torch.tensor([[1.]], requires_grad=True)
    outputs['pred_objectness'] = torch.tensor([[2.]])
    outputs['pred_boxes'] = outputs['pred_boxes'][:, :1]
    known, labels = [1, 4], torch.tensor([1])
    gap = known_unknown_log_ratio(outputs['pred_logits'][0], outputs['pred_novelty'][0], labels, known, scoring)
    prediction = DiscoveryPostProcess(known_ids=known, max_detections=3, scoring=scoring)(
        outputs, torch.tensor([[100, 100]]))[0]
    actual_known = prediction['scores'][prediction['labels'] == 1][0]
    actual_unknown = prediction['scores'][prediction['labels'] == 91][0]
    torch.testing.assert_close(gap[0], (actual_unknown / actual_known).log())
    known_grad, novelty_grad = torch.autograd.grad(matched_known_ranking_loss(gap),
        (outputs['pred_logits'], outputs['pred_novelty']))
    assert known_grad[0, 0, 1] < 0 and novelty_grad[0, 0] > 0
    assert known_grad[0, 0, [0, 2, 3, 5, 6]].count_nonzero() == 0
    improved = known_unknown_log_ratio((outputs['pred_logits'] - .1 * known_grad)[0],
        (outputs['pred_novelty'] - .1 * novelty_grad)[0], labels, known, scoring)
    assert improved[0] < gap[0]


@pytest.mark.parametrize('scoring', ['hierarchical', 'focal'])
def test_extreme_logits_and_empty_gt_have_finite_ranking_gradients(scoring):
    logits = torch.tensor([[-1000., 1000., -1000., -1000., -1000.]], requires_grad=True)
    novelty = torch.tensor([1000.], requires_grad=True)
    loss = matched_known_ranking_loss(known_unknown_log_ratio(logits, novelty, torch.tensor([4]), [1, 4], scoring))
    gradients = torch.autograd.grad(loss, (logits, novelty))
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
    empty = logits[:0]
    loss = matched_known_ranking_loss(known_unknown_log_ratio(empty, novelty[:0], torch.empty(0, dtype=torch.long), [1, 4], scoring))
    loss.backward()
    assert loss == 0 and logits.grad.count_nonzero() == novelty.grad.count_nonzero() == 0


@pytest.mark.parametrize('counts', [(3, 0), (1, 0), (0, 0), (1, 2)])
def test_rank_reduction_matches_global_objective_with_uneven_or_empty_ranks(monkeypatch, counts):
    # Deterministic collective substitute checks the objective and gradients;
    # test_runtime_distributed.py separately exercises real Gloo collectives.
    import models.graph_local.discovery as discovery
    observed_counts = []

    def reduce_count(value):
        observed_counts.append(int(value.item()))
        value.fill_(sum(counts))

    monkeypatch.setattr(discovery, 'is_dist_avail_and_initialized', lambda: True)
    monkeypatch.setattr(discovery, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce_count)
    joined = torch.linspace(-3, 3, sum(counts), requires_grad=True)
    reference = torch.nn.functional.softplus(joined).sum() / max(sum(counts), 1)
    reference_gradient = torch.autograd.grad(reference, joined)[0]
    losses, gradients = [], []
    for rank, count in enumerate(counts):
        start = sum(counts[:rank])
        local = joined.detach()[start:start + count].clone().requires_grad_(True)
        loss = matched_known_ranking_loss(local)
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, local)[0])
    assert observed_counts == list(counts)
    torch.testing.assert_close(torch.stack(losses).mean(), reference.detach())
    torch.testing.assert_close(torch.cat(gradients) / 2, reference_gradient)


def test_rank_adds_only_released_known_constraint_and_keeps_unknown_learning():
    outputs, targets = evidence()
    legacy, ranked = criterion(0.), criterion()
    reference = legacy(outputs, targets)
    actual = ranked(copy.deepcopy(outputs), targets)
    for name in reference:
        torch.testing.assert_close(reference[name], actual[name], rtol=0, atol=0)
    for a, b in zip(legacy.evidence, ranked.evidence):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual['discovery_known_count'] == 1
    assert actual['discovery_known_rank_violations'] == 1
    loss = ranked(outputs, targets)['loss_discovery_rank']
    gradient = torch.autograd.grad(loss, outputs['pred_novelty'], retain_graph=True)[0]
    assert gradient[0, 0] > 0 and gradient[0, 1:].count_nonzero() == 0
    fresh, targets = evidence()
    unknown_loss = ranked(fresh, targets)['loss_unknown']
    gradient = torch.autograd.grad(unknown_loss, fresh['pred_novelty'])[0]
    assert gradient[0, 1] < 0 and gradient[0, 2] == 0
    future = [dict(labels=torch.cat([targets[0]['labels'], torch.tensor([6])]),
        boxes=torch.cat([targets[0]['boxes'], fresh['pred_boxes'][0, 1:2].detach()]))]
    hidden = ranked(fresh, future)
    released = ranked(fresh, targets)
    for name in released:
        torch.testing.assert_close(hidden[name], released[name], rtol=0, atol=0)


@pytest.mark.parametrize('scale', [0., .1, 1.])
def test_gradient_guard_preserves_graph_forward_and_parameter_learning(scale):
    torch.manual_seed(42)
    graph = AffinityGraph(4)
    memory = PrototypeMemory(4, 2)
    memory.observe(torch.eye(4)[:2], torch.tensor([1, -1]), torch.tensor([1., .3]))
    source = torch.randn(1, 3, 4, requires_grad=True)
    reference = graph(source, memory)
    loss = reference['pred_objectness'].sum() + reference['pred_novelty'].sum()
    variables = (source, *graph.parameters())
    baseline = torch.autograd.grad(loss, variables)
    guarded = graph(scale_gradient(source, scale), memory)
    for key in reference:
        torch.testing.assert_close(reference[key], guarded[key], rtol=0, atol=0)
    loss = guarded['pred_objectness'].sum() + guarded['pred_novelty'].sum()
    gradients = torch.autograd.grad(loss, variables)
    torch.testing.assert_close(gradients[0], baseline[0] * scale)
    for before, after in zip(baseline[1:], gradients[1:]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_box_consistency_guard_scales_only_box_gradient_and_not_its_value():
    outputs, targets = evidence()
    with torch.no_grad():
        outputs['pred_boxes'][0, 1, 0] += .02
    guarded_outputs = copy.deepcopy(outputs)
    plain = criterion(0., 1.)(outputs, targets)
    guarded = criterion(0., .1)(guarded_outputs, targets)
    for name in plain:
        torch.testing.assert_close(plain[name], guarded[name], rtol=0, atol=0)
    a = torch.autograd.grad(plain['loss_discovery_consistency'], outputs['pred_boxes'])[0]
    b = torch.autograd.grad(guarded['loss_discovery_consistency'], guarded_outputs['pred_boxes'])[0]
    assert a.abs().sum() > 0
    torch.testing.assert_close(b, a * .1)


def test_task4_disables_unknown_ranking_and_unknown_statistics():
    outputs, targets = evidence()
    outputs['pred_logits'] = torch.zeros(1, 3, 91, requires_grad=True)
    loss_fn = criterion()
    loss_fn.known_ids = loss_fn.base.active_class_ids = list(range(1, 81))
    losses = loss_fn(outputs, targets)
    assert losses['loss_discovery_rank'] == 0
    assert losses['discovery_known_rank_violations'] == losses['discovery_known_unknown_score_sum'] == 0


@pytest.mark.parametrize('field,value', [('discovery_known_rank_coef', -1.),
    ('discovery_known_rank_coef', float('nan')), ('discovery_feature_grad_scale', -1.),
    ('discovery_feature_grad_scale', 1.1), ('discovery_feature_grad_scale', float('inf'))])
def test_invalid_stability_parameters_are_rejected(field, value):
    configuration = args()
    setattr(configuration, field, value)
    with pytest.raises(ValueError, match=field):
        DiscoveryCriterion(SetCriterion(7, HungarianMatcher(), {}, ['labels']), None, configuration)
