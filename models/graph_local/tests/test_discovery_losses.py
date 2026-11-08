"""Loss ablations preserve known supervision and actual unknown learning."""
import copy

import pytest
import torch

import models.graph_local.discovery as discovery
from models.graph_local.tests.test_discovery import evidence
from models.graph_local.tests.test_discovery_ramp import criterion


def test_count_normalization_keeps_absolute_confidence_and_known_gradient():
    logits = torch.tensor([[.4, -.2, .1, .6]], requires_grad=True)
    target = torch.tensor([[0., 1., 1., 0.]])
    confidence = torch.tensor([[1., .4, .2, 0.]])
    ramp = torch.tensor([[1., 1 / 3, 1 / 3, 1.]])
    reference = torch.autograd.grad(discovery.balanced_bce(
        logits, target, confidence, ramp, normalization='count'), logits)[0]
    low = confidence.clone()
    low[:, 1:3] *= .1
    reduced = torch.autograd.grad(discovery.balanced_bce(
        logits, target, low, ramp, normalization='count'), logits)[0]
    torch.testing.assert_close(reduced[:, 1:3], reference[:, 1:3] * .1)
    torch.testing.assert_close(reduced[:, [0, 3]], reference[:, [0, 3]])
    assert reduced[0, 1] < 0 and reduced[0, 2] < 0  # Unknown learning survives.
    assert reference[0, 3] == 0  # Unselected slots never become novelty negatives.
    legacy = torch.autograd.grad(discovery.balanced_bce(logits, target, confidence, ramp), logits)[0]
    torch.testing.assert_close(reference[:, 0], legacy[:, 0])
    legacy_low = torch.autograd.grad(discovery.balanced_bce(logits, target, low, ramp), logits)[0]
    torch.testing.assert_close(legacy, legacy_low)


def test_focal_concentrates_on_high_score_negatives_without_losing_pseudo_weight():
    logits = torch.tensor([[1., -.3, .3, -5.]], requires_grad=True)
    foreground = torch.tensor([[True, True, False, False]])
    confidence = torch.tensor([[1., .2, 1., 1.]])
    full = torch.autograd.grad(discovery.proposal_focal_loss(logits, foreground, confidence), logits)[0]
    ramp = torch.tensor([[1., 1 / 3, 1., 1.]])
    partial = torch.autograd.grad(discovery.proposal_focal_loss(logits, foreground, confidence, ramp), logits)[0]
    assert full[0, 0] < 0 and full[0, 1] < 0
    assert full[0, 2] > 100 * full[0, 3] > 0
    torch.testing.assert_close(partial[:, 1], full[:, 1] / 3)
    torch.testing.assert_close(partial[:, [0, 2, 3]], full[:, [0, 2, 3]])
    lower = confidence.clone()
    lower[:, 1] *= .1
    low = torch.autograd.grad(discovery.proposal_focal_loss(logits, foreground, lower), logits)[0]
    torch.testing.assert_close(low[:, 1], full[:, 1] * .1)
    torch.testing.assert_close(low[:, [0, 2, 3]], full[:, [0, 2, 3]])


def test_focal_ddp_matches_global_objective_even_with_empty_positive_rank(monkeypatch):
    joined = torch.tensor([[.7, -.4, 1.1], [.4, -.8, -2.]], requires_grad=True)
    foreground = torch.tensor([[True, True, False], [False, False, False]])
    weights = torch.tensor([[1., .2, 1.], [1., 1., 1.]])
    reference_loss = discovery.proposal_focal_loss(joined, foreground, weights)
    reference_grad = torch.autograd.grad(reference_loss, joined)[0]
    monkeypatch.setattr(discovery, 'is_dist_avail_and_initialized', lambda: True)
    monkeypatch.setattr(discovery, 'get_world_size', lambda: 2)
    losses = []
    for rank in range(2):
        remote_count = int(foreground[1 - rank].sum())
        monkeypatch.setattr(torch.distributed, 'all_reduce', lambda count: count.add_(remote_count))
        local = joined.detach()[rank:rank + 1].clone().requires_grad_(True)
        loss = discovery.proposal_focal_loss(local, foreground[rank:rank + 1], weights[rank:rank + 1])
        grad = torch.autograd.grad(loss, local)[0]
        torch.testing.assert_close(grad / 2, reference_grad[rank:rank + 1])
        losses.append(loss.detach())
    torch.testing.assert_close(torch.stack(losses).mean(), reference_loss.detach())


@pytest.mark.parametrize('mode,changed', [('objectness', {'loss_objectness'}),
                                         ('confidence', {'loss_unknown'}),
                                         ('combined', {'loss_objectness', 'loss_unknown'})])
def test_each_ablation_changes_only_its_losses_and_keeps_state_evidence(mode, changed):
    outputs, targets = evidence()
    baseline = criterion(3, 2)
    treatment = criterion(3, 2)
    if mode in ('objectness', 'combined'):
        treatment.objectness_loss = 'focal'
    if mode in ('confidence', 'combined'):
        treatment.pseudo_normalization = 'count'
    before, after = baseline(outputs, targets), treatment(copy.deepcopy(outputs), targets)
    assert before.keys() == after.keys()
    for name in before:
        if name not in changed:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        else:
            assert not torch.allclose(before[name], after[name])
    for a, b in zip(baseline.evidence, treatment.evidence):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('epoch,empty', [(0, False), (0, True), (2, True)])
def test_combined_empty_warmup_and_future_gt_are_safe(epoch, empty):
    outputs, targets = evidence()
    if empty:
        targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    loss_fn = criterion(3, epoch)
    loss_fn.objectness_loss = 'focal'
    loss_fn.pseudo_normalization = 'count'
    released = loss_fn(outputs, targets)
    extra = [dict(labels=torch.cat([targets[0]['labels'], torch.tensor([6])]),
                  boxes=torch.cat([targets[0]['boxes'], outputs['pred_boxes'][0, 1:2].detach()]))]
    hidden = loss_fn(outputs, extra)
    for name in released:
        assert torch.isfinite(released[name])
        torch.testing.assert_close(released[name], hidden[name], rtol=0, atol=0)
    sum(value for name, value in released.items() if name in loss_fn.weight_dict).backward()
    assert torch.isfinite(outputs['pred_objectness'].grad).all()
    assert torch.isfinite(outputs['pred_novelty'].grad).all()


def test_novelty_with_no_supervised_slots_is_zero_and_differentiable():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = discovery.balanced_bce(logits, torch.zeros_like(logits), torch.zeros_like(logits),
                                 normalization='count')
    loss.backward()
    assert loss == 0 and logits.grad.count_nonzero() == 0
