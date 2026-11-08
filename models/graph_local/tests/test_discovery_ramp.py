"""Check pseudo gradient strength, legacy equivalence and empty supervision."""
import copy

import pytest
import torch

from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.discovery import DiscoveryCriterion, balanced_bce, pseudo_supervision_scale
from models.graph_local.tests.test_discovery import args, evidence


def criterion(ramp_epochs, epoch):
    configuration = args()
    configuration.discovery_warmup = 1
    configuration.discovery_pseudo_ramp_epochs = ramp_epochs
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    result = DiscoveryCriterion(base, None, configuration)
    result.epoch = epoch
    return result


@pytest.mark.parametrize('ramp_epochs,expected', [
    (0, [0., 1., 1., 1., 1.]), (3, [0., 1 / 3, 2 / 3, 1., 1.])])
def test_schedule_has_full_strength_after_fixed_phase_local_budget(ramp_epochs, expected):
    assert [pseudo_supervision_scale(e, 1, ramp_epochs) for e in range(5)] == expected


def test_pseudo_scale_survives_group_normalization_with_only_one_pseudo():
    logits = torch.tensor([[.3, -.7]], requires_grad=True)
    target = torch.tensor([[0., 1.]])
    weights = torch.tensor([[1., .1]])
    full = torch.autograd.grad(balanced_bce(logits, target, weights), logits)[0]
    scaled = torch.autograd.grad(balanced_bce(
        logits, target, weights, torch.tensor([[1., .25]])), logits)[0]
    torch.testing.assert_close(scaled[:, :1], full[:, :1])
    torch.testing.assert_close(scaled[:, 1:], full[:, 1:] * .25)
    assert scaled[0, 1] < 0  # Unknown learning still receives a real gradient.


@pytest.mark.parametrize('loss_key,output_key', [
    ('loss_objectness', 'pred_objectness'), ('loss_unknown', 'pred_novelty'),
    ('loss_discovery_route', 'discovery_routes'), ('loss_discovery_consistency', 'pred_boxes')])
def test_ramp_only_reduces_pseudo_gradients_and_preserves_real_supervision(loss_key, output_key):
    outputs, targets = evidence()
    with torch.no_grad():
        outputs['pred_boxes'][0, 1, 0] += .02  # Nonzero teacher box consistency cost.
    full_outputs = copy.deepcopy(outputs)
    partial_losses = criterion(3, 1)(outputs, targets)
    full_losses = criterion(0, 1)(full_outputs, targets)
    variable = outputs[output_key][0] if output_key == 'discovery_routes' else outputs[output_key]
    reference = full_outputs[output_key][0] if output_key == 'discovery_routes' else full_outputs[output_key]
    partial_grad = torch.autograd.grad(partial_losses[loss_key], variable)[0]
    full_grad = torch.autograd.grad(full_losses[loss_key], reference)[0]
    assert full_grad[:, 1].abs().sum() > 0
    torch.testing.assert_close(partial_grad[:, 1], full_grad[:, 1] / 3)
    torch.testing.assert_close(partial_grad[:, [0, 2]], full_grad[:, [0, 2]])
    assert partial_losses['discovery_pseudo_per_image'] == 1
    assert partial_losses['discovery_pseudo_scale'] == pytest.approx(1 / 3)
    for key in ('loss_ce', 'loss_bbox', 'loss_giou'):
        torch.testing.assert_close(partial_losses[key], full_losses[key], rtol=0, atol=0)


def test_full_strength_is_identical_to_v5_including_prototype_evidence():
    outputs, targets = evidence()
    legacy, ramped = criterion(0, 3), criterion(3, 3)
    full = legacy(outputs, targets)
    recovered = ramped(outputs, targets)
    assert full.keys() == recovered.keys()
    for key in full:
        torch.testing.assert_close(full[key], recovered[key], rtol=0, atol=0)
    for a, b in zip(legacy.evidence, ramped.evidence):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('epoch,empty', [(0, False), (0, True), (1, True)])
def test_warmup_and_empty_known_targets_keep_losses_finite(epoch, empty):
    outputs, targets = evidence()
    if empty:
        targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    loss_fn = criterion(3, epoch)
    losses = loss_fn(outputs, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    if epoch == 0:
        assert losses['discovery_pseudo_count'] == losses['discovery_pseudo_scale'] == 0
    sum(value for name, value in losses.items() if name in loss_fn.weight_dict).backward()
    assert torch.isfinite(outputs['pred_objectness'].grad).all()
    assert torch.isfinite(outputs['pred_novelty'].grad).all()


def test_ramped_training_never_uses_future_ground_truth():
    outputs, targets = evidence()
    loss_fn = criterion(3, 1)
    released = loss_fn(outputs, targets)
    extra = [dict(labels=torch.tensor([1, 6]),
                  boxes=torch.cat([targets[0]['boxes'], outputs['pred_boxes'][0, 1:2].detach()]))]
    with_future = loss_fn(outputs, extra)
    for name in released:
        torch.testing.assert_close(released[name], with_future[name], rtol=0, atol=0)
