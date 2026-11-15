"""Background novelty must suppress trusted slots without erasing unknown learning."""
import copy

import pytest
import torch

from models.graph_local.tests.test_discovery import evidence
from models.graph_local.tests.test_discovery_ramp import criterion


def loss_fn(background=True, epoch=1):
    result = criterion(3, epoch)
    result.pseudo_normalization = 'count'
    result.background_negative = background
    return result


def test_new_gradient_is_limited_to_novelty_and_preserves_pseudo_ramp_and_evidence():
    before, targets = evidence()
    after = copy.deepcopy(before)
    reference, treatment = loss_fn(False), loss_fn()
    old, new = reference(before, targets), treatment(after, targets)
    old_grad = torch.autograd.grad(old['loss_unknown'], before['pred_novelty'])[0]
    new_grad = torch.autograd.grad(new['loss_unknown'], after['pred_novelty'])[0]
    assert old_grad[0, 2] == 0 < new_grad[0, 2]  # Descent lowers background novelty.
    assert new_grad[0, 1] < 0  # Pseudo objects still learn unknown.
    torch.testing.assert_close(new_grad[:, 1], old_grad[:, 1], rtol=0, atol=0)
    # Known and background share the existing negative-group mean.
    torch.testing.assert_close(new_grad[:, 0], old_grad[:, 0] / 2)
    changed = {'loss_unknown', 'discovery_ignored_fraction',
               'discovery_novelty_supervised_fraction', 'discovery_novelty_background_fraction'}
    for key in old.keys() - changed:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    for a, b in zip(reference.evidence, treatment.evidence):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert new['discovery_novelty_background_fraction'] == pytest.approx(1 / 3)
    assert new['discovery_novelty_supervised_fraction'] == 1
    full = loss_fn(epoch=3)(after, targets)
    full_grad = torch.autograd.grad(full['loss_unknown'], after['pred_novelty'])[0]
    torch.testing.assert_close(new_grad[:, 1], full_grad[:, 1] / 3)


@pytest.mark.parametrize('condition', ['warmup', 'no-teacher', 'evaluation', 'ambiguous', 'teacher-high'])
def test_background_is_ignored_without_two_view_trust(condition):
    outputs, targets = evidence()
    objective = loss_fn(epoch=0 if condition == 'warmup' else 1)
    if condition == 'no-teacher':
        del outputs['discovery_teacher']
    if condition == 'evaluation':
        objective.eval()
    if condition in ('ambiguous', 'teacher-high'):
        if condition == 'ambiguous':
            outputs['attention_feature'][:, :, :3, 7:] = .3
        teacher = outputs['discovery_teacher']
        teacher['attention_feature'][:, :, :3, :3] = .3 if condition == 'ambiguous' else 1.
    losses = objective(outputs, targets)
    gradient = torch.autograd.grad(losses['loss_unknown'], outputs['pred_novelty'])[0]
    assert gradient[0, 2] == 0
    assert losses['discovery_novelty_background_fraction'] == 0


@pytest.mark.parametrize('empty', [False, True])
def test_background_never_uses_future_gt_and_empty_known_targets_stay_finite(empty):
    outputs, targets = evidence()
    if empty:
        targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    objective = loss_fn()
    released = objective(outputs, targets)
    future = [dict(labels=torch.cat([targets[0]['labels'], torch.tensor([6])]),
                   boxes=torch.cat([targets[0]['boxes'], outputs['pred_boxes'][0, 1:2].detach()]))]
    hidden = objective(outputs, future)
    for name in released:
        assert torch.isfinite(released[name])
        torch.testing.assert_close(released[name], hidden[name], rtol=0, atol=0)
    released['loss_unknown'].backward()
    assert outputs['pred_novelty'].grad[0, 2] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_v9_real_cuda_detector_backward_increment_and_state_reload():
    from models.graph_local.tests.test_discovery import test_real_deformable_detector_cuda_discovery_and_checkpoint
    test_real_deformable_detector_cuda_discovery_and_checkpoint(
        3, 'balanced-bce', 'count', 1., 1., background_negative=True)
