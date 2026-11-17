"""Bounded candidate protection: loss gradients, real matching and teacher guards."""
import pytest
import torch

from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.discovery import DiscoveryCriterion
from models.graph_local.discovery_candidates import mine_candidates
from models.graph_local.tests.test_discovery import args, evidence


def candidate_scene(monkeypatch):
    # Real view matching, including repeated boxes; prescribed saliency keeps
    # every gate's boundary independent of heatmap interpolation details.
    boxes = torch.tensor([[[.1, .1, .1, .1], [.3, .3, .1, .1], [.3, .3, .1, .1],
                           [.6, .6, .1, .1], [.6, .6, .1, .1], [.1, .1, .1, .1],
                           [.9, .1, .1, .1], [.1, .9, .1, .1], [.9, .9, .1, .1]]])
    count = boxes.shape[1]
    logits = torch.full((1, count, 7), -4.)
    logits[0, 0, 1] = 4.
    features = torch.ones(1, count, 4)
    outputs = dict(pred_logits=logits.requires_grad_(), pred_boxes=boxes.requires_grad_(),
                   query_features=features.clone().requires_grad_(), attention_feature=torch.zeros(1, 1, 10, 10),
                   image_sizes=[(100, 100)], padded_size=(100, 100),
                   pred_objectness=torch.zeros(1, count, requires_grad=True),
                   pred_novelty=torch.zeros(1, count, requires_grad=True),
                   discovery_routes=[torch.zeros(1, count, 3, requires_grad=True)],
                   discovery_embedding=features.clone().requires_grad_(),
                   affinities=torch.full((1, count, 93), -1e4, requires_grad=True))
    teacher = {key: value.detach().clone() if torch.is_tensor(value) else value
               for key, value in outputs.items()}
    teacher['pred_boxes'][..., 0] = 1 - teacher['pred_boxes'][..., 0]
    teacher['query_features'][0, 7] *= -1  # A strong heatmap alone is insufficient.
    outputs['discovery_teacher'] = teacher
    student_saliency = torch.tensor([[.9, .9, .85, .6, .55, .95, .1, .6, .6]])
    teacher_saliency = student_saliency.clone()
    teacher_saliency[0, 8] = .4
    def saliency(feature, *unused):
        return student_saliency if feature is outputs['attention_feature'] else teacher_saliency
    monkeypatch.setattr('models.graph_local.discovery_candidates.saliency_in_boxes', saliency)
    targets = [dict(labels=torch.tensor([1]), boxes=boxes[0, :1].detach())]
    return outputs, targets


def make_criterion(policy='cross-view-protected', *, warmup=0):
    configuration = args()
    configuration.discovery_version = 7
    configuration.discovery_top_k = 1
    configuration.discovery_negative_policy = policy
    configuration.discovery_warmup = warmup
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    return DiscoveryCriterion(base, None, configuration)


def test_protects_only_independent_cross_view_evidence_and_keeps_all_other_negatives(monkeypatch):
    outputs, targets = candidate_scene(monkeypatch)
    criterion = make_criterion()
    losses = criterion(outputs, targets)
    assert losses['discovery_pseudo_count'] == 1
    assert losses['discovery_protected_count'] == 1
    assert losses['discovery_proposal_negative_fraction'] == pytest.approx(6 / 9)
    sum(losses[k] for k in ('loss_objectness', 'loss_discovery_route', 'loss_unknown', 'loss_ce')).backward()
    object_grad = outputs['pred_objectness'].grad[0]
    route_grad = outputs['discovery_routes'][0].grad[0]
    novelty_grad = outputs['pred_novelty'].grad[0]
    assert (object_grad[:2] < 0).all()  # Released GT and pseudo still learn foreground.
    assert object_grad[3] == 0 and route_grad[3].count_nonzero() == 0
    assert novelty_grad[3] == 0  # Protection does not fabricate unknown supervision.
    assert novelty_grad[1] < 0 and route_grad[1, 2] < 0
    for query in (2, 4, 5, 6, 7, 8):
        assert object_grad[query] > 0 and route_grad[query, 0] < 0
    assert outputs['pred_logits'].grad[0, 3, 1] > 0  # Dense known loss remains active.
    assert outputs['pred_logits'].grad[..., 6].count_nonzero() == 0
    assert criterion.evidence[1].tolist() == [1, -1]  # Protected slot never enters memory.


def test_protection_requires_teacher_and_finished_warmup(monkeypatch):
    outputs, targets = candidate_scene(monkeypatch)
    criterion = make_criterion(warmup=1)
    assert criterion(outputs, targets)['discovery_protected_count'] == 0
    criterion.epoch = 1
    assert criterion(outputs, targets)['discovery_protected_count'] == 1
    outputs.pop('discovery_teacher')
    assert criterion(outputs, targets)['discovery_protected_count'] == 0


def test_future_labels_do_not_change_any_loss_or_protection(monkeypatch):
    outputs, targets = candidate_scene(monkeypatch)
    criterion = make_criterion()
    first = criterion(outputs, targets)
    second = criterion(outputs, [dict(labels=torch.tensor([1, 6]),
        boxes=torch.cat([targets[0]['boxes'], outputs['pred_boxes'][0, 3:4].detach()]))])
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_dense_policy_still_trains_unselected_candidate_as_negative(monkeypatch):
    outputs, targets = candidate_scene(monkeypatch)
    losses = make_criterion('dense')(outputs, targets)
    (losses['loss_objectness'] + losses['loss_discovery_route']).backward()
    assert losses['discovery_protected_count'] == 0
    assert losses['discovery_objectness_supervised_fraction'] == 1
    assert outputs['pred_objectness'].grad[0, 3] > 0
    assert outputs['discovery_routes'][0].grad[0, 3, 0] < 0


def test_real_heatmap_moderate_candidate_is_protected_without_becoming_pseudo():
    outputs, targets = evidence()
    outputs['attention_feature'][:, :, :4, 6:] = .55
    outputs['discovery_teacher']['attention_feature'] = outputs['attention_feature'].flip(-1)
    criterion = make_criterion()
    losses = criterion(outputs, targets)
    losses['loss_objectness'].backward()
    assert losses['discovery_pseudo_count'] == 1
    assert losses['discovery_protected_count'] == 1
    assert outputs['pred_objectness'].grad[0, 2] == 0
    assert criterion.evidence[1].tolist() == [1, -1]


@pytest.mark.parametrize('count', [2, 20])
def test_overflow_confident_candidates_are_bounded_and_default_pseudo_is_unchanged(monkeypatch, count):
    # All boxes distinct; no threshold ambiguity. Top-k pseudo and protection
    # each have their own cap, leaving the rest as dense negatives.
    centers = torch.linspace(.05, .95, count)
    boxes = torch.stack([centers, centers, centers * 0 + .01, centers * 0 + .01], -1)[None]
    teacher_boxes = boxes.clone(); teacher_boxes[..., 0] = 1 - teacher_boxes[..., 0]
    outputs = dict(pred_boxes=boxes, pred_logits=torch.zeros(1, count, 7),
                   query_features=torch.ones(1, count, 4), attention_feature=torch.ones(1, 1, 10, 10),
                   image_sizes=[(100, 100)], padded_size=(100, 100))
    teacher = dict(outputs, pred_boxes=teacher_boxes)
    monkeypatch.setattr('models.graph_local.discovery_candidates.saliency_in_boxes',
                        lambda *args: torch.ones(1, count))
    targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    dense = mine_candidates(outputs, targets, indices, [1], teacher, top_k=1)
    optimized = mine_candidates(outputs, targets, indices, [1], teacher, top_k=1, protect_unselected=True)
    assert optimized['pseudo'].sum() == optimized['protected'].sum() == 1
    assert not (optimized['pseudo'] & optimized['protected']).any()
    for key in dense:
        if key != 'protected':
            assert torch.equal(dense[key], optimized[key]), key


def test_protected_mask_cannot_disable_positive_supervision():
    from models.graph_local.discovery import proposal_supervision
    positive, pseudo = torch.tensor([[True, False]]), torch.tensor([[False, True]])
    foreground, weights, proposals = proposal_supervision(
        positive, pseudo, torch.tensor([[0., .4]]), torch.ones_like(positive))
    assert foreground.all() and torch.equal(proposals, weights)
    assert proposals.tolist() == [[1., pytest.approx(.4)]]
