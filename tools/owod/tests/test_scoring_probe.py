"""Training-only scoring audit must count duplicate FPs and preserve difficult GT."""
import copy
from types import SimpleNamespace

import pytest
import torch

from tools.owod.scoring_probe import TrainingScoreAudit


def example():
    # Two ordinary objects and one difficult object of the same released class.
    boxes = [[.2, .2, .2, .2], [.2, .2, .2, .2], [.5, .5, .2, .2], [.8, .8, .2, .2]]
    outputs = dict(pred_logits=torch.tensor([[[0., 8., -8.], [0., 7., -8.],
                                             [0., 6., -8.], [0., 5., -8.]]]),
        pred_boxes=torch.tensor([boxes]), pred_objectness=torch.full((1, 4), 10.),
        pred_novelty=torch.full((1, 4), -10.))
    coco = SimpleNamespace(imgs={10: dict(owod_source_id='train-10')}, imgToAnns={10: [
        dict(category_id=1, voc_bbox=[11., 11., 30., 30.]),
        dict(category_id=1, voc_bbox=[41., 41., 60., 60.], difficult=1),
        dict(category_id=1, voc_bbox=[71., 71., 90., 90.])]})
    saved = SimpleNamespace(owod_stage=0, owod_phase='increment', owod_known_class_ids=[1, 2],
                            owod_current_class_ids=[1, 2], owod_max_detections=4)
    spec = dict(image_source_ids=['official-test'], iou_threshold=.5, ap_mode='integral', wi_recall=.8)
    targets = [dict(image_id=torch.tensor(10), orig_size=torch.tensor([100, 100]))]
    return coco, saved, spec, outputs, targets


def test_same_forward_scores_use_voc_one_to_one_duplicates_and_difficult_without_mutation():
    coco, saved, spec, outputs, targets = example()
    original_outputs, original_spec, original_coco = copy.deepcopy(outputs), copy.deepcopy(spec), copy.deepcopy(coco)
    audit = TrainingScoreAudit(coco, [10], saved, spec)
    audit.add(outputs, targets)
    result = audit.summarize()
    assert result['train_subset_only'] and result['diagnostic_only']
    focal = result['metrics']['focal']
    # Recall 1/2 at precision 1, then a duplicate FP, difficult ignored,
    # recall 1 at precision 2/3 => class 1 AP = 5/6; absent class 2 AP = 0.
    assert focal['train_known_ap50'] == pytest.approx(5 / 12)
    assert focal['known_predictions'] == 4 and focal['unknown_predictions'] == 0
    assert set(focal) == {'train_known_ap50', 'known_predictions', 'unknown_predictions'}
    assert result['metrics']['known_sigmoid_only']['train_known_ap50'] == pytest.approx(5 / 12)
    assert spec == original_spec and coco == original_coco
    assert all(torch.equal(outputs[k], v) for k, v in original_outputs.items())
    assert not any(p.requires_grad for pred in audit.predictions['focal'].values() for p in pred.values())


def test_unknowns_compete_for_the_same_topk_budget_without_fake_unknown_recall():
    coco, saved, spec, outputs, targets = example()
    outputs['pred_novelty'].fill_(10.)
    audit = TrainingScoreAudit(coco, [10], saved, spec)
    audit.add(outputs, targets)
    metrics = audit.summarize()['metrics']
    assert metrics['hierarchical']['unknown_predictions'] == saved.owod_max_detections
    assert metrics['hierarchical']['train_known_ap50'] == 0
    assert metrics['known_sigmoid_only']['unknown_predictions'] == 0
    assert metrics['known_sigmoid_only']['train_known_ap50'] > 0


def test_rejects_incomplete_repeated_extra_images_future_labels_and_task2():
    coco, saved, spec, outputs, targets = example()
    audit = TrainingScoreAudit(coco, [10], saved, spec)
    with pytest.raises(ValueError, match='cover'):
        audit.summarize()
    wrong = copy.deepcopy(targets)
    wrong[0]['image_id'] = torch.tensor(99)
    with pytest.raises(ValueError, match='Unexpected'):
        audit.add(outputs, wrong)
    audit.add(outputs, targets)
    with pytest.raises(ValueError, match='repeated'):
        audit.add(outputs, targets)
    for ids in ([], [10, 10]):
        with pytest.raises(ValueError, match='unique'):
            TrainingScoreAudit(coco, ids, saved, spec)
    saved.owod_stage = 1
    with pytest.raises(ValueError, match='Task 1'):
        TrainingScoreAudit(coco, [10], saved, spec)
    saved.owod_stage = 0
    coco.imgToAnns[10][0]['category_id'] = 8
    with pytest.raises(ValueError, match='unreleased'):
        TrainingScoreAudit(coco, [10], saved, spec)
