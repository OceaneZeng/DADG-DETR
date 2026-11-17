import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.analysis.audit_unknown_detection import (
    author_boxes, coverage, measure_image, summarize, validate_checkpoint_identity,
)
from tools.owod.protocol import file_sha256


def predictions(boxes, objects, novelty, limit=50):
    outputs = dict(pred_boxes=torch.tensor([boxes], dtype=torch.float64),
                   pred_objectness=torch.logit(torch.tensor([objects], dtype=torch.float64)),
                   pred_novelty=torch.logit(torch.tensor([novelty], dtype=torch.float64)),
                   pred_logits=torch.zeros(1, len(boxes), 2, dtype=torch.float64))
    result = DiscoveryPostProcess(known_ids=[1], max_detections=limit, unknown_threshold=.5)(
        outputs, torch.tensor([[100, 100]]))[0]
    return outputs, result


def test_audit_separates_box_objectness_novelty_product_and_topk_losses():
    boxes = [[.1 + .18 * i, .3, .1, .1] for i in range(5)]
    outputs, result = predictions(boxes, [.4, .9, .7, .9, .999], [.9, .4, .7, .9, .001], limit=1)
    absolute = author_boxes(result['boxes'].tolist())
    assert result['labels'].tolist() == [1]
    annotations = [dict(category_id=2, voc_bbox=[6 + 18 * i, 26, 15 + 18 * i, 35]) for i in range(4)]
    before = {key: value.clone() for key, value in outputs.items()}
    row = measure_image(outputs, result, annotations, [1], [100, 100], .5, .5)
    assert row['unknown_gt'] == 4 and row['raw_box_coverage'] == 4
    assert row['objectness_gate_coverage'] == 3 and row['novelty_gate_coverage'] == 3
    assert row['joint_score_gate_coverage'] == 1 and row['emitted_unknown_coverage'] == 0
    assert absolute.shape == (1, 4)
    assert all(torch.equal(value, before[key]) for key, value in outputs.items())


def test_unknown_output_does_not_remove_same_box_known_open_set_error():
    from models.owod_benchmark_eval import evaluate_author_benchmark
    from tools.owod.benchmark_spec import specification
    outputs, result = predictions([[.5, .5, .2, .2]], [.9], [.9])
    annotations = [dict(category_id=2, voc_bbox=[41, 41, 60, 60])]
    row = measure_image(outputs, result, annotations, [1], [100, 100], .5, .5)
    assert row['unknown_tp_local'] == 1
    assert row['known_on_unknown'] == row['known_on_unknown_same_box_as_unknown'] == 1
    coco = SimpleNamespace(imgs={1: {'owod_source_id': 'one'}}, imgToAnns={1: annotations})
    spec = {**specification('s-owodb'), 'image_source_ids': ['one']}
    actual = evaluate_author_benchmark({1: result}, coco, [1], [], [1], spec, .5)
    assert actual['u_recall'] == 1. and actual['a_ose'] == row['known_on_unknown']


def test_coverage_is_not_one_to_one_and_difficult_unknown_is_separate():
    outputs, result = predictions([[.5, .5, .2, .2]], [.9], [.9])
    annotations = [dict(category_id=2, voc_bbox=[41, 41, 60, 60]) for _ in range(2)]
    row = measure_image(outputs, result, annotations, [1], [100, 100], .5, .5)
    assert row['emitted_unknown_coverage'] == 2 and row['unknown_tp_local'] == 1
    annotations = [dict(category_id=2, voc_bbox=[41, 41, 60, 60], difficult=1)]
    row = measure_image(outputs, result, annotations, [1], [100, 100], .5, .5)
    assert row['unknown_gt'] == 0 and row['unknown_on_difficult'] == 1
    assert row['known_on_unknown'] == 1  # Match the author's A-OSE convention.


def test_error_bins_separate_duplicate_known_overlap_and_other():
    outputs, result = predictions([[.2, .2, .1, .1], [.2, .2, .1, .1],
                                   [.5, .5, .1, .1], [.8, .8, .1, .1]], [.9] * 4, [.9] * 4)
    annotations = [dict(category_id=2, voc_bbox=[16, 16, 25, 25]),
                   dict(category_id=1, voc_bbox=[46, 46, 55, 55])]
    row = measure_image(outputs, result, annotations, [1], [100, 100], .5, .5)
    assert row['unknown_tp_local'] == row['unknown_duplicate'] == 1
    assert row['unknown_on_known'] == row['unknown_other'] == 1


def test_author_quantization_inclusive_iou_and_strict_threshold():
    boxes = author_boxes([[0, 0, 10, 10]])
    assert boxes.tolist() == [[1., 1., 10., 10.]]
    assert coverage(boxes, [[1, 1, 5, 10]], .5) == 0  # IoU exactly .5 is not a TP.
    assert coverage(boxes, [[1, 1, 6, 10]], .5) == 1
    assert coverage(np.empty((0, 4)), [[1, 1, 6, 10]], .5) == 0


def test_summary_keeps_author_list_multiplicity_and_absent_denominator():
    rows = {'a': dict(images=1, unknown_gt=2, raw_box_coverage=1),
            'b': dict(images=1, unknown_gt=1, raw_box_coverage=1)}
    totals = summarize(rows, ['a', 'b', 'a'])
    assert totals['images'] == 3 and totals['unknown_gt'] == 5
    assert totals['coverage_fractions']['raw_box_coverage'] == pytest.approx(.6)
    assert summarize({'a': dict(images=1)}, ['a'])['coverage_fractions']['raw_box_coverage'] is None


def checkpoint_args(manifest, digest):
    return SimpleNamespace(owod_discovery=True, discovery_version=7, owod_stage=1,
                           owod_protocol=manifest['protocol'], owod_evaluation=manifest['evaluation'],
                           owod_known_class_ids=manifest['stages'][1]['active_classes'],
                           owod_current_class_ids=manifest['stages'][1]['classes'],
                           owod_previous_class_ids=manifest['stages'][0]['active_classes'],
                           owod_manifest_sha256=digest)


def test_identity_rejects_different_protocol_order_and_unsupported_branch():
    path = Path(__file__).resolve().parents[3] / 'data/owod/s-owodb/split_manifest.json'
    manifest = json.loads(path.read_text())
    saved = checkpoint_args(manifest, file_sha256(path))
    before = copy.deepcopy(vars(saved))
    assert validate_checkpoint_identity(path, saved, manifest)['proof'] == 'identical_manifest'
    assert vars(saved) == before
    saved.owod_current_class_ids = list(reversed(saved.owod_current_class_ids))
    with pytest.raises(ValueError, match='class order'):
        validate_checkpoint_identity(path, saved, manifest)
    saved = checkpoint_args(manifest, file_sha256(path))
    saved.discovery_version = 8  # mowodb's different model must use its own checkout.
    with pytest.raises(ValueError, match='v5/v6/v7'):
        validate_checkpoint_identity(path, saved, manifest)


def test_exact_legacy_manifest_recovery_for_task2_is_read_only():
    path = Path(__file__).resolve().parents[3] / 'data/owod/s-owodb/split_manifest.json'
    manifest = json.loads(path.read_text())
    saved = checkpoint_args(manifest, '13a1f3b3082358e690e04d0030f10edd32d5a6e16e9b1a90a6e2faa7c6f8771d')
    saved.owod_manifest = '/home/top/disks/new-hdd/zhy/Tree-DETR/data/owod/s-owodb/split_manifest.json'
    saved.owod_image_root = '/home/top/disks/new-hdd/zhy/Tree-DETR/data/raw/coco'
    before = copy.deepcopy(vars(saved))
    assert validate_checkpoint_identity(path, saved, manifest)['proof'] == 'exact_precanonical_serialization_sha256'
    assert vars(saved) == before
    saved.owod_manifest_sha256 = 'changed'
    with pytest.raises(ValueError, match='SHA-256'):
        validate_checkpoint_identity(path, saved, manifest)
