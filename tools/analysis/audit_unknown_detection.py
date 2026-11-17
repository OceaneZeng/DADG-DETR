"""Read-only discovery unknown coverage/error audit, never a training entry point."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def author_boxes(boxes):
    """Apply exactly the prediction coordinate convention of the author evaluator."""
    import numpy as np
    return np.asarray([[float(f'{float(x) + (1. if i < 2 else 0.):.1f}')
                        for i, x in enumerate(box)] for box in boxes], dtype=float).reshape(-1, 4)


def coverage(boxes, ground_truth, threshold):
    """Any-box coverage is a diagnostic upper bound, not one-to-one recall."""
    import numpy as np
    from models.owod_benchmark_eval import overlaps
    return sum(bool((overlaps(np.asarray(box, dtype=float), boxes) > threshold).any())
               for box in ground_truth)


def measure_image(outputs, prediction, annotations, known_ids, size, unknown_threshold, iou_threshold):
    """Use full annotations only after forward; they never reach a loss or candidate miner."""
    import numpy as np
    import torch
    from models.owod_benchmark_eval import overlaps
    from util.box_ops import box_cxcywh_to_xyxy

    known = set(known_ids)
    unseen = [ann for ann in annotations if ann['category_id'] not in known]
    unknown_gt = [ann['voc_bbox'] for ann in unseen if not ann.get('difficult', 0)]
    all_unknown_gt = [ann['voc_bbox'] for ann in unseen]
    known_gt = [ann['voc_bbox'] for ann in annotations if ann['category_id'] in known]
    obj = outputs['pred_objectness'][0].sigmoid().detach().cpu().numpy()
    novelty = outputs['pred_novelty'][0].sigmoid().detach().cpu().numpy()
    h, w = size
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0]).detach().cpu()
    boxes = author_boxes((boxes * torch.tensor([w, h, w, h])).tolist())
    if not np.isfinite(boxes).all() or not np.isfinite(obj).all() or not np.isfinite(novelty).all():
        raise ValueError('Non-finite raw boxes or discovery scores')
    pred_boxes = author_boxes(prediction['boxes'].tolist())
    labels = prediction['labels'].tolist()
    scores = np.asarray([float(f'{float(x):.3f}') for x in prediction['scores']])
    unknown_ids = [i for i, label in enumerate(labels) if label == 91]
    known_ids_pred = [i for i, label in enumerate(labels) if label in known]
    unknown_boxes = pred_boxes[unknown_ids]
    result = dict(images=1, unknown_gt=len(unknown_gt),
                  raw_box_coverage=coverage(boxes, unknown_gt, iou_threshold),
                  objectness_gate_coverage=coverage(boxes[obj >= unknown_threshold], unknown_gt, iou_threshold),
                  novelty_gate_coverage=coverage(boxes[novelty >= unknown_threshold], unknown_gt, iou_threshold),
                  joint_score_gate_coverage=coverage(
                      boxes[obj * novelty >= unknown_threshold], unknown_gt, iou_threshold),
                  emitted_unknown_coverage=coverage(unknown_boxes, unknown_gt, iou_threshold),
                  unknown_predictions=len(unknown_ids), known_predictions=len(known_ids_pred),
                  unknown_tp_local=0, unknown_duplicate=0, unknown_on_known=0,
                  unknown_on_difficult=0, unknown_other=0, known_on_unknown=0,
                  known_on_unknown_same_box_as_unknown=0)
    # The exact official metrics are computed separately over the complete
    # sequence. Local tie sorting can differ from a global author's np.argsort.
    matched = set()
    for offset in np.argsort(-scores[unknown_ids]):
        box = unknown_boxes[offset]
        ious = overlaps(box, all_unknown_gt)
        best = int(ious.argmax()) if len(ious) else -1
        if best >= 0 and ious[best] > iou_threshold:
            if unseen[best].get('difficult', 0):
                result['unknown_on_difficult'] += 1
            elif best in matched:
                result['unknown_duplicate'] += 1
            else:
                matched.add(best)
                result['unknown_tp_local'] += 1
        elif (overlaps(box, known_gt) > iou_threshold).any():
            result['unknown_on_known'] += 1
        else:
            # This includes localization failures and unannotated content;
            # it must not be relabelled as certain background.
            result['unknown_other'] += 1
    for index in known_ids_pred:
        box = pred_boxes[index]
        if (overlaps(box, all_unknown_gt) > iou_threshold).any():
            result['known_on_unknown'] += 1
            if len(unknown_boxes) and np.any(np.all(unknown_boxes == box, axis=1)):
                result['known_on_unknown_same_box_as_unknown'] += 1
    return result


def summarize(rows, sequence):
    totals = Counter()
    for source_id in sequence:
        totals.update(rows[source_id])
    result = dict(totals)
    denominator = result.get('unknown_gt', 0)
    result['coverage_fractions'] = {
        key: result.get(key, 0) / denominator if denominator else None
        for key in ('raw_box_coverage', 'objectness_gate_coverage', 'novelty_gate_coverage',
                    'joint_score_gate_coverage', 'emitted_unknown_coverage')}
    return result


def validate_checkpoint_identity(path, saved, manifest):
    """Prove identical/byte-equivalent manifest without altering saved arguments."""
    from tools.owod.protocol import file_sha256
    if (not getattr(saved, 'owod_discovery', False)
            or getattr(saved, 'owod_screening', '')
            or getattr(saved, 'owod_screening_sha256', None)
            or getattr(saved, 'discovery_version', None) not in (5, 6, 7)):
        raise ValueError('This main-branch audit supports full-data discovery v5/v6/v7 only')
    stage = getattr(saved, 'owod_stage', -1)
    if stage not in (0, 1, 2):
        raise ValueError('Unknown audit requires Task 1, 2 or 3; Task 4 has no unknown classes')
    record = manifest['stages'][stage]
    previous = manifest['stages'][stage - 1]['active_classes'] if stage else []
    if (saved.owod_protocol != manifest['protocol'] or saved.owod_evaluation != manifest['evaluation']
            or saved.owod_known_class_ids != record['active_classes']
            or saved.owod_current_class_ids != record['classes']
            or list(saved.owod_previous_class_ids or []) != previous):
        raise ValueError('Checkpoint protocol, evaluation or class order differs from the manifest')
    current = file_sha256(path)
    if current == saved.owod_manifest_sha256:
        return dict(proof='identical_manifest', canonical_manifest_sha256=current)
    # Only exact serialization recovery is accepted, as in the existing S
    # diagnostic. This is read-only compatibility, never formal resume.
    if manifest['protocol'] != 's-owodb':
        raise ValueError('Checkpoint manifest mismatch')
    from tools.owod.task2_parent_provenance import legacy_manifest_bytes
    recovered = hashlib.sha256(legacy_manifest_bytes(manifest, saved)).hexdigest()
    if recovered != saved.owod_manifest_sha256:
        raise ValueError('Cannot reproduce the original manifest SHA-256 exactly')
    return dict(proof='exact_precanonical_serialization_sha256', canonical_manifest_sha256=current,
                original_manifest_sha256=recovered)


def git_identity():
    def read(*args):
        return subprocess.run(['git', *args], cwd=ROOT, check=True, text=True,
                              capture_output=True).stdout.strip()
    return dict(commit=read('rev-parse', 'HEAD'), branch=read('branch', '--show-current'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=256, help='Author test-prefix images; 0 = complete author test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    opt = parser.parse_args()
    if opt.images < 0 or opt.num_workers < 0:
        parser.error('--images and --num-workers must be nonnegative')
    if opt.output.exists():
        parser.error('--output already exists; preserve it and choose a new report path')
    import torch
    from torch.utils.data import DataLoader, Subset
    from models.graph_local.discovery import DiscoveryPostProcess
    from models.owod_benchmark_eval import evaluate_author_benchmark
    from tools.owod.gnn_calibration import build_calibration_dataset, load_detector
    from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
    from tools.owod.task2_learning_probe import state_sha256
    from util.misc import collate_fn

    started = time.monotonic()
    torch.set_num_threads(2)
    source_hash = file_sha256(opt.checkpoint)
    manifest_hash = file_sha256(opt.manifest)
    payload = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    manifest = read_json(opt.manifest)
    proof = validate_checkpoint_identity(opt.manifest, saved, manifest)
    manifest, files = stage_files(opt.manifest, saved.owod_stage)
    identity = dict(checkpoint=str(opt.checkpoint.resolve()), checkpoint_sha256=source_hash,
                    checkpoint_epoch=payload.get('epoch'), source_diagnostic_only=payload.get('diagnostic_only', False),
                    manifest=str(opt.manifest.resolve()), manifest_proof=proof,
                    full_val_sha256=manifest['stages'][saved.owod_stage]['sha256']['full_val'],
                    protocol=manifest['protocol'], task=saved.owod_stage + 1, phase=saved.owod_phase,
                    code=git_identity(), audit_sha256=file_sha256(Path(__file__)),
                    discovery_source_sha256=file_sha256(ROOT / 'models/graph_local/discovery.py'),
                    evaluator_source_sha256=file_sha256(ROOT / 'models/owod_benchmark_eval.py'),
                    method={key: getattr(saved, key, None) for key in
                            ('discovery_version', 'discovery_scoring', 'graph_routing', 'graph_localization',
                             'unknown_threshold', 'owod_max_detections', 'seed')})
    print(json.dumps(identity, ensure_ascii=False), flush=True)
    del payload
    device = torch.device(opt.device)
    model, _criterion = load_detector(saved, opt.checkpoint, device)
    del _criterion
    model.eval().requires_grad_(False)
    before = state_sha256(model)
    dataset = build_calibration_dataset(resolve_image_root(opt.manifest, manifest), files['full_val'])
    dataset_positions = {image_id: index for index, image_id in enumerate(dataset.ids)}
    source_indices = {image['owod_source_id']: dataset_positions[image_id]
                      for image_id, image in dataset.coco.imgs.items()}
    author_sequence = manifest['evaluation']['image_source_ids']
    unique = list(dict.fromkeys(author_sequence))
    selected = unique[:opt.images] if opt.images else unique
    full_test = len(selected) == len(unique)
    sequence = author_sequence if full_test else selected
    kwargs = dict(batch_size=1, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    if opt.num_workers:
        kwargs['multiprocessing_context'] = 'spawn'
    loader = DataLoader(Subset(dataset, [source_indices[source] for source in selected]), **kwargs)
    postprocess = DiscoveryPostProcess(known_ids=saved.owod_known_class_ids,
                                      max_detections=saved.owod_max_detections,
                                      unknown_threshold=saved.unknown_threshold, scoring=saved.discovery_scoring)
    rows, predictions = {}, {}
    inference_started = time.monotonic()
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            image_id = int(targets[0]['image_id'].item())
            size = targets[0]['orig_size'].tolist()
            # Only pixels enter the model. No criterion, teacher, miner, EMA
            # or optimizer sees the evaluation labels.
            outputs = model(images.to(device))
            prediction = {key: value.cpu() for key, value in
                          postprocess(outputs, targets[0]['orig_size'][None].to(device))[0].items()}
            source = selected[index]
            rows[source] = measure_image(outputs, prediction, dataset.coco.imgToAnns[image_id],
                                         saved.owod_known_class_ids, size, saved.unknown_threshold,
                                         manifest['evaluation']['iou_threshold'])
            predictions[image_id] = prediction
            if (index + 1) % 50 == 0 or index + 1 == len(selected):
                print(f'Audited {index + 1}/{len(selected)} unique author images; '
                      f'inference seconds={time.monotonic() - inference_started:.1f}', flush=True)
    metrics = None
    if full_test:
        metrics = evaluate_author_benchmark(
            predictions, dataset.coco, saved.owod_known_class_ids, saved.owod_previous_class_ids,
            saved.owod_current_class_ids, manifest['evaluation'], saved.unknown_threshold)
    if before != state_sha256(model) or source_hash != file_sha256(opt.checkpoint):
        raise ValueError('Model/checkpoint changed during audit; use an immutable numbered checkpoint')
    if manifest_hash != file_sha256(opt.manifest):
        raise ValueError('Manifest changed during audit')
    if identity['full_val_sha256'] != file_sha256(files['full_val']):
        raise ValueError('Evaluation annotations changed during audit')
    report = dict(diagnostic_only=True, read_only=True, complete_author_test=full_test,
                  identity=identity, unique_images=len(selected), sequence_entries=len(sequence),
                  image_source_ids=sequence, totals=summarize(rows, sequence),
                  author_metrics=metrics, per_image=rows, seconds=time.monotonic() - started,
                  notes=[
                      'Coverage counts any box per GT; it is not one-to-one U-Recall or AP.',
                      'objectness/novelty gate coverage are separate necessary conditions, not sequential attribution.',
                      'joint_score_gate to emitted_unknown coverage measures loss at the actual shared Top-K.',
                      'Prefix runs never report official metrics. Full runs preserve author sequence multiplicity.',
                      'Unknown local TP/error bins use per-image score ties; official metrics use global author sorting.',
                      'Unknown other includes localization/unannotated content and is not certain background.',
                      'Same-box dual output is evidence to inspect, not the effect of a suppression intervention.',
                      'Evaluation future labels are for diagnosis only; do not select thresholds or epochs from this report.',
                      'A read-only evaluation does not establish improved training or unknown-to-new transfer.'])
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = opt.output.with_suffix(opt.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(opt.output)
    print(json.dumps(dict(report=str(opt.output.resolve()), totals=report['totals'],
                          author_metrics=metrics), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
