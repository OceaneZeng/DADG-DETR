"""Trace the old annotations lost by the Task 2 hard-negative finetune."""

from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_previous_hard_negative_head import OUTPUT as HEAD, PARENT_SHA, REFERENCE_SHAS
from tools.owod.run_task2_m_s_previous_hard_negatives import MANIFESTS, OUTPUT as PARENT, SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = PARENT.with_name('task2-m-s-previous-hard-negative-misses')
HEAD_SHA = 'c9d06aa7a29986bdc98a8f715f2227fd7304a1d444343261602f8a2627e037eb'
OUTCOMES = ('no_compatible_box', 'outside_top50', 'voc_assignment')


def match_ids(report):
    return {index for rows in report['previous_matches']['native'].values() for index in rows}


@torch.inference_mode()
def replay(model, dataset, loader, ref, previous, current, limit, threshold):
    post = DiscoveryPostProcess(known_ids=previous + current, scoring='focal', max_detections=limit)
    positions = {label: index for index, label in enumerate(previous + current)}
    lost = match_ids(ref['control']) - match_ids(ref['candidate'])
    wanted = defaultdict(list)
    for ann_id in sorted(lost):
        ann = dataset.coco.anns[ann_id]
        if ann.get('difficult') or ann['category_id'] not in previous:
            raise ValueError(f'Unexpected lost previous-class annotation: {ann_id}')
        wanted[ann['image_id']].append(ann)
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        images = [int(target['image_id']) for target in targets]
        if any(i in wanted for i in images):
            output = model(samples.to('cuda'))
            probabilities, _, _ = post.candidate_scores(output)
            size = torch.stack([t['orig_size'] for t in targets]).cuda()
            results = post(output, size, include_top_indices=True)
            boxes = box_cxcywh_to_xyxy(output['pred_boxes'])
            for i, image in enumerate(images):
                if image not in wanted:
                    continue
                labels = results[i]['labels'].tolist()
                counts = dict(previous=sum(label in previous for label in labels),
                              current=sum(label in current for label in labels), unknown=labels.count(91))
                if counts != ref['candidate']['per_image'][str(image)]['native']:
                    raise ValueError(f'Changed native candidate counts: {image}')
                h, w = targets[i]['orig_size'].tolist()
                voc_boxes = [voc_prediction_box(box) for box in (boxes[i] * boxes.new_tensor([w, h, w, h])).tolist()]
                class_count = probabilities.shape[-1]
                top = set(results[i]['top_indices'].tolist())
                for ann in wanted[image]:
                    ious = overlaps(np.asarray(ann['voc_bbox']), voc_boxes)
                    compatible = np.flatnonzero(ious > threshold).tolist()
                    gold_indices = [query * class_count + positions[ann['category_id']] for query in compatible]
                    emitted = bool(top.intersection(gold_indices))
                    best_score = float(probabilities[i].flatten()[gold_indices].max()) if compatible else None
                    cutoff = float(results[i]['scores'][-1])
                    if compatible and not emitted and best_score > cutoff:
                        raise ValueError(f'Inconsistent native selection: {ann["id"]}')
                    outcome = 'voc_assignment' if emitted else 'outside_top50' if compatible else 'no_compatible_box'
                    rows.append(dict(annotation=ann['id'], image=image, category=ann['category_id'], outcome=outcome,
                                     max_iou=float(ious.max()), compatible_queries=len(compatible),
                                     best_same_class_score=best_score, top50_cutoff=cutoff))
        if step % 128 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    if len(rows) != len(lost) or {row['annotation'] for row in rows} != lost:
        raise ValueError('Incomplete lost match replay')
    return rows


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    head = verified(HEAD / 'comparison.json', HEAD_SHA)
    source = verified(PARENT / 'comparison.json', PARENT_SHA)
    parent = verified(PARENT / 'plan.json', source['plan_sha256'])
    if head['status'] != 'complete' or any(
            row['old_rows_explain_most'] for folds in head['protocols'].values() for row in folds.values()):
        raise ValueError('Expected an inconclusive classifier-row attribution')
    count = sum(row['top50_voc_matches']['previous']['lost']
                for folds in source['protocols'].values() for row in folds.values())
    if count != 90:
        raise ValueError('Changed lost target population')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                head_sha256=HEAD_SHA, source_sha256=PARENT_SHA, source_plan_sha256=source['plan_sha256'],
                reference_shas=REFERENCE_SHAS, outcomes=list(OUTCOMES), lost_matches=count, paper_comparable=False,
                decision='Prefer a mechanism only if its outcome covers at least half of lost matches in each protocol')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in parent['jobs'].items():
        report = verified(PARENT / f'{protocol}.json', REFERENCE_SHAS[protocol])
        candidate = PARENT / protocol / 'finetune/checkpoint.pth'
        if file_sha256(candidate) != report['candidate_sha256']:
            raise ValueError(f'Changed candidate endpoint: {protocol}')
        manifest = verified(MANIFESTS[protocol], job['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        model, _ = load_detector(None, candidate, torch.device('cuda'))
        model.eval()
        results[protocol] = {}
        for fold, cohort in job['folds'].items():
            marker = OUTPUT / f'{protocol}-{fold}.json'
            if not marker.exists():
                released = SOURCE / cohort['released']['file']
                if file_sha256(released) != cohort['released']['sha256']:
                    raise ValueError(f'Changed released annotations: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold}', flush=True)
                dataset, loader = make_loader(root, released, parent['eval_batch'], 2)
                rows = replay(model, dataset, loader, {arm: report[arm][fold] for arm in ('control', 'candidate')},
                              job['previous'], job['current'], parent['max_detections'],
                              manifest['evaluation']['iou_threshold'])
                write_json(marker, dict(plan_sha256=digest, rows=rows, counts=dict(Counter(r['outcome'] for r in rows))))
            row = read_json(marker)
            if row['plan_sha256'] != digest or len(row['rows']) != source['protocols'][protocol][fold]['top50_voc_matches']['previous']['lost']:
                raise ValueError(f'Changed outcome report: {protocol}/{fold}')
            results[protocol][fold] = row['counts']
    pooled = {protocol: {k: sum(fold.get(k, 0) for fold in folds.values()) for k in OUTCOMES}
              for protocol, folds in results.items()}
    preferred = [k for k in OUTCOMES if all(row[k] * 2 >= sum(row.values()) for row in pooled.values())]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                                pooled=pooled, preferred=preferred))


if __name__ == '__main__':
    main()
