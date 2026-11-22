"""Trace the old annotations lost by a pinned Task 2 finetune."""

import argparse
from collections import Counter, defaultdict
from math import isclose
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
from tools.owod.run_task2_m_s_previous_hard_negatives import BOX_OUTPUT, MANIFESTS, OUTPUT as PARENT, SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = PARENT.with_name('task2-m-s-previous-hard-negative-misses')
SCORES = PARENT.with_name('task2-m-s-previous-hard-negative-scores')
BOX_MISSES = BOX_OUTPUT.with_name('task2-m-s-current-low-iou-box-misses')
BOX_COMPARISON_SHA = 'eced6ebf4702a663073d5f061be4639aa35e2ad4ec797259b991ab8ed8e69610'
BOX_REPORT_SHAS = {
    'm-owodb': 'a5d09197a68df4ad79d7207696c3c7405aa108a4b1365ada55aa2cb13c07ff63',
    's-owodb': 'c963148291c8d2c65769d1d9f316abfd33a12c0f2c1a0bc0425de61e369ae1dd',
}
MISSES_SHA = 'c0dd63960c62b950e7ae1ecd0c4ffc0ada7e03a77306df125487851e665a4786'
MISS_REPORTS = {
    'm-owodb-first': '32da40cc31c6cbcca359382d20d564ccbf95711c05a3406c38c49291005de4b7',
    'm-owodb-second': '7c960c278ca950a82b748bdf8c3fd4c3be3ab389cc33e7cc38f523ed6dd1a98d',
    's-owodb-first': '73a40ed16e7e4908a042c68acaf7a9530a7143615bd1433b58461837dab3c192',
    's-owodb-second': 'db907c0b6303aa5de358aa4e9bf698791ccf87d8ebb1240e075a2fdd82bc8847',
}
HEAD_SHA = 'c9d06aa7a29986bdc98a8f715f2227fd7304a1d444343261602f8a2627e037eb'
OUTCOMES = ('no_compatible_box', 'outside_top50', 'voc_assignment')
COMPONENTS = ('class_probability', 'objectness', 'both')


def match_ids(report):
    return {index for rows in report['previous_matches']['native'].values() for index in rows}


def voc_output_boxes(boxes, target):
    h, w = target['orig_size'].tolist()
    scaled = boxes * boxes.new_tensor([w, h, w, h])
    return [voc_prediction_box(box) for box in scaled.tolist()]


def checked_top_indices(result, report, image, previous, current):
    labels = result['labels'].tolist()
    counts = dict(previous=sum(label in previous for label in labels),
                  current=sum(label in current for label in labels), unknown=labels.count(91))
    if counts != report['per_image'][str(image)]['native']:
        raise ValueError(f'Changed native counts for image {image}')
    return set(result['top_indices'].tolist())


def restore_scores(output, control, image, probabilities, object_queries, label, column, top50):
    control_cls = control['pred_logits'][image, object_queries, label].sigmoid()
    control_obj = control['pred_objectness'][image, object_queries].sigmoid()
    model_cls = output['pred_logits'][image, object_queries, label].sigmoid()
    model_obj = output['pred_objectness'][image, object_queries].sigmoid()
    class_count = probabilities.shape[-1]
    slots = object_queries * class_count + column
    replacements = (control_cls * model_obj, model_cls * control_obj, control_cls * control_obj)
    effects = {}
    for name, values in zip(COMPONENTS, replacements, strict=True):
        patched = probabilities.flatten().index_copy(0, slots, values)
        selected = patched.topk(top50).indices
        effects[name] = bool(torch.isin(selected, slots).any())
    return effects


@torch.inference_mode()
def replay(model, dataset, loader, ref, previous, current, limit, threshold, *, donor=None, pinned=None):
    post = DiscoveryPostProcess(known_ids=previous + current, scoring='focal', max_detections=limit)
    positions = {label: index for index, label in enumerate(previous + current)}
    lost = match_ids(ref['control']) - match_ids(ref['candidate'])
    if pinned is not None:
        if not pinned.keys() <= lost:
            raise ValueError('Rank misses differ from the frozen lost annotations')
        lost = pinned.keys()
    wanted = defaultdict(list)
    for ann_id in sorted(lost):
        ann = dataset.coco.anns[ann_id]
        if ann.get('difficult') or ann['category_id'] not in previous:
            raise ValueError(f'Unexpected lost previous-class annotation: {ann_id}')
        wanted[ann['image_id']].append(ann)
    rows = []
    check_control = donor is not None and pinned is None
    for step, (samples, targets) in enumerate(loader, 1):
        images = [int(target['image_id']) for target in targets]
        if any(i in wanted for i in images):
            samples = samples.to('cuda')
            output = model(samples)
            control = donor(samples) if donor is not None else None
            probabilities, _, _ = post.candidate_scores(output)
            size = torch.stack([t['orig_size'] for t in targets]).cuda()
            results = post(output, size, include_top_indices=True)
            control_results = post(control, size, include_top_indices=True) if check_control else None
            boxes = box_cxcywh_to_xyxy(output['pred_boxes'])
            control_boxes = box_cxcywh_to_xyxy(control['pred_boxes']) if control is not None else None
            for i, image in enumerate(images):
                if image not in wanted:
                    continue
                voc_boxes = voc_output_boxes(boxes[i], targets[i])
                control_voc = voc_output_boxes(control_boxes[i], targets[i]) if control_boxes is not None else None
                class_count = probabilities.shape[-1]
                top = checked_top_indices(results[i], ref['candidate'], image, previous, current)
                if check_control:
                    control_top = checked_top_indices(control_results[i], ref['control'], image, previous, current)
                for ann in wanted[image]:
                    target_box = np.asarray(ann['voc_bbox'])
                    ious = overlaps(target_box, voc_boxes)
                    compatible = np.flatnonzero(ious > threshold).tolist()
                    label = ann['category_id']
                    column = positions[label]
                    gold_indices = [query * class_count + column for query in compatible]
                    emitted = bool(top.intersection(gold_indices))
                    best_score = float(probabilities[i].flatten()[gold_indices].max()) if compatible else None
                    cutoff = float(results[i]['scores'][-1])
                    if compatible and not emitted and best_score > cutoff:
                        raise ValueError(f'Inconsistent native selection: {ann["id"]}')
                    outcome = 'voc_assignment' if emitted else 'outside_top50' if compatible else 'no_compatible_box'
                    row = dict(annotation=ann['id'], image=image, category=ann['category_id'], outcome=outcome,
                               max_iou=float(ious.max()), compatible_queries=len(compatible),
                               best_same_class_score=best_score, top50_cutoff=cutoff)
                    if check_control:
                        control_iou = overlaps(target_box, control_voc)
                        selected = [control_iou[slot // class_count] for slot in control_top
                                    if slot % class_count == column]
                        best_selected = float(max(selected, default=0.))
                        if best_selected <= threshold:
                            raise ValueError(f'Control lost its qualifying Top-50 box: {ann["id"]}')
                        row.update(control_max_iou=float(control_iou.max()),
                                   control_best_top50_same_class_iou=best_selected)
                    elif control_voc is not None:
                        original = pinned[ann['id']]
                        if (outcome != 'outside_top50' or len(compatible) != original['compatible_queries']
                                or not isclose(cutoff, original['top50_cutoff'], rel_tol=1e-5)
                                or not isclose(best_score, original['best_same_class_score'], rel_tol=1e-5)):
                            raise ValueError(f'Changed frozen rank miss: {ann["id"]}')
                        donor_iou = overlaps(target_box, control_voc)
                        aligned = probabilities.new_tensor([q for q in compatible if donor_iou[q] > threshold],
                                                           dtype=torch.long)
                        effects = {k: False for k in COMPONENTS}
                        if aligned.numel():
                            effects = restore_scores(output, control, i, probabilities[i], aligned, label,
                                                     column, limit)
                        row.update(aligned_queries=int(aligned.numel()), recovered=effects)
                    rows.append(row)
        if step % 128 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    if len(rows) != len(lost) or {row['annotation'] for row in rows} != lost:
        raise ValueError('Incomplete lost match replay')
    return rows


def main(score_components=False, low_iou_boxes=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    if low_iou_boxes:
        parent_output, output = BOX_OUTPUT, BOX_MISSES
        source_sha, reference_shas, expected_count = BOX_COMPARISON_SHA, BOX_REPORT_SHAS, 216
    else:
        parent_output, output = PARENT, OUTPUT
        source_sha, reference_shas, expected_count = PARENT_SHA, REFERENCE_SHAS, 90
    source = verified(parent_output / 'comparison.json', source_sha)
    parent = verified(parent_output / 'plan.json', source['plan_sha256'])
    count = sum(row['top50_voc_matches']['previous']['lost']
                for folds in source['protocols'].values() for row in folds.values())
    if count != expected_count:
        raise ValueError('Changed lost target population')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256=source_sha, source_plan_sha256=source['plan_sha256'],
                reference_shas=reference_shas, outcomes=list(OUTCOMES), lost_matches=count, paper_comparable=False,
                decision='Prefer a mechanism only if its outcome covers at least half of lost matches in each protocol')
    if low_iou_boxes:
        if source['status'] != 'complete' or source['follow_up'] or parent['max_detections'] != 50:
            raise ValueError('Expected the completed low-IoU box screen with native Top-50')
        plan.update(max_detections=parent['max_detections'], eval_batch=parent['eval_batch'],
                    replay_control=True)
    else:
        head = verified(HEAD / 'comparison.json', HEAD_SHA)
        if head['status'] != 'complete' or any(
                row['old_rows_explain_most'] for folds in head['protocols'].values() for row in folds.values()):
            raise ValueError('Expected an inconclusive classifier-row attribution')
        plan.update(head_sha256=HEAD_SHA)
    if score_components:
        output = SCORES
        misses = verified(OUTPUT / 'comparison.json', MISSES_SHA)
        if misses['status'] != 'complete' or misses['preferred'] != ['outside_top50']:
            raise ValueError('Expected completed rank-miss attribution')
        plan.update(miss_report_shas=MISS_REPORTS, misses_sha256=MISSES_SHA,
                    outcomes=list(COMPONENTS), lost_matches=58,
                    decision='One factor must recover at least half of all rank misses in each protocol. '
                             'Per-target replay transfers only the selected class entry of queries with VOC IoU > 0.5 in both arms.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    results = {}
    for protocol, job in parent['jobs'].items():
        report = verified(parent_output / f'{protocol}.json', reference_shas[protocol])
        candidate = parent_output / protocol / 'finetune/checkpoint.pth'
        if file_sha256(candidate) != report['candidate_sha256']:
            raise ValueError(f'Changed candidate endpoint: {protocol}')
        manifest = verified(MANIFESTS[protocol], job['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        model, _ = load_detector(None, candidate, torch.device('cuda'))
        model.eval()
        donor = None
        if score_components or low_iou_boxes:
            control = Path(job['endpoints']['finetune']['path'])
            if file_sha256(control) != report['control_sha256']:
                raise ValueError(f'Changed control endpoint: {protocol}')
            donor, _ = load_detector(None, control, torch.device('cuda'))
            donor.eval()
        results[protocol] = {}
        for fold, cohort in job['folds'].items():
            label = f'{protocol}-{fold}'
            marker = output / f'{label}.json'
            pinned = None
            if score_components:
                previous_report = verified(OUTPUT / marker.name, MISS_REPORTS[label])
                pinned = {r['annotation']: r for r in previous_report['rows'] if r['outcome'] == 'outside_top50'}
            reference = {arm: report[arm][fold] for arm in ('control', 'candidate')}
            expected = len(pinned) if score_components else source['protocols'][protocol][fold]['top50_voc_matches']['previous']['lost']
            if low_iou_boxes:
                lost = match_ids(reference['control']) - match_ids(reference['candidate'])
                if len(lost) != expected:
                    raise ValueError(f'Changed lost annotation population: {label}')
            if not marker.exists():
                released = SOURCE / cohort['released']['file']
                if file_sha256(released) != cohort['released']['sha256']:
                    raise ValueError(f'Changed released annotations: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold}', flush=True)
                dataset, loader = make_loader(root, released, parent['eval_batch'], 2)
                rows = replay(model, dataset, loader, reference,
                              job['previous'], job['current'], parent['max_detections'],
                              manifest['evaluation']['iou_threshold'], donor=donor, pinned=pinned)
                if score_components:
                    counts = {k: sum(r['recovered'][k] for r in rows) for k in COMPONENTS}
                    counts.update(aligned=sum(r['aligned_queries'] > 0 for r in rows), rank_misses=len(rows))
                else:
                    counts = dict(Counter(r['outcome'] for r in rows))
                write_json(marker, dict(plan_sha256=digest, rows=rows, counts=counts))
            row = read_json(marker)
            if row['plan_sha256'] != digest or len(row['rows']) != expected:
                raise ValueError(f'Changed outcome report: {protocol}/{fold}')
            if low_iou_boxes and {target['annotation'] for target in row['rows']} != lost:
                raise ValueError(f'Changed replayed annotations: {label}')
            results[protocol][fold] = row['counts']
    outcomes = (*COMPONENTS, 'aligned', 'rank_misses') if score_components else OUTCOMES
    pooled = {protocol: {k: sum(fold.get(k, 0) for fold in folds.values()) for k in outcomes}
              for protocol, folds in results.items()}
    decision_outcomes = COMPONENTS[:2] if score_components else OUTCOMES
    preferred = [k for k in decision_outcomes
                 if all(row[k] * 2 >= (row['rank_misses'] if score_components else sum(row.values()))
                        for row in pooled.values())]
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                                pooled=pooled, preferred=preferred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--score-components', action='store_true')
    modes.add_argument('--low-iou-boxes', action='store_true')
    args = parser.parse_args()
    main(args.score_components, args.low_iou_boxes)
