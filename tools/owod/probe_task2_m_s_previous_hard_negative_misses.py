"""Trace the old annotations lost by the Task 2 hard-negative finetune."""

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
from tools.owod.run_task2_m_s_previous_hard_negatives import MANIFESTS, OUTPUT as PARENT, SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = PARENT.with_name('task2-m-s-previous-hard-negative-misses')
SCORES = PARENT.with_name('task2-m-s-previous-hard-negative-scores')
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
    for step, (samples, targets) in enumerate(loader, 1):
        images = [int(target['image_id']) for target in targets]
        if any(i in wanted for i in images):
            samples = samples.to('cuda')
            output = model(samples)
            control = donor(samples) if donor is not None else None
            probabilities, _, _ = post.candidate_scores(output)
            size = torch.stack([t['orig_size'] for t in targets]).cuda()
            results = post(output, size, include_top_indices=True)
            boxes = box_cxcywh_to_xyxy(output['pred_boxes'])
            control_boxes = box_cxcywh_to_xyxy(control['pred_boxes']) if control is not None else None
            for i, image in enumerate(images):
                if image not in wanted:
                    continue
                labels = results[i]['labels'].tolist()
                counts = dict(previous=sum(label in previous for label in labels),
                              current=sum(label in current for label in labels), unknown=labels.count(91))
                if counts != ref['candidate']['per_image'][str(image)]['native']:
                    raise ValueError(f'Changed native candidate counts: {image}')
                voc_boxes = voc_output_boxes(boxes[i], targets[i])
                control_voc = voc_output_boxes(control_boxes[i], targets[i]) if control_boxes is not None else None
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
                    row = dict(annotation=ann['id'], image=image, category=ann['category_id'], outcome=outcome,
                               max_iou=float(ious.max()), compatible_queries=len(compatible),
                               best_same_class_score=best_score, top50_cutoff=cutoff)
                    if control_voc is not None:
                        original = pinned[ann['id']]
                        if (outcome != 'outside_top50' or len(compatible) != original['compatible_queries']
                                or not isclose(cutoff, original['top50_cutoff'], rel_tol=1e-5)
                                or not isclose(best_score, original['best_same_class_score'], rel_tol=1e-5)):
                            raise ValueError(f'Changed frozen rank miss: {ann["id"]}')
                        donor_iou = overlaps(np.asarray(ann['voc_bbox']), control_voc)
                        aligned = probabilities.new_tensor([q for q in compatible if donor_iou[q] > threshold],
                                                           dtype=torch.long)
                        effects = {k: False for k in COMPONENTS}
                        if aligned.numel():
                            label = ann['category_id']
                            effects = restore_scores(output, control, i, probabilities[i], aligned, label,
                                                     positions[label], limit)
                        row.update(aligned_queries=int(aligned.numel()), recovered=effects)
                    rows.append(row)
        if step % 128 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    if len(rows) != len(lost) or {row['annotation'] for row in rows} != lost:
        raise ValueError('Incomplete lost match replay')
    return rows


def main(score_components=False):
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
    output = OUTPUT
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
        report = verified(PARENT / f'{protocol}.json', REFERENCE_SHAS[protocol])
        candidate = PARENT / protocol / 'finetune/checkpoint.pth'
        if file_sha256(candidate) != report['candidate_sha256']:
            raise ValueError(f'Changed candidate endpoint: {protocol}')
        manifest = verified(MANIFESTS[protocol], job['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        model, _ = load_detector(None, candidate, torch.device('cuda'))
        model.eval()
        donor = None
        if score_components:
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
            if not marker.exists():
                released = SOURCE / cohort['released']['file']
                if file_sha256(released) != cohort['released']['sha256']:
                    raise ValueError(f'Changed released annotations: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold}', flush=True)
                dataset, loader = make_loader(root, released, parent['eval_batch'], 2)
                rows = replay(model, dataset, loader, {arm: report[arm][fold] for arm in ('control', 'candidate')},
                              job['previous'], job['current'], parent['max_detections'],
                              manifest['evaluation']['iou_threshold'], donor=donor, pinned=pinned)
                if score_components:
                    counts = {k: sum(r['recovered'][k] for r in rows) for k in COMPONENTS}
                    counts.update(aligned=sum(r['aligned_queries'] > 0 for r in rows), rank_misses=len(rows))
                else:
                    counts = dict(Counter(r['outcome'] for r in rows))
                write_json(marker, dict(plan_sha256=digest, rows=rows, counts=counts))
            row = read_json(marker)
            expected = len(pinned) if score_components else source['protocols'][protocol][fold]['top50_voc_matches']['previous']['lost']
            if row['plan_sha256'] != digest or len(row['rows']) != expected:
                raise ValueError(f'Changed outcome report: {protocol}/{fold}')
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
    parser.add_argument('--score-components', action='store_true')
    main(parser.parse_args().score_components)
