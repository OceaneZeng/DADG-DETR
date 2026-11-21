"""Pair geometric coverage and class-only ranks for the S duration holdout."""
import os
from pathlib import Path
from statistics import median
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_score_competition import ARMS, MANIFEST, OUTPUT as SOURCE
from tools.owod.screening_data import write_json
from tools.owod.supervision_probe import current_candidate_rank_rows
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SOURCE.with_name('s-instance-task2-paired-targets')
IOU = .5
DOMINANT_FRACTION = .6


def candidates(model, samples, sizes, processor):
    outputs = model(samples)
    outputs['pred_novelty'] = torch.full_like(outputs['pred_novelty'], -torch.inf)
    scores, categories, _ = processor.candidate_scores(outputs)
    selected = processor(outputs, sizes, include_top_indices=True)
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    return boxes, (scores, categories, [item['top_indices'] for item in selected])


def target_rows(target, image_index, views):
    labels = target['labels'].to('cuda')
    truth = box_cxcywh_to_xyxy(target['boxes'].to('cuda'))
    per_arm = {}
    for arm, (boxes, scores) in views.items():
        overlap = box_iou(boxes[image_index], truth)[0]
        matched = iter(current_candidate_rank_rows(labels, overlap, image_index, [scores]).tolist())
        per_arm[arm] = []
        for iou in overlap.amax(0).tolist():
            _, rank, hit = next(matched) if iou > IOU else (None, None, False)
            per_arm[arm].append(dict(best_iou=iou, best_class_rank=rank, top50_hit=bool(hit)))
    image_id = int(target['image_id'])
    return [dict(image_id=image_id, gt_index=index, class_id=label,
                 **{arm: per_arm[arm][index] for arm in ARMS})
            for index, label in enumerate(target['labels'].tolist())]


def summarize(rows):
    short, long = ARMS
    matched = [(r[short], r[long]) for r in rows]
    common = [(a, b) for a, b in matched if a['best_iou'] > IOU and b['best_iou'] > IOU]
    lost = [(a, b) for a, b in matched if a['top50_hit'] and not b['top50_hit']]
    return dict(gt=len(rows), box_coverage={arm: sum(r[arm]['best_iou'] > IOU for r in rows) for arm in ARMS},
                covered_both=len(common), coverage_lost=sum(a['best_iou'] > IOU >= b['best_iou'] for a, b in matched),
                coverage_gained=sum(b['best_iou'] > IOU >= a['best_iou'] for a, b in matched),
                median_paired_iou_change=median(b['best_iou']-a['best_iou'] for a, b in common) if common else None,
                rank_worsened=sum(b['best_class_rank'] > a['best_class_rank'] for a, b in common),
                rank_improved=sum(b['best_class_rank'] < a['best_class_rank'] for a, b in common),
                top50_hits={arm: sum(r[arm]['top50_hit'] for r in rows) for arm in ARMS},
                top50_lost=len(lost), top50_gained=sum(b['top50_hit'] and not a['top50_hit'] for a, b in matched),
                top50_lost_no_box=sum(b['best_iou'] <= IOU for _, b in lost),
                top50_lost_box_kept_rank_worse=sum(b['best_class_rank'] is not None and
                    b['best_class_rank'] > a['best_class_rank'] for a, b in lost))


@torch.no_grad()
def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for frozen paired inference')
    source_plan, source_report = read_json(SOURCE / 'plan.json'), read_json(SOURCE / 'comparison.json')
    manifest, annotation = read_json(MANIFEST), Path(source_plan['annotation'])
    if (source_report['status'] != 'complete' or source_report['follow_up'] or
            source_report['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or
            file_sha256(MANIFEST) != source_plan['manifest_sha256'] or
            file_sha256(annotation) != source_plan['annotation_sha256']):
        raise ValueError('Expected the completed S holdout scoring comparison and its registered inputs')
    inputs = source_plan['checkpoints']
    expected = {}
    for arm in ARMS:
        view_path = SOURCE / f'{arm}.json'
        report = read_json(view_path)
        if (file_sha256(Path(inputs[arm]['path'])) != inputs[arm]['sha256'] or
                report['plan_sha256'] != source_report['plan_sha256'] or
                report['checkpoint_sha256'] != inputs[arm]['sha256']):
            raise ValueError(f'Changed frozen source for {arm}')
        expected[arm] = {group: report['view'][f'raw_{group}_box_recall_iou50'] for group in ('old', 'current')}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', seed=source_plan['seed'], images=source_plan['images'], paper_comparable=False,
                source_plan_sha256=source_report['plan_sha256'], source_result_sha256=file_sha256(SOURCE / 'comparison.json'),
                manifest_sha256=source_plan['manifest_sha256'], annotation=str(annotation),
                annotation_sha256=source_plan['annotation_sha256'],
                checkpoints=inputs, expected_coverage=expected, gpu=torch.cuda.get_device_name(),
                device=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=str(torch.__version__), batch_size=2,
                max_detections=50, iou_threshold=IOU, dominant_fraction=DOMINANT_FRACTION,
                criterion='Among old targets lost from class-only Top-50, at least 60% lose all geometric coverage '
                          'or at least 60% retain coverage with a worse true-class rank. Otherwise inspect precision.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    plan_hash = file_sha256(plan_path)
    device = torch.device('cuda')
    models = {arm: load_detector(None, Path(inputs[arm]['path']), device)[0].eval() for arm in ARMS}
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    known = manifest['stages'][1]['active_classes']
    previous = set(manifest['stages'][0]['active_classes'])
    groups = {'old': previous, 'current': set(manifest['stages'][1]['classes'])}
    processor = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal_class_only')
    seen, rows = set(), []
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to(device)
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        views = {arm: candidates(model, samples, sizes, processor) for arm, model in models.items()}
        for image_index, target in enumerate(targets):
            image_id = int(target['image_id'])
            if image_id in seen:
                raise ValueError(f'Duplicate diagnostic image: {image_id}')
            seen.add(image_id)
            rows.extend(target_rows(target, image_index, views))
        if step % 64 == 0 or step == len(loader):
            print(f'Paired inference {step}/{len(loader)} batches, {len(seen)} images', flush=True)
    totals = {group: summarize([r for r in rows if r['class_id'] in classes]) for group, classes in groups.items()}
    if (len(seen) != plan['images'] or sum(report['gt'] for report in totals.values()) != len(rows) or
            any(totals[group]['gt'] != expected[arm][group]['gt'] or
                totals[group]['box_coverage'][arm] != expected[arm][group]['hits']
                for group in groups for arm in ARMS)):
        raise ValueError('Paired raw coverage differs from the archived scoring comparison')
    old = totals['old']
    attribution = 'inspect_precision'
    if old['top50_lost']:
        if old['top50_lost_no_box'] / old['top50_lost'] >= DOMINANT_FRACTION:
            attribution = 'geometry'
        elif old['top50_lost_box_kept_rank_worse'] / old['top50_lost'] >= DOMINANT_FRACTION:
            attribution = 'class_rank'
    write_json(OUTPUT / 'targets.json', dict(plan_sha256=plan_hash, rows=rows))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
                                                images=len(seen), totals=totals, next_focus=attribution))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
