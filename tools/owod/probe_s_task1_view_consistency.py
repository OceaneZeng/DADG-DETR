"""Decompose the frozen S Task 1 student–EMA agreement gate."""

from collections import Counter
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task1_novelty_budget import PROTOCOL
from tools.owod.probe_s_task1_pseudo_coverage import OUTPUT as SOURCE
from tools.owod.probe_task1_discovery_gates import candidate_batches, frozen_miner
from tools.owod.probe_task1_paired_discovery import LOCKED, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, OUTPUT as DATA
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SOURCE.with_name('s-task1-view-consistency')
SOURCE_PLAN = '962688e98d94edb3ac9530ed1a81eed8f0f8bed4e314f037675fc4f83175afc6'
SOURCE_RESULT = '54dcfc6a029e4d26cf2a80d2a8400e3f2e0e396f4c3fe757b7d878a1c126b214'
SOURCE_REPORTS = {
    'first': '278666413feb71fa75fd33d66ce922e827fc131d936c0f12b1baf515564a7afc',
    'second': '78a6e94af2b5e166f67bb224ab164bdc30322a48b5cb388e76bd6fed18904034',
}


@torch.inference_mode()
def measure(model, criterion, ema, dataset, loader, reference):
    rows = []
    threshold = criterion.consistency_threshold
    for step, (targets, student, gates, mined, teacher) in enumerate(
            candidate_batches(model, criterion, ema, loader, include_teacher=True), 1):
        for i, target in enumerate(targets):
            image_id = int(target['image_id'])
            anns = [ann for ann in dataset.coco.imgToAnns[image_id] if ann['id'] in reference]
            if not anns:
                continue
            h, w = target['orig_size'].tolist()
            eligible = torch.where(gates[i]['eligible'] & gates[i]['salient'])[0]
            student_boxes = student['pred_boxes'][i, eligible]
            teacher_boxes = mined['teacher_boxes'][i]
            student_voc = voc_coords(student_boxes, w, h)
            teacher_voc = voc_coords(teacher_boxes, w, h)
            inter = box_iou(box_cxcywh_to_xyxy(student_boxes.float()),
                            box_cxcywh_to_xyxy(teacher_boxes.float()))[0]
            mapping = mined['matches'][i, eligible]
            safe = mapping.clamp_min(0)
            cosine = F.cosine_similarity(student['query_features'][i, eligible].float(),
                                         teacher['query_features'][i, safe].float())
            assigned_iou = inter[torch.arange(len(eligible), device=inter.device), safe]
            score = torch.where(mapping >= 0, assigned_iou * cosine.clamp(0, 1), 0)
            torch.testing.assert_close(score, mined['consistency'][i, eligible])
            consistent = gates[i]['consistent'][eligible].tolist()
            confident = gates[i]['confident'][eligible].tolist()
            selected = mined['pseudo'][i, eligible].tolist()
            queries = eligible.tolist()
            partners = mapping.tolist()
            ious, cosines, scores = assigned_iou.tolist(), cosine.tolist(), score.tolist()
            matrix = inter.cpu().numpy()
            for ann in anns:
                truth = np.asarray(ann['voc_bbox'])
                matches = np.flatnonzero(overlaps(truth, student_voc) > .5)
                teacher_iou = overlaps(truth, teacher_voc)
                options = []
                for j in matches:
                    teacher_id = partners[j]
                    options.append(dict(query=queries[j], partner=teacher_id,
                        partner_gt_iou=float(teacher_iou[teacher_id]) if teacher_id >= 0 else None,
                        view_iou=ious[j] if teacher_id >= 0 else None,
                        cosine=cosines[j] if teacher_id >= 0 else None, score=scores[j],
                        any_teacher_view_iou=float(matrix[j].max()),
                        any_accurate_teacher_view_iou=float(matrix[j][teacher_iou > .5].max())
                        if (teacher_iou > .5).any() else None,
                        consistent=consistent[j], confident=confident[j], selected=selected[j]))
                old = reference[ann['id']]
                actual = dict(student_salient=bool(options), **{
                    name: any(row[name] for row in options) for name in ('consistent','confident','selected')})
                if actual != {k: old[k] for k in actual}:
                    raise ValueError(f'Changed native gate path for target {ann["id"]}')
                rows.append(dict(annotation_id=ann['id'], image_id=image_id,
                                 best_teacher_gt_iou=float(teacher_iou.max()), candidates=options))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(reference) or {r['annotation_id'] for r in rows} != set(reference):
        raise ValueError('Incomplete target replay')
    return rows


def summarize(rows, threshold):
    counts = Counter()
    for row in rows:
        counts['targets'] += 1
        options = row['candidates']
        best = max(options, key=lambda x: x['score'])
        if best['consistent']:
            counts['passes_consistency'] += 1
        else:
            counts['fails_consistency'] += 1
            if row['best_teacher_gt_iou'] <= .5:
                counts['no_accurate_teacher_box'] += 1
            elif not any((x['partner_gt_iou'] or 0) > .5 for x in options):
                counts['accurate_teacher_unassigned'] += 1
            counts['all_assigned_view_iou_below_threshold'] += all(
                (x['view_iou'] or 0) < threshold for x in options)
            counts['all_assigned_cosine_below_threshold'] += all(
                (x['cosine'] or 0) < threshold for x in options)
            counts['passes_iou_alone'] += any((x['view_iou'] or 0) >= threshold for x in options)
            counts['any_accurate_alternative_can_pass_view_iou'] += any(
                (x['any_accurate_teacher_view_iou'] or 0) >= threshold for x in options)
    matched = [x for row in rows for x in row['candidates'] if x['partner'] >= 0]
    return dict(counts=dict(counts), candidate_pairs=len(matched),
                medians={k: median(x[k] for x in matched) for k in ('view_iou','cosine','score')})


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE/'plan.json', SOURCE_PLAN)
    completed = verified(SOURCE/'comparison.json', SOURCE_RESULT)
    checkpoint = prior['checkpoints']['parent12']
    path = Path(checkpoint['path'])
    if (completed['status'] != 'complete' or completed['plan_sha256'] != SOURCE_PLAN
            or file_sha256(path) != checkpoint['sha256']):
        raise ValueError('Changed frozen evidence')
    manifest_path = MANIFESTS[PROTOCOL]
    manifest = verified(manifest_path, LOCKED[PROTOCOL]['manifest_sha256'])
    root = resolve_image_root(manifest_path, manifest)
    model, criterion, ema = frozen_miner(path, manifest)
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_plan=SOURCE_PLAN, source_result=SOURCE_RESULT, reports=SOURCE_REPORTS,
                checkpoint=checkpoint, inputs=prior['slices'], batch_size=prior['batch_size'],
                consistency_threshold=criterion.consistency_threshold, paper_comparable=False)
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    results = {}
    for fold, spec in plan['inputs'].items():
        annotation = DATA/spec['file']
        verified(annotation, spec['sha256'])
        old = verified(SOURCE/f'parent12-{fold}.json', SOURCE_REPORTS[fold])
        if old['plan_sha256'] != SOURCE_PLAN:
            raise ValueError(f'Changed prior provenance in {fold}')
        reference = {r['annotation_id']: r['coverage'] for r in old['rows'] if not r['released']
                     and r['size'] != '32_sq_plus' and r['coverage']['student_salient']}
        dest = OUTPUT/f'{fold}.json'
        if not dest.exists():
            ds, loader = make_loader(root, annotation, plan['batch_size'], 2)
            print(f'Evaluating {fold} with original student–EMA pairs', flush=True)
            write_json(dest, dict(plan_sha256=digest, rows=measure(model, criterion, ema, ds, loader, reference)))
        report = read_json(dest)
        if report['plan_sha256'] != digest or {r['annotation_id'] for r in report['rows']} != set(reference):
            raise ValueError(f'Unpaired report {fold}')
        results[fold] = summarize(report['rows'], plan['consistency_threshold'])
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=results))


if __name__ == '__main__':
    main()
