"""Measure native Current AP and its coverage/ranking losses at both S Task 2 phases."""
from hashlib import sha256
from functools import partial
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION
from tools.owod.run_s_task2_paired_targets import OUTPUT as PAIRED, summarize as summarize_paired
from tools.owod.run_s_task2_score_competition import ARMS, OUTPUT as SCORING
from tools.owod.screening_data import write_json
from tools.owod.supervision_probe import current_candidate_rank_rows
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/s-instance-task2-current-phase'
PHASES = ('increment', 'finetune')
FINETUNE_DROP = .005
SOURCES = {
    DURATION: '4f698b6c4c608cb835f063960416ba546af06e557356f623555b4158eee9172f',
    SCORING: 'fce5044b5267e88cc8da470d3636f22527420b6d8670aa685ee4253374fd4d8a',
    PAIRED: 'bba29f03c007a6c4d4d4783f216e915d92ba070a65f2e40c3d9267f4e36c7aab',
}


def summarize(view):
    current = view['native']['current_class_stats']
    ranks = view['current_candidate_ranks']['overall']
    geometry = view['raw_current_box_recall_iou50']
    covered = ranks['gt_with_box']
    hits = ranks['views']['native']['top50_hits']
    if covered != geometry['hits'] or not 0 <= hits <= covered <= geometry['gt']:
        raise ValueError('Inconsistent native coverage partition')
    ceiling = mean(c['recall_ceiling_ap50'] for c in current.values())
    ap = view['native']['metrics']['current_ap50']
    return dict(**view['native']['metrics'], gt=geometry['gt'], no_box=geometry['gt']-covered,
                box_without_correct_top50=covered-hits, top50_hits=hits, macro_fixed_tp_ceiling=ceiling,
                macro_ranking_penalty=ceiling-ap, voc_tp=sum(c['tp'] for c in current.values()),
                voc_fp=sum(c['fp'] for c in current.values()),
                fp_before_first_tp=sum(c['fp_before_first_tp'] or 0 for c in current.values()),
                classes_without_tp=sum(c['tp'] == 0 for c in current.values()),
                fp_in_classes_without_tp=sum(c['fp'] for c in current.values() if not c['tp']))


def record_targets(rows, outputs, targets, batch_images, score_tables):
    """Recover historical per-GT class-only ranks; cross-check hits by decoded class ID."""
    table = score_tables[1]
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    for index, image_id in batch_images.items():
        labels = targets[index]['labels'].to(boxes.device)
        truth = box_cxcywh_to_xyxy(targets[index]['boxes'].to(boxes.device))
        overlap = box_iou(boxes[index], truth)[0]
        ranked = iter(current_candidate_rank_rows(labels, overlap, index, [table]).tolist())
        _, categories, positions = table
        query = positions[index] // len(categories)
        predicted_labels = categories[positions[index] % len(categories)]
        direct = ((overlap[query] > .5) & (predicted_labels[:, None] == labels)).any(0).tolist()
        for gt_index, (label, iou) in enumerate(zip(labels.tolist(), overlap.amax(0).tolist())):
            rank, hit = None, False
            if iou > .5:
                ranked_label, rank, hit = next(ranked)
                if ranked_label != label:
                    raise ValueError('Covered target order differs')
            if bool(hit) != direct[gt_index]:
                raise ValueError('Rank result differs from direct selected query/label match')
            rows.append(dict(image_id=image_id, gt_index=gt_index, class_id=label, best_iou=iou,
                             best_class_rank=rank, top50_hit=bool(hit)))


def corrected_history(current):
    history = read_json(PAIRED / 'targets.json')
    if history['plan_sha256'] != file_sha256(PAIRED / 'plan.json'):
        raise ValueError('Unexpected archived target rows')
    def identity(row):
        return row['image_id'], row['gt_index'], row['class_id']
    corrected = {identity(row): dict(class_id=row['class_id']) for row in history['rows']}
    changes = {}
    for arm in ARMS:
        rows = read_json(OUTPUT / f'{arm}-finetune.json')['class_only_targets']
        if len(rows) != len(corrected) or {identity(row) for row in rows} != corrected.keys():
            raise ValueError('Target identity differs from the paired study')
        for row in rows:
            corrected[identity(row)][arm] = row
        changes[arm] = {}
        for group, is_current in [('current', True), ('old', False)]:
            pairs = [(r[arm], corrected[identity(r)][arm]) for r in history['rows']
                     if (r['class_id'] in current) == is_current]
            if any((a['best_iou'] > .5) != (b['best_iou'] > .5) for a, b in pairs):
                raise ValueError('Historical geometric coverage differs')
            changes[arm][group] = dict(changed_rank=sum(a['best_class_rank'] != b['best_class_rank'] for a,b in pairs),
                                       changed_hit=sum(a['top50_hit'] != b['top50_hit'] for a,b in pairs))
    totals = {}
    for group, is_current in [('old', False), ('current', True)]:
        totals[group] = summarize_paired([r for r in corrected.values() if (r['class_id'] in current) == is_current])
    return dict(changes=changes, totals=totals)


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for four frozen endpoints')
    for folder, digest in SOURCES.items():
        result = read_json(folder / 'comparison.json')
        if (file_sha256(folder / 'comparison.json') != digest or result['status'] != 'complete' or
                result['plan_sha256'] != file_sha256(folder / 'plan.json')):
            raise ValueError(f'Changed source study: {folder}')
    duration = read_json(DURATION / 'comparison.json')['arms']
    scoring = read_json(SCORING / 'plan.json')
    paired = read_json(PAIRED / 'comparison.json')['totals']['current']
    historical_hash = file_sha256(PAIRED / 'targets.json')
    if historical_hash != 'b1008786cd4114c609aad263c271c0c7dee4f7bd2dfa93e5fe7232dd79d64207':
        raise ValueError('The historical paired targets differ from the archived report')
    annotation = Path(scoring['annotation'])
    manifest = read_json(MANIFEST)
    data = read_json(annotation)
    order = [im['owod_source_id'] for im in data['images']]
    current = manifest['stages'][1]['classes']
    previous = manifest['stages'][0]['active_classes']
    known = manifest['stages'][1]['active_classes']
    if (file_sha256(annotation) != scoring['annotation_sha256'] or file_sha256(MANIFEST) != scoring['manifest_sha256']
            or manifest['evaluation']['ap_mode'] != 'voc2007_11point' or len(set(order)) != len(order) or len(order) != 1024
            or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != scoring['eval_order_sha256']
            or {a['category_id'] for a in data['annotations']} != set(known)):
        raise ValueError('Changed release-only holdout, order or evaluation convention')
    checkpoints = {}
    for arm in ARMS:
        for phase in PHASES:
            folder = DURATION / arm / phase
            path = folder / 'checkpoint.pth'
            digest = file_sha256(path)
            epoch = int(arm.removesuffix('epochs'))-1 if phase == 'increment' else 0
            if digest != duration[arm][phase]['checkpoint_sha256'] or read_json(folder/'training_complete.json')['last_epoch'] != epoch:
                raise ValueError(f'Incomplete or changed endpoint: {path}')
            checkpoints[f'{arm}-{phase}'] = dict(path=str(path), sha256=digest)
        if read_json(DURATION / arm / 'finetune/input.json')['--pretrained'] != checkpoints[f'{arm}-increment']['sha256']:
            raise ValueError(f'Broken phase chain: {arm}')

    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', paper_comparable=False, seed=42, gpu=torch.cuda.get_device_name(),
                device=os.environ['CUDA_VISIBLE_DEVICES'], batch_size=2, torch_version=str(torch.__version__),
                manifest_sha256=file_sha256(MANIFEST), annotation=str(annotation), annotation_sha256=file_sha256(annotation),
                eval_order_sha256=scoring['eval_order_sha256'], images=len(order), checkpoints=checkpoints,
                sources={k.name: v for k, v in SOURCES.items()}, historical_targets_sha256=historical_hash,
                max_detections=50, ap_mode=manifest['evaluation']['ap_mode'],
                primary='Native focal Current AP50; compare finetune minus increment at 3 and 6 epochs.',
                criterion='Prioritize balanced finetuning if Current declines by at least 0.50 pp at both durations. '
                          'If neither crosses that threshold inspect increment-stage limitations; if only one crosses '
                          'report duration dependence. Quantify raw coverage, correct-class Top-50 and VOC FP ranking.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    plan_hash = file_sha256(plan_path)
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    spec = dict(manifest['evaluation'], image_source_ids=order, known_only_screening=True, diagnostic_current_precision=True)
    variants = {'native': (known, 'focal', 50), 'known_class_only': (known, 'focal_class_only', 50)}
    rows, differences = {}, {}
    for arm in ARMS:
        rows[arm] = {}
        reference = read_json(SCORING / f'{arm}.json')
        if reference['plan_sha256'] != file_sha256(SCORING / 'plan.json') or reference['checkpoint_sha256'] != checkpoints[f'{arm}-finetune']['sha256']:
            raise ValueError(f'Changed finetune reference: {arm}')
        for phase in PHASES:
            name = f'{arm}-{phase}'
            entry = checkpoints[name]
            path = OUTPUT / f'{name}.json'
            if not path.exists():
                print(f'Evaluating {name} native and class-only held-out views', flush=True)
                model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
                target_rows = []
                observer = None
                if phase == 'finetune':
                    observer = partial(record_targets, target_rows)
                view = evaluate_views(model, loader, dataset.coco, variants, spec, previous, current, 'cuda',
                                      include_current_stats=True, compare_current_ranks=True,
                                      exclude_unknown=('known_class_only',), print_freq=64, rank_observer=observer)
                write_json(path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], view=view,
                                      class_only_targets=target_rows))
                del model
            report = read_json(path)
            if report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != entry['sha256']:
                raise ValueError(f'Invalid stored report: {name}')
            view = report['view']
            rows[arm][phase] = summarize(view)
            if phase == 'finetune':
                for key in variants:
                    if any(view[key][k] != reference['view'][key][k] for k in ('metrics','composition')):
                        raise ValueError(f'Prior finetune replay differs: {name}, {key}')
                ranks = view['current_candidate_ranks']['overall']
                if ranks['gt_with_box'] != paired['box_coverage'][arm]:
                    raise ValueError(f'Historical raw geometric coverage differs: {arm}')
        inc, ft = (rows[arm][p] for p in PHASES)
        differences[arm] = {k:ft[k]-inc[k] for k in ft}
    drops = sum(d['current_ap50'] <= -FINETUNE_DROP for d in differences.values())
    focus = ('increment', 'duration_dependent', 'finetune')[drops]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
               arms=rows, finetune_minus_increment=differences, corrected_class_only=corrected_history(current),
               next_focus=focus))
    print('Current phase comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
