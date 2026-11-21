"""Locate missed low-frequency S classes in the frozen detector's native query scores."""
from collections import Counter
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


PARENT = DURATION.with_name('s-instance-task2-increment-tail-positives')
OUTPUT = DURATION.with_name('s-instance-task2-tail-query-ranking')
EXPECTED = '98c4aa7785b34f5af6988fd559669fa4c710d1f06d914854a436500649a6d785'
CRITERION = ('At each of the four slice/checkpoint pairs, at least half of the low-frequency GT missed by '
             'native correct-class Top-50 must have a query with VOC IoU > 0.5. Of these covered misses at '
             'least half must be covered by their own class top-1 query, with recoveries in at least two classes. '
             'Rank within class by the native unrounded focal score, resolving ties by lower query index. '
             'These are GT coverage counts, separate from native one-to-one VOC matches.')


def prepare():
    source = read_json(PARENT/'plan.json')
    result = read_json(PARENT/'comparison.json')
    manifest = read_json(MANIFEST)
    if (file_sha256(PARENT/'comparison.json') != EXPECTED or result['status'] != 'complete' or result['follow_up']
            or result['plan_sha256'] != file_sha256(PARENT/'plan.json')
            or file_sha256(MANIFEST) != source['identity']['manifest_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpus'][0] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed training evidence, evaluation protocol or environment')
    checkpoints = {'control': source['control']['finetune'],
                   'candidate': dict(path=str(PARENT/'finetune/checkpoint.pth'),
                                     sha256=result['phases']['finetune']['checkpoint_sha256'])}
    for entry in checkpoints.values():
        path = Path(entry['path'])
        if file_sha256(path) != entry['sha256'] or read_json(path.parent/'training_complete.json')['last_epoch'] != 0:
            raise ValueError(f'Changed finetune endpoint: {path}')
    slices = {}
    excluded = set(manifest['evaluation']['image_source_ids'])
    for name, prefix, order_field, suffix in [('original', '', 'eval_order_sha256', ''),
                                               ('independent', 'independent_', 'independent_order_sha256', '-independent')]:
        ann = Path(source[prefix+'annotation'])
        order = [im['owod_source_id'] for im in read_json(ann)['images']]
        if (file_sha256(ann) != source[prefix+'annotation_sha256'] or len(order) != 1024 or len(set(order)) != 1024
                or set(order).intersection(excluded) or sha256('\n'.join(order).encode()).hexdigest() != source[order_field]):
            raise ValueError(f'Changed validation slice: {name}')
        excluded.update(order)
        refs = {}
        for arm, checkpoint in checkpoints.items():
            ref = PARENT/f'{arm}-finetune{suffix}-heldout.json'
            d = read_json(ref)
            if d['plan_sha256'] != result['plan_sha256'] or d['checkpoint_sha256'] != checkpoint['sha256']:
                raise ValueError(f'Unpaired reference: {ref}')
            refs[arm] = dict(path=str(ref), sha256=file_sha256(ref))
        slices[name] = dict(annotation=str(ann), annotation_sha256=file_sha256(ann), order_sha256=source[order_field],
                            references=refs)
    plan = dict(protocol=source['protocol'], seed=source['seed'], paper_comparable=False,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                device='0', gpu=torch.cuda.get_device_name(), batch_size=2, torch_version=source['torch_version'],
                manifest_sha256=file_sha256(MANIFEST), parent_comparison_sha256=EXPECTED, slices=slices,
                checkpoints=checkpoints, tail_class_ids=source['tail_class_ids'], criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest


def record_targets(rows, coco, tail, threshold, outputs, targets, batch_images, score_tables):
    (scores, categories, selected), = score_tables
    columns = {c: i for i, c in enumerate(categories.tolist())}
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    for batch, image_id in batch_images.items():
        annotations = [a for a in coco.imgToAnns.get(image_id, []) if a['category_id'] in tail and not a.get('difficult')]
        if not annotations:
            continue
        h, w = targets[batch]['orig_size'].tolist()
        scaled = boxes[batch] * boxes.new_tensor([w, h, w, h])
        query_boxes = np.asarray([voc_prediction_box(b) for b in scaled.cpu().tolist()])
        native = scores[batch].cpu().numpy()
        positions = selected[batch].cpu().numpy()
        for ann in annotations:
            column = columns[ann['category_id']]
            ious = overlaps(np.asarray(ann['voc_bbox']), query_boxes)
            covered = ious > threshold
            selected_queries = positions[positions % len(columns) == column] // len(columns)
            row = dict(image=coco.imgs[image_id]['owod_source_id'], annotation_id=ann['id'], category=ann['category_id'],
                       best_iou=float(ious.max()), native_hit=bool(covered[selected_queries].any()),
                       within_class_rank=None, global_rank=None)
            if covered.any():
                order = np.argsort(-native[:, column], kind='stable')
                rank = int(np.flatnonzero(covered[order])[0])
                query = int(order[rank])
                score = native[query, column]
                row.update(within_class_rank=rank+1, global_rank=1+int((native > score).sum()),
                           query=query, focal_score=float(score))
            rows.append(row)


def summarize(rows, view, reference, tail):
    if view['metrics'] != reference['metrics'] or view['composition'] != reference['composition']:
        raise ValueError('Native AP or prediction composition changed')
    if len({r['annotation_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate low-frequency annotation')
    fp = Counter()
    per_class = {}
    for c in tail:
        original = reference['current_class_stats'][str(c)]
        item = view['current_class_stats'][str(c)]
        selected = [r for r in rows if r['category'] == c]
        native_hits = sum(r['native_hit'] for r in selected)
        if (any(item[k] != original[k] for k in ('ap50','gt','tp','fp'))
                or len(selected) != item['gt'] or native_hits < item['tp']):
            raise ValueError(f'Native VOC replay or annotation coverage differs for class {c}')
        missed = [r for r in selected if not r['native_hit']]
        ranks = [r['within_class_rank'] for r in missed if r['within_class_rank'] is not None]
        families = {k:v['fp'] for k,v in item['fp_sources']['families'].items()}
        if sum(families.values()) != item['fp']:
            raise ValueError('Unaccounted native false positives')
        fp.update(families)
        per_class[str(c)] = dict(gt=len(selected), native_covered=native_hits, missed=len(missed), reachable=len(ranks),
                                 top1=sum(i == 1 for i in ranks), top5=sum(i <= 5 for i in ranks), fp=families)
    totals = {k:sum(r[k] for r in per_class.values()) for k in ('gt','native_covered','missed','reachable','top1','top5')}
    geometry_pass = totals['missed'] > 0 and 2*totals['reachable'] >= totals['missed']
    rank_pass = (totals['reachable'] > 0 and 2*totals['top1'] >= totals['reachable']
                 and sum(r['top1'] > 0 for r in per_class.values()) >= 2)
    return dict(per_class=per_class, totals=totals, fp_sources=dict(fp), geometry_pass=geometry_pass, rank_pass=rank_pass)


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    prev, cur = manifest['stages'][:2]
    tail = set(plan['tail_class_ids'])
    arms = {}
    for name, entry in plan['slices'].items():
        ann = Path(entry['annotation'])
        ds, loader = make_loader(resolve_image_root(MANIFEST,manifest), ann, 2, 2)
        order = [im['owod_source_id'] for im in read_json(ann)['images']]
        spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_fp_sources=True)
        for arm, checkpoint in plan['checkpoints'].items():
            key = name+'-'+arm
            path = OUTPUT/(key+'.json')
            if not path.exists():
                print(f'Evaluating {key} on {len(ds)} images', flush=True)
                model, _ = load_detector(None, Path(checkpoint['path']), torch.device('cuda'))
                rows = []
                observer = partial(record_targets, rows, ds.coco, tail, spec['iou_threshold'])
                result = evaluate_views(model,loader,ds.coco,{'native':(cur['active_classes'],'focal',50)},
                                        spec,prev['classes'],cur['classes'],'cuda',include_current_stats=True,
                                        rank_observer=observer,print_freq=64)
                write_json(path, dict(view=result['native'], rows=rows, plan_sha256=digest, checkpoint_sha256=checkpoint['sha256']))
                del model
            d = read_json(path)
            if d['plan_sha256'] != digest or d['checkpoint_sha256'] != checkpoint['sha256']:
                raise ValueError(f'Changed frozen replay: {key}')
            reference = read_json(Path(entry['references'][arm]['path']))['views']['released']
            arms[key] = summarize(d['rows'],d['view'],reference,tail)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest, arms=arms,
               supported=all(v['geometry_pass'] and v['rank_pass'] for v in arms.values())))
    print('Tail query ranking comparison saved', flush=True)


if __name__ == '__main__':
    main()
