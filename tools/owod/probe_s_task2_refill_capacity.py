"""Measure reachable VOC targets for the fixed same-image, same-class refill slots."""
from collections import Counter
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import evaluate_author_benchmark, overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_shared_query_refill import OUTPUT as SOURCE, SharedQueryRefill
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-refill-capacity')
HASHES = {'original': '94ec77e1ff5b296210803cc7ea1104af2e76be7c85f1403fa56b73de1a91a02d',
          'independent': 'ecd4bee779127c7acb9b146cca5a4c52f2e7cd672028c9e2b6ac9ee37f156856'}
CRITERION = ('Investigate a learned replacement ranker only if on both slices label-assisted, TP-maximizing '
             'fill with unchanged detector scores raises released Current AP50 by at least 1 pp vs native, '
             'and the real refill captures less than half of the maximum additional VOC TPs above retained '
             'outputs. Distinguish deletion-only, actual refill and diagnostic fill. No masked-label tuning.')
ARMS = ('native', 'retained', 'refill', 'diagnostic')


def prepare():
    src = read_json(SOURCE/'plan.json')
    report = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != '121989fbf385c4156382b5cf44eb146b718f135e48c09080f68d22b285abbca2'
            or file_sha256(SOURCE/'plan.json') != report['plan_sha256'] or report['status'] != 'complete'
            or report['follow_up'] or file_sha256(MANIFEST) != src['manifest_sha256']
            or file_sha256(Path(src['checkpoint']['path'])) != src['checkpoint']['sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed conflict evidence, checkpoint or environment')
    for name, entry in src['slices'].items():
        if file_sha256(Path(entry['annotation'])) != entry['annotation_sha256'] or file_sha256(SOURCE/f'{name}.json') != HASHES[name]:
            raise ValueError(f'Changed held-out data or prior endpoint: {name}')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                references=HASHES, parent_comparison_sha256=file_sha256(SOURCE/'comparison.json'), criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan


def best_targets(voc_boxes, anns, threshold):
    """VOC selects the highest-overlap same-class annotation before checking difficult."""
    ids = []
    targets = [a['voc_bbox'] for a in anns]
    for box in voc_boxes:
        iou = overlaps(np.asarray(box), targets)
        best = int(iou.argmax()) if len(iou) else -1
        ids.append(anns[best]['id'] if best >= 0 and iou[best] > threshold and not anns[best].get('difficult') else None)
    return ids


def trace_image(native, refill, scores, categories, boxes, anns, previous, threshold):
    columns = len(categories)
    common = set(native) & set(refill)
    retained = [pos for pos in native if pos in common]
    dropped = [pos for pos in native if pos not in common]
    budget = Counter(pos % columns for pos in dropped)
    promoted = set(refill)-common
    if Counter(pos % columns for pos in promoted) != budget:
        raise ValueError('Different same-class budgets')
    old_max = np.full(len(boxes), -np.inf)
    for pos in native:
        q, c = divmod(pos, columns)
        if categories[c] in previous:
            old_max[q] = max(old_max[q], scores[q, c])
    native_set = set(native)
    voc = [voc_prediction_box(b) for b in boxes.tolist()]
    rows, additions = [], []
    for c, slots in budget.items():
        category = categories[c]
        targets = [a for a in anns if a['category_id'] == category]
        hit = best_targets(voc, targets, threshold)
        seen = {hit[pos//columns] for pos in retained if pos%columns == c} - {None}
        removed = {hit[pos//columns] for pos in dropped if pos%columns == c} - seen - {None}
        actual = {hit[pos//columns] for pos in promoted if pos%columns == c} - seen - {None}
        allowed = [int(q) for q in np.argsort(-scores[:, c], kind='stable')
                   if q*columns+c not in native_set and scores[q, c] >= old_max[q]]
        if set(allowed[:slots]) != {pos//columns for pos in promoted if pos%columns == c}:
            raise ValueError('Reconstructed eligible ranking differs from actual refill')
        # Each query has one best VOC target. Distinct new targets require distinct queries.
        best = {}
        for q in allowed:
            if hit[q] is not None and hit[q] not in seen:
                best.setdefault(hit[q], q)
        picked = list(best.values())[:slots]
        chosen = set(picked)
        picked.extend(q for q in allowed if q not in chosen)
        additions.extend(q*columns+c for q in picked[:slots])
        rows.append(dict(category=category, slots=slots, valid_gt=sum(not a.get('difficult', 0) for a in targets),
                         removed_unique=sorted(removed), actual_extra=sorted(actual), available_targets=list(best),
                         diagnostic_extra=list(best)[:slots], max_extra=min(slots, len(best))))
    # Match the existing refill convention: original order for retained ties, flattened ID for new ties.
    merged = retained + sorted(additions)
    diagnostic = sorted(merged, key=lambda pos: -scores.flat[pos])
    return retained, diagnostic, rows


@torch.inference_mode()
def evaluate(model, ds, loader, manifest, order, reference):
    previous, current = (s['classes'] for s in manifest['stages'][:2])
    known = manifest['stages'][1]['active_classes']
    native_post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=50)
    refill_post = SharedQueryRefill(known, previous)
    spec = dict(manifest['evaluation'], image_source_ids=order)
    predictions = {arm: {} for arm in ARMS}
    records = {}
    model.eval()
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = native_post(out, sizes, include_top_indices=True)
        refill = refill_post(out, sizes, include_top_indices=True)
        scores, categories, _ = native_post.candidate_scores(out)
        h, w = sizes.unbind(1)
        boxes = (box_cxcywh_to_xyxy(out['pred_boxes'])*torch.stack((w,h,w,h), 1)[:,None]).cpu()
        scores, categories = scores.cpu(), categories.cpu()
        for i, target in enumerate(targets):
            image_id = int(target['image_id'])
            key = ds.coco.imgs[image_id]['owod_source_id']
            n, r = native[i]['top_indices'].tolist(), refill[i]['top_indices'].tolist()
            kept, diagnostic, rows = trace_image(n, r, scores[i].numpy(), categories.tolist(), boxes[i],
                                                  ds.coco.imgToAnns.get(image_id, []), previous, spec['iou_threshold'])
            records[key] = rows
            for arm, selection in zip(ARMS, (n, kept, r, diagnostic)):
                ids = torch.tensor(selection, dtype=torch.long)
                predictions[arm][image_id] = dict(boxes=boxes[i, ids//len(categories)], labels=categories[ids%len(categories)],
                                                  scores=scores[i].flatten()[ids])
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(records)} images', flush=True)
    views = {}
    for arm in ARMS:
        print(f'Computing released VOC metrics for {arm}', flush=True)
        metrics = evaluate_author_benchmark(predictions.pop(arm), ds.coco, known, previous, current, spec,
                                            include_current_stats=True, include_matched_ids=True)
        matched = metrics.pop('matched_annotation_ids')
        stats = metrics.pop('current_class_stats')
        if arm in ('native', 'refill'):
            ref = reference['views'][f'released_{arm}']
            for k, row in stats.items():
                if row != {m:v for m,v in ref['current_class_stats'][str(k)].items() if m!='fp_sources'}:
                    raise ValueError('Original class AP, TP or FP changed')
            if metrics != ref['metrics']:
                raise ValueError('Original metrics changed')
        views[arm] = dict(metrics=metrics, current_class_stats=stats, matched_current=sorted(i for c in current for i in matched[c]))
    rows = [row for group in records.values() for row in group]
    retained = set(views['retained']['matched_current'])
    for arm, field in [('native','removed_unique'), ('refill','actual_extra'), ('diagnostic','diagnostic_extra')]:
        extra = {i for r in rows for i in r[field]}
        if retained | extra != set(views[arm]['matched_current']) or retained & extra:
            raise ValueError(f'Traced unique GT does not reconstruct {arm}')
    if (len(records)!=1024 or refill_post.replacements!=reference['audits']['released_refill']['replacements']
            or sum(r['slots'] for r in rows)!=refill_post.replacements):
        raise ValueError('Incomplete same-image quota replay')
    return dict(views=views, images=records)


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    manifest = read_json(MANIFEST)
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    sums = {}
    for name, entry in plan['slices'].items():
        path = OUTPUT/f'{name}.json'
        if not path.exists():
            ann = Path(entry['annotation'])
            ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
            order = [im['owod_source_id'] for im in read_json(ann)['images']]
            print(f'Evaluating fixed refill capacity on {name}, {len(ds)} images', flush=True)
            evidence = evaluate(model, ds, loader, manifest, order, read_json(SOURCE/f'{name}.json'))
            write_json(path, dict(plan_sha256=digest, **evidence))
        data = read_json(path)
        if data['plan_sha256'] != digest:
            raise ValueError('Changed endpoint identity')
        rows = [row for group in data['images'].values() for row in group]
        capacity = sum(r['max_extra'] for r in rows)
        actual = sum(len(r['actual_extra']) for r in rows)
        ap = {arm: v['metrics']['current_ap50'] for arm, v in data['views'].items()}
        sums[name] = dict(slots=sum(r['slots'] for r in rows), slots_without_class_gt=sum(r['slots'] for r in rows if not r['valid_gt']),
                          maximum_extra_tp=capacity, actual_extra_tp=actual, removed_unique_tp=sum(len(r['removed_unique']) for r in rows),
                          current_ap50=ap, ranking_opportunity=ap['diagnostic']-ap['native'] >= .01 and 2*actual < capacity)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                               slices=sums, ranking_opportunity=all(v['ranking_opportunity'] for v in sums.values())))
    print('Fixed refill capacity comparison saved', flush=True)


if __name__ == '__main__':
    main()
