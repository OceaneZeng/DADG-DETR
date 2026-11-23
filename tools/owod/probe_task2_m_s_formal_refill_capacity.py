"""Trace eligible slots and one-detection VOC recovery on frozen Task 2 endpoints."""

from collections import Counter
import json
from math import ceil
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess, UnknownRefillScore
from models.owod_benchmark_eval import evaluate_author_benchmark, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_refill_capacity import best_targets
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_formal_refill_screen import (
    COHORT_ROOT, LONG_OUTPUT, MANIFESTS, OUTPUT as SHORT_OUTPUT,
    OVERLAP_OUTPUT, SCREEN_ENDPOINTS_SHA, SCREEN_RESULT_SHA,
)
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SHORT_OUTPUT.with_name('task2-m-s-formal-refill-capacity')
SOURCES = {
    'three': (SHORT_OUTPUT, SCREEN_RESULT_SHA, SCREEN_ENDPOINTS_SHA),
    'six': (LONG_OUTPUT, '74ebc6a501e29251f02188d09ae36a1d84397434d31b189a33d99c2aa80d4760',
            '40e435522d36cc83ea9359521034e2c483b8b97abf89ccdb6ec02602a5071890'),
}
ARMS = ('native', 'unknown_refill')
DEVICE = torch.device('cuda')


def prepare():
    jobs = {}
    settings = None
    for duration, (source, comparison_hash, endpoints_hash) in SOURCES.items():
        comparison = verified(source / 'comparison.json', comparison_hash)
        original = verified(source / 'plan.json', comparison['plan_sha256'])
        endpoints_file = source / 'endpoint-sha256.txt'
        if comparison['status'] != 'complete' or file_sha256(endpoints_file) != endpoints_hash:
            raise ValueError(f'Changed archived Task 2 evidence: {duration}')
        endpoints = {name: digest for digest, name in
                     (line.split() for line in endpoints_file.read_text(encoding='utf-8').splitlines())}
        keys = ('cohort_plan_sha256', 'batch_size', 'limit', 'min_same_class_iou', 'protected_group',
                'minimum_unknown_recall_gain', 'refill_ap_tolerance', 'reference_sha256')
        current_settings = {key: original[key] for key in keys}
        if settings is not None and current_settings != settings:
            raise ValueError('The short and long endpoints must use identical replay settings')
        settings = current_settings
        jobs[duration] = {}
        for protocol, job in original['jobs'].items():
            directory = Path(job['benchmark']['runs'][-1]['output_dir'])
            checkpoint = directory / 'checkpoint.pth'
            digest = endpoints[checkpoint.resolve().relative_to(source.resolve()).as_posix()]
            manifest = MANIFESTS[protocol]
            manifest_digest = job['benchmark']['manifest_sha256']
            verified(manifest, manifest_digest)
            folds = {}
            for fold, entry in job['folds'].items():
                label = f'{protocol}-{fold}'
                report_path = source / f'{label}.json'
                report = read_json(report_path)
                if (report['checkpoint_sha256'] != digest or report['plan_sha256'] != comparison['plan_sha256']
                        or report['summary'] != comparison['protocols'][protocol][fold]):
                    raise ValueError(f'Changed replay reference: {duration}/{label}')
                folds[fold] = dict(entry, reference=str(report_path), reference_sha256=file_sha256(report_path))
            jobs[duration][protocol] = dict(
                checkpoint=str(checkpoint), checkpoint_sha256=digest, manifest=str(manifest),
                manifest_sha256=manifest_digest, previous=job['previous'], current=job['current'], folds=folds)
    if settings['limit'] != 50 or settings['protected_group'] != 'current' or settings['min_same_class_iou'] != .70:
        raise ValueError('Expected the registered Top-50 overlap rule')
    for protocol, job in jobs['three'].items():
        for fold, entry in job['folds'].items():
            other = jobs['six'][protocol]['folds'][fold]
            if any(entry[key] != other[key] for key in ('order', 'released', 'full', 'shared')):
                raise ValueError(f'Changed held-out fold: {protocol}/{fold}')
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                sources={name: dict(comparison_sha256=ref, endpoints_sha256=endpoints)
                         for name, (_, ref, endpoints) in SOURCES.items()},
                jobs=jobs, settings=settings, training=False, paper_comparable=False)


def trace_batch(outputs, sizes, native, refill, native_post, refill_post):
    scores, categories, unknown = native_post.candidate_scores(outputs)
    columns = len(categories)
    positions = torch.stack([row['top_indices'] for row in native])
    queries, classes = positions // columns, positions % columns
    ranks = torch.arange(positions.shape[1], device=positions.device)
    earlier = ranks[:, None] > ranks[None, :]
    repeated = ((queries[:, :, None] == queries[:, None, :]) & earlier).any(dim=-1)
    previous = (classes != columns - 1) & ~torch.isin(
        categories[classes], categories.new_tensor(refill_post.protected_ids))
    eligible = repeated & previous & refill_post.same_class_overlap(outputs['pred_boxes'], queries, classes, earlier)
    used = torch.zeros_like(unknown, dtype=torch.bool).scatter_(1, queries, True)
    # The production max takes the first query at an exact score tie.
    unused_scores = unknown.masked_fill(used, -torch.inf)
    unused_order = unused_scores.argsort(descending=True, stable=True)
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    h, w = sizes.unbind(1)
    boxes = boxes * torch.stack((w, h, w, h), dim=-1)[:, None]
    return zip(boxes.tolist(), unknown.tolist(), unused_order.tolist(), (~used).sum(1).tolist(),
               eligible.tolist(), positions.tolist(), [row['top_indices'].tolist() for row in refill], strict=True)


def trace_image(trace, anns, known, threshold):
    boxes, scores, order, unused_count, eligibility, native, refill = trace
    columns = len(known) + 1
    eligible = [index for index, flag in enumerate(eligibility) if flag]
    unused = order[:unused_count]
    removed = set(native) - set(refill)
    added = set(refill) - set(native)
    drop = native[eligible[-1]] if eligible and unused else None
    promoted = unused[0] if drop is not None else None
    if drop is None:
        if removed or added:
            raise ValueError('The rule changed an image without an eligible slot')
    elif removed != {drop} or added != {promoted * columns + columns - 1}:
        raise ValueError('Reconstructed slot or top unknown differs from the actual rule')

    unknown_anns = [ann for ann in anns if ann['category_id'] not in known]
    voc_boxes = [voc_prediction_box(box) for box in boxes]
    hits = best_targets(voc_boxes, unknown_anns, threshold)
    native_hits = {hits[pos // columns] for pos in native if pos % columns == columns - 1} - {None}
    reachable = set(hits) - native_hits - {None}
    unused_hits = {hits[q] for q in unused} - native_hits - {None}
    best_rank = next((rank for rank, q in enumerate(unused, 1) if hits[q] in unused_hits), None)
    new_hit = hits[promoted] if promoted is not None and hits[promoted] not in native_hits else None

    removed_tp = None
    if drop is not None:
        query, column = divmod(drop, columns)
        category = known[column]
        retained = [pos // columns for pos in refill if pos % columns == column]
        old_anns = [ann for ann in anns if ann['category_id'] == category]
        old_hits = best_targets([voc_boxes[q] for q in [query, *retained]], old_anns, threshold)
        if old_hits[0] is not None and old_hits[0] not in old_hits[1:]:
            removed_tp = (category, old_hits[0])
    return dict(eligible_slots=len(eligible), unused_queries=unused_count, dropped=drop, added_query=promoted,
                native_unknown_ids=sorted(native_hits), reachable_unknown_ids=sorted(reachable),
                unused_unknown_ids=sorted(unused_hits), best_unused_rank=best_rank,
                best_unused_hit_score=scores[unused[best_rank - 1]] if best_rank else None,
                top_unused_score=scores[unused[0]] if unused else None,
                added_unknown_tp=new_hit, removed_previous_tp=removed_tp)


@torch.inference_mode()
def evaluate(model, job, entry, settings):
    previous, current = job['previous'], job['current']
    known = previous + current
    manifest_path = Path(job['manifest'])
    manifest = verified(manifest_path, job['manifest_sha256'])
    released = COHORT_ROOT / entry['released']['file']
    full = OVERLAP_OUTPUT / entry['full']['file']
    for path, key in ((released, 'released'), (full, 'full')):
        if file_sha256(path) != entry[key]['sha256']:
            raise ValueError(f'Changed annotations: {path}')
    root = resolve_image_root(manifest_path, manifest)
    dataset, loader = make_loader(root, released, settings['batch_size'], 2)
    coco = COCO(str(full))
    spec = dict(manifest['evaluation'], image_source_ids=entry['order'])
    native = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=settings['limit'])
    refill = UnknownRefillScore(known_ids=known, scoring='focal', max_detections=settings['limit'],
                               protected_ids=current, min_same_class_iou=settings['min_same_class_iou'])
    predictions = {arm: {} for arm in ARMS}
    images = {}
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to(DEVICE))
        sizes = torch.stack([target['orig_size'] for target in targets]).to(DEVICE)
        baseline = native(outputs, sizes, include_top_indices=True)
        filled = refill(outputs, sizes, include_top_indices=True)
        traces = trace_batch(outputs, sizes, baseline, filled, native, refill)
        for target, base, fill, trace in zip(targets, baseline, filled, traces, strict=True):
            image = int(target['image_id'])
            if coco.imgs[image]['owod_source_id'] != dataset.coco.imgs[image]['owod_source_id']:
                raise ValueError(f'Changed annotation identity: {image}')
            images[image] = trace_image(trace, coco.imgToAnns[image], known, spec['iou_threshold'])
            for arm, pred in zip(ARMS, (base, fill)):
                predictions[arm][image] = {key: pred[key].cpu() for key in ('scores', 'boxes', 'labels')}
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)

    reference = verified(Path(entry['reference']), entry['reference_sha256'])
    matched = {}
    for arm in ARMS:
        metrics = evaluate_author_benchmark(predictions[arm], coco, known, previous, current, spec,
                                            include_current_stats=True, include_previous_stats=True,
                                            include_matched_ids=True)
        matched[arm] = metrics.pop('matched_annotation_ids')
        if json.loads(json.dumps(metrics)) != reference['report']['views'][arm]['metrics']:
            raise ValueError(f'Author VOC replay changed: {entry["reference"]}/{arm}')
    native_ids = {gt for row in images.values() for gt in row['native_unknown_ids']}
    added_ids = {row['added_unknown_tp'] for row in images.values()} - {None}
    removed_previous = {(label, gt) for label in previous
                        for gt in set(matched['native'][label]) - set(matched['unknown_refill'][label])}
    traced_previous = {row['removed_previous_tp'] for row in images.values()} - {None}
    if (native_ids != set(matched['native'][-1]) or native_ids | added_ids != set(matched['unknown_refill'][-1])
            or removed_previous != traced_previous):
        raise ValueError('Per-image trace did not reconstruct the author VOC matches')
    return dict(summary=summarize(images, reference['summary'], settings), images=images)


def summarize(images, reference, settings):
    rows = list(images.values())
    eligible = [row for row in rows if row['dropped'] is not None]
    rank_counts = Counter(row['best_unused_rank'] for row in eligible if row['best_unused_rank'] is not None)
    actual = sum(row['added_unknown_tp'] is not None for row in eligible)
    old_losses = sum(row['removed_previous_tp'] is not None for row in eligible)
    unknown_gt = reference['open_metrics']['native']['unknown_gt']
    if (len(eligible) != reference['refills'] or old_losses != reference['previous_matches_lost']
            or actual != rank_counts[1] or actual != round(reference['unknown_recall_gain'] * unknown_gt)):
        raise ValueError('Unexpected refill or VOC TP counts')
    return dict(reference=reference, image_count=len(rows), eligible_images=len(eligible),
                eligible_detections=sum(row['eligible_slots'] for row in rows),
                images_with_any_query_hit=sum(bool(row['reachable_unknown_ids']) for row in rows),
                images_with_unused_query_hit=sum(bool(row['unused_unknown_ids']) for row in rows),
                uncovered_unknown_gt_with_any_query_hit=sum(len(row['reachable_unknown_ids']) for row in rows),
                uncovered_unknown_gt_with_unused_query_hit=sum(len(row['unused_unknown_ids']) for row in rows),
                uncovered_unknown_gt_on_eligible_images=sum(len(row['unused_unknown_ids']) for row in eligible),
                compatible_rank_counts=dict(sorted(rank_counts.items())),
                actual_unknown_tp_gain=actual, max_one_refill_unknown_tp_gain=sum(rank_counts.values()),
                required_unknown_tp_gain=ceil(settings['minimum_unknown_recall_gain'] * unknown_gt),
                previous_tp_lost=old_losses)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for duration, jobs in plan['jobs'].items():
        results[duration] = {}
        for protocol, job in jobs.items():
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed frozen checkpoint: {checkpoint}')
            model, _ = load_detector(None, checkpoint, DEVICE)
            model.eval()
            results[duration][protocol] = {}
            for fold, entry in job['folds'].items():
                path = OUTPUT / f'{duration}-{protocol}-{fold}.json'
                if not path.exists():
                    print(f'Evaluating {duration}/{protocol}/{fold}', flush=True)
                    result = evaluate(model, job, entry, plan['settings'])
                    write_json(path, dict(plan_sha256=digest, **result))
                result = read_json(path)
                if result['plan_sha256'] != digest:
                    raise ValueError(f'Changed diagnostic plan: {path}')
                results[duration][protocol][fold] = result['summary']
            del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, endpoints=results))


if __name__ == '__main__':
    main()
