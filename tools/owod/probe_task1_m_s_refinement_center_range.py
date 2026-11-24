"""Estimate reachable released coverage from size-preserving center updates."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess

from scipy.optimize import linear_sum_assignment
import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_quality_folds import PROTECTED_FOLD_OUTPUT, PROTECTED_PREFLIGHT_SHA
from tools.owod.probe_task1_m_s_refinement_center_residual import (
    ASSIGNMENT_OUTPUT, ASSIGNMENT_SHA, NEAR_IOU, learned_query_ids,
)
from tools.owod.probe_task1_m_s_refinement_center_selector import source_ids
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_protected_residual import OUTPUT as PROTECTED_OUTPUT
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = PROTECTED_OUTPUT.with_name('task1-m-s-refinement-center-range')
FIT_PAIRING_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-fit-pairing')
FOLDS_SHA = '0d94655b3d6e2b8b3abab2d0fd2133a369156fe7d46a83ab8efb5a6ebed4b32e'
ONE_PER_MISS_FOLDS_SHA = '1f0bbcdd607ad1fb4fb644079b83d32224c636fb5a126c28784e186df5a184c6'
LIMITS = {'quarter': .25, 'half': .5, 'unrestricted': None}


def center_hits(boxes, truth, misses, limit):
    centers = truth[misses, :2]
    offset = centers[None] - boxes[:, None, :2]
    extent = boxes[:, None, 2:]
    if limit is not None:
        offset = torch.maximum(torch.minimum(offset, limit * extent), -limit * extent)
    revised = torch.cat((boxes[:, None, :2] + offset, extent.expand_as(offset)), dim=-1)
    hits = (box_iou(box_cxcywh_to_xyxy(revised.flatten(0, 1)), box_cxcywh_to_xyxy(truth))[0] > .5
            ).reshape(len(boxes), len(misses), len(truth))
    return hits, offset, extent


def coverage_at_limit(boxes, truth, misses, native_hits, limit):
    hits, offset, extent = center_hits(boxes, truth, misses, limit)
    reached = hits[:, torch.arange(len(misses), device=misses.device), misses]
    safe = reached & ~(native_hits[:, None] & ~hits).any(dim=-1)
    # Prefer the target with the fewest safe options and use each query once.
    available = safe.clone()
    allocated = torch.zeros(len(misses), dtype=torch.bool, device=misses.device)
    for target in safe.sum(dim=0).argsort(stable=True).tolist():
        queries = torch.where(available[:, target])[0]
        if len(queries):
            distance = (offset[queries, target] / extent[queries, 0].clamp_min(1e-6)).square().sum(dim=1)
            query = queries[distance.argmin()]
            allocated[target] = True
            available[query] = False
    return reached.any(dim=0), safe.any(dim=0), allocated


def fit_pairing_counts(boxes, truth, small, misses, native_iou, limit):
    hits, _, _ = center_hits(boxes, truth, misses, limit)
    reached = hits[:, torch.arange(len(misses), device=misses.device), misses]
    native_hits = native_iou > .5
    has_no_hit = ~native_hits.any(dim=1, keepdim=True)
    safe = reached & ~(native_hits[:, None] & ~hits).any(dim=-1)
    near = native_iou[:, misses] >= NEAR_IOU
    preferred = native_iou.masked_fill(~small[None], -1.).argmax(dim=1)
    original = reached & near & has_no_hit & (preferred[:, None] == misses)
    counts = Counter(original_positive_rows=int(original.sum()), original_targets=int(original.any(dim=0).sum()))
    for name, eligible in (('any_target', reached & near & has_no_hit),
                           ('preserve_hits', safe & near), ('all_safe', safe)):
        edges = eligible.cpu().numpy()
        query, target = linear_sum_assignment(edges, maximize=True)
        counts.update({f'{name}_targets': int(eligible.any(dim=0).sum()),
                       f'{name}_matched': int(edges[query, target].sum())})
    return counts


@torch.inference_mode()
def measure(model, loader, plan, known, *, fit_pairing=False):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    counts, events = Counter(), []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        unit = final.new_ones((len(targets), 2))
        scored = post(outputs, unit, include_top_indices=True)
        for batch, (target, scored_row) in enumerate(zip(targets, scored, strict=True)):
            truth = target['boxes'].cuda()
            small = target['source_box_area'].cuda() < plan['small_area']
            native_iou = box_iou(box_cxcywh_to_xyxy(final[batch]), box_cxcywh_to_xyxy(truth))[0]
            native = native_iou > .5
            covered = native.any(dim=0)
            misses = torch.where(small & ~covered)[0]
            counts.update(gt=int(small.sum()), final=int((covered & small).sum()), misses=len(misses))
            if not len(misses):
                continue
            queries = learned_query_ids(final[batch], outputs['geometry'], batch)
            emitted = scored_row['top_indices'][scored_row['labels'] != 91] // (len(known) + 1)
            queries = queries[~torch.isin(queries, emitted)]
            if fit_pairing:
                if len(queries):
                    counts.update(fit_pairing_counts(final[batch, queries], truth, small, misses,
                                                     native_iou[queries], plan['move_limit']))
                continue
            groups = {}
            for name, limit in LIMITS.items():
                result = coverage_at_limit(final[batch, queries], truth, misses, native[queries], limit)
                groups[name] = result
                counts.update({f'{name}_{key}': int(mask.sum())
                               for key, mask in zip(('reachable', 'safe', 'allocated'), result, strict=True)})
            for target_index, pos in enumerate(misses.tolist()):
                events.append(dict(image=int(target['image_id']), target_index=pos,
                                   ranges={key: [bool(mask[target_index]) for mask in values]
                                           for key, values in groups.items()}))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['fit_images' if fit_pairing else 'images']:
        raise ValueError('Incomplete released range replay')
    return counts, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-pairing', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    if args.fit_pairing:
        assignments = verified(ASSIGNMENT_OUTPUT / 'comparison.json', ASSIGNMENT_SHA)
        fold_output = OUTPUT.with_name('task1-m-s-refinement-protected-center-one-per-miss-folds')
        one_per_miss_folds = verified(fold_output / 'comparison.json', ONE_PER_MISS_FOLDS_SHA)
        if assignments['status'] != 'complete' or one_per_miss_folds['status'] != 'complete' or one_per_miss_folds['passed']:
            raise ValueError('Expected the completed fit audit and stopped one-positive fold replay')
        plan = verified(ASSIGNMENT_OUTPUT / 'plan.json', assignments['plan_sha256'])
        plan.update(assignment_sha256=ASSIGNMENT_SHA, one_per_miss_folds_sha256=ONE_PER_MISS_FOLDS_SHA,
                    factor='Measure which released missed small targets get an eligible quarter-range fit query. '
                           'Compare best-native-IoU target pairing, any target, preservation of query hits, '
                           'and removal of the near-IoU prerequisite in order.',
                    advance='Identify a meaningful shortfall in one-to-one reachable fit targets before changing labels.')
        output = FIT_PAIRING_OUTPUT
    else:
        preflight = verified(PROTECTED_OUTPUT / 'comparison.json', PROTECTED_PREFLIGHT_SHA)
        folds = verified(PROTECTED_FOLD_OUTPUT / 'comparison.json', FOLDS_SHA)
        if (preflight['status'] != 'complete' or not preflight['passed']
                or folds['status'] != 'complete' or folds['passed']):
            raise ValueError('Expected the protected preflight and its completed fold result')
        plan = verified(PROTECTED_OUTPUT / 'plan.json', preflight['plan_sha256'])
        plan.update(protected_folds_sha256=FOLDS_SHA, limits=LIMITS,
                    factor='Use released target centers for fixed-size quarter, half, and unrestricted shifts. '
                           'Exclude native known Top-50 queries and count individual safe shifts and one assignment per query.')
        output = OUTPUT
    plan['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = output / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            inputs = (checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest), job)
            loader, order = training_loader(*inputs, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            if args.fit_pairing:
                replay = loader
                loader, order = training_loader(*inputs, dict(plan, images=plan['fit_images']), source_box_area=True,
                                                offset=plan['images'])
                if order != plan['fit_order_sha256'][protocol] or set(source_ids(loader)) & set(source_ids(replay)):
                    raise ValueError(f'Changed released fit order: {protocol}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            print(f'Evaluating {protocol} released center range', flush=True)
            counts, events = measure(model, loader, plan, job['classes'], fit_pairing=args.fit_pairing)
            if args.fit_pairing:
                histogram = assignments['protocols'][protocol]['targets_by_positive_count']['missed']
                assigned = sum(targets for positives, targets in histogram.items() if int(positives))
                positive_rows = sum(int(positives) * targets for positives, targets in histogram.items())
                if (counts['misses'], counts['original_targets'], counts['original_positive_rows']) != (
                        sum(histogram.values()), assigned, positive_rows):
                    raise ValueError(f'Changed frozen fit assignments: {protocol}')
            else:
                reference = preflight['protocols'][protocol]['groups']['small']
                if any(counts[key] != reference[key] for key in ('gt', 'final')):
                    raise ValueError(f'Changed frozen released geometry: {protocol}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, coverage=counts, events=events))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed range plan: {dest}')
        protocols[protocol] = report['coverage']
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
