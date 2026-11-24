"""Measure released localization gains from the existing in-box center candidates."""

from collections import Counter
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_quality_folds import OUTPUT as FOLD_OUTPUT
from tools.owod.probe_task1_m_s_refinement_center_selector import CENTER_COMPARISON_SHA, TRACE_COMPARISON_SHA, TRACE_OUTPUT
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import CENTER_OUTPUT, prepare, refine_learned_boxes
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = FOLD_OUTPUT.with_name('task1-m-s-refinement-center-candidates')
FOLD_SHA = 'ccf18c98556468be5b081edc97cba0f48bda502ad09ba1c4ef935e112454b0e6'


def candidate_coverage(boxes, captured, truth, native, small):
    queries, centers, scores = captured
    width_height = boxes[queries, None, 2:].expand_as(centers)
    alternatives = torch.cat((centers, width_height), dim=-1)
    hit = box_iou(box_cxcywh_to_xyxy(alternatives.flatten(0, 1)), truth)[0]
    hit = hit.reshape(*scores.shape, len(truth)) > .5
    eligible = scores.isfinite()
    safe = eligible & ~(native[queries, None] & ~hit).any(dim=-1)
    candidates = (eligible[..., None] & hit).any(dim=(0, 1))
    safe_candidates = (safe[..., None] & hit).any(dim=(0, 1))

    # Select at most one proposal per query and count new small targets only once.
    missing = small & ~native.any(dim=0)
    chosen = torch.zeros_like(missing)
    for query in torch.where((safe[..., None] & hit & missing).any(dim=(1, 2)))[0].tolist():
        options = (safe[query, :, None] & hit[query])[:, missing & ~chosen]
        gains = options.sum(dim=1)
        best = scores[query].masked_fill(~safe[query] | (gains < gains.max()), -torch.inf).argmax()
        chosen |= hit[query, best] & missing
    return candidates, safe_candidates, chosen


@torch.inference_mode()
def measure(model, loader, plan):
    counts = {name: Counter() for name in ('all', 'small')}
    events = []
    images = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        captured = {}

        def observe(batch, queries, centers, scores):
            captured[batch] = (queries, centers, scores)

        midpoints = refine_learned_boxes(outputs, center_only=True, observe_centers=observe)
        for batch, target in enumerate(targets):
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            small = target['source_box_area'].cuda() < plan['small_area']
            native, midpoint = (box_iou(box_cxcywh_to_xyxy(boxes[batch]), truth)[0] > .5
                                for boxes in (final, midpoints))
            covered, revised = native.any(dim=0), midpoint.any(dim=0)
            safe_midpoint = torch.where((native & ~midpoint).any(dim=1, keepdim=True), native, midpoint)
            protected = safe_midpoint.any(dim=0)
            options, safe_options, feasible = (torch.zeros_like(covered) for _ in range(3))
            if batch in captured:
                options, safe_options, feasible = candidate_coverage(final[batch], captured[batch],
                                                                       truth, native, small)
            for name, mask in (('all', torch.ones_like(small)), ('small', small)):
                missed = ~covered & mask
                counts[name].update(gt=int(mask.sum()), final=int((covered & mask).sum()),
                                    midpoint_gained=int((revised & missed).sum()),
                                    midpoint_lost=int((covered & ~revised & mask).sum()),
                                    protected_midpoint_gained=int((protected & missed).sum()),
                                    any_candidate_gained=int((options & missed).sum()),
                                    safe_candidate_gained=int((safe_options & missed).sum()),
                                    feasible_small_gained=int((feasible & missed).sum()))
            for pos in torch.where(small & ~covered & options)[0].tolist():
                events.append(dict(image=int(target['image_id']), target_index=pos, midpoint=bool(revised[pos]),
                                   protected_midpoint=bool(protected[pos]), safe_candidate=bool(safe_options[pos]),
                                   feasible=bool(feasible[pos])))
        images += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if images != plan['images']:
        raise ValueError('Incomplete released center replay')
    return counts, events


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    folds = verified(FOLD_OUTPUT / 'comparison.json', FOLD_SHA)
    if folds['status'] != 'complete' or folds['passed']:
        raise ValueError('Expected the completed fixed midpoint fold screen')
    source = verified(CENTER_OUTPUT / 'comparison.json', CENTER_COMPARISON_SHA)
    original = verified(CENTER_OUTPUT / 'plan.json', source['plan_sha256'])
    trace = verified(TRACE_OUTPUT / 'comparison.json', TRACE_COMPARISON_SHA)
    plan = prepare()
    if any(plan[key] != original[key] for key in ('jobs', 'seed', 'images', 'batch_size', 'small_area')):
        raise ValueError('Changed released center replay identity')
    plan.update(folds_sha256=FOLD_SHA, midpoint_sha256=CENTER_COMPARISON_SHA, trace_sha256=TRACE_COMPARISON_SHA,
                factor='Compare the best encoder-quality midpoint and all 16 eligible in-box midpoints. '
                       'Use released target IoU for diagnostic gains, protect all old query hits, and separately '
                       'count a feasible greedy placement with at most one midpoint per query.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            manifest = verified(MANIFESTS[protocol], job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            image_root = resolve_image_root(MANIFESTS[protocol], manifest)
            loader, order = training_loader(checkpoint, Path(job['training']), image_root,
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            print(f'Evaluating {protocol} released center candidates', flush=True)
            summary, events = measure(model, loader, plan)
            for name, row in summary.items():
                reference = trace['protocols'][protocol][name]
                for key in ('gt', 'final', 'midpoint_gained', 'midpoint_lost'):
                    if row[key] != reference[key]:
                        raise ValueError(f'Changed frozen midpoint coverage: {protocol}/{name}/{key}')
                if row['protected_midpoint_gained'] != reference['protected_gained']:
                    raise ValueError(f'Changed protected midpoint coverage: {protocol}/{name}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, summary=summary, events=events))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen candidate plan: {dest}')
        protocols[protocol] = report['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
