"""Replay released-only dense box assignments through the frozen Task 1 postprocessor."""

from collections import Counter
import subprocess

from scipy.optimize import linear_sum_assignment
import torch

from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_protected_residual import protect_known
from tools.owod.probe_task1_m_s_refinement_trajectory import (
    CAPACITY_OUTPUT, eligible_queries, hits, near_dense_cells, run_released_probe,
)
from tools.owod.probe_task1_paired_discovery import verified
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = CAPACITY_OUTPUT.with_name('task1-m-s-refinement-dense-postprocess-oracle')
CAPACITY_SHA = 'fe476eac7ede7b94f7ca62009dae67de6d61fa5d5c62887b7f63169e0545c421'
ARMS = ('local', 'global')


def positive_pairs(iou):
    eligible = iou > .5
    rows = torch.where(eligible.any(dim=1))[0]
    # Maximizing the match count takes precedence over the IoU tie-break.
    weights = torch.where(eligible[rows], 1 + iou[rows] / (iou.shape[1] + 1), 0.)
    left, right = linear_sum_assignment(weights.cpu().numpy(), maximize=True)
    source = rows[left]
    targets = torch.as_tensor(right, device=iou.device)
    keep = eligible[source, targets]
    return source[keep], targets[keep]


def pair_candidates(final, geometry, batch, truth, native, misses, scored, columns):
    _, unknown = eligible_queries(native, scored, columns)
    queries = torch.where(unknown)[0]
    targets = torch.where(misses)[0]
    valid = torch.where(geometry['valid'][batch])[0]
    boxes = geometry['boxes'][batch]
    dense_iou = box_iou(box_cxcywh_to_xyxy(boxes[valid]), truth[targets])[0]

    near = near_dense_cells(final[queries], geometry, batch, valid)
    if near.shape[1]:
        local_iou, positions = dense_iou[torch.searchsorted(valid, near)].max(dim=1)
        local_queries, local_targets = positive_pairs(local_iou)
        local_cells = near[local_queries, positions[local_queries, local_targets]]
    else:
        local_queries = local_targets = local_cells = queries[:0]

    dense_cells, dense_targets = positive_pairs(dense_iou)
    dense_cells = valid[dense_cells]
    distance = torch.cdist(final[queries, :2], boxes[dense_cells, :2])
    global_queries, assigned = linear_sum_assignment(distance.cpu().numpy())
    return {
        'local': (queries[local_queries], local_cells, targets[local_targets]),
        'global': (queries[global_queries], dense_cells[assigned], targets[dense_targets[assigned]]),
    }


def output_hits(row, truth, labels):
    overlap = box_iou(row['boxes'], truth)[0] > .5
    unknown = row['labels'] == 91
    known_matches = (overlap & (row['labels'][:, None] == labels)).any(dim=0)
    return overlap[unknown].any(dim=0), known_matches


@torch.inference_mode()
def measure(model, loader, known, plan):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    counts = {arm: Counter() for arm in ARMS}
    protection = {arm: Counter(restored_known_queries=0, reverted_images=0) for arm in ARMS}
    coverage = Counter()
    columns = len(known) + 1
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        unit = final.new_ones((len(targets), 2))
        native_results = post(outputs, unit, include_top_indices=True)
        revised = {arm: final.clone() for arm in ARMS}
        evidence = []
        for batch, (target, scored) in enumerate(zip(targets, native_results, strict=True)):
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            native = hits(final[batch], truth)
            covered = native.any(dim=0)
            small = target['source_box_area'].cuda() < plan['small_area']
            misses = small & ~covered
            coverage.update(gt=int(small.sum()), final=int((small & covered).sum()))
            assignments = pair_candidates(final[batch], outputs['geometry'], batch, truth, native, misses, scored, columns)
            for arm, (queries, cells, _) in assignments.items():
                revised[arm][batch, queries] = outputs['geometry']['boxes'][batch, cells]
            evidence.append((truth, target['labels'].cuda(), small, scored, assignments))

        for arm in ARMS:
            transform = protect_known(lambda _: revised[arm], post, known, protection[arm])
            protected = transform(outputs)
            replay = post(dict(outputs, pred_boxes=protected), unit, include_top_indices=True)
            for batch, (truth, labels, small, scored, assignments) in enumerate(evidence):
                row = replay[batch]
                queries, cells, target_ids = assignments[arm]
                surviving = row['top_indices'][row['labels'] == 91]
                native_unknown = scored['top_indices'][scored['labels'] == 91]
                native_hits, native_known = output_hits(scored, truth, labels)
                revised_hits, revised_known = output_hits(row, truth, labels)
                if (native_known & ~revised_known).any():
                    raise ValueError('Lost a native known match after protection')
                retained = torch.isin(queries * columns + len(known), surviving)
                retained &= hits(protected[batch, queries], truth)[torch.arange(len(queries), device=queries.device),
                                                                   target_ids]
                counts[arm].update(
                    assigned=len(queries), unique_cells=len(cells.unique()), retained=int(retained.sum()),
                    lost_native_unknown_queries=int((~torch.isin(native_unknown, surviving)).sum()),
                    gained_unknown_small=int((small & revised_hits & ~native_hits).sum()),
                    lost_unknown_small=int((small & native_hits & ~revised_hits).sum()),
                    native_known=int(native_known.sum()), revised_known=int(revised_known.sum()),
                    native_known_small=int((native_known & small).sum()),
                    revised_known_small=int((revised_known & small).sum()),
                )
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released dense oracle replay')
    return dict(coverage=coverage, groups={arm: dict(counts[arm], **protection[arm]) for arm in ARMS})


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released dense oracle')
    capacity = verified(CAPACITY_OUTPUT / 'comparison.json', CAPACITY_SHA)
    plan = verified(CAPACITY_OUTPUT / 'plan.json', capacity['plan_sha256'])
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    if capacity['status'] != 'complete' or not quarter['passed']:
        raise ValueError('Expected completed released capacity and passed quarter-cap probe')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                capacity_sha256=CAPACITY_SHA, arms=ARMS,
                factor='Change only boxes at natively emitted, unprotected unknown queries with no original hit. '
                       'Use released labels to assign distinct targets to native query IDs. Compare nearest-16 fine-level '
                       'cells with unconstrained dense cells under the same scores and known output protection.',
                advance='Measure gains, losses and assigned queries surviving the locked Top-50 postprocessor.')

    def verify_measure(model, loader, known, plan):
        result = measure(model, loader, known, plan)
        protocol = next(name for name, job in plan['jobs'].items() if job['classes'] == known)
        expected = capacity['protocols'][protocol]['groups']['all_small']
        for arm in ARMS:
            group = result['groups'][arm]
            if group['assigned'] != expected[f'{arm}_unknown_matched']:
                raise ValueError(f'Changed released assignment count: {protocol}/{arm}')
            reference = quarter['protocols'][protocol]['groups']
            if (group['native_known'], group['native_known_small']) != (
                    reference['all']['native_known_tp'], reference['small']['native_known_tp']):
                raise ValueError(f'Changed native known matches: {protocol}/{arm}')
        return result

    run_released_probe(OUTPUT, plan, quarter, verify_measure)


if __name__ == '__main__':
    main()
