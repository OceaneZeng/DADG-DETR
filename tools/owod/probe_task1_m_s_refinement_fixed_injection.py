"""Test whether accurate released boxes survive fixed-count injection on the refinement checkpoint."""

import argparse
from collections import Counter
from functools import partial
import subprocess

import torch

from tools.owod.probe_task1_m_s_prescreen_replacement import MAX_REPLACEMENTS, replace_cells
from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure
from tools.owod.probe_task1_m_s_refinement_dense_oracle import positive_pairs
from tools.owod.probe_task1_m_s_refinement_first_extent import REGRESSION_OUTPUT
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_trajectory import hits, run_released_probe
from tools.owod.probe_task1_paired_discovery import verified
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = REGRESSION_OUTPUT.with_name('task1-m-s-refinement-fixed-injection')
NATIVE_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-fine-seed-reservation')
REGRESSION_SHA = 'd0055702a199e1317a7dd0f8366e24a1fe33b248847ff631c3e6b7106476d2b0'
ORACLE_SHA = '557f726288345153a448b95f970c66eb193856154154619b343215a4b4b5d5e9'
NMS_IOU = .6


def fine_candidates(geometry, batch):
    boxes = geometry['boxes'][batch]
    finest = geometry['masks'][0][batch].numel()
    area_limit = geometry['anchors'][batch, 0, 2:].prod()
    valid = torch.where(geometry['valid'][batch, :finest] & (boxes[:finest, 2:].prod(dim=1) <= area_limit))[0]
    selected = geometry['chosen'][batch, geometry['active'][batch]]
    return valid[~torch.isin(valid, selected)]


def choose_native(native, batch, detections, columns, *, scores=None):
    geometry = native['geometry']
    active = torch.where(geometry['active'][batch])[0]
    selected = geometry['chosen'][batch, active]
    known_queries = detections['top_indices'][detections['labels'] != 91] // columns
    eligible = torch.where(~torch.isin(active, known_queries))[0][-MAX_REPLACEMENTS:]
    boxes = geometry['boxes'][batch]
    candidates = fine_candidates(geometry, batch)
    scores = geometry['logits'][batch, candidates] if scores is None else scores
    priority = scores.argsort(descending=True, stable=True)
    candidates = candidates[priority]
    candidate_boxes = box_cxcywh_to_xyxy(boxes[candidates])
    used = ~scores[priority].isfinite()
    positions = torch.arange(len(active), device=active.device)
    placed = selected.clone()
    slots, cells = [], []
    for position in eligible.flip(0):
        retained = box_cxcywh_to_xyxy(boxes[placed[positions != position]])
        overlaps = box_iou(candidate_boxes, retained)[0]
        available = torch.where(~used & ~(overlaps > NMS_IOU).any(dim=1))[0]
        if len(available):
            best = available[0]
            used[best] = True
            placed[position] = candidates[best]
            slots.append(active[position])
            cells.append(candidates[best])
    if not slots:
        return None
    return torch.stack(slots), torch.stack(cells), None


def match_new_seeds(native, batch, slots, cells, truth, small):
    geometry = native['geometry']
    selected = geometry['chosen'][batch, geometry['active'][batch]]
    seeds = hits(geometry['boxes'][batch, selected], truth).any(dim=0)
    final = hits(native['pred_boxes'][batch], truth).any(dim=0)
    misses = torch.where(small & ~seeds & ~final)[0]
    boxes = box_cxcywh_to_xyxy(geometry['boxes'][batch, cells])
    rows, cols = positive_pairs(box_iou(boxes, truth[misses])[0])
    return slots[rows], cells[rows], misses[cols]


def choose(native, batch, truth, small, detections, columns, counts):
    geometry = native['geometry']
    active = torch.where(geometry['active'][batch])[0]
    chosen = geometry['chosen'][batch, active]
    seed_hits = hits(geometry['boxes'][batch, chosen], truth)
    final_hits = hits(native['pred_boxes'][batch], truth)
    misses = torch.where(small & ~seed_hits.any(dim=0) & ~final_hits.any(dim=0))[0]
    native_known = detections['top_indices'][detections['labels'] != 91] // columns
    eligible = active[~(seed_hits | final_hits[active]).any(dim=1) & ~torch.isin(active, native_known)]
    valid = torch.where(geometry['valid'][batch])[0]
    unused = valid[~torch.isin(valid, chosen)]
    dense = box_iou(box_cxcywh_to_xyxy(geometry['boxes'][batch, unused]), truth[misses])[0]
    accurate = (dense > .5).any(dim=0)
    counts.update(uncovered_by_seed_and_final=len(misses), dense_available=int(accurate.sum()),
                  dense_without_slot=int(accurate.sum()) if not len(eligible) else 0)
    if not len(eligible) or not accurate.any():
        return None
    cells, target_ids = positive_pairs(dense)
    limit = min(MAX_REPLACEMENTS, len(eligible), len(cells))
    priority = dense[cells, target_ids].argsort(descending=True, stable=True)[:limit]
    return eligible[-limit:], unused[cells[priority]], misses[target_ids[priority]]


def record(revised, candidates, truths, post, columns, counts):
    detections = post(revised, revised['pred_boxes'].new_ones((len(truths), 2)), include_top_indices=True)
    for batch, (slots, cells, targets) in candidates.items():
        truth = truths[batch][targets]
        if not hits(revised['geometry']['boxes'][batch, cells], truth).diagonal().all():
            raise ValueError('Replaced encoder cells lost their intended released boxes')
        first = hits(revised['aux_outputs'][0]['pred_boxes'][batch, slots], truth).diagonal()
        final = hits(revised['pred_boxes'][batch], truth)
        same_slot = final[slots].diagonal()
        row = detections[batch]
        unknown = row['labels'] == 91
        ids = row['top_indices'][unknown] // columns
        actual = (box_iou(row['boxes'][unknown], truth)[0] > .5).any(dim=0)
        counts.update(assigned=len(slots), images=1, first_same_slot=int(first.sum()),
                      final_same_slot=int(same_slot.sum()), final_any=int(final.any(dim=0).sum()),
                      unknown_query_emitted=int(torch.isin(slots, ids).sum()),
                      unknown_same_slot=int((same_slot & torch.isin(slots, ids)).sum()),
                      unknown_any=int(actual.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-selection', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released fixed-count injection')
    regression = verified(REGRESSION_OUTPUT / 'comparison.json', REGRESSION_SHA)
    previous = verified(REGRESSION_OUTPUT / 'plan.json', regression['plan_sha256'])
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(PROTECTED_FIT_OUTPUT / 'plan.json', quarter['plan_sha256'])
    fixed = ('jobs', 'images', 'batch_size', 'seed', 'small_area', 'limit')
    if regression['status'] != 'complete' or not quarter['passed'] or any(plan[key] != previous[key] for key in fixed):
        raise ValueError('Expected matching frozen released replay and regression attribution')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                regression_sha256=REGRESSION_SHA, max_replacements=MAX_REPLACEMENTS,
                factor='Assign accurate unused dense boxes to released small misses, replacing at most two active '
                       'slots with no native seed or final overlap hit and no emitted known result. Match dense '
                       'cells and targets one-to-one; select the highest-IoU assignments up to the budget.',
                advance='Measure first/final same-slot transfer, actual Top-50 unknown geometry and collateral '
                        'known-match changes with the refined detector. Released labels are used only in this audit.')

    output = OUTPUT
    if args.native_selection:
        oracle = verified(OUTPUT / 'comparison.json', ORACLE_SHA)
        oracle_plan = verified(OUTPUT / 'plan.json', oracle['plan_sha256'])
        if oracle['status'] != 'complete' or any(plan[key] != oracle_plan[key] for key in fixed):
            raise ValueError('Expected the same fixed-budget oracle cohort')
        output = NATIVE_OUTPUT
        plan.update(oracle_sha256=ORACLE_SHA, nms_iou=NMS_IOU,
                    factor='Reserve up to two existing low-priority active slots outside native known Top-50. '
                           'Choose unused fine-level valid boxes at most one native anchor area, sorted by the existing '
                           'cell quality and filtered against retained boxes at original NMS IoU 0.6.',
                    advance='Use the same accepted released dense and known Top-50 gates. Compare new seed hits '
                            'and same-slot transfer with the label-guided fixed-budget reference.')

    def replay(model, loader, known, plan):
        counts = Counter()
        post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
        columns = len(known) + 1
        proposal = model.detr.transformer.instance_proposals

        def predict(samples, native, targets):
            final = native['pred_boxes']
            detections = post(native, final.new_ones((len(targets), 2)), include_top_indices=True)
            candidates = {}
            truths = []
            for batch, target in enumerate(targets):
                truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
                truths.append(truth)
                small = target['source_box_area'].cuda() < plan['small_area']
                picked = (choose_native(native, batch, detections[batch], columns) if args.native_selection
                          else choose(native, batch, truth, small, detections[batch], columns, counts))
                if picked is not None:
                    candidates[batch] = picked
            if not candidates:
                return native
            positions = {batch: entry[0] for batch, entry in candidates.items()}
            replacements = {batch: entry[1] for batch, entry in candidates.items()}
            hook = partial(replace_cells, replacements=replacements, positions=positions)
            with proposal.register_forward_hook(hook):
                revised = model(samples)
            expected = native['geometry']['chosen'].clone()
            for batch, (slots, cells, _) in candidates.items():
                expected[batch, slots] = cells
            if (not torch.equal(revised['geometry']['active'], native['geometry']['active'])
                    or not torch.equal(revised['geometry']['chosen'], expected)):
                raise ValueError('Replacement changed the original query budget or untouched cells')
            if args.native_selection:
                counts['replacement_slots'] += sum(len(entry[0]) for entry in candidates.values())
                candidates = {
                    batch: match_new_seeds(native, batch, slots, cells, truths[batch],
                                           targets[batch]['source_box_area'].cuda() < plan['small_area'])
                    for batch, (slots, cells, _) in candidates.items()
                }
            record(revised, candidates, truths, post, columns, counts)
            return revised

        report = measure(model, known, loader, plan, replay=predict)
        protocol = next(name for name, job in plan['jobs'].items() if job['classes'] == known)
        groups = report['groups']
        reference = quarter['protocols'][protocol]['groups']
        if any(row['native_known_tp'] != reference[name]['native_known_tp'] for name, row in groups.items()):
            raise ValueError(f'Changed native class matches: {protocol}')
        groups['transfer'] = counts
        groups['changes'] = report['changes']
        if args.native_selection:
            groups['passed'] = (groups['small']['revised'] - groups['small']['final'] >= int(protocol == 's-owodb')
                                and all(groups[name]['revised_known_tp'] >= groups[name]['native_known_tp']
                                        for name in ('all', 'small')))
        return dict(groups=groups, coverage={key: groups['small'][key] for key in ('gt', 'final')})

    run_released_probe(output, plan, quarter, replay)


if __name__ == '__main__':
    main()
