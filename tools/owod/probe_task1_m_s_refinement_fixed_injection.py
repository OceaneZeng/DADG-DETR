"""Test whether accurate released boxes survive fixed-count injection on the refinement checkpoint."""

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
REGRESSION_SHA = 'd0055702a199e1317a7dd0f8366e24a1fe33b248847ff631c3e6b7106476d2b0'


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
                picked = choose(native, batch, truth, small, detections[batch], columns, counts)
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
        return dict(groups=groups, coverage={key: groups['small'][key] for key in ('gt', 'final')})

    run_released_probe(OUTPUT, plan, quarter, replay)


if __name__ == '__main__':
    main()
