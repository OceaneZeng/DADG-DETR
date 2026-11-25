"""Measure final Top-50 recovery after removing only replacement-to-retained suppression edges."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_fine_rank import mined_selector
from tools.owod.probe_task1_m_s_refinement_fixed_injection import match_new_seeds, measure_injection
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_trajectory import hits
from tools.owod.probe_task1_m_s_reservation_anchor_identity import OUTPUT as SOURCE, cohort_loaders, prepare_replays
from tools.owod.probe_task1_m_s_reservation_attention import evaluate, known_matches
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('task1-m-s-reservation-anchor-guard')
LIVE_OUTPUT = OUTPUT.with_name('task1-m-s-reservation-anchor-guard-live-decoder')
PROTECTED_SHA = '244f0997c765eaccd06709d8a6e5c4f1fcf29fb6fcc2ab4997161db3ff79f3f8'
SOURCE_SHA = '774325a61e6009fed14eefdd226c6aba03fad682ec3925c93355d186b50ab1e9'
REPORT_SHA = {
    'm-owodb-first': 'e89d58c5cca5d261d04861dc2cefc1e6cb464f479f78ad33a9894d14ce600700',
    'm-owodb-holdout': '56fb650b5a4ef87bd9a5e44939b71f773c884eb5639f3dcac587335be524517b',
    's-owodb-first': '60fc8b9ee6e8257f16203b2c7a746df8091d28db2f00fcf7dfa4ea4eb7fed408',
    's-owodb-holdout': '7a8d1403826d50d4aef8ee3286e3e49c8897c0244887e730d26e1624db306b8d',
}


class GuardReplacementAnchors(CompeteWithEmittedUnknown):
    def overlap_eligibility(self, outputs, batch, anchors, eligible):
        replaced = outputs['replaced_query_mask'][batch]
        return eligible & (replaced[:, None] | ~replaced[anchors][None])


class Counterfactual:
    def __init__(self, job, plan, source, *, baseline='protected'):
        self.guard = GuardReplacementAnchors(known_ids=job['classes'], scoring='focal', max_detections=plan['limit'])
        self.columns = len(job['classes']) + 1
        self.small_area = plan['small_area']
        self.baseline = baseline
        self.locked = defaultdict(dict)
        for row in source['rows']:
            self.locked[row['image_id']][row['target_index']] = row
        self.lost_targets = []
        self.totals = {size: Counter() for size in ('all', 'small')}
        self.unknown = Counter()
        self.proxy = {name: set() for name in (baseline, 'guarded')}

    def __call__(self, post, native, replacement, picks, targets, witnesses=None, _losses=None):
        unit = replacement['pred_boxes'].new_ones((len(targets), 2))
        standard = post(replacement, unit, include_top_indices=True)
        native_detections = post(native, unit) if witnesses is None else None
        changed = torch.zeros_like(replacement['pred_boxes'][..., 0], dtype=torch.bool)
        for batch, (slots, _, _) in picks.items():
            changed[batch, slots] = True
        candidate = self.guard(dict(replacement, replaced_query_mask=changed), unit, include_top_indices=True)

        for batch, target in enumerate(targets):
            image_id = int(target['image_id'])
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            labels = target['labels'].cuda()
            small = target['source_box_area'].cuda() < self.small_area
            before, after = (known_matches(items[batch], truth, labels).any(dim=0) for items in (standard, candidate))
            native_matches = (witnesses[batch][0] if witnesses is not None
                              else known_matches(native_detections[batch], truth, labels))
            original = native_matches.any(dim=0)
            for size, mask in (('all', torch.ones_like(small)), ('small', small)):
                counts = self.totals[size]
                counts.update(native=int((original & mask).sum()), guarded=int((after & mask).sum()))
                counts[self.baseline] += int((before & mask).sum())
                counts[f'gained_vs_{self.baseline}'] += int((after & ~before & mask).sum())
                counts[f'lost_vs_{self.baseline}'] += int((~after & before & mask).sum())
            for index, row in self.locked.get(image_id, {}).items():
                if (not original[index] or (self.baseline == 'protected' and before[index])
                        or int(labels[index]) != row['class_id'] or bool(small[index]) != row['small']):
                    raise ValueError('Changed known target identity or the protected replay no longer misses it')
                entry = dict(image_id=image_id, target_index=index, recovered=bool(after[index] & ~before[index]))
                if self.baseline == 'replaced':
                    entry.update(replaced_hit=bool(before[index]), guarded_hit=bool(after[index]))
                self.lost_targets.append(entry)

            emitted = {}
            for name, items in ((self.baseline, standard), ('guarded', candidate)):
                detection = items[batch]
                unknown = (detection['labels'] == 91) & detection['scores'].isfinite()
                emitted[name] = detection['top_indices'][unknown] // self.columns
                self.unknown[name] += len(emitted[name])
            if batch in picks:
                slots, cells, _ = picks[batch]
                slots, _, indices = match_new_seeds(native, batch, slots, cells, truth, small)
                geometry = hits(replacement['pred_boxes'][batch, slots], truth[indices]).diagonal()
                for name, queries in emitted.items():
                    matched = geometry & torch.isin(slots, queries)
                    self.proxy[name].update((image_id, index, slot) for index, slot in
                                            zip(indices[matched].tolist(), slots[matched].tolist(), strict=True))

    def finish(self, counts, original_proxy, *, require_recovery=False, geometry=None):
        correct = {(image, index) for image, indices in self.locked.items() for index in indices}
        observed = {(row['image_id'], row['target_index']) for row in self.lost_targets}
        if observed != correct or len(observed) != len(self.lost_targets):
            raise ValueError('Incomplete paired evaluation of the locked missing targets')
        for size, baseline_counts in counts.items():
            if any(self.totals[size][name] != count for name, count in baseline_counts.items()):
                raise ValueError('Changed paired known-match accounting')
        if len(self.proxy[self.baseline]) != original_proxy:
            raise ValueError('Changed direct unknown proxy hit accounting')
        recovered = sum(row['recovered'] for row in self.lost_targets)
        missing_proxy = self.proxy[self.baseline] - self.proxy['guarded']
        gates = dict(known=all(counts['guarded'] >= counts['native'] for counts in self.totals.values()),
                     proxy=not missing_proxy)
        if geometry is None:
            gates['recovery'] = not require_recovery or 2 * recovered >= len(correct)
        else:
            small = geometry['small']
            gates['geometry'] = 2 * (small['guarded'] - small['native']) >= small['replaced'] - small['native']
        summary = dict(known=self.totals, unknown=self.unknown, tracked_known_losses=len(correct),
                       recovered=recovered, proxy={name: len(hits) for name, hits in self.proxy.items()},
                       lost_proxy=len(missing_proxy), gates=gates, passed=all(gates.values()))
        if geometry is not None:
            summary['geometry'] = geometry
        return dict(summary=summary, target_results=self.lost_targets,
                    proxy_hits={name: sorted(values) for name, values in self.proxy.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-decoder', action='store_true', help='Use ordinary replacement without native attention replay')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the matched Top-50 counterfactual')
    plan, holdout, mined, first, _ = prepare_replays()
    source = verified(SOURCE / 'comparison.json', SOURCE_SHA)
    source_plan = verified(SOURCE / 'plan.json', source['plan_sha256'])
    if (source['status'] != 'complete' or not source['consistent_replacement_origin']
            or plan['jobs'] != source_plan['jobs'] or plan['holdout_order_sha256'] != source_plan['holdout_order_sha256']):
        raise ValueError('Expected the pinned four-slice source attribution')
    plan.update(source_sha256=SOURCE_SHA, source_plan_sha256=source['plan_sha256'], source_reports_sha256=REPORT_SHA,
                factor='On the frozen protected decode, block replacement-to-retained overlap edges only. '
                       'Run original native NMS and final Top-50 with unchanged boxes and logits.',
                advance='Recover half of original S overlap losses on each slice after Top-50. Require all/small '
                        'known hits at native levels or better, and preserve every original same-slot unknown proxy hit.')
    output = OUTPUT
    if args.live_decoder:
        prior = verified(OUTPUT / 'comparison.json', PROTECTED_SHA)
        prior_plan = verified(OUTPUT / 'plan.json', prior['plan_sha256'])
        if prior['status'] != 'complete' or not prior['passed'] or prior_plan['source_sha256'] != SOURCE_SHA:
            raise ValueError('Expected the passed targeted guard with native attention replay')
        output = LIVE_OUTPUT
        plan.update(protected_comparison_sha256=PROTECTED_SHA,
                    factor='Compare native and ordinary replacements with unchanged decoder dynamics. On the same '
                           'replaced outputs, block replacement-to-retained overlap edges only. Keep the exact same '
                           'slot selector, NMS, scores and Top-50.',
                    advance='All/small known matches at native levels or above on all slices, preserve at least half '
                            'the small-box geometry gain and retain every ordinary replacement same-slot proxy hit.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        loaders = cohort_loaders(protocol, job, plan)
        model, _ = load_detector(None, Path(job['checkpoint']), device)
        selector = mined_selector(mined[protocol]['ranker'])
        references = {'first': first['protocols'][protocol]['groups'], 'holdout': holdout[protocol]['mined']}
        slices = {}
        for name, (loader, order) in loaders.items():
            key = f'{protocol}-{name}'
            previous = verified(SOURCE / f'{key}.json', REPORT_SHA[key])
            if (previous['plan_sha256'] != source['plan_sha256'] or previous['order_sha256'] != order
                    or previous['summary'] != source['protocols'][protocol][name]):
                raise ValueError(f'Changed prior identity binding: {key}')
            dest = output / f'{key}.json'
            if not dest.exists():
                print(f'Evaluating {key} actual Top-50 recovery', flush=True)
                if args.live_decoder:
                    compare = Counterfactual(job, plan, previous, baseline='replaced')
                    replay = measure_injection(model.eval(), loader, job['classes'], plan, protocol, True,
                                               selector, observe=compare)
                    groups = replay['groups']
                    if groups != references[name]:
                        raise ValueError(f'Changed ordinary replacement replay: {key}')
                    counts = {size: dict(native=groups[size]['native_known_tp'], replaced=groups[size]['revised_known_tp'])
                              for size in ('all', 'small')}
                    geometry = {size: dict(native=groups[size]['final'], replaced=groups[size]['revised'],
                                           guarded=groups[size]['revised']) for size in counts}
                    result = compare.finish(counts, groups['transfer']['unknown_same_slot'], geometry=geometry)
                else:
                    compare = Counterfactual(job, plan, previous)
                    replay = evaluate(model.eval(), loader, protocol, job, plan, selector, references[name], observe=compare)
                    if replay != previous['replay']:
                        raise ValueError(f'Changed frozen replay: {key}')
                    counts = {size: {view: replay['groups'][size][view]['known_tp'] for view in ('native', 'protected')}
                              for size in ('all', 'small')}
                    result = compare.finish(counts, replay['transfer']['protected']['unknown_same_slot'],
                                            require_recovery=protocol == 's-owodb')
                write_json(dest, dict(plan_sha256=digest, order_sha256=order, source_sha256=REPORT_SHA[key],
                                      **result))
            report = read_json(dest)
            if (report['plan_sha256'], report['order_sha256'], report['source_sha256']) != (digest, order, REPORT_SHA[key]):
                raise ValueError(f'Changed output binding: {key}')
            slices[name] = report['summary']
        protocols[protocol] = slices
        del model
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(s['passed'] for group in protocols.values() for s in group.values())))


if __name__ == '__main__':
    main()
