"""Measure final Top-50 recovery after removing only replacement-to-retained suppression edges."""

from collections import Counter, defaultdict
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_fine_rank import mined_selector
from tools.owod.probe_task1_m_s_refinement_fixed_injection import match_new_seeds
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
    def __init__(self, job, plan, source):
        self.guard = GuardReplacementAnchors(known_ids=job['classes'], scoring='focal', max_detections=plan['limit'])
        self.columns = len(job['classes']) + 1
        self.small_area = plan['small_area']
        self.locked = defaultdict(dict)
        for row in source['rows']:
            self.locked[row['image_id']][row['target_index']] = row
        self.lost_targets = []
        self.totals = {size: Counter() for size in ('all', 'small')}
        self.unknown = Counter()
        self.proxy = {name: set() for name in ('protected', 'guarded')}

    def __call__(self, post, native, protected, picks, targets, witnesses, losses):
        unit = protected['pred_boxes'].new_ones((len(targets), 2))
        standard = post(protected, unit, include_top_indices=True)
        changed = torch.zeros_like(protected['pred_boxes'][..., 0], dtype=torch.bool)
        for batch, (slots, _, _) in picks.items():
            changed[batch, slots] = True
        candidate = self.guard(dict(protected, replaced_query_mask=changed), unit, include_top_indices=True)

        for batch, target in enumerate(targets):
            image_id = int(target['image_id'])
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            labels = target['labels'].cuda()
            small = target['source_box_area'].cuda() < self.small_area
            before, after = (known_matches(items[batch], truth, labels).any(dim=0) for items in (standard, candidate))
            original = witnesses[batch][0].any(dim=0)
            for size, mask in (('all', torch.ones_like(small)), ('small', small)):
                self.totals[size].update(native=int((original & mask).sum()), protected=int((before & mask).sum()),
                                         guarded=int((after & mask).sum()),
                                         gained_vs_protected=int((after & ~before & mask).sum()),
                                         lost_vs_protected=int((~after & before & mask).sum()))
            for index, row in self.locked.get(image_id, {}).items():
                if not losses[batch][index] or int(labels[index]) != row['class_id'] or bool(small[index]) != row['small']:
                    raise ValueError('Known loss identity did not reproduce on the protected replay')
                self.lost_targets.append(dict(image_id=image_id, target_index=index, recovered=bool(after[index])))

            emitted = {}
            for name, items in (('protected', standard), ('guarded', candidate)):
                detection = items[batch]
                unknown = (detection['labels'] == 91) & detection['scores'].isfinite()
                emitted[name] = detection['top_indices'][unknown] // self.columns
                self.unknown[name] += len(emitted[name])
            if batch in picks:
                slots, cells, _ = picks[batch]
                slots, _, indices = match_new_seeds(native, batch, slots, cells, truth, small)
                geometry = hits(protected['pred_boxes'][batch, slots], truth[indices]).diagonal()
                for name, queries in emitted.items():
                    matched = geometry & torch.isin(slots, queries)
                    self.proxy[name].update((image_id, index, slot) for index, slot in
                                            zip(indices[matched].tolist(), slots[matched].tolist(), strict=True))

    def finish(self, replay, protocol):
        correct = {(image, index) for image, indices in self.locked.items() for index in indices}
        observed = {(row['image_id'], row['target_index']) for row in self.lost_targets}
        if observed != correct or len(observed) != len(self.lost_targets):
            raise ValueError('Incomplete paired evaluation of the locked missing targets')
        for size, counts in self.totals.items():
            if any(counts[name] != replay['groups'][size][name]['known_tp'] for name in ('native', 'protected')):
                raise ValueError('Changed paired known-match accounting')
        if len(self.proxy['protected']) != replay['transfer']['protected']['unknown_same_slot']:
            raise ValueError('Changed direct unknown proxy hit accounting')
        recovered = sum(row['recovered'] for row in self.lost_targets)
        missing_proxy = self.proxy['protected'] - self.proxy['guarded']
        gates = dict(recovery=protocol != 's-owodb' or 2 * recovered >= len(correct),
                     known=all(counts['guarded'] >= counts['native'] for counts in self.totals.values()),
                     proxy=not missing_proxy)
        summary = dict(known=self.totals, unknown=self.unknown, tracked_known_losses=len(correct),
                       recovered=recovered, proxy={name: len(hits) for name, hits in self.proxy.items()},
                       lost_proxy=len(missing_proxy), gates=gates, passed=all(gates.values()))
        return dict(summary=summary, target_results=self.lost_targets,
                    proxy_hits={name: sorted(values) for name, values in self.proxy.items()})


def main():
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
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
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
            dest = OUTPUT / f'{key}.json'
            if not dest.exists():
                print(f'Evaluating {key} actual Top-50 recovery', flush=True)
                compare = Counterfactual(job, plan, previous)
                replay = evaluate(model.eval(), loader, protocol, job, plan, selector, references[name], observe=compare)
                if replay != previous['replay']:
                    raise ValueError(f'Changed frozen replay: {key}')
                write_json(dest, dict(plan_sha256=digest, order_sha256=order, source_sha256=REPORT_SHA[key],
                                      **compare.finish(replay, protocol)))
            report = read_json(dest)
            if (report['plan_sha256'], report['order_sha256'], report['source_sha256']) != (digest, order, REPORT_SHA[key]):
                raise ValueError(f'Changed output binding: {key}')
            slices[name] = report['summary']
        protocols[protocol] = slices
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(s['passed'] for group in protocols.values() for s in group.values())))


if __name__ == '__main__':
    main()
