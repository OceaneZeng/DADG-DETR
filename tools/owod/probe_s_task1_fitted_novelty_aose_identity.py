"""Pair A-OSE target identities under native and fitted S Task 1 output allocation."""

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_fitted_novelty_quota import (
    OUTPUT as QUOTA, MatchedFinalQuota, evaluate_fold, verify_quota,
)
from tools.owod.probe_s_task1_novelty_head_transfer import (
    FrozenNoveltyViews, TransferredNoveltyPostprocess, require_one_gpu,
)
from tools.owod.probe_s_task1_rank_independent import PROTOCOL
from tools.owod.probe_s_task1_same_query_competition import METRICS
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_score_order import verified_endpoint
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = QUOTA.with_name('task1-s-fitted-novelty-aose-identity')
QUOTA_SHA = 'c3eee0e09902153a7f733339cbd221c95f2cd0c8bfc7ec3465db5742e5bed200'


def prepare():
    previous = verified(QUOTA / 'comparison.json', QUOTA_SHA)
    source = verified(QUOTA / 'plan.json', previous['plan_sha256'])
    if previous['status'] != 'complete' or previous['passed'] or any(
            fold['delta']['a_ose'] <= 0 or fold['quota_checks']['fallbacks'] for fold in previous['folds'].values()):
        raise ValueError('Expected remaining A-OSE in both exact matched-quota folds')
    for name, field in (('control', 'checkpoint_sha256'), ('fitted', 'donor_sha256')):
        arm = source[name]
        if arm['sha256'] != previous[field] or file_sha256(Path(arm['checkpoint'])) != arm['sha256']:
            raise ValueError(f'Changed frozen endpoint: {name}')
    plan = {key: source[key] for key in ('control', 'fitted', 'manifest_sha256', 'batch_size', 'limit', 'folds')}
    plan.update(quota_sha256=QUOTA_SHA, quota_plan_sha256=previous['plan_sha256'],
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, scope='Pair VOC A-OSE by image and unknown GT identity on both locked folds.')
    pin(OUTPUT / 'plan.json', plan)
    return plan, previous['folds']


def compare_targets(before, after):
    def group(rows):
        result = defaultdict(Counter)
        for row in rows:
            identity = row['image_id'], row['unknown_gt']
            result[identity][row['known_prediction']] += 1
        return result

    old, new = group(before), group(after)
    gained, removed, shared = new.keys() - old.keys(), old.keys() - new.keys(), old.keys() & new.keys()
    added_labels, removed_labels = Counter(), Counter()
    for identity in gained:
        added_labels.update(new[identity])
    for identity in removed:
        removed_labels.update(old[identity])
    recurring_delta = sum(new[i].total() - old[i].total() for i in shared)
    return dict(new_targets=len(gained), removed_targets=len(removed), recurring_targets=len(shared),
                detections_on_new_targets=added_labels.total(), detections_on_removed_targets=removed_labels.total(),
                recurring_detection_delta=recurring_delta,
                recurring_known_label_changed=sum(old[i].keys() != new[i].keys() for i in shared),
                new_by_known_prediction=dict(added_labels), removed_by_known_prediction=dict(removed_labels))


def main():
    require_one_gpu()
    plan, previous = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    base, fitted = plan['control'], plan['fitted']
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    binding = dict(plan_sha256=digest, checkpoint_sha256=base['sha256'], donor_sha256=fitted['sha256'])
    detector, _ = load_detector(None, Path(base['checkpoint']), torch.device('cuda'))
    model = FrozenNoveltyViews(detector, Path(fitted['checkpoint']))
    settings = dict(known_ids=manifest['stages'][0]['classes'], scoring='focal', max_detections=plan['limit'])
    quota = MatchedFinalQuota(**settings)
    processors = dict(native=CompeteWithEmittedUnknown(**settings),
                      fitted=TransferredNoveltyPostprocess(**settings), quota=quota)
    pairs = (('native', 'fitted'), ('native', 'quota'), ('fitted', 'quota'))
    folds = {}
    for name, entry in plan['folds'].items():
        annotations = Path(entry['path'])
        if file_sha256(annotations) != entry['sha256']:
            raise ValueError(f'Changed S {name} holdout')
        dest = OUTPUT / f'{name}.json'
        if not dest.is_file():
            print(f'Evaluating S {name} per-target A-OSE identities', flush=True)
            report = evaluate_fold(model, manifest, annotations, plan, processors, trace_detections=True)
            report['quota_checks'] = quota.take_checks()
            write_json(dest, dict(**binding, report=report))
        report = verified_endpoint(dest, binding)
        verify_quota(report, 'native', plan['limit'])
        if report['quota_checks'] != previous[name]['quota_checks']:
            raise ValueError(f'Changed final keep counts on {name}')
        detections = {}
        for view, trace in report['attribution'].items():
            rows = trace['detections']
            aggregates = {key: value for key, value in trace.items() if key != 'detections'}
            metrics = {key: report[view]['metrics'][key] for key in METRICS}
            if (aggregates != previous[name]['attribution'][view] or metrics != previous[name]['metrics'][view]
                    or len(rows) != aggregates['a_ose']):
                raise ValueError(f'A-OSE identities did not reproduce the original {name}/{view} output')
            detections[view] = rows
        comparisons = {}
        for first, second in pairs:
            change = compare_targets(detections[first], detections[second])
            delta = (change['detections_on_new_targets'] - change['detections_on_removed_targets']
                     + change['recurring_detection_delta'])
            if delta != len(detections[second]) - len(detections[first]):
                raise ValueError('Paired target terms must sum to the observed A-OSE difference')
            comparisons[f'{second}_vs_{first}'] = dict(**change, a_ose_delta=delta)
        folds[name] = comparisons
    write_json(OUTPUT / 'comparison.json', dict(status='complete', **binding, folds=folds))


if __name__ == '__main__':
    main()
