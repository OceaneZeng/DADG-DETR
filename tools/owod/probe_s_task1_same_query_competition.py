"""Limit unknown-to-known overlap suppression to the same query on locked S folds."""

import subprocess
from pathlib import Path

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_half_rank_quota import SUPPRESSION_OUTPUT
from tools.owod.probe_s_task1_rank_independent import OUTPUT as INDEPENDENT, PROTOCOL
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown, DeduplicateKnown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import QUALITY_OUTPUT, QUALITY_SHA
from tools.owod.screening_data import source_ids, write_json


OUTPUT = SUPPRESSION_OUTPUT.with_name('task1-s-same-query-competition')
SUPPRESSION_SHA = '3c0847c427330583b7486fb1bb0401afe0f8ac886e7d2e3b139bee1fd5082b9e'
VIEWS = ('native', 'same_query', 'known_nms_only')
METRICS = ('known_ap50', 'u_recall', 'unknown_precision', 'unknown_predictions', 'a_ose', 'wi', 'h_score')
MIN_AP_CHANGE = {'full': -.005, 'half': .005}


def prepare():
    diagnostic = verified(SUPPRESSION_OUTPUT / 'comparison.json', SUPPRESSION_SHA)
    source = verified(SUPPRESSION_OUTPUT / 'plan.json', diagnostic['plan_sha256'])
    if (diagnostic['status'] != 'complete'
            or diagnostic['arms']['half']['unknown_overlap_loss'] <= diagnostic['arms']['full']['unknown_overlap_loss']):
        raise ValueError('Expected extra unknown overlap losses in the completed paired trace')
    quality = verified(QUALITY_OUTPUT / 'comparison.json', QUALITY_SHA)
    baseline = verified(QUALITY_OUTPUT / 'plan.json', quality['plan_sha256'])
    job = baseline['jobs'][PROTOCOL]
    if source['manifest_sha256'] != job['manifest_sha256']:
        raise ValueError('Changed S fold manifest')
    disjoint = source_ids(verified(INDEPENDENT / 'released-holdout.json', source['holdout_sha256']))
    for name, entry in job['folds'].items():
        ids = source_ids(verified(Path(entry['path']), entry['sha256']))
        if len(ids) != 1024 or ids & disjoint:
            raise ValueError(f'Expected 1024 disjoint S validation images for {name}')
        disjoint.update(ids)
    arms = source['arms']
    for arm in arms.values():
        if file_sha256(Path(arm['checkpoint'])) != arm['sha256']:
            raise ValueError('Changed frozen S checkpoint')
    manifest = verified(MANIFESTS[PROTOCOL], source['manifest_sha256'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                suppression_trace_sha256=SUPPRESSION_SHA, paper_comparable=False, arms=arms,
                folds=job['folds'], manifest_sha256=source['manifest_sha256'], batch_size=source['batch_size'],
                limit=source['limit'], min_ap_change=MIN_AP_CHANGE,
                known_references={fold: {k: row['metrics'][k] for k in METRICS}
                                  for fold, row in quality['protocols'][PROTOCOL].items()},
                max_wi_increase=baseline['gates']['wi'], max_aose_fraction=baseline['gates']['a_ose'])
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


def evaluate(model, manifest, path, plan):
    known = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, path, plan['batch_size'], 2)
    spec = dict(manifest['evaluation'], image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    settings = dict(known_ids=known, scoring='focal', max_detections=plan['limit'])
    processors = dict(native=CompeteWithEmittedUnknown(**settings),
                      same_query=CompeteWithEmittedUnknown(same_query=True, **settings),
                      known_nms_only=DeduplicateKnown(**settings))
    return evaluate_views(model, loader, dataset.coco, {key: (known, 'focal', plan['limit']) for key in VIEWS},
                          spec, [], known, torch.device('cuda'), include_current_stats=True,
                          postprocessors=processors, print_freq=128)


def compare(report, plan, arm, fold):
    views = {name: {key: report[name]['metrics'][key] for key in METRICS} for name in VIEWS}
    native, candidate = views['native'], views['same_query']
    if arm == 'full' and native != plan['known_references'][fold]:
        raise ValueError(f'Full-gradient native metrics do not match the archived {fold} endpoint')
    delta = {key: candidate[key] - native[key] for key in METRICS}
    gates = dict(known_ap50=delta['known_ap50'] >= plan['min_ap_change'][arm], u_recall=delta['u_recall'] >= 0.,
                 wi=delta['wi'] <= plan['max_wi_increase'],
                 a_ose=delta['a_ose'] <= max(1., plan['max_aose_fraction'] * native['a_ose']))
    return dict(views=views, delta=delta, gates=gates, passed=all(gates.values()))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 to replay both S Task 1 folds')
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    outcomes = {}
    for arm, endpoint in plan['arms'].items():
        model, _ = load_detector(None, Path(endpoint['checkpoint']), torch.device('cuda'))
        outcomes[arm] = {}
        for fold, entry in plan['folds'].items():
            dest = OUTPUT / f'{arm}-{fold}.json'
            if not dest.is_file():
                print(f'Evaluating S Task 1 {arm}/{fold} same-query competition', flush=True)
                report = evaluate(model, manifest, Path(entry['path']), plan)
                result = compare(report, plan, arm, fold)
                write_json(dest, dict(plan_sha256=digest, report=report, outcome=result))
            record = read_json(dest)
            if record['plan_sha256'] != digest:
                raise ValueError('Changed S Task 1 competition replay plan')
            outcomes[arm][fold] = record['outcome']
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, outcomes=outcomes,
                                               passed=all(row['passed'] for folds in outcomes.values()
                                                          for row in folds.values())))


if __name__ == '__main__':
    main()
