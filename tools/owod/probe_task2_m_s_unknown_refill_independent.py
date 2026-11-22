"""Validate one-slot unknown refill on independent paired Task 2 folds."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_fp_open_set import OPEN_KEYS, append_unknown
from tools.owod.probe_task2_m_s_current_score import measure, summarize
from tools.owod.probe_task2_m_s_current_top_class import UnknownRefillScore
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE, prepare
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import annotation_digest, raw_evaluation_source
from tools.owod.screening_data import subset, write_json


OUTPUT = SOURCE.with_name('task2-m-s-unknown-refill-independent')
PARENT = SOURCE.with_name('task2-m-s-unknown-refill')
PROTECTED_DISCOVERY = SOURCE.with_name('task2-m-s-refill-protect-current')
PROTECTED_OUTPUT = SOURCE.with_name('task2-m-s-refill-protect-current-independent')
OVERLAP_DISCOVERY = SOURCE.with_name('task2-m-s-refill-same-class-overlap')
OVERLAP_OUTPUT = SOURCE.with_name('task2-m-s-refill-same-class-overlap-independent')
PARENT_RESULT = '5d352e7889bda076c3b430a3d080b1c46d5cf8116dd83e384e75dcf54bb694d7'
PROTECTED_RESULT = '43b7d4533df8b38a0fa725bdd0bd5fb94cd76a8b750e7c9d8cafe8eee06290c5'
OVERLAP_RESULT = 'e291b6cc0a269949a721a492e6f67b5c37233bfe5faa3190da4d8577f361985a'
SOURCE_PLAN = 'de4bb126945143a69b5a60e60b83b7761c6270e6e71734fe127129948fd35712'
FOLD_SEEDS = {'first': 65, 'second': 66}
POOL_SEED = 67
PROTECTED_FOLD_SEEDS = {'first': 68, 'second': 69}
PROTECTED_POOL_SEED = 70
OVERLAP_FOLD_SEEDS = {'first': 71, 'second': 72}
OVERLAP_POOL_SEED = 73
PREPARED_COHORT = '26dd2b7e3579687b4fb247018017029aec9796ab3aae3db5a4cc5118e598cf39'


def summarize_refill(report, shared, plan):
    protect_current = plan.get('protected_group') == 'current'
    flipped = 0
    for image, arms in report['per_image'].items():
        native, refill = (arms[k] for k in ('native', 'unknown_refill'))
        queries = report['queries_per_image'][image]
        extra = refill['unknown'] - native['unknown']
        if (extra not in (0, 1) or queries['unknown_refill'] - queries['native'] != extra
                or any(refill[k] > native[k] for k in ('previous', 'current'))):
            raise ValueError(f'Invalid single-slot refill: {image}')
        if protect_current and refill['current'] != native['current']:
            raise ValueError(f'Changed protected Current output count: {image}')
        flipped += extra
    if protect_current:
        native, refill = (report['views'][arm] for arm in ('native', 'unknown_refill'))
        if (native['matched'] != refill['matched']
                or any(native['metrics'][key] != refill['metrics'][key]
                       for key in ('current_ap50', 'current_class_stats'))):
            raise ValueError('Changed protected Current author metrics')
    pair = dict(views=dict(native=report['views']['native'], perturbed=report['views']['unknown_refill']),
                per_image={image: dict(native=v['native'], perturbed=v['unknown_refill'])
                           for image, v in report['per_image'].items()})
    summary = summarize(pair, shared)
    metrics = {arm: {key: view['metrics'][key] for key in OPEN_KEYS}
               for arm, view in report['views'].items()}
    native, refill = (metrics[k] for k in ('native', 'unknown_refill'))
    gain = refill['u_recall'] - native['u_recall']
    passed = (gain >= plan['minimum_unknown_recall_gain']
              and all(delta >= -plan['refill_ap_tolerance'] for delta in summary['deltas'].values())
              and refill['a_ose'] <= native['a_ose'])
    result = dict(summary, refills=flipped, unknown_recall_gain=gain, open_metrics=metrics, passed=passed)
    if 'previous_matches' in report:
        before, after = (report['previous_matches'][arm] for arm in ('native', 'unknown_refill'))
        result['previous_matches_lost'] = sum(len(set(ids) - set(after[category]))
                                              for category, ids in before.items())
    return result


def main(protect_current=False, same_class_overlap=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    result = verified(PARENT / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete' or not result['accepted']:
        raise ValueError('Expected the completed passing discovery comparison')
    original = verified(PARENT / 'plan.json', result['plan_sha256'])
    original_folds = verified(SOURCE / 'plan.json', SOURCE_PLAN)
    excluded = {name for spec in original_folds['cohorts'].values() for name in spec['order']}
    output, fold_seeds, pool_seed = OUTPUT, FOLD_SEEDS, POOL_SEED
    protected_plan = None
    overlap_plan = None
    protect_current = protect_current or same_class_overlap
    if protect_current:
        output, fold_seeds, pool_seed = PROTECTED_OUTPUT, PROTECTED_FOLD_SEEDS, PROTECTED_POOL_SEED
        selection = verified(PROTECTED_DISCOVERY / 'comparison.json', PROTECTED_RESULT)
        if selection['status'] != 'complete' or selection['folds_passing'] != 4:
            raise ValueError('Expected four passing protected discovery folds')
        protected_plan = verified(PROTECTED_DISCOVERY / 'plan.json', selection['plan_sha256'])
        if (protected_plan['protected_group'] != 'current'
                or any(protected_plan[k] != original[k] for k in
                       ('minimum_unknown_recall_gain', 'refill_ap_tolerance'))):
            raise ValueError('Changed protected refill rule or acceptance limits')
        discovery_cohorts = verified(OUTPUT / 'cohorts/plan.json', protected_plan['cohort_plan'])
        excluded.update(name for spec in discovery_cohorts['cohorts'].values() for name in spec['order'])
    if same_class_overlap:
        output, fold_seeds, pool_seed = OVERLAP_OUTPUT, OVERLAP_FOLD_SEEDS, OVERLAP_POOL_SEED
        selection = verified(OVERLAP_DISCOVERY / 'comparison.json', OVERLAP_RESULT)
        rows = [row for folds in selection['protocols'].values() for row in folds.values()]
        if selection['status'] != 'complete' or len(rows) != 4 or not all(row['passed'] for row in rows):
            raise ValueError('Expected four passing overlap discovery folds')
        overlap_plan = verified(OVERLAP_DISCOVERY / 'plan.json', selection['plan_sha256'])
        if (overlap_plan['protected_group'] != 'current' or overlap_plan['min_same_class_iou'] != .7
                or any(overlap_plan[key] != original[key] for key in
                       ('minimum_unknown_recall_gain', 'refill_ap_tolerance'))):
            raise ValueError('Changed overlap rule or acceptance limits')
        overlap_cohorts = verified(PROTECTED_OUTPUT / 'cohorts/plan.json', overlap_plan['cohort_plan'])
        excluded.update(name for spec in overlap_cohorts['cohorts'].values() for name in spec['order'])
    print('Preparing two independent paired Task 2 folds', flush=True)
    cohort_dir = output / 'cohorts'
    cohort_path = cohort_dir / 'plan.json'
    run_plan_path = output / 'plan.json'
    if cohort_path.exists() and (run_plan_path.exists() or not protect_current):
        locked = read_json(run_plan_path)['cohort_plan'] if run_plan_path.exists() else PREPARED_COHORT
        source = verified(cohort_path, locked)
        manifests = {p: verified(MANIFESTS[p], info['manifest_sha256']) for p, info in source['protocols'].items()}
        roots = {p: resolve_image_root(MANIFESTS[p], manifest) for p, manifest in manifests.items()}
    else:
        source, manifests, roots = prepare(output=cohort_dir, fold_seeds=fold_seeds,
                                           pool_seed=pool_seed, excluded_sources=excluded,
                                           allow_smaller_pool=protect_current)
    used = set(excluded)
    for cohort in source['cohorts'].values():
        if used.intersection(cohort['order']):
            raise ValueError('Repeated images in independent folds')
        used.update(cohort['order'])
    if source['protocols'] != original_folds['protocols']:
        raise ValueError('Changed protocol or checkpoint identity')
    reports = {}
    for protocol, manifest in manifests.items():
        names = {name for spec in source['cohorts'].values() for name in spec['order']}
        training = set(manifest['splits']['t1_train']) | set(manifest['splits']['t2_train'])
        unknown = set(manifest['stages'][1]['unknown_classes'])
        future = raw_evaluation_source(protocol, manifest, training - names, allowed=unknown)
        for fold, cohort in source['cohorts'].items():
            entry = cohort['by_protocol'][protocol]
            released = verified(cohort_dir / entry['file'], entry['sha256'])
            full = append_unknown(released, subset(future, set(cohort['order']), unknown))
            path = output / f'{protocol}-{fold}-full.json'
            if not path.exists():
                write_json(path, full)
            expected = annotation_digest(full)
            if file_sha256(path) != expected:
                raise ValueError(f'Changed full evaluation inputs: {protocol}/{fold}')
            reports[f'{protocol}-{fold}'] = dict(file=path.name, sha256=expected)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_result=PARENT_RESULT, parent_plan=result['plan_sha256'], discovery_cohort_plan=SOURCE_PLAN,
                cohort_plan=file_sha256(cohort_dir / 'plan.json'), reports=reports,
                minimum_unknown_recall_gain=original['minimum_unknown_recall_gain'],
                refill_ap_tolerance=original['refill_ap_tolerance'],
                max_refills_per_image=original['max_refills_per_image'],
                batch_size=source['batch_size'], limit=source['max_detections'], paper_comparable=False)
    if protected_plan is not None:
        plan.update(protected_group='current', protected_discovery_result=PROTECTED_RESULT,
                    protected_discovery_cohort=protected_plan['cohort_plan'])
    if overlap_plan is not None:
        plan.update(overlap_discovery_result=OVERLAP_RESULT, overlap_discovery_cohort=overlap_plan['cohort_plan'],
                    endpoints=overlap_plan['endpoints'], min_same_class_iou=overlap_plan['min_same_class_iou'])
    if run_plan_path.exists():
        pin(run_plan_path, plan)
    else:
        prepared = {'cohorts', 'run.lock', 'console.log'} | {entry['file'] for entry in reports.values()}
        if any(path.name not in prepared for path in output.iterdir()):
            raise ValueError('Unexpected files before the independent replay plan')
        write_json(run_plan_path, plan)
    digest = file_sha256(run_plan_path)
    summaries = {}
    for protocol, info in source['protocols'].items():
        endpoint = plan['endpoints'][protocol] if same_class_overlap else info['endpoints']['finetune']
        checkpoint = Path(endpoint['path'])
        if file_sha256(checkpoint) != endpoint['sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        known = info['previous'] + info['current']
        protected = info['current'] if protect_current else ()
        post = UnknownRefillScore(known_ids=known, max_detections=plan['limit'], scoring='focal',
                                  protected_ids=protected, min_same_class_iou=plan.get('min_same_class_iou'))
        summaries[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            dest = output / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} on independent images', flush=True)
                annotation = cohort['by_protocol'][protocol]
                dataset, loader = make_loader(roots[protocol], cohort_dir / annotation['file'], plan['batch_size'], 2)
                evaluation = dict(manifests[protocol]['evaluation'], image_source_ids=cohort['order'])
                full = COCO(str(output / reports[label]['file']))
                report = measure(model, dataset, loader, known, info['previous'], info['current'], evaluation,
                                 plan['limit'], evaluation_coco=full, extra_posts=dict(unknown_refill=post),
                                 audit_query_counts=True, include_perturbed=False,
                                 audit_previous_matches=same_class_overlap)
                write_json(dest, dict(plan_sha256=digest, **report))
            report = read_json(dest)
            if report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order']):
                raise ValueError(f'Changed independent replay: {label}')
            pairs = verified(cohort_dir / cohort['paired_file'], cohort['paired_sha256'])
            shared = {pair['annotation_ids'][protocol] for pair in pairs}
            summaries[protocol][fold] = summarize_refill(report, shared, plan)
        del model
    accepted = all(row['passed'] for folds in summaries.values() for row in folds.values())
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=summaries,
               accepted=accepted, paper_comparable=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--protect-current', action='store_true')
    mode.add_argument('--same-class-overlap', action='store_true')
    args = parser.parse_args()
    main(args.protect_current, args.same_class_overlap)
