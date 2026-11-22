"""Validate one-slot unknown refill on independent paired Task 2 folds."""

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
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import annotation_digest, raw_evaluation_source
from tools.owod.screening_data import subset, write_json


OUTPUT = SOURCE.with_name('task2-m-s-unknown-refill-independent')
PARENT = SOURCE.with_name('task2-m-s-unknown-refill')
PARENT_RESULT = '5d352e7889bda076c3b430a3d080b1c46d5cf8116dd83e384e75dcf54bb694d7'
SOURCE_PLAN = 'de4bb126945143a69b5a60e60b83b7761c6270e6e71734fe127129948fd35712'
FOLD_SEEDS = {'first': 65, 'second': 66}
POOL_SEED = 67


def summarize_refill(report, shared, plan):
    flipped = 0
    for image, arms in report['per_image'].items():
        native, refill = (arms[k] for k in ('native', 'unknown_refill'))
        queries = report['queries_per_image'][image]
        extra = refill['unknown'] - native['unknown']
        if (extra not in (0, 1) or queries['unknown_refill'] - queries['native'] != extra
                or any(refill[k] > native[k] for k in ('previous', 'current'))):
            raise ValueError(f'Invalid single-slot refill: {image}')
        flipped += extra
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
    return dict(summary, refills=flipped, unknown_recall_gain=gain, open_metrics=metrics, passed=passed)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    result = verified(PARENT / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete' or not result['accepted']:
        raise ValueError('Expected the completed passing discovery comparison')
    original = verified(PARENT / 'plan.json', result['plan_sha256'])
    original_folds = verified(SOURCE / 'plan.json', SOURCE_PLAN)
    excluded = {name for spec in original_folds['cohorts'].values() for name in spec['order']}
    print('Preparing two independent paired Task 2 folds', flush=True)
    cohort_dir = OUTPUT / 'cohorts'
    source, manifests, roots = prepare(output=cohort_dir, fold_seeds=FOLD_SEEDS,
                                       pool_seed=POOL_SEED, excluded_sources=excluded)
    if any(excluded.intersection(cohort['order']) for cohort in source['cohorts'].values()):
        raise ValueError('Repeated discovery images in independent folds')
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
            path = OUTPUT / f'{protocol}-{fold}-full.json'
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
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for protocol, info in source['protocols'].items():
        checkpoint = Path(info['endpoints']['finetune']['path'])
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        known = info['previous'] + info['current']
        post = UnknownRefillScore(known_ids=known, max_detections=plan['limit'], scoring='focal')
        summaries[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} on independent images', flush=True)
                annotation = cohort['by_protocol'][protocol]
                dataset, loader = make_loader(roots[protocol], cohort_dir / annotation['file'], plan['batch_size'], 2)
                evaluation = dict(manifests[protocol]['evaluation'], image_source_ids=cohort['order'])
                full = COCO(str(OUTPUT / reports[label]['file']))
                report = measure(model, dataset, loader, known, info['previous'], info['current'], evaluation,
                                 plan['limit'], evaluation_coco=full, extra_posts=dict(unknown_refill=post),
                                 audit_query_counts=True, include_perturbed=False)
                write_json(dest, dict(plan_sha256=digest, **report))
            report = read_json(dest)
            if report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order']):
                raise ValueError(f'Changed independent replay: {label}')
            pairs = verified(cohort_dir / cohort['paired_file'], cohort['paired_sha256'])
            shared = {pair['annotation_ids'][protocol] for pair in pairs}
            summaries[protocol][fold] = summarize_refill(report, shared, plan)
        del model
    accepted = all(row['passed'] for folds in summaries.values() for row in folds.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=summaries,
               accepted=accepted, paper_comparable=False))


if __name__ == '__main__':
    main()
