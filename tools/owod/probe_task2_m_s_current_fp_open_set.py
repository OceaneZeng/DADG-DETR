"""Check unresolved Current false positives against unreleased author annotations."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import OUTPUT as SCORES, measure
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import annotation_digest, raw_evaluation_source
from tools.owod.screening_data import subset, write_json


OUTPUT = SCORES.with_name('task2-m-s-current-fp-open-set')
FP_REPORTS = SCORES.with_name('task2-m-s-current-fp')
FP_PLAN = '080f4d1ec0dfd8754fb3b1fd52b8ebab170706d9192b0bb6228fbe7d344c5a3f'
FP_RESULT = '613c1ad9125e1228b97cd1f77e5dfc17877c6623e5c74b677fd74553156a0e4c'
OPEN_KEYS = ('unknown_gt', 'unknown_predictions', 'u_recall', 'wi', 'a_ose')
OPEN_RESULT = '5626099bf497f1467272789a293eec0e0164a7597bc7a779973b68f326f0ae38'
RESIDUAL_BANDS = ('zero', 'positive_below_point_one', 'at_least_point_one')


def append_unknown(released, future):
    by_source = {im['owod_source_id']: im for im in future['images']}
    if by_source != {im['owod_source_id']: im for im in released['images']}:
        raise ValueError('Changed author image metadata')
    known_rows = released['annotations']
    first_id = max(row['id'] for row in known_rows) + 1
    return dict(released, annotations=[*known_rows, *(
        dict(row, id=i) for i, row in enumerate(future['annotations'], first_id)
    )])


def summarize(report, prior):
    if report['per_image'] != prior['per_image']:
        raise ValueError('Changed per-image Top-50 composition')
    sizes, metrics = {}, {}
    for name, view in report['views'].items():
        baseline = prior['views'][name]
        counts = view['metrics']['current_class_stats']
        for category, row in counts.items():
            original = baseline['metrics']['current_class_stats'][category]
            if dict(row, fp_sources={k: v for k, v in row['fp_sources'].items()
                                    if k not in ('unresolved_unknown_overlap', 'residual_overlap')}) != original:
                raise ValueError(f'Changed Current VOC metrics or false-positive tally: {name}/{category}')
        for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'unknown_predictions'):
            if view['metrics'][key] != baseline['metrics'][key]:
                raise ValueError(f'Changed frozen metrics: {name}/{key}')
        if view['matched'] != baseline['matched']:
            raise ValueError(f'Changed released true positives: {name}')
        total = sum(row['fp_sources']['families']['unresolved']['fp'] for row in counts.values())
        overlap = sum(row['fp_sources']['unresolved_unknown_overlap'] for row in counts.values())
        if not 0 <= overlap <= total:
            raise ValueError('Invalid unknown overlap count')
        sizes[name] = dict(unresolved=total, overlap=overlap, no_overlap=total-overlap)
        metrics[name] = {k: view['metrics'][k] for k in OPEN_KEYS}
    delta = {key: sizes['perturbed'][key] - sizes['native'][key] for key in sizes['native']}
    return dict(arms=sizes, delta=delta, majority_unknown_overlap=2*delta['overlap'] > delta['unresolved'],
                open_metrics=metrics)


def summarize_residual(report, prior, base):
    arms = {}
    for name, view in report['views'].items():
        original = prior['views'][name]
        stats = view['metrics']['current_class_stats']
        stripped = {}
        bands = dict.fromkeys((*RESIDUAL_BANDS, 'difficult_high_overlap'), 0)
        for category, row in stats.items():
            fp = dict(row['fp_sources'])
            residual = fp.pop('residual_overlap')
            if sum(residual[k] for k in RESIDUAL_BANDS) != (
                    fp['families']['unresolved']['fp'] - fp['unresolved_unknown_overlap']):
                raise ValueError(f'Incomplete residual attribution: {name}/{category}')
            stripped[category] = dict(row, fp_sources=fp)
            for key in bands:
                bands[key] += residual[key]
        if (dict(view['metrics'], current_class_stats=stripped) != original['metrics']
                or view['matched'] != original['matched']
                or report['per_image'] != prior['per_image']):
            raise ValueError(f'Changed parent open-set report: {name}')
        arms[name] = bands
    delta = {k: arms['perturbed'][k] - arms['native'][k] for k in arms['native']}
    if sum(delta[k] for k in RESIDUAL_BANDS) != base['delta']['no_overlap']:
        raise ValueError('Changed net residual count')
    below = delta['zero'] + delta['positive_below_point_one']
    return dict(arms=arms, delta=delta, below_point_one=below,
                majority_below_point_one=2*below > base['delta']['no_overlap'])


def main(residual_overlap=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    previous_plan = verified(FP_REPORTS / 'plan.json', FP_PLAN)
    previous = verified(FP_REPORTS / 'comparison.json', FP_RESULT)
    if previous['status'] != 'complete' or previous['plan_sha256'] != FP_PLAN or previous['shared_priority'] != 'unresolved':
        raise ValueError('Missing completed false-positive attribution')
    source = verified(SOURCE / 'plan.json', previous_plan['parent_plan'])
    output = OUTPUT
    open_base = open_plan = None
    if residual_overlap:
        open_base = verified(OUTPUT / 'comparison.json', OPEN_RESULT)
        open_plan = verified(OUTPUT / 'plan.json', open_base['plan_sha256'])
        if (open_base['status'] != 'complete' or open_plan['prior_plan'] != FP_PLAN
                or open_plan['prior_result'] != FP_RESULT):
            raise ValueError('Changed full-annotation baseline')
        output = OUTPUT.with_name('task2-m-s-current-fp-residual')
    reports, manifests, assets = {}, {}, {}
    for protocol, entry in source['protocols'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = manifests[protocol] = verified(manifest_path, entry['manifest_sha256'])
        if residual_overlap:
            for fold in source['cohorts']:
                label = f'{protocol}-{fold}'
                reports[label] = dict(open_plan['reports'][label],
                                      open_report_sha256=file_sha256(OUTPUT / f'{label}.json'))
                verified(OUTPUT / reports[label]['file'], reports[label]['sha256'])
            continue
        all_names = {item for c in source['cohorts'].values() for item in c['order']}
        train_names = set(manifest['splits']['t1_train']) | set(manifest['splits']['t2_train'])
        unknown = set(manifest['stages'][1]['unknown_classes'])
        future = raw_evaluation_source(protocol, manifest, train_names - all_names, allowed=unknown)
        for fold, cohort in source['cohorts'].items():
            path = cohort['by_protocol'][protocol]
            released = verified(SOURCE / path['file'], path['sha256'])
            extra = subset(future, set(cohort['order']), unknown)
            full = append_unknown(released, extra)
            label = f'{protocol}-{fold}'
            filename = f'{label}-annotations.json'
            assets[filename] = full
            reports[label] = dict(file=filename, sha256=annotation_digest(full),
                                  source_report_sha256=file_sha256(FP_REPORTS / f'{label}.json'),
                                  additional_annotations=len(extra['annotations']))
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                prior_plan=FP_PLAN, prior_result=FP_RESULT, parent_plan=previous_plan['parent_plan'],
                multiplier=previous_plan['multiplier'], max_detections=previous_plan['max_detections'],
                batch_size=previous_plan['batch_size'], paper_comparable=False, reports=reports)
    if residual_overlap:
        plan.update(open_base_result=OPEN_RESULT, open_base_plan=open_base['plan_sha256'],
                    residual_iou_bands=RESIDUAL_BANDS)
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    for name, data in assets.items():
        path = OUTPUT / name
        if not path.exists():
            write_json(path, data)
        if file_sha256(path) != annotation_digest(data):
            raise ValueError(f'Changed full evaluation input: {name}')
    results = {}
    for protocol, entry in source['protocols'].items():
        manifest = manifests[protocol]
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        known = entry['previous'] + entry['current']
        endpoint = entry['endpoints']['finetune']
        checkpoint = Path(endpoint['path'])
        if file_sha256(checkpoint) != endpoint['sha256']:
            raise ValueError(f'Changed frozen checkpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        results[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            dest = output / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} with unreleased labels used for scoring', flush=True)
                released = SOURCE / cohort['by_protocol'][protocol]['file']
                dataset, loader = make_loader(root, released, plan['batch_size'], 2)
                evaluation_coco = COCO(str(OUTPUT / reports[label]['file']))
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'],
                            diagnostic_current_fp_sources=True, diagnostic_unresolved_unknown_overlap=True)
                if residual_overlap:
                    spec['diagnostic_residual_overlap'] = True
                result = measure(model, dataset, loader, known, entry['previous'], entry['current'], spec,
                                 plan['max_detections'], evaluation_coco=evaluation_coco)
                write_json(dest, dict(plan_sha256=digest, **result))
            report = read_json(dest)
            prior = verified(FP_REPORTS / dest.name, reports[label]['source_report_sha256'])
            if report['plan_sha256'] != digest or prior['plan_sha256'] != FP_PLAN:
                raise ValueError(f'Changed evaluation run: {label}')
            results[protocol][fold] = summarize(report, prior)
            if residual_overlap:
                older = verified(OUTPUT / dest.name, reports[label]['open_report_sha256'])
                base = open_base['protocols'][protocol][fold]
                if older['plan_sha256'] != open_base['plan_sha256'] or results[protocol][fold] != base:
                    raise ValueError(f'Changed open-set baseline: {label}')
                results[protocol][fold]['residual'] = summarize_residual(report, older, base)
        del model
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=results, paper_comparable=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-overlap', action='store_true')
    main(parser.parse_args().residual_overlap)
