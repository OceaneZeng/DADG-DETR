"""Replay Task 2 unknown refill with score-order or Current-class controls."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_current_top_class import UnknownRefillScore
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS
from tools.owod.probe_task2_m_s_unknown_refill_independent import OUTPUT as SOURCE, summarize_refill
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-refill-score-order')
PROTECTED_OUTPUT = SOURCE.with_name('task2-m-s-refill-protect-current')
PARENT_RESULT = '8e51e2e4b4cdedf5c32a273fba893e3c9a6d17b5638aa27c77e2539424b9a5d3'
SCORE_ORDER_RESULT = '2e37540ea15f8ad2236fdb49eb9427580136552bf21cdb959ea0a4cab34fa723'
ORDERS = ('stable_quantized', 'stable_raw')
KEYS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')


def summarize(report, reference):
    if any(report[key] != reference[key] for key in ('views', 'per_image', 'queries_per_image')):
        raise ValueError('Changed author-order replay')
    author = {name: view['metrics'] for name, view in report['views'].items()}
    rows = {}
    for order, arms in dict(author=author, **report['score_order_views']).items():
        native, refill = (arms[k] for k in ('native', 'unknown_refill'))
        rows[order] = dict(
            deltas={key: refill[key] - native[key] for key in KEYS},
            zebra={name: arm['current_class_stats'].get('24') for name, arm in arms.items()},
        )
    return rows


def summarize_protected(report, reference, shared, plan):
    if (report['views']['native'] != reference['views']['native']
            or report['per_image'].keys() != reference['per_image'].keys()):
        raise ValueError('Changed native author-order replay')
    for image, arms in report['per_image'].items():
        if (arms['native'] != reference['per_image'][image]['native']
                or arms['native']['current'] != arms['unknown_refill']['current']):
            raise ValueError(f'Changed native or protected Current output counts: {image}')
    native, refill = (report['views'][arm] for arm in ('native', 'unknown_refill'))
    if (native['matched'] != refill['matched']
            or any(native['metrics'][key] != refill['metrics'][key]
                   for key in ('current_ap50', 'current_class_stats'))):
        raise ValueError('Changed protected Current author metrics')
    return summarize_refill(report, shared, plan)


def main(protect_current=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    result = verified(SOURCE / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete':
        raise ValueError('Missing completed independent refill')
    parent = verified(SOURCE / 'plan.json', result['plan_sha256'])
    cohort_dir = SOURCE / 'cohorts'
    cohorts = verified(cohort_dir / 'plan.json', parent['cohort_plan'])
    references = {name: file_sha256(SOURCE / f'{name}.json') for name in parent['reports']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan=result['plan_sha256'], parent_result=PARENT_RESULT,
                cohort_plan=parent['cohort_plan'], references=references, orders=list(ORDERS),
                minimum_recovery_fraction=.5, paper_comparable=False)
    output = PROTECTED_OUTPUT if protect_current else OUTPUT
    if protect_current:
        ordering = verified(OUTPUT / 'comparison.json', SCORE_ORDER_RESULT)
        if ordering['status'] != 'complete' or not ordering['ranking_sensitive']:
            raise ValueError('Missing the completed score-order diagnosis')
        del plan['minimum_recovery_fraction']
        plan.update(orders=[], source_score_order=SCORE_ORDER_RESULT, protected_group='current',
                    minimum_unknown_recall_gain=parent['minimum_unknown_recall_gain'],
                    refill_ap_tolerance=parent['refill_ap_tolerance'])
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    reports = {}
    for protocol, info in cohorts['protocols'].items():
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        checkpoint = Path(info['endpoints']['finetune']['path'])
        if file_sha256(checkpoint) != info['endpoints']['finetune']['sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        known = info['previous'] + info['current']
        protected = info['current'] if protect_current else ()
        post = UnknownRefillScore(known_ids=known, max_detections=parent['limit'], scoring='focal',
                                  protected_ids=protected)
        reports[protocol] = {}
        for fold, cohort in cohorts['cohorts'].items():
            label = f'{protocol}-{fold}'
            reference = verified(SOURCE / f'{label}.json', references[label])
            if reference['plan_sha256'] != result['plan_sha256']:
                raise ValueError(f'Changed independent source: {label}')
            annotation = cohort['by_protocol'][protocol]
            path = cohort_dir / annotation['file']
            verified(path, annotation['sha256'])
            full = parent['reports'][label]
            full_path = SOURCE / full['file']
            verified(full_path, full['sha256'])
            dest = output / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold}', flush=True)
                root = resolve_image_root(MANIFESTS[protocol], manifest)
                dataset, loader = make_loader(root, path, parent['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                replay = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 parent['limit'], evaluation_coco=COCO(str(full_path)),
                                 extra_posts=dict(unknown_refill=post), audit_query_counts=True,
                                 include_perturbed=False, diagnostic_score_orders=plan['orders'])
                write_json(dest, dict(plan_sha256=digest, **replay))
            replay = read_json(dest)
            if replay['plan_sha256'] != digest:
                raise ValueError(f'Changed score-order replay: {label}')
            if protect_current:
                pairs = verified(cohort_dir / cohort['paired_file'], cohort['paired_sha256'])
                shared = {pair['annotation_ids'][protocol] for pair in pairs}
                reports[protocol][fold] = summarize_protected(replay, reference, shared, plan)
            else:
                reports[protocol][fold] = summarize(replay, reference)
        del model
    if protect_current:
        passed = sum(row['passed'] for folds in reports.values() for row in folds.values())
        write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest,
                   protocols=reports, folds_passing=passed, paper_comparable=False))
        return
    focus = reports['m-owodb']['second']
    author = focus['author']['deltas']['current_ap50']
    sensitive = author < 0 and all(
        focus[mode]['deltas']['current_ap50'] - author >= -plan['minimum_recovery_fraction'] * author
        for mode in ORDERS)
    write_json(OUTPUT / 'comparison.json', dict(
        status='complete', plan_sha256=digest, protocols=reports, ranking_sensitive=sensitive,
        paper_comparable=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protect-current', action='store_true')
    main(parser.parse_args().protect_current)
