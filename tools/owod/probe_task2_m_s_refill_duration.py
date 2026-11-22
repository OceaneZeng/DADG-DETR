"""Compare three- and six-epoch Task 2 endpoints on the same protected-refill folds."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import UnknownRefillScore
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS
from tools.owod.probe_task2_m_s_unknown_refill_independent import PROTECTED_OUTPUT, summarize_refill
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = PROTECTED_OUTPUT.with_name('task2-m-s-refill-duration')
PREVIOUS_ORDER_OUTPUT = OUTPUT.with_name('task2-m-s-refill-previous-order')
OVERLAP_OUTPUT = OUTPUT.with_name('task2-m-s-refill-same-class-overlap')
SHORT_RESULT = '235e6303e8df7511fcf3271f2c33268341066dd808e169deb6f23c8ac974cb9b'
DURATION_RESULT = 'dd8eb5343ccb41605a4cc89358582cd9e4b1614ee43f6ee227e806511fe8acde'
PREVIOUS_ORDER_RESULT = '8a7f7deae0857ffaa8685f3274ac72ad8f7a17dd62afeb82dc374267859d10f3'
M_ENDPOINTS = 'f1bfb8aa346039aa631f17b9f710f2f039af1c6d5829650c5d8895225fa7fd83'
S_ENDPOINTS = '4f698b6c4c608cb835f063960416ba546af06e557356f623555b4158eee9172f'
DURATION_MARGIN = .01
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')
ORDERS = ('stable_quantized', 'stable_raw')
MIN_SAME_CLASS_IOU = .7


def checkpoints(cohorts):
    m = verified('output/m-instance-task2-duration-holdout/plan.json', M_ENDPOINTS)['checkpoints']
    s_root = Path('output/s-instance-task2-duration')
    s = verified(s_root / 'comparison.json', S_ENDPOINTS)
    if s['status'] != 'complete':
        raise ValueError('Missing completed S duration pair')
    expected = {}
    for epochs in (3, 6):
        arm = f'{epochs}epochs'
        entry = s['arms'][arm]['finetune']
        expected['m-owodb', epochs] = dict(path=m[f'seed42-{arm}']['path'],
                                          sha256=m[f'seed42-{arm}']['sha256'])
        expected['s-owodb', epochs] = dict(path=str((s_root / arm / 'finetune/checkpoint.pth').resolve()),
                                          sha256=entry['checkpoint_sha256'])
    for protocol, spec in cohorts['protocols'].items():
        short = expected[protocol, 3]
        if short != {key: spec['endpoints']['finetune'][key] for key in ('path', 'sha256')}:
            raise ValueError(f'Changed three-epoch reference: {protocol}')
        long = Path(expected[protocol, 6]['path'])
        end = read_json(long.parent / 'training_complete.json')
        inc = long.parent.parent / 'increment'
        inputs = read_json(long.parent / 'input.json')
        if (end['last_epoch'] != 0 or read_json(inc / 'training_complete.json')['last_epoch'] != 5
                or inputs['--pretrained'] != file_sha256(inc / 'checkpoint.pth')
                or expected[protocol, 6]['sha256'] != file_sha256(long)):
            raise ValueError(f'Changed six-epoch checkpoint chain: {long}')
    return {protocol: expected[protocol, 6] for protocol in cohorts['protocols']}


def compare(short, long, refill_summary):
    deltas = {}
    for arm in ('native', 'unknown_refill'):
        before, after = (report['views'][arm]['metrics'] for report in (short, long))
        deltas[arm] = {key: after[key] - before[key] for key in METRICS}
    score = deltas['native']
    return dict(long=refill_summary, six_minus_three=deltas, duration_passed=(
        score['current_ap50'] >= DURATION_MARGIN and score['previous_ap50'] >= -DURATION_MARGIN
        and score['known_ap50'] >= 0))


def compare_previous_order(reference, report, tolerance):
    arms = ('native', 'unknown_refill')
    for arm in arms:
        observed = report['views'][arm]
        original = reference['views'][arm]
        metrics = {key: value for key, value in observed['metrics'].items() if key != 'previous_class_stats'}
        if metrics != original['metrics'] or observed['matched'] != original['matched']:
            raise ValueError(f'Changed six-epoch author-order replay: {arm}')
    if any(report[key] != reference[key] for key in ('per_image', 'queries_per_image')):
        raise ValueError('Changed six-epoch output budget or queries')

    native, refill = (report['previous_matches'][arm] for arm in arms)
    before, after = (report['views'][arm]['metrics'] for arm in arms)
    classes = {}
    for category, ids in native.items():
        kept = set(refill[category])
        earlier = set(ids)
        ap_before, ap_after = (score['previous_class_stats'][category]['ap50'] for score in (before, after))
        classes[category] = dict(lost=sorted(earlier - kept), gained=sorted(kept - earlier),
                                 previous_ap50_delta=ap_after - ap_before)
    author = after['previous_ap50'] - before['previous_ap50']
    order_deltas = {}
    for order, views in report['score_order_views'].items():
        earlier, later = (views[arm]['previous_ap50'] for arm in arms)
        order_deltas[order] = later - earlier
    lost = sum(len(row['lost']) for row in classes.values())
    gained = sum(len(row['gained']) for row in classes.values())
    sensitive = (author < -tolerance and not (lost or gained)
                 and all(delta >= -tolerance for delta in order_deltas.values()))
    return dict(author_previous_ap50_delta=author, diagnostic_previous_ap50_deltas=order_deltas,
                lost=lost, gained=gained, classes=classes, ranking_sensitive=sensitive)


def compare_overlap(reference, report, summary):
    if (report['views']['native'] != reference['views']['native']
            or report['previous_matches']['native'] != reference['previous_matches']['native']
            or report['per_image'].keys() != reference['per_image'].keys()):
        raise ValueError('Changed native six-epoch reference')
    for image, arms in report['per_image'].items():
        if arms['native'] != reference['per_image'][image]['native']:
            raise ValueError(f'Changed native candidate count: {image}')
        if arms['unknown_refill']['unknown'] > reference['per_image'][image]['unknown_refill']['unknown']:
            raise ValueError(f'Overlap condition increased the number of refill slots: {image}')
    native, refill = (report['previous_matches'][arm] for arm in ('native', 'unknown_refill'))
    lost = {category: sorted(set(ids) - set(refill[category])) for category, ids in native.items()}
    return dict(summary, previous_matches_lost=sum(map(len, lost.values())),
                previous_lost_by_class={category: ids for category, ids in lost.items() if ids})


def main(audit_previous=False, same_class_overlap=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    short = verified(PROTECTED_OUTPUT / 'comparison.json', SHORT_RESULT)
    if short['status'] != 'complete' or not short['accepted']:
        raise ValueError('Missing completed protected refill reference')
    parent = verified(PROTECTED_OUTPUT / 'plan.json', short['plan_sha256'])
    cohort_dir = PROTECTED_OUTPUT / 'cohorts'
    cohorts = verified(cohort_dir / 'plan.json', parent['cohort_plan'])
    print('Verifying six-epoch endpoints and the frozen independent folds', flush=True)
    endpoints = checkpoints(cohorts)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                short_result=SHORT_RESULT, short_plan=short['plan_sha256'], cohort_plan=parent['cohort_plan'],
                endpoints=endpoints, metadata_hashes={'m-owodb': M_ENDPOINTS, 's-owodb': S_ENDPOINTS},
                duration_margin=DURATION_MARGIN, protected_group='current',
                minimum_unknown_recall_gain=parent['minimum_unknown_recall_gain'],
                refill_ap_tolerance=parent['refill_ap_tolerance'], paper_comparable=False)
    output = PREVIOUS_ORDER_OUTPUT if audit_previous else OUTPUT
    if audit_previous:
        duration = verified(OUTPUT / 'comparison.json', DURATION_RESULT)
        previous = verified(OUTPUT / 'plan.json', duration['plan_sha256'])
        if any(previous[key] != plan[key] for key in plan if key != 'commit'):
            raise ValueError('Changed six-epoch replay plan')
        references = {label: file_sha256(OUTPUT / f'{label}.json') for label in parent['reports']}
        plan.update(duration_result=DURATION_RESULT, duration_plan=duration['plan_sha256'],
                    references=references, diagnostic_score_orders=list(ORDERS))
    elif same_class_overlap:
        output = OVERLAP_OUTPUT
        previous_order = verified(PREVIOUS_ORDER_OUTPUT / 'comparison.json', PREVIOUS_ORDER_RESULT)
        checked = verified(PREVIOUS_ORDER_OUTPUT / 'plan.json', previous_order['plan_sha256'])
        if any(checked[key] != plan[key] for key in plan if key != 'commit'):
            raise ValueError('Changed previous-order replay plan')
        references = {label: file_sha256(PREVIOUS_ORDER_OUTPUT / f'{label}.json') for label in parent['reports']}
        plan.update(previous_order_result=PREVIOUS_ORDER_RESULT, previous_order_plan=previous_order['plan_sha256'],
                    references=references, min_same_class_iou=MIN_SAME_CLASS_IOU)
    plan_path = output / 'plan.json'
    pin(plan_path, plan)
    digest = file_sha256(plan_path)
    results = {}
    for protocol, info in cohorts['protocols'].items():
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        model, _ = load_detector(None, Path(endpoints[protocol]['path']), torch.device('cuda'))
        model.eval()
        known = info['previous'] + info['current']
        post = UnknownRefillScore(known_ids=known, max_detections=parent['limit'], scoring='focal',
                                  protected_ids=info['current'], min_same_class_iou=plan.get('min_same_class_iou'))
        results[protocol] = {}
        for fold, cohort in cohorts['cohorts'].items():
            label = f'{protocol}-{fold}'
            annotation = cohort['by_protocol'][protocol]
            path = cohort_dir / annotation['file']
            verified(path, annotation['sha256'])
            full = parent['reports'][label]
            full_path = PROTECTED_OUTPUT / full['file']
            verified(full_path, full['sha256'])
            dest = output / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at the six-epoch finetune endpoint', flush=True)
                dataset, loader = make_loader(root, path, parent['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                report = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 parent['limit'], evaluation_coco=COCO(str(full_path)),
                                 extra_posts=dict(unknown_refill=post), audit_query_counts=True,
                                 include_perturbed=False, audit_previous_matches=audit_previous or same_class_overlap,
                                 diagnostic_score_orders=plan.get('diagnostic_score_orders', ()))
                write_json(dest, dict(plan_sha256=digest, **report))
            report = read_json(dest)
            if report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order']):
                raise ValueError(f'Changed duration replay: {label}')
            if audit_previous:
                reference = verified(OUTPUT / f'{label}.json', plan['references'][label])
                if reference['plan_sha256'] != plan['duration_plan']:
                    raise ValueError(f'Changed duration reference: {label}')
                results[protocol][fold] = compare_previous_order(reference, report, plan['refill_ap_tolerance'])
                continue
            pairs = verified(cohort_dir / cohort['paired_file'], cohort['paired_sha256'])
            shared = {pair['annotation_ids'][protocol] for pair in pairs}
            summary = summarize_refill(report, shared, plan)
            if same_class_overlap:
                reference = verified(PREVIOUS_ORDER_OUTPUT / f'{label}.json', plan['references'][label])
                if reference['plan_sha256'] != plan['previous_order_plan']:
                    raise ValueError(f'Changed previous-order reference: {label}')
                results[protocol][fold] = compare_overlap(reference, report, summary)
                continue
            reference = read_json(PROTECTED_OUTPUT / f'{label}.json')
            if (reference['plan_sha256'] != short['plan_sha256']
                    or reference['per_image'].keys() != report['per_image'].keys()):
                raise ValueError(f'Changed three-epoch matched cohort: {label}')
            results[protocol][fold] = compare(reference, report, summary)
        del model
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=results, paper_comparable=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-previous', action='store_true')
    mode.add_argument('--same-class-overlap', action='store_true')
    args = parser.parse_args()
    main(args.audit_previous, args.same_class_overlap)
