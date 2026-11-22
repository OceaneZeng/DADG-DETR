"""Compare three- and six-epoch Task 2 endpoints on the same protected-refill folds."""

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
SHORT_RESULT = '235e6303e8df7511fcf3271f2c33268341066dd808e169deb6f23c8ac974cb9b'
M_ENDPOINTS = 'f1bfb8aa346039aa631f17b9f710f2f039af1c6d5829650c5d8895225fa7fd83'
S_ENDPOINTS = '4f698b6c4c608cb835f063960416ba546af06e557356f623555b4158eee9172f'
DURATION_MARGIN = .01
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')


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


def main():
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
    plan_path = OUTPUT / 'plan.json'
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
                                  protected_ids=info['current'])
        results[protocol] = {}
        for fold, cohort in cohorts['cohorts'].items():
            label = f'{protocol}-{fold}'
            annotation = cohort['by_protocol'][protocol]
            path = cohort_dir / annotation['file']
            verified(path, annotation['sha256'])
            full = parent['reports'][label]
            full_path = PROTECTED_OUTPUT / full['file']
            verified(full_path, full['sha256'])
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at the six-epoch finetune endpoint', flush=True)
                dataset, loader = make_loader(root, path, parent['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                report = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 parent['limit'], evaluation_coco=COCO(str(full_path)),
                                 extra_posts=dict(unknown_refill=post), audit_query_counts=True,
                                 include_perturbed=False)
                write_json(dest, dict(plan_sha256=digest, **report))
            report = read_json(dest)
            if report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order']):
                raise ValueError(f'Changed duration replay: {label}')
            pairs = verified(cohort_dir / cohort['paired_file'], cohort['paired_sha256'])
            shared = {pair['annotation_ids'][protocol] for pair in pairs}
            reference = read_json(PROTECTED_OUTPUT / f'{label}.json')
            if (reference['plan_sha256'] != short['plan_sha256']
                    or reference['per_image'].keys() != report['per_image'].keys()):
                raise ValueError(f'Changed three-epoch matched cohort: {label}')
            results[protocol][fold] = compare(reference, report, summarize_refill(report, shared, plan))
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=results, paper_comparable=False))


if __name__ == '__main__':
    main()
