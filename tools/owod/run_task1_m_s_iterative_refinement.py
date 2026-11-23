"""Screen iterative decoder box refinement from the shared Task 1 initialization."""

import argparse
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_formal_refill_capacity import SOURCES
from tools.owod.probe_task2_m_s_pseudo_budget_negatives import OUTPUT as BUDGET
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.run_task1_m_s_released_small_extent import (
    GATES as EXTENT_GATES, MANIFESTS, OUTPUT as SLICES, measure,
)
from tools.owod.screening_data import screening_files, source_ids, write_json


OUTPUT = BUDGET.with_name('task1-m-s-iterative-refinement')
BUDGET_SHA = '4b8c310cfcea94e1f7e6d48cff0618b08d1f193741ccaf6f3512da29960433bb'
SLICES_SHA = 'fd056b5878413cdb35917a3b5c3e3c9fefbcde55cad22ab89c0cff08a6cf3691'
GATES = {key: value for key, value in EXTENT_GATES.items() if key != 'small_dense'}
GATES['small_final'] = EXTENT_GATES['small_dense']
ARMS = ('control', 'refinement')


def prepare():
    evidence = verified(BUDGET / 'comparison.json', BUDGET_SHA)
    if evidence['status'] != 'complete' or evidence['passed']:
        raise ValueError('Expected the completed pseudo-budget audit')
    directory, result_hash, endpoints_hash = SOURCES['three']
    comparison = verified(directory / 'comparison.json', result_hash)
    original = verified(directory / 'plan.json', comparison['plan_sha256'])
    slices = verified(SLICES / 'plan.json', SLICES_SHA)
    endpoint_file = directory / 'endpoint-sha256.txt'
    if file_sha256(endpoint_file) != endpoints_hash:
        raise ValueError('Changed shared-backbone control checkpoint digests')
    endpoints = {name: digest for digest, name in
                 (line.split() for line in endpoint_file.read_text(encoding='utf-8').splitlines())}
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if gpu_ids != original['gpu_ids'] or gpus != original['device_names']:
        raise ValueError('Use the same devices as the three-epoch control')
    jobs = {}
    for protocol, source in original['jobs'].items():
        benchmark = source['benchmark']
        stage = benchmark['runs'][0]
        if (stage['stage'], stage['phase'], stage['epochs']) != (0, 'increment', 3):
            raise ValueError(f'Changed Task 1 control stage: {protocol}')
        checkpoint = Path(stage['output_dir']) / 'checkpoint.pth'
        control = endpoints[checkpoint.resolve().relative_to(directory.resolve()).as_posix()]
        if file_sha256(checkpoint) != control:
            raise ValueError(f'Changed Task 1 control checkpoint: {protocol}')
        marker = read_json(checkpoint.with_name('training_complete.json'))
        if marker['last_epoch'] != 2:
            raise ValueError(f'Incomplete Task 1 control endpoint: {protocol}')
        manifest_path = MANIFESTS[protocol]
        manifest_digest = benchmark['manifest_sha256']
        verified(manifest_path, manifest_digest)
        command = list(stage['command'])
        if any(flag in command for flag in ('--with_box_refine', '--resume', '--pretrained')):
            raise ValueError('Expected a fresh Task 1 control without iterative refinement')
        screen = Path(command[command.index('--owod-screening') + 1])
        backbone = Path(command[command.index('--backbone-weights') + 1])
        if (file_sha256(screen) != benchmark['screening_sha256']
                or file_sha256(backbone) != benchmark['backbone_weights_sha256']):
            raise ValueError(f'Changed screening or DINO inputs: {protocol}')
        manifest, files = screening_files(manifest_path, screen, 0)
        if slices['protocols'][protocol]['manifest_sha256'] != manifest_digest:
            raise ValueError('Changed validation protocol')
        excluded = set(manifest['evaluation']['image_source_ids']) | source_ids(read_json(files['increment_train']))
        folds = {}
        for fold, entry in slices['protocols'][protocol]['slices'].items():
            path = SLICES / entry['file']
            payload = verified(path, entry['sha256'])
            names = source_ids(payload)
            if len(names) != entry['images'] or names & excluded:
                raise ValueError(f'Overlapping or incomplete Task 1 validation fold: {protocol}/{fold}')
            excluded.update(names)
            folds[fold] = dict(path=str(path), sha256=entry['sha256'])
        command[command.index('--output_dir') + 1] = str((OUTPUT / protocol / 'increment').resolve())
        command.append('--with_box_refine')
        jobs[protocol] = dict(command=command, control=str(checkpoint), control_sha256=control,
                              manifest_sha256=manifest_digest, screen_sha256=benchmark['screening_sha256'],
                              backbone_sha256=benchmark['backbone_weights_sha256'], folds=folds)
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=BUDGET_SHA, control_plan_sha256=comparison['plan_sha256'],
                slices_plan_sha256=SLICES_SHA, jobs=jobs, device_ids=gpu_ids, devices=gpus,
                eval_batch=original['batch_size'], max_detections=original['limit'],
                seed=original['seed'], epochs=3, gates=GATES, paper_comparable=False,
                factor='Standard Deformable DETR iterative box refinement with untied per-layer detection heads.')


def compare(control, candidate, criteria):
    small_control, small_candidate = (row['groups']['small'] for row in (control, candidate))
    count = small_control['gt']
    if small_candidate['gt'] != count or count <= 0:
        raise ValueError('Changed small unknown ground-truth denominator')
    base, current = (row['metrics'] for row in (control, candidate))
    deltas = dict(small_final=(small_candidate['final'] - small_control['final']) / count,
                  small_u_recall=(small_candidate['unknown_tp'] - small_control['unknown_tp']) / count,
                  all_u_recall=current['u_recall'] - base['u_recall'],
                  **{key: current[key] - base[key] for key in ('known_ap50', 'wi', 'a_ose')})
    gates = {key: deltas[key] >= criteria[key]
             for key in ('small_final', 'small_u_recall', 'all_u_recall', 'known_ap50')}
    gates['wi'] = not (control['wi_denominator'] and candidate['wi_denominator']) or deltas['wi'] <= criteria['wi']
    gates['a_ose'] = deltas['a_ose'] <= max(1, criteria['a_ose'] * base['a_ose'])
    return dict(deltas=deltas, gates=gates, passed=all(gates.values()))


def evaluate(checkpoint, protocol, job, plan, digest, arm):
    report = OUTPUT / f'{protocol}-{arm}.json'
    identity = dict(plan_sha256=digest, checkpoint_sha256=file_sha256(checkpoint))
    if not report.exists():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        if model.detr.with_box_refine != (arm == 'refinement'):
            raise ValueError(f'Wrong saved decoder configuration: {protocol}/{arm}')
        folds = {}
        for fold, entry in job['folds'].items():
            annotation = Path(entry['path'])
            if file_sha256(annotation) != entry['sha256']:
                raise ValueError(f'Changed validation annotations: {protocol}/{fold}')
            print(f'Evaluating {protocol}/{arm}/{fold}', flush=True)
            folds[fold] = measure(model, manifest, root, annotation, plan['eval_batch'], plan['max_detections'])
        write_json(report, dict(identity, folds=folds))
    saved = read_json(report)
    if any(saved[key] != value for key, value in identity.items()):
        raise ValueError(f'Changed paired evaluation: {protocol}/{arm}')
    return saved['folds']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    if args.plan_only:
        return
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        print(f'Training {protocol} Task 1 iterative refinement', flush=True)
        candidate, saved = run_job(job['command'], job['screen_sha256'])
        if not saved.with_box_refine or saved.backbone_weights_sha256 != job['backbone_sha256']:
            raise ValueError(f'Unexpected refinement training identity: {protocol}')
        checkpoints = dict(control=Path(job['control']), refinement=candidate)
        observations = {arm: evaluate(checkpoints[arm], protocol, job, plan, digest, arm) for arm in ARMS}
        results[protocol] = {fold: compare(observations['control'][fold], observations['refinement'][fold], plan['gates'])
                             for fold in job['folds']}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                               passed=all(row['passed'] for folds in results.values() for row in folds.values())))


if __name__ == '__main__':
    main()
