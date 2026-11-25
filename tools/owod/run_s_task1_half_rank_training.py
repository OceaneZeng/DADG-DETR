"""Train the S Task 1 midpoint between full and stopped ranking gradients."""

import argparse
import os
import subprocess
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_rank_independent import OUTPUT as INDEPENDENT, PROTOCOL, evaluate
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import QUALITY_FLAG, QUALITY_OUTPUT, QUALITY_SHA
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.screening_data import screening_files, write_json


OUTPUT = INDEPENDENT.with_name('task1-s-half-rank-gradient')
INDEPENDENT_SHA = '66a3df5d4fe4fc4f521a59c8b4dc9f0924c6701e97d94afeded9e85835eb3eeb'
SCALE = 0.5


def prepare(output):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if gpu_names != ['NVIDIA GeForce RTX 3090'] * 2 or len(set(devices)) != 2 or not all(d.isdecimal() for d in devices):
        raise ValueError('Expected two pinned RTX 3090 GPUs for the fresh S run')
    comparison = verified(INDEPENDENT / 'comparison.json', INDEPENDENT_SHA)
    reference = verified(INDEPENDENT / 'plan.json', comparison['plan_sha256'])
    annotations = INDEPENDENT / 'released-holdout.json'
    if comparison['status'] != 'complete' or file_sha256(annotations) != reference['holdout_sha256']:
        raise ValueError('Expected the completed independent replay and its pinned holdout')
    original = verified(QUALITY_OUTPUT / 'comparison.json', QUALITY_SHA)
    source = verified(QUALITY_OUTPUT / 'plan.json', original['plan_sha256'])
    control = source['jobs'][PROTOCOL]
    command = control['command'].copy()
    if QUALITY_FLAG not in command or '--pretrained' in command or '--resume' in command:
        raise ValueError('Expected a fresh quality-rank training command')
    manifest_path = MANIFESTS[PROTOCOL]
    screen = Path(command[command.index('--owod-screening') + 1])
    backbone = Path(command[command.index('--backbone-weights') + 1])
    if (file_sha256(screen) != control['screen_sha256'] or file_sha256(backbone) != control['backbone_sha256']
            or control['screen_sha256'] != reference['trained_screen_sha256']
            or control['manifest_sha256'] != reference['manifest_sha256']):
        raise ValueError('Changed fresh training inputs')
    manifest = verified(manifest_path, reference['manifest_sha256'])
    screening_files(manifest_path, screen, 0)
    command[command.index('--output_dir') + 1] = str(output / PROTOCOL / 'increment')
    command.extend(['--screen-rank-feature-grad-scale', str(SCALE)])
    best_ap = max(arm['known_ap50'] for arm in comparison['arms'].values())
    best_small = max(arm['small']['covered'] for arm in comparison['arms'].values())
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, independent_comparison_sha256=INDEPENDENT_SHA,
                quality_comparison_sha256=QUALITY_SHA, screen_sha256=control['screen_sha256'],
                backbone_sha256=control['backbone_sha256'], manifest_sha256=reference['manifest_sha256'],
                holdout_sha256=reference['holdout_sha256'], holdout=str(annotations),
                rank_feature_grad_scale=SCALE, epoch_count=source['epochs'], seed=source['seed'],
                devices=devices, gpu_names=gpu_names, batch_size=reference['batch_size'], limit=reference['limit'],
                command=command, reference=comparison['arms'], min_known_ap50=best_ap, min_small_coverage=best_small)
    return plan, manifest, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT,
                        help='Use a new directory for a fresh attempt with different hardware')
    output = parser.parse_args().output_dir.resolve()
    plan, manifest, annotations = prepare(output)
    plan_path = output / 'plan.json'
    pin(plan_path, plan)
    plan_sha = file_sha256(plan_path)
    print(f'Fitting S Task 1 with auxiliary rank feature gradient scale {SCALE}', flush=True)
    checkpoint, saved = run_job(plan['command'], plan['screen_sha256'])
    if saved.backbone_weights_sha256 != plan['backbone_sha256'] or saved.screen_rank_feature_grad_scale != SCALE:
        raise ValueError('Unexpected final midpoint checkpoint')
    checkpoint_sha = file_sha256(checkpoint)
    endpoint = output / 'independent.json'
    if not endpoint.is_file():
        print('Evaluating S Task 1 midpoint on the pinned independent holdout', flush=True)
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        report = evaluate(model, manifest, annotations, plan)
        write_json(endpoint, dict(plan_sha256=plan_sha, checkpoint_sha256=checkpoint_sha, report=report))
    evidence = read_json(endpoint)
    if evidence['plan_sha256'] != plan_sha or evidence['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Changed midpoint plan or final evaluation checkpoint')
    report = evidence['report']
    ap = report['emitted']['metrics']['known_ap50']
    coverage = report['released_small_learned_query_coverage']
    if any(arm['small']['released'] != coverage['released'] for arm in plan['reference'].values()):
        raise ValueError('Midpoint coverage denominator differs from the independent controls')
    passed = (ap >= plan['min_known_ap50'] and coverage['covered'] >= plan['min_small_coverage']
              and (ap > plan['min_known_ap50'] or coverage['covered'] > plan['min_small_coverage']))
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=plan_sha, passed=passed,
                                               checkpoint_sha256=checkpoint_sha, known_ap50=ap, small=coverage))


if __name__ == '__main__':
    main()
