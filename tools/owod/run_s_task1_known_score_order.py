"""Train the fixed S midpoint recipe with matched released-class score ordering."""

import os
import subprocess
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_half_rank_quota import MIDPOINT_SHA, RETRY
from tools.owod.probe_s_task1_rank_independent import PROTOCOL, evaluate as evaluate_released
from tools.owod.probe_s_task1_same_query_competition import ANCHOR_OUTPUT, METRICS
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.screening_data import screening_files, write_json


OUTPUT = RETRY.with_name('task1-s-known-score-order')
ANCHOR_SHA = '6c3122eb33f76c106f7031b312f1299a1ee0a2496fb4c0e4b5d03a60a1d56505'
RANK_WEIGHT = 1.
MATCHED_RANK_FLAG = '--discovery-known-rank-coef'


def prepare(output, rank_flag):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if gpus != ['NVIDIA GeForce RTX 3090'] * 2 or len(set(devices)) != 2 or not all(d.isdecimal() for d in devices):
        raise ValueError('Use two pinned RTX 3090 GPUs for the fresh score-order run')
    middle = verified(RETRY / 'comparison.json', MIDPOINT_SHA)
    source = verified(RETRY / 'plan.json', middle['plan_sha256'])
    anchor = verified(ANCHOR_OUTPUT / 'comparison.json', ANCHOR_SHA)
    locked = verified(ANCHOR_OUTPUT / 'plan.json', anchor['plan_sha256'])
    if (any(row['status'] != 'complete' or row['passed'] for row in (middle, anchor))
            or locked['arms']['half']['sha256'] != middle['checkpoint_sha256']
            or locked['manifest_sha256'] != source['manifest_sha256']):
        raise ValueError('Expected the completed, below-gate controls for the same midpoint model')
    command = source['command'].copy()
    if any(item in command for item in (MATCHED_RANK_FLAG, rank_flag, '--resume', '--pretrained')):
        raise ValueError('Use only the fresh midpoint recipe without prior score ordering')
    screen = Path(command[command.index('--owod-screening') + 1])
    backbone = Path(command[command.index('--backbone-weights') + 1])
    if file_sha256(screen) != source['screen_sha256'] or file_sha256(backbone) != source['backbone_sha256']:
        raise ValueError('Changed fresh training inputs')
    manifest_path = MANIFESTS[PROTOCOL]
    manifest = verified(manifest_path, source['manifest_sha256'])
    screening_files(manifest_path, screen, 0)
    verified(Path(source['holdout']), source['holdout_sha256'])
    for entry in locked['folds'].values():
        verified(Path(entry['path']), entry['sha256'])
    command[command.index('--output_dir') + 1] = str(output / PROTOCOL / 'increment')
    command.extend([rank_flag, str(RANK_WEIGHT)])
    plan = {key: source[key] for key in ('manifest_sha256', 'screen_sha256', 'backbone_sha256', 'holdout',
                                          'holdout_sha256', 'epoch_count', 'seed', 'batch_size', 'limit',
                                          'rank_feature_grad_scale', 'min_known_ap50')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, devices=devices, gpu_names=gpus,
                command=command, midpoint_sha256=MIDPOINT_SHA, anchor_sha256=ANCHOR_SHA,
                min_small_coverage=middle['small']['covered'], released_small_count=middle['small']['released'],
                folds=locked['folds'], max_wi_increase=locked['max_wi_increase'], max_aose_fraction=locked['max_aose_fraction'],
                fold_native={fold: row['views']['native'] for fold, row in anchor['outcomes']['half'].items()})
    return plan, manifest


def evaluate_endpoint(output, checkpoint, manifest, plan, digest):
    checkpoint_sha = file_sha256(checkpoint)
    model, _ = load_detector(None, checkpoint, torch.device('cuda'))
    binding = dict(plan_sha256=digest, checkpoint_sha256=checkpoint_sha)
    released_path = output / 'released.json'
    if not released_path.is_file():
        print('Evaluating the final checkpoint on the fixed released-only S holdout', flush=True)
        report = evaluate_released(model, manifest, Path(plan['holdout']), plan)
        write_json(released_path, dict(**binding, report=report))
    released = verified_endpoint(released_path, binding)
    ap = released['emitted']['metrics']['known_ap50']
    small = released['released_small_learned_query_coverage']
    if small['released'] != plan['released_small_count']:
        raise ValueError('Expected the independent released-small denominator')
    released_passed = (ap >= plan['min_known_ap50'] and small['covered'] >= plan['min_small_coverage']
                       and (ap > plan['min_known_ap50'] or small['covered'] > plan['min_small_coverage']))
    known = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    folds = {}
    for name, entry in plan['folds'].items():
        dest = output / f'{name}.json'
        if not dest.is_file():
            print(f'Evaluating the final checkpoint on S {name} fold', flush=True)
            report = measure(model, manifest, root, Path(entry['path']), plan['batch_size'], plan['limit'], postprocessor=post)
            write_json(dest, dict(**binding, report=report))
        report = verified_endpoint(dest, binding)
        native = plan['fold_native'][name]
        metrics = {key: report['metrics'][key] for key in METRICS}
        delta = {key: metrics[key] - native[key] for key in METRICS}
        gates = dict(known_ap50=delta['known_ap50'] >= 0., u_recall=delta['u_recall'] >= 0.,
                     wi=delta['wi'] <= plan['max_wi_increase'],
                     a_ose=delta['a_ose'] <= max(1., plan['max_aose_fraction'] * native['a_ose']))
        folds[name] = dict(metrics=metrics, delta=delta, gates=gates, passed=all(gates.values()))
    write_json(output / 'comparison.json', dict(status='complete', **binding, known_ap50=ap, small=small,
                                               released_passed=released_passed, folds=folds,
                                               passed=released_passed and all(row['passed'] for row in folds.values())))


def verified_endpoint(path, binding):
    record = read_json(path)
    if any(record[key] != value for key, value in binding.items()):
        raise ValueError(f'Changed evaluation input for {path}')
    return record['report']


def main():
    plan, manifest = prepare(OUTPUT, MATCHED_RANK_FLAG)
    plan['matched_known_rank_coef'] = RANK_WEIGHT
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    print('Fitting S Task 1 with matched released-class score ordering', flush=True)
    checkpoint, saved = run_job(plan['command'], plan['screen_sha256'])
    if saved.discovery_known_rank_coef != RANK_WEIGHT or saved.backbone_weights_sha256 != plan['backbone_sha256']:
        raise ValueError('Unexpected score-order endpoint')
    evaluate_endpoint(OUTPUT, checkpoint, manifest, plan, digest)


if __name__ == '__main__':
    main()
