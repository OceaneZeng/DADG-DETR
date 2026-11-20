"""Repeat the 1.0x versus 0.75x decoder/bbox comparison with training seed 43."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_decoder_bbox_lr_pair import CONTROL, THREE_QUARTER, LR_MULTIPLIERS, treatment_commands
from tools.owod.screening_data import write_json

SEED = 43
OUTPUT = ROOT / 'output' / f'm-instance-decoder-bbox-seed{SEED}'
SCREEN = OUTPUT / 'data/screening.json'
REFERENCE = ROOT / 'output' / f'm-instance-{THREE_QUARTER}'


def seed_commands():
    parent, jobs = study.study_commands(output=OUTPUT, screen=SCREEN, seed=SEED, arms=(CONTROL,))
    jobs[THREE_QUARTER] = treatment_commands(jobs[CONTROL], OUTPUT / THREE_QUARTER, LR_MULTIPLIERS[THREE_QUARTER])
    return parent, jobs


def register_training_screen():
    # Keep the original sampling seed and every annotation/holdout identity.
    # Only the training seed and the location of this descriptor change.
    source = read_json(study.SCREEN)
    for stage in source['stages']:
        stage['files'] = {key: str(study.SCREEN.parent / path) for key, path in stage['files'].items()}
    source.update(training_seed=SEED, source_screen_sha256=file_sha256(study.SCREEN))
    pin(SCREEN, source)


def paired_ap_points(rows):
    result = {}
    for phase in study.PHASES:
        full = rows[CONTROL][phase]['metrics']
        reduced = rows[THREE_QUARTER][phase]['metrics']
        result[phase] = {
            'current_ap50_1x_pct': 100 * full['current_ap50'],
            'current_ap50_075x_pct': 100 * reduced['current_ap50'],
            'current_gap_1x_minus_075x_pp': 100 * (full['current_ap50'] - reduced['current_ap50']),
            'previous_ap50_1x_pct': 100 * full['previous_ap50'],
            'previous_ap50_075x_pct': 100 * reduced['previous_ap50'],
            'previous_gain_075x_minus_1x_pp': 100 * (reduced['previous_ap50'] - full['previous_ap50']),
        }
    return result


def run(parent, jobs):
    import torch

    old_plan = read_json(study.OUTPUT / 'plan.json')
    pair_plan = read_json(REFERENCE / 'plan.json')
    reference = read_json(REFERENCE / 'comparison.json')
    old_parent, old_jobs = study.study_commands(arms=(CONTROL,))
    old_reduced = treatment_commands(old_jobs[CONTROL], REFERENCE, LR_MULTIPLIERS[THREE_QUARTER])
    if (old_plan['seed'] != 42 or reference['seed'] != 42 or reference['status'] != 'complete'
            or reference['primary_endpoint'] != 'finetune'
            or reference['decoder_bbox_lr_mult'] != LR_MULTIPLIERS[THREE_QUARTER]
            or old_parent != old_plan['task1'] or old_jobs[CONTROL] != old_plan['arms'][CONTROL]
            or pair_plan['commands'] != old_reduced):
        raise ValueError('Expected the completed seed-42 reference pair')
    for path, digest in ((study.OUTPUT / 'plan.json', pair_plan['source_plan_sha256']),
                         (study.OUTPUT / 'comparison.json', pair_plan['source_comparison_sha256']),
                         (study.SCREEN, pair_plan['screen_sha256']),
                         (pilot.MANIFEST, old_plan['manifest_sha256']), (pilot.WEIGHTS, old_plan['weights_sha256'])):
        if file_sha256(path) != digest:
            raise ValueError(f'Seed-42 input changed: {path}')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if (gpus != old_plan['gpus'] or devices != old_plan['devices']
            or str(torch.__version__) != old_plan['torch_version']):
        raise ValueError('Use the original two GPUs and tree-detr environment')
    identity = dict(task1=parent, arms=jobs, seed=SEED, calibration_seed=SEED,
                    selection_seed=read_json(study.SCREEN)['seed'], source_screen_sha256=pair_plan['screen_sha256'],
                    manifest_sha256=old_plan['manifest_sha256'], weights_sha256=old_plan['weights_sha256'],
                    reference_plan_sha256=file_sha256(REFERENCE / 'plan.json'),
                    reference_comparison_sha256=file_sha256(REFERENCE / 'comparison.json'),
                    gpus=gpus, devices=devices, torch_version=str(torch.__version__),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    pin(OUTPUT / 'plan.json', identity)
    register_training_screen()
    study.check_data_and_cuda()
    report = study.train_shared_parent_arms(parent, jobs, OUTPUT, SCREEN, SEED)

    old, new = paired_ap_points(reference['rows']), paired_ap_points(report['rows'])
    a, b = old['finetune'], new['finetune']
    current = 'current_gap_1x_minus_075x_pp'
    previous = 'previous_gain_075x_minus_1x_pp'
    summary = dict(paired_ap_by_seed={'42': old, str(SEED): new},
                   current_gap_same_direction=a[current] * b[current] > 0,
                   current_gap_change_pp=b[current] - a[current],
                   previous_gain_same_direction=a[previous] * b[previous] > 0)
    report.update(status='complete', seed_comparison=summary)
    write_json(OUTPUT / 'comparison.json', report)
    write_json(OUTPUT / 'progress.json', report)
    print(json.dumps(summary, indent=2))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the five training jobs without running them')
    args = parser.parse_args()
    parent, jobs = seed_commands()
    if args.plan:
        print(json.dumps(dict(seed=SEED, calibration_seed=SEED, source_screen=str(study.SCREEN),
                              task1=parent, arms=jobs), indent=2))
    else:
        run(parent, jobs)
