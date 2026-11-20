"""Run the lower-LR Task 2 arm against the completed stronger-parent study."""
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
from tools.owod.run_cross_attention_study import differences, pin
from tools.owod.run_schedule_pair import endpoint
from tools.owod.screening_data import screening_files, write_json

OUTPUT = ROOT / 'output/m-instance-decoder-bbox-lr-half'
LR_MULTIPLIER = 0.5
CONTROL = 'decoder-bbox'
ARM = 'decoder-bbox-lr-half'


def treatment_commands(reference):
    commands = {phase: list(reference[phase]) for phase in study.PHASES}
    for phase, command in commands.items():
        command[command.index('--output_dir') + 1] = str(OUTPUT / phase)
        command.extend(['--decoder-bbox-lr-mult', str(LR_MULTIPLIER)])
    finetune = commands['finetune']
    finetune[finetune.index('--pretrained') + 1] = str(OUTPUT / 'increment/checkpoint.pth')
    return commands


def run_pair(task1, arms):
    import torch
    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    from tools.owod.validate_discovery import load_checkpoint_metadata, validate_plan_arguments
    from util.misc import collate_fn

    source = study.OUTPUT
    plan_path, results_path = source / 'plan.json', source / 'comparison.json'
    original, results = read_json(plan_path), read_json(results_path)
    if (results['status'] != 'complete' or original['task1'] != task1 or original['arms'] != arms
            or original['seed'] != results['seed']):
        raise ValueError('Expected the completed stronger-parent study with its original commands')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (gpus != original['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != original['devices']
            or str(torch.__version__) != original['torch_version']):
        raise ValueError('Use the original two GPUs and tree-detr environment')

    parent = source / 'off/task1/checkpoint.pth'
    gnn = source / 'off/gnn/gnn_stage0.pt'
    parent_hash, gnn_hash = results['task1']['checkpoint_sha256'], results['common_gnn_sha256']
    parent_args = load_checkpoint_metadata(parent)['args']
    screen_hash = parent_args.owod_screening_sha256
    for path, digest in ((pilot.MANIFEST, original['manifest_sha256']), (pilot.WEIGHTS, original['weights_sha256']),
                         (study.SCREEN, screen_hash), (parent, parent_hash), (gnn, gnn_hash)):
        if file_sha256(path) != digest:
            raise ValueError(f'Original study input changed: {path}')
    pilot.run([sys.executable, str(ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(ROOT / 'data/owod')])
    _, files = screening_files(pilot.MANIFEST, study.SCREEN, 1)
    cache = source / 'retrieval' / f'instances-stage-1-{parent_hash}.json'
    cache_hash = file_sha256(cache)
    rows = {name: results['rows'][name] for name in ('frozen', CONTROL)}
    for name, phases in rows.items():
        for phase, row in phases.items():
            folder = source / name / phase
            if endpoint(folder, phase) != row['metrics']:
                raise ValueError(f'Original endpoint differs from the completed study: {folder}')
    control_increment = load_checkpoint_metadata(source / CONTROL / 'increment/checkpoint.pth')['args']
    validate_plan_arguments({'command': arms[CONTROL]['increment']}, control_increment)
    if control_increment.graph_adapter_specs[-1]['instances']['cache_sha256'] != cache_hash:
        raise ValueError('Original instance cache changed')

    commands = treatment_commands(original['arms'][CONTROL])
    pin(OUTPUT / 'plan.json', dict(commands=commands, commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_plan_sha256=file_sha256(plan_path), source_comparison_sha256=file_sha256(results_path),
        parent_sha256=parent_hash, gnn_sha256=gnn_hash, screen_sha256=screen_hash, cache_sha256=cache_hash))
    dataset = build_calibration_dataset(Path(parent_args.owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    rows[ARM] = {}
    report = dict(status='partial', rows=rows, seed=results['seed'], paper_comparable=False,
                  primary_endpoint='finetune', decoder_bbox_lr_mult=LR_MULTIPLIER, units=results['units'])
    write_json(OUTPUT / 'progress.json', report)
    for phase, cmd in commands.items():
        path, args = study.run_job(cmd, screen_hash)
        row = study.study_endpoint(path, args, phase, loader, parent_hash, gnn_hash, cache_hash)
        if row['trainable_parameters'] != rows[CONTROL][phase]['trainable_parameters']:
            raise ValueError(f'Trainable parameter count changed: {path}')
        rows[ARM][phase] = row
        write_json(OUTPUT / 'progress.json', report)

    deltas = {name: {phase: differences(rows[ARM][phase]['metrics'], baseline[phase]['metrics'])
                     for phase in study.PHASES}
              for name, baseline in rows.items() if name != ARM}
    final = deltas['frozen']['finetune']
    passed = final['current_ap50'] >= .01 and final['previous_ap50'] >= -.01
    report.update(status='complete', deltas=deltas, capacity_screen_passed=passed)
    write_json(OUTPUT / 'comparison.json', report)
    write_json(OUTPUT / 'progress.json', report)
    print(json.dumps(dict(deltas=deltas, capacity_screen_passed=passed), indent=2))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the two jobs without training')
    options = parser.parse_args()
    task1, arms = study.study_commands()
    if options.plan:
        print(json.dumps(treatment_commands(arms[CONTROL]), indent=2))
    else:
        run_pair(task1, arms)
