"""Compare three Task 2 adaptation scopes from a more thoroughly trained v10 Task 1 parent."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod import run_instance_pilot as pilot
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import differences, endpoint_coverage, pin
from tools.owod.run_schedule_pair import endpoint
from tools.owod.screening_data import prepare_screen, screening_files, write_json

OUTPUT = ROOT / 'output/m-instance-stronger-parent'
SCREEN = OUTPUT / 'data/screening.json'
PARENT_IMAGES, PARENT_EPOCHS = 12288, 12
ARMS = ('frozen', 'cross-attn-last', 'decoder-bbox')
PHASES = ('increment', 'finetune')


def study_commands():
    task1 = pilot.command('off', 'task1', output=OUTPUT, screen=SCREEN, epochs=PARENT_EPOCHS)
    parent = OUTPUT / 'off/task1/checkpoint.pth'
    gnn = OUTPUT / 'off/gnn/gnn_stage0.pt'
    jobs = {}
    for arm in ARMS:
        jobs[arm] = {}
        for phase in PHASES:
            previous = parent if phase == 'increment' else OUTPUT / arm / 'increment/checkpoint.pth'
            cmd = pilot.command('off', phase, previous, gnn, output=OUTPUT, screen=SCREEN)
            cmd[cmd.index('--output_dir') + 1] = str(OUTPUT / arm / phase)
            cmd[cmd.index('--graph-localization') + 1] = arm
            cmd.extend(['--screen-branch-checkpoint', str(parent)])
            if phase == 'finetune':
                cmd.extend(['--discovery-pseudo-epoch-offset', str(pilot.EPOCHS['increment'])])
            jobs[arm][phase] = cmd
    return task1, jobs


def run_job(command, screen_hash):
    import torch
    from tools.owod.validate_discovery import validate_plan_arguments

    folder = Path(command[command.index('--output_dir') + 1])
    checkpoint = folder / 'checkpoint.pth'
    marker = folder / 'training_complete.json'
    epochs = int(command[command.index('--epochs') + 1])
    inputs = dict(command=command, screen_sha256=screen_hash)
    for flag in ('--pretrained', '--graph-gnn-checkpoint'):
        if flag in command:
            inputs[flag] = file_sha256(Path(command[command.index(flag) + 1]))
    pin(folder / 'input.json', inputs)
    if not marker.exists():
        pilot.run(command + (['--resume', str(checkpoint)] if checkpoint.exists() else []))
    saved = torch.load(checkpoint, map_location='meta', weights_only=False)
    if saved['epoch'] != epochs - 1 or read_json(marker)['last_epoch'] != epochs - 1:
        raise ValueError(f'Unexpected training endpoint: {checkpoint}')
    validate_plan_arguments({'command': command}, saved['args'])
    return checkpoint, saved['args']


def study_endpoint(path, args, phase, loader, parent_hash, gnn_hash, cache_hash):
    spec = args.graph_adapter_specs[-1]
    if (spec['source_checkpoint_sha256'] != parent_hash
            or spec['gnn_sha256'] != gnn_hash
            or spec['instances']['cache_sha256'] != cache_hash
            or spec['alignment_summary']['applied'] != 0):
        raise ValueError(f'Different parent/GNN/cache or nonzero alignment: {path}')
    metrics = endpoint(path.parent, phase)
    if phase == 'finetune' and metrics['discovery_pseudo_scale'] != 1.:
        raise ValueError(f'Finetuning did not continue the pseudo schedule: {path}')
    last = json.loads((path.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    return dict(metrics=metrics, current_classes=last['test_owod_current_class_stats'],
                trainable_parameters=last['n_parameters'],
                peak_allocated_mib_rank0=last['train_peak_allocated_mib'],
                heldout=endpoint_coverage(path, loader, path.parent / 'coverage.json'))


def run_study(task1, jobs):
    import torch
    from torch.utils.data import DataLoader
    from models.graph_local.instance_transfer import bank_settings
    from tools.owod.gnn_calibration import build_calibration_dataset
    from tools.owod.instance_coverage import task1_coverage
    from util.misc import collate_fn

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpus) != 2 or any('3090' not in device for device in gpus):
        raise RuntimeError('Select two RTX 3090 devices with CUDA_VISIBLE_DEVICES')
    pin(OUTPUT / 'plan.json', dict(
        task1=task1, arms=jobs, commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        manifest_sha256=file_sha256(pilot.MANIFEST), weights_sha256=file_sha256(pilot.WEIGHTS),
        task1_images=PARENT_IMAGES, task2_images=pilot.TASK2_IMAGES, heldout_images=128,
        seed=42, instance_settings=bank_settings(expanded=True),
        devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpus=gpus, torch_version=str(torch.__version__)))
    pilot.run([sys.executable, str(ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(ROOT / 'data/owod')])
    pilot.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
               str(ROOT / 'tools/owod/server_cuda_check.py'), '--instances', '--decoder-bbox'])

    screen = prepare_screen(pilot.MANIFEST, SCREEN.parent, PARENT_IMAGES, 128, 42,
                            task2_images=pilot.TASK2_IMAGES)
    screen_hash = file_sha256(screen)
    _, files = screening_files(pilot.MANIFEST, screen, 1)
    parent, parent_args = run_job(task1, screen_hash)
    parent_hash = file_sha256(parent)
    parent_report = parent.parent / 'candidate_coverage.json'
    if not parent_report.exists():
        write_json(parent_report, dict(checkpoint_sha256=parent_hash,
                                       coverage=task1_coverage(parent, pilot.MANIFEST, screen)))
    observed = read_json(parent_report)
    if observed['checkpoint_sha256'] != parent_hash:
        raise ValueError('Task 1 coverage belongs to a different checkpoint')
    rows = {}
    report = dict(status='partial', paper_comparable=False, seed=42, primary_endpoint='finetune',
                  units='AP50, recall and precision are fractions',
                  task1=dict(checkpoint_sha256=parent_hash, coverage=observed['coverage'],
                             metrics=pilot.metrics(parent.parent, PARENT_EPOCHS - 1)), rows=rows)
    write_json(OUTPUT / 'progress.json', report)

    gnn = pilot.calibrate('off', parent, output=OUTPUT, screen=screen)
    gnn_hash = file_sha256(gnn)
    report['common_gnn_sha256'] = gnn_hash
    dataset = build_calibration_dataset(Path(parent_args.owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    cache = OUTPUT / 'retrieval' / f'instances-stage-1-{parent_hash}.json'
    for arm, phases in jobs.items():
        rows[arm] = {}
        for phase, cmd in phases.items():
            path, args = run_job(cmd, screen_hash)
            rows[arm][phase] = study_endpoint(path, args, phase, loader, parent_hash, gnn_hash, file_sha256(cache))
            write_json(OUTPUT / 'progress.json', report)
    deltas = {
        arm: {phase: differences(rows[arm][phase]['metrics'], rows['frozen'][phase]['metrics'])
              for phase in PHASES}
        for arm in ARMS[1:]
    }
    screen_passed = {arm: v['finetune']['current_ap50'] >= .01 and v['finetune']['previous_ap50'] >= -.01
                     for arm, v in deltas.items()}
    report.update(status='complete', deltas_vs_frozen=deltas, capacity_screen_passed=screen_passed)
    write_json(OUTPUT / 'comparison.json', report)
    write_json(OUTPUT / 'progress.json', report)
    print(json.dumps(dict(deltas_vs_frozen=deltas, capacity_screen_passed=screen_passed), indent=2))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the planned jobs without training')
    args = parser.parse_args()
    task1, jobs = study_commands()
    if args.plan:
        print(json.dumps(dict(task1_images=PARENT_IMAGES, task1=task1, arms=jobs), indent=2))
    else:
        run_study(task1, jobs)
