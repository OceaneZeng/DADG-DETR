"""Run paired localization or classifier diagnostics for the expanded M-OWODB Task 2 pilot."""
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
from tools.owod.run_schedule_pair import coverage, endpoint
from tools.owod.screening_data import screening_files, write_json

OUTPUT = ROOT / 'output/m-instance-localization-pair'
CLASSIFIER_OUTPUT = ROOT / 'output/m-instance-classifier-pair'
PHASES = ('increment', 'finetune')


def treatment_command(folder, arm, phase, parent, previous, gnn):
    cmd = pilot.command('off', phase, previous, gnn)
    cmd[cmd.index('--output_dir') + 1] = str(folder)
    if arm == 'bbox-last':
        cmd[cmd.index('--graph-localization') + 1] = 'bbox-last'
        # Pin the completed Task 1 checkpoint at the permitted diagnostic fork.
        cmd += ['--screen-branch-checkpoint', str(parent)]
    elif phase == 'increment':
        cmd += ['--class_embed_lr_mult', '2']
    return cmd


def verified_pilot_inputs():
    import torch

    source = pilot.OUTPUT
    old = read_json(source / 'plan.json')
    comparison = read_json(source / 'comparison.json')
    if comparison.get('status') != 'complete' or old['epochs'] != pilot.EPOCHS:
        raise ValueError('Expected the completed expanded pilot')
    if (torch.cuda.device_count() != 2 or [torch.cuda.get_device_name(i) for i in range(2)] != old['gpus']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != old['devices'] or str(torch.__version__) != old['torch_version']):
        raise ValueError('Use the same two GPUs and tree-detr environment as the completed pilot')
    _, files = screening_files(pilot.MANIFEST, pilot.SCREEN, 1)
    parent = source / 'matched/task1/checkpoint.pth'
    gnn = source / 'matched/gnn/gnn_stage0.pt'
    parent_hash = comparison['common_parent_sha256']
    cache = source / 'retrieval' / f'instances-stage-1-{parent_hash}.json'
    for path, expected in ((pilot.MANIFEST, old['manifest']), (pilot.SCREEN, old['screen']),
                           (pilot.WEIGHTS, old['weights']), (parent, parent_hash),
                           (gnn, comparison['common_gnn_sha256']), (cache, comparison['evidence_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'Original input changed: {path}')
    return comparison, files, parent, gnn


def endpoint_args(path, phase, command, comparison):
    import torch
    from tools.owod.validate_discovery import validate_plan_arguments
    saved = torch.load(path, map_location='meta', weights_only=False)
    if saved['epoch'] != pilot.EPOCHS[phase] - 1:
        raise ValueError(f'Unexpected endpoint: {path}')
    validate_plan_arguments({'command': command}, saved['args'])
    spec = saved['args'].graph_adapter_specs[-1]
    if (spec['source_checkpoint_sha256'] != comparison['common_parent_sha256']
            or spec['gnn_sha256'] != comparison['common_gnn_sha256']
            or spec['instances']['cache_sha256'] != comparison['evidence_sha256']):
        raise ValueError(f'Checkpoint did not use the shared pilot inputs: {path}')
    return saved['args']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier-lr', action='store_true',
                        help='Double the classifier LR only in Task 2 increment')
    args = parser.parse_args(argv)
    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    from util.misc import collate_fn

    source = pilot.OUTPUT
    comparison, files, parent, gnn = verified_pilot_inputs()

    output, arm = (CLASSIFIER_OUTPUT, 'classifier-lr2x') if args.classifier_lr else (OUTPUT, 'bbox-last')
    frozen = {phase: source / 'off' / phase / 'checkpoint.pth' for phase in PHASES}
    trainable = {phase: output / arm / phase / 'checkpoint.pth' for phase in PHASES}
    commands = {
        phase: treatment_command(trainable[phase].parent, arm, phase, parent, previous, gnn)
        for phase, previous in zip(PHASES, (parent, trainable['increment']))
    }
    rows = {'frozen': {}, arm: {}}

    for phase, previous in zip(PHASES, (parent, frozen['increment'])):
        saved_args = endpoint_args(frozen[phase], phase, pilot.command('off', phase, previous, gnn), comparison)
        control = endpoint(frozen[phase].parent, phase)
        if any(control[key] != value for key, value in comparison['rows']['off'][phase].items()):
            raise ValueError(f'Frozen metrics differ from the completed pilot: {phase}')
        rows['frozen'][phase] = control

    plan = output / 'plan.json'
    identity = dict(commands=commands, source_plan=file_sha256(source / 'plan.json'),
                    source_results=file_sha256(source / 'comparison.json'),
                    frozen_checkpoints={phase: file_sha256(path) for phase, path in frozen.items()},
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if plan.exists():
        if read_json(plan) != identity:
            raise ValueError(f'Changed paired comparison inputs: {plan}')
    else:
        if any(path.parent.exists() for path in trainable.values()):
            raise ValueError('Missing plan for existing paired run')
        write_json(plan, identity)

    for phase, path in trainable.items():
        cmd = commands[phase]
        if not (path.parent / 'training_complete.json').is_file():
            resume = ['--resume', str(path)] if path.exists() else []
            pilot.run(cmd + resume)
        endpoint_args(path, phase, cmd, comparison)
        rows[arm][phase] = endpoint(path.parent, phase)

    dataset = build_calibration_dataset(Path(saved_args.owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    observed = {
        name: {phase: coverage(path, loader, include_top300=True, include_classification=args.classifier_lr)
               for phase, path in checkpoints.items()}
        for name, checkpoints in [('frozen', frozen), (arm, trainable)]
    }
    deltas = {
        phase: {key: rows[arm][phase][key] - rows['frozen'][phase][key]
                for key in (*pilot.METRICS, 'unknown_tp')}
        for phase in PHASES
    }
    delta_name = f'{arm.replace("-", "_")}_minus_frozen'
    report = dict(rows=rows, coverage=observed,
                  coverage_images=len(dataset), author_max_detections=50,
                  coverage_output_budgets=[50, 300], status='complete')
    report[delta_name] = deltas
    summary = {delta_name: deltas}
    if args.classifier_lr:
        summary['classifier_lr_multiplier'] = {'increment': 2, 'finetune': 1}
        summary['coverage_overall'] = {
            name: {phase: {**view['overall'], **view['classification_at_best_iou']['overall']}
                   for phase, view in phases.items()}
            for name, phases in observed.items()
        }
        report['summary'] = summary
    path = output / 'comparison.json'
    write_json(path, report)
    print(json.dumps(summary if args.classifier_lr else deltas, indent=2), flush=True)
    print('Comparison saved:', path, flush=True)


if __name__ == '__main__':
    main()
