"""Compare frozen and trainable box projection in the expanded M-OWODB Task 2 pilot."""
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
PHASES = ('increment', 'finetune')


def localization_command(phase, parent, previous, gnn):
    folder = OUTPUT / 'bbox-last' / phase
    cmd = pilot.command('off', phase, previous, gnn)
    cmd[cmd.index('--output_dir') + 1] = str(folder)
    cmd[cmd.index('--graph-localization') + 1] = 'bbox-last'
    # Pin the completed Task 1 checkpoint at the permitted diagnostic fork.
    cmd += ['--screen-branch-checkpoint', str(parent)]
    return cmd


def main():
    import torch
    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    from tools.owod.validate_discovery import validate_plan_arguments
    from util.misc import collate_fn

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

    frozen = {phase: source / 'off' / phase / 'checkpoint.pth' for phase in PHASES}
    trainable = {phase: OUTPUT / 'bbox-last' / phase / 'checkpoint.pth' for phase in PHASES}
    commands = {
        'increment': localization_command('increment', parent, parent, gnn),
        'finetune': localization_command('finetune', parent, trainable['increment'], gnn),
    }
    rows = {'frozen': {}, 'bbox-last': {}}

    def read_endpoint(path, phase, command):
        saved = torch.load(path, map_location='meta', weights_only=False)
        if saved['epoch'] != pilot.EPOCHS[phase] - 1:
            raise ValueError(f'Unexpected endpoint: {path}')
        validate_plan_arguments({'command': command}, saved['args'])
        spec = saved['args'].graph_adapter_specs[-1]
        if (spec['source_checkpoint_sha256'] != parent_hash
                or spec['gnn_sha256'] != comparison['common_gnn_sha256']
                or spec['instances']['cache_sha256'] != comparison['evidence_sha256']):
            raise ValueError(f'Checkpoint did not use the shared pilot inputs: {path}')
        return endpoint(path.parent, phase), saved['args']

    for phase, previous in zip(PHASES, (parent, frozen['increment'])):
        control, saved_args = read_endpoint(frozen[phase], phase, pilot.command('off', phase, previous, gnn))
        if any(control[key] != value for key, value in comparison['rows']['off'][phase].items()):
            raise ValueError(f'Frozen metrics differ from the completed pilot: {phase}')
        rows['frozen'][phase] = control

    plan = OUTPUT / 'plan.json'
    identity = dict(commands=commands, source_plan=file_sha256(source / 'plan.json'),
                    source_results=file_sha256(source / 'comparison.json'),
                    frozen_checkpoints={phase: file_sha256(path) for phase, path in frozen.items()},
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if plan.exists():
        if read_json(plan) != identity:
            raise ValueError(f'Changed localization comparison inputs: {plan}')
    else:
        if any(path.parent.exists() for path in trainable.values()):
            raise ValueError('Missing plan for existing localization run')
        write_json(plan, identity)

    for phase, path in trainable.items():
        cmd = commands[phase]
        if not (path.parent / 'training_complete.json').is_file():
            resume = ['--resume', str(path)] if path.exists() else []
            pilot.run(cmd + resume)
        rows['bbox-last'][phase], _ = read_endpoint(path, phase, cmd)

    dataset = build_calibration_dataset(Path(saved_args.owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    observed = {
        arm: {phase: coverage(path, loader, include_top300=True) for phase, path in checkpoints.items()}
        for arm, checkpoints in [('frozen', frozen), ('bbox-last', trainable)]
    }
    deltas = {
        phase: {key: rows['bbox-last'][phase][key] - rows['frozen'][phase][key]
                for key in (*pilot.METRICS, 'unknown_tp')}
        for phase in PHASES
    }
    report = dict(rows=rows, bbox_last_minus_frozen=deltas, coverage=observed,
                  coverage_images=len(dataset), author_max_detections=50,
                  coverage_output_budgets=[50, 300], status='complete')
    path = OUTPUT / 'comparison.json'
    write_json(path, report)
    print(json.dumps(deltas, indent=2), flush=True)
    print('Comparison saved:', path, flush=True)


if __name__ == '__main__':
    main()
