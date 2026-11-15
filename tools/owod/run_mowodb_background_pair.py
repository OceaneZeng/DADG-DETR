#!/usr/bin/env python
"""Paired full-data M Task 1 diagnostic: focal v8 vs trusted-background v9."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod import run_benchmark
from tools.owod.paired_initialization import atomic_json, runtime_sha256

EPOCHS = 8
DEFAULT_OUTPUT = 'output/mowodb-task1-background-8epochs-2gpu'
METRICS = ('current_ap50', 'known_ap50', 'u_recall', 'unknown_precision',
           'unknown_predictions', 'unknown_gt', 'wi', 'a_ose', 'h_score')


def gpu_guard():
    """Use physical nvidia-smi indices 0/1 and then pin their UUIDs in CUDA."""
    def query(options):
        return subprocess.check_output(['nvidia-smi', *options, '--format=csv,noheader,nounits'],
                                       text=True, timeout=20)
    devices = list(csv.reader(io.StringIO(query(['--query-gpu=index,uuid,name']))))
    selected = {int(row[0].strip()): [item.strip() for item in row[1:]]
                for row in devices if int(row[0].strip()) in (0, 1)}
    if set(selected) != {0, 1} or any('3090' not in row[1] for row in selected.values()):
        raise ValueError('Expected RTX 3090 at physical GPU indices 0 and 1')
    busy = query(['--query-compute-apps=gpu_uuid,pid']).strip()
    uuids = [selected[index][0] for index in (0, 1)]
    for row in csv.reader(io.StringIO(busy)):
        if row and row[0].strip() in uuids:
            raise ValueError(f'GPU 0/1 occupied by a compute process: {row}; wait for the existing job')
    print('Selected physical GPUs 0,1:', selected, flush=True)
    return uuids


def benchmark_arguments(output, variant, manifest, weights, gpus='0,1'):
    command = ['--manifest', str(manifest), '--recipe', 'ow-detr-paper',
        '--method', 'discovery', '--task1-only', '--backbone-weights', str(weights),
        '--output-dir', str(output / variant), '--gpus', gpus,
        '--nproc-per-node', '2', '--batch-size', '2', '--num-workers', '4', '--seed', '42',
        '--activation-checkpointing', '--discovery-lr-scale', '0.25',
        '--graph-routing', 'hard', '--graph-localization', 'frozen',
        '--discovery-scoring', 'focal', '--discovery-pseudo-ramp-epochs', '3',
        '--discovery-pseudo-normalization', 'count', '--discovery-known-rank-coef', '1.0',
        '--discovery-feature-grad-scale', '1.0', '--pause-after-epochs', str(EPOCHS),
        '--initialization-audit', str(output / 'initialization.json')]
    if variant == 'background':
        command.append('--discovery-background-negative')
    elif variant != 'reference':
        raise ValueError('Unknown paired variant')
    return command


def paired_plans(output, manifest, weights):
    plans = {name: run_benchmark.build_plan(run_benchmark.get_args_parser().parse_args(
        benchmark_arguments(output, name, manifest, weights))) for name in ('reference', 'background')}
    def common(plan):
        command = list(plan['runs'][0]['command'])
        index = command.index('--output_dir')
        del command[index:index + 2]
        if '--discovery-background-negative' in command:
            command.remove('--discovery-background-negative')
        return command
    if (any(len(plan['runs']) != 1 or plan['protocol'] != 'm-owodb' for plan in plans.values())
            or common(plans['reference']) != common(plans['background'])):
        raise ValueError('The pair must differ only in background novelty supervision and output paths')
    for field in ('manifest_sha256', 'backbone_weights_sha256', 'recipe'):
        if plans['reference'][field] != plans['background'][field]:
            raise ValueError(f'Paired input changed while building plans: {field}')
    return plans


def read_result(plan):
    from tools.owod.strict_setting import validate_checkpoint
    from tools.owod.validate_discovery import load_checkpoint_metadata, validate_plan_arguments
    run = plan['runs'][0]
    directory = Path(run['output_dir'])
    status = json.loads((directory / 'training_paused.json').read_text(encoding='utf-8'))
    if status.get('last_epoch') != EPOCHS - 1 or status.get('epochs') != run['epochs']:
        raise ValueError('Require the fixed eighth-epoch pause, not a completed/other budget')
    checkpoint = load_checkpoint_metadata(directory / 'checkpoint.pth')
    if {'model', 'optimizer', 'lr_scheduler', 'discovery_ema', 'args', 'epoch'} - checkpoint.keys():
        raise ValueError('Incomplete paired training checkpoint')
    if checkpoint.get('epoch') != EPOCHS - 1:
        raise ValueError('Checkpoint is not at the fixed eighth epoch')
    saved = checkpoint['args']
    if getattr(saved, 'pause_after_epochs', 0) != EPOCHS:
        raise ValueError('Checkpoint did not use the paired execution budget')
    if (saved.owod_manifest_sha256 != plan['manifest_sha256']
            or saved.backbone_weights_sha256 != plan['backbone_weights_sha256']
            or saved.epochs != run['epochs'] or saved.owod_stage != run['stage']
            or saved.owod_phase != run['phase']):
        raise ValueError('Checkpoint provenance differs from paired plan')
    validate_plan_arguments(run, saved)
    validate_checkpoint(saved, checkpoint, resume=True)
    initial = json.loads((directory / 'initialization.json').read_text(encoding='utf-8'))
    shared = json.loads(Path(saved.initialization_audit).read_text(encoding='utf-8'))
    if initial != shared or initial['model_sha256'] != saved.initialization_sha256:
        raise ValueError('Initialization proof differs from checkpoint')
    if initial['source_sha256'] != runtime_sha256():
        raise ValueError('Runtime source changed since initialization')
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    fixed = [row for row in rows if row['epoch'] == EPOCHS - 1]
    if len(fixed) != 1:
        raise ValueError('Require exactly one fixed-epoch evaluation')
    row = fixed[0]
    metrics = {}
    for metric in METRICS:
        key = 'test_owod_' + metric
        if key not in row or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
            raise ValueError(f'Missing/nonfinite required metric: {key}')
        metrics[metric] = row[key]
    resume_invocations = sum(bool(json.loads(path.read_text(encoding='utf-8')).get('resume'))
                             for path in (directory / 'run_history').glob('*.json'))
    return dict(metrics=metrics, initialization=initial, n_parameters=row['n_parameters'],
        discovery_version=saved.discovery_version, output_dir=str(directory),
        resume_invocations=resume_invocations,
        training_diagnostics={key: value for key, value in row.items()
                              if key.startswith('train_discovery_')})


def comparison(plans):
    before, after = (read_result(plans[name]) for name in ('reference', 'background'))
    if before['initialization'] != after['initialization'] or before['n_parameters'] != after['n_parameters']:
        raise ValueError('Model initialization or parameter count differs')
    if (before['discovery_version'], after['discovery_version']) != (8, 9):
        raise ValueError('Expected v8 reference and v9 background treatment')
    metrics = {name: dict(reference=before['metrics'][name], background=after['metrics'][name],
        delta=after['metrics'][name] - before['metrics'][name]) for name in METRICS}
    print('Fixed epoch 8; AP/recall/precision in %, WI and counts in native units:')
    for name, values in metrics.items():
        scale = 100 if name.endswith('ap50') or name in ('u_recall', 'unknown_precision', 'h_score') else 1
        print(f"{name:24s} {values['reference'] * scale:12.4f} -> "
              f"{values['background'] * scale:12.4f}  delta={values['delta'] * scale:+.4f}")
    return dict(status='measured_single_seed_diagnostic', protocol='m-owodb', task=1,
        completed_epochs=EPOCHS, registered_epochs=50, previous_ap50=None, metrics=metrics,
        reference=before, background=after,
        metric_scale='AP/recall/precision are fractions in JSON; console multiplies them by 100.',
        limitation='Early Task 1 screening; no claim about final 50-epoch or four-task performance. '
                   'Interrupted runs restore existing optimizer/EMA checkpoints; legacy rank RNG is not restored.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--manifest', type=Path, default=Path('data/owod/m-owodb/split_manifest.json'))
    parser.add_argument('--weights', type=Path, default=Path('pretrained/dino_resnet50.pth'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--gpu-check', action='store_true')
    args = parser.parse_args(argv)
    if args.gpu_check:
        gpu_guard()
        return 0
    output, manifest, weights = (path.resolve() for path in (args.output, args.manifest, args.weights))
    plans = paired_plans(output, manifest, weights)
    identity = dict(version=1, completed_epochs=EPOCHS, source_sha256=runtime_sha256(), plans=plans)
    registered = output / 'paired_plan.json'
    if registered.exists():
        if json.loads(registered.read_text(encoding='utf-8')) != identity:
            raise ValueError('Paired source/data/plan changed; preserve this directory and start an independent run')
    elif output.exists() and any(path.name not in ('run.lock', 'console.log') for path in output.iterdir()):
        raise ValueError('Unregistered output already contains artifacts; use a clean paired directory')
    if args.check:
        print(json.dumps(identity, indent=2))
        return 0
    # A --check does not initialize CUDA or allocate an output plan.
    uuids = gpu_guard()
    atomic_json(registered, identity)
    for name, plan in plans.items():
        run = plan['runs'][0]
        directory = Path(run['output_dir'])
        if (directory / 'training_paused.json').is_file():
            read_result(plan)
            print(f'{name}: fixed epoch 8 already measured; keeping existing results', flush=True)
            continue
        gpu_guard()
        arguments = benchmark_arguments(output, name, manifest, weights, ','.join(uuids))
        if (directory.parents[1] / 'benchmark_plan.json').is_file():
            arguments.append('--resume')
        print(f'Start {name}: full author Task 1, 8/50 epochs, physical GPUs 0,1', flush=True)
        result = run_benchmark.main(arguments)
        if result:
            return result
        read_result(plan)
    atomic_json(output / 'comparison.json', comparison(plans))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
