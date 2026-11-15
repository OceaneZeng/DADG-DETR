"""Bind the paired experiment to actual initialization, fixed metrics and lineage."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tools.owod import run_benchmark
from tools.owod import run_mowodb_background_pair as pair
from tools.owod.paired_initialization import (audit_initialization, recover_final_evaluation,
                                             runtime_sha256, state_sha256)
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.tests.test_discovery_v7 import configured, inputs
from tools.owod.validate_discovery import validate_plan_arguments


def test_default_preserves_old_plans_and_checkpoints_and_v9_cannot_mix(inputs):
    inputs.discovery_known_rank_coef = 1.
    inputs.discovery_pseudo_normalization = 'count'
    legacy = run_benchmark.build_plan(inputs)
    original = configured(legacy)
    assert original.discovery_version == 8
    del original.discovery_background_negative, original.initialization_audit
    validate_plan_arguments(legacy['runs'][0], original)
    validate_checkpoint(configured(legacy), dict(args=original, epoch=7), resume=True)
    inputs.discovery_background_negative = False
    assert run_benchmark.build_plan(inputs) == legacy
    inputs.discovery_background_negative = True
    current = run_benchmark.build_plan(inputs)
    args = configured(current)
    assert args.discovery_version == 9
    validate_plan_arguments(current['runs'][0], args)
    validate_checkpoint(args, dict(args=copy.deepcopy(args), epoch=7), resume=True)
    for resume in (True, False):
        with pytest.raises(ValueError, match='discovery_background_negative'):
            validate_checkpoint(args, dict(args=original, epoch=49), resume=resume)
    inputs.task1_only = False
    plan = run_benchmark.build_plan(inputs)
    phases = [run for run in plan['runs'] if run.get('kind') != 'calibration']
    assert len(phases) == 7
    assert all('--discovery-background-negative' in run['command'] for run in phases)
    inputs.method = 'control'
    with pytest.raises(ValueError, match='require discovery'):
        run_benchmark.build_plan(inputs)
    args.owod_discovery = False
    with pytest.raises(ValueError, match='require discovery'):
        configure_run(args)


def audit_args(tmp_path):
    return SimpleNamespace(initialization_audit=str(tmp_path / 'common.json'),
        output_dir=str(tmp_path / 'reference'), seed=42, world_size=2,
        owod_manifest_sha256='manifest', backbone_weights_sha256='DINO', resume='')


def test_initialization_audit_preserves_rng_rejects_different_weights_and_resume_source(tmp_path, monkeypatch):
    args = audit_args(tmp_path)
    model = torch.nn.Linear(4, 3)
    rng = torch.get_rng_state().clone()
    before = copy.deepcopy(model.state_dict())
    audit_initialization(model, args)
    assert torch.equal(rng, torch.get_rng_state())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert args.initialization_sha256 == state_sha256(model)
    args.output_dir = str(tmp_path / 'background')
    audit_initialization(copy.deepcopy(model), args)
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.bias.add_(.1)
    with pytest.raises(ValueError, match='initialization or source differs'):
        audit_initialization(changed, args)
    args.resume = 'checkpoint.pth'
    audit_initialization(changed, args)  # Resuming checks original proof, not learned weights.
    monkeypatch.setattr('tools.owod.paired_initialization.runtime_sha256', lambda: 'changed')
    with pytest.raises(ValueError, match='provenance changed'):
        audit_initialization(changed, args)


def test_interrupted_last_evaluation_is_recovered_exactly_once(tmp_path):
    args = SimpleNamespace(pause_after_epochs=8, start_epoch=8)
    log = tmp_path / 'metrics.jsonl'
    stats = dict(owod_known_ap50=.4, owod_u_recall=.1)
    recover_final_evaluation(log, args, stats, 100)
    recover_final_evaluation(log, args, stats, 100)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['epoch'] == 7
    assert rows[0]['test_owod_known_ap50'] == .4
    assert not any(key.startswith('train_') for key in rows[0])
    with log.open('a') as handle:
        handle.write(json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        recover_final_evaluation(log, args, stats, 100)


def test_gpu_guard_uses_physical_zero_one_and_rejects_small_busy_jobs(monkeypatch):
    def query(command, **_kwargs):
        if '--query-gpu=index,uuid,name' in command:
            return '0, GPU-a, NVIDIA GeForce RTX 3090\n1, GPU-b, NVIDIA GeForce RTX 3090\n2, GPU-c, RTX 3090\n'
        return 'GPU-c, 123\n'
    monkeypatch.setattr(subprocess, 'check_output', query)
    assert pair.gpu_guard() == ['GPU-a', 'GPU-b']
    def occupied(command, **kwargs):
        return query(command, **kwargs) if '--query-gpu=index,uuid,name' in command else 'GPU-a, 456\n'
    monkeypatch.setattr(subprocess, 'check_output', occupied)
    with pytest.raises(ValueError, match='occupied'):
        pair.gpu_guard()


def fixture_results(inputs, tmp_path):
    plans = pair.paired_plans(tmp_path / 'pair', inputs.manifest, inputs.backbone_weights)
    initialization = dict(model_sha256='same', source_sha256=runtime_sha256())
    for name, plan in plans.items():
        args = configured(plan)
        args.pause_after_epochs = 8
        args.initialization_sha256 = 'same'
        directory = Path(args.output_dir)
        directory.mkdir(parents=True)
        Path(args.initialization_audit).write_text(json.dumps(initialization))
        (directory / 'initialization.json').write_text(json.dumps(initialization))
        root = directory.parents[1]
        (root / 'benchmark_plan.json').write_text(json.dumps(plan))
        (directory / 'training_paused.json').write_text(json.dumps(dict(last_epoch=7, epochs=50)))
        torch.save(dict(args=args, epoch=7, model={}, optimizer={}, lr_scheduler={},
                        discovery_ema={}), directory / 'checkpoint.pth')
        row = dict(epoch=7, n_parameters=100, **{'test_owod_' + metric:
            .5 if name == 'reference' else .6 for metric in pair.METRICS})
        (directory / 'metrics.jsonl').write_text(json.dumps(row) + '\n')
    return plans


def test_comparison_requires_fixed_epoch_complete_metrics_common_init_and_matching_plan(inputs, tmp_path):
    plans = fixture_results(inputs, tmp_path)
    result = pair.comparison(plans)
    assert result['previous_ap50'] is None
    assert result['status'] == 'measured_single_seed_diagnostic'
    assert result['metrics']['known_ap50']['delta'] == pytest.approx(.1)
    directory = Path(plans['background']['runs'][0]['output_dir'])
    metrics = directory / 'metrics.jsonl'
    original = metrics.read_text()
    metrics.write_text(original + original)
    with pytest.raises(ValueError, match='exactly one'):
        pair.comparison(plans)
    row = json.loads(original)
    del row['test_owod_unknown_precision']
    metrics.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='required metric'):
        pair.comparison(plans)
    metrics.write_text(original)
    changed = copy.deepcopy(plans)
    changed['background']['manifest_sha256'] = 'changed'
    with pytest.raises(ValueError, match='provenance differs'):
        pair.comparison(changed)
    proof = directory / 'initialization.json'
    proof.write_text(json.dumps(dict(model_sha256='different', source_sha256=runtime_sha256())))
    with pytest.raises(ValueError, match='Initialization proof'):
        pair.comparison(plans)


def test_runner_checks_without_cuda_and_automatically_resumes_partial_group(inputs, tmp_path, monkeypatch):
    output = tmp_path / 'pair'
    arguments = ['--output', str(output), '--manifest', str(inputs.manifest),
                 '--weights', str(inputs.backbone_weights)]
    monkeypatch.setattr(pair, 'gpu_guard', lambda: pytest.fail('CPU check touched GPUs'))
    assert pair.main(arguments + ['--check']) == 0
    assert not output.exists()
    monkeypatch.setattr(pair, 'gpu_guard', lambda: ['GPU-a', 'GPU-b'])
    calls = []
    def run(argv):
        calls.append(argv)
        planned = run_benchmark.get_args_parser().parse_args(argv)
        seed = Path(planned.output_dir) / 'm-owodb/seed_42'
        seed.mkdir(parents=True, exist_ok=True)
        (seed / 'benchmark_plan.json').write_text('{}')
        return 7
    monkeypatch.setattr(run_benchmark, 'main', run)
    assert pair.main(arguments) == 7
    assert '--resume' not in calls[-1]
    assert pair.main(arguments) == 7
    assert '--resume' in calls[-1]
    identity = output / 'paired_plan.json'
    content = json.loads(identity.read_text())
    content['source_sha256'] = 'changed'
    identity.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='source/data/plan changed'):
        pair.main(arguments)


def test_completed_pair_is_verified_and_skipped_without_training(inputs, tmp_path, monkeypatch):
    output = tmp_path / 'pair'
    plans = fixture_results(inputs, tmp_path)
    identity = dict(version=1, completed_epochs=8, source_sha256=runtime_sha256(), plans=plans)
    (output / 'paired_plan.json').write_text(json.dumps(identity))
    monkeypatch.setattr(pair, 'gpu_guard', lambda: ['GPU-a', 'GPU-b'])
    monkeypatch.setattr(run_benchmark, 'main', lambda *_args: pytest.fail('Completed run retrained'))
    arguments = ['--output', str(output), '--manifest', str(inputs.manifest),
                 '--weights', str(inputs.backbone_weights)]
    assert pair.main(arguments) == 0
    assert (output / 'comparison.json').is_file()
    (output / 'paired_plan.json').unlink()
    with pytest.raises(ValueError, match='Unregistered output'):
        pair.main(arguments)


def test_actual_main_recovers_checkpoint_saved_before_final_evaluation(inputs, tmp_path, monkeypatch):
    import main as training
    import models.graph_local.discovery as discovery

    class TinyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer('observations', torch.zeros((), dtype=torch.long))

        def train_discovery(self):
            pass

    class TinyCriterion(torch.nn.Module):
        weight_dict = {}

    def distributed(args):
        args.distributed, args.rank, args.world_size = False, 0, 1

    updates, interrupted = [], []
    def train(model, criterion, loader, optimizer, device, epoch, *args, discovery_teacher=None, **kwargs):
        optimizer.zero_grad()
        loss = model.weight.square().sum()
        loss.backward()
        optimizer.step()
        model.observations.add_(1)
        discovery.update_ema(discovery_teacher, model)
        updates.append(epoch)
        return dict(loss=loss.item())

    def evaluate(model, *_args, **_kwargs):
        if model.observations.item() == 8 and not interrupted:
            interrupted.append(True)
            raise RuntimeError('controlled interruption after checkpoint before evaluation')
        return ({'owod_' + name: .5 for name in pair.METRICS}, SimpleNamespace(coco_eval={}))

    monkeypatch.setattr(training.utils, 'init_distributed_mode', distributed)
    monkeypatch.setattr(training.utils, 'get_sha', lambda: 'fixture')
    monkeypatch.setattr(discovery, 'build', lambda _args: (TinyDetector(), TinyCriterion(), {}))
    monkeypatch.setattr(training, 'build_dataset', lambda **kwargs: [0, 1])
    monkeypatch.setattr(training, 'get_coco_api_from_dataset', lambda _dataset: None)
    monkeypatch.setattr(training, 'train_one_epoch', train)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    inputs.pause_after_epochs = 8
    inputs.initialization_audit = str(tmp_path / 'proof.json')
    inputs.discovery_background_negative = True
    args = configured(run_benchmark.build_plan(inputs))
    args.pause_after_epochs, args.device, args.no_file_log = 8, 'cpu', True
    with pytest.raises(RuntimeError, match='controlled interruption'):
        training.main(args)
    directory = Path(args.output_dir)
    checkpoint = directory / 'checkpoint.pth'
    initial = json.loads((directory / 'initialization.json').read_text())
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['args'].initialization_sha256 == initial['model_sha256']
    formal = copy.deepcopy(args)
    formal.initialization_audit = ''
    with pytest.raises(ValueError, match='diagnostic checkpoints and formal chains'):
        validate_checkpoint(formal, saved, resume=True)
    args.resume = str(checkpoint)
    training.main(args)
    assert updates == list(range(8))  # The eighth epoch is evaluated, never retrained.
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == list(range(8))
    assert rows[-1]['evaluation_recovered_after_resume']
    assert json.loads((directory / 'training_paused.json').read_text())['last_epoch'] == 7


@pytest.mark.parametrize('mode', ['check', 'data-failure', 'busy', 'locked', 'normal'])
def test_bash_launcher_stops_before_training_on_failed_preflight(tmp_path, mode):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    root = tmp_path / 'repo'
    script = root / 'tools/owod/run_mowodb_background_pair.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(pair.ROOT / script.relative_to(root), script)
    startup = tmp_path / 'startup.sh'
    startup.write_text('''task_python() {
  printf '%s\n' "$*" >> "$TEST_TRACE"
  if [[ $TEST_MODE == data-failure && $* == *manage_data.py* ]]; then return 7; fi
  if [[ $TEST_MODE == busy && $* == *--gpu-check* ]]; then return 8; fi
  return 0
}
export -f task_python
flock() { if [[ $TEST_MODE == locked ]]; then return 1; fi; return 0; }
git() { printf 'fixture-git\n'; return 0; }
''')
    # An executable wrapper, so the launcher's interpreter existence check also runs.
    wrapper = tmp_path / 'env/bin/task-python'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('#!/usr/bin/env bash\ntask_python "$@"\n')
    wrapper.chmod(0o755)
    trace = tmp_path / 'trace'
    env = dict(os.environ, BASH_ENV=startup.as_posix(), TREE_DETR_PYTHON=wrapper.as_posix(),
               TEST_TRACE=trace.as_posix(), TEST_MODE=mode, TMUX='test-session')
    command = [bash, script.as_posix()] + (['--check'] if mode == 'check' else [])
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
    expected = 7 if mode == 'data-failure' else 8 if mode == 'busy' else 1 if mode == 'locked' else 0
    assert result.returncode == expected, result.stdout + result.stderr
    calls = trace.read_text().splitlines() if trace.exists() else []
    training = [line for line in calls if line.startswith('-u tools/owod/run_mowodb_background_pair.py')]
    assert bool(training) == (mode == 'normal')
    if mode == 'check':
        assert not any('--gpu-check' in line for line in calls)
