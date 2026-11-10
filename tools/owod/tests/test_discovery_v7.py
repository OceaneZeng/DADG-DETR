"""Bind loss ablations to checkpoint lineage and keep diagnostic pauses resumable."""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from main import get_args_parser
from tools.owod import run_benchmark
from tools.owod.protocol import build_official_manifest
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.tests.test_protocol import fixture_split, make_tree
from util.training_runtime import training_epoch_limit, write_training_status


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.owod.protocol.official_split', fixture_split)
    make_tree(tmp_path)
    manifest = build_official_manifest(tmp_path, 'm-owodb', 'author-release-fixture')
    manifest['image_root'] = str(tmp_path)
    path = tmp_path / 'split_manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    for stem in ('100001', '100002', '900001'):
        (tmp_path / (stem + '.jpg')).write_bytes(b'path fixture, never decoded')
    weights = tmp_path / 'dino.pth'
    weights.write_bytes(b'hash fixture, never loaded')
    return SimpleNamespace(manifest=path, recipe='ow-detr-paper', finetune_epochs=None,
        backbone_weights=weights, output_dir=tmp_path / 'runs', seed=42, task1_only=True,
        method='discovery', graph_adapter_rank=8, graph_adapter_last_n=2, graph_risk_beta=2.,
        graph_route_loss_coef=1., graph_sketch_max_images=12, nproc_per_node=2,
        batch_size=2, num_workers=0, discovery_pseudo_ramp_epochs=3)


def configured(plan):
    command = plan['runs'][0]['command']
    index = next(i for i, value in enumerate(command) if Path(value).name == 'main.py')
    args = get_args_parser().parse_args(command[index + 1:])
    args.world_size = 2
    args.graph_adapter_specs = []
    configure_run(args)
    return args


@pytest.mark.parametrize('objectness,normalization', [
    ('focal', 'weight-sum'), ('balanced-bce', 'count'), ('focal', 'count')])
def test_v7_modes_bind_actual_losses_and_reject_v6_and_other_modes(inputs, objectness, normalization):
    legacy_plan = run_benchmark.build_plan(inputs)
    legacy = configured(legacy_plan)
    assert legacy.discovery_version == 6
    inputs.discovery_objectness_loss = objectness
    inputs.discovery_pseudo_normalization = normalization
    args = configured(run_benchmark.build_plan(inputs))
    assert args.discovery_version == 7
    assert args.discovery_objectness_loss == objectness
    assert args.discovery_pseudo_normalization == normalization
    payload = dict(args=copy.deepcopy(args), epoch=7)
    validate_checkpoint(args, payload, resume=True)
    for resume in (False, True):
        with pytest.raises(ValueError, match='discovery_'):
            validate_checkpoint(args, dict(args=legacy, epoch=49), resume=resume)
    for field, other in [('discovery_objectness_loss', 'balanced-bce' if objectness == 'focal' else 'focal'),
                         ('discovery_pseudo_normalization', 'count' if normalization == 'weight-sum' else 'weight-sum')]:
        changed = copy.deepcopy(args)
        setattr(changed, field, other)
        with pytest.raises(ValueError, match=field):
            validate_checkpoint(changed, payload, resume=True)


def test_defaults_keep_legacy_plan_and_old_checkpoint_compatible(inputs):
    original = run_benchmark.build_plan(inputs)
    inputs.discovery_objectness_loss = 'balanced-bce'
    inputs.discovery_pseudo_normalization = 'weight-sum'
    assert run_benchmark.build_plan(inputs) == original
    args = configured(original)
    old = copy.deepcopy(args)
    del old.discovery_objectness_loss, old.discovery_pseudo_normalization, old.pause_after_epochs
    validate_checkpoint(args, dict(args=old, epoch=7), resume=True)


def test_pause_preserves_schedule_and_cannot_promote_partial_task(inputs, tmp_path):
    inputs.discovery_objectness_loss = 'focal'
    unbounded = run_benchmark.build_plan(inputs)
    inputs.pause_after_epochs = 8
    assert run_benchmark.build_plan(inputs) == unbounded
    args = configured(unbounded)
    args.pause_after_epochs = 8
    assert args.epochs == args.lr_drop == 50
    assert training_epoch_limit(args) == 8
    saved = copy.deepcopy(args)
    args.pause_after_epochs = 0
    assert training_epoch_limit(args) == 50
    validate_checkpoint(args, dict(args=saved, epoch=7), resume=True)
    write_training_status(tmp_path, 50, 7, 123)
    assert not (tmp_path / 'training_complete.json').exists()
    assert json.loads((tmp_path / 'training_paused.json').read_text())['stage_complete'] is False
    args.owod_stage = 1
    with pytest.raises(ValueError, match='completed preceding phase'):
        validate_checkpoint(args, dict(args=saved, epoch=7))
    write_training_status(tmp_path, 50, 49, 456)
    assert not (tmp_path / 'training_paused.json').exists()
    assert json.loads((tmp_path / 'training_complete.json').read_text())['last_epoch'] == 49


@pytest.mark.parametrize('change', [dict(task1_only=False), dict(method='graph-adapter'),
                                   dict(pause_after_epochs=-1), dict(pause_after_epochs=51)])
def test_pause_rejects_unsafe_scopes(inputs, change):
    inputs.pause_after_epochs = 8
    for name, value in change.items():
        setattr(inputs, name, value)
    with pytest.raises(ValueError, match='pause-after-epochs'):
        run_benchmark.build_plan(inputs)


def test_execution_pause_is_not_in_plan_and_resume_can_continue_to_full_budget(tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    directory = root / 'm-owodb/seed_42/stage_0/increment'
    command = ['python', 'main.py', '--epochs', '50', '--lr_drop', '50']
    plan = dict(protocol='m-owodb', runs=[dict(stage=0, phase='increment', epochs=50,
                                            output_dir=str(directory), command=command)])
    monkeypatch.setattr(run_benchmark, 'build_plan', lambda args: plan)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'checkpoint.pth').write_bytes(b'runner fixture, never loaded')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_benchmark.subprocess, 'run', execute)
    common = ['--manifest', str(tmp_path / 'manifest.json'), '--recipe', 'ow-detr-paper',
              '--output-dir', str(root), '--method', 'discovery', '--task1-only']
    assert run_benchmark.main(common + ['--pause-after-epochs', '8']) == 0
    assert commands[0][-2:] == ['--pause-after-epochs', '8']
    plan_path = root / 'm-owodb/seed_42/benchmark_plan.json'
    assert '--pause-after-epochs' not in plan_path.read_text()
    before = plan_path.read_bytes()
    assert run_benchmark.main(common + ['--resume']) == 0
    assert '--resume' in commands[1] and '--pause-after-epochs' not in commands[1]
    assert plan_path.read_bytes() == before


@pytest.mark.parametrize('variant', ['objectness', 'confidence', 'combined'])
def test_launcher_runs_variant_with_fixed_gpu_budget_and_captures_data_failures(tmp_path, variant):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    script = tmp_path / 'tools/owod/run_discovery_mowodb_3090_v7.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / script.name, script)
    trace = tmp_path / 'calls.txt'
    startup = tmp_path / 'test-env.sh'
    startup.write_text('''python() {
  printf '%q ' "$@" >> "$TEST_TRACE"
  printf '\\n' >> "$TEST_TRACE"
  if [[ ${TEST_FAIL_VERIFY:-0} == 1 && "$*" == *manage_data.py* ]]; then return 7; fi
  return 0
}
''', encoding='utf-8')
    env = dict(os.environ, BASH_ENV=startup.as_posix(), TEST_TRACE=trace.as_posix())
    result = subprocess.run([bash, script.as_posix(), variant, '--dry-run'], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [shlex.split(line) for line in trace.read_text().splitlines()]
    command = next(call for call in calls if 'tools/owod/run_benchmark.py' in call)
    assert command[command.index('--gpus') + 1] == '0,1'
    assert command[command.index('--batch-size') + 1] == '2'
    assert command[command.index('--pause-after-epochs') + 1] == '8'
    assert '--task1-only' in command and '--dry-run' in command
    assert ('--discovery-objectness-loss' in command) == (variant != 'confidence')
    assert ('--discovery-pseudo-normalization' in command) == (variant != 'objectness')
    trace.write_text('')
    full = subprocess.run([bash, script.as_posix(), variant, '--resume', '--full'], env=env,
                          capture_output=True, text=True, timeout=20)
    assert full.returncode == 0, full.stdout + full.stderr
    command = next(shlex.split(line) for line in trace.read_text().splitlines()
                   if 'run_benchmark.py' in line)
    assert '--resume' in command and command[command.index('--pause-after-epochs') + 1] == '0'
    trace.write_text('')
    env['TEST_FAIL_VERIFY'] = '1'
    failed = subprocess.run([bash, script.as_posix(), variant], env=env,
                            capture_output=True, text=True, timeout=20)
    assert failed.returncode == 7
    assert 'run_benchmark.py' not in trace.read_text()
    logfile = tmp_path / f'output/tree-detr-m-v7-{variant}-3090-logs/console.log'
    assert f'variant={variant}' in logfile.read_text()


@pytest.mark.parametrize('rank_coef,grad_scale', [(0., 1.), (1., .1)])
def test_actual_training_loop_pauses_and_restores_all_state_without_shortening_schedule(inputs, monkeypatch, rank_coef, grad_scale):
    import torch
    from torch import nn
    import main as training
    import models.graph_local.discovery as discovery

    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer('observations', torch.zeros((), dtype=torch.long))

        def train_discovery(self):
            pass

    class TinyCriterion(nn.Module):
        weight_dict = {}

    def distributed(args):
        args.distributed = False
        args.rank = 0
        args.world_size = 1

    updates = []
    def train(model, criterion, loader, optimizer, device, epoch, *args, discovery_teacher=None, **kwargs):
        optimizer.zero_grad()
        loss = model.weight.square().sum()
        loss.backward()
        optimizer.step()
        model.observations.add_(1)
        discovery.update_ema(discovery_teacher, model)
        updates.append(epoch)
        return {'loss': float(loss.detach())}

    monkeypatch.setattr(training.utils, 'init_distributed_mode', distributed)
    monkeypatch.setattr(training.utils, 'get_sha', lambda: 'test fixture')
    monkeypatch.setattr(discovery, 'build', lambda args: (TinyDetector(), TinyCriterion(), {}))
    monkeypatch.setattr(training, 'build_dataset', lambda **kwargs: [0, 1])
    monkeypatch.setattr(training, 'get_coco_api_from_dataset', lambda dataset: None)
    monkeypatch.setattr(training, 'train_one_epoch', train)
    monkeypatch.setattr(training, 'evaluate', lambda *args, **kwargs: ({}, SimpleNamespace(coco_eval={})))
    inputs.discovery_objectness_loss = 'focal'
    inputs.discovery_known_rank_coef = rank_coef
    inputs.discovery_feature_grad_scale = grad_scale
    fresh = configured(run_benchmark.build_plan(inputs))
    fresh.device = 'cpu'
    paused = copy.deepcopy(fresh)
    paused.pause_after_epochs = 8
    training.main(paused)
    directory = Path(paused.output_dir)
    checkpoint = directory / 'checkpoint.pth'
    payload = torch.load(checkpoint, weights_only=False)
    assert updates == list(range(8))
    assert payload['epoch'] == 7 and payload['args'].epochs == payload['args'].lr_drop == 50
    assert payload['lr_scheduler']['last_epoch'] == 8
    assert payload['discovery_ema']['ema_update_count'] == 8
    assert (directory / 'checkpoint0007.pth').is_file()
    assert (directory / 'training_paused.json').is_file()
    assert not (directory / 'training_complete.json').exists()
    resumed = copy.deepcopy(fresh)
    resumed.resume = str(checkpoint)
    training.main(resumed)
    restored = torch.load(checkpoint, weights_only=False)
    assert updates == list(range(50))
    assert restored['epoch'] == restored['lr_scheduler']['last_epoch'] - 1 == 49
    assert restored['discovery_ema']['ema_update_count'] == restored['model']['observations'] == 50
    assert (directory / 'training_complete.json').is_file()
    assert not (directory / 'training_paused.json').exists()
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == list(range(50))
    uninterrupted = copy.deepcopy(fresh)
    uninterrupted.output_dir = str(directory.parent / 'uninterrupted')
    training.main(uninterrupted)
    reference = torch.load(Path(uninterrupted.output_dir) / 'checkpoint.pth', weights_only=False)
    for section in ('model', 'discovery_ema'):
        for key in reference[section]:
            torch.testing.assert_close(restored[section][key], reference[section][key], rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(restored['optimizer']['state'][0][key],
                                   reference['optimizer']['state'][0][key], rtol=0, atol=0)
