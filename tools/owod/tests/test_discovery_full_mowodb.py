"""Protect the low-LR full M task chain and its independent launch directory."""
import copy
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from tools.owod import run_benchmark
from tools.owod.strict_setting import validate_checkpoint
from tools.owod.tests.test_discovery_v7 import configured, inputs


def ranking_full(inputs):
    inputs.task1_only = False
    inputs.discovery_pseudo_normalization = 'count'
    inputs.discovery_known_rank_coef = 1.
    return run_benchmark.build_plan(inputs)


def test_full_chain_scales_only_training_lrs_and_binds_resume(inputs):
    standard = ranking_full(inputs)
    inputs.discovery_lr_scale = 1.
    assert run_benchmark.build_plan(inputs) == standard
    inputs.discovery_lr_scale = .25
    scaled = run_benchmark.build_plan(inputs)
    training = [r for r in scaled['runs'] if r.get('kind') != 'calibration']
    assert [(r['stage'], r['phase'], r['epochs']) for r in training] == [
        (0, 'increment', 50), (1, 'increment', 50), (1, 'finetune', 20),
        (2, 'increment', 50), (2, 'finetune', 20),
        (3, 'increment', 50), (3, 'finetune', 20)]
    assert len(scaled['runs']) == 10  # Seven training phases and three GNN calibrations.
    previous = None
    for before, after in zip(standard['runs'], scaled['runs']):
        if after.get('kind') == 'calibration':
            assert before == after
            assert after['command'][after['command'].index('--checkpoint') + 1] == previous
            continue
        command = after['command']
        expected_lr = 5e-5 if after['phase'] == 'increment' else 5e-6
        assert float(command[command.index('--lr') + 1]) == expected_lr
        assert float(command[command.index('--lr_backbone') + 1]) == expected_lr / 10
        original = list(before['command'])
        for option in ('--lr', '--lr_backbone'):
            original[original.index(option) + 1] = command[command.index(option) + 1]
        assert original == command
        if previous:
            assert command[command.index('--pretrained') + 1] == previous
            assert '--old-class-distillation' in command
        else:
            assert '--pretrained' not in command
        previous = str(Path(after['output_dir']) / 'checkpoint.pth')
    assert scaled['discovery_optimization']['lr_scale'] == .25
    assert scaled['discovery_optimization']['finetune_lr'] == 5e-6
    assert 'not the original optimization recipe' in scaled['recipe_source']
    args = configured(scaled)
    assert args.discovery_version == 8
    validate_checkpoint(args, dict(args=copy.deepcopy(args), epoch=7), resume=True)
    with pytest.raises(ValueError, match='cannot change lr'):
        validate_checkpoint(args, dict(args=configured(standard), epoch=7), resume=True)


@pytest.mark.parametrize('value', [0., -1., 1.5, float('nan'), float('inf')])
def test_discovery_lr_scale_rejects_invalid_values(inputs, value):
    inputs.discovery_lr_scale = value
    with pytest.raises(ValueError, match='discovery-lr-scale'):
        ranking_full(inputs)


@pytest.mark.parametrize('method', ['control', 'graph-adapter'])
def test_discovery_lr_scale_does_not_modify_other_methods(inputs, method):
    inputs.discovery_lr_scale = .25
    inputs.method = method
    with pytest.raises(ValueError, match='discovery-lr-scale'):
        run_benchmark.build_plan(inputs)


def test_launcher_full_chain_protects_results_and_logs_failures(tmp_path):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    script = tmp_path / 'tools/owod/run_discovery_mowodb_3090_full.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / script.name, script)
    trace = tmp_path / 'calls.txt'
    startup = tmp_path / 'test-env.sh'
    startup.write_text('''python() {
  printf '%q ' "$@" >> "$TEST_TRACE"
  printf '\\n' >> "$TEST_TRACE"
  if [[ ${TEST_FAIL_VERIFY:-0} == 1 && "$*" == *manage_data.py* ]]; then return 7; fi
  if [[ ${TEST_FAIL_TRAIN:-0} == 1 && "$*" == *run_benchmark.py* ]]; then return 9; fi
  return 0
}
''', encoding='utf-8')
    env = dict(os.environ, BASH_ENV=startup.as_posix(), TEST_TRACE=trace.as_posix())

    def execute(*args):
        trace.write_text('')
        return subprocess.run([bash, script.as_posix(), *args], env=env,
                              capture_output=True, text=True, timeout=20)

    result = execute('--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert '--task1-only' not in command and '--pause-after-epochs' not in command
    for option, expected in [('--discovery-lr-scale', '0.25'), ('--method', 'discovery'),
        ('--gpus', '0,1'), ('--nproc-per-node', '2'), ('--batch-size', '2'),
        ('--discovery-known-rank-coef', '1.0'), ('--discovery-pseudo-normalization', 'count')]:
        assert command[command.index(option) + 1] == expected
    assert execute('--resume').returncode == 2
    assert not trace.read_text()
    root = tmp_path / 'output/tree-detr-m-v8-ranking-lr-quarter-3090-full/m-owodb/seed_42'
    checkpoint = root / 'stage_0/increment/checkpoint.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'existing formal chain')
    (root / 'benchmark_plan.json').write_text('{}')
    assert execute().returncode == 2
    assert checkpoint.read_bytes() == b'existing formal chain'
    assert not trace.read_text()
    assert execute('--resume').returncode == 0
    assert '--resume' in trace.read_text()
    env['TEST_FAIL_VERIFY'] = '1'
    assert execute('--resume').returncode == 7
    assert 'run_benchmark.py' not in trace.read_text()
    env['TEST_FAIL_VERIFY'] = '0'
    env['TEST_FAIL_TRAIN'] = '1'
    assert execute('--resume').returncode == 9
    assert execute('--full').returncode == 2
    logfile = tmp_path / 'output/tree-detr-m-v8-ranking-lr-quarter-3090-full-logs/console.log'
    assert 'tasks=4 training_phases=7 schedule=50/20' in logfile.read_text()
