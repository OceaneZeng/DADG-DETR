"""Keep stability experiments traceable, resumable and separate from legacy runs."""
import copy
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from tools.owod import run_benchmark
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.tests.test_discovery_v7 import configured, inputs
from tools.owod.validate_discovery import validate_plan_arguments


@pytest.mark.parametrize('rank_coef,grad_scale', [(1., 1.), (0., .1), (1., .1)])
def test_v8_modes_lock_both_options_and_reject_old_checkpoint(inputs, rank_coef, grad_scale):
    inputs.discovery_pseudo_normalization = 'count'
    legacy = configured(run_benchmark.build_plan(inputs))
    assert legacy.discovery_version == 7
    inputs.discovery_known_rank_coef = rank_coef
    inputs.discovery_feature_grad_scale = grad_scale
    plan = run_benchmark.build_plan(inputs)
    args = configured(plan)
    assert args.discovery_version == 8
    assert args.discovery_known_rank_coef == rank_coef
    assert args.discovery_feature_grad_scale == grad_scale
    checkpoint = dict(args=copy.deepcopy(args), epoch=7)
    validate_checkpoint(args, checkpoint, resume=True)
    validate_plan_arguments(plan['runs'][0], checkpoint['args'])
    for resume in (False, True):
        with pytest.raises(ValueError, match='discovery_'):
            validate_checkpoint(args, dict(args=legacy, epoch=49), resume=resume)
    for field, other in [('discovery_known_rank_coef', rank_coef + .5),
                         ('discovery_feature_grad_scale', .2 if grad_scale == 1. else 1.)]:
        changed = copy.deepcopy(args)
        setattr(changed, field, other)
        with pytest.raises(ValueError, match=field):
            validate_checkpoint(changed, checkpoint, resume=True)
        with pytest.raises(ValueError, match=field):
            validate_plan_arguments(plan['runs'][0], changed)


@pytest.mark.parametrize('ramp,objectness,normalization', [
    (0, 'balanced-bce', 'weight-sum'), (3, 'balanced-bce', 'weight-sum'),
    (3, 'focal', 'weight-sum'), (3, 'balanced-bce', 'count'), (3, 'focal', 'count')])
def test_old_plans_and_checkpoints_missing_v8_fields_keep_their_behavior(inputs, ramp, objectness, normalization):
    inputs.discovery_pseudo_ramp_epochs = ramp
    inputs.discovery_objectness_loss = objectness
    inputs.discovery_pseudo_normalization = normalization
    plan = run_benchmark.build_plan(inputs)
    args = configured(plan)
    saved = copy.deepcopy(args)
    del saved.discovery_known_rank_coef, saved.discovery_feature_grad_scale
    validate_checkpoint(args, dict(args=saved, epoch=7), resume=True)
    validate_plan_arguments(plan['runs'][0], saved)
    inputs.discovery_known_rank_coef = 0.
    inputs.discovery_feature_grad_scale = 1.
    assert run_benchmark.build_plan(inputs) == plan


@pytest.mark.parametrize('field,value', [('discovery_known_rank_coef', -1.),
    ('discovery_known_rank_coef', float('nan')), ('discovery_known_rank_coef', float('inf')),
    ('discovery_feature_grad_scale', -.1), ('discovery_feature_grad_scale', 1.1),
    ('discovery_feature_grad_scale', float('nan'))])
def test_invalid_options_fail_in_both_plan_and_training_preflight(inputs, field, value):
    args = configured(run_benchmark.build_plan(inputs))
    setattr(inputs, field, value)
    setattr(args, field, value)
    with pytest.raises(ValueError, match='stability configuration'):
        run_benchmark.build_plan(inputs)
    with pytest.raises(ValueError, match='stability configuration'):
        configure_run(args)


def test_stability_options_require_discovery_and_follow_the_entire_chain(inputs):
    args = configured(run_benchmark.build_plan(inputs))
    inputs.discovery_known_rank_coef = args.discovery_known_rank_coef = 1.
    inputs.method = 'control'
    args.owod_discovery = False
    with pytest.raises(ValueError, match='require'):
        run_benchmark.build_plan(inputs)
    with pytest.raises(ValueError, match='require'):
        configure_run(args)
    inputs.method = 'discovery'
    inputs.task1_only = False
    inputs.discovery_pseudo_normalization = 'count'
    inputs.discovery_feature_grad_scale = .1
    plan = run_benchmark.build_plan(inputs)
    phases = [run for run in plan['runs'] if run.get('kind') != 'calibration']
    assert len(phases) == 7
    for run in phases:
        command = run['command']
        assert command[command.index('--discovery-known-rank-coef') + 1] == '1.0'
        assert command[command.index('--discovery-feature-grad-scale') + 1] == '0.1'
        assert command[command.index('--discovery-pseudo-normalization') + 1] == 'count'


@pytest.mark.parametrize('variant', ['ranking', 'guarded', 'combined'])
def test_launcher_binds_variant_keeps_results_and_stops_on_bad_data(tmp_path, variant):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    script = tmp_path / 'tools/owod/run_discovery_mowodb_3090_v8.sh'
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

    def execute(*arguments):
        return subprocess.run([bash, script.as_posix(), variant, *arguments], env=env,
            capture_output=True, text=True, timeout=20)

    result = execute('--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert command[command.index('--gpus') + 1] == '0,1'
    assert command[command.index('--nproc-per-node') + 1] == '2'
    assert command[command.index('--batch-size') + 1] == '2'
    assert command[command.index('--pause-after-epochs') + 1] == '8'
    assert command[command.index('--discovery-pseudo-normalization') + 1] == 'count'
    assert '--task1-only' in command and '--dry-run' in command
    assert '--discovery-objectness-loss' not in command
    assert ('--discovery-known-rank-coef' in command) == (variant != 'guarded')
    assert ('--discovery-feature-grad-scale' in command) == (variant != 'ranking')
    trace.write_text('')
    result = execute('--dry-run', '--profile-runtime')
    assert result.returncode == 0, result.stdout + result.stderr
    profiled = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert profiled[profiled.index('--profile-runtime-steps') + 1] == '50'
    assert profiled[profiled.index('--profile-runtime-warmup') + 1] == '20'
    directory = tmp_path / f'output/tree-detr-m-v8-{variant}-3090/m-owodb/seed_42/stage_0/increment'
    directory.mkdir(parents=True)
    checkpoint = directory / 'checkpoint.pth'
    checkpoint.write_bytes(b'Existing results must be retained')
    trace.write_text('')
    assert execute().returncode == 2
    assert checkpoint.read_bytes() == b'Existing results must be retained'
    trace.write_text('')
    result = execute('--resume', '--full')
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert '--resume' in command and command[command.index('--pause-after-epochs') + 1] == '0'
    trace.write_text('')
    env['TEST_FAIL_VERIFY'] = '1'
    assert execute('--resume').returncode == 7
    assert 'run_benchmark.py' not in trace.read_text()
    logfile = tmp_path / f'output/tree-detr-m-v8-{variant}-3090-logs/console.log'
    assert f'variant={variant}' in logfile.read_text()
