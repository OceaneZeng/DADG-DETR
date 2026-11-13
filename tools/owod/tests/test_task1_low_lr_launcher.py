"""Low-LR gradient/scoring comparisons, unchanged schedule and isolated recovery."""
import copy
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from tools.owod import run_benchmark
from tools.owod.tests.test_discovery_v7 import configured, inputs


def test_low_lr_comparison_changes_only_feature_gradient_and_locks_resume(inputs):
    from tools.owod.strict_setting import validate_checkpoint
    inputs.discovery_pseudo_normalization = 'count'
    inputs.discovery_known_rank_coef = 1.
    inputs.discovery_lr_scale = .25
    inputs.pause_after_epochs = 8
    reference = run_benchmark.build_plan(inputs)
    inputs.discovery_feature_grad_scale = .1
    protected = run_benchmark.build_plan(inputs)
    assert len(reference['runs']) == len(protected['runs']) == 1
    command = list(protected['runs'][0]['command'])
    offset = command.index('--discovery-feature-grad-scale')
    del command[offset:offset + 2]
    assert command == reference['runs'][0]['command']
    args = configured(protected)
    assert args.lr == 5e-5 and args.epochs == args.lr_drop == 50
    assert args.discovery_known_rank_coef == 1. and args.discovery_feature_grad_scale == .1
    with pytest.raises(ValueError, match='discovery_feature_grad_scale'):
        validate_checkpoint(args, dict(args=configured(reference), epoch=7), resume=True)
    inputs.pause_after_epochs = 0
    assert run_benchmark.build_plan(inputs) == protected


def test_focal_changes_only_scoring_from_ranking_and_rejects_cross_variant_resume(inputs):
    from tools.owod.strict_setting import validate_checkpoint
    inputs.discovery_pseudo_normalization = 'count'
    inputs.discovery_known_rank_coef = 1.
    inputs.discovery_lr_scale = .25
    inputs.pause_after_epochs = 8
    reference = run_benchmark.build_plan(inputs)
    inputs.discovery_scoring = 'focal'
    focal = run_benchmark.build_plan(inputs)
    command = list(focal['runs'][0]['command'])
    offset = command.index('--discovery-scoring') + 1
    assert command[offset] == 'focal'
    command[offset] = 'hierarchical'
    assert command == reference['runs'][0]['command']
    args = configured(focal)
    assert args.discovery_version == 8
    assert args.discovery_feature_grad_scale == args.discovery_known_rank_coef == 1.
    assert args.discovery_objectness_loss == 'balanced-bce'
    assert args.lr == 5e-5 and args.epochs == args.lr_drop == 50
    validate_checkpoint(args, dict(args=copy.deepcopy(args), epoch=7), resume=True)
    for active, saved in ((args, configured(reference)), (configured(reference), args)):
        with pytest.raises(ValueError, match='discovery_scoring'):
            validate_checkpoint(active, dict(args=saved, epoch=7), resume=True)
    changed = copy.deepcopy(args)
    changed.discovery_feature_grad_scale = .1
    with pytest.raises(ValueError, match='discovery_feature_grad_scale'):
        validate_checkpoint(args, dict(args=changed, epoch=7), resume=True)
    inputs.pause_after_epochs = 0
    assert run_benchmark.build_plan(inputs) == focal


@pytest.mark.parametrize('variant,scale,scoring', [
    ('ranking', '1.0', 'hierarchical'), ('combined', '0.1', 'hierarchical'),
    ('focal', '1.0', 'focal'), (None, '0.1', 'hierarchical')])
def test_launcher_preserves_variants_schedules_existing_outputs_and_failures(tmp_path, variant, scale, scoring):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    script = tmp_path / 'tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh'
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
        return subprocess.run([bash, script.as_posix(), *([variant] if variant else []), *args], env=env,
                              capture_output=True, text=True, timeout=20)

    result = execute('--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    for option, expected in [('--discovery-lr-scale', '0.25'), ('--gpus', '0,1'),
        ('--nproc-per-node', '2'), ('--batch-size', '2'), ('--pause-after-epochs', '8'),
        ('--discovery-known-rank-coef', '1.0'), ('--discovery-feature-grad-scale', scale),
        ('--discovery-scoring', scoring),
        ('--output-dir', f'output/tree-detr-m-v8-{variant or "combined"}-lr-quarter-3090-task1')]:
        assert command[command.index(option) + 1] == expected
    assert '--task1-only' in command and '--pretrained' not in command
    assert execute('--resume').returncode == 2 and not trace.read_text()
    root = tmp_path / f'output/tree-detr-m-v8-{variant or "combined"}-lr-quarter-3090-task1/m-owodb/seed_42'
    checkpoint = root / 'stage_0/increment/checkpoint.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'preserve existing result')
    (root / 'benchmark_plan.json').write_text('{}')
    assert execute().returncode == 2 and not trace.read_text()
    assert checkpoint.read_bytes() == b'preserve existing result'
    assert execute('--resume', '--full').returncode == 0
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert '--resume' in command and command[command.index('--pause-after-epochs') + 1] == '0'
    env['TEST_FAIL_VERIFY'] = '1'
    assert execute('--resume').returncode == 7 and 'run_benchmark.py' not in trace.read_text()
    env['TEST_FAIL_VERIFY'] = '0'
    env['TEST_FAIL_TRAIN'] = '1'
    assert execute('--resume').returncode == 9
    assert execute('--unknown-option').returncode == 2
