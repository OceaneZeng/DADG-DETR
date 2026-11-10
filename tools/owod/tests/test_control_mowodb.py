"""Keep the M control diagnostic bounded, attributable and safe to resume."""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.owod import diagnose_control_inputs as diagnostic, run_benchmark
from tools.owod.strict_setting import validate_checkpoint
from tools.owod.tests.test_discovery_v7 import configured, inputs
from util.training_runtime import training_epoch_limit


def control_args(inputs):
    inputs.method = 'control'
    inputs.discovery_pseudo_ramp_epochs = 0
    return configured(run_benchmark.build_plan(inputs))


def test_control_pause_keeps_recipe_and_rejects_discovery_or_partial_transition(inputs):
    args = control_args(inputs)
    original = run_benchmark.build_plan(inputs)
    inputs.pause_after_epochs = 3
    assert run_benchmark.build_plan(inputs) == original
    assert len(original['runs']) == 1
    command = original['runs'][0]['command']
    assert '--owod-discovery' not in command and '--graph-routed-adapter' not in command
    assert '--pretrained' not in command and '--resume' not in command
    assert args.epochs == args.lr_drop == 50
    assert args.lr == 2e-4 and args.lr_backbone == 2e-5 and args.batch_size == 2
    args.pause_after_epochs = 3
    assert training_epoch_limit(args) == 3
    saved = copy.deepcopy(args)
    args.pause_after_epochs = 0
    assert training_epoch_limit(args) == 50
    validate_checkpoint(args, dict(args=saved, epoch=2), resume=True)
    discovery = copy.deepcopy(args)
    discovery.owod_discovery = True
    with pytest.raises(ValueError, match='discovery architecture'):
        validate_checkpoint(discovery, dict(args=saved, epoch=2), resume=True)
    args.owod_stage = 1
    with pytest.raises(ValueError, match='completed preceding phase'):
        validate_checkpoint(args, dict(args=saved, epoch=2))


def test_lr_quarter_changes_only_optimization_and_is_bound_to_resume(inputs):
    standard = control_args(inputs)
    original = run_benchmark.build_plan(inputs)
    inputs.control_lr_scale = 1.
    assert run_benchmark.build_plan(inputs) == original
    inputs.control_lr_scale = .25
    plan = run_benchmark.build_plan(inputs)
    quarter = configured(plan)
    assert quarter.lr == 5e-5 and quarter.lr_backbone == 5e-6
    assert quarter.epochs == quarter.lr_drop == 50
    before = list(original['runs'][0]['command'])
    for option in ('--lr', '--lr_backbone'):
        before[before.index(option) + 1] = plan['runs'][0]['command'][before.index(option) + 1]
    assert before == plan['runs'][0]['command']
    assert plan['control_optimization']['lr_scale'] == .25
    assert 'not the original optimization recipe' in plan['recipe_source']
    validate_checkpoint(quarter, dict(args=copy.deepcopy(quarter), epoch=2), resume=True)
    with pytest.raises(ValueError, match='cannot change lr'):
        validate_checkpoint(quarter, dict(args=standard, epoch=2), resume=True)


@pytest.mark.parametrize('value', [0., -1., 1.5, float('nan'), float('inf')])
def test_lr_scale_rejects_invalid_values(inputs, value):
    control_args(inputs)
    inputs.control_lr_scale = value
    with pytest.raises(ValueError, match='control-lr-scale'):
        run_benchmark.build_plan(inputs)


@pytest.mark.parametrize('change', [dict(method='discovery'), dict(method='graph-adapter'),
                                  dict(task1_only=False)])
def test_lr_scale_cannot_silently_change_other_experiments(inputs, change):
    control_args(inputs)
    inputs.control_lr_scale = .25
    for key, value in change.items():
        setattr(inputs, key, value)
    with pytest.raises(ValueError, match='control-lr-scale'):
        run_benchmark.build_plan(inputs)


@pytest.mark.parametrize('field,value', [('owod_stage', 1), ('owod_phase', 'finetune'),
    ('owod_screening', 'screen.json'), ('owod_manifest', ''), ('graph_routed_adapter', True),
    ('paper_baseline', 'ew-detr'), ('with_tree', True), ('pause_after_epochs', -1),
    ('pause_after_epochs', 51)])
def test_direct_control_pause_rejects_other_scopes(inputs, field, value):
    args = control_args(inputs)
    args.pause_after_epochs = 3
    setattr(args, field, value)
    with pytest.raises(ValueError, match='pause-after-epochs'):
        training_epoch_limit(args)


@pytest.mark.parametrize('lr_scale', [1., .25])
def test_actual_control_loop_retains_each_epoch_and_resumes_same_optimizer(inputs, monkeypatch, lr_scale):
    import main as training

    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    def distributed(args):
        args.distributed = False
        args.rank = 0
        args.world_size = 1

    updates = []

    def train(model, criterion, loader, optimizer, device, epoch, *args, **kwargs):
        assert kwargs['discovery_teacher'] is None
        optimizer.zero_grad()
        loss = model.weight.square().sum()
        loss.backward()
        optimizer.step()
        updates.append(epoch)
        return {'loss': float(loss.detach())}

    monkeypatch.setattr(training.utils, 'init_distributed_mode', distributed)
    monkeypatch.setattr(training.utils, 'get_sha', lambda: 'control fixture')
    monkeypatch.setattr(training, 'build_model', lambda args: (TinyDetector(), nn.Module(), {}))
    monkeypatch.setattr(training, 'build_dataset', lambda **kwargs: [0, 1])
    monkeypatch.setattr(training, 'get_coco_api_from_dataset', lambda dataset: None)
    monkeypatch.setattr(training, 'train_one_epoch', train)
    monkeypatch.setattr(training, 'evaluate', lambda *args, **kwargs: ({}, SimpleNamespace(coco_eval={})))
    inputs.control_lr_scale = lr_scale
    args = control_args(inputs)
    args.device = 'cpu'
    args.pause_after_epochs = 3
    training.main(copy.deepcopy(args))
    directory = Path(args.output_dir)
    checkpoint = directory / 'checkpoint.pth'
    paused = torch.load(checkpoint, weights_only=False)
    assert updates == [0, 1, 2]
    assert paused['args'].epochs == paused['args'].lr_drop == 50
    assert paused['epoch'] == 2 and paused['lr_scheduler']['last_epoch'] == 3
    assert 'discovery_ema' not in paused
    assert all((directory / f'checkpoint{epoch:04}.pth').is_file() for epoch in range(3))
    assert not (directory / 'training_complete.json').exists()
    assert json.loads((directory / 'training_paused.json').read_text())['completed_epochs'] == 3
    preserved = {epoch: (directory / f'checkpoint{epoch:04}.pth').read_bytes() for epoch in range(3)}
    resumed_args = copy.deepcopy(args)
    resumed_args.resume = str(checkpoint)
    resumed_args.pause_after_epochs = 4
    training.main(resumed_args)
    assert updates == [0, 1, 2, 3]
    restored = torch.load(checkpoint, weights_only=False)
    assert restored['epoch'] == 3 and restored['lr_scheduler']['last_epoch'] == 4
    for epoch, content in preserved.items():
        assert (directory / f'checkpoint{epoch:04}.pth').read_bytes() == content
    uninterrupted = copy.deepcopy(args)
    uninterrupted.pause_after_epochs = 4
    uninterrupted.output_dir = str(directory.parent / 'reference')
    training.main(uninterrupted)
    reference = torch.load(Path(uninterrupted.output_dir) / 'checkpoint.pth', weights_only=False)
    torch.testing.assert_close(reference['model']['weight'], restored['model']['weight'], rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(reference['optimizer']['state'][0][key],
                                   restored['optimizer']['state'][0][key], rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['standard', 'lr-quarter'])
def test_launcher_pins_control_budget_preserves_results_and_logs_failures(tmp_path, variant):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    script = tmp_path / 'tools/owod/run_control_mowodb_3090.sh'
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

    def execute(*arguments):
        trace.write_text('')
        return subprocess.run([bash, script.as_posix(), variant, *arguments], env=env,
                              capture_output=True, text=True, timeout=20)

    result = execute('--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [shlex.split(line) for line in trace.read_text().splitlines()]
    command = next(call for call in calls if 'tools/owod/run_benchmark.py' in call)
    for option, expected in [('--method', 'control'), ('--gpus', '0,1'), ('--nproc-per-node', '2'),
        ('--batch-size', '2'), ('--seed', '42'), ('--pause-after-epochs', '3'),
        ('--manifest', 'data/owod/m-owodb/split_manifest.json')]:
        assert command[command.index(option) + 1] == expected
    assert '--task1-only' in command and '--activation-checkpointing' in command
    assert ('--control-lr-scale' in command) == (variant == 'lr-quarter')
    if variant == 'lr-quarter':
        assert command[command.index('--control-lr-scale') + 1] == '0.25'
    assert not any('diagnose_control_inputs.py' in str(call) for call in calls)
    assert execute().returncode == 0
    assert 'diagnose_control_inputs.py' in trace.read_text()
    probe = next(shlex.split(line) for line in trace.read_text().splitlines() if 'diagnose_control_inputs.py' in line)
    assert probe[probe.index('--images') + 1] == '256'
    run_name = 'tree-detr-m-control-3090' if variant == 'standard' else 'tree-detr-m-control-lr-quarter-3090'
    directory = tmp_path / 'output' / run_name / 'm-owodb/seed_42'
    (directory / 'stage_0/increment').mkdir(parents=True)
    checkpoint = directory / 'stage_0/increment/checkpoint.pth'
    checkpoint.write_bytes(b'Existing checkpoint')
    (directory / 'benchmark_plan.json').write_text('{}')
    assert execute().returncode == 2
    assert not trace.read_text()
    assert checkpoint.read_bytes() == b'Existing checkpoint'
    assert execute('--resume').returncode == 0
    command = next(shlex.split(line) for line in trace.read_text().splitlines() if 'run_benchmark.py' in line)
    assert '--resume' in command and command[command.index('--pause-after-epochs') + 1] == '3'
    env['TEST_FAIL_VERIFY'] = '1'
    assert execute('--resume').returncode == 7
    assert 'run_benchmark.py' not in trace.read_text()
    env['TEST_FAIL_VERIFY'] = '0'
    env['TEST_FAIL_TRAIN'] = '1'
    assert execute('--resume').returncode == 9
    assert 'diagnose_control_inputs.py' not in trace.read_text()
    assert 'pause_after_epochs=3 schedule=50' in (tmp_path /
        f'output/{run_name}-logs/console.log').read_text()


class InputDetector(nn.Module):
    def __init__(self, ignore_input=False):
        super().__init__()
        self.ignore_input = ignore_input
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Module()
        self.transformer.encoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.transformer.encoder.layers:
            layer.norm2 = nn.Identity()
        self.register_buffer('unchanged', torch.ones(1))

    def forward(self, samples):
        value = samples.tensors.mean((-1, -2))[:, None, :]
        if self.ignore_input:
            value = torch.zeros_like(value)
        for layer in self.transformer.encoder.layers:
            value = layer.norm2(value)
        return {'pred_logits': value.expand(-1, 5, -1),
                'pred_boxes': value.mean(-1, keepdim=True).expand(-1, 5, 4)}


def image_fixture():
    return [(torch.full((3, 9 + index, 10 + index), float(index + 1)),
             {'image_id': torch.tensor(index)}) for index in range(24)]


@pytest.mark.parametrize('ignore_input', [False, True])
def test_input_probe_detects_image_blindness_and_keeps_model_state(ignore_input):
    model = InputDetector(ignore_input)
    previous = copy.deepcopy(model.state_dict())
    result = diagnostic.probe_inputs(model, image_fixture(), 'cpu')
    for comparison in ('mean_color', 'different_real_image'):
        summary = result['summary_mean_abs'][comparison]
        assert (summary['outputs']['pred_boxes'] == 0) == ignore_input
        assert (summary['encoder']['encoder_block_2'] == 0) == ignore_input
        assert (result['distributions'][comparison]['outputs']['pred_boxes']['fraction_below_1e_6'] == 1) == ignore_input
    assert model.training
    assert len(result['rows']) == 4
    assert all(row['image_id'] != row['other_image_id'] for row in result['rows'])
    for key, value in previous.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(not module._forward_hooks for module in model.modules())


def test_input_probe_uses_256_unique_training_images_and_reports_the_distribution():
    dataset = [(torch.full((3, 8, 8), float(index + 1)), {'image_id': torch.tensor(index)})
               for index in range(1024)]
    indices = diagnostic.probe_indices(len(dataset), 256)
    assert len(set(indices)) == len(indices) == 256
    assert indices[0] == 0 and max(indices) > .9 * len(dataset)
    assert diagnostic.probe_indices(len(dataset), 4) == [0, 7, 15, 23]
    result = diagnostic.probe_inputs(InputDetector(), dataset, 'cpu', indices=indices)
    assert len(result['rows']) == 256
    assert [row['image_id'] for row in result['rows']] == indices
    assert result['distributions']['mean_color']['outputs']['pred_boxes']['fraction_below_1e_6'] == 0
    assert result['distributions']['different_real_image']['outputs']['pred_boxes']['median'] > 0


def test_input_probe_cli_binds_epoch_reports_to_unchanged_checkpoints(inputs, monkeypatch):
    args = control_args(inputs)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True)
    for epoch in range(3):
        torch.save(dict(args=args, model={}, epoch=epoch), directory / f'checkpoint{epoch:04}.pth')
    content = {path.name: path.read_bytes() for path in directory.glob('*.pth')}
    monkeypatch.setattr('tools.owod.gnn_calibration.build_calibration_dataset', lambda *a: image_fixture())
    monkeypatch.setattr('tools.owod.gnn_calibration.load_detector', lambda *a: (InputDetector(), None))
    assert diagnostic.main(['--run-dir', str(directory), '--device', 'cpu', '--images', '4']) == 0
    reports = list(directory.glob('input_diagnostics/*/summary.json'))
    assert len(reports) == 1
    result = json.loads(reports[0].read_text())
    assert [row['completed_epochs'] for row in result['epochs']] == [1, 2, 3]
    assert all(len(row['rows']) == 4 for row in result['epochs'])
    for path in directory.glob('*.pth'):
        assert path.read_bytes() == content[path.name]
