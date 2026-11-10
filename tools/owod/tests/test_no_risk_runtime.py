"""A real no-controller plan must omit calibration without weakening lineage."""
import copy
from types import SimpleNamespace

import pytest

from tools.owod import run_benchmark, graph_adaptation
from tools.owod.gnn_calibration import dataset_indices_for_class
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.validate_discovery import validate_plan_arguments
from tools.owod.tests.test_discovery_v7 import inputs, configured


def test_disabled_risk_preserves_seven_phases_and_skips_calibration_and_sketches(inputs, tmp_path, monkeypatch):
    inputs.task1_only = False
    learned = run_benchmark.build_plan(inputs)
    assert len([run for run in learned['runs'] if run.get('kind') == 'calibration']) == 3
    inputs.graph_risk_mode = 'disabled'
    inputs.graph_adapter_compute = 'batched'
    plan = run_benchmark.build_plan(inputs)
    assert len(plan['runs']) == 7
    import tools.owod.gnn_calibration as calibration
    def reject(*args, **kwargs):
        raise AssertionError('No-risk chain must not compute sketches or load a GNN')
    monkeypatch.setattr(calibration, 'compute_gradient_sketches', reject)
    monkeypatch.setattr(graph_adaptation, 'load_gnn_checkpoint', reject)
    previous = None
    specs = []
    for run in plan['runs']:
        assert run.get('kind') != 'calibration'
        assert '--graph-gnn-checkpoint' not in run['command']
        args = configured(dict(runs=[run]))
        if previous is not None:
            validate_checkpoint(args, previous)
        if args.owod_stage > 0 and args.owod_phase == 'increment':
            parent = tmp_path / f'completed_parent_{args.owod_stage}.pth'
            parent.write_bytes(b'unit fixture, never loaded')
            spec = graph_adaptation.build_stage_spec(None, None, args, parent)
            assert spec['sketch_counts'] == {} and spec['gnn_sha256'] is None
            assert spec['score_semantics'] == 'disabled_no_interference_controller'
            specs.append(spec)
        args.graph_adapter_specs = copy.deepcopy(specs)
        assert args.graph_risk_beta == 0.
        assert args.graph_adapter_compute == 'batched'
        if args.owod_stage > 0:
            assert args.old_class_distillation
        previous = dict(args=copy.deepcopy(args), epoch=args.epochs - 1)
        validate_checkpoint(args, previous, resume=True)
        validate_plan_arguments(run, previous['args'])
    damaged = copy.deepcopy(previous)
    damaged['args'].graph_adapter_specs[-1]['risk'][0][0] = .1
    with pytest.raises(ValueError, match='controller state'):
        validate_checkpoint(args, damaged, resume=True)


@pytest.mark.parametrize('field,other', [('graph_risk_mode', 'disabled'), ('graph_adapter_compute', 'batched')])
def test_old_checkpoint_defaults_and_mode_changes_are_checked(inputs, field, other):
    plan = run_benchmark.build_plan(inputs)
    args = configured(plan)
    old = copy.deepcopy(args)
    del old.graph_risk_mode, old.graph_adapter_compute
    checkpoint = dict(args=old, epoch=7)
    validate_checkpoint(args, checkpoint, resume=True)
    validate_plan_arguments(plan['runs'][0], old)
    changed = copy.deepcopy(args)
    setattr(changed, field, other)
    for resume in (False, True):
        with pytest.raises(ValueError, match=field):
            validate_checkpoint(changed, checkpoint, resume=resume)
    inputs.graph_risk_mode = 'learned'
    inputs.graph_adapter_compute = 'loop'
    assert run_benchmark.build_plan(inputs) == plan


def test_disabled_risk_rejects_control_and_gnn_input(inputs):
    args = configured(run_benchmark.build_plan(inputs))
    args.graph_risk_mode = 'disabled'
    args.graph_gnn_checkpoint = 'must_not_be_used.pt'
    with pytest.raises(ValueError, match='cannot consume'):
        configure_run(args)
    inputs.graph_risk_mode = 'disabled'
    inputs.method = 'control'
    with pytest.raises(ValueError, match='require graph-routed'):
        run_benchmark.build_plan(inputs)


def test_calibration_class_index_cache_keeps_order_and_deduplicates_images():
    class CountingAnnotations(dict):
        calls = 0
        def get(self, *args):
            self.calls += 1
            return super().get(*args)
    annotations = CountingAnnotations({
        9: [{'category_id': 2}, {'category_id': 2}, {'category_id': 1}],
        3: [{'category_id': 1}], 7: []})
    dataset = SimpleNamespace(ids=[9, 7, 3], coco=SimpleNamespace(imgToAnns=annotations))
    assert dataset_indices_for_class(dataset, 1) == [0, 2]
    assert dataset_indices_for_class(dataset, 2) == [0]
    assert dataset_indices_for_class(dataset, 99) == []
    assert annotations.calls == 3
    detached = dataset_indices_for_class(dataset, 1)
    detached.clear()
    assert dataset_indices_for_class(dataset, 1) == [0, 2]


def test_profiling_is_dispatched_without_changing_resume_plan(tmp_path, monkeypatch):
    directory = tmp_path / 'runs/m-owodb/seed_42/stage_0/increment'
    plan = dict(protocol='m-owodb', runs=[dict(stage=0, phase='increment', epochs=50,
        output_dir=str(directory), command=['python', 'main.py', '--epochs', '50'])])
    monkeypatch.setattr(run_benchmark, 'build_plan', lambda args: plan)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'checkpoint.pth').write_bytes(b'unit runner fixture, never loaded')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_benchmark.subprocess, 'run', execute)
    common = ['--manifest', str(tmp_path / 'manifest.json'), '--recipe', 'ow-detr-paper',
              '--output-dir', str(tmp_path / 'runs'), '--method', 'discovery', '--task1-only']
    assert run_benchmark.main(common) == 0
    plan_path = directory.parents[1] / 'benchmark_plan.json'
    before = plan_path.read_bytes()
    assert run_benchmark.main(common + ['--resume', '--profile-runtime-steps', '5',
                                        '--profile-runtime-warmup', '1']) == 0
    assert plan_path.read_bytes() == before
    assert '--profile-runtime-steps' not in commands[0]
    assert commands[1][-4:] == ['--profile-runtime-steps', '5', '--profile-runtime-warmup', '1']
