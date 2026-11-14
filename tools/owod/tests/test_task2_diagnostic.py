import copy

import pytest
import torch

from models.graph_local.discovery_memory import PrototypeMemory
from tools.owod.train_task2_diagnostic import (
    make_optimizer, validate_diagnostic_resume, diagnostic_identity,
    capture_rank_state, restore_rank_state,
    diagnostic_evaluation_epochs, pending_diagnostic_evaluations, evaluate_diagnostic_milestone,
)
from tools.owod.strict_setting import validate_checkpoint


def test_bounded_memory_keeps_mature_slots_and_limits_global_recycling():
    memory = PrototypeMemory(4, capacity=3)
    memory.centers[91:] = torch.eye(4)[:3]
    memory.counts[91:] = torch.tensor([100., .2, .3])
    memory.enable_bounded(3.)
    # Many unrelated proposals in one observation must not replace the whole bank.
    for _ in range(96):
        memory.observe(torch.eye(4)[3:].repeat(4, 1), torch.full((4,), -1), torch.ones(4))
    assert memory.counts[91] == 100
    assert torch.equal(memory.centers[91], torch.eye(4)[0])
    assert memory.recycle_count == 1  # Later identical evidence matches the new slot.
    assert memory.counts[93] == pytest.approx(.3)
    restored = PrototypeMemory(4, capacity=3)
    restored.enable_bounded(3.)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()), strict=True)
    assert restored.last_recycle == memory.last_recycle
    assert restored.recycle_count == 1


def test_bounded_memory_can_eventually_recycle_a_long_stale_mature_slot():
    memory = PrototypeMemory(2, capacity=1)
    memory.centers[91] = torch.tensor([1., 0.])
    memory.counts[91] = 100
    memory.enable_bounded(3.)
    memory.clock.fill_(1022)
    evidence = (torch.tensor([[0., 1.]]), torch.tensor([-1]), torch.ones(1))
    memory.observe(*evidence)
    assert memory.counts[91] == 100
    memory.observe(*evidence)
    assert memory.counts[91] == 1 and memory.recycle_count == 1


def test_known_only_warmup_does_not_expire_v7_anonymous_memory():
    memory = PrototypeMemory(2, capacity=1)
    memory.enable_bounded(3.)
    for _ in range(40):
        memory.observe(torch.tensor([[1., 0.]]), torch.tensor([1]), torch.ones(1))
    assert memory.clock == 0
    memory.observe(torch.tensor([[float('nan'), 0.]]), torch.tensor([-1]), torch.ones(1))
    assert memory.clock == 0


def test_diagnostic_resume_is_pinned_and_never_a_formal_parent():
    payload = dict(diagnostic_only=True, diagnostic_identity={'parent': 'sha'}, policy=7, epoch=0)
    for field in ('model', 'optimizer', 'discovery_ema', 'torch_rng', 'cuda_rng',
                  'numpy_rng', 'python_rng', 'loader_rng', 'args'):
        payload[field] = {}
    validate_diagnostic_resume(payload, {'parent': 'sha'}, 7)
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(payload, {'parent': 'different'}, 7)
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(payload, {'parent': 'sha'}, 5)
    with pytest.raises(ValueError, match='Diagnostic checkpoints'):
        validate_checkpoint(None, payload)


def test_optimizer_preserves_masked_classifier_rows_with_zero_decay():
    from types import SimpleNamespace
    model = torch.nn.Module()
    model.class_embed = torch.nn.Linear(2, 2)
    args = SimpleNamespace(lr=2e-4, lr_backbone=2e-5, lr_backbone_names=['backbone'],
                           lr_linear_proj_names=['sampling_offsets'], lr_linear_proj_mult=.1,
                           class_embed_lr_mult=1., weight_decay=1e-4)
    optimizer = make_optimizer(model, args)
    before = model.class_embed.weight.detach().clone()
    model.class_embed.weight.grad = torch.zeros_like(before)
    model.class_embed.weight.grad[1] = 1
    optimizer.step()
    assert torch.equal(before[0], model.class_embed.weight[0])
    assert not torch.equal(before[1], model.class_embed.weight[1])


def test_two_gpu_plan_keeps_effective_batch_and_rejects_single_gpu_resume(tmp_path):
    files = [tmp_path / name for name in ('manifest', 'checkpoint', 'gnn')]
    for path in files:
        path.write_text('identity fixture')
    single = diagnostic_identity(*files)
    dual = diagnostic_identity(*files, world_size=2)
    assert single['batch_size'] * single['world_size'] == 2
    assert dual['batch_size'] == 1 and dual['world_size'] == 2
    assert dual['effective_batch_size'] == 2
    assert single['epochs'] == dual['epochs'] == 3  # Existing v5/v7 CLI keeps its budget.
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(dict(diagnostic_only=True, diagnostic_identity=single,
                                        policy=7, epoch=0), dual, 7)


def test_two_gpu_resume_requires_all_rank_random_states():
    identity = {'world_size': 2}
    generator = torch.Generator().manual_seed(42)
    state = capture_rank_state(generator)
    payload = dict(diagnostic_only=True, diagnostic_identity=identity, policy=7, epoch=0,
                   model={}, optimizer={}, discovery_ema={}, args={}, **state)
    with pytest.raises(ValueError, match='per-rank'):
        validate_diagnostic_resume(payload, identity, 7)
    payload['rank_states'] = [state, dict(state)]
    validate_diagnostic_resume(payload, identity, 7)
    payload['rank_states'][1].pop('loader_rng')
    with pytest.raises(ValueError, match='per-rank'):
        validate_diagnostic_resume(payload, identity, 7)


def test_rank_random_state_restores_training_and_loader_streams():
    import random
    import numpy as np
    generator = torch.Generator().manual_seed(43)
    before = capture_rank_state(generator)

    def draw():
        return (torch.rand(3), torch.rand(3, generator=generator),
                np.random.rand(3), random.random())

    expected = draw()
    draw()
    restore_rank_state(before, generator)
    actual = draw()
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert np.array_equal(expected[2], actual[2]) and expected[3] == actual[3]


def test_eight_epoch_resume_accepts_only_its_own_budget():
    state = capture_rank_state(torch.Generator())
    identity = dict(epochs=8, evaluation_epochs=[3, 8])
    payload = dict(diagnostic_only=True, diagnostic_identity=identity, policy=7, epoch=7,
                   model={}, optimizer={}, discovery_ema={}, args={}, **state)
    validate_diagnostic_resume(payload, identity, 7)
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(dict(payload, epoch=8), identity, 7)
    old = dict(epochs=3)
    legacy = dict(payload, diagnostic_identity=old, epoch=2)
    validate_diagnostic_resume(legacy, old, 7)
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(legacy, identity, 7)
    with pytest.raises(ValueError, match='identity'):
        validate_diagnostic_resume(dict(legacy, epoch=3), old, 7)


@pytest.mark.parametrize('milestones', [[], [8, 3], [3, 3, 8], [3], [3, 9], [0, 8]])
def test_diagnostic_rejects_invalid_evaluation_schedule(milestones):
    with pytest.raises(ValueError, match='evaluations'):
        diagnostic_evaluation_epochs(dict(epochs=8, evaluation_epochs=milestones))
    assert diagnostic_evaluation_epochs(dict(epochs=3)) == (3,)


def test_midpoint_measurement_preserves_training_rng_and_fixed_schedule(tmp_path):
    import json
    import random
    import numpy as np
    identity = dict(epochs=8, evaluation_epochs=[3, 8])
    generator = torch.Generator().manual_seed(43)
    before = capture_rank_state(generator)

    def draw():
        return (torch.rand(3), torch.rand(3, generator=generator),
                np.random.rand(3), random.random())

    expected = draw()
    restore_rank_state(before, generator)
    calls = []

    def measure():
        calls.append(True)
        draw()  # Simulate evaluation/probe worker creation and other random work.
        return {'owod_current_ap50': .02}

    for epoch in range(1, 9):
        evaluate_diagnostic_milestone(tmp_path, identity, epoch, generator, measure, measure)
    actual = draw()
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
    assert np.array_equal(expected[2], actual[2]) and expected[3] == actual[3]
    assert len(calls) == 4  # AP + probe, at epochs 3 and 8 only.
    assert json.loads((tmp_path / 'metrics_epoch_3.json').read_text())['owod_current_ap50'] == .02
    assert (tmp_path / 'epoch_8_learning_probe.json').is_file()
    evaluate_diagnostic_milestone(tmp_path, identity, 8, generator, measure, measure)
    assert len(calls) == 4  # Completed measurements are not repeated on resume.


def test_interrupted_midpoint_retries_before_checkpoint_advances(tmp_path):
    identity = dict(epochs=8, evaluation_epochs=[3, 8])
    generator = torch.Generator().manual_seed(42)
    before = capture_rank_state(generator)
    calls = []

    def evaluate():
        calls.append(True)
        torch.rand(3, generator=generator)
        return {'owod_current_ap50': .02}

    def interrupted_probe():
        torch.rand(3)
        raise RuntimeError('interrupted measurement')

    with pytest.raises(RuntimeError, match='interrupted'):
        evaluate_diagnostic_milestone(tmp_path, identity, 3, generator, evaluate, interrupted_probe)
    assert torch.equal(before['loader_rng'], generator.get_state())
    assert torch.equal(before['torch_rng'], torch.get_rng_state())
    assert pending_diagnostic_evaluations(tmp_path, identity, 3, with_probe=True) == [3]
    assert not (tmp_path / 'evaluation_epoch_3_complete.json').exists()
    with pytest.raises(ValueError, match='Cannot reconstruct fixed epoch 3'):
        pending_diagnostic_evaluations(tmp_path, identity, 4, with_probe=True)
    # Resume uses the still-current epoch-3 checkpoint, finishing evaluation first.
    evaluate_diagnostic_milestone(tmp_path, identity, 3, generator, evaluate, lambda: {'matched': 1})
    assert len(calls) == 2
    assert pending_diagnostic_evaluations(tmp_path, identity, 4, with_probe=True) == []
    assert pending_diagnostic_evaluations(tmp_path, identity, 8, with_probe=True) == [8]


def test_two_process_checkpoint_restores_each_rank_rng():
    from tools.owod.tests.test_runtime_distributed import _run_workers
    _run_workers(r'''
import io
import random
import numpy as np
import torch
import torch.distributed as dist
from tools.owod.train_task2_diagnostic import capture_rank_state, restore_rank_state
dist.init_process_group('gloo')
rank = dist.get_rank()
torch.manual_seed(42 + rank)
np.random.seed(42 + rank)
random.seed(42 + rank)
generator = torch.Generator().manual_seed(42 + rank)
states = [None, None]
dist.all_gather_object(states, capture_rank_state(generator))
def draw():
    return (torch.rand(3), torch.rand(3, generator=generator), np.random.rand(3), random.random())
expected = draw()
# Exercise serialization of the same list saved in the rank-zero checkpoint.
buffer = io.BytesIO()
torch.save({'rank_states': states}, buffer)
buffer.seek(0)
saved = torch.load(buffer, weights_only=False)
draw()
restore_rank_state(saved['rank_states'][rank], generator)
actual = draw()
assert torch.equal(expected[0], actual[0])
assert torch.equal(expected[1], actual[1])
assert np.array_equal(expected[2], actual[2]) and expected[3] == actual[3]
assert not torch.equal(states[0]['torch_rng'], states[1]['torch_rng'])
dist.destroy_process_group()
''')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration check')
def test_v7_real_detector_backward_and_diagnostic_restore(tmp_path):
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema, update_ema
    from tools.owod.gnn_calibration import load_detector
    config = get_args_parser().parse_args([])
    config.owod_discovery = True
    config.owod_known_class_ids = [1]
    config.owod_previous_class_ids = []
    config.owod_max_detections = 3
    config.discovery_version = 7
    config.backbone = 'resnet18'
    config.hidden_dim = 32
    config.dim_feedforward = 64
    config.enc_layers = config.dec_layers = config.graph_adapter_last_n = 1
    config.num_queries = 3
    config.skip_backbone_initialization = True
    model, criterion, _ = build(config)
    model.cuda().train()
    criterion.cuda().train()
    teacher = make_ema(model)
    optimizer = make_optimizer(model, config)
    outputs = model([torch.rand(3, 96, 96, device='cuda')])
    targets = [dict(labels=torch.tensor([1], device='cuda'),
                    boxes=torch.tensor([[.5, .5, .3, .3]], device='cuda'))]
    losses = criterion(outputs, targets)
    total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
    assert torch.isfinite(total)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    model.observe_training(criterion.evidence)
    update_ema(teacher, model)
    path = tmp_path / 'diagnostic.pth'
    torch.save(dict(diagnostic_only=True, model=model.state_dict(), args=config), path)
    restored, _ = load_detector(config, path, torch.device('cuda'))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
