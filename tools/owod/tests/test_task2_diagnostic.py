import copy

import pytest
import torch

from models.graph_local.discovery_memory import PrototypeMemory
from tools.owod.train_task2_diagnostic import make_optimizer, validate_diagnostic_resume
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
