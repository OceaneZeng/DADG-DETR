"""Regression evidence for bounded memory and consistent old/unknown targets."""
import copy

import pytest
import torch
from torch.nn import functional as F

from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.discovery import DiscoveryCriterion, balanced_route_loss
from models.graph_local.discovery_memory import PrototypeMemory
from models.graph_local.tests.test_discovery import args, evidence


def test_full_memory_recycles_stale_slot_without_merging_or_growing():
    memory = PrototypeMemory(4, capacity=1, adaptive=True)
    first, second = torch.eye(4)[:2]
    observe = lambda feature: memory.observe(feature[None], torch.tensor([-1]), torch.tensor([1.]))
    assert observe(first) == [0]
    assert observe(second) == []  # Recent prototype is protected.
    for _ in range(31):
        observe(second)
    assert torch.equal(memory.centers[91], second)
    assert memory.counts[91] == 1 and memory.variance[91].count_nonzero() == 0
    assert memory.centers.shape == (92, 4)
    restored = PrototypeMemory(4, capacity=1, adaptive=True)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()), strict=True)
    assert torch.equal(restored.last_seen, memory.last_seen)
    assert restored.clock == memory.clock
    for instance in (memory, restored):
        instance.observe(first[None], torch.tensor([-1]), torch.tensor([1.]))
    for key, value in memory.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def test_nonfinite_evidence_cannot_corrupt_memory():
    memory = PrototypeMemory(4, capacity=1, adaptive=True)
    memory.observe(torch.eye(4)[:2], torch.tensor([-1, -1]), torch.tensor([float('nan'), float('inf')]))
    assert memory.counts.count_nonzero() == 0


def test_router_unknown_gradient_does_not_vanish_with_background_count():
    def gradient(background_count):
        labels = torch.tensor([[1, 2] + [0] * background_count])
        logits = torch.zeros(1, labels.numel(), 3, requires_grad=True)
        weights = torch.ones_like(labels, dtype=torch.float)
        weights[0, 1] = .2
        values = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        balanced_route_loss(values, labels, weights).backward()
        return logits.grad[0, 1]
    assert torch.allclose(gradient(1), gradient(98))
    assert gradient(98)[2] < 0


def test_frozen_old_teacher_prevents_old_object_from_unknown_supervision():
    outputs, targets = evidence()
    configuration = args()
    configuration.discovery_version = 6
    configuration.owod_known_class_ids = [1, 2]
    configuration.owod_previous_class_ids = [2]
    base = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    base.active_class_ids = [1, 2]
    criterion = DiscoveryCriterion(base, None, configuration)
    teacher = dict(pred_logits=torch.full((1, 3, 7), -12.),
                   pred_boxes=outputs['pred_boxes'].detach().clone())
    teacher['pred_logits'][0, 1, 2] = 12.
    outputs['discovery_old_teacher'] = teacher
    losses = criterion(outputs, targets)
    assert losses['discovery_pseudo_count'] == 0
    assert sorted(criterion.evidence[1].tolist()) == [1, 2]
    losses['loss_unknown'].backward()
    assert outputs['pred_novelty'].grad[0, 1] > 0  # Known old, never anonymous.
    assert targets[0]['labels'].tolist() == [1]  # Source annotations stay intact.
    del outputs['discovery_old_teacher']
    with pytest.raises(ValueError, match='frozen old-class teacher'):
        criterion(outputs, targets)


def test_v5_memory_state_shape_and_refusal_remain_compatible():
    memory = PrototypeMemory(4, 1)
    assert set(memory.state_dict()) == {'centers', 'counts', 'variance'}
    memory.observe(torch.eye(4)[:1], torch.tensor([-1]), torch.tensor([1.]))
    assert memory.observe(torch.eye(4)[1:2], torch.tensor([-1]), torch.tensor([1.])) == []


def test_checkpoint_restore_skips_missing_pretraining_without_changing_batchnorm(tmp_path):
    from main import get_args_parser
    from models.graph_local.discovery import build
    from models.backbone import FrozenBatchNorm2d
    from tools.owod.gnn_calibration import load_detector
    configuration = get_args_parser().parse_args([])
    configuration.owod_discovery = True
    configuration.owod_known_class_ids = [1]
    configuration.owod_max_detections = 3
    configuration.backbone = 'resnet18'
    configuration.backbone_weights = str(tmp_path / 'missing-original-pretraining.pth')
    configuration.hidden_dim = 32
    configuration.dim_feedforward = 64
    configuration.enc_layers = configuration.dec_layers = configuration.graph_adapter_last_n = 1
    configuration.num_queries = 3
    configuration.skip_backbone_initialization = True
    model, _, _ = build(configuration)
    configuration.skip_backbone_initialization = False
    checkpoint = tmp_path / 'detector.pth'
    torch.save({'args': configuration, 'model': model.state_dict()}, checkpoint)
    restored, _ = load_detector(configuration, checkpoint, torch.device('cpu'))
    assert any(isinstance(m, FrozenBatchNorm2d) for m in restored.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in restored.modules())
    assert not configuration.skip_backbone_initialization
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
