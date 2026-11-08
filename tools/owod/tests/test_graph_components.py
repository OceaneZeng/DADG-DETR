from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from engine import _coco_ap50_for_categories
from models.graph_local.distillation import old_class_distillation_losses
from models.graph_local.lora import (freeze_for_class_ids, inject_decoder_lora,
                                     merge_decoder_lora)
from models.graph_local.losses import local_margin_loss, projection_loss
from models.graph_local.protection import build_off_neighborhood_basis
from models.graph_local.pseudo_labels import select_teacher_pseudo_labels
from tools.owod.calibrate_interference_gnn import resolve_gnn_ablation, source_masks


def test_internal_gnn_ablations_disable_exactly_one_component():
    base = {
        "gnn_message_steps": 2,
        "gnn_ranking_margin": 0.25,
    }
    expected = {
        "full": (True, 2, 1.0),
        "no_node_encoder": (False, 2, 1.0),
        "no_message_passing": (True, 0, 1.0),
        "no_ranking_loss": (True, 2, 0.0),
    }
    for name, values in expected.items():
        config = resolve_gnn_ablation(SimpleNamespace(gnn_ablation=name, **base))
        assert (config["use_node_encoder"], config["message_steps"],
                config["ranking_weight"]) == values


def test_source_holdout_masks_entire_rows_without_label_leakage():
    class_ids = [1, 2, 3, 4, 5]
    valid = ~torch.eye(len(class_ids), dtype=torch.bool)
    train, validation, held_out = source_masks(class_ids, valid, fraction=0.4, seed=42)

    held_indices = [class_ids.index(class_id) for class_id in held_out]
    assert len(held_out) == 2
    assert not train[held_indices].any()
    assert torch.equal(validation[held_indices], valid[held_indices])
    assert not (train & validation).any()
    assert torch.equal(train | validation, valid)


def test_source_holdout_ignores_unprobed_rows():
    class_ids = [1, 2, 3, 4]
    valid = torch.zeros(4, 4, dtype=torch.bool)
    valid[0, 1:] = True
    valid[2, [0, 1, 3]] = True
    train, validation, held_out = source_masks(class_ids, valid, fraction=0.5, seed=1)

    assert len(held_out) == 1
    assert held_out[0] in {1, 3}
    assert not train[1].any() and not validation[1].any()
    assert not train[3].any() and not validation[3].any()
    assert torch.equal(train | validation, valid)


def test_category_ap50_uses_requested_category_subset():
    precision = np.full((10, 101, 3, 4, 3), -1.0)
    precision[0, :, 0, 0, 2] = 0.2
    precision[0, :, 1, 0, 2] = 0.6
    precision[0, :, 2, 0, 2] = 0.9
    evaluator = SimpleNamespace(
        eval={"precision": precision},
        params=SimpleNamespace(
            catIds=[1, 2, 3],
            iouThrs=np.linspace(0.5, 0.95, 10),
            areaRngLbl=["all", "small", "medium", "large"],
            maxDets=[1, 10, 100],
        ),
    )

    assert np.isclose(_coco_ap50_for_categories(evaluator, [1, 2]), 0.4)


def test_old_class_distillation_is_zero_at_initialization_and_detects_drift():
    teacher = {
        "pred_logits": torch.tensor([[[2.0, -1.0], [0.5, -2.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
    }
    student = {key: value.clone().requires_grad_(True) for key, value in teacher.items()}
    equal, count = old_class_distillation_losses(
        student, teacher, [0], score_threshold=0.0, max_queries_per_image=2)
    assert count == 2
    assert abs(float(equal["loss_distill_cls"])) < 1e-6
    assert abs(float(equal["loss_distill_bbox"])) < 1e-6

    shifted = {key: value.clone().requires_grad_(True) for key, value in teacher.items()}
    shifted["pred_logits"] = (shifted["pred_logits"] + 1.0).detach().requires_grad_(True)
    shifted["pred_boxes"] = (shifted["pred_boxes"] + 0.1).detach().requires_grad_(True)
    drift, _ = old_class_distillation_losses(
        shifted, teacher, [0], score_threshold=0.0, max_queries_per_image=2)
    total = drift["loss_distill_cls"] + drift["loss_distill_bbox"]
    assert float(total) > 0
    total.backward()
    assert shifted["pred_logits"].grad is not None


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 4)

    def forward(self, value):
        return self.linear2(torch.relu(self.linear1(value)))


class _TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList(
            [_TinyDecoderLayer(), _TinyDecoderLayer()])
        head = nn.Linear(4, 5)
        self.class_embed = nn.ModuleList([head, head])

    def forward(self, value):
        for layer in self.transformer.decoder.layers:
            value = layer(value)
        return self.class_embed[-1](value)


def test_local_margin_is_zero_when_margin_is_met_and_positive_otherwise():
    def matcher(_outputs, _targets):
        return [(torch.tensor([0]), torch.tensor([0]))]

    targets = [{"labels": torch.tensor([1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    good = {"pred_logits": torch.tensor([[[0.0, 3.0, 1.0]]]), "pred_boxes": boxes}
    bad = {"pred_logits": torch.tensor([[[0.0, 1.2, 1.0]]]), "pred_boxes": boxes}
    assert float(local_margin_loss(good, targets, matcher, [1, 2], margin=1.0)) == 0.0
    assert float(local_margin_loss(bad, targets, matcher, [1, 2], margin=1.0)) > 0.0


def test_off_neighborhood_basis_is_orthonormal_and_projection_has_gradients():
    sketches = {
        1: torch.arange(1, 9, dtype=torch.float32),
        2: torch.arange(8, 0, -1, dtype=torch.float32),
        10: torch.ones(8),
    }
    basis = build_off_neighborhood_basis(sketches, excluded_classes=[10], max_rank=2)
    assert basis.shape == (8, 2)
    assert torch.allclose(basis.t() @ basis, torch.eye(2), atol=1e-5)
    delta = torch.randn(8, requires_grad=True)
    loss = projection_loss(delta, basis)
    loss.backward()
    assert float(loss) >= 0.0
    assert delta.grad is not None and float(delta.grad.abs().sum()) > 0.0


def test_teacher_completion_filters_ground_truth_overlap_and_adds_old_boxes():
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.1, 0.1, 0.1, 0.1]]]),
    }
    targets = [{
        "labels": torch.tensor([2]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "area": torch.tensor([0.04]),
        "iscrowd": torch.tensor([0]),
        "size": torch.tensor([100, 100]),
    }]
    completed, counts = select_teacher_pseudo_labels(
        outputs, targets, old_class_ids=[0, 1], score_threshold=0.5,
        ground_truth_iou=0.5)
    assert counts == [1]
    assert completed[0]["labels"].tolist() == [2, 1]
    assert completed[0]["boxes"].shape[0] == 2


def test_teacher_completion_keeps_overlapping_boxes_from_different_classes():
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.5, 0.5, 0.2, 0.2]]]),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.long),
                "boxes": torch.empty(0, 4)}]
    completed, counts = select_teacher_pseudo_labels(
        outputs, targets, old_class_ids=[0, 1], score_threshold=0.5)
    assert counts == [2]
    assert completed[0]["labels"].tolist() == [0, 1]


def test_only_lora_and_current_classifier_rows_update_and_merge_is_equivalent():
    torch.manual_seed(0)
    model = _TinyDetector()
    inject_decoder_lora(model, rank=2, last_n=1)
    handles, parameters = freeze_for_class_ids(model, [3, 4])
    old_head = model.class_embed[0].weight[:3].detach().clone()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)
    loss = model(torch.randn(3, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.equal(model.class_embed[0].weight[:3], old_head)
    for name, before in frozen.items():
        assert torch.equal(dict(model.named_parameters())[name], before)

    for module in model.modules():
        if hasattr(module, "lora_b"):
            nn.init.normal_(module.lora_b)
    inputs = torch.randn(2, 4)
    before_merge = model(inputs).detach()
    assert merge_decoder_lora(model, last_n=1) == 2
    after_merge = model(inputs).detach()
    assert torch.allclose(before_merge, after_merge, atol=1e-5, rtol=1e-5)
    for handle in handles:
        handle.remove()
