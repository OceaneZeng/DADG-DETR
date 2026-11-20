"""Frozen-teacher output preservation for incremental Deformable DETR."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def select_distillation_queries(confidence: torch.Tensor, score_threshold: float,
                               max_queries_per_image: int) -> torch.Tensor:
    """Select the teacher-confident query slots used by old-class distillation."""
    selected = torch.zeros_like(confidence, dtype=torch.bool)
    limit = max(0, int(max_queries_per_image))
    for image_index in range(confidence.shape[0]):
        candidates = torch.nonzero(
            confidence[image_index] >= float(score_threshold), as_tuple=False,
        ).flatten()
        if limit and candidates.numel() > limit:
            ranking = confidence[image_index, candidates].argsort(descending=True)
            candidates = candidates[ranking[:limit]]
        selected[image_index, candidates] = True
    return selected


def old_class_distillation_losses(student_outputs: dict[str, torch.Tensor],
                                  teacher_outputs: dict[str, torch.Tensor],
                                  old_class_ids: Iterable[int],
                                  score_threshold: float = 0.3,
                                  max_queries_per_image: int = 20,
                                  temperature: float = 1.0
                                  ) -> tuple[dict[str, torch.Tensor], int]:
    """Preserve old-class logits and boxes on teacher-confident query slots."""
    class_ids = sorted({int(value) for value in old_class_ids})
    student_logits = student_outputs["pred_logits"]
    teacher_logits = teacher_outputs["pred_logits"].detach()
    if not class_ids:
        zero = student_logits.sum() * 0.0
        return {"loss_distill_cls": zero, "loss_distill_bbox": zero}, 0
    if class_ids[0] < 0 or class_ids[-1] >= student_logits.shape[-1]:
        raise ValueError("old_class_ids contains an index outside the classifier")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")

    student_old = student_logits[..., class_ids] / temperature
    teacher_old = teacher_logits[..., class_ids] / temperature
    teacher_probability = teacher_old.sigmoid()
    confidence = teacher_probability.amax(dim=-1)
    selected = select_distillation_queries(confidence, score_threshold, max_queries_per_image)

    if not selected.any():
        zero = student_logits.sum() * 0.0
        return {"loss_distill_cls": zero, "loss_distill_bbox": zero}, 0

    # Bernoulli KL is zero when student and teacher logits agree, unlike soft-label BCE.
    student_log_probability = torch.stack(
        (F.logsigmoid(student_old), F.logsigmoid(-student_old)), dim=-1)
    teacher_bernoulli = torch.stack(
        (teacher_probability, 1.0 - teacher_probability), dim=-1)
    per_logit_kl = F.kl_div(
        student_log_probability, teacher_bernoulli, reduction="none",
    ).sum(dim=-1)
    classification = per_logit_kl[selected].mean() * (temperature ** 2)

    student_boxes = student_outputs["pred_boxes"]
    teacher_boxes = teacher_outputs["pred_boxes"].detach()
    box = F.smooth_l1_loss(student_boxes[selected], teacher_boxes[selected], reduction="mean")
    return {"loss_distill_cls": classification, "loss_distill_bbox": box}, int(selected.sum())
