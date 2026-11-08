"""Graph-local continual adaptation for Deformable DETR."""

from .gnn import (
    ClassInterferenceGNN,
    compress_gradient_sketches,
    fit_interference_gnn,
    harm_prediction_loss,
    load_gnn_checkpoint,
    save_gnn_checkpoint,
    select_gnn_neighbors,
)
from .lora import (
    LoRALinear,
    expand_classification_head,
    freeze_for_class_ids,
    freeze_for_increment,
    inject_decoder_lora,
    lora_delta_vector,
    merge_decoder_lora,
)
from .losses import local_margin_loss, projection_loss, weighted_detection_loss
from .protection import build_off_neighborhood_basis
from .pseudo_labels import select_teacher_pseudo_labels
from .distillation import old_class_distillation_losses

__all__ = [
    "LoRALinear",
    "ClassInterferenceGNN",
    "build_off_neighborhood_basis",
    "compress_gradient_sketches",
    "expand_classification_head",
    "freeze_for_class_ids",
    "freeze_for_increment",
    "fit_interference_gnn",
    "harm_prediction_loss",
    "inject_decoder_lora",
    "load_gnn_checkpoint",
    "lora_delta_vector",
    "local_margin_loss",
    "merge_decoder_lora",
    "old_class_distillation_losses",
    "projection_loss",
    "select_gnn_neighbors",
    "select_teacher_pseudo_labels",
    "save_gnn_checkpoint",
    "weighted_detection_loss",
]
