"""Low-rank updates for the final Deformable-DETR decoder FFNs."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


TARGET_ATTRIBUTES = ("linear1", "linear2")


def _detector(model: nn.Module) -> nn.Module:
    return model.detr if hasattr(model, "detr") else model


def target_decoder_layers(model: nn.Module, last_n: int = 2) -> Sequence[nn.Module]:
    layers = _detector(model).transformer.decoder.layers
    if last_n <= 0 or last_n > len(layers):
        raise ValueError(f"last_n must be in [1, {len(layers)}], got {last_n}")
    return layers[-last_n:]


def open_decoder_and_bbox(model: nn.Module):
    """Train the base decoder and bbox head while preserving adapter and class masks."""
    detector = _detector(model)
    for name, parameter in detector.transformer.decoder.named_parameters():
        if not any(part in name for part in ('graph_adapter', 'discovery_adapter', 'class_embed')):
            parameter.requires_grad_(True)
    detector.bbox_embed.requires_grad_(True)


def target_base_weights(model: nn.Module, last_n: int = 2) -> List[nn.Parameter]:
    """Return the dense target weights in a stable layer/linear order."""
    weights: List[nn.Parameter] = []
    for layer in target_decoder_layers(model, last_n):
        for attr in TARGET_ATTRIBUTES:
            module = getattr(layer, attr)
            if isinstance(module, LoRALinear):
                module = module.base
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at decoder.{attr}, got {type(module)!r}")
            weights.append(module.weight)
    return weights


class LoRALinear(nn.Module):
    """Frozen linear layer plus a mergeable low-rank weight delta."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
        return self.base


def inject_decoder_lora(model: nn.Module, rank: int = 8, last_n: int = 2,
                        alpha: float | None = None, dropout: float = 0.0
                        ) -> List[LoRALinear]:
    """Wrap both FFN matrices in each of the final ``last_n`` decoder layers."""
    wrappers: List[LoRALinear] = []
    for layer in target_decoder_layers(model, last_n):
        for attr in TARGET_ATTRIBUTES:
            current = getattr(layer, attr)
            if isinstance(current, LoRALinear):
                raise RuntimeError(f"Decoder attribute {attr} already contains LoRA")
            if not isinstance(current, nn.Linear):
                raise TypeError(f"Expected nn.Linear at decoder.{attr}, got {type(current)!r}")
            wrapped = LoRALinear(current, rank=rank, alpha=alpha, dropout=dropout)
            setattr(layer, attr, wrapped)
            wrappers.append(wrapped)
    return wrappers


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def lora_delta_vector(model: nn.Module) -> torch.Tensor:
    deltas = [module.delta_weight().reshape(-1) for module in iter_lora_modules(model)]
    if not deltas:
        raise RuntimeError("No LoRA modules are attached")
    return torch.cat(deltas)


def merge_decoder_lora(model: nn.Module, last_n: int = 2) -> int:
    """Merge all target LoRA modules and restore plain ``nn.Linear`` layers."""
    merged = 0
    for layer in target_decoder_layers(model, last_n):
        for attr in TARGET_ATTRIBUTES:
            current = getattr(layer, attr)
            if isinstance(current, LoRALinear):
                setattr(layer, attr, current.merge())
                merged += 1
    return merged


def expand_classification_head(model: nn.Module, new_num_classes: int,
                               new_bias: float = -4.59511985013459,
                               initialization: str = "mean_old"
                               ) -> Tuple[int, int]:
    """Append deterministic classifier rows while preserving shared head modules.

    ``mean_old`` keeps the new row deterministic while giving the new-class
    loss a non-zero path into shared decoder features.  A zero row is a valid
    identity-style initialization, but its first classification gradient does
    not inform the target FFN weights, which makes a new-class interference
    sketch unnecessarily dominated by box losses.
    """
    detector = _detector(model)
    old_heads = list(detector.class_embed)
    if not old_heads:
        raise RuntimeError("Detector has no classification heads")
    old_num_classes = old_heads[0].out_features
    if new_num_classes <= old_num_classes:
        raise ValueError(
            f"new_num_classes must exceed {old_num_classes}, got {new_num_classes}"
        )

    replacements = {}
    expanded = []
    for old in old_heads:
        if old.out_features != old_num_classes:
            raise ValueError("All classification heads must have the same output size")
        key = id(old)
        if key not in replacements:
            new = nn.Linear(old.in_features, new_num_classes, bias=old.bias is not None)
            new = new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:old_num_classes].copy_(old.weight)
            if initialization == "mean_old":
                new.weight[old_num_classes:].copy_(
                    old.weight.mean(dim=0, keepdim=True).expand(
                        new_num_classes - old_num_classes, -1))
            elif initialization == "zero":
                new.weight[old_num_classes:].zero_()
            else:
                raise ValueError(
                    f"Unknown classifier-row initialization: {initialization}")
            if old.bias is not None:
                new.bias.fill_(new_bias)
                new.bias[:old_num_classes].copy_(old.bias)
            replacements[key] = new
        expanded.append(replacements[key])

    detector.class_embed = nn.ModuleList(expanded)
    if detector.two_stage:
        detector.transformer.decoder.class_embed = detector.class_embed
    return old_num_classes, new_num_classes


def _row_mask(size: int, trainable_class_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(size, device=device)
    ids = torch.as_tensor(sorted({int(value) for value in trainable_class_ids}),
                          dtype=torch.long, device=device)
    ids = ids[(ids >= 0) & (ids < size)]
    if ids.numel():
        mask[ids] = 1.0
    return mask


def freeze_for_increment(model: nn.Module, old_num_classes: int) -> Tuple[List, List[nn.Parameter]]:
    """Freeze the detector except LoRA factors and newly appended classifier rows.

    PyTorch optimizers operate on whole tensors, so classifier hooks zero the
    old-row gradients. The returned handles must stay alive through training.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable: List[nn.Parameter] = []
    for module in iter_lora_modules(model):
        module.lora_a.requires_grad_(True)
        module.lora_b.requires_grad_(True)
        trainable.extend([module.lora_a, module.lora_b])

    handles = []
    seen = set()
    for head in _detector(model).class_embed:
        if id(head) in seen:
            continue
        seen.add(id(head))
        if old_num_classes >= head.out_features:
            raise ValueError("Classifier has no appended rows to train")
        row_mask = _row_mask(head.out_features, range(old_num_classes, head.out_features),
                             head.weight.device)
        head.weight.requires_grad_(True)
        handles.append(head.weight.register_hook(lambda grad, m=row_mask: grad * m[:, None]))
        trainable.append(head.weight)
        if head.bias is not None:
            head.bias.requires_grad_(True)
            handles.append(head.bias.register_hook(lambda grad, m=row_mask: grad * m))
            trainable.append(head.bias)
    return handles, trainable


def freeze_for_class_ids(model: nn.Module, trainable_class_ids: Iterable[int],
                         train_detection_heads: bool = False
                         ) -> Tuple[List, List[nn.Parameter]]:
    """Freeze detector weights and expose only selected classifier rows.

    COCO/M-OWODB keeps the original category IDs (including gaps), so an
    increment cannot be represented by a contiguous ``old_num_classes``
    boundary.  This helper is the row-mask equivalent for arbitrary IDs.
    LoRA factors, when attached, remain trainable as the shared adaptation
    path. With train_detection_heads, both complete detection heads are
    trainable instead of masking classifier rows; the feature extractor stays
    frozen except for LoRA.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable: List[nn.Parameter] = []
    for module in iter_lora_modules(model):
        module.lora_a.requires_grad_(True)
        module.lora_b.requires_grad_(True)
        trainable.extend([module.lora_a, module.lora_b])

    if train_detection_heads:
        detector = _detector(model)
        seen_parameters = {id(parameter) for parameter in trainable}
        for heads in (detector.class_embed, detector.bbox_embed):
            for parameter in heads.parameters():
                parameter.requires_grad_(True)
                if id(parameter) not in seen_parameters:
                    seen_parameters.add(id(parameter))
                    trainable.append(parameter)
        return [], trainable

    handles = []
    seen = set()
    requested = {int(value) for value in trainable_class_ids}
    for head in _detector(model).class_embed:
        if id(head) in seen:
            continue
        seen.add(id(head))
        valid = requested & set(range(head.out_features))
        if not valid:
            raise ValueError("No requested classifier IDs fit this head")
        row_mask = _row_mask(head.out_features, valid, head.weight.device)
        head.weight.requires_grad_(True)
        handles.append(head.weight.register_hook(lambda grad, m=row_mask: grad * m[:, None]))
        trainable.append(head.weight)
        if head.bias is not None:
            head.bias.requires_grad_(True)
            handles.append(head.bias.register_hook(lambda grad, m=row_mask: grad * m))
            trainable.append(head.bias)
    return handles, trainable
