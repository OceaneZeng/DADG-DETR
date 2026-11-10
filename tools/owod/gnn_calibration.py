"""Detector-derived features and empirical harm probes for OWOD GNNs."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.coco import CocoDetection, make_coco_transforms
from main import load_local_checkpoint
from models import build_model
from models.graph_local.lora import (freeze_for_class_ids, inject_decoder_lora,
                                     target_base_weights)
from models.graph_local.losses import weighted_detection_loss
import util.misc as utils


def flatten_gradients(gradients: Iterable[torch.Tensor | None],
                      parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    chunks = []
    for gradient, parameter in zip(gradients, parameters):
        chunks.append((torch.zeros_like(parameter) if gradient is None else gradient.detach()).reshape(-1))
    if not chunks:
        raise ValueError("No target gradients were provided")
    return torch.cat(chunks)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_calibration_dataset(coco_path: Path, annotation: Path,
                              lightweight: bool = False) -> CocoDetection:
    """Read train images with deterministic transforms for comparable losses."""
    return CocoDetection(
        coco_path,
        annotation,
        transforms=make_coco_transforms("val", lightweight=lightweight),
        return_masks=False,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    )


def dataset_indices_for_class(dataset: CocoDetection, class_id: int) -> list[int]:
    # Calibration annotations/ids are immutable for this dataset's lifetime.
    # Build every class index once rather than rescanning all images for each
    # source-target probe. Preserve image order and count an image only once.
    if not hasattr(dataset, '_calibration_class_indices'):
        by_class = {}
        for index, image_id in enumerate(dataset.ids):
            categories = {int(annotation['category_id'])
                          for annotation in dataset.coco.imgToAnns.get(image_id, [])}
            for category in categories:
                by_class.setdefault(category, []).append(index)
        dataset._calibration_class_indices = {category: tuple(indices)
                                              for category, indices in by_class.items()}
    return list(dataset._calibration_class_indices.get(int(class_id), ()))


def class_loader(dataset: CocoDetection, class_id: int, batch_size: int,
                 num_workers: int, max_images: int | None) -> DataLoader:
    indices = dataset_indices_for_class(dataset, class_id)
    if max_images is not None and max_images > 0:
        indices = indices[:max_images]
    if not indices:
        raise RuntimeError(f"No calibration images contain class {class_id}")
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_targets(targets: Sequence[dict], device: torch.device) -> list[dict]:
    return [{key: value.to(device) if torch.is_tensor(value) else value
             for key, value in target.items()} for target in targets]


def target_for_class(target: Mapping, class_id: int) -> dict:
    filtered = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in target.items()
    }
    labels = target["labels"]
    keep = labels == int(class_id)
    length = int(labels.numel())
    for key, value in target.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == length:
            filtered[key] = value[keep].clone()
    return filtered


def load_detector(args, checkpoint_path: Path,
                  device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    payload = load_local_checkpoint(checkpoint_path)
    saved = payload.get('args') if isinstance(payload, Mapping) else None
    # Reconstruct the actual completed detector, including persistent banks.
    model_args = saved if saved is not None else args
    if getattr(model_args, 'owod_discovery', False):
        from models.graph_local.discovery import build as build_discovery
        model, criterion, _postprocessors = build_discovery(model_args)
    else:
        model, criterion, _postprocessors = build_model(model_args)
    from models.graph_local.routed_adapter import install_specs
    install_specs(model, getattr(saved, 'graph_adapter_specs', []))
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint is not a state dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected
                  if not name.endswith(("total_params", "total_ops"))]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the requested detector configuration. "
            f"Missing={missing}; unexpected={unexpected}"
        )
    model.to(device)
    criterion.to(device)
    return model, criterion


def compute_class_gradient_sketch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataset: CocoDetection,
    class_id: int,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 4,
    max_images: int = 12,
    last_decoder_layers: int = 2,
) -> tuple[torch.Tensor, int]:
    """Average decoder-FFN gradients for one class on a frozen detector state."""
    loader = class_loader(dataset, class_id, batch_size, num_workers, max_images)
    model.eval()
    weights = target_base_weights(model, last_decoder_layers)
    # Restored frozen method backbones still need gradients for controller sketches.
    for parameter in weights:
        parameter.requires_grad_(True)
    total = None
    matched = 0
    for samples, targets in loader:
        samples = samples.to(device)
        targets = move_targets(targets, device)
        filtered = [target_for_class(target, class_id) for target in targets]
        batch_count = sum(int(target["labels"].numel()) for target in filtered)
        if batch_count == 0:
            continue
        scalar = weighted_detection_loss(
            criterion(model(samples), filtered), criterion.weight_dict)
        gradients = torch.autograd.grad(scalar, weights, allow_unused=True)
        vector = flatten_gradients(gradients, weights).detach().cpu()
        weighted = vector * batch_count
        total = weighted if total is None else total + weighted
        matched += batch_count
    if total is None or matched == 0:
        raise RuntimeError(f"No matched calibration annotations for class {class_id}")
    return total / matched, matched


def compute_gradient_sketches(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    datasets: Mapping[int, CocoDetection],
    class_ids: Iterable[int],
    device: torch.device,
    cache_dir: Path | None = None,
    reuse_cache: bool = True,
    cache_identity: Mapping[str, object] | None = None,
    progress: Callable[[str], None] = print,
    **kwargs,
) -> tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Compute or resume per-class detector gradient sketches."""
    sketches: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_config = {
        "batch_size": kwargs.get("batch_size", 1),
        "max_images": kwargs.get("max_images", 12),
        "last_decoder_layers": kwargs.get("last_decoder_layers", 2),
        **dict(cache_identity or {}),
    }
    ordered = [int(value) for value in class_ids]
    for position, class_id in enumerate(ordered, start=1):
        cache = cache_dir / f"class_{class_id}.pt" if cache_dir is not None else None
        payload = None
        if reuse_cache and cache is not None and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload is not None and payload.get("cache_config") == cache_config:
            sketch = payload["sketch"].float()
            count = int(payload["matched_annotations"])
            progress(f"Sketch [{position}/{len(ordered)}] class={class_id} cache hit")
        else:
            progress(f"Sketch [{position}/{len(ordered)}] class={class_id} computing")
            sketch, count = compute_class_gradient_sketch(
                model, criterion, datasets[class_id], class_id, device, **kwargs)
            if cache is not None:
                atomic_torch_save({"class_id": class_id, "sketch": sketch,
                                   "matched_annotations": count,
                                   "cache_config": cache_config}, cache)
        sketches[class_id] = sketch
        counts[class_id] = count
    return sketches, counts


def class_loss(model: torch.nn.Module, criterion: torch.nn.Module,
               dataset: CocoDetection, class_id: int, device: torch.device,
               batch_size: int, num_workers: int, max_images: int) -> tuple[float, int]:
    loader = class_loader(dataset, class_id, batch_size, num_workers, max_images)
    was_training = model.training
    model.eval()
    total = 0.0
    matched = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = move_targets(targets, device)
            filtered = [target_for_class(target, class_id) for target in targets]
            count = sum(int(target["labels"].numel()) for target in filtered)
            if count == 0:
                continue
            value = weighted_detection_loss(
                criterion(model(samples), filtered), criterion.weight_dict)
            total += float(value.item()) * count
            matched += count
    model.train(was_training)
    return (total / matched if matched else math.nan), matched


def run_source_probe(base_model: torch.nn.Module, criterion: torch.nn.Module,
                     dataset: CocoDetection, source_class: int,
                     device: torch.device, batch_size: int, num_workers: int,
                     max_images: int, steps: int, probe_lr: float,
                     weight_decay: float, rank: int,
                     last_decoder_layers: int, clip_max_norm: float) -> tuple[torch.nn.Module, list[float]]:
    """Fit a temporary source-class LoRA update used only to label harm."""
    if steps <= 0:
        raise ValueError("Probe steps must be positive")
    probe = copy.deepcopy(base_model)
    inject_decoder_lora(probe, rank=rank, last_n=last_decoder_layers)
    handles, parameters = freeze_for_class_ids(probe, [source_class])
    optimizer = torch.optim.AdamW(parameters, lr=probe_lr, weight_decay=weight_decay)
    loader = class_loader(dataset, source_class, batch_size, num_workers, max_images)
    losses: list[float] = []
    probe.train()
    try:
        while len(losses) < steps:
            produced = False
            for samples, targets in loader:
                produced = True
                samples = samples.to(device)
                targets = move_targets(targets, device)
                filtered = [target_for_class(target, source_class) for target in targets]
                objective = weighted_detection_loss(
                    criterion(probe(samples), filtered), criterion.weight_dict)
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(parameters, clip_max_norm)
                optimizer.step()
                losses.append(float(objective.detach().cpu()))
                if len(losses) >= steps:
                    break
            if not produced:
                raise RuntimeError(f"No probe batches for source class {source_class}")
    finally:
        for handle in handles:
            handle.remove()
    return probe, losses


def empirical_harm_row(base_losses: Mapping[int, float], probe_model: torch.nn.Module,
                       criterion: torch.nn.Module, dataset: CocoDetection,
                       source_class: int, target_class_ids: Sequence[int],
                       device: torch.device, batch_size: int, num_workers: int,
                       max_images: int) -> tuple[Dict[int, float], Dict[int, float], Dict[int, int]]:
    harm: Dict[int, float] = {}
    after_losses: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for target_class in target_class_ids:
        if int(target_class) == int(source_class):
            continue
        after, count = class_loss(
            probe_model, criterion, dataset, target_class, device,
            batch_size, num_workers, max_images)
        before = float(base_losses[int(target_class)])
        after_losses[int(target_class)] = after
        counts[int(target_class)] = count
        harm[int(target_class)] = max(0.0, after - before)
    return harm, after_losses, counts
