#!/usr/bin/env python
"""Calibrate a trainable class-interference GNN on an earlier OWOD stage."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import get_args_parser
from models.graph_local.gnn import (ClassInterferenceGNN,
                                    compress_gradient_sketches,
                                    fit_interference_gnn, save_gnn_checkpoint)
from models.graph_local.lora import lora_delta_vector
from tools.owod.gnn_calibration import (build_calibration_dataset, class_loss,
                                        compute_gradient_sketches, atomic_torch_save,
                                        empirical_harm_row, load_detector,
                                        run_source_probe, set_seed)
from tools.owod.protocol import stage_files, file_sha256
from main import load_local_checkpoint
from util.experiment_log import start_file_logging, stop_file_logging
import util.misc as utils


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, parents=[get_args_parser()], conflict_handler="resolve")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", default=0, type=int,
                        help="completed stage whose train labels supervise the GNN")
    parser.add_argument("--output-dir", "--output_dir", required=True, type=Path)
    parser.add_argument("--feature-dim", default=128, type=int)
    parser.add_argument("--sketch-max-images", default=12, type=int)
    parser.add_argument("--probe-max-images", default=12, type=int)
    parser.add_argument("--probe-steps", default=3, type=int)
    parser.add_argument("--probe-lr", default=1e-4, type=float)
    parser.add_argument("--probe-weight-decay", default=1e-4, type=float)
    parser.add_argument("--probe-rank", default=8, type=int)
    parser.add_argument("--source-limit", default=0, type=int,
                        help="probe only the first N sources for a smoke test; 0 means all")
    parser.add_argument("--last-decoder-layers", default=2, type=int)
    parser.add_argument("--gnn-hidden-dim", default=64, type=int)
    parser.add_argument("--gnn-message-steps", default=2, type=int)
    parser.add_argument("--gnn-dropout", default=0.0, type=float)
    parser.add_argument(
        "--gnn-ablation",
        choices=("full", "no_node_encoder", "no_message_passing", "no_ranking_loss"),
        default="full",
        help="remove exactly one GNN component for the primary ablation table",
    )
    parser.add_argument("--gnn-ranking-margin", default=0.25, type=float)
    parser.add_argument("--gnn-epochs", default=400, type=int)
    parser.add_argument("--gnn-lr", default=1e-3, type=float)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    parser.add_argument("--validation-k", default=5, type=int)
    parser.add_argument("--force", action="store_true",
                        help="discard reusable per-class sketch/probe caches")
    parser.add_argument("--allow-unverified-protocol", action="store_true",
                        help="internal pilot only; marks the run as not paper-comparable")
    parser.set_defaults(dataset_file="coco", masks=False, cache_mode=False,
                        two_stage=False, num_classes=91)
    return parser


def resolve_gnn_ablation(args: argparse.Namespace) -> dict[str, object]:
    """Translate one named ablation into explicit, checkpointed settings."""
    return {
        "name": args.gnn_ablation,
        "use_node_encoder": args.gnn_ablation != "no_node_encoder",
        "message_steps": (0 if args.gnn_ablation == "no_message_passing"
                          else int(args.gnn_message_steps)),
        "ranking_weight": (0.0 if args.gnn_ablation == "no_ranking_loss" else 1.0),
        "ranking_margin": float(args.gnn_ranking_margin),
    }


def ablation_suffix(name: str) -> str:
    return "" if name == "full" else f"_{name}"


def source_masks(class_ids: list[int], valid: torch.Tensor, fraction: float,
                 seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")
    eligible = [index for index in range(len(class_ids)) if valid[index].any()]
    if len(eligible) < 2:
        raise ValueError("At least two probed source rows are required for validation")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator).tolist()
    count = min(len(eligible) - 1, max(1, round(len(eligible) * fraction)))
    held_out = sorted(class_ids[eligible[offset]] for offset in order[:count])
    held_indices = {class_ids.index(class_id) for class_id in held_out}
    train = valid.clone()
    validation = torch.zeros_like(valid)
    for index in held_indices:
        validation[index] = valid[index]
        train[index] = False
    return train, validation, held_out


def prediction_metrics(model: ClassInterferenceGNN, stage: dict,
                       mask: torch.Tensor, k: int) -> dict:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = model(stage["features"].to(device))["edge_prob"].cpu()
    harm = stage["harm"].float()
    maes = []
    overlaps = []
    correlations = []
    for source in range(harm.shape[0]):
        valid = mask[source]
        if not valid.any():
            continue
        truth = harm[source][valid]
        prediction = scores[source][valid]
        truth = truth / truth.max().clamp_min(1e-8)
        maes.append(float((prediction - truth).abs().mean()))
        top = min(max(1, k), truth.numel())
        truth_top = set(torch.topk(truth, top).indices.tolist())
        predicted_top = set(torch.topk(prediction, top).indices.tolist())
        overlaps.append(len(truth_top & predicted_top) / top)
        if truth.numel() > 1 and float(truth.std()) > 0 and float(prediction.std()) > 0:
            centered_truth = truth - truth.mean()
            centered_prediction = prediction - prediction.mean()
            correlations.append(float(
                (centered_truth * centered_prediction).sum() /
                (centered_truth.square().sum().sqrt() *
                 centered_prediction.square().sum().sqrt()).clamp_min(1e-8)))
    return {
        "source_rows": len(maes),
        "normalized_mae": sum(maes) / max(1, len(maes)),
        f"top_{k}_overlap": sum(overlaps) / max(1, len(overlaps)),
        "pearson": sum(correlations) / max(1, len(correlations)),
    }


def main() -> int:
    args = get_parser().parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing detector checkpoint: {args.checkpoint}")
    if not args.manifest.is_file():
        raise SystemExit(f"Missing OWOD manifest: {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ablation_suffix(args.gnn_ablation)
    write_json(args.output_dir / f"run_config{suffix}.json", {
        "runner": "tools/owod/calibrate_interference_gnn.py",
        "arguments": vars(args),
        "argv": sys.argv,
        "git": utils.get_sha(),
    })
    log_path = (Path(args.log_file).resolve() if args.log_file
                else args.output_dir / f"calibration{suffix}.log")
    args.log_file = str(log_path)
    log_state = start_file_logging(args, is_main_process=True)
    try:
        return run(args)
    finally:
        stop_file_logging(log_state)


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    ablation = resolve_gnn_ablation(args)
    suffix = ablation_suffix(args.gnn_ablation)
    manifest = read_json(args.manifest)
    stages = manifest.get("stages", [])
    if args.stage < 0 or args.stage >= len(stages):
        raise ValueError(f"stage {args.stage} is outside the manifest")
    class_ids = [int(value) for value in stages[args.stage]["active_classes"]]
    if args.source_limit < 0:
        raise ValueError("source-limit must be non-negative")
    probe_sources = class_ids[:args.source_limit] if args.source_limit else class_ids
    if len(probe_sources) < 2:
        raise ValueError("Calibration requires at least two source classes")
    checked_manifest, files = stage_files(
        args.manifest, args.stage, allow_unverified=args.allow_unverified_protocol)
    if getattr(args, 'owod_screening', ''):
        from tools.owod.screening_data import screening_files
        checked_manifest, files = screening_files(args.manifest, Path(args.owod_screening), args.stage)
        checked_manifest = {**checked_manifest, 'paper_comparable': False,
                            'validation_mode': 'training_only_screen'}
    annotation = files["train"]
    checkpoint_payload = load_local_checkpoint(args.checkpoint)
    checkpoint_args = checkpoint_payload.get('args')
    screen_hash = file_sha256(Path(args.owod_screening)) if getattr(args, 'owod_screening', '') else None
    if getattr(checkpoint_args, 'owod_screening_sha256', None) != screen_hash:
        raise ValueError('Calibration cannot mix screening and full-data checkpoints')
    expected_phase = 'increment' if args.stage == 0 else 'finetune'
    if (checkpoint_args is None
            or getattr(checkpoint_args, 'owod_manifest_sha256', None) != file_sha256(args.manifest)
            or getattr(checkpoint_args, 'owod_stage', None) != args.stage
            or getattr(checkpoint_args, 'owod_phase', None) != expected_phase
            or checkpoint_payload.get('epoch') != checkpoint_args.epochs - 1):
        raise ValueError('GNN calibration requires the completed official preceding task checkpoint')
    identity = {'checkpoint_sha256': file_sha256(args.checkpoint),
                'manifest_sha256': file_sha256(args.manifest),
                'annotation_sha256': file_sha256(annotation)}
    if screen_hash:
        identity['screening_sha256'] = screen_hash
    if args.force:
        print("--force ignores existing sketch and harm-row cache files")

    device = torch.device(args.device)
    def clock():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()
    started = clock()
    timing = {}
    cache_hits = dict(sketches=0, base_losses=0, harm_rows=0)
    print(json.dumps({
        "method": "empirical_train_loss_interference",
        "stage": args.stage,
        "classes": class_ids,
        "probe_sources": probe_sources,
        "checkpoint": str(args.checkpoint),
        "annotation": str(annotation),
        "device": str(device),
        "deterministic_calibration_transforms": True,
        "uses_validation_labels": False,
        "gnn_ablation": ablation,
        "protocol_validation": checked_manifest.get("validation_mode", "official"),
        "paper_comparable": bool(checked_manifest.get("paper_comparable", False)),
    }, indent=2))
    dataset = build_calibration_dataset(Path(checked_manifest['image_root']), annotation,
                                        lightweight=args.lightweight)
    model, criterion = load_detector(args, args.checkpoint, device)
    criterion.active_class_ids = class_ids
    now = clock()
    timing['dataset_and_detector_load'] = now - started
    phase_start = now

    sketch_cache = args.output_dir / "sketches"
    def sketch_progress(message):
        if 'cache hit' in message:
            cache_hits['sketches'] += 1
        print(message)
    sketches, sketch_counts = compute_gradient_sketches(
        model, criterion, {class_id: dataset for class_id in class_ids}, class_ids,
        device, cache_dir=sketch_cache, reuse_cache=not args.force,
        progress=sketch_progress,
        cache_identity={**identity, "checkpoint": str(args.checkpoint),
                        "annotation": str(annotation)},
        batch_size=args.batch_size,
        num_workers=args.num_workers, max_images=args.sketch_max_images,
        last_decoder_layers=args.last_decoder_layers)
    ordered_ids, features = compress_gradient_sketches(sketches, args.feature_dim)
    now = clock()
    timing['gradient_sketches'] = now - phase_start
    phase_start = now

    base_loss_path = args.output_dir / "base_class_losses.json"
    base_losses: dict[int, float] = {}
    base_counts: dict[int, int] = {}
    base_loss_config = {
        **identity,
        "checkpoint": str(args.checkpoint),
        "annotation": str(annotation),
        "batch_size": args.batch_size,
        "max_images": args.probe_max_images,
    }
    if base_loss_path.is_file() and not args.force:
        saved = read_json(base_loss_path)
        if saved.get("cache_config") == base_loss_config:
            base_losses = {int(key): float(value) for key, value in saved["losses"].items()}
            base_counts = {int(key): int(value) for key, value in saved["matched_annotations"].items()}
    for position, class_id in enumerate(class_ids, start=1):
        if class_id in base_losses:
            cache_hits['base_losses'] += 1
            print(f"Base loss [{position}/{len(class_ids)}] class={class_id} cache hit")
            continue
        print(f"Base loss [{position}/{len(class_ids)}] class={class_id} computing")
        value, count = class_loss(
            model, criterion, dataset, class_id, device, args.batch_size,
            args.num_workers, args.probe_max_images)
        base_losses[class_id] = value
        base_counts[class_id] = count
        write_json(base_loss_path, {
            "losses": base_losses, "matched_annotations": base_counts,
            "annotation": str(annotation), "deterministic_transforms": True,
            "cache_config": base_loss_config,
        })

    now = clock()
    timing['base_class_losses'] = now - phase_start
    phase_start = now
    harm_dir = args.output_dir / "harm_rows"
    harm_dir.mkdir(parents=True, exist_ok=True)
    harm_rows: dict[int, dict[int, float]] = {}
    harm_config = {
        **base_loss_config,
        "probe_steps": args.probe_steps,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "probe_rank": args.probe_rank,
        "last_decoder_layers": args.last_decoder_layers,
        "clip_max_norm": args.clip_max_norm,
    }
    for position, source_class in enumerate(probe_sources, start=1):
        row_path = harm_dir / f"source_{source_class}.json"
        if row_path.is_file() and not args.force:
            saved = read_json(row_path)
            if saved.get("cache_config") == harm_config:
                cache_hits['harm_rows'] += 1
                harm_rows[source_class] = {
                    int(key): float(value) for key, value in saved["harm"].items()}
                print(f"Harm probe [{position}/{len(probe_sources)}] source={source_class} cache hit")
                continue
        print(f"Harm probe [{position}/{len(probe_sources)}] source={source_class} computing")
        set_seed(args.seed + source_class)
        probe, probe_losses = run_source_probe(
            model, criterion, dataset, source_class, device, args.batch_size,
            args.num_workers, args.probe_max_images, args.probe_steps,
            args.probe_lr, args.probe_weight_decay, args.probe_rank,
            args.last_decoder_layers, args.clip_max_norm)
        harm, after_losses, counts = empirical_harm_row(
            base_losses, probe, criterion, dataset, source_class, class_ids,
            device, args.batch_size, args.num_workers, args.probe_max_images)
        probe_delta_norm = float(lora_delta_vector(probe).detach().norm().cpu())
        if probe_delta_norm == 0.0:
            print(f"WARNING: source={source_class} produced a zero LoRA update")
        harm_rows[source_class] = harm
        write_json(row_path, {
            "source_class": source_class,
            "harm": harm,
            "before_loss": base_losses,
            "after_loss": after_losses,
            "matched_annotations": counts,
            "probe_losses": probe_losses,
            "probe_steps": args.probe_steps,
            "probe_lr": args.probe_lr,
            "last_decoder_layers": args.last_decoder_layers,
            "probed_source_classes": probe_sources,
            "production_ready": len(probe_sources) == len(class_ids),
            "probe_delta_norm": probe_delta_norm,
            "cache_config": harm_config,
        })
        del probe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    now = clock()
    timing['source_probes_and_target_losses'] = now - phase_start
    phase_start = now

    size = len(ordered_ids)
    index = {class_id: offset for offset, class_id in enumerate(ordered_ids)}
    harm = torch.zeros(size, size, dtype=torch.float32)
    valid = torch.zeros(size, size, dtype=torch.bool)
    for source_class, row in harm_rows.items():
        for target_class, value in row.items():
            if source_class == target_class:
                continue
            harm[index[source_class], index[target_class]] = max(0.0, float(value))
            valid[index[source_class], index[target_class]] = True
    artifact = {
        "schema_version": 3,
        "class_ids": ordered_ids,
        "features": features,
        "harm": harm,
        "valid_mask": valid,
        "metadata": {
            **identity,
            "supervision": "empirical_train_loss_increase",
            "feature_source": "detector_decoder_ffn_gradient_sketch",
            "last_decoder_layers": args.last_decoder_layers,
            "checkpoint": str(args.checkpoint),
            "annotation": str(annotation),
            "stage": args.stage,
            "uses_validation_labels": False,
            "deterministic_calibration_transforms": True,
            "sketch_matched_annotations": sketch_counts,
            "probe_steps": args.probe_steps,
            "probe_lr": args.probe_lr,
            "probed_source_classes": probe_sources,
            "production_ready": len(probe_sources) == len(class_ids),
            "gnn_ablation": ablation,
            "protocol_validation": checked_manifest.get("validation_mode", "official"),
            "paper_comparable": bool(checked_manifest.get("paper_comparable", False)),
        },
    }
    artifact_path = args.output_dir / f"empirical_stage{args.stage}.pt"
    atomic_torch_save(artifact, artifact_path)
    positive_edges = int(((harm > 0) & valid).sum())
    if positive_edges == 0:
        raise RuntimeError(
            "All empirical harm labels are zero. Increase --probe-steps or "
            "--probe-lr; no meaningful GNN checkpoint was written."
        )

    train_mask, validation_mask, held_out = source_masks(
        ordered_ids, valid, args.validation_fraction, args.seed)
    validation_model = ClassInterferenceGNN(
        input_dim=args.feature_dim, hidden_dim=args.gnn_hidden_dim,
        message_steps=ablation["message_steps"], dropout=args.gnn_dropout,
        use_node_encoder=ablation["use_node_encoder"]).to(device)
    validation_history = fit_interference_gnn(
        validation_model, [{**artifact, "valid_mask": train_mask}],
        epochs=args.gnn_epochs, lr=args.gnn_lr,
        ranking_margin=ablation["ranking_margin"],
        ranking_weight=ablation["ranking_weight"])
    validation_metrics = prediction_metrics(
        validation_model, artifact, validation_mask, args.validation_k)
    print("Held-out source validation:", json.dumps({
        "classes": held_out, **validation_metrics}, sort_keys=True))
    now = clock()
    timing['artifact_and_validation_gnn'] = now - phase_start
    phase_start = now

    set_seed(args.seed)
    final_model = ClassInterferenceGNN(
        input_dim=args.feature_dim, hidden_dim=args.gnn_hidden_dim,
        message_steps=ablation["message_steps"], dropout=args.gnn_dropout,
        use_node_encoder=ablation["use_node_encoder"]).to(device)
    final_history = fit_interference_gnn(
        final_model, [artifact], epochs=args.gnn_epochs, lr=args.gnn_lr,
        ranking_margin=ablation["ranking_margin"],
        ranking_weight=ablation["ranking_weight"])
    now = clock()
    timing['final_gnn_fit'] = now - phase_start
    timing['total_before_save'] = now - started
    checkpoint_path = args.output_dir / f"gnn_stage{args.stage}{suffix}.pt"
    metadata = {
        **artifact["metadata"],
        "artifact": str(artifact_path),
        "training_classes": ordered_ids,
        "probed_source_classes": probe_sources,
        "production_ready": len(probe_sources) == len(class_ids),
        "held_out_source_validation": {"classes": held_out, **validation_metrics},
        "epochs": args.gnn_epochs,
        "final_training_loss": final_history[-1],
        "gnn_ablation": ablation,
        "protocol_validation": checked_manifest.get("validation_mode", "official"),
        "paper_comparable": bool(checked_manifest.get("paper_comparable", False)),
    }
    save_gnn_checkpoint(final_model, checkpoint_path, extra=metadata)
    summary = {
        "artifact": str(artifact_path),
        "gnn_checkpoint": str(checkpoint_path),
        "class_count": size,
        "valid_edges": int(valid.sum()),
        "positive_edges": positive_edges,
        "probed_source_classes": probe_sources,
        "production_ready": len(probe_sources) == len(class_ids),
        "validation": {"classes": held_out, **validation_metrics},
        "validation_training_loss": validation_history[-1],
        "final_training_loss": final_history[-1],
        "gnn_ablation": ablation,
        "timing_seconds": timing,
        "cache_hits": cache_hits,
    }
    write_json(args.output_dir / f"calibration_summary{suffix}.json", summary)
    write_json(args.output_dir / f"gnn_history{suffix}.json", {
        "validation_model": validation_history, "final_model": final_history})
    write_json(args.output_dir / f"calibration_complete{suffix}.json", {
        "stage": args.stage, "class_count": size,
        "gnn_checkpoint": str(checkpoint_path),
        "production_ready": len(probe_sources) == len(class_ids),
        "gnn_ablation": ablation,
        "protocol_validation": checked_manifest.get("validation_mode", "official"),
        "paper_comparable": bool(checked_manifest.get("paper_comparable", False))})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
