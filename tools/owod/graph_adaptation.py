"""Build graph controller inputs exclusively from official training annotations."""
from pathlib import Path
import torch

from models.graph_local.gnn import load_gnn_checkpoint, compress_gradient_sketches
from models.graph_local.lora import target_base_weights
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.screening_data import experiment_files


def validate_gnn_metadata(metadata, args, source_checkpoint):
    expected = {
        'supervision': 'empirical_train_loss_increase',
        'production_ready': True,
        'uses_validation_labels': False,
        'manifest_sha256': args.owod_manifest_sha256,
        'checkpoint_sha256': file_sha256(Path(source_checkpoint)),
        'stage': args.owod_stage - 1,
        'last_decoder_layers': args.graph_adapter_last_n,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'GNN provenance mismatch: {key}; recalibrate the preceding task')
    if metadata.get('screening_sha256') != getattr(args, 'owod_screening_sha256', None):
        raise ValueError('GNN screening/full-data provenance mismatch')
    if set(metadata.get('training_classes', [])) != set(args.owod_previous_class_ids):
        raise ValueError('GNN calibration must use only the preceding task known classes')


def build_stage_spec(model, criterion, args, source_checkpoint):
    # Lazy import: calibration utilities import main for the checkpoint loader.
    from tools.owod.gnn_calibration import build_calibration_dataset, compute_gradient_sketches
    gnn, metadata = load_gnn_checkpoint(Path(args.graph_gnn_checkpoint), device='cpu')
    validate_gnn_metadata(metadata, args, source_checkpoint)
    manifest, files = experiment_files(args, args.owod_stage)
    current = list(args.owod_current_class_ids)
    old = list(args.owod_previous_class_ids)
    # Released balanced memory supplies old-class controller evidence. It never
    # changes the increment loader, optimizer exposure, or author image budget.
    image_root = resolve_image_root(Path(args.owod_manifest), manifest)
    old_dataset = build_calibration_dataset(image_root, files['train'])
    new_dataset = build_calibration_dataset(image_root, files['increment_train'])
    datasets = {c: old_dataset for c in old}
    datasets.update({c: new_dataset for c in current})
    weights = target_base_weights(model, args.graph_adapter_last_n)
    original = [p.requires_grad for p in weights]
    try:
        for parameter in weights:
            parameter.requires_grad_(True)
        sketches, counts = compute_gradient_sketches(
            model, criterion, datasets, old + current, next(model.parameters()).device,
            batch_size=1, num_workers=args.num_workers,
            max_images=args.graph_sketch_max_images,
            last_decoder_layers=args.graph_adapter_last_n)
    finally:
        for parameter, enabled in zip(weights, original):
            parameter.requires_grad_(enabled)
    ordered, features = compress_gradient_sketches(sketches, gnn.input_dim)
    with torch.no_grad():
        scores = gnn(features)['edge_prob']
    positions = {c: i for i, c in enumerate(ordered)}
    risk = scores[[positions[c] for c in current]][:, [positions[c] for c in old]]
    return {
        'stage': args.owod_stage, 'class_ids': current, 'old_ids': old,
        'rank': args.graph_adapter_rank, 'last_n': args.graph_adapter_last_n,
        'risk_beta': args.graph_risk_beta, 'risk': risk.tolist(),
        'routing': getattr(args, 'graph_routing', 'hard'),
        'gnn_sha256': file_sha256(Path(args.graph_gnn_checkpoint)),
        'source_checkpoint_sha256': file_sha256(Path(source_checkpoint)),
        'training_annotation_sha256': {key: file_sha256(files[key])
                                       for key in ('train', 'increment_train')},
        'sketch_max_images': args.graph_sketch_max_images, 'sketch_counts': counts,
        'score_semantics': 'relative_directed_train_loss_increase',
    }


@torch.no_grad()
def collect_promotion_features(model, criterion, args):
    """Use only newly released train labels; no historical instance overlap required."""
    from tools.owod.gnn_calibration import build_calibration_dataset, class_loader, move_targets, target_for_class
    manifest, files = experiment_files(args, args.owod_stage)
    dataset = build_calibration_dataset(resolve_image_root(Path(args.owod_manifest), manifest),
                                        files['increment_train'])
    means, route_means = {}, {}
    model.eval()
    device = next(model.parameters()).device
    for class_id in args.owod_current_class_ids:
        chunks = []
        routed_chunks = [[] for _ in range(args.graph_adapter_last_n)]
        for images, targets in class_loader(dataset, class_id, 1, args.num_workers, args.graph_sketch_max_images):
            targets = [target_for_class(t, class_id) for t in move_targets(targets, device)]
            outputs = model(images.to(device))
            for batch, (queries, _) in enumerate(criterion.matcher(outputs, targets)):
                chunks.append(outputs['query_features'][batch, queries])
                for index, features in enumerate(outputs['discovery_inputs']):
                    routed_chunks[index].append(features[batch, queries])
        if chunks:
            means[class_id] = torch.cat(chunks).mean(0)
            route_means[class_id] = [torch.cat(values).mean(0) for values in routed_chunks]
    return means, route_means
