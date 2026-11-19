"""Disposable parameter-capacity checks; never alter production training policy."""
import torch

from models.graph_local.lora import open_decoder_and_bbox as open_detection_parameters


@torch.no_grad()
def isolated_predictions(outputs, reference, processor, old, current, sizes):
    """Keep source scores/boxes intact, then apply one shared candidate budget.

    Scores use each branch's full known-class denominator. No GT routing,
    score calibration, per-branch quota, or unknown suppression is applied.
    """
    from models.graph_local.discovery import DiscoveryPostProcess
    limit = (len(processor.known_ids) + 1) * max(outputs['pred_logits'].shape[1], reference['pred_logits'].shape[1])
    full = DiscoveryPostProcess(known_ids=processor.known_ids, max_detections=limit,
                               scoring=processor.scoring, unknown_threshold=processor.unknown_threshold)
    fresh, frozen = full(outputs, sizes), full(reference, sizes)
    results = []
    for new, previous in zip(fresh, frozen):
        new_mask = torch.isin(new['labels'], new['labels'].new_tensor(current))
        old_mask = torch.isin(previous['labels'], previous['labels'].new_tensor([*old, 91]))
        merged = {key: torch.cat((previous[key][old_mask], new[key][new_mask])) for key in previous}
        keep = merged['scores'].topk(min(processor.max_detections, len(merged['scores']))).indices
        results.append({key: value[keep] for key, value in merged.items()})
    return results


def parameter_group(name):
    if 'class_embed' in name:
        return 'classifier'
    if 'bbox_embed' in name:
        return 'bbox_heads'
    if 'graph_adapter' in name:
        return 'class_adapters'
    if 'discovery_adapter' in name or 'affinity_graph' in name:
        return 'discovery'
    if 'transformer.decoder' in name:
        return 'base_decoder'
    return 'other'


def gradient_paths(model, losses, weights):
    """Measure task-specific gradients before clipping, without optimizer updates."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    result = {}
    for group, prefixes in (('classification', ('loss_ce',)), ('localization', ('loss_bbox', 'loss_giou'))):
        terms = [value * weights[key] for key, value in losses.items()
                 if key in weights and key.startswith(prefixes)]
        total = sum(terms)
        norms = {}
        if terms and total.requires_grad:
            gradients = torch.autograd.grad(total, [p for _, p in named], retain_graph=True, allow_unused=True)
            for (name, _), gradient in zip(named, gradients):
                key = parameter_group(name)
                norms[key] = norms.get(key, 0.) + (float(gradient.detach().square().sum()) if gradient is not None else 0.)
        result[group] = norms
    return result


@torch.no_grad()
def probe_training(model, criterion, loader, known, device):
    from tools.owod.supervision_probe import probe_batch, summarize
    from tools.owod.gnn_calibration import move_targets
    model.eval()
    rows = []
    for samples, targets in loader:
        rows.extend(probe_batch(model(samples.to(device)), move_targets(targets, torch.device(device)),
                                criterion.matcher, known))
    return summarize(rows)
