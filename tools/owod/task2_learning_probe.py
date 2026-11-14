"""Read-only released-class routing/box/score measurements for Task 2."""
import hashlib
from pathlib import Path


def state_sha256(model):
    import torch
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(str((key, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def per_class_probe_indices(dataset, current_ids, per_class=12):
    """Deterministic train-only class coverage; this does not register a split."""
    if per_class < 1:
        raise ValueError('Probe images per class must be positive')
    selected = set()
    for category in current_ids:
        matches = [i for i, image in enumerate(dataset.ids)
                   if any(a['category_id'] == category for a in dataset.coco.imgToAnns.get(image, []))]
        if not matches:
            raise ValueError(f'Probe has no released training examples for class {category}')
        selected.update(matches[:per_class])
    return sorted(selected)


def measure_batch(outputs, targets, indices, known_ids, stages, threshold=.5, limit=50,
                  scoring='hierarchical'):
    """Class-aware candidate coverage, not AP, using the actual Top-K rule."""
    import torch
    from util.box_ops import box_cxcywh_to_xyxy, box_iou
    rows = {}
    totals = dict(images=len(targets), detections=0, unknown_predictions=0)
    for batch, target in enumerate(targets):
        logits = outputs['pred_logits'][batch, :, known_ids]
        obj = outputs['pred_objectness'][batch].sigmoid()
        novelty = outputs['pred_novelty'][batch].sigmoid()
        known = ((logits.softmax(-1) * (obj * (1 - novelty))[:, None])
                 if scoring == 'hierarchical' else logits.sigmoid() * obj[:, None])
        unknown = obj * novelty
        joint = torch.cat([known, torch.where(unknown >= threshold, unknown,
                                             torch.full_like(unknown, -torch.inf))[:, None]], -1)
        selected = {}
        for name, matrix in [('joint', joint), ('known_only', known),
                             ('classifier_only', logits.sigmoid())]:
            values, positions = matrix.flatten().topk(min(limit, matrix.numel()))
            positions = positions[torch.isfinite(values)]
            keep = torch.zeros_like(matrix, dtype=torch.bool).flatten()
            keep[positions] = True
            selected[name] = keep.reshape_as(matrix)[:, :len(known_ids)]
            if name == 'joint':
                totals['detections'] += len(positions)
                totals['unknown_predictions'] += int((positions % matrix.shape[1] == len(known_ids)).sum())
        overlap = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][batch]),
                          box_cxcywh_to_xyxy(target['boxes']))[0]
        queries, matches = indices[batch]
        for gt_index, label in enumerate(target['labels'].tolist()):
            if label not in known_ids:
                raise ValueError('Training probe received an unreleased label')
            row = rows.setdefault(str(label), dict(gt=0, raw_box_coverage=0, joint_coverage=0,
                         known_only_coverage=0, classifier_only_coverage=0, layers={}))
            covered = overlap[:, gt_index] >= .5
            row['gt'] += 1
            row['raw_box_coverage'] += int(covered.any())
            for name, keep in selected.items():
                row[name + '_coverage'] += int((covered & keep[:, known_ids.index(label)]).any())
            matched = queries[matches == gt_index]
            for name, stage in stages:
                if label not in stage.class_ids or not matched.numel():
                    continue
                values = row['layers'].setdefault(name, dict(matched=0, correct_route=0,
                           null_route=0, positive_gate=0, gate_sum=0.))
                column = stage.class_ids.index(label)
                # Forward clears last_logits into outputs['adapter_routes'];
                # retain actual logits with a temporary forward hook in probe().
                prediction = stage.probe_logits[batch, matched].argmax(-1)
                gate = stage.last_gates[batch, matched, column]
                values['matched'] += matched.numel()
                values['correct_route'] += int((prediction == column).sum())
                values['null_route'] += int((prediction == len(stage.class_ids)).sum())
                values['positive_gate'] += int((gate > 0).sum())
                values['gate_sum'] += float(gate.sum())
    return dict(classes=rows, totals=totals)


def merge_measurements(batches):
    result = dict(classes={}, totals={})
    for batch in batches:
        for key, value in batch['totals'].items():
            result['totals'][key] = result['totals'].get(key, 0) + value
        for category, row in batch['classes'].items():
            target = result['classes'].setdefault(category, dict(layers={}))
            for key, value in row.items():
                if key != 'layers':
                    target[key] = target.get(key, 0) + value
            for name, values in row['layers'].items():
                layer = target['layers'].setdefault(name, {})
                for key, value in values.items():
                    layer[key] = layer.get(key, 0) + value
    return result


def parameter_audit(model, parent_state, current_ids, previous_ids):
    import torch
    state = model.state_dict()
    key = next(k for k in state if 'class_embed' in k and k.endswith('.weight') and k in parent_state)
    delta = state[key].detach().cpu() - parent_state[key].detach().cpu()
    current = delta[current_ids].norm(dim=1)
    return dict(classifier_key=key, previous_max_abs_change=float(delta[previous_ids].abs().max()),
                current_classifier_row_delta_norm=dict(zip(map(str, current_ids), current.tolist())),
                adapter_up_norms={name: float(value.norm()) for name, value in state.items()
                                  if '.graph_adapter.' in name and name.endswith('.up.weight')},
                note='Parameter changes/nonzero factors are not AP or proof of effective learning.')


def probe(model, criterion, args, manifest_path, rank=0, world=1, per_class=12):
    import torch
    from torch.utils.data import DataLoader, Subset
    from models.graph_local.routed_adapter import RiskRoutedStage
    from tools.owod.protocol import stage_files, resolve_image_root
    from tools.owod.gnn_calibration import build_calibration_dataset, move_targets
    from util.misc import collate_fn
    manifest, files = stage_files(manifest_path, 1)
    dataset = build_calibration_dataset(resolve_image_root(manifest_path, manifest), files['increment_train'])
    selection = per_class_probe_indices(dataset, args.owod_current_class_ids, per_class)
    loader = DataLoader(Subset(dataset, selection[rank::world]), batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, multiprocessing_context='spawn')
    stages = [(name, module) for name, module in model.named_modules() if isinstance(module, RiskRoutedStage)]
    def keep_logits(module, inputs, output):
        module.probe_logits = module.last_logits.detach()
    hooks = [stage.register_forward_hook(keep_logits) for _, stage in stages]
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for images, targets in loader:
                targets = move_targets(targets, next(model.parameters()).device)
                outputs = model(images.to(next(model.parameters()).device))
                rows.append(measure_batch(outputs, targets, criterion.matcher(outputs, targets),
                                          args.owod_known_class_ids, stages,
                                          args.unknown_threshold, args.owod_max_detections,
                                          args.discovery_scoring))
        summary = merge_measurements(rows)
        if world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, summary)
            summary = merge_measurements(gathered)
        summary.update(diagnostic_only=True, image_indices=selection,
                       images_per_current_class=per_class, training_annotation_sha256=manifest['stages'][1]['sha256']['increment_train'],
                       note='Released Task 2 training labels only. Class-covered deterministic sample, '
                            'not a benchmark split, AP or representative recall. Matching excludes teacher '
                            'pseudo labels. known_only removes unknown competition; classifier_only also '
                            'removes objectness/novelty and changes the class link. Neither changes formal scoring.')
        return summary
    finally:
        for hook in hooks:
            hook.remove()
        for _, stage in stages:
            if hasattr(stage, 'probe_logits'):
                del stage.probe_logits
        model.train(was_training)


def evaluate_reference(opt, identity):
    """Full author AP plus train-only measurements, without any optimization."""
    import copy
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from engine import evaluate
    from models.graph_local.discovery import DiscoveryPostProcess
    from tools.owod.gnn_calibration import load_detector, build_calibration_dataset
    from tools.owod.protocol import file_sha256, stage_files, resolve_image_root
    from tools.owod.train_task2_diagnostic import dump
    from util.misc import collate_fn
    directory = opt.output / 'reference'
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').is_file():
        import json
        complete = json.loads((directory / 'complete.json').read_text())
        if (complete.get('source_checkpoint_sha256') != identity['reference']['checkpoint_sha256']
                or not all((directory / name).is_file() for name in ('learning_probe.json', 'metrics.json'))):
            raise ValueError('Incomplete or changed reference measurements')
        return
    saved = torch.load(opt.reference_checkpoint, map_location='cpu', weights_only=False)['args']
    manifest, files = stage_files(opt.manifest, 1)
    if (saved.owod_evaluation != manifest['evaluation']
            or saved.owod_known_class_ids != manifest['stages'][1]['active_classes']
            or saved.owod_current_class_ids != manifest['stages'][1]['classes']):
        raise ValueError('Reference evaluation/classes do not match the canonical protocol')
    args = copy.deepcopy(saved)
    args.rank, args.gpu, args.world_size = opt.rank, opt.gpu, opt.world_size
    device = torch.device('cuda', opt.gpu)
    model, criterion = load_detector(args, opt.reference_checkpoint, device)
    model.eval()
    before = state_sha256(model)
    report = probe(model, criterion, args, opt.manifest, opt.rank, opt.world_size)
    validation = build_calibration_dataset(resolve_image_root(opt.manifest, manifest), files['full_val'])
    sampler = DistributedSampler(validation, num_replicas=opt.world_size, rank=opt.rank, shuffle=False)
    loader = DataLoader(validation, batch_size=1, sampler=sampler, num_workers=2,
                        collate_fn=collate_fn, multiprocessing_context='spawn')
    metrics = evaluate(model, criterion, {'bbox': DiscoveryPostProcess(
        known_ids=args.owod_known_class_ids, max_detections=args.owod_max_detections,
        unknown_threshold=args.unknown_threshold, scoring=args.discovery_scoring)}, loader,
        validation.coco, device, str(directory), owod_known_class_ids=args.owod_known_class_ids,
        owod_previous_class_ids=args.owod_previous_class_ids,
        owod_current_class_ids=args.owod_current_class_ids, owod_unknown_threshold=args.unknown_threshold,
        owod_specification=manifest['evaluation'])[0]
    if state_sha256(model) != before or file_sha256(opt.reference_checkpoint) != identity['reference']['checkpoint_sha256']:
        raise ValueError('Reference weights changed during read-only measurements')
    if opt.rank == 0:
        dump(directory / 'learning_probe.json', report)
        dump(directory / 'metrics.json', metrics)
        dump(directory / 'complete.json', dict(diagnostic_only=True, weight_state_sha256=before,
             source_checkpoint_sha256=identity['reference']['checkpoint_sha256']))
    if opt.world_size > 1:
        torch.distributed.barrier()
    del model, criterion, loader, validation
    torch.cuda.empty_cache()
