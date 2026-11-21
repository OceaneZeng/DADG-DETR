#!/usr/bin/env python
"""Read-only score probes and disposable tiny-fit diagnostics; no checkpoints exported."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/ops'))

from tools.owod.protocol import file_sha256, read_json
from tools.owod.screening_data import screening_files, write_json


def score_variants(old, known):
    return {'task1_hierarchical_top50': (old, 'hierarchical', 50),
            'expanded_hierarchical_top50': (known, 'hierarchical', 50),
            'expanded_focal_top50': (known, 'focal', 50),
            'expanded_hierarchical_top300': (known, 'hierarchical', 300)}


def task2_score_variants(known):
    return {f'{scoring}_top{limit}': (known, scoring, limit)
            for scoring in ('hierarchical', 'focal') for limit in (50, 300)}


def score_summary(report):
    """Display AP in percent; retain the original fractional metrics in scores.json."""
    return {key: {
        'ap50_percent': {name: 100 * value for name, value in row['metrics'].items()
                         if name.endswith('ap50') and value is not None},
        'candidate_percent': {group: 100 * row['composition'][group] /
                              max(1, row['composition']['detections'])
                              for group in ('old', 'current', 'unknown')}}
        for key, row in report.items() if isinstance(row, dict) and 'metrics' in row}


def checked_checkpoint(path, manifest_hash, screen_hash, stage, phase='increment'):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    saved = payload.get('args')
    expected = {'owod_manifest_sha256': manifest_hash, 'owod_screening_sha256': screen_hash,
                'owod_stage': stage, 'owod_phase': phase, 'owod_discovery': True,
                'paper_comparable_protocol': False}
    if any(getattr(saved, key, None) != value for key, value in expected.items()):
        raise ValueError('Diagnostic checkpoint provenance mismatch')
    if payload.get('epoch', -2) != getattr(saved, 'epochs', 0) - 1:
        raise ValueError('Diagnostic input must be a completed phase checkpoint')
    if stage == 1 and (saved.graph_routing != 'hard' or saved.graph_localization != 'frozen'):
        raise ValueError('Tiny fit requires the hard/frozen Task 2 diagnostic baseline')
    return saved


def make_loader(image_root, annotation, batch, workers, shuffle=False, seed=42):
    import torch
    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    import util.misc as utils
    dataset = build_calibration_dataset(Path(image_root), Path(annotation))
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                        collate_fn=utils.collate_fn, generator=torch.Generator().manual_seed(seed))
    return dataset, loader


def evaluate_views(model, loader, coco, variants, specification, old, current, device, protected_model=None,
                   *, include_current_stats=False, include_previous_stats=False, compare_current_ranks=False,
                   exclude_unknown=(), postprocessors=None, print_freq=0, include_wi_sources=False,
                   rank_observer=None):
    import torch
    from models.graph_local.discovery import DiscoveryPostProcess
    from models.owod_benchmark_eval import evaluate_author_benchmark
    from util.box_ops import box_cxcywh_to_xyxy, box_iou
    processors = dict(postprocessors or {})
    if rank_observer is not None and not compare_current_ranks:
        raise ValueError('Recording target ranks requires paired scoring views')
    for key, (ids, scoring, limit) in variants.items():
        if key not in processors:
            processors[key] = DiscoveryPostProcess(known_ids=ids, max_detections=limit, scoring=scoring)
    if exclude_unknown and (protected_model is not None or
                            any(variants[key][1] == 'hierarchical' for key in exclude_unknown)):
        raise ValueError('Unknown suppression requires focal views of a single model')
    if compare_current_ranks:
        if (protected_model is not None or len(variants) != 2
                or any(limit != 50 or not set(current).issubset(ids) for ids, _, limit in variants.values())):
            raise ValueError('Current candidate ranks require two Top-50 views of the same outputs')
        from tools.owod.supervision_probe import current_candidate_rank_rows, summarize_candidate_ranks
        rank_rows = []
    predictions = {key: {} for key in variants}
    stats = {key: dict(detections=0, old=0, current=0, unknown=0, score_sum=0.,
                       old_score_sum=0., current_score_sum=0.) for key in variants}
    coverage = {group: {'hits': 0, 'gt': 0} for group in ('old', 'current')}
    old_set, current_set = set(old), set(current)
    seen = set()
    model.eval()
    if protected_model is not None:
        protected_model.eval()
    with torch.no_grad():
        for step, (samples, targets) in enumerate(loader, 1):
            outputs = model(samples.to(device))
            known_outputs = outputs
            if exclude_unknown:
                # Zero novelty removes unknown scores without changing focal known scores.
                known_outputs = dict(outputs, pred_novelty=torch.full_like(outputs['pred_novelty'], -torch.inf))
            reference = protected_model(samples.to(device)) if protected_model is not None else None
            sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            current_overlaps = {}
            batch_images = {}
            for index, target in enumerate(targets):
                image_id = int(target['image_id'])
                if image_id in seen:
                    continue
                seen.add(image_id)
                batch_images[index] = image_id
                for group, ids in (('old', old), ('current', current)):
                    selected = torch.isin(target['labels'], torch.tensor(ids, dtype=torch.long))
                    boxes = target['boxes'][selected].to(device)
                    coverage[group]['gt'] += len(boxes)
                    if len(boxes):
                        overlap = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][index]),
                                          box_cxcywh_to_xyxy(boxes))[0]
                        coverage[group]['hits'] += int((overlap.amax(0) > .5).sum())
                        if compare_current_ranks and group == 'current':
                            current_overlaps[index] = (target['labels'][selected].to(device), overlap)
            score_tables = []
            for key, processor in processors.items():
                if reference is not None:
                    from tools.owod.capacity_probe import isolated_predictions
                    processed = isolated_predictions(outputs, reference, processor, old, current, sizes)
                else:
                    view_outputs = known_outputs if key in exclude_unknown else outputs
                    capture_scores = bool(current_overlaps) or rank_observer is not None
                    processed = processor(view_outputs, sizes, include_top_indices=capture_scores)
                    if capture_scores:
                        candidates, categories, _ = processor.candidate_scores(view_outputs)
                        score_tables.append((candidates, categories, [result['top_indices'] for result in processed]))
                for index, image_id in batch_images.items():
                    result = {k: processed[index][k].cpu() for k in ('boxes', 'labels', 'scores')}
                    predictions[key][image_id] = result
                    labels = result['labels'].tolist()
                    scores = result['scores'].tolist()
                    stats[key]['detections'] += len(labels)
                    stats[key]['old'] += sum(label in old_set for label in labels)
                    stats[key]['current'] += sum(label in current_set for label in labels)
                    stats[key]['unknown'] += labels.count(91)
                    stats[key]['score_sum'] += float(result['scores'].sum())
                    stats[key]['old_score_sum'] += sum(s for s, label in zip(scores, labels) if label in old_set)
                    stats[key]['current_score_sum'] += sum(s for s, label in zip(scores, labels) if label in current_set)
            for index, (labels, overlap) in current_overlaps.items():
                rank_rows.append(current_candidate_rank_rows(labels, overlap, index, score_tables))
            if rank_observer is not None:
                rank_observer(outputs, targets, batch_images, score_tables)
            if print_freq and (step % print_freq == 0 or step == len(loader)):
                print(f'Inference {step}/{len(loader)} batches, {len(seen)} images', flush=True)
    report = {}
    for key, (ids, scoring, limit) in variants.items():
        values = stats[key]
        count = max(1, values['detections'])
        if print_freq:
            print(f'Computing author metrics for {key}', flush=True)
        metrics = evaluate_author_benchmark(predictions.pop(key), coco, ids,
                           [c for c in old if c in ids], [c for c in current if c in ids], specification,
                           include_current_stats=include_current_stats, include_previous_stats=include_previous_stats,
                           include_wi_sources=include_wi_sources)
        wi_sources = metrics.pop('wi_sources', None)
        current_stats = metrics.pop('current_class_stats', None)
        previous_stats = metrics.pop('previous_class_stats', None)
        if specification.get('diagnostic_current_only'):
            metrics = {'current_ap50': metrics['current_ap50']}
        elif specification.get('diagnostic_previous_only'):
            metrics = {'previous_ap50': metrics['previous_ap50']}
        report[key] = {'metrics': metrics,
                       'composition': {**values, 'mean_score': values['score_sum'] / count,
                                       'mean_old_score': values['old_score_sum'] / max(1, values['old']),
                                       'mean_current_score': values['current_score_sum'] / max(1, values['current'])},
                       'scoring': scoring, 'max_detections': limit}
        if current_stats is not None:
            report[key]['current_class_stats'] = current_stats
        if previous_stats is not None:
            report[key]['previous_class_stats'] = previous_stats
        if wi_sources is not None:
            report[key]['wi_sources'] = wi_sources
    for group, values in coverage.items():
        report[f'raw_{group}_box_recall_iou50'] = {**values,
            'recall': values['hits'] / values['gt'] if values['gt'] else None,
            'definition': 'class-agnostic all-query geometric coverage; not author U-Recall/AP'}
    if protected_model is not None:
        report['coverage_note'] = 'Raw box coverage describes the trained branch only, not the combined predictions.'
    if compare_current_ranks:
        report['current_candidate_ranks'] = summarize_candidate_ranks(rank_rows, tuple(variants), current)
    return report


def parameter_views(model, old, current):
    """Separate old/new classifier rows from trainable adapter parameters."""
    for name, parameter in model.named_parameters():
        if 'class_embed' in name:
            yield 'new_classifier', name, parameter, current
            yield 'old_classifier', name, parameter, old
        elif parameter.requires_grad:
            from tools.owod.capacity_probe import parameter_group
            group = parameter_group(name)
            if group not in ('class_adapters', 'bbox_heads', 'base_decoder'):
                group = 'discovery_and_other'
            yield group, name, parameter, None


def parameter_snapshot(model, old, current):
    return {(group, name): (p.detach()[rows] if rows is not None else p.detach()).cpu().clone()
            for group, name, p, rows in parameter_views(model, old, current)}


def parameter_report(model, old, current, initial):
    result = {}
    for group, name, p, rows in parameter_views(model, old, current):
        row = result.setdefault(group, dict(gradient_l2_squared=0., delta_l2_squared=0., trainable_tensors=0))
        row['trainable_tensors'] += int(p.requires_grad)
        value = (p.detach()[rows] if rows is not None else p.detach()).cpu()
        row['delta_l2_squared'] += float((value - initial[group, name]).square().sum())
        if p.grad is not None:
            gradient = p.grad[rows] if rows is not None else p.grad
            row['gradient_l2_squared'] += float(gradient.detach().square().sum())
    return result


def fit(args, source_args, checkpoint, parent, manifest, files, identity):
    import torch
    from tools.owod.gnn_calibration import load_detector, set_seed, move_targets
    from models.graph_local.routed_adapter import configure_trainable, mask_inactive_adapter_gradients
    from models.graph_local.discovery import make_ema, update_ema
    from models.graph_local.discovery_candidates import flip_samples
    from models.graph_local.distillation import old_class_distillation_losses
    from tools.owod.manage_data import prepare_fit_data

    old = list(source_args.owod_previous_class_ids)
    current = list(source_args.owod_current_class_ids)
    known = list(source_args.owod_known_class_ids)
    annotation = prepare_fit_data(args.manifest, args.screening, args.output / 'data', args.images, args.seed)
    dataset, _ = make_loader(manifest['image_root'], annotation, args.batch_size, args.workers)
    _, eval_loader = make_loader(manifest['image_root'], annotation, args.batch_size, args.workers)
    val_dataset, val_loader = make_loader(manifest['image_root'], files['full_val'], args.batch_size, args.workers)
    specification = {**manifest['evaluation'], 'known_only_screening': True, 'diagnostic_current_only': True,
                     'image_source_ids': [image['owod_source_id'] for image in dataset.coco.dataset['images']]}
    val_spec = {**specification, 'diagnostic_current_only': False,
                'image_source_ids': read_json(args.screening)['heldout_source_ids']}
    capacity = args.mode in ('capacity', 'isolation')
    isolation = args.mode == 'isolation'
    variants = task2_score_variants(known) if capacity else {'hierarchical': (known, 'hierarchical', 50)}
    treatments = (['decoder_bbox_open'] if isolation else ['frozen', 'decoder_bbox_open'] if capacity else
                  ['distill_on', 'distill_off'] if args.mode == 'distill' else ['distill_on'])
    for treatment in treatments:
        set_seed(args.seed)
        model, criterion = load_detector(source_args, checkpoint, torch.device(args.device))
        handles = configure_trainable(model, current, known, localization='frozen')
        model.train_discovery()
        if treatment == 'decoder_bbox_open':
            from tools.owod.capacity_probe import open_detection_parameters
            open_detection_parameters(model)
        criterion.weight_dict['loss_adapter_route'] = source_args.graph_route_loss_coef
        criterion.epoch = source_args.epochs
        ema = make_ema(model)
        teacher = None
        protected = None
        if isolation:
            protected, _ = load_detector(source_args, checkpoint, torch.device(args.device))
            protected.eval().requires_grad_(False)
        if treatment == 'distill_on':
            teacher, _ = load_detector(source_args, parent, torch.device(args.device))
            teacher.eval().requires_grad_(False)
        groups = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lr, decay = source_args.lr, source_args.weight_decay
            if 'class_embed' in name:
                lr *= source_args.class_embed_lr_mult
                decay = 0.
            elif any(word in name for word in source_args.lr_backbone_names):
                lr = source_args.lr_backbone
            elif any(word in name for word in source_args.lr_linear_proj_names):
                lr *= source_args.lr_linear_proj_mult
            groups.append({'params': [p], 'lr': lr, 'weight_decay': decay})
        optimizer = torch.optim.AdamW(groups, lr=source_args.lr)
        initial = parameter_snapshot(model, old, current)
        directory = args.output / treatment
        directory.mkdir()
        if capacity:
            from tools.owod.capacity_probe import gradient_paths, probe_training
            write_json(directory / 'parameter_scope.json', {
                'trainable': {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
                'classifier_rows': current, 'old_classifier_rows_frozen': old,
                'dropout_disabled': True, 'old_output_distillation': False,
                'learning_rate': source_args.lr, 'constant_lr_fresh_adamw': True})
            write_json(directory / 'probe_before.json', probe_training(model, criterion, eval_loader, known, args.device))
        write_json(directory / 'before.json', evaluate_views(model, eval_loader, dataset.coco, variants,
                                                            specification, old, current, args.device))
        write_json(directory / 'heldout_before.json', evaluate_views(model, val_loader, val_dataset.coco, variants,
                                                                    val_spec, old, current, args.device))
        if isolation:
            write_json(directory / 'isolated_heldout_before.json', evaluate_views(model, val_loader, val_dataset.coco,
                variants, val_spec, old, current, args.device, protected_model=protected))
        # Evaluation must not change the minibatch/dropout RNG used by the paired fits.
        set_seed(args.seed)
        _, loader = make_loader(manifest['image_root'], annotation, args.batch_size, args.workers, True, args.seed)
        iterator = iter(loader)
        for step in range(1, args.steps + 1):
            try:
                samples, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                samples, targets = next(iterator)
            samples, targets = samples.to(args.device), move_targets(targets, torch.device(args.device))
            # Eval mode disables dropout but does not disable autograd. Both arms
            # retain identical loss/EMA/memory rules, only trainable scope differs.
            model.train(not capacity); criterion.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(samples)
            with torch.no_grad():
                outputs['discovery_teacher'] = ema(flip_samples(samples))
            losses = criterion(outputs, targets)
            weights = dict(criterion.weight_dict)
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(samples)
                extra, _ = old_class_distillation_losses(outputs, teacher_outputs, old,
                    source_args.distill_score_threshold, source_args.distill_max_queries)
                losses.update(extra)
                weights.update(loss_distill_cls=source_args.distill_class_coef,
                               loss_distill_bbox=source_args.distill_bbox_coef)
            total = sum(losses[key] * weight for key, weight in weights.items() if key in losses)
            if not torch.isfinite(total):
                raise RuntimeError(f'Nonfinite diagnostic loss at step {step}')
            if capacity and step == 1:
                write_json(directory / 'gradient_paths.json', gradient_paths(model, losses, weights))
            total.backward()
            mask_inactive_adapter_gradients(model)
            monitor = step == 1 or step % args.report_interval == 0 or step == args.steps
            gradients = parameter_report(model, old, current, initial) if monitor else None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), source_args.clip_max_norm, error_if_nonfinite=True)
            optimizer.step()
            model.observe_training(criterion.evidence)
            update_ema(ema, model)
            if monitor:
                changes = parameter_report(model, old, current, initial)
                for group in changes:
                    changes[group]['gradient_l2_squared'] = gradients[group]['gradient_l2_squared']
                record = {'step': step, 'loss': float(total.detach()), 'preclip_grad_norm': float(norm),
                          'parameters': changes,
                          'weighted_losses': {key: float(losses[key].detach()) * weight
                                              for key, weight in weights.items() if key in losses}}
                with (directory / 'updates.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(treatment, step, record['loss'], flush=True)
            del outputs, losses, total
            if teacher is not None:
                del teacher_outputs, extra
        write_json(directory / 'after.json', evaluate_views(model, eval_loader, dataset.coco, variants,
                                                           specification, old, current, args.device))
        write_json(directory / 'heldout_after.json', evaluate_views(model, val_loader, val_dataset.coco, variants,
                                                                   val_spec, old, current, args.device))
        if capacity:
            write_json(directory / 'probe_after.json', probe_training(model, criterion, eval_loader, known, args.device))
        if isolation:
            write_json(directory / 'isolated_heldout_after.json', evaluate_views(model, val_loader, val_dataset.coco,
                variants, val_spec, old, current, args.device, protected_model=protected))
            write_json(directory / 'isolation_scope.json', {
                'old_and_unknown_source': str(checkpoint), 'current_source': 'disposable trained branch',
                'shared_global_topk': True, 'per_branch_quota': False,
                'raw_box_coverage_source': 'trained branch only; not isolated combined coverage',
                'guarantee': 'Frozen old/unknown scores and boxes before global top-k; AP may change through competition',
                'deployment': 'diagnostic two-forward upper-bound design, no exported model'})
        for handle in handles:
            handle.remove()
        del model, criterion, teacher, protected, ema, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output / 'complete.json', {'kind': 'disposable_learning_diagnostic', 'steps': args.steps,
                                               'checkpoint_exported': False, **identity})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('scores', 'scores-task2', 'supervision', 'capacity', 'isolation', 'fit', 'distill'))
    parser.add_argument('--source-root', type=Path, default=ROOT / 'output/screening-s-8k')
    parser.add_argument('--comparison-root', type=Path, default=ROOT / 'output/task2-shared-s-8k')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-phase', choices=('increment', 'finetune'), default='increment',
                        help='Checkpoint phase for scores-task2 or supervision')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--images', type=int, default=160)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--capacity-lr', type=float, default=2e-4,
                        help='Constant diagnostic LR for both capacity arms; not a paper schedule')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report-interval', type=int, default=50)
    args = parser.parse_args(argv)
    if args.score_phase != 'increment' and args.mode not in ('scores-task2', 'supervision', 'capacity', 'isolation'):
        parser.error('--score-phase finetune requires scores-task2, supervision, capacity or isolation')
    if min(args.images, args.steps, args.batch_size, args.report_interval) < 1 or args.workers < 0:
        parser.error('Invalid diagnostic budget')
    import math
    if not math.isfinite(args.capacity_lr) or args.capacity_lr <= 0:
        parser.error('capacity-lr must be positive and finite')
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu)
    args.screening = args.source_root / 'data/screening.json'
    manifest, files = screening_files(args.manifest, args.screening, 1)
    plan = read_json(args.comparison_root / 'comparison_plan.json')
    identity = {'manifest_sha256': file_sha256(args.manifest), 'screening_sha256': file_sha256(args.screening)}
    if any(identity[key] != plan[key] for key in identity):
        raise ValueError('Comparison and diagnostic data differ')
    parent = Path(plan['checkpoint'])
    if file_sha256(parent) != plan['checkpoint_sha256']:
        raise ValueError('Common Task 1 checkpoint changed')
    saved = checked_checkpoint(parent, identity['manifest_sha256'], identity['screening_sha256'], 0)
    if args.seed != saved.seed:
        raise ValueError('Use the original paired diagnostic seed')
    checkpoint = args.comparison_root / 'screen/hard' / manifest['protocol'] / f'seed_{args.seed}/stage_1/{args.score_phase}/checkpoint.pth'
    if args.mode != 'scores':
        saved = checked_checkpoint(checkpoint, identity['manifest_sha256'], identity['screening_sha256'], 1, args.score_phase)
        if getattr(saved, 'screen_branch_checkpoint_sha256', None) != plan['checkpoint_sha256']:
            raise ValueError('Task 2 checkpoint has a different common parent')
        if getattr(saved, 'screen_branch_gnn_sha256', None) != plan['gnn_sha256']:
            raise ValueError('Task 2 checkpoint has a different GNN source')
        if (list(saved.owod_current_class_ids) != manifest['stages'][1]['classes']
                or list(saved.owod_known_class_ids) != manifest['stages'][1]['active_classes']):
            raise ValueError('Task 2 diagnostic class order differs from the manifest')
    identity.update(parent_sha256=plan['checkpoint_sha256'], mode=args.mode,
                    checkpoint_sha256=file_sha256(parent if args.mode == 'scores' else checkpoint),
                    paper_comparable=False, checkpoint_exported=False, score_phase=args.score_phase)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty diagnostic output directory; no overwrite/resume')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'diagnostic_config.json', {**identity, 'arguments': vars(args) | {'manifest': str(args.manifest),
               'source_root': str(args.source_root), 'comparison_root': str(args.comparison_root),
               'screening': str(args.screening), 'output': str(args.output)}})
    if args.mode in ('fit', 'distill', 'capacity', 'isolation'):
        if args.mode in ('capacity', 'isolation'):
            import copy
            saved = copy.deepcopy(saved)
            saved.lr = args.capacity_lr
        fit(args, saved, checkpoint, parent, manifest, files, identity)
        return 0
    if args.mode == 'supervision':
        from tools.owod.supervision_probe import run_probe
        run_probe(args, saved, checkpoint, manifest, identity)
        return 0
    import torch
    from tools.owod.gnn_calibration import load_detector
    task2 = args.mode == 'scores-task2'
    model, _ = load_detector(saved, checkpoint if task2 else parent, torch.device(args.device))
    if not task2:
        _, files = screening_files(args.manifest, args.screening, 0)
    dataset, loader = make_loader(manifest['image_root'], files['full_val'], args.batch_size, args.workers)
    old, known = manifest['stages'][0]['active_classes'], manifest['stages'][1]['active_classes']
    current = manifest['stages'][1]['classes']
    spec = {**manifest['evaluation'], 'image_source_ids': read_json(args.screening)['heldout_source_ids'],
            'known_only_screening': True, 'diagnostic_previous_only': not task2}
    variants = task2_score_variants(known) if task2 else score_variants(old, known)
    report = evaluate_views(model, loader, dataset.coco, variants, spec, old, current, args.device)
    report['note'] = (f'Task 2 {args.score_phase} checkpoint; known-only diagnostic holdout, not official benchmark results.'
                      if task2 else 'Previous AP50 only: GT contains Task 1 classes; no current-class performance is claimed.')
    write_json(args.output / 'scores.json', report)
    summary = score_summary(report)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'complete.json', {'kind': 'score_diagnostic', **identity})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
