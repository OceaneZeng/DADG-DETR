#!/usr/bin/env python
"""Read-only Task 1/2 probes on released training labels, with optional Task 1 score audit."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def select_training_indices(dataset, classes, count, seed):
    """Stable train-only sample with class coverage before filling the budget."""
    if count < 1:
        raise ValueError('images must be positive')
    labels = {i: {int(a['category_id']) for a in dataset.coco.imgToAnns.get(image_id, [])}
              for i, image_id in enumerate(dataset.ids)}
    order = sorted(labels, key=lambda i: hashlib.sha256(
        f'{seed}:{dataset.coco.imgs[dataset.ids[i]].get("owod_source_id", dataset.ids[i])}'.encode()).digest())
    selected, covered = [], set()
    for category in classes:
        if category not in covered:
            candidate = next((i for i in order if category in labels[i] and i not in selected), None)
            if candidate is not None and len(selected) < count:
                selected.append(candidate)
                covered.update(labels[candidate])
    chosen = set(selected)
    selected.extend(i for i in order if i not in chosen)
    return selected[:count]


@contextmanager
def capture_routes(model):
    """Read transient gates before the detector consumes per-layer logits."""
    from models.graph_local.routed_adapter import RiskRoutedStage
    captured, handles = {}, []

    def record(name):
        def hook(module, _inputs, _output):
            captured[name] = dict(class_ids=module.class_ids,
                probabilities=module.last_logits.detach().softmax(-1),
                gates=module.last_gates.detach().clone())
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, RiskRoutedStage):
                handles.append(module.register_forward_hook(record(name)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def routing_rows(captured, indices, targets):
    """Actual matched-GT gates; positive gates do not prove optimizer updates."""
    rows = []
    for layer, trace in captured.items():
        probabilities = trace['probabilities']
        class_ids = trace['class_ids']
        selected = probabilities.argmax(-1)
        for batch, (queries, matches) in enumerate(indices):
            for query, match in zip(queries.tolist(), matches.tolist()):
                category = int(targets[batch]['labels'][match])
                if category not in class_ids:
                    continue
                column = class_ids.index(category)
                rows.append(dict(layer=layer, class_id=category,
                    image_id=int(targets[batch]['image_id'].item()),
                    selected_correct=int(selected[batch, query]) == column,
                    selected_null=int(selected[batch, query]) == len(class_ids),
                    probability=float(probabilities[batch, query, column]),
                    effective_gate=float(trace['gates'][batch, query, column])))
    return rows


def summarize_routes(rows):
    import torch
    result = {}
    for category in sorted({r['class_id'] for r in rows}):
        group = [r for r in rows if r['class_id'] == category]
        gates = torch.tensor([r['effective_gate'] for r in group])
        result[str(category)] = dict(matched_layer_observations=len(group),
            correct_argmax_fraction=sum(r['selected_correct'] for r in group) / len(group),
            null_argmax_fraction=sum(r['selected_null'] for r in group) / len(group),
            mean_probability=sum(r['probability'] for r in group) / len(group),
            positive_gate_fraction=float((gates > 0).float().mean()),
            gate_p10_p50_p90=gates.quantile(torch.tensor([.1, .5, .9])).tolist())
    return result


def memory_report(model, feature_sums, feature_counts, current_ids):
    import torch
    from models.graph_local.discovery import DiscoveryFFN
    memory = model.memory
    centers = memory.centers[memory.known_slots:]
    counts = memory.counts[memory.known_slots:]
    supported = counts >= model.min_support
    per_class = {}
    with torch.no_grad():
        for category in current_ids:
            observed = feature_counts.get(category, 0)
            if not observed:
                per_class[str(category)] = dict(observed_gt=0, reason='no_sampled_GT')
                continue
            mean = feature_sums[category] / observed
            similarity = model.affinity_graph.similarities(mean[None], centers)[0]
            eligible = supported & (similarity >= model.transfer_threshold)
            reason = ('transfer_disabled' if not model.transfer_enabled else
                      'no_supported_slots' if not supported.any() else
                      'below_affinity_threshold' if not eligible.any() else 'eligible_now')
            per_class[str(category)] = dict(observed_gt=observed, reason=reason,
                max_supported_affinity=float(similarity[supported].max()) if supported.any() else None,
                eligible_slots=eligible.nonzero().flatten().tolist())
        snapshots = {name: [float((up @ down).norm()) for up, down in
                     zip(module.snapshot_up, module.snapshot_down)]
                     for name, module in model.named_modules() if isinstance(module, DiscoveryFFN)}
    return dict(support=counts.tolist(), supported_slots=int(supported.sum()),
        largest_support_fraction=float(counts.max() / counts.sum()) if counts.sum() > 0 else None,
        min_support=model.min_support, transfer_threshold=model.transfer_threshold,
        current_feature_affinity=per_class, snapshot_effective_norms=snapshots,
        note='Current matched-GT means, not the historical promotion input or a replay of its decision.')


def checked_inputs(saved, manifest_path):
    from tools.owod.protocol import file_sha256, stage_files
    stage, phase = getattr(saved, 'owod_stage', None), getattr(saved, 'owod_phase', None)
    if (not getattr(saved, 'owod_discovery', False) or stage not in (0, 1)
            or phase not in ('increment', 'finetune') or (stage == 0 and phase == 'finetune')
            or getattr(saved, 'owod_screening', None)
            or not getattr(saved, 'paper_comparable_protocol', False)):
        raise ValueError('Expected a full-data discovery Task 1/2 phase checkpoint, not a screening checkpoint')
    if file_sha256(manifest_path) != saved.owod_manifest_sha256:
        raise ValueError('Checkpoint manifest mismatch')
    manifest, files = stage_files(manifest_path, stage)
    record = manifest['stages'][stage]
    if (list(saved.owod_known_class_ids) != record['active_classes']
            or list(saved.owod_current_class_ids) != record['classes']):
        raise ValueError('Checkpoint class order mismatch')
    if saved.owod_evaluation != manifest['evaluation']:
        raise ValueError('Checkpoint evaluation convention mismatch')
    return manifest, files['increment_train' if phase == 'increment' else 'train']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-audit', action='store_true',
                        help='Compare existing scores with one-to-one training-subset AP (Task 1 only)')
    options = parser.parse_args(argv)
    if options.images < 1:
        parser.error('images must be positive')
    if options.output.suffix.lower() != '.json':
        parser.error('output must be a new .json file')
    if options.output.exists():
        raise FileExistsError(f'Refusing to overwrite {options.output}')
    import torch
    from tools.owod.protocol import file_sha256
    from tools.owod.gnn_calibration import load_detector, build_calibration_dataset, move_targets, set_seed
    from tools.owod.diagnose_discovery_task1 import score_competition, summarize_competition
    from tools.owod.supervision_probe import probe_batch, summarize
    from models.graph_local.discovery import make_ema
    from models.graph_local.discovery_candidates import mine_candidates, flip_samples
    from util.misc import collate_fn

    torch.set_num_threads(2)
    set_seed(options.seed)
    identities = {str(options.checkpoint): file_sha256(options.checkpoint),
                  str(options.manifest): file_sha256(options.manifest)}
    payload = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    manifest, annotation = checked_inputs(saved, options.manifest)
    if options.score_audit and saved.owod_stage != 0:
        raise ValueError('Score audit requires Task 1 increment training labels')
    identities[str(annotation)] = file_sha256(annotation)
    known, current = list(saved.owod_known_class_ids), list(saved.owod_current_class_ids)
    dataset = build_calibration_dataset(Path(manifest['image_root']), annotation)
    if any(int(a['category_id']) not in known for a in dataset.coco.anns.values()):
        raise ValueError('Training probe received unreleased labels')
    classes = known if saved.owod_phase == 'finetune' else current
    selection = select_training_indices(dataset, classes, options.images, options.seed)
    if not selection:
        raise ValueError('No training images available')
    audit = None
    if options.score_audit:
        from tools.owod.scoring_probe import TrainingScoreAudit
        audit = TrainingScoreAudit(dataset.coco, [dataset.ids[i] for i in selection],
                                   saved, manifest['evaluation'])
    device = torch.device(options.device)
    model, criterion = load_detector(saved, options.checkpoint, device)
    model.eval()
    teacher = None
    if 'discovery_ema' in payload:
        teacher = make_ema(model)
        teacher.load_state_dict(payload['discovery_ema'], strict=True)
        teacher.eval()
    memory_before = {name: value.clone() for name, value in model.memory.named_buffers()}
    rows, routes, competition, candidates, image_ids = [], [], [], [], []
    feature_sums, feature_counts = {}, {}
    with torch.no_grad(), capture_routes(model) as captured:
        for number, index in enumerate(selection, 1):
            samples, targets = collate_fn([dataset[index]])
            targets = move_targets(targets, device)
            samples = samples.to(device)
            outputs = model(samples)
            if audit is not None:
                audit.add(outputs, targets)
            image_id = int(targets[0]['image_id'].item())
            image_ids.append(image_id)
            indices = criterion.matcher(outputs, targets)
            rows.extend(probe_batch(outputs, targets, criterion.matcher, known))
            routes.extend(routing_rows(captured, indices, targets))
            competition.append(score_competition(outputs, targets, indices, known,
                limit=saved.owod_max_detections, scoring=getattr(saved, 'discovery_scoring', 'hierarchical')))
            for query, match in zip(indices[0][0].tolist(), indices[0][1].tolist()):
                category = int(targets[0]['labels'][match])
                feature = outputs['query_features'][0, query]
                feature_sums[category] = feature_sums.get(category, torch.zeros_like(feature)) + feature
                feature_counts[category] = feature_counts.get(category, 0) + 1
            if teacher is not None:
                mined = mine_candidates(outputs, targets, indices, known, teacher(flip_samples(samples)),
                    top_k=saved.discovery_top_k, saliency_threshold=saved.discovery_saliency_threshold,
                    consistency_threshold=saved.discovery_consistency_threshold)
                selected = mined['pseudo'][0].nonzero().flatten().tolist()
                candidates.append(dict(image_id=image_id, selected=[dict(query=q,
                    box_cxcywh=outputs['pred_boxes'][0, q].tolist(), weight=float(mined['weights'][0, q]),
                    saliency=float(mined['saliency'][0, q]), consistency=float(mined['consistency'][0, q]),
                    objectness=float(outputs['pred_objectness'][0, q].sigmoid()),
                    novelty=float(outputs['pred_novelty'][0, q].sigmoid())) for q in selected]))
            if number % 32 == 0:
                print(f'Probe {number}/{len(selection)}: {len(rows)} released GT', flush=True)
    if any(not torch.equal(value, memory_before[name]) for name, value in model.memory.named_buffers()):
        raise RuntimeError('Read-only probe changed prototype memory')
    if any(file_sha256(Path(path)) != identity for path, identity in identities.items()):
        raise ValueError('Input changed during probe; use a numbered, immutable checkpoint')
    report = dict(task=saved.owod_stage + 1, phase=saved.owod_phase, epoch=payload['epoch'] + 1,
        input_sha256=identities, protocol=manifest['protocol'], ap_mode=manifest['evaluation']['ap_mode'],
        diagnostic_only=True, optimization_steps=0,
        configuration=dict(version=getattr(saved, 'discovery_version', None), lr=saved.lr,
            known_rank_coef=getattr(saved, 'discovery_known_rank_coef', 0.),
            feature_grad_scale=getattr(saved, 'discovery_feature_grad_scale', 1.),
            scoring=getattr(saved, 'discovery_scoring', 'hierarchical'),
            routing=getattr(saved, 'graph_routing', 'hard'),
            localization=getattr(saved, 'graph_localization', 'frozen')),
        sampling=dict(seed=options.seed, requested_images=options.images, image_ids=image_ids,
                      rule='train-only source-ID hash order with released-class coverage'),
        supervision=summarize(rows), per_class={str(c): summarize([r for r in rows if r['class_id'] == c])
                                              for c in classes},
        score_competition=summarize_competition(competition), routing=summarize_routes(routes),
        routing_observations=routes,
        anonymous_memory=memory_report(model, feature_sums, feature_counts, current),
        recorded_promotion=[s.get('promotion', []) for s in getattr(saved, 'graph_adapter_specs', [])],
        ema_candidates=dict(available=teacher is not None, images=candidates,
            pseudo_enabled_at_checkpoint=payload['epoch'] >= saved.discovery_warmup),
        note='No optimizer, official AP measurement, test-label selection, EMA update or memory update. '
             'Deterministic train views do not replay augmented training batches. Coverage is not '
             'one-to-one AP/U-Recall. Candidate validity is not known from missing annotations. '
             'Nonzero route gates are not proof of parameter updates.')
    if audit is not None:
        report['score_audit'] = audit.summarize()
    serialized = json.dumps(report, indent=2, allow_nan=False)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x', encoding='utf-8') as handle:
        handle.write(serialized + '\n')
    print(json.dumps(dict(task=report['task'], epoch=report['epoch'],
        supervision=report['supervision'], routing=report['routing'],
        supported_anonymous_slots=report['anonymous_memory']['supported_slots']), indent=2))
    return report


if __name__ == '__main__':
    main()
