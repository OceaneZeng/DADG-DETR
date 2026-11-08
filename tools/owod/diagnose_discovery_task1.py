"""Read-only training-image probe; no optimizer, split changes or AP selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def score_competition(outputs, targets, indices, known, limit=50, scoring='hierarchical'):
    """Counterfactual known-only ranking on the same forward, never deployment."""
    import torch
    from util.box_ops import box_iou, box_cxcywh_to_xyxy
    if scoring not in ('hierarchical', 'focal'):
        raise ValueError('Unknown discovery scoring rule')
    result = dict(gt=0, coverage=dict(raw=0, joint=0, known_actual=0, known_hierarchical=0, known_sigmoid=0),
                  detections=0, unknown=0, groups={}, scoring=scoring)
    with torch.no_grad():
        for b, target in enumerate(targets):
            obj = outputs['pred_objectness'][b].sigmoid()
            novelty = outputs['pred_novelty'][b].sigmoid()
            logits = outputs['pred_logits'][b, :, known]
            unknown = obj * novelty
            scores = logits.softmax(-1) * (obj * (1 - novelty))[:, None]
            actual_known = scores if scoring == 'hierarchical' else logits.sigmoid() * obj[:, None]
            joint = torch.cat([actual_known, unknown[:, None]], -1)
            selection = joint.flatten().topk(min(limit, joint.numel())).indices
            result['detections'] += len(selection)
            result['unknown'] += int(((selection % joint.shape[1]) == len(known)).sum())
            overlap = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][b]),
                              box_cxcywh_to_xyxy(target['boxes']))[0]
            matched = torch.zeros_like(obj, dtype=torch.bool)
            matched[indices[b][0]] = True
            nearby = overlap.amax(-1) > .3 if overlap.shape[1] else torch.zeros_like(matched)
            values = torch.stack([obj, novelty, unknown, logits.sigmoid().amax(-1),
                                  logits.softmax(-1).amax(-1), actual_known.amax(-1)], -1)
            for name, mask in [('matched', matched), ('unmatched_known_overlap', ~matched & nearby),
                               ('unmatched_no_known_overlap', ~matched & ~nearby)]:
                result['groups'].setdefault(name, []).extend(values[mask].cpu().tolist())
            selected = {}
            for name, matrix in [('joint', joint), ('known_actual', actual_known), ('known_hierarchical', scores),
                                 ('known_sigmoid', logits.sigmoid())]:
                keep = torch.zeros_like(matrix, dtype=torch.bool).flatten()
                keep[matrix.flatten().topk(min(limit, matrix.numel())).indices] = True
                selected[name] = keep.reshape_as(matrix)[:, :len(known)]
            for column, label in enumerate(target['labels']):
                category = known.index(int(label))
                covered = overlap[:, column] > .5
                result['gt'] += 1
                result['coverage']['raw'] += int(covered.any())
                for name, keep in selected.items():
                    result['coverage'][name] += int((covered & keep[:, category]).any())
    return result


def summarize_competition(batches):
    import torch
    scoring = {b['scoring'] for b in batches}
    if len(scoring) != 1:
        raise ValueError('Require batches from one discovery scoring rule')
    groups = {}
    for batch in batches:
        for name, rows in batch['groups'].items():
            groups.setdefault(name, []).extend(rows)
    gt = sum(b['gt'] for b in batches)
    detections = sum(b['detections'] for b in batches)
    unknown = sum(b['unknown'] for b in batches)
    coverage = {name: sum(b['coverage'][name] for b in batches)
                for name in ('raw', 'joint', 'known_actual', 'known_hierarchical', 'known_sigmoid')}
    return dict(gt=gt, scoring=scoring.pop(),
                coverage=coverage,
                coverage_fraction={name: count / gt if gt else None for name, count in coverage.items()},
                detections=detections, unknown=unknown,
                unknown_fraction=unknown / detections if detections else None,
                score_columns=['objectness', 'novelty', 'unknown', 'max_known_sigmoid',
                               'max_known_softmax', 'max_known_actual_score'],
                groups={name: dict(count=len(rows),
                                   mean=torch.tensor(rows).mean(0).tolist() if rows else None,
                                   p10_p50_p90=torch.tensor(rows).quantile(
                                       torch.tensor([.1, .5, .9]), dim=0).tolist() if rows else None)
                        for name, rows in groups.items()})


def main():
    import torch
    from tools.owod.protocol import file_sha256, stage_files
    from tools.owod.gnn_calibration import load_detector, build_calibration_dataset, move_targets
    from tools.owod.supervision_probe import probe_batch, summarize
    from models.graph_local.discovery_candidates import mine_candidates
    from util.box_ops import box_iou, box_cxcywh_to_xyxy
    from util.misc import collate_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, help='Save the diagnostic JSON, independently of training outputs')
    options = parser.parse_args()
    if options.images < 1:
        parser.error('--images must be positive')
    torch.set_num_threads(2)
    identity = file_sha256(options.checkpoint)
    payload = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    if (not getattr(saved, 'owod_discovery', False) or saved.owod_stage != 0
            or saved.owod_phase != 'increment' or getattr(saved, 'owod_screening', None)):
        raise ValueError('Expected a full-data discovery Task 1 checkpoint')
    if file_sha256(options.manifest) != saved.owod_manifest_sha256:
        raise ValueError('Checkpoint manifest mismatch')
    manifest, files = stage_files(options.manifest, 0)
    annotation_identity = file_sha256(files['increment_train'])
    known = manifest['stages'][0]['active_classes']
    if list(saved.owod_known_class_ids) != known:
        raise ValueError('Checkpoint class order mismatch')
    device = torch.device(options.device)
    model, criterion = load_detector(saved, options.checkpoint, device)
    model.eval()
    teacher = None
    if 'discovery_ema' in payload:
        from models.graph_local.discovery import make_ema
        teacher = make_ema(model)
        teacher.load_state_dict(payload['discovery_ema'], strict=True)
    from models.graph_local.discovery_candidates import flip_samples
    dataset = build_calibration_dataset(Path(manifest['image_root']), files['increment_train'])
    rows, image_ids, competition, candidate_rows = [], [], [], []
    counts = dict(queries=0, background_without_teacher=0, unmatched_known_overlap=0)
    with torch.no_grad():
        for index in range(min(options.images, len(dataset))):
            samples, targets = collate_fn([dataset[index]])
            targets = move_targets(targets, device)
            samples = samples.to(device)
            outputs = model(samples)
            rows.extend(probe_batch(outputs, targets, criterion.matcher, known))
            image_ids.append(int(targets[0]['image_id'].item()))
            indices = criterion.matcher(outputs, targets)
            competition.append(score_competition(outputs, targets, indices, known,
                                                 limit=saved.owod_max_detections,
                                                 scoring=getattr(saved, 'discovery_scoring', 'hierarchical')))
            candidates = mine_candidates(outputs, targets, indices, known)
            if teacher is not None:
                actual = mine_candidates(
                    outputs, targets, indices, known, teacher(flip_samples(samples)),
                    top_k=saved.discovery_top_k, saliency_threshold=saved.discovery_saliency_threshold,
                    consistency_threshold=saved.discovery_consistency_threshold)
                selected = torch.where(actual['pseudo'][0])[0]
                candidate_rows.append(dict(image_id=image_ids[-1],
                    candidates=[dict(query=int(q), box_cxcywh=outputs['pred_boxes'][0, q].tolist(),
                                     weight=float(actual['weights'][0, q]),
                                     saliency=float(actual['saliency'][0, q]),
                                     consistency=float(actual['consistency'][0, q]),
                                     semantic_weight=float(actual['semantic_weights'][0, q]),
                                     objectness=float(outputs['pred_objectness'][0, q].sigmoid()),
                                     novelty=float(outputs['pred_novelty'][0, q].sigmoid()))
                                for q in selected]))
            overlaps = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]),
                               box_cxcywh_to_xyxy(targets[0]['boxes']))[0]
            duplicate = (overlaps.amax(-1) > .3 if overlaps.shape[1] else
                         torch.zeros(overlaps.shape[0], device=device, dtype=torch.bool))
            duplicate[indices[0][0]] = False
            counts['queries'] += outputs['pred_boxes'].shape[1]
            counts['background_without_teacher'] += int(candidates['background'].sum())
            counts['unmatched_known_overlap'] += int(duplicate.sum())
    if identity != file_sha256(options.checkpoint):
        raise ValueError('Checkpoint changed during probe; use a numbered epoch checkpoint')
    if (file_sha256(options.manifest) != saved.owod_manifest_sha256
            or file_sha256(files['increment_train']) != annotation_identity):
        raise ValueError('Manifest or training annotations changed during probe')
    counts_anonymous = model.memory.counts[model.memory.known_slots:].cpu()
    report = dict(checkpoint_sha256=identity, epoch=payload['epoch'],
                          manifest_sha256=saved.owod_manifest_sha256,
                          train_annotation_sha256=annotation_identity,
                          discovery_version=getattr(saved, 'discovery_version', None),
                          pseudo_ramp_epochs=getattr(saved, 'discovery_pseudo_ramp_epochs', 0),
                          image_ids=image_ids, summary=summarize(rows), supervision=counts,
                          score_competition=summarize_competition(competition),
                          ema_candidates=dict(available=teacher is not None, images=candidate_rows,
                              pseudo_enabled_at_checkpoint=payload['epoch'] >= saved.discovery_warmup,
                              note='Reconstructed EMA evidence on deterministic training views; '
                                   'not the original augmented minibatches or candidate accuracy.'),
                          anonymous_memory=dict(support=counts_anonymous.tolist(),
                              supported_slots=int((counts_anonymous >= saved.discovery_min_support).sum()),
                              largest_support_fraction=(float(counts_anonymous.max() / counts_anonymous.sum())
                                                        if counts_anonymous.sum() > 0 else None)),
                          note='Training-image diagnostic only, not AP or U-Recall. '
                               'Deterministic dataset prefix; not representative accuracy. '
                               'background_without_teacher is an upper bound on two-view trust. '
                               'Coverage includes any correctly labelled IoU>0.5 candidate, '
                               'not one-to-one AP matching; score variants are diagnostic only. '
                               'No overlap with known GT does not prove background: future GT is never read.')
    rendered = json.dumps(report, indent=2) + '\n'
    if options.output:
        # A diagnostic destination must never replace a checkpoint or an input.
        if options.output.suffix.lower() != '.json' or options.output.exists():
            raise ValueError('Use a new .json diagnostic output path')
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
