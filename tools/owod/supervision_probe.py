"""GT-conditioned read-only diagnostics, not detection AP or unknown recall."""
import torch
from util.box_ops import box_cxcywh_to_xyxy, box_iou


def current_endpoint_hits(outputs, batch, target, result, known, current, top300=None, *,
                          include_classification=False, score_table=None):
    """Per current GT: box, correct argmax, and coverage at each output budget."""
    labels = target['labels']
    keep = torch.isin(labels, labels.new_tensor(current))
    labels = labels[keep]
    gt = box_cxcywh_to_xyxy(target['boxes'][keep])
    raw = box_cxcywh_to_xyxy(outputs['pred_boxes'][batch])
    overlaps = box_iou(raw, gt)[0]
    covers = overlaps > .5
    classes = labels.new_tensor(known)
    argmax = classes[outputs['pred_logits'][batch][:, classes].argmax(dim=1)]
    height, width = target['orig_size'].unbind()
    scale = torch.stack((width, height, width, height))

    def output_hits(predictions):
        boxes = predictions['boxes'] / scale
        return ((box_iou(boxes, gt)[0] > .5) & (predictions['labels'][:, None] == labels)).any(0)

    hits = [labels, covers.any(0), (covers & (argmax[:, None] == labels)).any(0), output_hits(result)]
    if top300 is not None:
        hits.append(output_hits(top300))
    if include_classification:
        best_class = argmax[overlaps.argmax(0)]
        hits.append(torch.where(covers.any(0), best_class, -1))
    if score_table is not None:
        ranks = torch.zeros_like(labels)
        rows = current_candidate_rank_rows(labels, overlaps, batch, [score_table])
        ranks[covers.any(0)] = rows[:, 1].to(ranks.device)
        hits.append(ranks)
    return torch.stack(hits, dim=1)


def summarize_endpoint(image_hits, current, *, include_classification=False, include_ranks=False):
    hits = torch.cat(image_hits)
    if include_ranks:
        ranks, hits = hits[:, -1], hits[:, :-1]
    coverage = hits[:, :-1] if include_classification else hits
    names = ['any_box', 'correct_argmax', 'final_output']
    if coverage.shape[1] == 5:
        names.append('top300_output')

    def counts(rows):
        return dict(gt=len(rows), **dict(zip(names, rows[:, 1:].sum(0).tolist())))
    report = dict(images=len(image_hits), overall=counts(coverage),
                  per_class={str(c): counts(coverage[coverage[:, 0] == c]) for c in current})
    if include_ranks:
        covered_ranks = ranks[ranks > 0]
        report['rank_at_iou50'] = dict(
            gt_with_box=len(covered_ranks),
            median_best_true_class_rank=float(covered_ranks.float().quantile(.5)) if len(covered_ranks) else None)
    if include_classification:
        covered = hits[hits[:, -1] >= 0]
        labels, predicted = covered[:, 0], covered[:, -1]
        correct = predicted == labels
        predicted_current = torch.isin(predicted, predicted.new_tensor(current))
        per_class = {}
        for c in current:
            ids, totals = predicted[labels == c].unique(return_counts=True)
            per_class[str(c)] = dict(zip(map(str, ids.tolist()), totals.tolist()))
        report['classification_at_best_iou'] = {
            'overall': dict(best_iou_correct=int(correct.sum()), predicted_previous=int((~predicted_current).sum()),
                            predicted_other_current=int((predicted_current & ~correct).sum())),
            'predicted_class_counts_by_gt': per_class,
            'definition': ('Among current GT with IoU > 0.5, use the query with the largest IoU '
                           'and its top known-class logit. The table maps GT class IDs to predicted class counts.'),
        }
    return report


def current_candidate_rank_rows(labels, overlap, index, score_tables):
    """Rank the best true-class candidate covering each GT across the whole image."""
    covered = overlap > .5
    keep = covered.any(0)
    labels, covered = labels[keep], covered[:, keep]
    rows = [labels]
    for batch_scores, categories, top_indices in score_tables:
        scores = batch_scores[index]
        columns = (categories[:, None] == labels).nonzero(as_tuple=True)[0]
        best = scores[:, columns].masked_fill(~covered, -torch.inf).amax(0)
        flat = scores.flatten()
        # Count scores strictly above the best match and measure the actual
        # selected set separately, so ties at the cutoff follow torch.topk.
        rank = (flat[:, None] > best).sum(0) + 1
        selected = top_indices[index]
        queries = selected // len(categories)
        hits = (covered[queries] & ((selected % len(categories))[:, None] == columns)).any(0)
        rows.extend((rank, hits))
    return torch.stack(rows, dim=1).cpu()


def summarize_candidate_ranks(rows, view_names, current):
    values = torch.cat(rows) if rows else torch.empty((0, 5), dtype=torch.long)

    def summarize_group(group):
        reference_rank, reference_hit, treatment_rank, treatment_hit = group[:, 1:].unbind(1)
        variants = {}
        for name, rank, hit in zip(view_names, (reference_rank, treatment_rank), (reference_hit, treatment_hit)):
            variants[name] = dict(median_best_rank=float(rank.float().quantile(.5)) if len(rank) else None,
                                  top50_hits=int(hit.sum()))
        return dict(gt_with_box=len(group), views=variants,
                    rank_improved=int((treatment_rank < reference_rank).sum()),
                    rank_worsened=int((treatment_rank > reference_rank).sum()),
                    entered_top50=int(((reference_hit == 0) & (treatment_hit == 1)).sum()),
                    left_top50=int(((reference_hit == 1) & (treatment_hit == 0)).sum()))

    return dict(overall=summarize_group(values),
                per_class={str(c): summarize_group(values[values[:, 0] == c]) for c in current},
                definition=('Per unique test image, valid new-class GT with a query at IoU > 0.5. '
                            'Best rank is 1 plus the number of strictly higher known/unknown scores. '
                            'Top-50 hits use the actual selected candidates.'))


@torch.no_grad()
def probe_batch(outputs, targets, matcher, known):
    """One row per visible GT. Detection coverage is not one-to-one AP matching."""
    indices = matcher(outputs, targets)
    rows = []
    categories = torch.as_tensor(known, device=outputs['pred_logits'].device)
    for b, target in enumerate(targets):
        logits = outputs['pred_logits'][b, :, known]
        labels = target['labels'].to(logits.device)
        if not torch.isin(labels, categories).all():
            raise ValueError('Probe received a label outside the released known classes')
        if not len(labels):
            continue
        overlaps = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][b]),
                          box_cxcywh_to_xyxy(target['boxes'].to(logits.device)))[0]
        obj = outputs['pred_objectness'][b].sigmoid()
        novelty = outputs['pred_novelty'][b].sigmoid()
        unknown = obj * novelty
        predictions = categories[logits.argmax(-1)]
        scores = {'hierarchical': logits.softmax(-1) * (obj * (1 - novelty))[:, None],
                  'focal': logits.sigmoid() * obj[:, None],
                  # Isolate novelty suppression: same softmax and same unknown score.
                  'hierarchical_without_novelty_gate': logits.softmax(-1) * obj[:, None]}
        selected = {}
        for mode, score in scores.items():
            combined = torch.cat((score, unknown[:, None]), -1)
            for limit in (50, 300):
                keep = torch.zeros_like(combined, dtype=torch.bool).flatten()
                keep[combined.flatten().topk(min(limit, combined.numel())).indices] = True
                selected[f'{mode}_top{limit}'] = keep.reshape_as(combined)[:, :-1]
        matched = {int(t): int(q) for q, t in zip(*indices[b])}
        for t, label in enumerate(labels):
            column = int(torch.where(categories == label)[0][0])
            covered = overlaps[:, t] > .5
            row = {'image_id': int(target['image_id'].item()), 'class_id': int(label),
                   'gt_index': t, 'any_box_iou50': bool(covered.any()),
                   'correct_argmax_box_iou50': bool((covered & (predictions == label)).any()),
                   'best_iou': float(overlaps[:, t].max()), 'matched': t in matched}
            for name, keep in selected.items():
                row[name] = bool((covered & keep[:, column]).any())
            if t in matched:
                q = matched[t]
                row.update(matched_iou=float(overlaps[q, t]),
                           matched_correct_argmax=bool(predictions[q] == label),
                           matched_objectness=float(obj[q]), matched_novelty=float(novelty[q]),
                           matched_gt_sigmoid=float(logits[q, column].sigmoid()),
                           matched_gt_softmax=float(logits[q].softmax(-1)[column]),
                           matched_novelty_above_half=bool(novelty[q] > .5))
            rows.append(row)
    return rows


def summarize(rows):
    flags = ['any_box_iou50', 'correct_argmax_box_iou50', 'matched'] + [
        f'{mode}_top{limit}' for mode in ('hierarchical', 'focal', 'hierarchical_without_novelty_gate')
        for limit in (50, 300)]
    matched = [r for r in rows if r['matched']]
    result = {'gt_count': len(rows), 'counts': {key: sum(r[key] for r in rows) for key in flags},
              'percent_of_gt': {key: 100 * sum(r[key] for r in rows) / len(rows) if rows else None
                                for key in flags}}
    keys = ('matched_iou', 'matched_objectness', 'matched_novelty', 'matched_gt_sigmoid',
            'matched_gt_softmax', 'matched_correct_argmax', 'matched_novelty_above_half')
    result['matched_means'] = {key: sum(r[key] for r in matched) / len(matched) if matched else None
                               for key in keys}
    return result


def run_probe(args, saved, checkpoint, manifest, identity):
    import json
    from tools.owod.diagnose_learning import make_loader
    from tools.owod.manage_data import prepare_fit_data
    from tools.owod.gnn_calibration import load_detector, move_targets
    from tools.owod.screening_data import write_json
    annotation = prepare_fit_data(args.manifest, args.screening, args.output / 'data', args.images, args.seed)
    _, loader = make_loader(manifest['image_root'], annotation, args.batch_size, args.workers)
    model, criterion = load_detector(saved, checkpoint, torch.device(args.device))
    model.eval()
    rows = []
    with torch.no_grad(), (args.output / 'objects.jsonl').open('w') as handle:
        for index, (samples, targets) in enumerate(loader, 1):
            targets = move_targets(targets, torch.device(args.device))
            outputs = model(samples.to(args.device))
            batch = probe_batch(outputs, targets, criterion.matcher, saved.owod_known_class_ids)
            rows.extend(batch)
            for row in batch:
                handle.write(json.dumps(row) + '\n')
            if index % 20 == 0:
                print(f'Probe: {index}/{len(loader)} batches, {len(rows)} visible GT', flush=True)
    report = {'overall': summarize(rows),
              'per_class': {str(c): summarize([r for r in rows if r['class_id'] == c])
                            for c in saved.owod_current_class_ids},
              'note': 'Current-class GT from deterministic increment training subset; eval transforms, '
                      'no optimization, no EMA reconstruction. GT coverage is not AP/U-Recall; '
                      'one query may cover multiple GT. Counterfactual gate removal is diagnostic only.',
              **identity}
    write_json(args.output / 'supervision.json', report)
    write_json(args.output / 'complete.json', {'kind': 'supervision_probe', **identity})
    print(json.dumps(report['overall'], indent=2), flush=True)
