"""Detached unknown evidence with padded-image-safe cross-view matching."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import NestedTensor


@torch.no_grad()
def semantic_weights(outputs, targets, indices, known_ids):
    """Relative known confidence from released GT; never a hard unknown veto.

    Focal logits are not calibrated probabilities. Their foreground reference
    is the matched, correctly labelled train objects in this very batch.
    Prototype similarity contributes evidence without bootstrapping on an
    untrained novelty head. A floor prevents both cues from starving discovery.
    """
    logits = outputs['pred_logits'][..., known_ids].amax(-1)
    anchors = []
    for batch, (queries, matches) in enumerate(indices):
        if len(queries):
            anchors.append(outputs['pred_logits'][batch, queries, targets[batch]['labels'][matches]])
    if anchors:
        anchors = torch.cat(anchors).detach()
        reference = anchors.median()
        scale = (anchors - reference).abs().median().clamp_min(.5)
        confidence = ((logits - reference) / scale).sigmoid()
    else:
        confidence = logits.sigmoid()
    novelty = 1 - confidence
    affinities = outputs.get('affinities')
    if affinities is not None:
        known_affinity = affinities[..., known_ids]
        valid = known_affinity.amax(-1) > -100
        distance = (1 - known_affinity.amax(-1)).clamp(0, 2) / 2
        novelty = torch.where(valid, (novelty + distance) / 2, novelty)
    return .25 + .75 * novelty.clamp(0, 1)


def flip_samples(samples):
    image = samples.tensors.clone()
    for batch, mask in enumerate(samples.mask):
        height = int((~mask).any(1).sum())
        width = int((~mask).any(0).sum())
        image[batch, :, :height, :width] = image[batch, :, :height, :width].flip(-1)
    return NestedTensor(image, samples.mask.clone())


@torch.no_grad()
def saliency_in_boxes(feature, boxes, image_sizes, padded_size):
    heat = feature.detach().abs().mean(1)
    results = []
    for batch, (height, width) in enumerate(image_sizes):
        fh = max(1, round(heat.shape[1] * float(height) / padded_size[0]))
        fw = max(1, round(heat.shape[2] * float(width) / padded_size[1]))
        valid = heat[batch, :fh, :fw]
        low, high = torch.quantile(valid.flatten().float(), valid.new_tensor([.1, .9]).float())
        normalized = ((valid - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        xyxy = box_cxcywh_to_xyxy(boxes[batch]).clamp(0, 1)
        points = torch.linspace(.15, .85, 3, device=boxes.device)
        x = xyxy[:, 0, None] + (xyxy[:, 2] - xyxy[:, 0])[:, None] * points
        y = xyxy[:, 1, None] + (xyxy[:, 3] - xyxy[:, 1])[:, None] * points
        grid = torch.stack([x[:, None, :].expand(-1, 3, -1),
                            y[:, :, None].expand(-1, -1, 3)], -1).reshape(1, -1, 9, 2) * 2 - 1
        values = F.grid_sample(normalized[None, None], grid, align_corners=False)
        results.append(values[0, 0].mean(-1))
    return torch.stack(results)


@torch.no_grad()
def view_matches(outputs, teacher):
    boxes = teacher['pred_boxes'].detach().clone()
    boxes[..., 0] = 1 - boxes[..., 0]
    matches, confidence = [], []
    for batch in range(boxes.shape[0]):
        corners, valid_ids = [], []
        for name, prediction, features in (
                ('student', outputs['pred_boxes'][batch], outputs['query_features'][batch]),
                ('teacher', boxes[batch], teacher['query_features'][batch])):
            if not torch.isfinite(prediction).all() or not torch.isfinite(features).all():
                raise ValueError(f'Cross-view matching: non-finite {name} boxes/features in batch {batch}')
            # Convert before geometry arithmetic to avoid half-precision collapse.
            # Even finite positive cxcywh sizes can round to zero in xyxy.
            xyxy = box_cxcywh_to_xyxy(prediction.float())
            size = xyxy[:, 2:] - xyxy[:, :2]
            valid = (size > 0).all(-1) & (size.prod(-1) > 0)
            corners.append(xyxy)
            valid_ids.append(torch.where(valid)[0])
        student_ids, teacher_ids = valid_ids
        mapping = torch.full((len(corners[0]),), -1, device=boxes.device, dtype=torch.long)
        score = corners[0].new_zeros(len(corners[0]))
        if student_ids.numel() and teacher_ids.numel():
            overlaps = box_iou(corners[0][student_ids], corners[1][teacher_ids])[0]
            if not torch.isfinite(overlaps).all():
                raise ValueError(f'Cross-view matching: non-finite IoU in batch {batch}')
            source, target = linear_sum_assignment(-overlaps.cpu().numpy())
            source = torch.as_tensor(source, device=boxes.device)
            target = torch.as_tensor(target, device=boxes.device)
            queries, partners = student_ids[source], teacher_ids[target]
            mapping[queries] = partners
            cosine = F.cosine_similarity(outputs['query_features'][batch, queries].float(),
                                         teacher['query_features'][batch, partners].float())
            score[queries] = overlaps[source, target] * cosine.clamp(0, 1)
        matches.append(mapping)
        confidence.append(score)
    return torch.stack(matches), torch.stack(confidence), boxes


def select_distinct_boxes(boxes, eligible, scores, limit):
    ranked = torch.where(eligible)[0]
    ranked = ranked[torch.argsort(scores[ranked], descending=True)]
    chosen = []
    for query in ranked:
        if len(chosen) >= limit:
            break
        if chosen:
            overlap = box_iou(box_cxcywh_to_xyxy(boxes[query:query + 1]),
                              box_cxcywh_to_xyxy(boxes[torch.stack(chosen)]))[0]
            if overlap.max() > .7:
                continue
        chosen.append(query)
    return torch.stack(chosen) if chosen else ranked[:0]


@torch.no_grad()
def mine_candidates(outputs, targets, indices, known_ids, teacher=None, top_k=5,
                    saliency_threshold=.65, consistency_threshold=.5, *, gate_trace=None):
    saliency = saliency_in_boxes(outputs['attention_feature'], outputs['pred_boxes'],
                                outputs['image_sizes'], outputs['padded_size'])
    known = outputs['pred_logits'][..., known_ids].sigmoid().amax(-1)
    semantic = semantic_weights(outputs, targets, indices, known_ids)
    pseudo = torch.zeros_like(known, dtype=torch.bool)
    background = torch.zeros_like(pseudo)
    weights = torch.zeros_like(known)
    maps, consistency, teacher_boxes = (None, torch.zeros_like(known), None)
    teacher_saliency = None
    if teacher is not None:
        maps, consistency, teacher_boxes = view_matches(outputs, teacher)
        teacher_saliency = saliency_in_boxes(teacher['attention_feature'], teacher['pred_boxes'],
                                            teacher['image_sizes'], teacher['padded_size'])
    for batch, (queries, _) in enumerate(indices):
        box = outputs['pred_boxes'][batch]
        valid = torch.isfinite(box).all(-1) & (box[:, 2:] > .005).all(-1)
        valid &= box[:, 2:].prod(-1) < .85
        exclude = torch.zeros_like(valid)
        exclude[queries] = True
        if targets[batch]['boxes'].numel():
            overlap = box_iou(box_cxcywh_to_xyxy(box),
                              box_cxcywh_to_xyxy(targets[batch]['boxes']))[0]
            exclude |= overlap.amax(-1) > .3
        eligible = valid & ~exclude
        salient = saliency[batch] >= saliency_threshold
        consistent = consistency[batch] >= consistency_threshold
        confident = eligible & salient & consistent
        if teacher_saliency is not None:
            matched_saliency = teacher_saliency[batch, maps[batch].clamp_min(0)]
            other_salient = (maps[batch] >= 0) & (matched_saliency >= saliency_threshold)
            confident &= other_salient
        if gate_trace is not None:
            gate_trace.append(dict(valid=valid, eligible=eligible, salient=salient, consistent=consistent,
                                   other_salient=other_salient if teacher_saliency is not None else torch.ones_like(valid),
                                   confident=confident,
                                   matched_saliency=matched_saliency if teacher_saliency is not None else None))
        quality = saliency[batch] * consistency[batch] * semantic[batch]
        if 'geometry_quality' in outputs:
            quality = quality * outputs['geometry_quality'][batch].detach()
        chosen = select_distinct_boxes(box, confident, quality, top_k)
        if chosen.numel():
            pseudo[batch, chosen] = True
            weights[batch, chosen] = quality[chosen]
        # Lack of a semantic label is insufficient evidence of background.
        background[batch] = eligible & (saliency[batch] < .15)
        if teacher is not None:
            background[batch] &= maps[batch] >= 0
            background[batch] &= teacher_saliency[batch, maps[batch].clamp_min(0)] < .15
    return {'pseudo': pseudo, 'background': background, 'weights': weights,
            'matches': maps, 'teacher_boxes': teacher_boxes, 'consistency': consistency,
            'saliency': saliency, 'semantic_weights': semantic}
