"""VOC-style OWOD evaluation following the author evaluators, not COCO AP.

Raw VOC boxes, difficult flags, score/coordinate quantization, classwise
confidence sorting, one-to-one TP matching, and WI at recall=0.8 are retained.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def voc_ap(recall, precision, mode):
    if mode == 'voc2007_11point':
        return sum(float(np.max(precision[recall >= t])) if np.any(recall >= t) else 0.0
                   for t in np.arange(0., 1.1, .1)) / 11.
    if mode != 'integral':
        raise ValueError(f'Unknown AP convention: {mode}')
    rec = np.concatenate(([0.], recall, [1.]))
    prec = np.concatenate(([0.], precision, [0.]))
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    changes = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[changes + 1] - rec[changes]) * prec[changes + 1]))


def overlaps(box, boxes):
    if not len(boxes):
        return np.empty(0)
    boxes = np.asarray(boxes, dtype=float)
    intersection = np.maximum(np.minimum(box[2:], boxes[:, 2:]) -
                              np.maximum(box[:2], boxes[:, :2]) + 1., 0.).prod(axis=1)
    union = (box[2:] - box[:2] + 1.).prod() + (boxes[:, 2:] - boxes[:, :2] + 1.).prod(axis=1) - intersection
    return intersection / np.maximum(union, np.finfo(float).eps)


def evaluate_author_benchmark(predictions, coco, known_ids, previous_ids, current_ids,
                              specification, *, include_current_stats=False, include_previous_stats=False,
                              include_wi_sources=False):
    known = set(known_ids)
    gt, source_to_id = {}, {}
    hidden_classes = defaultdict(list)
    for image_id, image in coco.imgs.items():
        source_to_id[image['owod_source_id']] = image_id
        objects = defaultdict(list)
        for ann in coco.imgToAnns.get(image_id, []):
            label = ann['category_id'] if ann['category_id'] in known else -1
            if 'voc_bbox' not in ann:
                raise ValueError('Author evaluation requires preserved raw voc_bbox annotations')
            objects[label].append((ann['voc_bbox'], bool(ann.get('difficult', 0))))
            if include_wi_sources and label == -1:
                hidden_classes[image_id].append(ann['category_id'])
        gt[image_id] = objects
    sequence = [source_to_id[stem] for stem in specification['image_source_ids']]
    if set(predictions) != set(sequence):
        raise ValueError('Evaluation predictions must cover the entire official test split')
    detections = defaultdict(list)
    # Preserve the literal released evaluation sequence. DDP sampler padding
    # is removed in engine.py before applying official list multiplicity here.
    for image_id in sequence:
        prediction = predictions[image_id]
        for score, label, box in zip(prediction['scores'].tolist(),
                                     prediction['labels'].tolist(), prediction['boxes'].tolist()):
            label = int(label)
            if label == 91:
                label = -1
            elif label not in known:
                raise ValueError(f'Prediction uses an unseen or invalid category: {label}')
            box = [float(f'{float(value) + (1. if i < 2 else 0.):.1f}')
                   for i, value in enumerate(box)]
            detections[label].append((float(f'{score:.3f}'), image_id, box))
    ap, curves = {}, {}
    a_ose, fp_at_recall, total_at_recall = 0, [], []
    current_stats, previous_stats = {}, {}
    wi_sources = {}
    unknown_gt = sum(sum(not difficult for _, difficult in gt[i][-1]) for i in sequence)
    unknown_tp = unknown_count = 0
    for label in [*known_ids, -1]:
        track_previous = include_previous_stats and label in previous_ids
        records = {i: {'boxes': [b for b, _ in gt[i][label]],
                       'difficult': [d for _, d in gt[i][label]],
                       'matched': set()} for i in gt}
        npos = sum(sum(not d for _, d in gt[i][label]) for i in sequence)
        items = detections[label]
        order = np.argsort(-np.asarray([item[0] for item in items]))
        tp, fp, fp_open = np.zeros(len(items)), np.zeros(len(items)), np.zeros(len(items))
        duplicate_fp = np.zeros(len(items)) if track_previous else None
        proxy_hits = []
        for rank, index in enumerate(order):
            _, image_id, box = items[index]
            record = records[image_id]
            ious = overlaps(np.asarray(box), record['boxes'])
            best = int(ious.argmax()) if len(ious) else -1
            if best >= 0 and ious[best] > specification['iou_threshold']:
                if not record['difficult'][best]:
                    if best in record['matched']:
                        fp[rank] = 1.
                        if track_previous:
                            duplicate_fp[rank] = 1.
                    else:
                        tp[rank] = 1.
                        record['matched'].add(best)
            else:
                fp[rank] = 1.
            if label != -1:
                unknown_iou = overlaps(np.asarray(box), [b for b, _ in gt[image_id][-1]])
                if len(unknown_iou) and unknown_iou.max() > specification['iou_threshold']:
                    fp_open[rank] = 1.
                    if include_wi_sources:
                        proxy_hits.append((rank, hidden_classes[image_id][int(unknown_iou.argmax())]))
        tp, fp, fp_open = np.cumsum(tp), np.cumsum(fp), np.cumsum(fp_open)
        recall = tp / max(1, npos)
        precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
        ap[label] = voc_ap(recall, precision, specification['ap_mode'])
        curves[label] = {'recall': recall, 'precision': precision}
        if include_current_stats and label in current_ids:
            current_stats[label] = dict(ap50=ap[label], gt=npos,
                                        tp=int(tp[-1]) if len(tp) else 0,
                                        fp=int(fp[-1]) if len(fp) else 0)
        if track_previous:
            # Hold the matched TP set fixed and rank every TP above the FPs.
            ceiling = voc_ap(recall[-1:], np.ones(1), specification['ap_mode']) if len(tp) and tp[-1] else 0.
            filtered_fp = fp - np.cumsum(duplicate_fp)
            filtered_precision = tp / np.maximum(tp + filtered_fp, np.finfo(float).eps)
            previous_stats[label] = dict(ap50=ap[label], gt=npos, recall_ceiling_ap50=ceiling,
                                         ap50_without_duplicate_fp=voc_ap(recall, filtered_precision, specification['ap_mode']),
                                         tp=int(tp[-1]) if len(tp) else 0, fp=int(fp[-1]) if len(fp) else 0,
                                         fp_duplicates=int(duplicate_fp.sum()))
        if label == -1:
            unknown_tp = int(tp[-1]) if len(tp) else 0
            unknown_count = len(items)
        else:
            a_ose += int(fp_open[-1]) if len(fp_open) else 0
            if len(recall):
                index = int(np.argmin(np.abs(recall - specification['wi_recall'])))
                fp_at_recall.append(fp_open[index])
                total_at_recall.append(tp[index] + fp[index])
                if include_wi_sources:
                    by_proxy = defaultdict(int)
                    for rank, proxy_class in proxy_hits:
                        if rank <= index:
                            by_proxy[proxy_class] += 1
                    wi_sources[label] = dict(rank=index + 1, recall=float(recall[index]),
                                             max_recall=float(recall[-1]), open=int(fp_open[index]),
                                             total=int(tp[index] + fp[index]), by_proxy=dict(by_proxy),
                                             full_open=int(fp_open[-1]))
    group_ap = {}
    for name, ids in [('Previous', previous_ids), ('Current', current_ids), ('Known', known_ids)]:
        if ids:
            group_ap[name] = float(np.mean([ap[c] for c in ids]))
    result = {f'{name.lower()}_ap50': value for name, value in group_ap.items()}
    if include_wi_sources:
        result['wi_sources'] = wi_sources
    if include_current_stats:
        result['current_class_stats'] = current_stats
    if include_previous_stats:
        result['previous_class_stats'] = previous_stats
    if specification.get('known_only_screening'):
        return result
    result.update({'unknown_gt': float(unknown_gt), 'unknown_predictions': float(unknown_count)})
    if unknown_gt:
        result.update({'u_recall': unknown_tp / unknown_gt,
                       'unknown_precision': unknown_tp / max(1, unknown_count),
                       'a_ose': float(a_ose),
                       'wi': float(sum(fp_at_recall) / max(1., sum(total_at_recall)))})
        k, u = group_ap['Known'], result['u_recall']
        result['h_score'] = 2 * k * u / (k + u) if k + u else 0.
    return result
