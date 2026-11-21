"""Can a single detection from four fixed crops recover released small objects?"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_unknown_factors import best_target
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST
from tools.owod.run_s_task1_released_seed import OUTPUT as RELEASED
from tools.owod.run_s_task1_small_slot import OUTPUT as PREVIOUS
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata
from util.misc import nested_tensor_from_tensor_list


OUTPUT = PREVIOUS.with_name('s-instance-task1-crop-refill')


def crop_window(width, height, right, bottom):
    # Each crop covers 5/8 along each axis, sharing a strip across the midpoint.
    w, h = (5 * width + 7) // 8, (5 * height + 7) // 8
    x, y = (width - w if right else 0), (height - h if bottom else 0)
    return x, y, x + w, y + h


def crop_predictions(model, post, dataset, keys, sizes):
    photos = [dataset.get_image(dataset.coco.imgs[k]['file_name']) for k in keys]
    if [list(photo.size[::-1]) for photo in photos] != sizes.tolist():
        raise ValueError('Original image dimensions differ from the evaluation target')
    groups = [[] for _ in photos]
    for tile in range(4):
        right, bottom = tile % 2, tile // 2
        windows = [crop_window(*photo.size, right, bottom) for photo in photos]
        crops = [dataset._transforms(photo.crop(window), None)[0] for photo, window in zip(photos, windows)]
        out = model(nested_tensor_from_tensor_list(crops).to('cuda'))
        probabilities, categories, _ = post.candidate_scores(out)
        scores, labels = probabilities.max(-1)
        for i, ((x, y, x2, y2), photo) in enumerate(zip(windows, photos)):
            w, h = photo.size
            b = out['pred_boxes'][i]
            boxes = b * b.new_tensor([(x2-x)/w, (y2-y)/h, (x2-x)/w, (y2-y)/h])
            boxes[:, :2] += b.new_tensor([x/w, y/h])
            owned = ((boxes[:, 0] >= .5) == bool(right)) & ((boxes[:, 1] >= .5) == bool(bottom))
            # Keep a stable identity for the crop and its original query.
            ids = torch.where(owned)[0]
            groups[i].append((boxes[ids], scores[i, ids], categories[labels[i, ids]], ids + tile * len(b)))
    return [tuple(torch.cat(parts) for parts in zip(*group)) for group in groups]


def refill(candidates, native_boxes, native_labels, size, truth, designated):
    boxes, scores, labels, identities = candidates
    area = boxes[:, 2:].prod(-1) * size.prod()
    valid = (area > 0) & (area <= 2048)
    if len(boxes):
        valid &= overlap(boxes, native_boxes[:-1]).amax(1) < .5
    ids = torch.where(valid)[0]
    record = dict(pool=len(ids), candidate=None, pool_covers=False, winner_covers=False)
    if not len(ids):
        return native_boxes, native_labels, record
    j = ids[scores[ids].argmax()]
    record['candidate'] = dict(id=int(identities[j]), label=int(labels[j]), score=float(scores[j]), box=boxes[j].tolist())
    # The candidate is fixed before annotation-based measurements.
    if designated is not None:
        match = overlap(boxes[ids], truth[designated:designated + 1])[:, 0]
        record['pool_covers'] = bool((match >= .5).any())
        record['winner_covers'] = bool((match[ids == j] >= .5).item())
    return torch.cat((native_boxes[:-1], boxes[j].unsqueeze(0))), torch.cat((native_labels[:-1], labels[j].view(1))), record


def score(boxes, labels, truth, anns):
    assigned = best_target(boxes, truth).tolist()
    return sorted({anns[j]['id'] for j, label in zip(assigned, labels.tolist())
                   if j >= 0 and label == anns[j]['category_id']})


@torch.inference_mode()
def evaluate(model, post, loader, ref):
    images = {}
    totals = {name: Counter() for name in ('native', 'full', 'crops')}
    for step, (samples, targets) in enumerate(loader, 1):
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        keys = [int(t['image_id'].item()) for t in targets]
        native = model(samples.to('cuda'))
        dets = post(native, sizes, include_top_indices=True)
        probabilities, categories, _ = post.candidate_scores(native)
        scores, best = probabilities.max(-1)
        cropped = crop_predictions(model, post, loader.dataset, keys, sizes)
        for i, (key, target, det) in enumerate(zip(keys, targets, dets)):
            prior = ref[str(key)]
            anns = valid_annotations(loader.dataset, key)
            if target['labels'].tolist() != [a['category_id'] for a in anns]:
                raise ValueError('Changed released annotation order')
            truth = target['boxes'].cuda()
            eligible = prior['eligible_id']
            by_id = {a['id']: j for j, a in enumerate(anns)}
            designated = by_id[eligible] if eligible is not None else None
            small = {a['id'] for a in anns if a['area'] < 32**2}
            queries = det['top_indices'] // len(categories)
            boxes = native['pred_boxes'][i]
            selected, labels = boxes[queries], det['labels']
            if len(queries) != 50:
                raise ValueError('Expected the native joint Top-50')
            ids = torch.arange(len(boxes), device=boxes.device)
            ids = ids[~torch.isin(ids, queries)]
            full = (boxes[ids], scores[i, ids], categories[best[i, ids]], ids)
            outcomes = {}
            for name, pool in [('native', None), ('full', full), ('crops', cropped[i])]:
                if pool is None:
                    final, predictions, record = selected, labels, {}
                else:
                    final, predictions, record = refill(pool, selected, labels, sizes[i], truth, designated)
                hits = score(final, predictions, truth, anns)
                if name == 'native' and hits != prior['outcomes']['native']['known']:
                    raise ValueError(f'Changed native known detections on {key}')
                record.update(correct=hits, unknown_outputs=int((predictions == 91).sum()))
                outcomes[name] = record
                totals[name].update(images=1, known=len(hits), small_known=len(small.intersection(hits)),
                                    eligible_recovered=int(eligible in hits), unknown_outputs=record['unknown_outputs'],
                                    substituted=int(record.get('candidate') is not None),
                                    pool_covers=int(record.get('pool_covers', False)), winner_covers=int(record.get('winner_covers', False)))
            images[str(key)] = dict(eligible_id=eligible, outcomes=outcomes)
        if step == 1 or step % 32 == 0 or step == len(loader):
            print(f'{datetime.now().isoformat(timespec="seconds")} released {step}/{len(loader)} batches', flush=True)
    if images.keys() != ref.keys():
        raise ValueError('Incomplete released slice')
    return images, totals


def main():
    base = read_json(PREVIOUS / 'plan.json')
    recovery = read_json(PREVIOUS / 'recovery/plan.json') if (PREVIOUS / 'recovery/plan.json').exists() else None
    result = read_json(PREVIOUS / 'comparison.json')
    checkpoint = Path(base['job']['parent'])
    data = RELEASED / 'released2048.json'
    reference = PREVIOUS / 'released.json'
    digest = file_sha256(PREVIOUS / 'plan.json')
    if (result['status'] != 'complete' or result['follow_up'] or result['plan_sha256'] != digest
            or file_sha256(checkpoint) != base['job']['parent_sha256'] or file_sha256(data) != base['data_sha256']['released']
            or file_sha256(MANIFEST) != base['manifest_sha256'] or torch.cuda.get_device_name() != base['gpu']
            or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed previous endpoint, weights, data or environment')
    if recovery and (result['execution_commit'] != recovery['commit'] or recovery['original_plan_sha256'] != digest
                     or file_sha256(reference) != recovery['released_sha256']):
        raise ValueError('Changed recovered released endpoint')
    plan = {k: base[k] for k in ('job', 'seed', 'gpu', 'torch_version', 'batch_size', 'protocol', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), devices=os.environ['CUDA_VISIBLE_DEVICES'],
                reference_sha256=file_sha256(reference), parent_plan_sha256=digest, parent_result_sha256=file_sha256(PREVIOUS / 'comparison.json'),
                released_sha256=file_sha256(data), rule='Keep native top 49. Replace rank 50 by at most one small detection from unused '
                'full-frame queries or four overlapping 5/8-size crops transformed at 800/1333. Own quadrant by predicted center. '
                'Each query uses its highest native focal category score (including unknown); choose highest score among boxes '
                'of area <=2048 px^2 and IoU <.5 to every retained box. No candidate keeps native rank 50. Same detection budget, '
                'higher crop inference compute. Record pool coverage after selection; released-only diagnostic.',
                criterion='Only consider independent unknown-proxy evaluation if crops recover >=11/456 designated targets, '
                'recover strictly more than full-frame refill, increase all small-known hits over native, and maintain total known hits.')
    pin(OUTPUT / 'plan.json', plan)
    ref = read_json(reference)
    if ref['plan_sha256'] != digest or len(ref['images']) != 2048 or sum(r['eligible_id'] is not None for r in ref['images'].values()) != 456:
        raise ValueError('Unexpected released reference population')
    manifest = read_json(MANIFEST)
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    post = DiscoveryPostProcess(known_ids=manifest['stages'][0]['classes'], max_detections=50, scoring='focal')
    _, loader = make_loader(resolve_image_root(MANIFEST, manifest), data, plan['batch_size'], 2)
    rows, totals = evaluate(model, post, loader, ref['images'])
    digest = file_sha256(OUTPUT / 'plan.json')
    write_json(OUTPUT / 'paired.json', dict(plan_sha256=digest, images=rows))
    c, n = totals['crops'], totals['native']
    passes = (c['eligible_recovered'] >= 11 and c['eligible_recovered'] > totals['full']['eligible_recovered']
              and c['small_known'] > n['small_known'] and c['known'] >= n['known'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals,
                                                follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
