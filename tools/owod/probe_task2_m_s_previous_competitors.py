"""Count old negatives that occupy slots ahead of matched new-class training targets."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader, Subset

from datasets import build_dataset
from main import get_args_parser
from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_previous_hard_negative_misses import SCORES
from tools.owod.run_cross_attention_study import pin
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_task2_m_s_previous_hard_negatives import OUTPUT as TRAINING
from tools.owod.screening_data import write_json
from tools.owod.strict_setting import configure_run
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import collate_fn


OUTPUT = TRAINING.with_name('task2-m-s-previous-competitor-training-probe')
RANK_OUTPUT = TRAINING.with_name('task2-m-s-current-training-ranks')
COMPETITOR_SHA = '7138a048c637626791f9ab8ee55d5edc7f821517f3ea1e12d69873481491f04e'
SCORE_SHA = '03a17968abd29ece715bc40410166689bc832a23b2b614dcf3b9a4f19b8fb244'
TRAIN_PLAN_SHA = '9b44fe67f57833d067c1534381695ee09cd93ed98691cf123ba942d518fa7c83'
DEVICE = torch.device('cuda')
SAMPLE_IMAGES = 256
SEED = 42
TOP50 = 50
MAX_CANDIDATE_RANK = 60
MIN_ADMITTED_IMAGES = 5
IOU = .5
RANK_MISSES = ('no_compatible_box', 'rank_51_to_60', 'rank_61_to_100', 'rank_past_100')


def select_old_negatives(scores, boxes, target, matches, known, previous_count):
    """Find the fewest eligible old scores whose removal admits the nearest current match."""
    queries, gt = (indices.to(boxes.device) for indices in matches)
    ious, _ = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target['boxes']))
    labels = target['labels']
    columns = (labels[gt, None] == known).long().argmax(dim=1)
    current = (columns >= previous_count) & (ious[queries, gt] > IOU)
    class_count = scores.shape[1]
    gold = queries[current] * class_count + columns[current]
    flat = scores.flatten()
    native = flat.topk(TOP50).indices
    near = flat.topk(MAX_CANDIDATE_RANK).indices
    contenders = near[TOP50:]
    candidate = torch.isin(contenders, gold) & ~torch.isin(contenders, native)
    distances = candidate.nonzero().flatten()
    if not distances.numel():
        return dict(near_current=0, admitted_current=0, old_terms=[])

    old_slots = native[native.remainder(class_count) < previous_count]
    old_queries = old_slots.div(class_count, rounding_mode='floor')
    old_labels = known[old_slots.remainder(class_count)]
    near_old = ((ious[old_queries] > IOU) & (old_labels[:, None] == labels)).any(dim=1)
    eligible = old_slots[~near_old & ~torch.isin(old_queries, queries)]
    needed = int(distances[0]) + 1
    if len(eligible) < needed:
        return dict(near_current=int(distances.numel()), admitted_current=0, old_terms=[])

    removed = eligible[-needed:]
    patched = flat.clone()
    patched[removed] = -torch.inf
    refreshed = patched.topk(TOP50).indices
    admitted = torch.isin(gold, refreshed) & ~torch.isin(gold, native)
    if not bool(admitted.any()) or bool(torch.isin(removed, refreshed).any()):
        raise ValueError('A planned removal failed to admit a current match')
    return dict(near_current=int(distances.numel()), admitted_current=int(admitted.sum()),
                old_terms=removed.tolist(), first_admitted_candidate_rank=TOP50+needed)


def current_annotation_ranks(scores, boxes, target, matches, known, previous_count):
    queries, matched_gt = (indices.to(boxes.device) for indices in matches)
    ious, _ = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target['boxes']))
    class_count = scores.shape[-1]
    flat = scores.flatten()
    native = flat.topk(TOP50).indices
    outside = flat.clone()
    outside[native] = -torch.inf
    order = outside.argsort(descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(TOP50 + 1, TOP50 + len(order) + 1, device=order.device)
    ranks[native] = torch.arange(1, TOP50 + 1, device=order.device)
    assigned = dict(zip(matched_gt.tolist(), queries.tolist(), strict=True))
    current_classes = {label: i for i, label in enumerate(known) if i >= previous_count}
    rows = []
    for gt, label in enumerate(target['labels'].tolist()):
        if label not in current_classes:
            continue
        position = current_classes[label]
        compatible = (ious[:, gt] > IOU).nonzero().flatten()
        best_rank = int(ranks[compatible * class_count + position].min()) if compatible.numel() else None
        matched = assigned.get(gt)
        assigned_iou = float(ious[matched, gt]) if matched is not None else None
        assigned_rank = int(ranks[matched * class_count + position]) if matched is not None else None
        rows.append(dict(target=gt, category=label, compatible_queries=int(compatible.numel()),
                         maximum_iou=float(ious[:, gt].max()), best_rank=best_rank,
                         assigned_iou=assigned_iou, assigned_rank=assigned_rank))
    return rows


def prepare(rank_analysis):
    evidence = verified(SCORES / 'comparison.json', SCORE_SHA)
    source = verified(TRAINING / 'plan.json', TRAIN_PLAN_SHA)
    if evidence['status'] != 'complete' or evidence['preferred'] != ['class_probability']:
        raise ValueError('Expected the dual-protocol class-probability result')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen training probe')
    jobs, datasets = {}, {}
    for protocol, job in source['jobs'].items():
        command = job['command']
        argv = next(i for i, token in enumerate(command) if Path(token).name == 'main.py') + 1
        args = get_args_parser().parse_args(command[argv:])
        configure_run(args)
        if file_sha256(Path(args.train_ann)) != job['train_sha256']:
            raise ValueError(f'Changed balanced finetune annotations: {protocol}')
        endpoint = job['endpoints']['increment']
        if file_sha256(Path(endpoint['path'])) != endpoint['sha256']:
            raise ValueError(f'Changed increment endpoint: {protocol}')
        dataset = build_dataset('train', args)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))[:SAMPLE_IMAGES].tolist()
        if len(indices) != SAMPLE_IMAGES:
            raise ValueError(f'Insufficient balanced training images: {protocol}')
        jobs[protocol] = dict(checkpoint=endpoint, train_sha256=job['train_sha256'], indices=indices,
                              image_ids=[dataset.ids[i] for i in indices], previous=job['previous'], current=job['current'])
        datasets[protocol] = Subset(dataset, indices)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                score_result_sha256=SCORE_SHA, train_plan_sha256=TRAIN_PLAN_SHA, jobs=jobs, seed=SEED, batch=2,
                device=torch.cuda.get_device_name(), torch_version=str(torch.__version__), sample_images=SAMPLE_IMAGES,
                native_limit=TOP50, maximum_rank=MAX_CANDIDATE_RANK, minimum_iou=IOU,
                minimum_admitted_images=MIN_ADMITTED_IMAGES, paper_comparable=False)
    output = OUTPUT
    if rank_analysis:
        output = RANK_OUTPUT
        reference = verified(OUTPUT / 'comparison.json', COMPETITOR_SHA)
        prior = verified(OUTPUT / 'plan.json', reference['plan_sha256'])
        if reference['follow_up'] or prior['jobs'] != jobs:
            raise ValueError('Expected the same frozen balanced training sample and endpoint')
        plan.update(competitor_result_sha256=COMPETITOR_SHA, competitor_plan_sha256=reference['plan_sha256'],
                    observation='current GT coverage, assigned query IoU and best compatible true-class score rank',
                    decision='Prefer geometry or ranking only if it accounts for at least half of the missing current annotations in both protocols.')
    pin(output / 'plan.json', plan)
    return plan, datasets


@torch.inference_mode()
def probe(protocol, job, dataset, rank_analysis):
    model, criterion = load_detector(None, Path(job['checkpoint']['path']), DEVICE)
    model.eval()
    base = criterion.base
    known = job['previous'] + job['current']
    if list(base.active_class_ids) != known:
        raise ValueError(f'Changed active class mapping: {protocol}')
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=TOP50)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(SEED))
    if not rank_analysis:
        base.enable_previous_hard_negatives(job['previous'])
        base.to(DEVICE)
    classes = torch.tensor(known, device=DEVICE)
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        targets = move_targets(targets, DEVICE)
        if any(not bool(torch.isin(t['labels'], classes).all()) for t in targets):
            raise ValueError(f'Unreleased target supervision: {protocol}')
        outputs = model(samples.to(DEVICE))
        scores, _, _ = post.candidate_scores(outputs)
        matches = base.matcher(outputs, targets)
        boxes = outputs['pred_boxes']
        if not rank_analysis:
            original_mask = base.previous_hard_negative_mask(outputs['pred_logits'][..., known], boxes, targets, matches, None)
        for i, (target, pair) in enumerate(zip(targets, matches, strict=True)):
            image = int(target['image_id'])
            if rank_analysis:
                measured = current_annotation_ranks(scores[i], boxes[i], target, pair, known, len(job['previous']))
                rows.append(dict(image=image, current_annotations=measured))
            else:
                result = select_old_negatives(scores[i], boxes[i], target, pair, classes, len(job['previous']))
                rows.append(dict(image=image, original_old_terms=int(original_mask[i].sum()), **result))
        if step % 32 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} training images', flush=True)
    if [r['image'] for r in rows] != job['image_ids']:
        raise ValueError(f'Changed sample order: {protocol}')
    return rows


def rank_range(rank):
    if rank is None:
        return 'no_compatible_box'
    if rank <= TOP50:
        return 'native_top50'
    if rank <= MAX_CANDIDATE_RANK:
        return 'rank_51_to_60'
    if rank <= 100:
        return 'rank_61_to_100'
    return 'rank_past_100'


def summarize_ranks(rows):
    annotations = [target for row in rows for target in row['current_annotations']]
    best = Counter(rank_range(a['best_rank']) for a in annotations)
    assigned = Counter(rank_range(a['assigned_rank'] if (a['assigned_iou'] or 0) > IOU else None)
                       for a in annotations)
    return dict(images=len(rows), current_annotations=len(annotations), best_compatible_rank=dict(best),
                matched_query_rank=dict(assigned))


def main(rank_analysis=False):
    output = RANK_OUTPUT if rank_analysis else OUTPUT
    plan, datasets = prepare(rank_analysis)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        marker = output / f'{protocol}.json'
        if not marker.exists():
            print(f'Evaluating {protocol} balanced finetune training images', flush=True)
            rows = probe(protocol, job, datasets[protocol], rank_analysis)
            counts = summarize_ranks(rows) if rank_analysis else dict(
                          images=len(rows), images_near=sum(r['near_current'] > 0 for r in rows),
                          images_admitted=sum(r['admitted_current'] > 0 for r in rows),
                          matches_admitted=sum(r['admitted_current'] for r in rows),
                          old_terms=sum(len(r['old_terms']) for r in rows),
                          original_old_terms=sum(r['original_old_terms'] for r in rows))
            write_json(marker, dict(plan_sha256=digest, rows=rows, counts=counts))
        report = read_json(marker)
        if report['plan_sha256'] != digest or report['counts']['images'] != SAMPLE_IMAGES:
            raise ValueError(f'Unexpected report: {protocol}')
        protocols[protocol] = report['counts']
    final = dict(status='complete', plan_sha256=digest, protocols=protocols)
    if rank_analysis:
        geometry_majority, rank_majority = [], []
        for rows in protocols.values():
            best = rows['best_compatible_rank']
            missing = sum(best.get(k, 0) for k in RANK_MISSES)
            geometry = best.get('no_compatible_box', 0)
            geometry_majority.append(missing > 0 and geometry * 2 >= missing)
            rank_majority.append(missing > 0 and (missing - geometry) * 2 >= missing)
        final['preferred'] = [name for name, flags in (('geometry', geometry_majority), ('ranking', rank_majority))
                              if all(flags)]
    else:
        final['follow_up'] = all(r['images_admitted'] >= MIN_ADMITTED_IMAGES for r in protocols.values())
    write_json(output / 'comparison.json', final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-ranks', action='store_true')
    main(parser.parse_args().current_ranks)
