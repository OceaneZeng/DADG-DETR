"""Count old negatives that occupy slots ahead of matched new-class training targets."""

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
SCORE_SHA = '03a17968abd29ece715bc40410166689bc832a23b2b614dcf3b9a4f19b8fb244'
TRAIN_PLAN_SHA = '9b44fe67f57833d067c1534381695ee09cd93ed98691cf123ba942d518fa7c83'
DEVICE = torch.device('cuda')
SAMPLE_IMAGES = 256
SEED = 42
TOP50 = 50
MAX_CANDIDATE_RANK = 60
MIN_ADMITTED_IMAGES = 5
IOU = .5


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


def prepare():
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
    pin(OUTPUT / 'plan.json', plan)
    return plan, datasets


@torch.inference_mode()
def probe(protocol, job, dataset):
    model, criterion = load_detector(None, Path(job['checkpoint']['path']), DEVICE)
    model.eval()
    base = criterion.base
    known = job['previous'] + job['current']
    if list(base.active_class_ids) != known:
        raise ValueError(f'Changed active class mapping: {protocol}')
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=TOP50)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(SEED))
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
        original_mask = base.previous_hard_negative_mask(outputs['pred_logits'][..., known], boxes, targets, matches, None)
        for i, (target, pair) in enumerate(zip(targets, matches, strict=True)):
            result = select_old_negatives(scores[i], boxes[i], target, pair, classes, len(job['previous']))
            rows.append(dict(image=int(target['image_id']), original_old_terms=int(original_mask[i].sum()), **result))
        if step % 32 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} training images', flush=True)
    if [r['image'] for r in rows] != job['image_ids']:
        raise ValueError(f'Changed sample order: {protocol}')
    return rows


def main():
    plan, datasets = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        marker = OUTPUT / f'{protocol}.json'
        if not marker.exists():
            print(f'Evaluating {protocol} balanced finetune training images', flush=True)
            rows = probe(protocol, job, datasets[protocol])
            counts = dict(images=len(rows), images_near=sum(r['near_current'] > 0 for r in rows),
                          images_admitted=sum(r['admitted_current'] > 0 for r in rows),
                          matches_admitted=sum(r['admitted_current'] for r in rows),
                          old_terms=sum(len(r['old_terms']) for r in rows),
                          original_old_terms=sum(r['original_old_terms'] for r in rows))
            write_json(marker, dict(plan_sha256=digest, rows=rows, counts=counts))
        report = read_json(marker)
        if report['plan_sha256'] != digest or report['counts']['images'] != SAMPLE_IMAGES:
            raise ValueError(f'Unexpected report: {protocol}')
        protocols[protocol] = report['counts']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                follow_up=all(r['images_admitted'] >= MIN_ADMITTED_IMAGES for r in protocols.values())))


if __name__ == '__main__':
    main()
