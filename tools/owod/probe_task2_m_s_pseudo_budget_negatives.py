"""Measure negative supervision of distinct proposals omitted by the Task 2 pseudo budget."""

from hashlib import sha256
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from datasets.coco import CocoDetection, make_coco_transforms
from engine import teacher_prediction
from main import load_local_checkpoint
from models.graph_local.discovery import make_ema
from models.graph_local.discovery_candidates import flip_samples, mine_candidates, select_distinct_boxes
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_formal_refill_capacity import SOURCES
from tools.owod.probe_task2_m_s_refill_quality_transfer import DEVICE, OUTPUT as QUALITY
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import collate_fn


OUTPUT = QUALITY.with_name('task2-m-s-pseudo-budget-negatives')
QUALITY_SHA = '05dda8092eaa99f902199308fbe5c4615bd7bb118a107caebdd5429c3079a22b'
SAMPLE_IMAGES = 512
SEED = 42


def prepare():
    comparison = verified(QUALITY / 'comparison.json', QUALITY_SHA)
    if comparison['status'] != 'complete' or comparison['passed']:
        raise ValueError('Expected the completed released localization-quality comparison')
    quality_plan = verified(QUALITY / 'plan.json', comparison['plan_sha256'])
    jobs = {}
    for duration, (directory, result_hash, endpoint_hash) in SOURCES.items():
        result = verified(directory / 'comparison.json', result_hash)
        plan = verified(directory / 'plan.json', result['plan_sha256'])
        endpoint_file = directory / 'endpoint-sha256.txt'
        if file_sha256(endpoint_file) != endpoint_hash:
            raise ValueError(f'Changed endpoint hashes: {duration}')
        hashes = {name: digest for digest, name in
                  (line.split() for line in endpoint_file.read_text(encoding='utf-8').splitlines())}
        jobs[duration] = {}
        for protocol, job in plan['jobs'].items():
            increment = next(run for run in job['benchmark']['runs']
                             if run['stage'] == 1 and run['phase'] == 'increment')
            checkpoint = Path(increment['output_dir']) / 'checkpoint.pth'
            manifest = quality_plan['jobs'][duration][protocol]['manifest']
            jobs[duration][protocol] = dict(
                checkpoint=str(checkpoint), checkpoint_sha256=hashes[checkpoint.resolve().relative_to(directory.resolve()).as_posix()],
                manifest=manifest, manifest_sha256=job['benchmark']['manifest_sha256'], epochs=increment['epochs'],
                fit=quality_plan['data'][protocol]['fit'], known=job['previous'] + job['current'],
                current=job['current'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=QUALITY_SHA, jobs=jobs, batch_size=quality_plan['batch_size'],
                sample_images=SAMPLE_IMAGES, seed=SEED, paper_comparable=False,
                minimum_affected_image_fraction=.05, minimum_gradient_mass_share=.01,
                decision='Advance budget-specific negative suppression only if both durations and protocols '
                         'show at least 5% affected images and 1% objectness negative gradient mass at logits.')


def load_inputs(job, plan):
    checkpoint = Path(job['checkpoint'])
    if file_sha256(checkpoint) != job['checkpoint_sha256']:
        raise ValueError(f'Changed increment checkpoint: {checkpoint}')
    marker = read_json(checkpoint.with_name('training_complete.json'))
    if marker['last_epoch'] != job['epochs'] - 1:
        raise ValueError(f'Changed completed increment epoch: {checkpoint}')
    manifest_path = Path(job['manifest'])
    manifest = verified(manifest_path, job['manifest_sha256'])
    root = resolve_image_root(manifest_path, manifest)
    annotation = Path(job['fit']['path'])
    released = verified(annotation, job['fit']['sha256'])
    if set(p['category_id'] for p in released['annotations']) != set(job['current']):
        raise ValueError(f'Expected only released Current annotations: {annotation}')
    if set(manifest['evaluation']['image_source_ids']) & {im['owod_source_id'] for im in released['images']}:
        raise ValueError('An official test source entered the training diagnostic')

    model, criterion = load_detector(None, checkpoint, DEVICE)
    model.eval()
    if list(model.known_ids) != job['known'] or job['epochs'] <= criterion.warmup:
        raise ValueError(f'Changed active classes or warmup at {checkpoint}')
    payload = load_local_checkpoint(checkpoint)
    ema = make_ema(model)
    ema.load_state_dict(payload['discovery_ema'], strict=True)
    args = payload['args']
    transforms = make_coco_transforms('train', lightweight=getattr(args, 'lightweight', False),
                                      no_augmentation=getattr(args, 'no_augmentation', False),
                                      no_random_crop=getattr(args, 'no_random_crop', False))
    dataset = CocoDetection(root, annotation, transforms=transforms, return_masks=False,
                            cache_mode=False, local_rank=0, local_size=1)
    if len(dataset) != job['fit']['images'] or plan['sample_images'] > len(dataset):
        raise ValueError('Changed registered training image count')
    choices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(plan['seed']))
    choices = choices[:plan['sample_images']].tolist()
    order = [dataset.coco.imgs[dataset.ids[i]]['owod_source_id'] for i in choices]
    identity = sha256('\n'.join(order).encode()).hexdigest()
    loader = DataLoader(Subset(dataset, choices), batch_size=plan['batch_size'], num_workers=2,
                        collate_fn=collate_fn, generator=torch.Generator().manual_seed(plan['seed']))
    return model, criterion, ema, loader, identity


def negative_gradient(logits, criterion):
    probability = logits.sigmoid()
    if criterion.objectness_loss != 'focal':
        return probability
    return ((1. - criterion.base.focal_alpha) * probability.square()
            * (probability + 2. * (1. - probability) * F.softplus(logits)))


@torch.inference_mode()
def audit(model, criterion, ema, loader):
    rows = {}
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to(DEVICE)
        targets = move_targets(targets, DEVICE)
        student = model(samples)
        teacher = teacher_prediction(ema, flip_samples(samples), 'discovery')
        matched = criterion.matcher(student, targets)
        gates = []
        mined = mine_candidates(student, targets, matched, model.known_ids, teacher, criterion.top_k,
                                criterion.saliency_threshold, criterion.consistency_threshold, gate_trace=gates)
        boxes = student['pred_boxes']
        decoded = box_cxcywh_to_xyxy(boxes)
        objectness = negative_gradient(student['pred_objectness'], criterion)
        route = torch.stack([1. - value.softmax(-1)[..., 0] for value in student['discovery_routes']]).mean(0)
        for i, target in enumerate(targets):
            selected = mined['pseudo'][i]
            negatives = ~selected.clone()
            negatives[matched[i][0]] = False
            confident = gates[i]['confident']
            held_back = confident & negatives
            duplicate = torch.zeros_like(negatives)
            if selected.any():
                iou, _ = box_iou(decoded[i], decoded[i, selected])
                duplicate = iou.amax(-1) > .70
            at_budget = int(selected.sum()) == criterion.top_k and criterion.top_k > 0
            distinct = held_back & ~duplicate if at_budget else torch.zeros_like(negatives)
            if not at_budget and (held_back & ~duplicate).any():
                raise ValueError('Unselected distinct candidate below the budget')
            score = mined['saliency'][i] * mined['consistency'][i] * mined['semantic_weights'][i]
            if 'geometry_quality' in student:
                score = score * student['geometry_quality'][i]
            unique = select_distinct_boxes(boxes[i], distinct, score, len(distinct))
            rows[int(target['image_id'])] = dict(
                budget_full=at_budget, selected=int(selected.sum()), confident=int(confident.sum()),
                confident_duplicates=int((held_back & duplicate).sum()), distinct=int(distinct.sum()),
                distinct_after_nms=len(unique), negatives=int(negatives.sum()),
                objectness_negative_mass=float(objectness[i, negatives].sum()),
                objectness_omitted_mass=float(objectness[i, distinct].sum()),
                router_negative_mass=float(route[i, negatives].sum()),
                router_omitted_mass=float(route[i, distinct].sum()))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    return rows


def summarize(rows, plan):
    if len(rows) != plan['sample_images']:
        raise ValueError('Changed diagnostic image coverage')
    values = list(rows.values())
    fraction = sum(row['distinct'] > 0 for row in values) / len(values)
    mass = {key: sum(row[key] for row in values) for key in values[0]}
    objectness = mass['objectness_omitted_mass'] / max(1e-12, mass['objectness_negative_mass'])
    router = mass['router_omitted_mass'] / max(1e-12, mass['router_negative_mass'])
    return dict(images=len(values), affected_image_fraction=fraction, objectness_gradient_share=objectness,
                router_gradient_share=router, totals=mass,
                passed=fraction >= plan['minimum_affected_image_fraction']
                and objectness >= plan['minimum_gradient_mass_share'])


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen training-source audit')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    outputs = {}
    for duration, jobs in plan['jobs'].items():
        outputs[duration] = {}
        for protocol, job in jobs.items():
            path = OUTPUT / f'{duration}-{protocol}.json'
            if not path.exists():
                model, criterion, ema, loader, identity = load_inputs(job, plan)
                print(f'Evaluating {duration}/{protocol} released training inputs with checkpoint EMA', flush=True)
                images = audit(model, criterion, ema, loader)
                write_json(path, dict(plan_sha256=digest, sample_order_sha256=identity,
                                      summary=summarize(images, plan), images=images))
                del model, criterion, ema, loader
            result = read_json(path)
            if result['plan_sha256'] != digest:
                raise ValueError(f'Changed negative-supervision plan: {path}')
            outputs[duration][protocol] = result['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, endpoints=outputs,
                                               passed=all(row['passed'] for d in outputs.values() for row in d.values())))


if __name__ == '__main__':
    main()
