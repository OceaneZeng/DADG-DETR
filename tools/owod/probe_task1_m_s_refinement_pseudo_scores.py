"""Measure known/unknown score competition on released Task 1 training images."""

from hashlib import sha256
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader, Subset

from datasets.coco import CocoDetection, make_coco_transforms
from main import load_local_checkpoint
from tools.owod.probe_task1_discovery_gates import candidate_batches, frozen_miner
from tools.owod.probe_task1_m_s_refinement_top_known import OUTPUT as TOP_KNOWN
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, OUTPUT as TRAINING
from tools.owod.screening_data import screening_files, write_json
from util.misc import collate_fn


OUTPUT = TRAINING.with_name('task1-m-s-refinement-pseudo-scores')
TOP_KNOWN_SHA = 'f3043f4e8e0fb4fcb57462b2c3a9a5c609bcc3381f14b2f4488dd48dfd14b6c7'
SAMPLE_IMAGES = 512
SEED = 42


def prepare():
    result = verified(TOP_KNOWN / 'comparison.json', TOP_KNOWN_SHA)
    if result['status'] != 'complete' or result['passed']:
        raise ValueError('Expected the completed single-known-label comparison')
    prior = verified(TOP_KNOWN / 'plan.json', result['plan_sha256'])
    training = verified(TRAINING / 'plan.json', prior['training_plan_sha256'])
    jobs = {}
    for protocol, job in training['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        command = job['command']
        screen = Path(command[command.index('--owod-screening') + 1])
        if file_sha256(screen) != job['screen_sha256']:
            raise ValueError(f'Changed Task 1 screen: {protocol}')
        _, annotations = screening_files(path, screen, 0)
        entry = prior['jobs'][protocol]['arms']['refinement']
        jobs[protocol] = dict(checkpoint=entry['checkpoint'], checkpoint_sha256=entry['checkpoint_sha256'],
                              manifest_sha256=job['manifest_sha256'],
                              classes=manifest['stages'][0]['classes'],
                              training=str(annotations['increment_train']),
                              training_sha256=file_sha256(annotations['increment_train']))
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=TOP_KNOWN_SHA, jobs=jobs, seed=SEED, images=SAMPLE_IMAGES,
                batch_size=prior['batch_size'], paper_comparable=False)


def training_loader(checkpoint, path, root, job, plan, *, source_box_area=False, offset=0):
    annotation = verified(path, job['training_sha256'])
    if {row['category_id'] for row in annotation['annotations']} != set(job['classes']):
        raise ValueError('Expected only released Task 1 training annotations')
    args = load_local_checkpoint(checkpoint)['args']
    transforms = make_coco_transforms('train', lightweight=getattr(args, 'lightweight', False),
                                      no_augmentation=getattr(args, 'no_augmentation', False),
                                      no_random_crop=getattr(args, 'no_random_crop', False))
    dataset = CocoDetection(root, path, transforms=transforms, return_masks=False,
                            cache_mode=False, local_rank=0, local_size=1, source_box_area=source_box_area)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(plan['seed']))
    indices = indices[offset:offset + plan['images']].tolist()
    order = [dataset.coco.imgs[dataset.ids[i]]['owod_source_id'] for i in indices]
    loader = DataLoader(Subset(dataset, indices), batch_size=plan['batch_size'], num_workers=2, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(plan['seed']))
    return loader, sha256('\n'.join(order).encode()).hexdigest()


def summary(pairs):
    differences = torch.cat([delta for delta, _ in pairs])
    weights = torch.cat([weight for _, weight in pairs])
    conflict = differences <= 0
    result = dict(queries=len(differences), conflicts=int(conflict.sum()),
                  conflict_share=float(conflict.float().mean()) if len(conflict) else None,
                  weighted_conflict_share=float(weights[conflict].sum() / weights.sum()) if weights.sum() > 0 else None,
                  median_margin=float(differences.median()) if len(differences) else None)
    return result


@torch.inference_mode()
def measure(model, criterion, ema, loader, count):
    rows = dict(known=[], pseudo=[], unselected=[])
    observed = 0
    for step, (targets, outputs, _trace, mined) in enumerate(candidate_batches(model, criterion, ema, loader), 1):
        pairs = criterion.matcher(outputs, targets)
        predicted = outputs['pred_logits'].sigmoid()
        novelty = outputs['pred_novelty'].sigmoid()
        max_known = predicted[..., model.known_ids].amax(-1)
        selected = mined['pseudo']
        positive = torch.zeros_like(selected)
        for i, (queries, annotations) in enumerate(pairs):
            positive[i, queries] = True
            labels = targets[i]['labels'][annotations]
            margin = predicted[i, queries, labels] - novelty[i, queries]
            rows['known'].append((margin.cpu(), torch.ones_like(margin).cpu()))
        if (selected & positive).any():
            raise ValueError('A known matched query entered the pseudo set')
        margin = novelty - max_known
        rows['pseudo'].append((margin[selected].cpu(), mined['weights'][selected].cpu()))
        other = ~(positive | selected)
        rows['unselected'].append((margin[other].cpu(), torch.ones_like(margin[other]).cpu()))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != count:
        raise ValueError('Incomplete training-source replay')
    return {kind: summary(pairs) for kind, pairs in rows.items()}


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            checkpoint = Path(job['checkpoint'])
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed refined Task 1 checkpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(path, manifest),
                                            job, plan)
            model, criterion, ema = frozen_miner(checkpoint, manifest)
            print(f'Evaluating {protocol} Task 1 pseudo score conflicts', flush=True)
            report = measure(model, criterion, ema, loader, plan['images'])
            write_json(dest, dict(plan_sha256=digest, sample_order_sha256=order, groups=report))
            del model, criterion, ema
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen diagnostic plan: {dest}')
        protocols[protocol] = report['groups']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
