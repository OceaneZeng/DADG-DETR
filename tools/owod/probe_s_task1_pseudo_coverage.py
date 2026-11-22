"""Measure native EMA pseudo-label coverage by target size on locked S Task 1 slices."""

from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task1_novelty_budget import OUTPUT as SOURCE, PROTOCOL
from tools.owod.probe_task1_discovery_gates import candidate_batches, frozen_miner, masks
from tools.owod.probe_task1_paired_discovery import LOCKED, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, OUTPUT as PARENT, size_group
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-task1-pseudo-coverage')
SOURCE_PLAN = '2069ff3cfcfc5ca0666ec1b77127b78fdc4b9cac2380270c88c8bb75f79b5c9d'
SOURCE_RESULT = '9bc304e77a41982944b879b7cf52503f3c0eab25d7eaef2a9adbbd7dcc58abf8'
STAGES = ('raw', 'valid', 'eligible', 'student_salient', 'consistent', 'confident', 'selected')
REMOVALS = ('without_student_saliency', 'without_consistency', 'without_teacher_saliency')


@torch.inference_mode()
def audit(model, criterion, ema, dataset, loader):
    known = set(model.known_ids)
    rows, images = [], []
    for step, (targets, student, trace, mined) in enumerate(candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            image_id = int(target['image_id'])
            height, width = target['orig_size'].tolist()
            boxes = voc_coords(student['pred_boxes'][i], width, height)
            gates = masks(trace[i], mined['pseudo'][i])
            if any((gates[right] & ~gates[left]).any() for left, right in zip(STAGES, STAGES[1:])):
                raise ValueError('Native candidate stages are not nested')
            gates = {stage: mask.numpy() for stage, mask in gates.items()}
            for ann in dataset.coco.imgToAnns[image_id]:
                if ann.get('difficult'):
                    continue
                correct = overlaps(np.asarray(ann['voc_bbox']), boxes) > .5
                hit = {name: bool((correct & gate).any()) for name, gate in gates.items()}
                rows.append(dict(annotation_id=ann['id'], image_id=image_id, size=size_group(ann),
                                 released=ann['category_id'] in known, coverage=hit))
            images.append(dict(image_id=image_id, queries={gate: int(gates[gate].sum()) for gate in STAGES}))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len({r['image_id'] for r in images}) != len(images) or len(images) != len(dataset):
        raise ValueError('Changed image coverage')
    return dict(rows=rows, images=images)


def summarize(report):
    counts = defaultdict(Counter)
    by_image = defaultdict(Counter)
    image_picks = {r['image_id']: r['queries']['selected'] for r in report['images']}
    groups = ('under_16_sq', '16_to_32_sq', '32_sq_plus')
    for role in ('proxy', 'released'):
        for size in (*groups, 'small', 'all'):
            counts[f'{role}/{size}'].update(dict.fromkeys(('targets', *STAGES, *REMOVALS), 0))
    for row in report['rows']:
        role = 'released' if row['released'] else 'proxy'
        size = row['size']
        for group in (size, 'all') if size == '32_sq_plus' else (size, 'small', 'all'):
            counts[f'{role}/{group}'].update(targets=1, **row['coverage'])
        if role == 'proxy' and size != '32_sq_plus':
            budget = image_picks[row['image_id']]
            group = '0' if not budget else '1–2' if budget <= 2 else '3+'
            by_image[group].update(targets=1, raw=row['coverage']['raw'], selected=row['coverage']['selected'])
    small = counts['proxy/small']
    losses = {b: small[a] - small[b] for a, b in zip(STAGES, STAGES[1:])}
    return dict(groups=dict(counts), small_by_image_pseudo_count=dict(by_image), small_gate_losses=losses,
                largest_small_gate_loss=max(losses, key=losses.get), total_pseudo=sum(image_picks.values()))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE/'plan.json', SOURCE_PLAN)
    result = verified(SOURCE/'comparison.json', SOURCE_RESULT)
    if result['status'] != 'complete' or result['plan_sha256'] != SOURCE_PLAN:
        raise ValueError('Changed S checkpoint comparison')
    manifest_path = MANIFESTS[PROTOCOL]
    manifest = verified(manifest_path, LOCKED[PROTOCOL]['manifest_sha256'])
    root = resolve_image_root(manifest_path, manifest)
    checkpoints = prior['checkpoint_inputs']
    for name, spec in checkpoints.items():
        if file_sha256(Path(spec['path'])) != spec['sha256']:
            raise ValueError(f'Changed {name} checkpoint')
    paths = {fold: PARENT/spec['file'] for fold, spec in prior['slices'].items()}
    for fold, path in paths.items():
        verified(path, prior['slices'][fold]['sha256'])
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                parent_plan=SOURCE_PLAN, parent_comparison=SOURCE_RESULT, checkpoints=checkpoints,
                slices=prior['slices'], batch_size=2, strict_voc_iou=.5, paper_comparable=False)
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    summaries = {}
    for name, spec in checkpoints.items():
        model, criterion, ema = frozen_miner(Path(spec['path']), manifest)
        for fold, path in paths.items():
            label = f'{name}-{fold}'
            target = OUTPUT/f'{label}.json'
            if not target.exists():
                ds, loader = make_loader(root, path, plan['batch_size'], 2)
                print(f'Evaluating {label} with real student and checkpoint EMA', flush=True)
                report = audit(model, criterion, ema, ds, loader)
                write_json(target, dict(plan_sha256=digest, **report))
            report = read_json(target)
            if report['plan_sha256'] != digest:
                raise ValueError(f'Unpaired report {label}')
            summaries[label] = summarize(report)
        del ema, model, criterion
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=summaries))


if __name__ == '__main__':
    main()
