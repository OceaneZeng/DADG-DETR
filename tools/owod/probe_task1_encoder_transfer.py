"""Rank dense residual proposals on held-back Task 1 classes and fresh training-source images."""
from collections import Counter
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from models.graph_local.discovery_candidates import flip_samples
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_encoder_residual as residual
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, select_images, source_ids, subset, write_json


OUTPUT = ROOT / 'output/m-instance-task1-encoder-transfer'


def identities(names):
    return hashlib.sha256(('\n'.join(sorted(names)) + '\n').encode()).hexdigest()


def prepare():
    plan, manifest, _, large = paired.paired_plan(
        residual.OUTPUT, 'Fit the mean within-object difference of unit encoder features between accurate '
        '(IoU >= .5) and nearby (.1 <= IoU < .5) residual cells on 15 released classes. On fresh images rank '
        'the same eligible cells by projection or original mirror-quality score; allow one choice per image.',
        'Each seed must improve top-1 hit images by >=5 percentage points among reachable images on the '
        'five held-back released classes. Top-10 is descriptive. No Task 2 targets or threshold fitting.')
    classes = manifest['stages'][0]['classes']
    held = classes[3::4]
    fit_classes = [c for c in classes if c not in held]
    _, official = stage_files(MANIFEST, 0)
    full = read_json(official['increment_train'])
    blocked = set(manifest['evaluation']['image_source_ids']) | source_ids(read_json(large))
    train_hashes = set()
    for job in plan['jobs'].values():
        screen = Path(job['screen'])
        if file_sha256(screen) != job['screen_sha256']:
            raise ValueError('Changed parent screening file')
        _, files = screening_files(MANIFEST, screen, 0)
        train = files['increment_train']
        train_hashes.add(file_sha256(train))
        blocked.update(source_ids(read_json(Path(job['small']))))
    if len(train_hashes) != 1:
        raise ValueError('Expected identical parent training images')
    parent = read_json(train)
    fit_names = select_images(parent, 2048, fit_classes, 42, excluded=blocked)
    val_names = select_images(full, 1024, [], 42, excluded=blocked | source_ids(parent))
    if fit_names & val_names or len(held) != 5:
        raise ValueError('Invalid released-only split')
    plan.update(fit_classes=fit_classes, held_classes=held, train_sha256=train_hashes.pop(),
                official_train_sha256=file_sha256(official['increment_train']), selection_seed=42,
                fit_images=len(fit_names), fit_names_sha256=identities(fit_names),
                val_images=len(val_names), val_names_sha256=identities(val_names))
    pin(OUTPUT / 'plan.json', plan)
    for label, data in [('fit', subset(parent, fit_names, fit_classes)), ('val', subset(full, val_names, classes))]:
        path = OUTPUT / f'{label}-data.json'
        if path.exists() and read_json(path) != data:
            raise ValueError(f'Changed {label} annotations')
        if not path.exists():
            write_json(path, data)
    return plan, manifest


def candidates(model, ema, loader, exclude_classes):
    excluded = torch.tensor(exclude_classes, device='cuda', dtype=torch.long)
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to('cuda')
        out = model(samples)
        other = ema(flip_samples(samples))['geometry']
        g = out['geometry']
        for row, t in enumerate(targets):
            boxes, labels = t['boxes'].cuda(), t['labels'].cuda()
            known = torch.isin(labels, excluded)
            base, legal = residual.residual_scores(g, other, row, boxes[known], out['pred_boxes'][row])
            ids = legal.nonzero().flatten()
            yield (int(t['image_id'].item()), ids, base[ids], g['boxes'][row, ids],
                   F.normalize(g['memory'][row, ids].float(), dim=-1), boxes[~known])
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Processed {step}/{len(loader)} batches', flush=True)


@torch.inference_mode()
def fit(model, ema, loader):
    total = None
    objects = positives = near_misses = images = 0
    for _, ids, _, boxes, memory, truth in candidates(model, ema, loader, []):
        images += 1
        if not len(truth) or not len(ids):
            continue
        quality, owner = paired.overlap(boxes, truth).max(1)
        for obj in owner[quality >= .5].unique():
            pos = (owner == obj) & (quality >= .5)
            near = (owner == obj) & (quality >= .1) & (quality < .5)
            if near.any():
                contrast = memory[pos].mean(0) - memory[near].mean(0)
                total = contrast.double() if total is None else total + contrast
                positives += int(pos.sum())
                near_misses += int(near.sum())
                objects += 1
    if total is None or not torch.isfinite(total).all() or total.norm() == 0:
        raise ValueError('No finite paired feature contrast')
    return dict(direction=F.normalize(total, dim=0).tolist(), objects=objects, positives=positives,
                near_misses=near_misses, images=images)


@torch.inference_mode()
def evaluate(model, ema, loader, fit_row, fit_classes):
    direction = torch.tensor(fit_row['direction'], dtype=torch.float32, device='cuda')
    images = {}
    totals = Counter()
    for im, ids, base, boxes, memory, truth in candidates(model, ema, loader, fit_classes):
        # Commit to both lists before reading held-class overlaps.
        orders = {key: torch.argsort(scores, descending=True, stable=True)[:10]
                  for key, scores in [('original', base), ('contrast', memory @ direction)]}
        edges = paired.overlap(boxes, truth) >= .5
        row = dict(cells=len(ids), gt=len(truth), reachable=int(edges.any()))
        totals.update(row)
        for key, order in orders.items():
            selected = edges[order]
            row[key] = dict(ids=ids[order].tolist(), gt_ids=[e.nonzero().flatten().tolist() for e in selected])
            totals[key + '_top1'] += int(selected[:1].any())
            totals[key + '_top10'] += int(selected.any())
        images[im] = row
    if set(images) != set(loader.dataset.ids):
        raise ValueError('Incomplete held-class slice')
    return dict(images=images, counts=totals)


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    fits, rows, passed = {}, {}, {}
    for seed, job in plan['jobs'].items():
        model, _, ema = gates.frozen_miner(Path(job['parent']), manifest)
        fit_path = OUTPUT / f'seed{seed}-fit.json'
        if not fit_path.exists():
            print(f'Seed {seed}: fitting released-class residual feature contrast', flush=True)
            _, loader = make_loader(root, OUTPUT / 'fit-data.json', plan['batch_size'], 2)
            write_json(fit_path, dict(plan_sha256=digest, **fit(model, ema, loader)))
        fitted = read_json(fit_path)
        if fitted['plan_sha256'] != digest or fitted['images'] != plan['fit_images']:
            raise ValueError('Changed released-class fit')
        fits[seed] = file_sha256(fit_path)
        path = OUTPUT / f'seed{seed}-val.json'
        if not path.exists():
            print(f'Seed {seed}: evaluating held classes on fresh images', flush=True)
            _, loader = make_loader(root, OUTPUT / 'val-data.json', plan['batch_size'], 2)
            result = evaluate(model, ema, loader, fitted, plan['fit_classes'])
            write_json(path, dict(plan_sha256=digest, fit_sha256=fits[seed], **result))
        saved = read_json(path)
        if (saved['plan_sha256'] != digest or saved['fit_sha256'] != fits[seed]
                or len(saved['images']) != plan['val_images']):
            raise ValueError('Changed held-class evaluation')
        c = rows[seed] = saved['counts']
        passed[seed] = c['reachable'] > 0 and 100 * (c['contrast_top1'] - c['original_top1']) >= 5 * c['reachable']
        del model, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, fit_sha256=fits,
                                               rows=rows, passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
