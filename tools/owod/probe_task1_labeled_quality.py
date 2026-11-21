"""Fit frozen proposal quality on released Task 1 boxes, then rank held-out anonymous candidates."""
from collections import Counter
import hashlib
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader, Subset

from models.graph_local.discovery_candidates import select_distinct_boxes
from models.graph_local.instance_transfer import paired_iou
from tools.owod import probe_task1_empty_allocation as allocation
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import probe_task2_slot_features as features
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, select_images, source_ids, write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task1-labeled-quality'
CUES = ('geometry', 'objectness', 'view_iou', 'consistency', 'student_saliency', 'teacher_saliency')
FIT_IMAGES = 4096
RIDGE = .01
MIN_IOU = .1


def cue_matrix(output, mined, trace, i):
    partners = mined['matches'][i]
    view_iou = paired_iou(output['pred_boxes'][i], mined['teacher_boxes'][i, partners.clamp_min(0)])
    return torch.stack([output['geometry_quality'][i], output['pred_objectness'][i].sigmoid(),
                        view_iou, mined['consistency'][i], mined['saliency'][i], trace['matched_saliency']], dim=1)


def neighborhood(boxes, truth, valid):
    if not truth.numel():
        return None
    overlap = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0]
    quality, anchor = overlap.max(1)
    keep = valid & (quality >= MIN_IOU)
    if not keep.any():
        return None
    objects = anchor[keep]
    weights = objects.bincount(minlength=len(truth))[objects].double().reciprocal()
    return keep, quality[keep], weights


def solve(gram):
    # The last two columns are the intercept and target IoU. Normalize features
    # using training-only moments; apply the ridge penalty only to the slopes.
    mass = gram[-2, -2]
    moments = gram / mass
    mean, target_mean = moments[:-2, -2], moments[-1, -2]
    covariance = moments[:-2, :-2] - torch.outer(mean, mean)
    scale = covariance.diagonal().clamp_min(1e-12).sqrt()
    system = covariance / torch.outer(scale, scale) + RIDGE * torch.eye(len(CUES), dtype=gram.dtype)
    slopes = torch.linalg.solve(system, (moments[:-2, -1] - mean * target_mean) / scale) / scale
    intercept = target_mean - slopes @ mean
    residual = torch.cat([slopes, intercept.reshape(1), gram.new_tensor([-1])])
    return dict(slopes=slopes.tolist(), intercept=float(intercept), represented_gt=round(float(mass)),
                train_rmse=float((residual @ moments @ residual).clamp_min(0).sqrt()),
                mean_only_rmse=float((moments[-1, -1] - target_mean.square()).clamp_min(0).sqrt()))


@torch.inference_mode()
def fit(model, criterion, ema, loader):
    gram = torch.zeros(len(CUES) + 2, len(CUES) + 2, dtype=torch.float64)
    observations = 0
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            valid = traces[i]['valid'] & (mined['matches'][i] >= 0)
            anchors = neighborhood(output['pred_boxes'][i], target['boxes'], valid)
            if anchors is None:
                continue
            keep, y, weights = anchors
            x = cue_matrix(output, mined, traces[i], i)[keep]
            data = torch.cat([x, torch.ones_like(y[:, None]), y[:, None]], dim=1).cpu().double()
            gram += data.T @ (weights.cpu()[:, None] * data)
            observations += len(y)
        if step == 64 or step % 256 == 0 or step == len(loader):
            provisional = solve(gram)
            print(f'Calibration {step}/{len(loader)} batches, {observations} boxes, '
                  f'{provisional["represented_gt"]} anchored GT, fitted RMSE {provisional["train_rmse"]:.4f}', flush=True)
    return dict(**solve(gram), observations=observations, images=len(loader.dataset), weighted_gram=gram.tolist())


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, fit_row, current, previous):
    slopes = torch.tensor(fit_row['slopes'], device='cuda', dtype=torch.float32)
    classes = torch.tensor(current, device='cuda')
    arms = ('original', 'soft', 'equal_cues', 'fitted')
    counts = {arm: Counter() for arm in arms}
    errors = {name: Counter() for name in ('known_task1', 'released_task2')}
    images = {}
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            original = mined['pseudo'][i]
            quota = int(original.sum())
            boxes = output['pred_boxes'][i]
            cues = cue_matrix(output, mined, traces[i], i)
            estimate = (cues @ slopes + fit_row['intercept']).clamp(0, 1)
            average = cues.clamp_min(0).prod(1).pow(1 / len(CUES))
            base_score, pool = soft.continuous_evidence(output, mined, traces[i], i)
            semantic = mined['semantic_weights'][i]
            masks = {'original': original}
            fallback = {}
            for arm, score in [('soft', base_score), ('equal_cues', average * semantic), ('fitted', estimate * semantic)]:
                ids = select_distinct_boxes(boxes, pool, score, quota)
                fallback[arm] = len(ids) != quota
                masks[arm] = original if fallback[arm] else torch.zeros_like(original).scatter_(0, ids, True)
            is_new = torch.isin(target['labels'], classes)
            truth = target['boxes'][is_new]
            edges = (box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0] >= .5).cpu()
            image = int(target['image_id'].item())
            row = dict(gt=len(truth), quota=quota, hits={}, ids={})
            for arm, mask in masks.items():
                mask = mask.cpu()
                ids = row['ids'][arm] = mask.nonzero().flatten().tolist()
                hits = row['hits'][arm] = gates.measure_boxes(edges, mask, quota)['one_to_one']
                if len(ids) != quota or (arm in ('original', 'soft') and
                       (ids != previous[str(image)]['query_ids'][arm] or hits != previous[str(image)]['hits'][arm])):
                    raise ValueError(f'Unpaired {arm} selection at {image}')
                counts[arm].update(hits=hits, gt=len(truth), proposals=quota, fallback=int(fallback.get(arm, False)))
            images[image] = row
            for group, gt in [('known_task1', target['boxes'][~is_new]), ('released_task2', truth)]:
                anchor = neighborhood(boxes, gt, traces[i]['valid'] & (mined['matches'][i] >= 0))
                if anchor is not None:
                    keep, y, weights = anchor
                    errors[group].update(weight=float(weights.sum()), fitted_sse=float((weights * (estimate[keep]-y).square()).sum()),
                                         geometry_sse=float((weights * (cues[keep, 0]-y).square()).sum()))
        if step % 64 == 0 or step == len(loader):
            print(f'Evaluation {step}/{len(loader)} batches, {len(images)} images', flush=True)
    if set(images) != set(loader.dataset.ids):
        raise ValueError('Incomplete held-out image set')
    return dict(counts=counts, quality_errors=errors, images=images)


def main():
    source = read_json(allocation.OUTPUT / 'plan.json')
    last = read_json(allocation.OUTPUT / 'comparison.json')
    large = holdout.OUTPUT / 'heldout.json'
    manifest = read_json(MANIFEST)
    if (last['status'] != 'complete' or last['plan_sha256'] != file_sha256(allocation.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(MANIFEST) or source['holdout_sha256'] != file_sha256(large)
            or source['torch_version'] != str(torch.__version__) or source['gpu'] != torch.cuda.get_device_name()):
        raise ValueError('Original inputs and completed allocation comparison are required')
    known = manifest['stages'][0]['classes']
    jobs = source['jobs']
    screens = read_json(features.OUTPUT / 'plan.json')['entries']
    train_hashes = set()
    for seed, job in jobs.items():
        job['screen'] = screens[seed]['screen']
        screen = Path(job['screen'])
        for key in ('screen', 'parent', 'small'):
            if file_sha256(Path(job[key])) != job[f'{key}_sha256']:
                raise ValueError(f'Changed seed {seed} {key}: {job[key]}')
        _, files = screening_files(MANIFEST, screen, 0)
        train_path = files['increment_train']
        train_hashes.add(file_sha256(train_path))
    if len(train_hashes) != 1:
        raise ValueError('Paired parents require identical released training examples')
    train = read_json(train_path)
    names = select_images(train, FIT_IMAGES, known, 42)
    excluded = set(manifest['evaluation']['image_source_ids']) | source_ids(read_json(large))
    for job in jobs.values():
        excluded.update(source_ids(read_json(Path(job['small']))))
    if names & excluded or {ann['category_id'] for ann in train['annotations']} != set(known):
        raise ValueError('Evaluation overlap or unexpected calibration labels')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=last['plan_sha256'], source_result_sha256=file_sha256(allocation.OUTPUT / 'comparison.json'),
                jobs=jobs, per_image_sha256=source['per_image_sha256'], manifest_sha256=source['manifest_sha256'],
                holdout_sha256=source['holdout_sha256'], train_sha256=train_hashes.pop(), fit_images=FIT_IMAGES,
                selected_names_sha256=hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest(),
                selection_seed=42, min_iou=MIN_IOU, ridge=RIDGE, cues=list(CUES), batch_size=2,
                torch_version=source['torch_version'], gpu=source['gpu'], paper_comparable=False,
                criterion='Both seeds: fitted >= 5 pp over original on holdout1024, above soft and untrained equal-cue controls there, and >= original on reference128.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    data, template = make_loader(root, train_path, 2, 2)
    indices = [i for i, k in enumerate(data.ids) if data.coco.imgs[k]['owod_source_id'] in names]
    if len(indices) != FIT_IMAGES:
        raise ValueError('Incomplete calibration selection')
    calibration = DataLoader(Subset(data, indices), batch_size=2, shuffle=False, num_workers=2, collate_fn=template.collate_fn)
    rows, passed = {}, {}
    for seed, job in jobs.items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        fit_path = OUTPUT / f'seed{seed}-fit.json'
        if not fit_path.exists():
            print(f'Seed {seed}: fitting Task 1 quality on {FIT_IMAGES} images', flush=True)
            write_json(fit_path, dict(plan_sha256=digest, **fit(model, criterion, ema, calibration)))
        fitted = read_json(fit_path)
        if fitted['plan_sha256'] != digest or fitted['images'] != FIT_IMAGES:
            raise ValueError('Unexpected saved fit')
        rows[seed] = {}
        for label, path in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            result_path = OUTPUT / f'seed{seed}-{label}.json'
            if not result_path.exists():
                old_path = soft.OUTPUT / f'seed{seed}-{label}.json'
                if file_sha256(old_path) != source['per_image_sha256'][seed][label]:
                    raise ValueError('Original per-image reference changed')
                _, loader = make_loader(root, path, 2, 2)
                print(f'Seed {seed}, {label}: comparing actual ranked boxes', flush=True)
                measured = evaluate(model, criterion, ema, loader, fitted, manifest['stages'][1]['classes'], read_json(old_path)['images'])
                write_json(result_path, dict(plan_sha256=digest, fit_sha256=file_sha256(fit_path), **measured))
            row = read_json(result_path)
            if row['plan_sha256'] != digest or row['fit_sha256'] != file_sha256(fit_path):
                raise ValueError('Result and fitted coefficients differ')
            rows[seed][label] = row['counts']
        small, big = rows[seed]['reference128'], rows[seed]['holdout1024']
        passed[seed] = (100 * (big['fitted']['hits'] - big['original']['hits']) >= 5 * big['original']['gt']
                        and big['fitted']['hits'] > max(big[a]['hits'] for a in ('soft', 'equal_cues'))
                        and small['fitted']['hits'] >= small['original']['hits'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
