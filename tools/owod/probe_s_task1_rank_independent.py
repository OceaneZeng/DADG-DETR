"""Replay three S Task 1 ranking arms on a separate released-only holdout."""

import subprocess
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_detached_rank_ap import DETACHED_SHA, OUTPUT as TRACE, PROTOCOL
from tools.owod.probe_s_task1_detached_rank_ap import prepare as traced_arms
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import (
    DETACHED_OUTPUT, MATCHED_COUNT_OUTPUT, MATCHED_COUNT_SHA, QUALITY_OUTPUT, QUALITY_SHA,
)
from tools.owod.run_task1_m_s_released_small_extent import annotation_digest, size_group
from tools.owod.screening_data import source_ids, subset, write_json


OUTPUT = TRACE.with_name('task1-s-rank-independent')
TRACE_SHA = '1caa2b866a2e57ba9e459352dfff8ac58430647733c7b82c4bd7ab7787a7c6ab'
NO_RANK_CHECKPOINT = '8e8b88b09f12587fe89c492841be0040016fbeecc74b865af08ea804ca56c931'
IMAGES = 2048
MIN_CLASS_IMAGES = 64
SEED = 53
PRIOR_COHORTS = tuple(Path('output') / name / filename for name, filenames in (
    ('task1-m-s-consistency-validation', ('s-owodb-first-annotations.json', 's-owodb-second-annotations.json')),
    ('task1-m-s-small-box-extent', ('s-owodb-released-annotations.json',)),
) for filename in filenames)


def cohort(source, known, excluded):
    eligible = {row['id']: row['owod_source_id'] for row in source['images']
                if row['owod_source_id'] not in excluded}
    by_class = defaultdict(set)
    for ann in source['annotations']:
        name = eligible.get(ann['image_id'])
        if name and not ann.get('difficult', 0) and min(ann['bbox'][2:]) > 0:
            by_class[ann['category_id']].add(name)
    order = sorted(eligible.values(), key=lambda name: sha256(f'{SEED}:{name}'.encode()).digest())
    chosen = set()
    for label in known:
        names = [name for name in order if name in by_class[label]]
        if len(names) < MIN_CLASS_IMAGES:
            raise ValueError(f'Insufficient independent author training images for class {label}')
        chosen.update(names[:MIN_CLASS_IMAGES])
    for name in order:
        if len(chosen) >= IMAGES:
            break
        chosen.add(name)
    if len(chosen) != IMAGES:
        raise ValueError('Cannot fill the pinned independent holdout')
    return subset(source, chosen, set(known)), {str(label): len(chosen & by_class[label]) for label in known}


def prepare():
    trace = verified(TRACE / 'comparison.json', TRACE_SHA)
    if trace['status'] != 'complete':
        raise ValueError('Expected the completed emitted-AP attribution')
    locked = traced_arms()
    arms = locked['arms']
    checkpoint = MATCHED_COUNT_OUTPUT / PROTOCOL / 'increment/checkpoint.pth'
    if file_sha256(checkpoint) != NO_RANK_CHECKPOINT:
        raise ValueError('Changed no-rank S Task 1 checkpoint')
    arms['no_rank'] = dict(checkpoint=str(checkpoint), checkpoint_sha256=NO_RANK_CHECKPOINT,
                           comparison_sha256=MATCHED_COUNT_SHA)
    configs = {}
    for name, folder, comparison_sha in (
        ('no_rank', MATCHED_COUNT_OUTPUT, MATCHED_COUNT_SHA),
        ('quality_rank', QUALITY_OUTPUT, QUALITY_SHA),
        ('detached_rank', DETACHED_OUTPUT, DETACHED_SHA),
    ):
        result = verified(folder / 'comparison.json', comparison_sha)
        config = verified(folder / 'plan.json', result['plan_sha256'])
        job = config['jobs'][PROTOCOL]
        command = job['command'].copy()
        command[command.index('--output_dir') + 1] = ''
        configs[name] = (command, job['backbone_sha256'], job['screen_sha256'])
    command, backbone, screen_sha = configs['no_rank']
    rank = '--screen-small-box-quality-rank'
    detach = '--screen-detach-rank-features'
    if configs['quality_rank'] != (command + [rank], backbone, screen_sha) or configs['detached_rank'] != (
            command + [rank, detach], backbone, screen_sha):
        raise ValueError('The three diagnostic arms must have the same training inputs and budget')
    screen_path = Path(command[command.index('--owod-screening') + 1])
    screen = verified(screen_path, screen_sha)
    record = screen['stages'][0]
    training = verified(screen_path.parent / record['files']['increment_train'], record['sha256']['increment_train'])
    path = MANIFESTS[PROTOCOL]
    verified(path, locked['manifest_sha256'])
    manifest, files = stage_files(path, 0)
    source = read_json(files['increment_train'])
    prior = {entry['path']: entry['sha256'] for entry in locked['folds'].values()}
    prior.update({str(path): file_sha256(path) for path in PRIOR_COHORTS})
    excluded = source_ids(training) | set(screen['heldout_source_ids']) | set(manifest['splits']['test'])
    for filename, digest in prior.items():
        excluded.update(source_ids(verified(Path(filename), digest)))
    known = manifest['stages'][0]['classes']
    annotations, counts = cohort(source, known, excluded)
    annotation_path = OUTPUT / 'released-holdout.json'
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, trace_sha256=TRACE_SHA, manifest_sha256=locked['manifest_sha256'],
                train_source_sha256=file_sha256(files['increment_train']), trained_screen_sha256=screen_sha,
                prior_cohorts=prior, image_count=IMAGES, min_class_images=MIN_CLASS_IMAGES,
                seed=SEED, images_by_class=counts, holdout_sha256=annotation_digest(annotations),
                batch_size=locked['batch_size'], limit=locked['limit'], arms=arms)
    pin(OUTPUT / 'plan.json', plan)
    if not annotation_path.is_file():
        write_json(annotation_path, annotations)
    if file_sha256(annotation_path) != plan['holdout_sha256']:
        raise ValueError('Changed pinned released holdout')
    return plan, manifest, annotation_path


def evaluate(model, manifest, path, plan, *, trace_query_origin=False, extra_postprocessors=None):
    known = manifest['stages'][0]['classes']
    threshold = manifest['evaluation']['iou_threshold']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, path, plan['batch_size'], 2)
    spec = dict(manifest['evaluation'], known_only_screening=True, diagnostic_current_precision=True,
                image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    processors = dict(emitted=CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit']))
    if extra_postprocessors:
        processors.update(extra_postprocessors)
    small = Counter(released=0, covered=0)
    origins = {}

    def observe(outputs, targets, batch_images, _score_tables):
        for index, image in batch_images.items():
            annotations = [ann for ann in dataset.coco.imgToAnns[image]
                           if not ann.get('difficult') and size_group(ann) != '32_sq_plus']
            if not annotations:
                continue
            boxes = outputs['pred_boxes'][index]
            active = outputs['geometry']['active'][index]
            learned = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
            learned[:len(active)] = ~active
            height, width = targets[index]['orig_size'].tolist()
            learned_coords = voc_coords(boxes[learned], width, height)
            if trace_query_origin:
                proposal_coords = voc_coords(boxes[~learned], width, height)
            for annotation in annotations:
                truth = np.asarray(annotation['voc_bbox'])
                hit = (overlaps(truth, learned_coords) > threshold).any().item()
                small['released'] += 1
                small['covered'] += hit
                if trace_query_origin:
                    origins[annotation['id']] = dict(
                        image_id=image, size=size_group(annotation), learned=hit,
                        proposal=(overlaps(truth, proposal_coords) > threshold).any().item())

    views = {name: (known, 'focal', plan['limit']) for name in processors}
    report = evaluate_views(model, loader, dataset.coco, views,
                            spec, [], known, torch.device('cuda'), include_current_stats=True,
                            postprocessors=processors, print_freq=128, rank_observer=observe)
    report['released_small_learned_query_coverage'] = small
    if trace_query_origin:
        if len(origins) != small['released']:
            raise ValueError('Expected a unique ID for every released small annotation')
        report['released_small_query_origin'] = origins
    return report


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the independent replay')
    plan, manifest, annotation = prepare()
    plan_sha = file_sha256(OUTPUT / 'plan.json')
    rows = {}
    for name, arm in plan['arms'].items():
        destination = OUTPUT / f'{name}.json'
        if not destination.is_file():
            print(f'Evaluating S Task 1 independent holdout: {name}', flush=True)
            model, _ = load_detector(None, Path(arm['checkpoint']), torch.device('cuda'))
            report = evaluate(model, manifest, annotation, plan)
            write_json(destination, dict(plan_sha256=plan_sha, report=report))
            del model
        saved = read_json(destination)
        if saved['plan_sha256'] != plan_sha:
            raise ValueError(f'Changed pinned replay report for {name}')
        report = saved['report']
        rows[name] = dict(known_ap50=report['emitted']['metrics']['known_ap50'],
                          small=report['released_small_learned_query_coverage'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_sha, arms=rows))


if __name__ == '__main__':
    main()
