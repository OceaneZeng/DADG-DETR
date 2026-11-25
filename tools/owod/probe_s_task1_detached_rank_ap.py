"""Attribute the S Task 1 detached-rank AP change on both locked holdouts."""

import subprocess
from collections import Counter
from pathlib import Path

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown, DeduplicateKnown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import (
    DETACHED_OUTPUT, QUALITY_CHECKPOINTS, QUALITY_OUTPUT, QUALITY_SHA,
)
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


PROTOCOL = 's-owodb'
OUTPUT = DETACHED_OUTPUT.with_name('task1-s-detached-rank-ap-trace')
DETACHED_SHA = 'b7b4f5f78bc7a676880e79cee941550259689f4c03df175abe946642e9ce9aab'
DETACHED_CHECKPOINT = '4dd72d434ce03e97327acc52067515c35ec1e9e0b2f9de1482d90665ef84b838'


def prepare():
    arms = {}
    shared = None
    for name, folder, comparison_sha, checkpoint_sha in (
        ('quality_rank', QUALITY_OUTPUT, QUALITY_SHA, QUALITY_CHECKPOINTS[PROTOCOL]),
        ('detached_rank', DETACHED_OUTPUT, DETACHED_SHA, DETACHED_CHECKPOINT),
    ):
        comparison = verified(folder / 'comparison.json', comparison_sha)
        plan = verified(folder / 'plan.json', comparison['plan_sha256'])
        job = plan['jobs'][PROTOCOL]
        settings = dict(manifest_sha256=job['manifest_sha256'], folds=job['folds'],
                        batch_size=plan['batch_size'], limit=plan['limit'])
        if comparison['status'] != 'complete' or (shared is not None and shared != settings):
            raise ValueError(f'Missing complete, matched Task 1 results for {name}')
        shared = settings
        checkpoint = folder / PROTOCOL / 'increment/checkpoint.pth'
        if file_sha256(checkpoint) != checkpoint_sha:
            raise ValueError(f'Changed S Task 1 checkpoint for {name}')
        arms[name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                          comparison_sha256=comparison_sha,
                          expected_ap={fold: row['metrics']['known_ap50']
                                       for fold, row in comparison['protocols'][PROTOCOL].items()})
    return dict(**shared, arms=arms, paper_comparable=False,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())


def evaluate(model, manifest, path, settings):
    known = manifest['stages'][0]['classes']
    threshold = manifest['evaluation']['iou_threshold']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, path, settings['batch_size'], 2)
    spec = dict(manifest['evaluation'], known_only_screening=True, diagnostic_current_precision=True,
                diagnostic_current_ap_anchors=True,
                image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    variants = {view: (known, 'focal', settings['limit'])
                for view in ('emitted', 'without_cross_suppression', 'known_only')}
    post_args = dict(known_ids=known, scoring='focal', max_detections=settings['limit'])
    processors = dict(emitted=CompeteWithEmittedUnknown(**post_args),
                      without_cross_suppression=DeduplicateKnown(**post_args),
                      known_only=DeduplicateKnown(**post_args))
    device = torch.device('cuda')
    known_ids = torch.tensor(known, device=device)
    category_hits = {category: Counter(gt=0, localized=0, correct_class=0) for category in known}

    def observe(outputs, targets, batch_images, _score_tables):
        top_class = known_ids[outputs['pred_logits'][..., known].argmax(dim=-1)]
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        for index in batch_images:
            target = targets[index]
            labels = target['labels'].to(device)
            released = torch.isin(labels, known_ids)
            labels = labels[released]
            targets_xy = box_cxcywh_to_xyxy(target['boxes'].to(device)[released])
            overlap = box_iou(boxes[index], targets_xy)[0] > threshold
            localized = overlap.any(dim=0)
            correct = (overlap & (top_class[index, :, None] == labels[None, :])).any(dim=0)
            for category, has_box, has_class in zip(labels.tolist(), localized.tolist(), correct.tolist(), strict=True):
                category_hits[category].update(dict(gt=1, localized=has_box, correct_class=has_class))

    report = evaluate_views(model, loader, dataset.coco, variants, spec, [], known, device,
                            include_current_stats=True, exclude_unknown=('known_only',),
                            postprocessors=processors, print_freq=64, rank_observer=observe)
    report['released_query_coverage_by_class'] = category_hits
    return report


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    plan_sha = file_sha256(plan_path)
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    results = {}
    for name, arm in plan['arms'].items():
        model, _ = load_detector(None, Path(arm['checkpoint']), torch.device('cuda'))
        results[name] = {}
        for fold, entry in plan['folds'].items():
            print(f'Evaluating S Task 1 {name}/{fold}', flush=True)
            path = Path(entry['path'])
            if file_sha256(path) != entry['sha256']:
                raise ValueError(f'Changed frozen holdout for {fold}')
            report = evaluate(model, manifest, path, plan)
            if report['emitted']['metrics']['known_ap50'] != arm['expected_ap'][fold]:
                raise ValueError(f'Original competition scores did not reproduce archived AP for {name}/{fold}')
            write_json(OUTPUT / f'{name}-{fold}.json', dict(plan_sha256=plan_sha, report=report))
            results[name][fold] = {view: report[view]['metrics']['known_ap50']
                                   for view in ('emitted', 'without_cross_suppression', 'known_only')}
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_sha, known_ap50=results))


if __name__ == '__main__':
    main()
