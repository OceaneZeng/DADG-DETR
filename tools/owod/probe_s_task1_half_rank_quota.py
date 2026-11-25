"""Trace released-class AP and output quotas for two frozen S Task 1 endpoints."""

import subprocess
from collections import Counter
from pathlib import Path

import torch

from models.owod_benchmark_eval import evaluate_author_benchmark
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_rank_independent import OUTPUT as INDEPENDENT, PROTOCOL
from tools.owod.probe_task1_m_s_refinement_aose import with_quota
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_half_rank_training import INDEPENDENT_SHA, OUTPUT as MIDPOINT
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = MIDPOINT.with_name('task1-s-half-rank-quota')
MIDPOINT_SHA = '58376179753f96f69fc837c62e3ec0409ccc91f34915f0ac9b309f2b08036675'
RETRY = MIDPOINT.with_name('task1-s-half-rank-gradient-retry')


def prepare():
    before = verified(INDEPENDENT / 'comparison.json', INDEPENDENT_SHA)
    after = verified(RETRY / 'comparison.json', MIDPOINT_SHA)
    full = verified(INDEPENDENT / 'plan.json', before['plan_sha256'])
    half = verified(RETRY / 'plan.json', after['plan_sha256'])
    if any(row['status'] != 'complete' for row in (before, after)) or after['passed']:
        raise ValueError('Expected the completed, below-gate midpoint and independent reference')
    for key in ('holdout_sha256', 'manifest_sha256', 'batch_size', 'limit'):
        if full[key] != half[key]:
            raise ValueError(f'Changed frozen replay input: {key}')
    annotations = INDEPENDENT / 'released-holdout.json'
    if file_sha256(annotations) != full['holdout_sha256']:
        raise ValueError('Changed released-only independent holdout')
    control = full['arms']['quality_rank']
    arms = dict(full=dict(checkpoint=control['checkpoint'], sha256=control['checkpoint_sha256'],
                          expected_ap=before['arms']['quality_rank']['known_ap50']),
                half=dict(checkpoint=str(RETRY / PROTOCOL / 'increment/checkpoint.pth'),
                          sha256=after['checkpoint_sha256'], expected_ap=after['known_ap50']))
    for arm in arms.values():
        if file_sha256(Path(arm['checkpoint'])) != arm['sha256']:
            raise ValueError('Changed final checkpoint for quota replay')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, independent_comparison_sha256=INDEPENDENT_SHA,
                midpoint_comparison_sha256=MIDPOINT_SHA, arms=arms,
                **{key: full[key] for key in ('holdout_sha256', 'manifest_sha256', 'batch_size', 'limit')})
    pin(OUTPUT / 'plan.json', plan)
    return plan, annotations


def replay(arm, manifest, annotations, plan, full_quotas=None):
    device = torch.device('cuda')
    model, _ = load_detector(None, Path(arm['checkpoint']), device)
    known = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, annotations, plan['batch_size'], 2)
    specification = dict(manifest['evaluation'], known_only_screening=True, diagnostic_current_precision=True,
                         image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    limit = plan['limit']
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=limit)
    threshold = specification['iou_threshold']
    counts = {str(category): Counter(gt=0, localized=0, correct_class=0, viable=0, emitted=0) for category in known}
    quotas, predictions = {}, {}
    known_ids = torch.as_tensor(known, device=device)
    lost_slots = 0

    def observe(outputs, targets, batch_images, score_tables):
        nonlocal lost_slots
        scores, categories, selected = score_tables[0]
        queries_xy = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        raw_class = known_ids[outputs['pred_logits'][..., known].argmax(dim=-1)]
        for index, image in batch_images.items():
            target = targets[index]
            labels = target['labels'].to(device)
            columns = (labels[:, None] == known_ids).nonzero(as_tuple=True)[1]
            matched = box_iou(queries_xy[index], box_cxcywh_to_xyxy(target['boxes'].to(device)))[0] > threshold
            native = selected[index]
            native_columns = native % len(categories)
            localized = matched.any(0)
            correct = (matched & (raw_class[index, :, None] == labels)).any(0)
            viable = (matched & scores[index][:, columns].isfinite()).any(0)
            emitted = (matched[native // len(categories)] & (native_columns[:, None] == columns)).any(0)
            for label, box, cls, survive, output in zip(labels.tolist(), localized.tolist(), correct.tolist(),
                                                        viable.tolist(), emitted.tolist(), strict=True):
                counts[str(label)].update(gt=1, localized=box, correct_class=cls, viable=survive, emitted=output)
            quotas[image] = int((native_columns != len(known)).sum())
            if full_quotas is not None:
                desired = full_quotas[image]
                available = int(scores[index, :, :-1].isfinite().sum())
                actual = min(desired, available)
                lost_slots += desired - actual
                height, width = target['orig_size'].tolist()
                boxes = queries_xy[index] * queries_xy.new_tensor([width, height, width, height])
                predictions[image] = with_quota(scores[index], categories, boxes, actual, limit)

    report = evaluate_views(model, loader, dataset.coco, {'emitted': (known, 'focal', limit)},
                            specification, [], known, device, include_current_stats=True,
                            postprocessors={'emitted': post}, print_freq=128, rank_observer=observe)
    if report['emitted']['metrics']['known_ap50'] != arm['expected_ap']:
        raise ValueError('Native Known AP did not exactly reproduce the archived endpoint')
    report['released_query_trace'] = counts
    if full_quotas is not None:
        adjusted = evaluate_author_benchmark(predictions, dataset.coco, known, [], known, specification,
                                             include_current_stats=True)
        report['reference_quota'] = dict(known_ap50=adjusted['known_ap50'], known_outputs=sum(full_quotas.values()) - lost_slots,
                                         requested_known_outputs=sum(full_quotas.values()), unfilled_known_slots=lost_slots,
                                         current_class_stats=adjusted['current_class_stats'])
    return report, quotas


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan, annotations = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    arms = plan['arms']
    print('Evaluating S Task 1 full-gradient endpoint', flush=True)
    control, quotas = replay(arms['full'], manifest, annotations, plan)
    write_json(OUTPUT / 'full.json', dict(plan_sha256=digest, report=control, known_quota_by_image=quotas))
    print('Evaluating S Task 1 half-gradient endpoint', flush=True)
    half, _ = replay(arms['half'], manifest, annotations, plan, full_quotas=quotas)
    write_json(OUTPUT / 'half.json', dict(plan_sha256=digest, report=half))
    control_ap = control['emitted']['metrics']['known_ap50']
    native_ap = half['emitted']['metrics']['known_ap50']
    matched_ap = half['reference_quota']['known_ap50']
    recovery = (matched_ap - native_ap) / (control_ap - native_ap)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
                                               full_known_ap50=control_ap, half_known_ap50=native_ap,
                                               half_reference_quota_ap50=matched_ap, ap_gap_recovery=recovery,
                                               unfilled_known_slots=half['reference_quota']['unfilled_known_slots']))


if __name__ == '__main__':
    main()
