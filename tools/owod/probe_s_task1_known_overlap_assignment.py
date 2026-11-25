"""Account for assignment status of emitted unknowns overlapping released S boxes."""

import subprocess
from collections import Counter
from pathlib import Path

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_half_rank_quota import MIDPOINT_SHA, RETRY
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown, OVERLAP
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_score_order import OUTPUT as SCORE_ORDER, PROTOCOL
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SCORE_ORDER.with_name('task1-s-known-overlap-assignment')
SCORE_ORDER_SHA = '4c141b1645c6f07bb644de460b2fcbd267abd513d02d5ac8897f0d556213be04'


def prepare():
    arms = {}
    shared = None
    for arm, output, sha in (('midpoint', RETRY, MIDPOINT_SHA), ('score_order', SCORE_ORDER, SCORE_ORDER_SHA)):
        result = verified(output / 'comparison.json', sha)
        source = verified(output / 'plan.json', result['plan_sha256'])
        spec = {key: source[key] for key in ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit')}
        if result['status'] != 'complete' or (shared is not None and shared != spec):
            raise ValueError('Expected two completed endpoints on identical released-only images')
        shared = spec
        checkpoint = output / PROTOCOL / 'increment/checkpoint.pth'
        if file_sha256(checkpoint) != result['checkpoint_sha256']:
            raise ValueError(f'Changed final checkpoint for {arm}')
        arms[arm] = dict(checkpoint=str(checkpoint), sha256=result['checkpoint_sha256'],
                         expected_ap=result['known_ap50'], comparison_sha256=sha)
    plan = dict(**shared, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                arms=arms, overlap=OVERLAP, paper_comparable=False)
    pin(OUTPUT / 'plan.json', plan)
    return plan


def evaluate(arm, manifest, path, plan):
    device = torch.device('cuda')
    model, criterion = load_detector(None, Path(arm['checkpoint']), device)
    known = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, path, plan['batch_size'], 2)
    known_ids = torch.as_tensor(known, device=device)
    spec = dict(manifest['evaluation'], known_only_screening=True,
                image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    counts = Counter(matched_queries=0, matched_violations=0, eligible_duplicates=0, duplicate_violations=0,
                     emitted_unknown=0, emitted_unknown_over_known=0, emitted_matched_over_known=0,
                     emitted_duplicate_over_known=0, emitted_other_unmatched_over_known=0)

    def observe(outputs, targets, batch_images, score_tables):
        targets = [{key: row[key].to(device) for key in ('labels', 'boxes')} for row in targets]
        assignments = criterion.matcher(outputs, targets)
        _, categories, selected = score_tables[0]
        top_known = known_ids[outputs['pred_logits'][..., known].argmax(-1)]
        queries_xy = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        logits = outputs['pred_logits']
        novelty = outputs['pred_novelty']
        for batch in batch_images:
            target = targets[batch]
            labels = target['labels']
            query, gt = (values.to(device) for values in assignments[batch])
            matched = torch.zeros_like(novelty[batch], dtype=torch.bool)
            matched[query] = True
            over = box_iou(queries_xy[batch], box_cxcywh_to_xyxy(target['boxes']))[0] > plan['overlap']
            duplicate = (~matched) & (over & (top_known[batch, :, None] == labels)).any(-1)
            indices = selected[batch]
            emitted = indices[indices % len(categories) == len(known)] // len(categories)
            emitted_over_known = emitted[over.any(dim=1)[emitted]]
            matched_hits = int(matched[emitted_over_known].sum())
            duplicate_hits = int(duplicate[emitted_over_known].sum())
            counts.update(matched_queries=len(query),
                          matched_violations=int((novelty[batch, query] >= logits[batch, query, labels[gt]]).sum()),
                          eligible_duplicates=int(duplicate.sum()),
                          duplicate_violations=int((novelty[batch, duplicate] >=
                                                   logits[batch, duplicate, top_known[batch, duplicate]]).sum()),
                          emitted_unknown=len(emitted), emitted_unknown_over_known=len(emitted_over_known),
                          emitted_matched_over_known=matched_hits, emitted_duplicate_over_known=duplicate_hits,
                          emitted_other_unmatched_over_known=len(emitted_over_known) - matched_hits - duplicate_hits)

    report = evaluate_views(model, loader, dataset.coco, {'native': (known, 'focal', plan['limit'])},
                            spec, [], known, device, postprocessors={'native': post},
                            print_freq=128, rank_observer=observe)
    if (report['native']['metrics']['known_ap50'] != arm['expected_ap']
            or report['native']['composition']['unknown'] != counts['emitted_unknown']):
        raise ValueError('Native scores or emitted unknown accounting did not reproduce')
    return dict(counts=counts, known_ap50=report['native']['metrics']['known_ap50'])


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen assignment audit')
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    path = Path(plan['holdout'])
    if file_sha256(path) != plan['holdout_sha256']:
        raise ValueError('Changed released-only holdout')
    rows = {}
    for name, arm in plan['arms'].items():
        print(f'Evaluating S Task 1 {name} known-overlap assignment', flush=True)
        rows[name] = evaluate(arm, manifest, path, plan)
        write_json(OUTPUT / f'{name}.json', dict(plan_sha256=digest, **rows[name]))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, arms=rows))


if __name__ == '__main__':
    main()
