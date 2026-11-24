"""Attribute released Top-50 match changes after the frozen center residual."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_residual import OUTPUT as RESIDUAL_OUTPUT, residual_transform
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = RESIDUAL_OUTPUT.with_name('task1-m-s-refinement-center-residual-trace')
RESIDUAL_SHA = '8af381886517ff0914429d94d648718ebae3a4a8440e7dce9b023152007bb56c'


@torch.inference_mode()
def trace(model, loader, plan, transform, known):
    columns = len(known) + 1
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    counts = {key: Counter() for key in ('all', 'small')}
    events = []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        original = outputs['pred_boxes']
        revised = dict(outputs, pred_boxes=transform(outputs))
        unit = original.new_ones((len(targets), 2))
        detections = [post(arm, unit, include_top_indices=True) for arm in (outputs, revised)]
        for batch, target in enumerate(targets):
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            labels = target['labels'].cuda()
            small = target['source_box_area'].cuda() < plan['small_area']
            before, after = (arm[batch] for arm in detections)
            old_hits, new_hits = (
                (box_iou(row['boxes'], truth)[0] > .5) & (row['labels'][:, None] == labels)
                for row in (before, after)
            )
            found_before, found_after = (hits.any(dim=0) for hits in (old_hits, new_hits))
            gained, lost = found_after & ~found_before, found_before & ~found_after
            for group, mask in (('all', torch.ones_like(small)), ('small', small)):
                counts[group].update(gt=int(mask.sum()), native_known_tp=int((found_before & mask).sum()),
                                     revised_known_tp=int((found_after & mask).sum()), gained=int((gained & mask).sum()),
                                     lost=int((lost & mask).sum()))
            if not (gained | lost).any():
                continue
            score_arms = [post.candidate_scores(arm)[0][batch].flatten() for arm in (outputs, revised)]
            dense_iou = [box_iou(box_cxcywh_to_xyxy(arm['pred_boxes'][batch]), truth)[0]
                         for arm in (outputs, revised)]
            for pos in torch.where(gained | lost)[0].tolist():
                is_lost = bool(lost[pos])
                hit_arm, other = (0, 1) if is_lost else (1, 0)
                row, peer = (before, after) if is_lost else (after, before)
                matches = old_hits if is_lost else new_hits
                ids = row['top_indices'][matches[:, pos]]
                queries = ids // columns
                qualifies = dense_iou[other][queries, pos] > .5
                eligible = score_arms[other][ids].isfinite()
                emitted = torch.isin(ids, peer['top_indices'])
                if (qualifies & emitted).any():
                    raise ValueError('Changed target has a surviving qualifying emission')
                reasons = torch.where(eligible, 2, 1).masked_fill(~qualifies, 0).tolist()
                source = ('geometry', 'suppressed', 'budget')
                events.append(dict(image=int(target['image_id']), target_index=pos, small=bool(small[pos]),
                                   change='lost' if is_lost else 'gained',
                                   reasons=sorted({source[i] for i in reasons}),
                                   queries=[dict(query=int(query), class_label=int(labels[pos]),
                                                 qualifying_iou=float(hit), other_iou=float(peer_iou),
                                                 other_emitted=bool(emission), reason=source[reason])
                                            for query, hit, peer_iou, emission, reason in zip(
                                                queries, dense_iou[hit_arm][queries, pos],
                                                dense_iou[other][queries, pos], emitted, reasons, strict=True)]))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released replay')
    return counts, events


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released center residual trace')
    source = verified(RESIDUAL_OUTPUT / 'comparison.json', RESIDUAL_SHA)
    if source['status'] != 'complete' or source['passed']:
        raise ValueError('Expected the stopped continuous center residual preflight')
    plan = verified(RESIDUAL_OUTPUT / 'plan.json', source['plan_sha256'])
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                residual_comparison_sha256=RESIDUAL_SHA,
                factor='Pair changed released Top-50 matches. Trace the matching query in the opposite arm '
                       'through strict IoU, locked geometry suppression, and output budget.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            frozen = source['protocols'][protocol]
            sampler = model.detr.transformer.instance_proposals.refined_features
            transform = residual_transform(frozen['residual'], sampler, device)
            print(f'Evaluating {protocol} released Top-50 residual changes', flush=True)
            counts, events = trace(model, loader, plan, transform, job['classes'])
            for group, row in counts.items():
                expected = frozen['groups'][group]
                if any(row[key] != expected[key] for key in ('gt', 'native_known_tp', 'revised_known_tp')):
                    raise ValueError(f'Changed frozen correct-class matches: {protocol}/{group}')
            attribution = {change: Counter('+'.join(row['reasons']) for row in events if row['change'] == change)
                           for change in ('gained', 'lost')}
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, counts=counts,
                                  attribution=attribution, events=events))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen trace plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('counts', 'attribution')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
