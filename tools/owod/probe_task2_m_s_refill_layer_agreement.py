"""Ablate two decoder-box agreement rules on the registered frozen Task 2 folds."""

import json
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess, UnknownRefillScore
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_formal_refill_capacity import (
    COHORT_ROOT, DEVICE, OVERLAP_OUTPUT, OUTPUT as CAPACITY,
)
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = CAPACITY.with_name('task2-m-s-refill-layer-agreement')
CAPACITY_SHA = '517293e0203622472085540df1a31a2080398142987e814369cda14957af0765'
VARIANTS = dict(guard=(True, False), rank=(False, True), combined=(True, True))
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')


def intersection_union(left, right):
    intersection = (torch.minimum(left[..., 2:], right[..., 2:])
                    - torch.maximum(left[..., :2], right[..., :2])).clamp_min(0).prod(dim=-1)
    left_area = (left[..., 2:] - left[..., :2]).clamp_min(0).prod(dim=-1)
    right_area = (right[..., 2:] - right[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection, left_area + right_area - intersection


class LayerAgreementRefill(UnknownRefillScore):
    def __init__(self, *, guard, rank, **options):
        super().__init__(**options)
        self.guard = guard
        self.rank = rank

    def candidate_scores(self, outputs):
        scores, categories, unknown = DiscoveryPostProcess.candidate_scores(self, outputs)
        original, _, _ = super().candidate_scores(outputs)
        columns = len(categories)
        n = self.max_detections
        values, positions = scores.flatten(1).topk(n, dim=1)
        queries, classes = positions // columns, positions % columns
        removed = ~original.flatten(1).gather(1, positions).isfinite()
        replace = removed.any(1)
        drop = removed.int().argmax(1)

        last_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        prev_boxes = box_cxcywh_to_xyxy(outputs['aux_outputs'][-1]['pred_boxes'])
        if self.guard:
            # Require the same higher-ranked known detection to support the drop at both layers.
            old = prev_boxes.gather(1, queries[..., None].expand(-1, -1, 4))
            intersection, union = intersection_union(old[:, :, None], old[:, None, :])
            penultimate_overlap = (union > 0) & (intersection >= self.min_same_class_iou * union)
            rank = torch.arange(n, device=scores.device)
            agrees = self.same_class_overlap(outputs['pred_boxes'], queries, classes,
                                             penultimate_overlap & (rank[:, None] > rank[None, :]))
            replace &= agrees.gather(1, drop[:, None]).squeeze(1)

        priority = unknown
        if self.rank:
            intersection, union = intersection_union(last_boxes, prev_boxes)
            agreement = torch.where(union > 0, intersection / union, 0.)
            priority = unknown * agreement
        used = torch.zeros_like(unknown, dtype=torch.bool).scatter_(1, queries, True)
        promoted = priority.masked_fill(used, -torch.inf).argmax(1)
        selected = torch.full_like(scores.flatten(1), -torch.inf).scatter_(1, positions, values)
        batch = torch.arange(scores.shape[0], device=scores.device)[replace]
        selected[batch, positions[batch, drop[batch]]] = -torch.inf
        selected[batch, promoted[batch] * columns + columns - 1] = unknown[batch, promoted[batch]]
        return selected.reshape_as(scores), categories, unknown


def summarize(report, reference, settings, previous):
    before = report['views']['native']['metrics']
    results = {}
    for arm, view in report['views'].items():
        if arm == 'native':
            continue
        after = view['metrics']
        deltas = {key: after[key] - before[key] for key in METRICS}
        unknown_native, unknown_after = (set(report['unknown_matches'][key]) for key in ('native', arm))
        old_native, old_after = ({(label, gt) for label in previous for gt in report['previous_matches'][key][label]}
                                 for key in ('native', arm))
        refills = 0
        for image, counts in report['per_image'].items():
            native, candidate = counts['native'], counts[arm]
            extra = candidate['unknown'] - native['unknown']
            if (extra not in (0, 1) or candidate['current'] != native['current']
                    or candidate['previous'] != native['previous'] - extra):
                raise ValueError(f'Invalid one-detection budget: {image}/{arm}')
            refills += extra
        if unknown_native - unknown_after or old_after - old_native:
            raise ValueError(f'Unexpected match direction for a single-slot refill: {arm}')
        passed = (deltas['u_recall'] >= settings['minimum_unknown_recall_gain']
                  and all(deltas[key] >= -settings['refill_ap_tolerance'] for key in METRICS[:3])
                  and deltas['a_ose'] <= 0)
        results[arm] = dict(deltas=deltas, refills=refills, passed=passed,
                            added_unknown_tp=len(unknown_after - unknown_native),
                            previous_tp_lost=len(old_native - old_after))
    for arm in ('native', 'unknown_refill'):
        if json.loads(json.dumps(report['views'][arm])) != reference['report']['views'][arm]:
            raise ValueError(f'Author VOC control replay changed: {arm}')
    if results['unknown_refill']['passed'] != reference['summary']['passed']:
        raise ValueError('Changed original refill acceptance')
    return results


def evaluate(model, job, entry, settings):
    previous, current = job['previous'], job['current']
    known = previous + current
    manifest_path = Path(job['manifest'])
    manifest = verified(manifest_path, job['manifest_sha256'])
    released = COHORT_ROOT / entry['released']['file']
    full = OVERLAP_OUTPUT / entry['full']['file']
    for path, key in ((released, 'released'), (full, 'full')):
        if file_sha256(path) != entry[key]['sha256']:
            raise ValueError(f'Changed annotations: {path}')
    root = resolve_image_root(manifest_path, manifest)
    dataset, loader = make_loader(root, released, settings['batch_size'], 2)
    spec = dict(manifest['evaluation'], image_source_ids=entry['order'])
    options = dict(known_ids=known, scoring='focal', max_detections=settings['limit'],
                   protected_ids=current, min_same_class_iou=settings['min_same_class_iou'])
    posts = dict(unknown_refill=UnknownRefillScore(**options))
    posts.update({name: LayerAgreementRefill(guard=guard, rank=rank, **options)
                  for name, (guard, rank) in VARIANTS.items()})
    reference = verified(Path(entry['reference']), entry['reference_sha256'])
    result = measure(model, dataset, loader, known, previous, current, spec, settings['limit'],
                     evaluation_coco=COCO(str(full)), extra_posts=posts, include_perturbed=False,
                     audit_previous_matches=True, audit_unknown_matches=True)
    return dict(summary=summarize(result, reference, settings, previous), report=result)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    capacity = verified(CAPACITY / 'comparison.json', CAPACITY_SHA)
    if capacity['status'] != 'complete':
        raise ValueError('The frozen endpoint diagnosis must be complete')
    source = verified(CAPACITY / 'plan.json', capacity['plan_sha256'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                capacity_sha256=CAPACITY_SHA, source_plan_sha256=capacity['plan_sha256'],
                settings=source['settings'], jobs=source['jobs'], training=False, paper_comparable=False,
                variants={name: dict(guard=guard, rank=rank) for name, (guard, rank) in VARIANTS.items()},
                guard_rule='Same higher-ranked same-class box with overlap at both last decoder layers.',
                rank_rule='Unknown score times aligned last-two-layer box IoU. Emit the original score.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for duration, jobs in plan['jobs'].items():
        results[duration] = {}
        for protocol, job in jobs.items():
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed frozen endpoint: {checkpoint}')
            model, _ = load_detector(None, checkpoint, DEVICE)
            model.eval()
            results[duration][protocol] = {}
            for fold, entry in job['folds'].items():
                path = OUTPUT / f'{duration}-{protocol}-{fold}.json'
                if not path.exists():
                    print(f'Evaluating {duration}/{protocol}/{fold}', flush=True)
                    write_json(path, dict(plan_sha256=digest, **evaluate(model, job, entry, plan['settings'])))
                data = read_json(path)
                if data['plan_sha256'] != digest:
                    raise ValueError(f'Changed diagnostic plan: {path}')
                results[duration][protocol][fold] = data['summary']
            del model
    arms = ('unknown_refill', *VARIANTS)
    passes = {duration: {arm: sum(row[arm]['passed'] for protocol in protocols.values() for row in protocol.values())
                         for arm in arms} for duration, protocols in results.items()}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, endpoints=results,
                                               folds_passing=passes,
                                               candidates=[arm for arm in VARIANTS
                                                           if all(r[arm] == 4 for r in passes.values())]))


if __name__ == '__main__':
    main()
