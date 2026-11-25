"""Identify the unknown queries suppressing native known matches after reservation."""

from collections import Counter
from functools import partial
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_fine_rank import DECODER_OUTPUT, mined_selector
from tools.owod.probe_task1_m_s_refinement_known_dedup import OVERLAP
from tools.owod.probe_task1_m_s_refinement_mined_seed_holdout import DECODER_SHA, holdout_loader
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_m_s_reservation_attention import OUTPUT as ATTENTION, evaluate, prepare
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ATTENTION.with_name('task1-m-s-reservation-anchor-identity')
ATTENTION_SHA = 'd6fec6df949b9890395b7b5d423a2b5ddc406c67fd69ff8c3a9262b3c3488bd6'


def shortlist_unknown(scores, limit):
    columns = scores.shape[-1]
    indices = scores.flatten(1).topk(limit, dim=1).indices
    return [row[row % columns == columns - 1] // columns for row in indices]


def trace(rows, small_area, post, native, protected, picks, targets, witnesses, losses):
    native_scores = post.scores_before_competition(native)[0]
    scores, categories, unknown = post.scores_before_competition(protected)
    final = post.candidate_scores(protected)[0].flatten(1)
    native_anchors = shortlist_unknown(native_scores, post.max_detections)
    anchors = shortlist_unknown(scores, post.max_detections)
    boxes = box_cxcywh_to_xyxy(protected['pred_boxes']).float()
    columns = len(categories)
    for batch, (matches, indices) in enumerate(witnesses):
        lost = losses[batch]
        if not lost.any():
            continue
        native_witnesses = matches & scores[batch].flatten()[indices].isfinite()[:, None]
        suppressed_targets = lost & native_witnesses.any(dim=0) & ~(matches & final[batch, indices].isfinite()[:, None]).any(dim=0)
        if not suppressed_targets.any():
            continue
        if batch not in picks:
            raise ValueError('Overlap loss occurred on an unchanged image')
        retained_witnesses = native_witnesses[:, suppressed_targets].any(dim=1)
        queries = indices[retained_witnesses] // columns
        current = anchors[batch]
        injected = torch.isin(current, picks[batch][0])
        previously_shortlisted = torch.isin(current, native_anchors[batch])
        iou = box_iou(boxes[batch, queries], boxes[batch, current])[0]
        known_scores = scores[batch, queries, :-1].amax(dim=1)
        unknown_scores = unknown[batch, current]
        edges = (iou > OVERLAP) & (unknown_scores[None] > known_scores[:, None])
        if not edges.any(dim=1).all() or (edges & (~injected & previously_shortlisted)[None]).any():
            raise ValueError('Witness edges disagree with the native overlap suppression and retained parity')

        flags = native_witnesses[retained_witnesses][:, suppressed_targets]
        for column, index in enumerate(torch.where(suppressed_targets)[0].tolist()):
            relevant = flags[:, column]
            triggering = (edges & relevant[:, None]).any(dim=0)
            without_injected = bool((relevant & ~(edges & ~injected[None]).any(dim=1)).any())
            without_retained = bool((relevant & ~(edges & injected[None]).any(dim=1)).any())
            anchors_for_target = [dict(query=int(current[i]), replaced=bool(injected[i]),
                                       in_native_shortlist=bool(previously_shortlisted[i]),
                                       witnesses=queries[relevant & edges[:, i]].tolist())
                                  for i in torch.where(triggering)[0].tolist()]
            target = targets[batch]
            rows.append(dict(image_id=int(target['image_id']), target_index=index, class_id=int(target['labels'][index]),
                             small=bool(target['source_box_area'][index] < small_area),
                             anchors=anchors_for_target, survives_without_injected_edges=without_injected,
                             survives_without_retained_edges=without_retained))


def summarize(rows, replay):
    if (len({(row['image_id'], row['target_index']) for row in rows}) != len(rows)
            or len(rows) != replay['known_losses']['all']['unknown_overlap']
            or sum(row['small'] for row in rows) != replay['known_losses']['small']['unknown_overlap']):
        raise ValueError('Suppression identities do not account for the original stage losses')
    counts = Counter()
    for row in rows:
        origins = {anchor['replaced'] for anchor in row['anchors']}
        origin = 'both' if len(origins) == 2 else 'replaced_only' if True in origins else 'retained_only'
        counts[origin] += 1
        counts['survives_without_injected_edges'] += row['survives_without_injected_edges']
        counts['survives_without_retained_edges'] += row['survives_without_retained_edges']
    return dict(overlap_losses=len(rows), small_overlap_losses=sum(row['small'] for row in rows), **counts)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the retained-anchor traceback')
    plan, holdout, mined = prepare()
    first = verified(DECODER_OUTPUT / 'comparison.json', DECODER_SHA)
    first_plan = verified(DECODER_OUTPUT / 'plan.json', first['plan_sha256'])
    attention = verified(ATTENTION / 'comparison.json', ATTENTION_SHA)
    prior = verified(ATTENTION / 'plan.json', attention['plan_sha256'])
    bindings = ('jobs', 'seed', 'images', 'batch_size', 'limit', 'small_area', 'mined_sha256')
    if (first['status'] != 'complete' or attention['status'] != 'complete'
            or any(plan[k] != first_plan[k] or plan[k] != prior[k] for k in bindings)):
        raise ValueError('Expected the unchanged paired reservation endpoints and cohorts')
    plan.update(attention_sha256=ATTENTION_SHA, first_sha256=DECODER_SHA, first_plan_sha256=first['plan_sha256'],
                factor='Identify every native witness lost to an emitted unknown/query overlap, with the exact '
                       'attention isolation and no changes to logits, boxes, ranking or suppression.',
                minimum_s_losses_each_slice=3,
                advance='Attribute a consistent replacement-specific mechanism only if, in both S slices, '
                        'at least 3 targets are lost to overlap and at least two thirds have a native witness '
                        'that survives after ignoring only edges from replaced queries at the same shortlist.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    device = torch.device('cuda')
    for protocol, job in plan['jobs'].items():
        loader, holdout_order = holdout_loader(protocol, job, plan)
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, job['manifest_sha256'])
        checkpoint = Path(job['checkpoint'])
        first_loader, first_order = training_loader(checkpoint, Path(job['training']),
                                                   resolve_image_root(manifest_path, manifest), job, plan,
                                                   source_box_area=True)
        if first_order != job['expected_order'] or holdout_order != plan['holdout_order_sha256'][protocol]:
            raise ValueError(f'Changed cohort order: {protocol}')
        model, _ = load_detector(None, checkpoint, device)
        selector = mined_selector(mined[protocol]['ranker'])
        slices = {}
        for name, cohort, order, reference in (
            ('first', first_loader, first_order, first['protocols'][protocol]['groups']),
            ('holdout', loader, holdout_order, holdout[protocol]['mined']),
        ):
            dest = OUTPUT / f'{protocol}-{name}.json'
            if not dest.exists():
                print(f'Tracing {protocol}/{name} unknown anchor identities', flush=True)
                rows = []
                replay = evaluate(model.eval(), cohort, protocol, job, plan, selector, reference,
                                  observe=partial(trace, rows, plan['small_area']))
                if any(replay['changed_rows'][key] for key in ('pred_novelty', 'pred_objectness')):
                    raise ValueError(f'Changed retained unknown scores: {protocol}/{name}')
                if name == 'holdout' and replay != attention['protocols'][protocol]:
                    raise ValueError(f'Changed isolated replay aggregates: {protocol}')
                write_json(dest, dict(plan_sha256=digest, order_sha256=order, replay=replay, rows=rows,
                                      summary=summarize(rows, replay)))
            report = read_json(dest)
            if (report['plan_sha256'] != digest or report['order_sha256'] != order
                    or report['summary'] != summarize(report['rows'], report['replay'])):
                raise ValueError(f'Inconsistent stored identity report: {dest}')
            slices[name] = report['summary']
        protocols[protocol] = slices
        del model
    passed = all(s['overlap_losses'] >= plan['minimum_s_losses_each_slice']
                 and 3 * s.get('survives_without_injected_edges', 0) >= 2 * s['overlap_losses']
                 for s in protocols['s-owodb'].values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               consistent_replacement_origin=passed))


if __name__ == '__main__':
    main()
