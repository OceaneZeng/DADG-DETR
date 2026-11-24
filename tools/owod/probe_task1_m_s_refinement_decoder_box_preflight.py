"""Measure frozen Task 1 box changes on released training images."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.instance_proposals import nearby_anchor_ids
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_known_dedup import COMPETITION_OUTPUT, CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_local_saliency import SOURCE_SHA as TRAIN_REPLAY_SHA
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import OUTPUT as TRAIN_REPLAY, training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-decoder-box-preflight')
RANK_OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-encoder-rank-preflight')
EXTENT_OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-learned-extent-preflight')
CENTER_OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-learned-center-preflight')
ATTRIBUTION_OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-final-slot-attribution')
GEOMETRY_OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-learned-geometry')
COMPETITION_SHA = 'b3801feb4573a0b6d9586d70f5479d8be541acea665678baaf5f76f974570f67'
DECODER_SHA = 'd16ba04de5eeaa1be2a67fcb5bcf8c514e70a874618339ed5d6b6d10d3561d95'
ATTRIBUTION_SHA = '31ed1449712d060261553d2666fad941e8f9e2701d15e357c910498cddd4f917'
GEOMETRY_SHA = 'e8919659633fdf48a6d0dddd19fb8cccf32c9ae5eab216d751e931777feb9b12'


def prepare():
    comparison = verified(COMPETITION_OUTPUT / 'comparison.json', COMPETITION_SHA)
    plan = verified(COMPETITION_OUTPUT / 'plan.json', comparison['plan_sha256'])
    if comparison['status'] != 'complete' or comparison['passed']:
        raise ValueError('Expected the completed known/unknown competition screen')
    source = verified(TRAIN_REPLAY / 'comparison.json', TRAIN_REPLAY_SHA)
    train = verified(TRAIN_REPLAY / 'plan.json', source['plan_sha256'])
    jobs = {}
    for protocol, job in plan['jobs'].items():
        reference = read_json(TRAIN_REPLAY / f'{protocol}.json')
        if reference['plan_sha256'] != source['plan_sha256'] or job['manifest_sha256'] != train['jobs'][protocol]['manifest_sha256']:
            raise ValueError(f'Changed released image replay: {protocol}')
        data = train['jobs'][protocol]
        jobs[protocol] = dict(checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
                              manifest_sha256=job['manifest_sha256'], classes=data['classes'],
                              training=data['training'], training_sha256=data['training_sha256'],
                              expected_order=reference['sample_order_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                competition_sha256=COMPETITION_SHA, training_replay_sha256=TRAIN_REPLAY_SHA,
                jobs=jobs, seed=train['seed'], images=train['images'], batch_size=train['batch_size'],
                limit=plan['limit'], paper_comparable=False, small_area=32**2,
                factor='Average the normalized boxes at decoder layers 5 and 6 by native query identity.',
                advance='Require positive released-small dense coverage in S and no decline in M. '
                        'Report class-correct Top-50 matches and the any-last-two-layer geometry upper bound.')


def refine_learned_boxes(outputs, *, center_only=False, observe_neighbors=None):
    """Refine one component of a learned query with a nearby fine-level box."""
    final = outputs['pred_boxes']
    geometry = outputs['geometry']
    revised = final.clone()
    finest_cells = geometry['masks'][0][0].numel()
    active = geometry['active']
    for batch, boxes in enumerate(final):
        anchors = geometry['anchors'][batch, :finest_cells]
        candidates = geometry['boxes'][batch, :finest_cells]
        valid = geometry['valid'][batch, :finest_cells]
        anchor_area = anchors[0, 2:].prod()
        area = candidates[:, 2:].prod(dim=-1)
        candidate_ids = torch.where(valid & (area <= anchor_area))[0]
        if not len(candidate_ids):
            continue
        # Use the same small-query cohort for both component changes.
        learned = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        learned[:active.shape[1]] = ~active[batch]
        queries = torch.where(learned & (boxes[:, 2:].prod(dim=-1) < 2 * anchor_area))[0]
        base = boxes[queries]
        near = nearby_anchor_ids(base, anchors, candidate_ids)
        peers = candidates[near]
        if center_only:
            centers = peers[..., :2]
            inside = ((centers - base[:, None, :2]).abs() <= base[:, None, 2:] * .5).all(dim=-1)
            scores = geometry['logits'][batch, near].masked_fill(~inside, -torch.inf)
            best, position = scores.max(dim=1)
            keep = best.isfinite()
            selected = candidates[near[keep, position[keep]], :2]
            revised[batch, queries[keep], :2] = base[keep, :2].lerp(selected, .5)
            continue
        base_xyxy = box_cxcywh_to_xyxy(base)[:, None]
        peer_xyxy = box_cxcywh_to_xyxy(peers)
        left = torch.maximum(base_xyxy[..., :2], peer_xyxy[..., :2])
        right = torch.minimum(base_xyxy[..., 2:], peer_xyxy[..., 2:])
        intersection = (right - left).clamp_min(0).prod(dim=-1)
        union = base[:, None, 2:].prod(dim=-1) + area[near] - intersection
        overlaps = intersection / union.clamp_min(1e-8)
        if observe_neighbors is not None:
            observe_neighbors(batch, queries, peers, overlaps)
        best, position = overlaps.max(dim=1)
        keep = best > .5
        peer_extent = candidates[near[keep, position[keep]], 2:]
        revised[batch, queries[keep], 2:] = (base[keep, 2:] * peer_extent).sqrt()
    return revised


@torch.inference_mode()
def measure(model, known, loader, plan, *, learned_component=None):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    stats = {key: Counter() for key in ('all', 'small')}
    changes = Counter()
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        layers = outputs['aux_outputs']
        if len(layers) != 5:
            raise ValueError('Expected the locked six-layer decoder')
        final = outputs['pred_boxes']
        previous = layers[-1]['pred_boxes']
        if learned_component is None:
            candidate = final.lerp(previous, .5)
        else:
            candidate = refine_learned_boxes(outputs, center_only=learned_component == 'center')
        revised = dict(outputs, pred_boxes=candidate)
        if learned_component is not None:
            changed = (candidate != final).any(dim=-1)
            changes.update(queries=int(changed.sum()), images=int(changed.any(dim=1).sum()))
        unit = final.new_ones((len(targets), 2))
        native, changed_outputs = post(outputs, unit), post(revised, unit)
        for batch, target in enumerate(targets):
            labels = target['labels'].cuda()
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            small = target['source_box_area'].cuda() < plan['small_area']
            dense = [(box_iou(box_cxcywh_to_xyxy(boxes[batch]), truth)[0] > .5).any(dim=0)
                     for boxes in (final, previous, candidate)]
            matches = [((box_iou(row['boxes'], truth)[0] > .5) & (row['labels'][:, None] == labels)).any(dim=0)
                       for row in (native[batch], changed_outputs[batch])]
            for name, mask in (('all', torch.ones_like(small)), ('small', small)):
                original, penultimate, updated = (x[mask] for x in dense)
                native_tp, updated_tp = (x[mask] for x in matches)
                stats[name].update(gt=int(mask.sum()), final=int(original.sum()),
                                   gained=int((updated & ~original).sum()), lost=int((original & ~updated).sum()),
                                   native_known_tp=int(native_tp.sum()))
                if learned_component is not None:
                    stats[name].update(revised=int(updated.sum()), revised_known_tp=int(updated_tp.sum()))
                else:
                    stats[name].update(penultimate=int(penultimate.sum()), blend=int(updated.sum()),
                                       either=int((original | penultimate).sum()), blend_known_tp=int(updated_tp.sum()))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images'] or stats['small']['gt'] <= 0:
        raise ValueError('Incomplete released replay or missing small targets')
    return dict(groups=stats, changes=changes) if learned_component is not None else stats


@torch.inference_mode()
def measure_ranks(model, known, loader, plan):
    groups = {key: Counter() for key in ('under_16_sq', '16_to_32_sq', 'all_small')}
    miss_rows = []
    observed = 0
    multiplier = model.detr.transformer.instance_proposals.shortlist_multiplier
    for step, (samples, targets) in enumerate(loader, 1):
        geo = model(samples.to('cuda'))['geometry']
        for batch, target in enumerate(targets):
            area = target['source_box_area']
            small = torch.where(area < plan['small_area'])[0]
            truth = target['boxes'][small].cuda()
            if not len(truth):
                continue
            if not set(target['labels'].tolist()) <= set(known):
                raise ValueError('Expected released training categories only')
            valid = torch.where(geo['valid'][batch])[0]
            scores = geo['logits'][batch, valid]
            chosen = geo['chosen'][batch, geo['active'][batch]]
            shortlist = scores.topk(min(len(scores), geo['chosen'].shape[1] * multiplier)).indices
            chosen_at = torch.searchsorted(valid, chosen)
            if not torch.isin(chosen_at, shortlist).all():
                raise ValueError('Selected cells are outside the reconstructed native shortlist')
            near = nearby_anchor_ids(truth, geo['anchors'][batch], valid)
            near_at = torch.searchsorted(valid, near)
            accurate = box_iou(box_cxcywh_to_xyxy(geo['boxes'][batch, valid]), box_cxcywh_to_xyxy(truth))[0] > .5
            dense_ok = accurate.any(dim=0)
            shortlist_ok = accurate[shortlist].any(dim=0)
            chosen_ok = accurate[chosen_at].any(dim=0)
            near_ok = accurate.T.gather(1, near_at).any(dim=1)
            flags = torch.stack((dense_ok, shortlist_ok, chosen_ok, near_ok), dim=1).tolist()
            for pos, ((dense, pool, keep, near_hit), original) in enumerate(zip(flags, small.tolist(), strict=True)):
                group = 'under_16_sq' if area[original] < 16**2 else '16_to_32_sq'
                for name in (group, 'all_small'):
                    groups[name].update(gt=1, dense=dense, shortlist=pool, selected=keep,
                                        dense_outside_shortlist=dense and not pool,
                                        outside_shortlist_near_positive=dense and not pool and near_hit,
                                        outside_shortlist_without_near_positive=dense and not pool and not near_hit,
                                        shortlist_lost_at_selection=pool and not keep)
                if dense and not pool:
                    high = scores[accurate[:, pos]].max()
                    miss_rows.append(dict(group=group, near_positive=near_hit,
                                          first_rank=1+int((scores > high).sum()), native_cut=len(shortlist)))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images'] or not groups['all_small']['gt']:
        raise ValueError('Incomplete released small-box ranking replay')
    return dict(groups=groups, shortlist_misses=miss_rows)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--encoder-ranks', action='store_true')
    modes.add_argument('--learned-extent', action='store_true')
    modes.add_argument('--learned-center', action='store_true')
    args = parser.parse_args()
    output = EXTENT_OUTPUT if args.learned_extent else RANK_OUTPUT if args.encoder_ranks else OUTPUT
    if args.learned_center:
        output = CENTER_OUTPUT
    learned_component = 'center' if args.learned_center else 'extent' if args.learned_extent else None
    plan = prepare()
    if args.encoder_ranks:
        decoder = verified(OUTPUT / 'comparison.json', DECODER_SHA)
        if decoder['status'] != 'complete' or decoder['passed']:
            raise ValueError('Expected the completed box-average preflight')
        plan.update(decoder_preflight_sha256=DECODER_SHA, factor='Trace native encoder scores and exact released near-16 supervision.',
                    advance='Use the source of actual shortlist misses to choose the next released supervision contrast.')
    if args.learned_extent:
        attribution = verified(ATTRIBUTION_OUTPUT / 'comparison.json', ATTRIBUTION_SHA)
        if attribution['status'] != 'complete':
            raise ValueError('Expected the completed final-query source trace')
        plan.update(attribution_comparison_sha256=ATTRIBUTION_SHA,
                    factor='For a learned query, find the best IoU among the 16 nearest valid finest anchors '
                           'whose encoder box area does not exceed the finest anchor area. At IoU > 0.5, '
                           'take the geometric mean of the decoder and encoder widths and heights, keeping '
                           'the decoder center and logits.')
    if args.learned_center:
        geometry = verified(GEOMETRY_OUTPUT / 'comparison.json', GEOMETRY_SHA)
        source_plan = verified(GEOMETRY_OUTPUT / 'plan.json', geometry['plan_sha256'])
        if geometry['status'] != 'complete' or any(plan[key] != source_plan[key] for key in ('jobs', 'seed', 'images')):
            raise ValueError('Expected the completed released component diagnosis on the same sources')
        plan.update(geometry_comparison_sha256=GEOMETRY_SHA,
                    factor='For a small learned query, take the highest frozen encoder quality score among '
                           'the same 16 nearby eligible fine-level candidates whose centers lie within the '
                           'decoder box. Move halfway toward that center, keeping decoder extent and logits.')
    if learned_component is not None:
        plan['advance'] = ('Improve released-small dense coverage in S with no drop in M. Require no drop '
                           'in class-correct small or all Top-50 matches on either protocol.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = output / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            print(f'Evaluating {protocol} released images in {output.name}', flush=True)
            if args.encoder_ranks:
                report = measure_ranks(model, job['classes'], loader, plan)
            elif learned_component is not None:
                report = measure(model, job['classes'], loader, plan, learned_component=learned_component)
            else:
                report = dict(groups=measure(model, job['classes'], loader, plan))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen plan: {dest}')
        protocols[protocol] = report['groups']
        if learned_component is not None:
            protocols[protocol] = dict(groups=report['groups'], changes=report['changes'])
    result = dict(status='complete', plan_sha256=digest, protocols=protocols)
    if learned_component is not None:
        for protocol, report in protocols.items():
            small = report['groups']['small']
            report['passed'] = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                                and all(row['revised_known_tp'] >= row['native_known_tp']
                                        for row in report['groups'].values()))
        result['passed'] = all(report['passed'] for report in protocols.values())
    elif not args.encoder_ranks:
        m, s = (protocols[key]['small'] for key in ('m-owodb', 's-owodb'))
        result['passed'] = m['blend'] >= m['final'] and s['blend'] > s['final']
    write_json(output / 'comparison.json', result)


if __name__ == '__main__':
    main()
