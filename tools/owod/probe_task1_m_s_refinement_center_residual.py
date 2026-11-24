"""Fit a bounded residual for released Task 1 learned-query centers."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import FIT_IMAGES, LOCAL_FEATURE_OUTPUT, source_ids
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure, prepare
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = LOCAL_FEATURE_OUTPUT.with_name('task1-m-s-refinement-center-residual')
PROTECTED_FIT_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-protected-center-fit')
HALF_RANGE_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-protected-center-half-range')
ASSIGNMENT_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-fit-assignment')
FIT_FOLDS_OUTPUT = PROTECTED_FIT_OUTPUT.with_name('task1-m-s-refinement-protected-center-fit-folds')
RANGE_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-range')
LOCAL_SHA = '448222b0494130f1bd6e65198440d667355166c761a423517127241d25bf0263'
RANGE_SHA = '39fe7dd79bcde92006f0a04d2400198d9fed7a9c7550392c0ddb92ec2ad0505f'
FIT_FOLDS_SHA = '8c7b498aa9c142805eeb647746339cdca770866f97b0af5bffc9f0f2b0eb25aa'
QUARTER_FIT_SHA = '99732ddbfcb08e8c5d4ff9a7a3393c4704663d1df856789a2e3d27bcbfa9b619'
MOVE_LIMIT = .25
NEAR_IOU = .1
KEEP_COST = 2.
RIDGE = .1


def learned_query_ids(final, state, batch):
    active = state['active'][batch]
    learned = torch.ones(len(final), dtype=torch.bool, device=final.device)
    learned[:len(active)] = ~active
    anchor_area = state['anchors'][batch, 0, 2:].prod()
    return torch.where(learned & (final[:, 2:].prod(dim=1) < 2 * anchor_area))[0]


def local_inputs(outputs, sampler, batch):
    state = outputs['geometry']
    final = outputs['pred_boxes'][batch]
    queries = learned_query_ids(final, state, batch)
    boxes = final[queries]
    row_state = dict(memory=state['memory'][batch:batch+1], masks=[mask[batch:batch+1] for mask in state['masks']])
    local = sampler(row_state, boxes[None])[0]
    features = torch.cat((outputs['query_features'][batch, queries], local, boxes[:, 2:].clamp_min(1e-6).log()),
                         dim=1).float()
    return queries, boxes, features


@torch.no_grad()
def training_rows(model, sampler, loader, plan, *, protect_post=None, known_columns=None, assignment=None):
    rows, residuals, movements = [], [], []
    observed = 0
    move_limit = plan['move_limit']
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        if protect_post is not None:
            unit = outputs['pred_boxes'].new_ones((len(targets), 2))
            scored = protect_post(outputs, unit, include_top_indices=True)
        for batch, target in enumerate(targets):
            if not len(target['boxes']):
                continue
            queries, boxes, features = local_inputs(outputs, sampler, batch)
            truth = target['boxes'].cuda()
            truth_xyxy = box_cxcywh_to_xyxy(truth)
            native_iou = box_iou(box_cxcywh_to_xyxy(boxes), truth_xyxy)[0]
            best_iou = native_iou.amax(dim=1)
            small = target['source_box_area'].cuda() < plan['small_area']
            overlap, index = native_iou.masked_fill(~small[None], -1.).max(dim=1)
            paired = truth[index]
            target_offset = ((paired[:, :2] - boxes[:, :2]) / boxes[:, 2:].clamp_min(1e-6)).clamp(
                -move_limit, move_limit)
            shifted = torch.cat(((boxes[:, :2] + target_offset * boxes[:, 2:]).clamp(0, 1), boxes[:, 2:]), dim=1)
            predicted = box_cxcywh_to_xyxy(shifted)
            expected = truth_xyxy[index]
            intersection = (torch.minimum(predicted[:, 2:], expected[:, 2:])
                            - torch.maximum(predicted[:, :2], expected[:, :2])).clamp_min(0).prod(dim=1)
            union = shifted[:, 2:].prod(dim=1) + paired[:, 2:].prod(dim=1) - intersection
            movable = (overlap >= NEAR_IOU) & (best_iou <= .5) & (intersection > .5 * union)
            keep = best_iou >= NEAR_IOU
            if protect_post is not None:
                row = scored[batch]
                protected = row['top_indices'][row['labels'] != 91] // (known_columns + 1)
                keep &= ~torch.isin(queries, protected)
            if assignment is not None:
                final = outputs['pred_boxes'][batch]
                covered = (box_iou(box_cxcywh_to_xyxy(final), truth_xyxy)[0] > .5).any(dim=0)
                per_target = torch.bincount(index[movable & keep], minlength=len(truth))
                assignment['missed'].update(per_target[small & ~covered].tolist())
                assignment['covered'].update(per_target[small & covered].tolist())
            rows.append(features[keep].cpu())
            residuals.append(torch.where(movable[:, None], target_offset, 0.)[keep].cpu())
            movements.append(movable[keep].cpu())
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Fitting features {step}/{len(loader)} batches', flush=True)
    if observed != plan['fit_images'] or not rows:
        raise ValueError('Incomplete released fit replay')
    return torch.cat(rows), torch.cat(residuals), torch.cat(movements)


def fit_residual(rows, targets, move):
    if not move.any() or move.all():
        raise ValueError('Continuous residual needs both moving and stationary released queries')
    rows, targets = rows.double(), targets.double()
    mean, scale = rows.mean(dim=0), rows.std(dim=0, unbiased=False).clamp_min(1e-4)
    rows = (rows - mean) / scale
    inputs = torch.cat((rows, rows.new_ones((len(rows), 1))), dim=1)
    weights = torch.where(move, 1. / int(move.sum()), KEEP_COST / int((~move).sum())).double()
    weighted = inputs.T * weights
    penalty = torch.eye(inputs.shape[1], dtype=inputs.dtype) * RIDGE
    penalty[-1, -1] = 0.
    solution = torch.linalg.solve(weighted @ inputs + penalty, weighted @ targets)
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=solution[:-1].tolist(),
                bias=solution[-1].tolist(), move=int(move.sum()), keep=int((~move).sum()))


def residual_transform(params, sampler, device, *, move_limit=MOVE_LIMIT):
    mean, scale, weights, bias = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weights', 'bias'))

    def transform(outputs):
        out = outputs['pred_boxes'].clone()
        for batch in range(len(out)):
            queries, boxes, features = local_inputs(outputs, sampler, batch)
            residual = (((features - mean) / scale) @ weights + bias).clamp(-move_limit, move_limit)
            out[batch, queries, :2] = (boxes[:, :2] + residual * boxes[:, 2:]).clamp(0, 1)
        return out

    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--protected-fit', action='store_true')
    modes.add_argument('--half-range', action='store_true')
    modes.add_argument('--fit-assignment', action='store_true')
    args = parser.parse_args()
    protected_fit = args.protected_fit or args.half_range or args.fit_assignment
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released center residual')
    source = verified(LOCAL_FEATURE_OUTPUT / 'comparison.json', LOCAL_SHA)
    prior = verified(LOCAL_FEATURE_OUTPUT / 'plan.json', source['plan_sha256'])
    plan = prepare()
    keys = ('jobs', 'seed', 'images', 'batch_size', 'limit', 'small_area')
    if source['status'] != 'complete' or source['passed'] or any(plan[key] != prior[key] for key in keys):
        raise ValueError('Expected the stopped midpoint selection and matched released cohorts')
    plan.update(local_comparison_sha256=LOCAL_SHA, fit_images=FIT_IMAGES, fit_order_sha256=prior['baseline_fit_orders'],
                move_limit=MOVE_LIMIT, near_iou=NEAR_IOU, keep_cost=KEEP_COST, ridge=RIDGE,
                factor='Fit a continuous XY residual using frozen decoder query and native local encoder features. '
                       'Use a clipped supervised center move only when a small released hit can be gained; '
                       'nearby queries that preserve their native center supervise zero displacement.',
                advance=prior['advance'])
    output = OUTPUT
    if protected_fit:
        from tools.owod.probe_task1_m_s_refinement_protected_residual import OUTPUT as PROTECTED_OUTPUT, protect_known

        protected_sha = 'ae7af8fdfce6ad0977fa732fb0917162d2047d3b5149240bf4d94b6d4f474a13'
        protected = verified(PROTECTED_OUTPUT / 'comparison.json', protected_sha)
        protected_plan = verified(PROTECTED_OUTPUT / 'plan.json', protected['plan_sha256'])
        evidence = verified(RANGE_OUTPUT / 'comparison.json', RANGE_SHA)
        fixed = (*keys, 'fit_images', 'fit_order_sha256', 'move_limit', 'near_iou', 'keep_cost', 'ridge')
        if (protected['status'] != 'complete' or not protected['passed'] or evidence['status'] != 'complete'
                or any(plan[key] != protected_plan[key] for key in fixed)):
            raise ValueError('Expected the protected residual and fixed-fit range replay')
        output = PROTECTED_FIT_OUTPUT
        plan.update(protected_comparison_sha256=protected_sha, range_comparison_sha256=RANGE_SHA,
                    factor='Exclude native emitted known Top-50 queries from residual fit examples. '
                           'Use the same fit sources, feature inputs, displacement limit and inference protection.')
    if args.half_range or args.fit_assignment:
        folds = verified(FIT_FOLDS_OUTPUT / 'comparison.json', FIT_FOLDS_SHA)
        baseline = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
        baseline_plan = verified(PROTECTED_FIT_OUTPUT / 'plan.json', baseline['plan_sha256'])
        if (folds['status'] != 'complete' or folds['passed'] or baseline['status'] != 'complete'
                or not baseline['passed'] or any(plan[key] != baseline_plan[key] for key in fixed)):
            raise ValueError('Expected the completed quarter-range fit and its fold replay')
        plan.update(fit_folds_sha256=FIT_FOLDS_SHA, quarter_fit_sha256=QUARTER_FIT_SHA)
    if args.half_range:
        output = HALF_RANGE_OUTPUT
        plan.update(move_limit=.5,
                    factor='Increase the protected fit center cap from a quarter to a half of the native extent. '
                           'Use the same features, query filter and protection.')
    if args.fit_assignment:
        output = ASSIGNMENT_OUTPUT
        plan.update(factor='Count quarter-range protected fit positives per released small target.',
                    advance='Assess repeated positive moves among small targets missed by every native query.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = output / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count checkpoint: {protocol}')
            inputs = (checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest), job)
            fit_loader, fit_order = training_loader(*inputs, dict(plan, images=FIT_IMAGES), source_box_area=True,
                                                    offset=plan['images'])
            loader, order = training_loader(*inputs, plan, source_box_area=True)
            if (order != job['expected_order'] or fit_order != plan['fit_order_sha256'][protocol]
                    or set(source_ids(fit_loader)) & set(source_ids(loader))):
                raise ValueError(f'Changed released image linkage: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            sampler = model.detr.transformer.instance_proposals.refined_features
            known = job['classes']
            post = (CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
                    if protected_fit else None)
            assignment = {'missed': Counter(), 'covered': Counter()} if args.fit_assignment else None
            print(f'Replaying {protocol} released fit sources' if args.fit_assignment
                  else f'Fitting {protocol} released continuous center residual', flush=True)
            rows = training_rows(model, sampler, fit_loader, plan, protect_post=post, known_columns=len(known),
                                 assignment=assignment)
            if args.fit_assignment:
                move = int(rows[2].sum())
                keep = len(rows[2]) - move
                expected = baseline['protocols'][protocol]['residual']
                audited_moves = sum(count * targets for group in assignment.values() for count, targets in group.items())
                if move != expected['move'] or keep != expected['keep'] or audited_moves != move:
                    raise ValueError(f'Changed quarter-range fit assignments: {protocol}')
                result = dict(move=move, keep=keep, targets_by_positive_count=assignment)
                write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order, **result))
                protocols[protocol] = result
                del model
                continue
            params = fit_residual(*rows)
            print(f'Evaluating {protocol} frozen center residual', flush=True)
            transform = residual_transform(params, sampler, device, move_limit=plan['move_limit'])
            protection = Counter(restored_known_queries=0, reverted_images=0)
            if protected_fit:
                transform = protect_known(transform, post, known, protection)
            report = measure(model, known, loader, plan, box_transform=transform)
            if protected_fit:
                report['protection'] = protection
            for group, row in report['groups'].items():
                reference = source['protocols'][protocol]['groups'][group]
                if any(row[key] != reference[key] for key in ('gt', 'final', 'native_known_tp')):
                    raise ValueError(f'Changed native released baseline: {protocol}/{group}')
            small = report['groups']['small']
            passed = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                      and all(row['revised_known_tp'] >= row['native_known_tp'] for row in report['groups'].values()))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order,
                                  residual=params, passed=passed, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen center residual plan: {dest}')
        if args.fit_assignment:
            protocols[protocol] = {key: report[key] for key in ('move', 'keep', 'targets_by_positive_count')}
            continue
        protocols[protocol] = {key: report[key] for key in ('residual', 'groups', 'changes', 'passed')}
        if protected_fit:
            protocols[protocol]['protection'] = report['protection']
    if args.fit_assignment:
        write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))
    else:
        write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                                   passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
