"""Fit a bounded residual for released Task 1 learned-query centers."""

from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import FIT_IMAGES, LOCAL_FEATURE_OUTPUT, source_ids
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure, prepare
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = LOCAL_FEATURE_OUTPUT.with_name('task1-m-s-refinement-center-residual')
LOCAL_SHA = '448222b0494130f1bd6e65198440d667355166c761a423517127241d25bf0263'
MOVE_LIMIT = .25
NEAR_IOU = .1
KEEP_COST = 2.
RIDGE = .1


def local_inputs(outputs, sampler, batch):
    state = outputs['geometry']
    final = outputs['pred_boxes'][batch]
    active = state['active'][batch]
    learned = torch.ones(len(final), dtype=torch.bool, device=final.device)
    learned[:len(active)] = ~active
    anchor_area = state['anchors'][batch, 0, 2:].prod()
    queries = torch.where(learned & (final[:, 2:].prod(dim=1) < 2 * anchor_area))[0]
    boxes = final[queries]
    row_state = dict(memory=state['memory'][batch:batch+1], masks=[mask[batch:batch+1] for mask in state['masks']])
    local = sampler(row_state, boxes[None])[0]
    features = torch.cat((outputs['query_features'][batch, queries], local, boxes[:, 2:].clamp_min(1e-6).log()),
                         dim=1).float()
    return queries, boxes, features


@torch.no_grad()
def training_rows(model, sampler, loader, plan):
    rows, residuals, movements = [], [], []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        for batch, target in enumerate(targets):
            if not len(target['boxes']):
                continue
            _, boxes, features = local_inputs(outputs, sampler, batch)
            truth = target['boxes'].cuda()
            truth_xyxy = box_cxcywh_to_xyxy(truth)
            native_iou = box_iou(box_cxcywh_to_xyxy(boxes), truth_xyxy)[0]
            best_iou = native_iou.amax(dim=1)
            small = target['source_box_area'].cuda() < plan['small_area']
            overlap, index = native_iou.masked_fill(~small[None], -1.).max(dim=1)
            paired = truth[index]
            target_offset = ((paired[:, :2] - boxes[:, :2]) / boxes[:, 2:].clamp_min(1e-6)).clamp(
                -MOVE_LIMIT, MOVE_LIMIT)
            shifted = torch.cat(((boxes[:, :2] + target_offset * boxes[:, 2:]).clamp(0, 1), boxes[:, 2:]), dim=1)
            predicted = box_cxcywh_to_xyxy(shifted)
            expected = truth_xyxy[index]
            intersection = (torch.minimum(predicted[:, 2:], expected[:, 2:])
                            - torch.maximum(predicted[:, :2], expected[:, :2])).clamp_min(0).prod(dim=1)
            union = shifted[:, 2:].prod(dim=1) + paired[:, 2:].prod(dim=1) - intersection
            movable = (overlap >= NEAR_IOU) & (best_iou <= .5) & (intersection > .5 * union)
            keep = best_iou >= NEAR_IOU
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


def residual_transform(params, sampler, device):
    mean, scale, weights, bias = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weights', 'bias'))

    def transform(outputs):
        out = outputs['pred_boxes'].clone()
        for batch in range(len(out)):
            queries, boxes, features = local_inputs(outputs, sampler, batch)
            residual = (((features - mean) / scale) @ weights + bias).clamp(-MOVE_LIMIT, MOVE_LIMIT)
            out[batch, queries, :2] = (boxes[:, :2] + residual * boxes[:, 2:]).clamp(0, 1)
        return out

    return transform


def main():
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
            print(f'Fitting {protocol} released continuous center residual', flush=True)
            params = fit_residual(*training_rows(model, sampler, fit_loader, plan))
            print(f'Evaluating {protocol} frozen center residual', flush=True)
            report = measure(model, job['classes'], loader, plan, box_transform=residual_transform(params, sampler, device))
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
        protocols[protocol] = {key: report[key] for key in ('residual', 'groups', 'changes', 'passed')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
