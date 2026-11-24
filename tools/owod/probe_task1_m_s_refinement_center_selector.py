"""Fit a selector for the frozen Task 1 small learned-query center candidate."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import (
    CENTER_OUTPUT, measure, prepare, refine_learned_boxes,
)
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = CENTER_OUTPUT.with_name('task1-m-s-refinement-center-selector')
QUALITY_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-quality-selector')
TRACE_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-selector-trace')
RANK_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-placement')
CANDIDATES_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-center-candidates')
CENTER_COMPARISON_SHA = '1df8e79d21bc4459dddb8efec1e34fdce1d6e418562cc0dba490a984cf0cc79a'
SELECTOR_COMPARISON_SHA = '9a32eb2af9fcf9988b341baf6893f1c75feb44b5c373928b5046f67d95d1f6a3'
TRACE_COMPARISON_SHA = '5885955be7819524332ed13daee33ec243677793cbe6877ae5064451f3274e34'
CANDIDATE_COMPARISON_SHA = '013497fa1cbfc364003dfd3ffb3a9d9819a627ed29e9a23fdd8787608f893636'
QUALITY_COMPARISON_SHA = '6db34143914d3cd53ee9361d4ff472633618812942494b443e2510af7f40d573'
FEATURES = ('encoder_quality', 'max_known_logit', 'objectness_logit', 'novelty_logit',
            'log_width', 'log_height', 'normalized_abs_dx', 'normalized_abs_dy')
QUALITY_FEATURE = 'shifted_minus_native_geometry_quality'
FIT_IMAGES = 1536
NEGATIVE_COST = 2.
L2 = .01


def features(outputs, batch, queries, center, quality, known, refined_quality=None):
    boxes = outputs['pred_boxes'][batch, queries]
    extent = boxes[:, 2:].clamp_min(1e-6)
    displacement = (center - boxes[:, :2]).abs() / extent
    known_max = outputs['pred_logits'][batch, queries][:, known].amax(dim=-1)
    scores = torch.stack((quality, known_max, outputs['pred_objectness'][batch, queries],
                          outputs['pred_novelty'][batch, queries]), dim=-1)
    values = torch.cat((scores, extent.log(), displacement), dim=-1).float()
    if refined_quality is not None:
        delta = values.new_empty((len(queries), 1))
        if len(queries):
            geometry = outputs['geometry']
            state = dict(memory=geometry['memory'][batch:batch+1],
                         masks=[mask[batch:batch+1] for mask in geometry['masks']])
            adjusted = torch.cat((center, boxes[:, 2:]), dim=-1)
            shifted_quality = refined_quality(state, adjusted[None]).sigmoid()[0]
            delta[:, 0] = shifted_quality - outputs['geometry_quality'][batch, queries]
        values = torch.cat((values, delta), dim=-1)
    return values


@torch.no_grad()
def fit_examples(model, loader, known, count, small_area, refined_quality=None, *, all_centers=False):
    examples, labels = [], []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))

        def collect(batch, queries, center, quality):
            target = targets[batch]
            boxes = outputs['pred_boxes'][batch, queries]
            xyxy = box_cxcywh_to_xyxy(target['boxes'].cuda())
            native_iou = box_iou(box_cxcywh_to_xyxy(boxes), xyxy)[0]
            shifted_iou = box_iou(box_cxcywh_to_xyxy(torch.cat((center, boxes[:, 2:]), dim=1)), xyxy)[0]
            old_hit, new_hit = native_iou > .5, shifted_iou > .5
            small = target['source_box_area'].cuda() < small_area
            gained = (new_hit & ~old_hit & small).any(dim=1)
            lost = (old_hit & ~new_hit).any(dim=1)
            relevant = gained | lost
            rows = features(outputs, batch, queries, center, quality, known, refined_quality)
            examples.append(rows[relevant].cpu())
            labels.append((gained & ~lost)[relevant].float().cpu())
            return relevant

        def collect_candidates(batch, queries, centers, quality):
            rows, columns = torch.where(quality.isfinite())
            collect(batch, queries[rows], centers[rows, columns], quality[rows, columns])

        if all_centers:
            refine_learned_boxes(outputs, center_only=True, observe_centers=collect_candidates)
        else:
            refine_learned_boxes(outputs, center_only=True, choose_centers=collect)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Fit examples {step}/{len(loader)} batches', flush=True)
    if observed != count or not examples:
        raise ValueError('Incomplete released fitting replay')
    return torch.cat(examples), torch.cat(labels)


def fit_selector(examples, labels):
    positive = labels.bool()
    if not positive.any() or positive.all():
        raise ValueError('Released fitting replay needs examples of both center decisions')
    mean, scale = examples.mean(dim=0), examples.std(dim=0, unbiased=False).clamp_min(1e-4)
    inputs = (examples - mean) / scale
    weight = torch.zeros(inputs.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS((weight, bias), max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        losses = F.binary_cross_entropy_with_logits(inputs @ weight + bias, labels, reduction='none')
        loss = losses[positive].mean() + NEGATIVE_COST * losses[~positive].mean() + L2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return dict(mean=mean.tolist(), scale=scale.tolist(), weight=weight.detach().tolist(),
                bias=bias.detach().item(), positive=int(positive.sum()), negative=int((~positive).sum()))


def correction_logits(params, known, device, refined_quality=None):
    mean, scale, weight = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weight'))

    def score(outputs, batch, queries, center, quality):
        values = features(outputs, batch, queries, center, quality, known, refined_quality)
        return ((values - mean) / scale) @ weight + params['bias']

    return score


def center_transform(params, known, device, refined_quality=None, *, rank_centers=False):
    score = correction_logits(params, known, device, refined_quality)

    def transform(outputs):
        def choose(batch, queries, center, quality):
            return score(outputs, batch, queries, center, quality) > 0

        def rank(batch, queries, centers, quality):
            rows, columns = torch.where(quality.isfinite())
            logits = score(outputs, batch, queries[rows], centers[rows, columns], quality[rows, columns])
            ranked = torch.full_like(quality, -torch.inf)
            ranked[rows, columns] = logits.masked_fill(logits <= 0, -torch.inf)
            return ranked

        if rank_centers:
            return refine_learned_boxes(outputs, center_only=True, rank_centers=rank)
        return refine_learned_boxes(outputs, center_only=True, choose_centers=choose)

    return transform


def source_ids(loader):
    subset = loader.dataset
    base = subset.dataset
    return [base.coco.imgs[base.ids[i]]['owod_source_id'] for i in subset.indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--quality-delta', action='store_true')
    modes.add_argument('--rank-centers', action='store_true')
    args = parser.parse_args()
    quality_delta = args.quality_delta or args.rank_centers
    output = RANK_OUTPUT if args.rank_centers else QUALITY_OUTPUT if quality_delta else OUTPUT
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    source = verified(CENTER_OUTPUT / 'comparison.json', CENTER_COMPARISON_SHA)
    old = verified(CENTER_OUTPUT / 'plan.json', source['plan_sha256'])
    if source['status'] != 'complete' or source['passed']:
        raise ValueError('Expected the completed midpoint preflight')
    plan = prepare()
    reference_keys = ('jobs', 'seed', 'images', 'batch_size', 'limit', 'small_area')
    if any(plan[key] != old[key] for key in reference_keys):
        raise ValueError('Changed released training sources')
    plan.update(center_comparison_sha256=CENTER_COMPARISON_SHA, fit_images=FIT_IMAGES, features=list(FEATURES),
                negative_cost=NEGATIVE_COST, l2=L2, advance=old['advance'],
                factor='Fit one standardized logistic selector per protocol on the next released images in the '
                       'locked replay permutation. Accept only a positive logit. A positive fitting example adds '
                       'a strict small-target localization hit without losing any existing hit of that query.')
    baseline_orders = {}
    if quality_delta:
        baseline = verified(OUTPUT / 'comparison.json', SELECTOR_COMPARISON_SHA)
        trace = verified(TRACE_OUTPUT / 'comparison.json', TRACE_COMPARISON_SHA)
        baseline_plan = verified(OUTPUT / 'plan.json', baseline['plan_sha256'])
        comparison_keys = (*reference_keys, 'fit_images', 'features', 'negative_cost', 'l2')
        if (baseline['status'] != 'complete' or baseline['passed'] or trace['status'] != 'complete'
                or any(plan[key] != baseline_plan[key] for key in comparison_keys)):
            raise ValueError('Expected the stopped frozen selector and its matched trace')
        for protocol, report in baseline['protocols'].items():
            endpoint = read_json(OUTPUT / f'{protocol}.json')
            if endpoint['plan_sha256'] != baseline['plan_sha256'] or endpoint['selector'] != report['selector']:
                raise ValueError(f'Changed frozen selector fitting source: {protocol}')
            baseline_orders[protocol] = endpoint['fit_order_sha256']
        plan.update(features=[*FEATURES, QUALITY_FEATURE], baseline_fit_orders=baseline_orders,
                    selector_comparison_sha256=SELECTOR_COMPARISON_SHA,
                    selector_trace_sha256=TRACE_COMPARISON_SHA,
                    factor='Add the frozen geometry head score difference for the shifted and native boxes. '
                           'Preserve data, candidate construction, fit labels, and logistic decision.')
    if args.rank_centers:
        candidates = verified(CANDIDATES_OUTPUT / 'comparison.json', CANDIDATE_COMPARISON_SHA)
        previous = verified(QUALITY_OUTPUT / 'comparison.json', QUALITY_COMPARISON_SHA)
        previous_plan = verified(QUALITY_OUTPUT / 'plan.json', previous['plan_sha256'])
        if (candidates['status'] != 'complete' or previous['status'] != 'complete' or not previous['passed']
                or any(plan[key] != previous_plan[key] for key in comparison_keys)):
            raise ValueError('Expected matched released midpoint and candidate diagnostics')
        plan.update(candidate_comparison_sha256=CANDIDATE_COMPARISON_SHA,
                    quality_comparison_sha256=QUALITY_COMPARISON_SHA,
                    factor='Fit on every eligible midpoint. Use the highest positive decision logit per learned '
                           'query, or preserve the native center. Reuse the same features and loss.')
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
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            loader_args = (checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest), job)
            fit_loader, fit_order = training_loader(*loader_args, dict(plan, images=FIT_IMAGES), source_box_area=True,
                                                    offset=plan['images'])
            loader, order = training_loader(*loader_args, plan, source_box_area=True)
            fit_ids, replay_ids = source_ids(fit_loader), source_ids(loader)
            if (order != job['expected_order'] or len(set(fit_ids)) != FIT_IMAGES
                    or len(set(replay_ids)) != plan['images'] or set(fit_ids) & set(replay_ids)):
                raise ValueError(f'Invalid released source identity separation: {protocol}')
            if quality_delta and fit_order != baseline_orders[protocol]:
                raise ValueError(f'Changed selector fitting order: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            refined_quality = model.detr.transformer.instance_proposals.refined_quality if quality_delta else None
            print(f'Fitting {protocol} released center selector', flush=True)
            training = fit_examples(model, fit_loader, known, FIT_IMAGES, plan['small_area'], refined_quality,
                                    all_centers=args.rank_centers)
            params = fit_selector(*training)
            if args.quality_delta and any(params[key] != baseline['protocols'][protocol]['selector'][key]
                                          for key in ('positive', 'negative')):
                raise ValueError(f'Changed selector fitting labels: {protocol}')
            print(f'Evaluating {protocol} frozen center selector', flush=True)
            transform = center_transform(params, known, device, refined_quality, rank_centers=args.rank_centers)
            report = measure(model, known, loader, plan, box_transform=transform)
            for size, row in report['groups'].items():
                reference = source['protocols'][protocol]['groups'][size]
                if any(row[key] != reference[key] for key in ('gt', 'final', 'native_known_tp')):
                    raise ValueError(f'Changed native released baseline: {protocol}/{size}')
            small = report['groups']['small']
            passed = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                      and all(row['revised_known_tp'] >= row['native_known_tp'] for row in report['groups'].values()))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order,
                                  selector=params, passed=passed, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed released selector plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('selector', 'groups', 'changes', 'passed')}
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(report['passed'] for report in protocols.values())))


if __name__ == '__main__':
    main()
