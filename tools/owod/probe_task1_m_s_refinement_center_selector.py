"""Fit a selector for the frozen Task 1 small learned-query center candidate."""

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
CENTER_COMPARISON_SHA = '1df8e79d21bc4459dddb8efec1e34fdce1d6e418562cc0dba490a984cf0cc79a'
FEATURES = ('encoder_quality', 'max_known_logit', 'objectness_logit', 'novelty_logit',
            'log_width', 'log_height', 'normalized_abs_dx', 'normalized_abs_dy')
FIT_IMAGES = 1536
NEGATIVE_COST = 2.
L2 = .01


def features(outputs, batch, queries, center, quality, known):
    boxes = outputs['pred_boxes'][batch, queries]
    extent = boxes[:, 2:].clamp_min(1e-6)
    displacement = (center - boxes[:, :2]).abs() / extent
    known_max = outputs['pred_logits'][batch, queries][:, known].amax(dim=-1)
    scores = torch.stack((quality, known_max, outputs['pred_objectness'][batch, queries],
                          outputs['pred_novelty'][batch, queries]), dim=-1)
    return torch.cat((scores, extent.log(), displacement), dim=-1).float()


@torch.no_grad()
def fit_examples(model, loader, known, count, small_area):
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
            examples.append(features(outputs, batch, queries, center, quality, known)[relevant].cpu())
            labels.append((gained & ~lost)[relevant].float().cpu())
            return relevant

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


def center_transform(params, known, device):
    mean, scale, weight = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weight'))

    def transform(outputs):
        def choose(batch, queries, center, quality):
            values = features(outputs, batch, queries, center, quality, known)
            return ((values - mean) / scale) @ weight + params['bias'] > 0

        return refine_learned_boxes(outputs, center_only=True, choose_centers=choose)

    return transform


def source_ids(loader):
    subset = loader.dataset
    base = subset.dataset
    return [base.coco.imgs[base.ids[i]]['owod_source_id'] for i in subset.indices]


def main():
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
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            args = (checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest), job)
            fit_loader, fit_order = training_loader(*args, dict(plan, images=FIT_IMAGES), source_box_area=True,
                                                    offset=plan['images'])
            loader, order = training_loader(*args, plan, source_box_area=True)
            fit_ids, replay_ids = source_ids(fit_loader), source_ids(loader)
            if (order != job['expected_order'] or len(set(fit_ids)) != FIT_IMAGES
                    or len(set(replay_ids)) != plan['images'] or set(fit_ids) & set(replay_ids)):
                raise ValueError(f'Invalid released source identity separation: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            print(f'Fitting {protocol} released center selector', flush=True)
            training = fit_examples(model, fit_loader, known, FIT_IMAGES, plan['small_area'])
            params = fit_selector(*training)
            print(f'Evaluating {protocol} frozen center selector', flush=True)
            report = measure(model, known, loader, plan, box_transform=center_transform(params, known, device))
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
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(report['passed'] for report in protocols.values())))


if __name__ == '__main__':
    main()
