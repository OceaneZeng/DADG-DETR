"""Isolate learned fine-level candidate ranking at the same released seed budget."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import L2, NEGATIVE_COST, fit_selector, source_ids
from tools.owod.probe_task1_m_s_refinement_dense_oracle import positive_pairs
from tools.owod.probe_task1_m_s_refinement_fixed_injection import (
    MAX_REPLACEMENTS, NATIVE_OUTPUT, choose_native, fine_candidates,
)
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_m_s_refinement_trajectory import hits
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = NATIVE_OUTPUT.with_name('task1-m-s-refinement-fine-seed-rank')
RESERVATION_SHA = '1e8b6b879f13943ba41338595391500d7158869f55143cdf1be62b6d81451bd3'
POSITIVES_PER_TARGET = 4
NEGATIVES_PER_IMAGE = 16


def missed_boxes(outputs, batch, target, small_area):
    truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
    small = target['source_box_area'].cuda() < small_area
    state = outputs['geometry']
    chosen = state['chosen'][batch, state['active'][batch]]
    seed = hits(state['boxes'][batch, chosen], truth).any(dim=0)
    final = hits(outputs['pred_boxes'][batch], truth).any(dim=0)
    return truth[small & ~seed & ~final]


def features(geometry, batch, cells):
    boxes = geometry['boxes'][batch, cells]
    anchors = geometry['anchors'][batch, cells]
    extent = boxes[:, 2:].clamp_min(1e-6)
    return torch.cat((geometry['memory'][batch, cells], geometry['logits'][batch, cells, None],
                      boxes[:, :2], extent.log(), (boxes[:, :2] - anchors[:, :2]) / extent), dim=1).float()


@torch.no_grad()
def fit_examples(model, loader, plan):
    samples_out, labels_out = [], []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        geometry = outputs['geometry']
        for batch, target in enumerate(targets):
            cells = fine_candidates(geometry, batch)
            truth = missed_boxes(outputs, batch, target, plan['small_area'])
            iou = box_iou(box_cxcywh_to_xyxy(geometry['boxes'][batch, cells]), truth)[0]
            top_iou, indices = iou.topk(min(POSITIVES_PER_TARGET, len(cells)), dim=0)
            positive = indices[top_iou > .5].unique()
            negatives = torch.where(~(iou > .5).any(dim=1))[0]
            priority = geometry['logits'][batch, cells[negatives]].argsort(descending=True, stable=True)
            negative = negatives[priority[:NEGATIVES_PER_IMAGE]]
            picked = torch.cat((positive, negative))
            if len(picked):
                values = features(geometry, batch, cells[picked])
                samples_out.append(values.cpu())
                labels_out.append((torch.arange(len(picked)) < len(positive)).float())
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Fitting rank features {step}/{len(loader)} batches', flush=True)
    if observed != plan['fit_images'] or not samples_out:
        raise ValueError('Incomplete released rank fitting replay')
    return torch.cat(samples_out), torch.cat(labels_out)


@torch.inference_mode()
def measure(model, loader, post, params, known, plan):
    keys = ('native', 'readout', 'label_rank')
    arms = {key: Counter(placed=0, matched=0) for key in keys}
    counts = Counter()
    observed = 0
    device = torch.device('cuda')
    mean, scale, weight = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weight'))
    columns = len(known) + 1
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to(device))
        geometry = outputs['geometry']
        scores = post(outputs, outputs['pred_boxes'].new_ones((len(targets), 2)), include_top_indices=True)
        for batch, (target, row) in enumerate(zip(targets, scores, strict=True)):
            cells = fine_candidates(geometry, batch)
            truth = missed_boxes(outputs, batch, target, plan['small_area'])
            iou = box_iou(box_cxcywh_to_xyxy(geometry['boxes'][batch, cells]), truth)[0]
            values = (features(geometry, batch, cells) - mean) / scale
            readout = values @ weight + params['bias']
            oracle = iou.amax(dim=1) if len(truth) else iou.new_full((len(cells),), -torch.inf)
            oracle = oracle.masked_fill(oracle <= .5, -torch.inf)
            _, reachable = positive_pairs(iou)
            counts.update(misses=len(truth), pool_reachable=int((iou > .5).any(dim=0).sum()),
                          two_cell_upper_bound=min(len(reachable), MAX_REPLACEMENTS))
            for key, priority in zip(keys, (None, readout, oracle), strict=True):
                reserve = choose_native(outputs, batch, row, columns, scores=priority)
                if reserve is not None:
                    picked = reserve[1]
                    _, matched = positive_pairs(iou[torch.searchsorted(cells, picked)])
                    arms[key].update(placed=len(picked), matched=len(matched))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released fine ranking replay')
    return dict(arms=arms, capacity=counts, improved=arms['readout']['matched'] > arms['native']['matched'])


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for released fine-level ranking')
    source = verified(NATIVE_OUTPUT / 'comparison.json', RESERVATION_SHA)
    plan = verified(NATIVE_OUTPUT / 'plan.json', source['plan_sha256'])
    if source['status'] != 'complete' or all(row['groups']['passed'] for row in source['protocols'].values()):
        raise ValueError('Expected the failed native fine-level reservation')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                reservation_sha256=RESERVATION_SHA, positives_per_target=POSITIVES_PER_TARGET,
                negatives_per_image=NEGATIVES_PER_IMAGE, l2=L2, negative_cost=NEGATIVE_COST,
                factor='Fit a linear rank readout over frozen encoder features, raw quality and box geometry. '
                       'Use accurate fine cells for released misses and high raw-score negatives. Change only '
                       'the fine-cell order under the same native protection, suppression and slot budget.',
                advance='Require increased correct replacements in both protocols before replaying the decoder.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    device = torch.device('cuda')
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            inputs = (checkpoint, Path(job['training']), resolve_image_root(path, manifest), job)
            replay, order = training_loader(*inputs, plan, source_box_area=True)
            fit, fit_order = training_loader(*inputs, dict(plan, images=plan['fit_images']), source_box_area=True,
                                            offset=plan['images'])
            if (order != job['expected_order'] or fit_order != plan['fit_order_sha256'][protocol]
                    or set(source_ids(fit)) & set(source_ids(replay))):
                raise ValueError(f'Changed released fit or replay identities: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
            print(f'Fitting {protocol} released fine rank', flush=True)
            params = fit_selector(*fit_examples(model, fit, plan))
            print(f'Comparing {protocol} frozen fine rank', flush=True)
            report = measure(model, replay, post, params, known, plan)
            reference = source['protocols'][protocol]['groups']['transfer']
            if report['arms']['native'] != dict(placed=reference['replacement_slots'], matched=reference['assigned']):
                raise ValueError(f'Changed native reservation on the same replay: {protocol}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order,
                                  ranker=params, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed released fine-rank plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('arms', 'capacity', 'ranker', 'improved')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['improved'] for row in protocols.values())))


if __name__ == '__main__':
    main()
