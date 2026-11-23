"""Assess localization quality transfer between released Task 2 categories."""

from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_refill_layer_agreement import (
    DEVICE, OUTPUT as AGREEMENT, intersection_union,
)
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_formal_refill_screen import OUTPUT as SHORT, SCREEN_RESULT_SHA
from tools.owod.screening_data import screening_files, source_ids, write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = AGREEMENT.with_name('task2-m-s-refill-quality-transfer')
AGREEMENT_SHA = '35317d940ad166841c39bc91ff5c0b69e0a51af30625f6bd5b5182dc39a27a08'
FEATURES = ('objectness_logit', 'log_width', 'log_height', 'iou_last_to_penultimate', 'iou_last_to_antepenultimate')
MARGIN_IOU = .10
POSITIVE_IOU = .50
RIDGE = .01


def prepare():
    result = verified(AGREEMENT / 'comparison.json', AGREEMENT_SHA)
    if result['status'] != 'complete' or result['candidates']:
        raise ValueError('Expected the completed fixed-rule screen')
    source = verified(AGREEMENT / 'plan.json', result['plan_sha256'])
    short_result = verified(SHORT / 'comparison.json', SCREEN_RESULT_SHA)
    short = verified(SHORT / 'plan.json', short_result['plan_sha256'])
    data = {}
    for protocol, job in source['jobs']['three'].items():
        manifest_path = Path(job['manifest'])
        verified(manifest_path, job['manifest_sha256'])
        original = short['jobs'][protocol]['benchmark']
        command = original['runs'][-1]['command']
        screen = Path(command[command.index('--owod-screening') + 1])
        if file_sha256(screen) != original['screening_sha256']:
            raise ValueError(f'Changed original screening spec: {protocol}')
        manifest, files = screening_files(manifest_path, screen, 1)
        blocked = set(manifest['evaluation']['image_source_ids'])
        blocked.update(image for fold in job['folds'].values() for image in fold['order'])
        entries = {}
        for role, path in (('fit', files['increment_train']), ('transfer', files['full_val'])):
            names = source_ids(read_json(path))
            if names & blocked:
                raise ValueError(f'Overlapping fit, transfer or OWOD sources: {protocol}/{role}')
            blocked.update(names)
            entries[role] = dict(path=str(path), sha256=file_sha256(path), images=len(names))
        data[protocol] = entries
    jobs = {duration: {protocol: {key: value for key, value in job.items() if key != 'folds'}
                       for protocol, job in protocols.items()} for duration, protocols in source['jobs'].items()}
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=AGREEMENT_SHA, jobs=jobs, data=data,
                batch_size=source['settings']['batch_size'], limit=source['settings']['limit'],
                features=list(FEATURES), fit_minimum_iou=MARGIN_IOU, hit_iou=POSITIVE_IOU, ridge=RIDGE,
                known_fit_group='current', known_transfer_group='previous', paper_comparable=False,
                criterion='On all four endpoint/protocol pairs improve MRR and retain at least as many top-1 hits '
                          'as both the raw and layer-IoU controls on released-class transfer images.')


def query_signals(outputs, native):
    scores, _, unknown = native.candidate_scores(outputs)
    queries = scores.flatten(1).topk(native.max_detections, dim=1).indices // scores.shape[-1]
    unused = torch.ones_like(unknown, dtype=torch.bool).scatter_(1, queries, False)
    final = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    agreement = []
    for offset in (-1, -2):
        earlier = box_cxcywh_to_xyxy(outputs['aux_outputs'][offset]['pred_boxes'])
        intersection, union = intersection_union(final, earlier)
        agreement.append(torch.where(union > 0, intersection / union, 0.))
    width, height = outputs['pred_boxes'][..., 2:].clamp_min(1e-6).log().unbind(-1)
    features = torch.stack((outputs['pred_objectness'], width, height, *agreement), dim=-1)
    return features, unknown, agreement[0], unused, final


def released_overlap(boxes, annotations, categories, size):
    rows = [ann['voc_bbox'] for ann in annotations if ann['category_id'] in categories and not ann.get('difficult')]
    if not rows:
        return boxes.new_zeros(boxes.shape[0])
    height, width = size.tolist()
    target = boxes.new_tensor(rows)
    target[:, :2] -= 1
    target /= boxes.new_tensor([width, height, width, height])
    overlaps, _ = box_iou(boxes, target)
    return overlaps.amax(dim=-1)


@torch.inference_mode()
def collect(model, root, entry, previous, current, plan, *, fitting):
    annotation = Path(entry['path'])
    if file_sha256(annotation) != entry['sha256']:
        raise ValueError(f'Changed released annotations: {annotation}')
    dataset, loader = make_loader(root, annotation, plan['batch_size'], 2)
    if len(dataset) != entry['images']:
        raise ValueError(f'Changed released image count: {annotation}')
    native = DiscoveryPostProcess(known_ids=previous + current, scoring='focal', max_detections=plan['limit'])
    wanted = set(current if fitting else previous)
    competing = set(current) if not fitting else set()
    for step, (samples, targets) in enumerate(loader, 1):
        features, unknown, agreement, unused, boxes = query_signals(model(samples.to(DEVICE)), native)
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            anns = dataset.coco.imgToAnns[image]
            nearby = released_overlap(boxes[i], anns, wanted, target['orig_size'])
            if fitting:
                selected = unused[i] & (nearby >= plan['fit_minimum_iou'])
            else:
                other = released_overlap(boxes[i], anns, competing, target['orig_size'])
                selected = unused[i]
                nearby = torch.where(nearby > other, nearby, 0.)
            yield features[i, selected].cpu().numpy(), nearby[selected].cpu().numpy(), (
                unknown[i, selected].cpu().numpy(), agreement[i, selected].cpu().numpy()
            )
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)


def fit_readout(samples, plan):
    feature_batches, overlap_batches = zip(*[(features, overlap) for features, overlap, _ in samples])
    features = np.concatenate(feature_batches).astype(np.float64)
    overlaps = np.concatenate(overlap_batches).astype(np.float64)
    target = (overlaps > plan['hit_iou']).astype(np.float64)
    if not 0 < target.sum() < len(target):
        raise ValueError('The released fit cohort must produce positive and partial-overlap queries')
    mean = features.mean(0)
    scale = features.std(0).clip(min=1e-6)
    norm = (features - mean) / scale

    def objective(weights):
        bias, coef = weights[0], weights[1:]
        logits = norm @ coef + bias
        cost = (np.logaddexp(0, logits) - target * logits).mean() + .5 * plan['ridge'] * (coef @ coef)
        residual = expit(logits) - target
        gradient = np.r_[residual.mean(), norm.T @ residual / len(target) + plan['ridge'] * coef]
        return cost, gradient

    result = minimize(objective, np.zeros(norm.shape[1] + 1), method='L-BFGS-B', jac=True)
    if not result.success:
        raise RuntimeError(f'Localization readout did not converge: {result.message}')
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x[1:].tolist(),
                bias=float(result.x[0]), samples=len(target), hits=int(target.sum()), log_loss=float(result.fun))


def compare(samples, readout, threshold):
    ranks = dict(raw=[], layer_iou=[], quality=[])
    for features, overlap, (unknown, agreement) in samples:
        hits = overlap > threshold
        if not hits.any():
            continue
        norm = (features - readout['mean']) / readout['scale']
        quality = expit(norm @ readout['coefficients'] + readout['bias'])
        for arm, priority in zip(ranks, (unknown, unknown * agreement, unknown * quality)):
            order = np.argsort(-priority, kind='stable')
            ranks[arm].append(int(np.flatnonzero(hits[order])[0]) + 1)
    if not ranks['raw']:
        raise ValueError('No transferable released-class box among unused queries')
    metrics = {arm: dict(top1=r.count(1), top5=sum(k <= 5 for k in r),
                         mrr=float(np.mean(1. / np.asarray(r))), ranks=r) for arm, r in ranks.items()}
    controls = ('raw', 'layer_iou')
    passed = (metrics['quality']['mrr'] > max(metrics[k]['mrr'] for k in controls)
              and metrics['quality']['top1'] >= max(metrics[k]['top1'] for k in controls))
    return dict(images_with_hit=len(ranks['raw']), metrics=metrics, passed=passed)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    endpoints = {}
    for duration, protocols in plan['jobs'].items():
        endpoints[duration] = {}
        for protocol, job in protocols.items():
            path = OUTPUT / f'{duration}-{protocol}.json'
            if not path.exists():
                checkpoint = Path(job['checkpoint'])
                if file_sha256(checkpoint) != job['checkpoint_sha256']:
                    raise ValueError(f'Changed frozen endpoint: {checkpoint}')
                manifest_path = Path(job['manifest'])
                manifest = verified(manifest_path, job['manifest_sha256'])
                root = resolve_image_root(manifest_path, manifest)
                model, _ = load_detector(None, checkpoint, DEVICE)
                model.eval()
                entries = plan['data'][protocol]
                groups = (job['previous'], job['current'])
                print(f'Fitting {duration}/{protocol} released Current quality readout', flush=True)
                readout = fit_readout(collect(model, root, entries['fit'], *groups, plan, fitting=True), plan)
                print(f'Evaluating {duration}/{protocol} released Previous transfer', flush=True)
                transfer = compare(collect(model, root, entries['transfer'], *groups, plan, fitting=False),
                                   readout, plan['hit_iou'])
                write_json(path, dict(plan_sha256=digest, readout=readout, transfer=transfer))
                del model
            result = read_json(path)
            if result['plan_sha256'] != digest:
                raise ValueError(f'Changed quality transfer plan: {path}')
            endpoints[duration][protocol] = result['transfer']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, endpoints=endpoints,
                                               passed=all(row['passed'] for rows in endpoints.values() for row in rows.values())))


if __name__ == '__main__':
    main()
