"""Fit a keep-or-replace selector for local full encoder boxes on released images."""

from collections import Counter
from pathlib import Path
import subprocess

import torch
from torch.nn.functional import pad

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_center_selector import L2, NEGATIVE_COST, fit_query_choice, source_ids
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure
from tools.owod.probe_task1_m_s_refinement_dense_oracle import OUTPUT as ORACLE_OUTPUT
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_protected_residual import protect_known
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_m_s_refinement_trajectory import near_dense_cells
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ORACLE_OUTPUT.with_name('task1-m-s-refinement-dense-local-selector')
ORACLE_SHA = 'a1006fe51faa25998a625a0b1361670eb0e39a2f445e997da80d0efafe3fb6db'


def candidate_cells(outputs, batch, row, known):
    unknown = row['labels'] == 91
    emitted = row['top_indices'] // (len(known) + 1)
    queries = emitted[unknown & ~torch.isin(emitted, emitted[~unknown])]
    geometry = outputs['geometry']
    valid = torch.where(geometry['valid'][batch])[0]
    near = near_dense_cells(outputs['pred_boxes'][batch, queries], geometry, batch, valid)
    return queries, near


def features(outputs, batch, queries, cells, known):
    native = outputs['pred_boxes'][batch, queries, None]
    geometry = outputs['geometry']
    candidate = geometry['boxes'][batch, cells]
    native_extent = native[..., 2:].clamp_min(1e-6)
    candidate_extent = candidate[..., 2:].clamp_min(1e-6)
    query = outputs['query_features'][batch, queries, None].expand(-1, cells.shape[1], -1)
    encoder = geometry['memory'][batch, cells]
    score = torch.stack((outputs['pred_logits'][batch, queries][:, known].amax(dim=1),
                         outputs['pred_objectness'][batch, queries], outputs['pred_novelty'][batch, queries],
                         outputs['geometry_quality'][batch, queries]), dim=1)
    return torch.cat((query, encoder, geometry['logits'][batch, cells, None],
                      score[:, None].expand(-1, cells.shape[1], -1),
                      native_extent.log().expand_as(candidate_extent),
                      (candidate_extent / native_extent).log(),
                      (candidate[..., :2] - native[..., :2]) / native_extent), dim=-1).float()


@torch.no_grad()
def fit_examples(model, loader, post, known, plan, device):
    examples, preferred, eligible = [], [], []
    fit_count = 0
    random = torch.Generator(device=device).manual_seed(plan['seed'])
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to(device))
        final = outputs['pred_boxes']
        scored = post(outputs, final.new_ones((len(targets), 2)), include_top_indices=True)
        for batch, (target, row) in enumerate(zip(targets, scored, strict=True)):
            queries, cells = candidate_cells(outputs, batch, row, known)
            if not cells.numel():
                continue
            truth = box_cxcywh_to_xyxy(target['boxes'].to(device))
            native_iou = box_iou(box_cxcywh_to_xyxy(final[batch]), truth)[0]
            small = target['source_box_area'].to(device) < plan['small_area']
            missed = small & ~(native_iou > .5).any(dim=0)
            boxes = outputs['geometry']['boxes'][batch, cells]
            iou = box_iou(box_cxcywh_to_xyxy(boxes.flatten(0, 1)), truth)[0]
            iou = iou.reshape(*cells.shape, len(truth))
            moves = ((iou[..., missed] > .5).any(dim=2)
                     & ~(native_iou[queries] > .5).any(dim=1, keepdim=True))
            learned = moves.any(dim=1)
            keep = torch.where(~learned)[0]
            if len(keep):
                # Include one hard keep (strongest native hit) and one other sampled keep per image.
                best_native = native_iou[queries[keep]].amax(dim=1) if len(truth) else final.new_zeros(len(keep))
                hard = best_native.argmax()
                sample = torch.randperm(len(keep), device=device, generator=random)
                keep = torch.cat((keep[hard].reshape(1), keep[sample[sample != hard][:1]]))
            fit = torch.cat((torch.where(learned)[0], keep))
            padding = (0, 16 - cells.shape[1])
            examples.append(pad(features(outputs, batch, queries[fit], cells[fit], known), (0, 0, *padding)).cpu())
            preferred.append(pad(moves[fit], padding).cpu())
            eligible.append(pad(torch.ones_like(moves[fit]), padding).cpu())
        fit_count += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Fitting features {step}/{len(loader)} batches', flush=True)
    if fit_count != plan['fit_images'] or not examples:
        raise ValueError('Incomplete released local chooser fitting replay')
    return torch.cat(examples), torch.cat(preferred), torch.cat(eligible)


def selector_transform(params, post, known, device):
    mean, scale, weight = (torch.tensor(params[key], device=device) for key in ('mean', 'scale', 'weight'))

    def transform(outputs):
        final = outputs['pred_boxes']
        revised = final.clone()
        scored = post(outputs, final.new_ones((len(final), 2)), include_top_indices=True)
        for batch, row in enumerate(scored):
            queries, cells = candidate_cells(outputs, batch, row, known)
            if not cells.numel():
                continue
            values = (features(outputs, batch, queries, cells, known) - mean) / scale
            logits = values @ weight + params['bias']
            best, candidate = logits.max(dim=1)
            replace = best > 0
            chosen = cells[replace, candidate[replace]]
            revised[batch, queries[replace]] = outputs['geometry']['boxes'][batch, chosen]
        return revised

    return transform


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released dense local chooser')
    oracle = verified(ORACLE_OUTPUT / 'comparison.json', ORACLE_SHA)
    oracle_plan = verified(ORACLE_OUTPUT / 'plan.json', oracle['plan_sha256'])
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(PROTECTED_FIT_OUTPUT / 'plan.json', quarter['plan_sha256'])
    if oracle['status'] != 'complete' or not quarter['passed'] or any(
            plan[key] != oracle_plan[key] for key in ('jobs', 'seed', 'images', 'fit_order_sha256')):
        raise ValueError('Expected identical released fit and replay sources')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                oracle_sha256=ORACLE_SHA, quarter_fit_sha256=QUARTER_FIT_SHA,
                factor='Choose between native geometry and 16 nearest eligible fine-level boxes at unprotected emitted '
                       'unknown slots. Train grouped logits on frozen query, encoder cell, relative geometry and scores. '
                       'A positive preserves native hits by using a query with no original hit and reaches a small miss. '
                       'Include all positive query groups and two keep groups per image if available.',
                selector_negative_cost=NEGATIVE_COST, selector_l2=L2,
                advance='Use the locked joint released dense and known Top-50 gates on the disjoint replay.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
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
                raise ValueError(f'Changed fit or replay identities: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
            print(f'Fitting {protocol} local full-box selector', flush=True)
            params = fit_query_choice(*fit_examples(model, fit, post, known, plan, device))
            counts = Counter(restored_known_queries=0, reverted_images=0)
            transform = protect_known(selector_transform(params, post, known, device), post, known, counts)
            print(f'Evaluating {protocol} frozen local full-box selector', flush=True)
            report = measure(model, known, replay, plan, box_transform=transform)
            reference = quarter['protocols'][protocol]['groups']
            for name, group in report['groups'].items():
                if any(group[key] != reference[name][key] for key in ('gt', 'final', 'native_known_tp')):
                    raise ValueError(f'Changed native released baseline: {protocol}/{name}')
            groups = report['groups']
            small = groups['small']
            passed = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                      and all(group['revised_known_tp'] >= group['native_known_tp'] for group in groups.values()))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order,
                                  selector=params, protection=counts, passed=passed, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed dense local selector plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('selector', 'groups', 'changes', 'protection', 'passed')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
