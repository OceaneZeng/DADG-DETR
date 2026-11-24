"""Separate center and extent errors of learned slots on released Task 1 images."""

from collections import Counter
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import (
    EXTENT_OUTPUT, GEOMETRY_OUTPUT, prepare, refine_learned_boxes,
)
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = GEOMETRY_OUTPUT
PREFLIGHT_SHA = 'a3a7da2fdfb9cc4e92aebd7e2b6b5c3a0f3f88e614ebcc5f8b3196cdd1d24de5'
IOU_THRESHOLD = .5


def component_oracles(learned, truth):
    centers = torch.cat((truth[:, :2].expand_as(learned[:, :2]), learned[:, 2:]), dim=-1)
    extents = torch.cat((learned[:, :2], truth[:, 2:].expand_as(learned[:, 2:])), dim=-1)
    return [box_iou(box_cxcywh_to_xyxy(rows), box_cxcywh_to_xyxy(truth))[0].amax().item()
            for rows in (centers, extents)]


def neighbor_oracles(final, neighbors, truth):
    queries, peers, overlap = neighbors
    base = final[queries][:, None].expand_as(peers)
    allowed = overlap > IOU_THRESHOLD
    truth_xyxy = box_cxcywh_to_xyxy(truth)
    extent = torch.cat((base[..., :2], (base[..., 2:] * peers[..., 2:]).sqrt()), dim=-1)
    center = torch.cat((base[..., :2].lerp(peers[..., :2], .5), base[..., 2:]), dim=-1)
    result = {}
    for name, boxes in (('neighbor_extent', extent), ('neighbor_center', center)):
        score = box_iou(box_cxcywh_to_xyxy(boxes.flatten(0, 1)), truth_xyxy)[0].reshape_as(overlap)
        result[name] = score.max().item() if score.numel() else 0.
        result[f'gated_{name}'] = score.masked_fill(~allowed, 0.).max().item() if score.numel() else 0.
    return result


@torch.inference_mode()
def measure(model, loader, plan, reference):
    totals, misses, lost = Counter(), [], []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        active = outputs['geometry']['active']
        neighbors = {}

        def capture(batch, queries, peers, overlap):
            neighbors[batch] = (queries, peers, overlap)

        revised = refine_learned_boxes(outputs, observe_neighbors=capture)
        for batch, target in enumerate(targets):
            ids = torch.where(target['source_box_area'] < plan['small_area'])[0]
            truth = target['boxes'][ids].cuda()
            xyxy = box_cxcywh_to_xyxy(truth)
            native_iou = box_iou(box_cxcywh_to_xyxy(final[batch]), xyxy)[0].amax(dim=0)
            revised_iou = box_iou(box_cxcywh_to_xyxy(revised[batch]), xyxy)[0].amax(dim=0)
            covered, rescored = native_iou > IOU_THRESHOLD, revised_iou > IOU_THRESHOLD
            totals.update(gt=len(ids), final=int(covered.sum()), revised=int(rescored.sum()),
                          gained=int((~covered & rescored).sum()), lost=int((covered & ~rescored).sum()))
            learned = torch.ones(final.shape[1], dtype=torch.bool, device=final.device)
            learned[:active.shape[1]] = ~active[batch]
            learned_boxes = final[batch, learned]
            image_id = int(target['image_id'])
            for pos in torch.where(covered & ~rescored)[0].tolist():
                lost.append(dict(image=image_id, target_index=int(ids[pos]),
                                 native_iou=float(native_iou[pos]), revised_iou=float(revised_iou[pos])))
            for pos in torch.where(~covered)[0].tolist():
                gt = truth[pos:pos+1]
                center, extent = component_oracles(learned_boxes, gt)
                scores = dict.fromkeys(('neighbor_extent', 'gated_neighbor_extent',
                                        'neighbor_center', 'gated_neighbor_center'), 0.)
                if batch in neighbors:
                    scores = neighbor_oracles(final[batch], neighbors[batch], gt)
                misses.append(dict(image=image_id, target_index=int(ids[pos]), native_iou=float(native_iou[pos]),
                                   revised_iou=float(revised_iou[pos]), center=center, extent=extent, **scores))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images'] or dict(totals) != {key: reference[key] for key in totals}:
        raise ValueError('Released image coverage differs from the locked learned-extent preflight')
    if len(misses) != totals['gt'] - totals['final'] or len(lost) != totals['lost']:
        raise ValueError('Incomplete released target accounting')
    return misses, lost, totals


def summarize(misses):
    counts = Counter()
    for row in misses:
        center, extent = row['center'] > IOU_THRESHOLD, row['extent'] > IOU_THRESHOLD
        counts.update(center_only=center and not extent, extent_only=extent and not center,
                      both=center and extent, neither=not (center or extent))
        for component in ('center', 'extent'):
            possible = row[component] > IOU_THRESHOLD
            isolated = possible and row['extent' if component == 'center' else 'center'] <= IOU_THRESHOLD
            for gate in ('neighbor', 'gated_neighbor'):
                name = f'{gate}_{component}'
                hit = row[name] > IOU_THRESHOLD
                counts.update({name: hit, f'{name}_on_{component}_only': hit and isolated})
    return counts


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    preflight = verified(EXTENT_OUTPUT / 'comparison.json', PREFLIGHT_SHA)
    if preflight['status'] != 'complete' or preflight['passed']:
        raise ValueError('Expected the stopped released geometry preflight')
    source_plan = verified(EXTENT_OUTPUT / 'plan.json', preflight['plan_sha256'])
    plan = prepare()
    inputs = ('jobs', 'seed', 'images', 'batch_size', 'limit', 'small_area')
    if any(plan[key] != source_plan[key] for key in inputs):
        raise ValueError('Changed released training sources')
    plan.update(extent_preflight_sha256=PREFLIGHT_SHA,
                factor='On native small misses, substitute only the reference center or only the reference extent '
                       'on each learned slot. Compare these diagnostics with the best same-component geometric '
                       'blend attainable by the same 16 nearby fine encoder candidates, with and without their native gate.',
                advance='Prefer a component supported by attainable recovery on the released training sources. '
                        'Report target-oracle coverage separately from deployable results.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count checkpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            print(f'Evaluating {protocol} released geometry components', flush=True)
            reference = preflight['protocols'][protocol]['groups']['small']
            misses, lost, totals = measure(model, loader, plan, reference)
            summary = dict(coverage=totals, recovery=summarize(misses))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, summary=summary, misses=misses, lost=lost))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen replay plan: {dest}')
        protocols[protocol] = report['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
