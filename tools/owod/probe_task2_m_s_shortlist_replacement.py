"""Test whether a released-class proposal survives decoding at the original query budget."""

import argparse
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.probe_task1_m_s_prescreen_replacement import replace_cells
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_previous_competitors import (
    DEVICE, IOU, SEED, SHORTLIST_OUTPUT, TOP50, TRAINING, TRAIN_PLAN_SHA, training_samples,
)
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import collate_fn


OUTPUT = TRAINING.with_name('task2-m-s-current-shortlist-replacement')
TRACE_OUTPUT = TRAINING.with_name('task2-m-s-current-replacement-trace')
REPLACEMENT_SHA = '6f5b5f906ca552da0bd75cc9e0f674eff76657b93bdd88d26b180f2ea42a8c6b'
REPLACEMENT_REPORTS = {
    'm-owodb': '899abef2c35a41bbabf032d056dbe3694ee2e58326173766e454b1719ed68100',
    's-owodb': '16a87b239fb78fce4922ffa1741de82bdcca5bf952817a2ce0a9a997125ff60a',
}
SOURCE_SHA = '84fae79e8dc058dc6786e415121f3cade6b06ea7c975a1ccb63c190908f1d66d'
REPORTS = {
    'm-owodb': '768624bd356fc0d7c0a0c7c6456422eff237a3b7e6cc2598c00e451f11e5a777',
    's-owodb': 'b7cfa08b75eb40413e8e19ba81a251b823690140b6d8833a83d39442c5e26f77',
}
MIN_FINAL = .25
MIN_TOP50 = .10


def top50_covered(output, index, target, post):
    scores, labels, _ = post.candidate_scores(output)
    flat = scores[index].flatten().topk(TOP50).indices
    slots = flat // len(labels)
    emitted = labels[flat % len(labels)]
    boxes = box_cxcywh_to_xyxy(output['pred_boxes'][index])
    iou, _ = box_iou(boxes, box_cxcywh_to_xyxy(target['boxes']))
    return iou, ((iou[slots] > IOU) & (emitted[:, None] == target['labels'])).any(0)


def trace_query(output, image, cell, target):
    geo = output['geometry']
    slots = torch.where((geo['chosen'][image] == cell) & geo['active'][image])[0]
    if slots.numel() != 1 or len(output['aux_outputs']) != 5:
        raise ValueError('Expected one injected query and six decoder layers')
    slot = int(slots[0])
    layers = torch.stack([layer['pred_boxes'][image, slot] for layer in (*output['aux_outputs'], output)])
    encoder = geo['boxes'][image, cell]
    first = layers[0]
    variants = torch.stack((torch.cat((encoder[:2], first[2:])), torch.cat((first[:2], encoder[2:]))))
    truth = box_cxcywh_to_xyxy(target[None])
    layer_iou, _ = box_iou(box_cxcywh_to_xyxy(layers), truth)
    variant_iou, _ = box_iou(box_cxcywh_to_xyxy(variants), truth)
    return dict(slot=slot, layer_iou=layer_iou.flatten().tolist(),
                encoder_center_iou=float(variant_iou[0, 0]), encoder_extent_iou=float(variant_iou[1, 0]))


@torch.inference_mode()
def replay(job, dataset, reference, trace=False):
    model, _ = load_detector(None, Path(job['checkpoint']['path']), DEVICE)
    model.eval()
    proposal = model.detr.transformer.instance_proposals
    post = DiscoveryPostProcess(known_ids=job['previous'] + job['current'], scoring='focal', max_detections=TOP50)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(SEED))
    rows, images = [], []
    for step, (samples, targets) in enumerate(loader, 1):
        names = [int(target['image_id']) for target in targets]
        images.extend(names)
        if any(reference[name] for name in names):
            samples = samples.to(DEVICE)
            targets = move_targets(targets, DEVICE)
            native = model(samples)
            geo = native['geometry']
            cells, offered, checks = {}, {}, {}
            for i, (image, target) in enumerate(zip(names, targets, strict=True)):
                misses = reference[image]
                if not misses:
                    continue
                indices = [r['target'] for r in misses]
                iou, hits = top50_covered(native, i, target, post)
                if (target['labels'][indices].tolist() != [r['category'] for r in misses]
                        or bool((iou[:, indices] > IOU).any())):
                    raise ValueError(f'Changed frozen targets: {image}')
                checks[i] = hits
                active = geo['chosen'][i, geo['active'][i]]
                if not len(active):
                    continue
                valid = torch.where(geo['valid'][i])[0]
                available = valid[~torch.isin(valid, active)]
                geometry_iou, _ = box_iou(box_cxcywh_to_xyxy(geo['boxes'][i, available]),
                                          box_cxcywh_to_xyxy(target['boxes'][indices]))
                best_iou, position = geometry_iou.max(dim=0)
                if not bool((best_iou > IOU).all()):
                    raise ValueError(f'Lost a compatible dense box: {image}')
                best_target = int(best_iou.argmax())
                cells[i] = available[position[best_target]].reshape(1)
                offered[i] = (indices[best_target], float(best_iou[best_target]))
            changed = native
            if cells:
                with proposal.register_forward_hook(lambda m, x, y: replace_cells(m, x, y, cells)):
                    changed = model(samples)
                if not torch.equal(changed['geometry']['active'], geo['active']):
                    raise ValueError('Changed native encoder quota')
            for i, before in checks.items():
                image, target = names[i], targets[i]
                iou, after = top50_covered(changed, i, target, post)
                old = torch.isin(target['labels'], target['labels'].new_tensor(job['previous']))
                selected = offered.get(i)
                if selected is not None:
                    actual = changed['geometry']['chosen'][i, geo['active'][i]]
                    if not bool(torch.isin(cells[i], actual).all()):
                        raise ValueError(f'Chosen cell did not reach the decoder: {image}')
                for rank_row in reference[image]:
                    gt = rank_row['target']
                    injected = selected is not None and selected[0] == gt
                    row = dict(image=image, target=gt, category=rank_row['category'],
                               injected=injected, injected_iou=selected[1] if injected else None,
                               final_covered=bool((iou[:, gt] > IOU).any()), in_top50=bool(after[gt]),
                               old_top50_lost=int((old & before & ~after).sum()) if injected else 0)
                    if trace and injected:
                        row['trace'] = trace_query(changed, i, int(cells[i][0]), target['boxes'][gt])
                    rows.append(row)
        if step % 32 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    if images != job['image_ids'] or len(rows) != sum(map(len, reference.values())):
        raise ValueError('Changed registered sample or shortlist target population')
    return rows


def main(trace=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    previous = verified(SHORTLIST_OUTPUT / 'comparison.json', SOURCE_SHA)
    source_plan = verified(SHORTLIST_OUTPUT / 'plan.json', previous['plan_sha256'])
    jobs, datasets = training_samples(verified(TRAINING / 'plan.json', TRAIN_PLAN_SHA))
    if jobs != source_plan['jobs'] or previous['depth_headroom']:
        raise ValueError('Expected the same fixed training sample and the closed shortlist-depth candidate')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256=SOURCE_SHA, source_reports=REPORTS, jobs=jobs, seed=SEED, batch=2,
                per_image_replacements=1, native_top=TOP50, iou=IOU, paper_comparable=False,
                minimum_final_fraction=MIN_FINAL, minimum_top50_fraction=MIN_TOP50)
    output = OUTPUT
    if trace:
        output = TRACE_OUTPUT
        replacement = verified(OUTPUT / 'comparison.json', REPLACEMENT_SHA)
        if replacement['headroom_supported']:
            raise ValueError('Expected the completed fixed-budget replacement')
        plan.update(replacement_sha256=REPLACEMENT_SHA, replacement_report_shas=REPLACEMENT_REPORTS,
                    decision='First layer is a shared bottleneck if the injected query loses IoU coverage there for half of each protocol.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    for protocol, job in jobs.items():
        marker = output / f'{protocol}.json'
        if not marker.exists():
            frozen = verified(SHORTLIST_OUTPUT / marker.name, REPORTS[protocol])
            reference = {row['image']: row['shortlist_misses'] for row in frozen['rows']}
            print(f'Evaluating {protocol} fixed-count shortlist replacement', flush=True)
            rows = replay(job, datasets[protocol], reference, trace=trace)
            injected = [row for row in rows if row['injected']]
            counts = dict(targets=len(rows), injected=len(injected),
                          final_covered=sum(row['final_covered'] for row in injected),
                          in_top50=sum(row['in_top50'] for row in injected),
                          old_top50_lost=sum(row['old_top50_lost'] for row in injected))
            if trace:
                parent = verified(OUTPUT / marker.name, REPLACEMENT_REPORTS[protocol])
                if [{k: v for k, v in row.items() if k != 'trace'} for row in rows] != parent['rows']:
                    raise ValueError(f'Fixed-budget replacement did not reproduce: {protocol}')
                traces = [row['trace'] for row in injected]
                lost_first = [item for item in traces if item['layer_iou'][0] <= IOU]
                counts.update(first_layer_misses=len(lost_first),
                              covered_by_layer=[sum(t['layer_iou'][i] > IOU for t in traces) for i in range(6)],
                              encoder_center_recovers=sum(t['encoder_center_iou'] > IOU for t in lost_first),
                              encoder_extent_recovers=sum(t['encoder_extent_iou'] > IOU for t in lost_first))
            write_json(marker, dict(plan_sha256=digest, rows=rows, counts=counts))
        report = read_json(marker)
        if report['plan_sha256'] != digest or report['counts']['targets'] != previous['protocols'][protocol]['shortlist_misses']:
            raise ValueError(f'Unexpected replay report: {protocol}')
        protocols[protocol] = report['counts']
    supported = all(row['injected'] and row['final_covered'] >= MIN_FINAL * row['injected']
                    and row['in_top50'] >= MIN_TOP50 * row['injected'] for row in protocols.values())
    result = dict(status='complete', plan_sha256=digest, protocols=protocols, headroom_supported=supported)
    if trace:
        result['first_layer_majority'] = all(row['first_layer_misses'] * 2 >= row['injected']
                                            for row in protocols.values())
    write_json(output / 'comparison.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-injected', action='store_true')
    main(parser.parse_args().trace_injected)
