"""Reserve one encoder slot for a small proposal using inference-time signals only."""
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_unknown_factors import best_target
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.run_s_task1_proposal_stages import OUTPUT as TRACE
from tools.owod.run_s_task1_released_seed import OUTPUT as RELEASED, replace
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = PARENT.with_name('s-instance-task1-small-slot')
RANKERS = ('quality', 'agreement')


def neighbor_agreement(boxes, masks, i):
    """Mean regressed-box IoU with valid horizontal and vertical grid neighbors."""
    parts = []
    offset = 0
    for mask in masks:
        valid = ~mask[i]
        height, width = valid.shape
        end = offset + height * width
        grid = box_cxcywh_to_xyxy(boxes[offset:end]).reshape(height, width, 4)
        area = (grid[..., 2:] - grid[..., :2]).prod(-1)
        total, count = torch.zeros_like(area), torch.zeros_like(area)
        for axis in (0, 1):
            left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            a, b = grid[left], grid[right]
            inter = (torch.minimum(a[..., 2:], b[..., 2:]) -
                     torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
            joined = valid[left] & valid[right]
            value = torch.where(joined, inter / (area[left] + area[right] - inter).clamp_min(1e-12), 0)
            for side in (left, right):
                total[side] += value
                count[side] += joined
        parts.append((total / count.clamp_min(1)).flatten())
        offset = end
    return torch.cat(parts)


def rank_cells(geo, i, size):
    boxes, logits = geo['boxes'][i], geo['logits'][i]
    live = torch.where(geo['active'][i])[0]
    empty = live[:0]
    if not len(live):
        return None, {name: empty for name in RANKERS}
    chosen = geo['chosen'][i, live]
    # Predicted area <= 2 * 32^2 is necessary for IoU .5 with a small box.
    area = boxes[:, 2:].prod(-1) * size.prod()
    valid = geo['valid'][i] & torch.isfinite(logits) & (area > 0) & (area <= 2 * 32**2)
    valid[chosen] = False
    ids = torch.where(valid)[0]
    if len(ids):
        ids = ids[overlap(boxes[ids], boxes[chosen]).amax(1) < .5]
    if not len(ids):
        return None, {name: ids for name in RANKERS}
    agreement = neighbor_agreement(boxes, geo['masks'], i)
    scores = dict(quality=logits[ids], agreement=logits[ids].sigmoid() * agreement[ids])
    ordered = {name: ids[torch.argsort(score, descending=True, stable=True)] for name, score in scores.items()}
    choice = dict(slot=int(live[-1]), donor=int(chosen[-1]), **{a: int(v[0]) for a, v in ordered.items()})
    return choice, ordered


def measure(out, i, det, truth, anns, known, proxy):
    queries = det['top_indices'] // (len(known) + 1)
    unknown = det['labels'] == 91
    record = dict(unknown_outputs=int(unknown.sum()))
    matched = {}
    for group, classes, use in [('known', known, ~unknown), ('proxy', proxy, unknown)]:
        idx = [j for j, a in enumerate(anns) if a['category_id'] in classes]
        assign = best_target(out['pred_boxes'][i], truth[idx])
        matched[group] = assign[queries] >= 0
        selected = assign[queries[use]].tolist()
        hits = {anns[idx[j]]['id'] for j, label in zip(selected, det['labels'][use].tolist())
                if j >= 0 and (group == 'proxy' or anns[idx[j]]['category_id'] == label)}
        record[group] = sorted(hits)
    record['unknown_only_known'] = int((unknown & matched['known'] & ~matched['proxy']).sum())
    return record


def first_cover(boxes, orders, truth, anns):
    """Offline ranks of the first IoU-.5 proposal per small GT, zero when absent."""
    small = [j for j, a in enumerate(anns) if a['area'] < 32**2]
    result = {}
    for arm, ids in orders.items():
        ranks = [0] * len(small)
        if len(ids) and small:
            edges = overlap(boxes[ids], truth[small]) >= .5
            ranks = ((edges.to(torch.uint8).argmax(0) + 1) * edges.any(0)).tolist()
        result[arm] = {anns[j]['id']: r for j, r in zip(small, ranks)}
    return result


@torch.inference_mode()
def evaluate(label, model, post, loader, known, proxy, reference):
    totals = {arm: Counter() for arm in ('native', *RANKERS)}
    images = {}
    module = model.detr.transformer.instance_proposals
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to('cuda')
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = model(samples)
        geo = native['geometry']
        plans = [rank_cells(geo, i, size) for i, size in enumerate(sizes)]
        choices = [c for c, _ in plans]
        rows = []
        for i, target in enumerate(targets):
            key = str(target['image_id'].item())
            anns = valid_annotations(loader.dataset, int(key))
            if [a['category_id'] for a in anns] != target['labels'].tolist():
                raise ValueError(f'Unexpected annotation order on {key}')
            ids = [a['id'] for a in anns]
            ref = reference[key]
            if ids != ref['annotation_ids']:
                raise ValueError(f'Unpaired ground truth on {key}')
            eligible = ref.get('choice')
            truth = target['boxes'].cuda()
            orders = plans[i][1]
            row = dict(choice=choices[i], pool=len(orders['quality']), eligible_id=eligible['annotation_id'] if eligible else None,
                       first_cover=first_cover(geo['boxes'][i], orders, truth, anns), outcomes={})
            images[key] = row
            rows.append((row, truth, anns, ref))
        for arm in totals:
            out = native
            if arm != 'native' and any(choices):
                hook = partial(replace, choices=choices, geometry=arm, content=arm)
                with module.register_forward_hook(hook):
                    out = model(samples)
            detections = post(out, sizes, include_top_indices=True)
            for i, (row, truth, anns, ref) in enumerate(rows):
                record = measure(out, i, detections[i], truth, anns, known, proxy)
                row['outcomes'][arm] = record
                if arm == 'native':
                    if label == 'proxy' and set(record['known'] + record['proxy']) != set(ref['emitted_best']):
                        raise ValueError('Changed native proxy-slice predictions')
                    if row['eligible_id'] is not None:
                        expected = {ref['annotation_ids'][j] for j in ref['outcomes']['native']['correct_ids']}
                        if set(record['known']) != expected:
                            raise ValueError('Changed native released-slice predictions')
                small = row['first_cover']['quality']
                totals[arm].update(images=1, eligible_recovered=int(row['eligible_id'] in record['known']),
                                   known=len(record['known']), proxy=len(record['proxy']),
                                   small_proxy=sum(j in small for j in record['proxy']),
                                   unknown_outputs=record['unknown_outputs'], unknown_only_known=record['unknown_only_known'])
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'{datetime.now().isoformat(timespec="seconds")} {label} {step}/{len(loader)} batches', flush=True)
    if images.keys() != reference.keys():
        raise ValueError('Incomplete paired slice')
    return images, totals


def main():
    base = read_json(RELEASED / 'plan.json')
    prior = read_json(RELEASED / 'comparison.json')
    trace = read_json(TRACE / 'plan.json')
    data = dict(released=RELEASED / 'released2048.json', proxy=PARENT / 'evaluation/fresh1024.json')
    checkpoint = Path(base['job']['parent'])
    refs = dict(released=RELEASED / 'paired.json', proxy=TRACE / 'stage-trace.json')
    if (prior['status'] != 'complete' or not prior['follow_up'] or prior['plan_sha256'] != file_sha256(RELEASED / 'plan.json')
            or trace['job'] != base['job'] or trace['manifest_sha256'] != base['manifest_sha256']
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(checkpoint) != base['job']['parent_sha256']
            or file_sha256(data['released']) != base['data_sha256'] or file_sha256(data['proxy']) != trace['fresh_sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed parent diagnostic, data, weights or environment')
    plan = {k: base[k] for k in ('seed', 'protocol', 'job', 'gpu', 'torch_version', 'batch_size', 'manifest_sha256', 'paper_comparable')}
    parents = dict(released=prior['plan_sha256'], proxy=file_sha256(TRACE / 'plan.json'))
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), devices=os.environ['CUDA_VISIBLE_DEVICES'],
                parents=parents, reference_sha256={k: file_sha256(v) for k, v in refs.items()},
                data_sha256={k: file_sha256(v) for k, v in data.items()},
                rule='One last live encoder slot replaced by same-cell geometry and memory. Valid non-selected cell, '
                     'area <= 2048 original pixels squared, IoU < .5 with all active proposals. Compare raw quality vs '
                     'sigmoid quality times mean regressed-box IoU with valid four-neighbors. Stable flat-index ties. '
                     'No labels used for ranking. Record posthoc first-cover ranks for each small GT.',
                criterion='Independently for either arm: recover >=11 of the previously designated released misses; '
                          'maintain total correct known hits on each slice and total emitted proxy hits; strictly increase '
                          'small emitted proxy hits. Passing warrants independent held-out verification only.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    totals = {}
    for label, path in data.items():
        ref = read_json(refs[label])
        if ref['plan_sha256'] != parents[label]:
            raise ValueError(f'Unpaired reference for {label}')
        _, loader = make_loader(resolve_image_root(MANIFEST, manifest), path, plan['batch_size'], 2)
        images, totals[label] = evaluate(label, model, post, loader, known, proxy, ref['images'])
        write_json(OUTPUT / f'{label}.json', dict(plan_sha256=digest, images=images))
    r, p = totals['released'], totals['proxy']
    passing = [a for a in RANKERS if r[a]['eligible_recovered'] >= 11 and r[a]['known'] >= r['native']['known']
               and p[a]['known'] >= p['native']['known'] and p[a]['proxy'] >= p['native']['proxy']
               and p[a]['small_proxy'] > p['native']['small_proxy']]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals,
                                                passing=passing, follow_up=bool(passing), paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
