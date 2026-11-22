"""Measure signed width and height errors on the unchanged native neighbors."""

from collections import Counter
from itertools import accumulate
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.instance_proposals import nearby_anchor_ids
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_small_box_geometry import OUTPUT as SOURCE
from tools.owod.probe_task1_paired_discovery import HOLDOUT, SCREENS, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, resolve_manifest_path
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.screening_data import select_images, source_ids, write_json


OUTPUT = SOURCE.with_name('task1-m-s-small-box-extent')
SOURCE_FILES = {
    'plan.json': 'c48a38144a0589d9e856083922948d32fc7506ea4e89c6f8c315051278b92c3a',
    'comparison.json': 'e8f9825608d08e0ba4df69ef2c5f51ba03af28bfdc6672e815cdca3466bc48b7',
    'first-annotations.json': '96fd21ff56bf12d91b303686c78103df6c12765e1c25cbfca5c08d68002e4c76',
    'second-annotations.json': '1ec01c36d8f124157489e723f8e77faa1f00646ffa14d313ff132cb989587215',
    'm-owodb-first.json': 'b3fae08ef589716f509a32c146b6cf2231a2f6a7c55e549e67219b6d29bb9425',
    'm-owodb-second.json': '1b00367b59950834aa62f52df7c8ac2434898ef29d7951973dcae2152d787b03',
    's-owodb-first.json': '33d7577fa3a5b06553a1b52e7362c9da46dbfe38bdb21b23239ce5a434cf8451',
    's-owodb-second.json': 'e4364aeebd56bc18ed2e10abcb95e75f79b074a0c3a82d8de56df76ac522ad4b',
}
SEED = 44
CONTROL_IMAGES = 192
DIRECTION_TOLERANCE = float(np.log(1.05))
FIXED_SCALE_SHARE = .70
MIN_STRATUM = 20


def released_controls(source, parent):
    protocols = parent['parents']
    manifests, roots, released = {}, {}, {}
    blocked = set().union(*(source_ids(source[f'{s}-annotations.json']) for s in parent['cohorts']))
    for path, digest in HOLDOUT.items():
        blocked.update(source_ids(verified(path, digest)))
    for protocol, weights in protocols.items():
        manifest_path = Path('data/owod', protocol, 'split_manifest.json').resolve()
        manifest = verified(manifest_path, weights['manifest_sha256'])
        manifests[protocol] = manifest
        roots[protocol] = resolve_image_root(manifest_path, manifest)
        blocked.update(manifest['evaluation']['image_source_ids'])
        screen_path = SCREENS[protocol]
        screen = verified(screen_path, weights['screen_sha256'])
        for stage, keys in ((0, ('increment_train', 'full_val')), (1, ('increment_train',))):
            rows = screen['stages'][stage]
            for key in keys:
                blocked.update(source_ids(verified(screen_path.parent / rows['files'][key], rows['sha256'][key])))
        phase = manifest['stages'][0]
        path = resolve_manifest_path(manifest_path, phase['files']['increment_train'])
        released[protocol] = verified(path, phase['sha256']['increment_train'])

    common = set.intersection(*(set(manifest['stages'][0]['classes']) for manifest in manifests.values()))
    controls = {}
    for protocol, data in released.items():
        images = {i['id']: i for i in data['images'] if i['owod_source_id'] not in blocked}
        small = []
        for ann in data['annotations']:
            if ann['image_id'] not in images or ann['category_id'] not in common:
                continue
            image = images[ann['image_id']]
            x, y, w, h = ann['bbox']
            if (not ann.get('difficult') and not ann.get('iscrowd') and w * h < 32**2
                    and min(x + w, image['width']) > max(x, 0) and min(y + h, image['height']) > max(y, 0)):
                small.append(ann)
        eligible = {ann['image_id'] for ann in small}
        pool = dict(data, images=[images[i] for i in sorted(eligible)], annotations=small)
        classes = sorted({ann['category_id'] for ann in small})
        names = select_images(pool, min(CONTROL_IMAGES, len(eligible)), classes, SEED)
        selected = [i for i in pool['images'] if i['owod_source_id'] in names]
        ids = {i['id'] for i in selected}
        controls[protocol] = dict(pool, images=selected, annotations=[a for a in small if a['image_id'] in ids])
    return controls, roots


def prepare():
    source = {name: verified(SOURCE / name, digest) for name, digest in SOURCE_FILES.items()}
    parent = source['plan.json']
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not device.isdecimal() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != parent['gpu'] or str(torch.__version__) != parent['torch_version']):
        raise ValueError('Changed frozen replay environment')
    comparison = source['comparison.json']
    if (comparison['status'], comparison['shared_priority'], comparison['plan_sha256']) != (
            'complete', 'extent', SOURCE_FILES['plan.json']):
        raise ValueError('Frozen replay did not establish an extent priority')
    for weights in parent['parents'].values():
        checkpoint = weights['checkpoint']
        if file_sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError(f'Changed frozen checkpoint: {checkpoint["path"]}')
    controls, roots = released_controls(source, parent)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol='m-s-paired', paper_comparable=False, parents=parent['parents'],
                source_files=SOURCE_FILES, device=device, gpu=parent['gpu'], torch_version=parent['torch_version'],
                batch_size=parent['batch_size'], cohorts=parent['cohorts'], control_seed=SEED,
                released_controls={p: dict(images=len(data['images']), targets=len(data['annotations']),
                                           classes=sorted({a['category_id'] for a in data['annotations']}))
                                   for p, data in controls.items()},
                direction_tolerance=DIRECTION_TOLERANCE, fixed_scale_share=FIXED_SCALE_SHARE,
                minimum_stratum=MIN_STRATUM)
    pin(OUTPUT / 'plan.json', plan)
    for protocol, data in controls.items():
        path = OUTPUT / f'{protocol}-released-annotations.json'
        if not path.exists():
            write_json(path, data)
        elif read_json(path) != data:
            raise ValueError(f'Changed released controls: {protocol}')
    return plan, roots, source


def candidate_ious(gt, boxes):
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) / 2
    gt_center, gt_half = (gt[:2] + gt[2:]) / 2, (gt[2:] - gt[:2]) / 2
    center_only = np.concatenate((gt_center - half, gt_center + half), axis=1)
    extent_only = np.concatenate((center - gt_half, center + gt_half), axis=1)
    return overlaps(gt, center_only), overlaps(gt, extent_only)


@torch.inference_mode()
def trace(model, loader, wanted, previous=None):
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        geo = model(samples.to('cuda'))['geometry']
        boundaries = list(accumulate(mask.shape[-2] * mask.shape[-1] for mask in geo['masks']))
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            annotations = valid_annotations(loader.dataset, image)
            if target['labels'].tolist() != [a['category_id'] for a in annotations]:
                raise ValueError(f'Annotation order mismatch on image {image}')
            indices = [j for j, ann in enumerate(annotations) if ann['id'] in wanted]
            if not indices:
                continue
            ids = torch.where(geo['valid'][i])[0]
            gt_boxes = target['boxes'][indices].to(ids.device)
            near = nearby_anchor_ids(gt_boxes, geo['anchors'][i], ids)
            if near.shape[1] != 16:
                raise ValueError(f'Expected 16 valid neighbors on image {image}')
            lookup = torch.searchsorted(ids, near).cpu().numpy()
            flat = near.cpu().numpy()
            levels = np.searchsorted(boundaries, flat, side='right')
            height, width = target['orig_size'].tolist()
            dense = voc_coords(geo['boxes'][i, ids], width, height)
            gt_wh = gt_boxes[:, None, 2:]
            anchor_error = (geo['anchors'][i, near, 2:] / gt_wh).log().cpu().numpy()
            dense_error = (geo['boxes'][i, near, 2:] / gt_wh).log().cpu().numpy()
            for k, j in enumerate(indices):
                ann = annotations[j]
                gt = np.asarray(ann['voc_bbox'])
                boxes = dense[lookup[k]]
                native = float(overlaps(gt, dense).max())
                center, extent = candidate_ious(gt, boxes)
                association = (native, float(center.max()), float(extent.max()))
                if previous is not None:
                    reference = previous[ann['id']]
                    old = (reference['native_iou'], reference['center_iou'], reference['extent_iou'])
                    if reference['category'] != ann['category_id'] or not np.allclose(association, old, atol=1e-8, rtol=0):
                        raise ValueError(f'Changed replay geometry for target {ann["id"]}')
                neighbors = [dict(cell=int(flat[k, idx]), level=int(levels[k, idx]),
                                  anchor_log_wh=anchor_error[k, idx].tolist(), dense_log_wh=dense_error[k, idx].tolist(),
                                  native_iou=float(iou), center_iou=float(center[idx]), extent_iou=float(extent[idx]))
                             for idx, iou in enumerate(overlaps(gt, boxes))]
                rows.append(dict(annotation_id=ann['id'], category=ann['category_id'], image_id=image,
                                 area=float(ann['bbox'][2] * ann['bbox'][3]), image_hw=[height, width],
                                 native_iou=native, neighbors=neighbors))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(wanted) or {row['annotation_id'] for row in rows} != wanted:
        raise ValueError('Incomplete target replay')
    return rows


def log_summary(values):
    if not len(values):
        return dict(n=0, directions={}, quantiles=None)
    errors = np.asarray(values)
    signs = (errors > DIRECTION_TOLERANCE).astype(int) - (errors < -DIRECTION_TOLERANCE).astype(int)
    symbol = {-1: 'under', 0: 'close', 1: 'over'}
    counts = Counter(f'{symbol[w]}/{symbol[h]}' for w, h in signs)
    return dict(n=len(values), directions=dict(counts), quantiles=np.quantile(errors, [.1, .5, .9], axis=0).tolist())


def summarize(rows):
    eligible = [(row, [n for n in row['neighbors'] if n['extent_iou'] > .5]) for row in rows]
    eligible = [(row, neighbors) for row, neighbors in eligible if neighbors]
    best_extent = [(row, max(neighbors, key=lambda n: n['center_iou'])) for row, neighbors in eligible]
    nearest = [(row, neighbors[0]) for row, neighbors in eligible]

    def pairs(selected):
        return dict(anchor=log_summary([n['anchor_log_wh'] for _, n in selected]),
                    dense=log_summary([n['dense_log_wh'] for _, n in selected]))

    return dict(targets=len(rows), native_covered=sum(row['native_iou'] > .5 for row in rows),
                recoverable=len(eligible), nearest_recoverable=pairs(nearest), best_extent=pairs(best_extent),
                native_nearest=pairs([(row, row['neighbors'][0]) for row in rows]),
                strata={name: {label: pairs([(r, n) for r, n in best_extent if low <= value(r, n) < high])
                               for label, low, high in bins}
                        for name, value, bins in (
                            ('area', lambda r, _: r['area'], (('under_16_sq', 0, 16**2), ('16_to_32_sq', 16**2, 32**2))),
                            ('resolution', lambda r, _: max(r['image_hw']),
                             (('under_640', 0, 640), ('640_to_1023', 640, 1024), ('1024_plus', 1024, float('inf')))),
                            ('level', lambda _, n: n['level'], ((str(i), i, i + 1) for i in range(4))))},
                all_recoverable_neighbors_by_level={str(level): pairs([
                    (row, n) for row, neighbors in eligible for n in neighbors if n['level'] == level
                ]) for level in range(4)})


def fixed_scale_direction(arms):
    for direction in ('over/over', 'under/under'):
        if all(arm['recoverable'] and arm['best_extent']['dense']['directions'].get(direction, 0)
               >= FIXED_SCALE_SHARE * arm['recoverable']
               and all(group['dense']['n'] < MIN_STRATUM or group['dense']['directions'].get(direction, 0)
                       > .5 * group['dense']['n'] for groups in arm['strata'].values() for group in groups.values())
               for arm in arms.values()):
            return direction
    return None


def main():
    plan, roots, source = prepare()
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    arms, controls = {}, {}
    for protocol, parent in plan['parents'].items():
        model, _ = load_detector(None, Path(parent['checkpoint']['path']), torch.device('cuda'))
        model.eval()
        for name, cohort in plan['cohorts'].items():
            key = f'{protocol}-{name}'
            previous = {row['annotation_id']: row for row in source[f'{key}.json']['rows']}
            _, loader = make_loader(roots['m-owodb'], SOURCE / f'{name}-annotations.json', plan['batch_size'], 2)
            print(f'Evaluating {key}: {cohort["images"]} identical replay images', flush=True)
            rows = trace(model, loader, set(cohort['target_ids']), previous)
            write_json(OUTPUT / f'{key}.json', dict(plan_sha256=plan_hash, rows=rows))
            arms[key] = summarize(rows)
        key = f'{protocol}-released'
        dataset, loader = make_loader(roots[protocol], OUTPUT / f'{key}-annotations.json', plan['batch_size'], 2)
        print(f'Evaluating {key}: {len(dataset)} independent control images', flush=True)
        rows = trace(model, loader, set(dataset.coco.anns))
        write_json(OUTPUT / f'{key}.json', dict(plan_sha256=plan_hash, rows=rows))
        controls[key] = summarize(rows)
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
               arms=arms, released_controls=controls, fixed_scale_direction=fixed_scale_direction(arms)))
    print('Paired small-object width and height comparison saved', flush=True)


if __name__ == '__main__':
    main()
