"""Separate center and extent errors on paired small-object dense misses."""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.instance_proposals import nearby_anchor_ids
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import OUTPUT as SOURCE, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.screening_data import subset, write_json


OUTPUT = SOURCE.with_name('task1-m-s-small-box-geometry-replay')
SOURCE_FILES = json.loads("{\"comparison.json\":\"1ca1175febb2e53cf6206024b74ebe817f33944123e2daaa5753dbba9da4eb06\",\"first-annotations.json\":\"640756b275954c84a8f89121166ded54a553f1a2055165800e0abce776374530\",\"second-annotations.json\":\"8bc8abd1483991915ba8aa78437724db2bb21b605f5bc02ae3baf9315dbc1b41\",\"m-owodb-first.json\":\"d01f3b6e90b84131ebef2b19e91eb29d25f7414b9208d239807e8576875a9946\",\"m-owodb-second.json\":\"05cf4b69309d9e7a74d23a9c73d5e06f44ac0aa811ba7fcf3f884a8a7d5112ac\",\"s-owodb-first.json\":\"97365f9a0cb023b69bd71e1dbb10f98469f7a4acd1ab8abcf00d8cb0b3012920\",\"s-owodb-second.json\":\"3060243c463c235cee9135c23212accf34d8ef56ce2696db0109c9691f6c0a42\"}")
EXPECTED = {'first': 151, 'second': 434}
CRITERION = (
    'On the same small-object misses, prefer center correction or extent correction only if one '
    'exceeds the other by at least 10 pp in both slices and both protocols. Use the unchanged 16 '
    'native nearest anchors and strict VOC IoU > .5. Replacement against GT is diagnostic only.'
)


def prepare():
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    source_plan = read_json(SOURCE / 'plan.json')
    source_sha = file_sha256(SOURCE / 'plan.json')
    source = {name: verified(SOURCE / name, digest) for name, digest in SOURCE_FILES.items()}
    cn = source['comparison.json']
    manifest_path = Path('data/owod/m-owodb/split_manifest.json')
    manifest = verified(manifest_path, source_plan['parents']['m-owodb']['manifest_sha256'])
    if (cn['plan_sha256'] != source_sha or cn['status'] != 'complete' or cn['shared_priority'] is not None
            or not device.isdecimal() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source_plan['gpu'] or str(torch.__version__) != source_plan['torch_version']):
        raise ValueError('Changed shared discovery evidence or replay environment')
    for parent in source_plan['parents'].values():
        entry = parent['checkpoint']
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Changed frozen Task 1 checkpoint')
    cohorts, assets = {}, {}
    for name, expected in EXPECTED.items():
        arm = [source[f'{p}-{name}.json'] for p in source_plan['parents']]
        if any(row['plan_sha256'] != source_sha for row in arm):
            raise ValueError(f'Changed paired source plan: {name}')
        missed = set.intersection(*({r['annotation_id'] for r in a['rows'] if r['missed_group'] == 'no_dense'}
                                    for a in arm))
        payload = source[f'{name}-annotations.json']
        selected = {a['id']: a for a in payload['annotations'] if a['id'] in missed and a['bbox'][2]*a['bbox'][3] < 32**2}
        if len(selected) != expected:
            raise ValueError(f'Changed fixed small-target cohort: {name}')
        wanted_images = {a['image_id'] for a in selected.values()}
        order = sorted(payload['images'], key=lambda image: image['id'])
        keep = set()
        for i in range(0, len(order), 2):
            batch = order[i:i + 2]
            if any(image['id'] in wanted_images for image in batch):
                # Retain the original batch partner and padding.
                keep.update(image['owod_source_id'] for image in batch)
        assets[name] = subset(payload, keep, source_plan['shared_classes'])
        cohorts[name] = dict(target_ids=sorted(selected), images=len(keep))
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol='m-s-paired', device=device, gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), batch_size=2, paper_comparable=False,
                cohorts=cohorts, parents=source_plan['parents'], parent_plan_sha256=source_sha,
                source_files=SOURCE_FILES, criterion=CRITERION)
    pin(OUTPUT / 'plan.json', plan)
    for name, payload in assets.items():
        path = OUTPUT / f'{name}-annotations.json'
        if not path.exists():
            write_json(path, payload)
        if read_json(path) != payload:
            raise ValueError(f'Changed replay annotation: {name}')
    return plan, resolve_image_root(manifest_path, manifest)


def corrections(gt, boxes):
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) / 2
    gt_centre, gt_half = (gt[:2] + gt[2:]) / 2, (gt[2:] - gt[:2]) / 2
    center_only = np.concatenate((gt_centre - half, gt_centre + half), axis=1)
    extent_only = np.concatenate((centre - gt_half, centre + gt_half), axis=1)
    return [float(overlaps(gt, a).max()) for a in (center_only, extent_only)]


@torch.inference_mode()
def trace(model, loader, wanted):
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        geo = model(samples.to('cuda'))['geometry']
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            anns = valid_annotations(loader.dataset, image)
            if target['labels'].tolist() != [a['category_id'] for a in anns]:
                raise ValueError(f'Annotation order mismatch on image {image}')
            indices = [j for j, ann in enumerate(anns) if ann['id'] in wanted]
            if not indices:
                continue
            ids = torch.where(geo['valid'][i])[0]
            near = nearby_anchor_ids(target['boxes'][indices].to(ids.device), geo['anchors'][i], ids)
            lookup = torch.searchsorted(ids, near).cpu().numpy()
            h, w = target['orig_size'].tolist()
            dense = voc_coords(geo['boxes'][i, ids], w, h)
            for j, neighbor_positions in zip(indices, lookup):
                ann = anns[j]
                gt = np.asarray(ann['voc_bbox'])
                native = float(overlaps(gt, dense).max())
                if native > .5:
                    raise ValueError(f'Previously uncovered target now has native dense coverage: {ann["id"]}')
                center, extent = corrections(gt, dense[neighbor_positions])
                rows.append(dict(annotation_id=ann['id'], category=ann['category_id'],
                                 native_iou=native, center_iou=center, extent_iou=extent))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(wanted) or {row['annotation_id'] for row in rows} != wanted:
        raise ValueError('Incomplete fixed target replay')
    return rows


def summarize(rows):
    counts = Counter((row['center_iou'] > .5, row['extent_iou'] > .5) for row in rows)
    delta = counts[True, False] - counts[False, True]
    return dict(targets=len(rows), center_only=counts[True, False], extent_only=counts[False, True],
                both=counts[True, True], neither=counts[False, False],
                priority=('center' if 10 * delta >= len(rows) else 'extent' if -10 * delta >= len(rows) else None))


def main():
    plan, root = prepare()
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    arms = {}
    for protocol, parent in plan['parents'].items():
        model, _ = load_detector(None, Path(parent['checkpoint']['path']), torch.device('cuda'))
        model.eval()
        for name, part in plan['cohorts'].items():
            key = f'{protocol}-{name}'
            path = OUTPUT / f'{key}.json'
            wanted = set(part['target_ids'])
            if not path.exists():
                _, loader = make_loader(root, OUTPUT / f'{name}-annotations.json', plan['batch_size'], 2)
                print(f'Evaluating {key} on {part["images"]} images', flush=True)
                rows = trace(model, loader, wanted)
                write_json(path, dict(plan_sha256=plan_hash, rows=rows))
            record = read_json(path)
            if (record['plan_sha256'] != plan_hash or len(record['rows']) != len(wanted)
                    or {r['annotation_id'] for r in record['rows']} != wanted):
                raise ValueError(f'Changed paired report: {key}')
            arms[key] = summarize(record['rows'])
        del model
    verdicts = {arm['priority'] for arm in arms.values()}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
               arms=arms, shared_priority=next(iter(verdicts)) if len(verdicts) == 1 else None))
    print('Paired small-box geometry comparison saved', flush=True)


if __name__ == '__main__':
    main()
