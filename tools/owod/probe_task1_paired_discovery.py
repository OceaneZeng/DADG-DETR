"""Trace the same released targets through frozen M and S Task 1 detectors."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.screening_data import select_images, source_ids, write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = Path('output/task1-m-s-paired-discovery')
LOCKED = json.loads("{\"m-owodb\":{\"manifest_sha256\":\"a5bd4b44dbded3f7f5e62bc3e95e2f69ac9f3eedcead3b3daeb64c32a59daa94\",\"screen_sha256\":\"a09b43a0eb901b3786056ef63df7956169ccfe237675b4159479a3fe5e26da14\",\"checkpoint\":{\"path\":\"/home/top/disks/new-hdd/zhy/Tree-DETR/output/m-instance-stronger-parent/off/task1/checkpoint.pth\",\"sha256\":\"262386ef1081866e670961242b5c9667a914cf021967bd7e4966552a5b3bfceb\"}},\"s-owodb\":{\"manifest_sha256\":\"0060c5f7b080ea417f88295fdcabd7308cc745610319a160ce9787e219d87640\",\"screen_sha256\":\"34f48520fefc31528c087111bdadc1371a1a6eebf5eedcdfec6a29e3240c70e0\",\"checkpoint\":{\"path\":\"/home/top/disks/new-hdd/zhy/Tree-DETR/output/s-instance-task1-known-fraction/off/task1/checkpoint.pth\",\"sha256\":\"7c7466e8f322c8286aa9a432a6df7d65be275058e3057122646bc8c96645b863\"}}}")
COHORTS = json.loads("{\"first\":{\"image_bytes_sha256\":\"82f82ca5de82c79859d1908c2d4857aa7fa158c1cbec47534041c8999279e36c\",\"images\":1024,\"order_sha256\":\"05c4db2fb6ba13dd9a26d49615adc84a276fc027dea8c5897f7883d1c0714bbb\",\"seed\":42,\"target_geometry_sha256\":\"8c28cb16e7de089ee8768764a633161d2546e0f6c3cbffdb753656990d45cfec\",\"targets\":2005},\"second\":{\"image_bytes_sha256\":\"6b90560067a64148bf395738f2019b5c66f15ddacd7fd9806a21fa9421dce4eb\",\"images\":1024,\"order_sha256\":\"16e32836d6d612feeb5593ebf930828e6759ebee752c355c9401f2d87cd8d150\",\"seed\":43,\"target_geometry_sha256\":\"56691279b918b0072d55b0cd1e9d1cba3a25053ba7fffe12bd9026a4739cc2d6\",\"targets\":2523}}")
SCREENS = {
    'm-owodb': Path('output/m-instance-stronger-parent/data/screening.json'),
    's-owodb': Path('output/s-instance-task1-known-fraction/data/screening.json'),
}
HOLDOUT = {
    'output/m-instance-task2-duration-holdout/heldout.json': 'f797928276533b12cb1124c534e7d96793fdb3806b2ff279e0a545c93445cc94',
    'output/s-instance-task2-duration/heldout.json': '551ae613045c06a4022824d95cb39266a9c63b8ac2cc91b5f9247d611bb9a9b3',
    'output/instance-task2-old-only-independent/s-owodb-heldout.json': 'fc24cbdae8461aab9c233906fd564db97a35050be16cd7ee76d6d015cdf62c60',
}
GEOMETRY_STAGES = ('dense', 'shortlist', 'selected', 'final')
GROUPS = ('no_dense', 'outside_shortlist', 'outside_selected', 'decoder', 'unknown_selection')


def verified(path, digest):
    path = Path(path)
    if file_sha256(path) != digest:
        raise ValueError(f'Changed input: {path}')
    return read_json(path)


def protocol_inputs(name, blocked):
    source = LOCKED[name]
    mp = Path('data/owod', name, 'split_manifest.json')
    verified(mp, source['manifest_sha256'])
    manifest, phases = stage_files(mp, 1)
    checkpoint = Path(source['checkpoint']['path'])
    if file_sha256(checkpoint) != source['checkpoint']['sha256']:
        raise ValueError(f'Changed Task 1 parent: {name}')
    config = read_json(checkpoint.parent / 'run_config.json')
    if (config['owod_protocol'], config['owod_stage'], config['discovery_version'],
            config['discovery_scoring'], config['instance_mode'], config['epochs']) != (name, 0, 10, 'focal', 'off', 12):
        raise ValueError(f'Unexpected Task 1 parent configuration: {name}')
    finished = read_json(checkpoint.parent / 'training_complete.json')
    if finished['last_epoch'] != 11:
        raise ValueError(f'Incomplete Task 1 parent: {name}')
    screen = SCREENS[name]
    data = verified(screen, source['screen_sha256'])
    if data['parent_manifest_sha256'] != source['manifest_sha256']:
        raise ValueError('The diagnostic data has a different manifest')
    for stage, labels in ((0, ('increment_train', 'full_val')), (1, ('increment_train',))):
        row = data['stages'][stage]
        for label in labels:
            blocked.update(source_ids(verified(screen.parent / row['files'][label], row['sha256'][label])))
    blocked.update(manifest['evaluation']['image_source_ids'])
    return read_json(phases['increment_train']), resolve_image_root(mp, manifest), manifest


def register(payload, eligible, classes):
    images = {row['owod_source_id']: row for row in payload['images'] if row['owod_source_id'] in eligible}
    names = {row['id']: name for name, row in images.items()}
    coords = {}
    for ann in payload['annotations']:
        i, c = ann['image_id'], ann['category_id']
        if i in names and c in classes and not ann.get('difficult') and not ann.get('iscrowd') and min(ann['bbox'][2:]) > 0:
            key = names[i], c, tuple(map(float, ann['voc_bbox']))
            coords[key] = None if key in coords else ann
    return images, coords


def locked_cohorts(m, s, blocked):
    m_data, m_root, mm = m
    s_data, s_root, sm = s
    common = [c for c in mm['stages'][1]['classes'] if c in sm['stages'][1]['classes']]
    old = set(mm['stages'][0]['classes']) | set(sm['stages'][0]['classes'])
    if old.intersection(common) or len(common) != 15:
        raise ValueError('Shared proxies must be unseen in both Task 1 parents')
    eligible = (source_ids(m_data) & source_ids(s_data)) - blocked
    mi, mo = register(m_data, eligible, set(common))
    si, so = register(s_data, eligible, set(common))
    by_image = defaultdict(list)
    for key in sorted(mo.keys() & so.keys()):
        if mo[key] is not None and so[key] is not None:
            by_image[key[0]].append(key)
    selection = dict(images=[im for name, im in mi.items() if name in by_image],
                     annotations=[mo[key] for group in by_image.values() for key in group])
    used, payloads = set(), {}
    for name, lock in COHORTS.items():
        images = select_images(selection, lock['images'], common, lock['seed'], excluded=used)
        used.update(images)
        order = sorted(images)
        image_digest, geometry_digest = sha256(), sha256()
        annotations = []
        for identity in order:
            left, right = mi[identity], si[identity]
            digest = file_sha256(m_root / left['file_name'])
            if ((left['width'], left['height']) != (right['width'], right['height'])
                    or digest != file_sha256(s_root / right['file_name'])):
                raise ValueError(f'Image bytes or dimensions differ: {identity}')
            image_digest.update(f'{identity}:{digest}\n'.encode())
            for key in by_image[identity]:
                geometry_digest.update((json.dumps(key, separators=(',', ':')) + '\n').encode())
                annotations.append(mo[key])
        actual = (sha256('\n'.join(order).encode()).hexdigest(), image_digest.hexdigest(),
                  geometry_digest.hexdigest(), len(annotations))
        expected = (lock['order_sha256'], lock['image_bytes_sha256'], lock['target_geometry_sha256'], lock['targets'])
        if actual != expected or {a['category_id'] for a in annotations} != set(common):
            raise ValueError(f'The registered paired slice changed: {name}')
        payloads[name] = dict(m_data, images=[mi[i] for i in order], annotations=annotations)
    return payloads, common


def prepare():
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not device.isdecimal() or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090'):
        raise ValueError('Expected one RTX 3090 for the paired frozen replay')
    blocked = set()
    for path, digest in HOLDOUT.items():
        blocked.update(source_ids(verified(path, digest)))
    m, s = (protocol_inputs(name, blocked) for name in SCREENS)
    payloads, common = locked_cohorts(m, s, blocked)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol='m-s-paired', device=device, gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), batch_size=2, paper_comparable=False,
                parents=LOCKED, cohorts=COHORTS, shared_classes=common,
                known={name: row[2]['stages'][0]['classes'] for name, row in zip(SCREENS, (m, s))},
                criterion='A shared priority requires the same missed-target category to explain a strict majority '
                          'of all targets without an unknown Top-50 box in every protocol and slice.')
    pin(OUTPUT / 'plan.json', plan)
    for name, payload in payloads.items():
        path = OUTPUT / f'{name}-annotations.json'
        if not path.exists():
            write_json(path, payload)
        if read_json(path) != payload:
            raise ValueError(f'Changed frozen replay annotation: {name}')
    return plan, m[1]


def voc_coords(boxes, width, height):
    xy = box_cxcywh_to_xyxy(boxes) * boxes.new_tensor([width, height, width, height])
    xy = xy.cpu().numpy().astype(np.float64)
    xy[:, :2] += 1
    return xy.round(decimals=1)


def missed_group(flags):
    for name, flag in zip(GROUPS, flags):
        if not flag:
            return name
    return 'unknown_selection'


@torch.inference_mode()
def trace(model, loader, known):
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    coco = loader.dataset.coco
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        detections = post(out, sizes, include_top_indices=True)
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            h, w = target['orig_size'].tolist()
            stages = stage_boxes(out, i)
            boxes = [voc_coords(stages[k], w, h) for k in GEOMETRY_STAGES]
            d = detections[i]
            indices = (d['top_indices'] // (len(known) + 1)).cpu().numpy()
            unknown = d['labels'].cpu().numpy() == 91
            for ann in coco.imgToAnns[image]:
                gt = np.asarray(ann['voc_bbox'])
                flags = [bool((overlaps(gt, b) > .5).any()) for b in boxes]
                hits = overlaps(gt, boxes[-1])[indices] > .5
                discovered = bool(hits[unknown].any())
                category = ('discovered' if discovered else
                            'unknown_selection' if flags[-1] else missed_group(flags))
                rows.append(dict(image=coco.imgs[image]['owod_source_id'], annotation_id=ann['id'],
                    category=ann['category_id'], coverage=dict(zip(GEOMETRY_STAGES, flags)),
                    unknown=discovered, overlapped_known=bool(hits[~unknown].any()), missed_group=category))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(coco.anns) or len({r['annotation_id'] for r in rows}) != len(rows):
        raise ValueError('The paired target identities did not replay exactly')
    return rows


def summarize(rows):
    counts = Counter(row['missed_group'] for row in rows)
    missed = len(rows) - counts['discovered']
    majority = next((name for name in GROUPS if counts[name] * 2 > missed), None)
    known = sum(row['missed_group'] == 'unknown_selection' and row['overlapped_known'] for row in rows)
    return dict(targets=len(rows), counts=dict(counts), majority=majority,
                selection_missed_with_overlapping_known=known)


def main():
    plan, root = prepare()
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, spec in plan['parents'].items():
        model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
        model.eval()
        for cohort, lock in plan['cohorts'].items():
            name = f'{protocol}-{cohort}'
            path = OUTPUT / f'{name}.json'
            if not path.exists():
                _, loader = make_loader(root, OUTPUT / f'{cohort}-annotations.json', plan['batch_size'], 2)
                print(f'Evaluating {name} on {lock["images"]} images', flush=True)
                rows = trace(model, loader, plan['known'][protocol])
                write_json(path, dict(plan_sha256=plan_hash, rows=rows))
            report = read_json(path)
            if report['plan_sha256'] != plan_hash or len(report['rows']) != lock['targets']:
                raise ValueError(f'Unpaired diagnostic report: {name}')
            results[name] = summarize(report['rows'])
        del model
    priorities = {row['majority'] for row in results.values()}
    common = priorities.pop() if len(priorities) == 1 else None
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
               plan_sha256=plan_hash, arms=results, shared_priority=common))
    print('Paired Task 1 discovery comparison saved', flush=True)


if __name__ == '__main__':
    main()
