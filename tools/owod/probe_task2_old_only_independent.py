"""Confirm old-only refill on two disjoint, release-only Task 2 validation slices."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import torch

from tools.owod import prepare_benchmarks
from tools.owod.probe_task1_known_queries import fresh_origins
from tools.owod.probe_task2_dedup_open_set import SOURCES, evaluate_protocol
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task1_known_fraction import proxy_data
from tools.owod.run_s_task2_old_dedup import DEDUP_IOU, LIMIT, OLD_POOL
from tools.owod.screening_data import source_ids, write_json


RUNS = ROOT / 'output'
PARENT = RUNS / 'instance-task2-old-only-refill'
OUTPUT = RUNS / 'instance-task2-old-only-independent'
SELECTION_SEED = 61
SCREENS = {
    'm-owodb': ['m-instance-stronger-parent', 'm-instance-decoder-bbox-seed43', 'm-instance-expanded',
                'm-instance-pilot', 'm-instance-task2-exposure-pair'],
    's-owodb': ['s-instance-task1-known-fraction'],
}
PRIOR_SLICES = {
    'm-owodb': ['m-instance-task1-encoder-transfer/val-data.json', 'm-instance-task1-visible-classes/val-data.json',
                'm-instance-task1-known-fraction/fresh1024.json', 'm-instance-task1-known-queries/fresh1024.json',
                'm-instance-task1-unknown-budget/fresh1024.json', 'm-instance-task2-duration-holdout/heldout.json'],
    's-owodb': ['s-instance-task1-crop-proxy/fresh1024.json', 's-instance-task2-duration/heldout.json',
                's-instance-task2-old-dedup-holdout/heldout.json'],
}


def independent_data(protocol, prior):
    manifest_path = SOURCES[protocol][0]
    manifest = read_json(manifest_path)
    if (file_sha256(manifest_path) != prior['manifest_sha256'] or
            manifest['evaluation']['ap_mode'] != prior['ap_mode'] or
            file_sha256(Path(prior['annotation'])) != prior['annotation_sha256']):
        raise ValueError(f'Changed source data: {protocol}')
    for checkpoint in prior['checkpoints'].values():
        if file_sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError(f'Changed frozen weights: {protocol}')
    excluded = set(manifest['evaluation']['image_source_ids'])
    hashes = {}
    for run in SCREENS[protocol]:
        path = RUNS / run / 'data/screening.json'
        screen = read_json(path)
        hashes[str(path)] = file_sha256(path)
        if screen['parent_manifest_sha256'] != prior['manifest_sha256']:
            raise ValueError(f'Unexpected screening manifest: {path}')
        excluded.update(screen['heldout_source_ids'])
        for stage in screen['stages'][:2]:
            for key, filename in stage['files'].items():
                ann = path.parent / filename
                if file_sha256(ann) != stage['sha256'][key]:
                    raise ValueError(f'Changed screening file: {ann}')
                excluded.update(source_ids(read_json(ann)))
    for filename in PRIOR_SLICES[protocol]:
        path = RUNS / filename
        hashes[str(path)] = file_sha256(path)
        excluded.update(source_ids(read_json(path)))
    if protocol == 's-owodb':
        for filename, key in [('s-instance-task1-known-fraction/selection/inputs.json', 'origins'),
                              ('s-instance-task1-crop-proxy/selection/origins.json', None),
                              ('s-instance-task1-region-ranking/plan.json', 'split')]:
            path = RUNS / filename
            hashes[str(path)] = file_sha256(path)
            record = read_json(path)
            excluded.update(record[key] if key else record)

    counts = (48, 976) if protocol == 'm-owodb' else (512, 512)
    origins = fresh_origins(manifest, excluded, SELECTION_SEED, counts)
    if set(origins) & excluded or Counter(origins.values()) != dict(zip(('t1_train', 't2_train'), counts)):
        raise ValueError(f'Invalid independent holdout sources: {protocol}')
    if protocol == 's-owodb':
        data = proxy_data(manifest, origins)
    else:
        root = resolve_image_root(manifest_path, manifest)
        records = {name: prepare_benchmarks.read_voc(root.parent, name) for name in origins}
        data = prepare_benchmarks.subset(origins, manifest['stages'][1]['active_classes'], records)
    order = [row['owod_source_id'] for row in data['images']]
    if order != list(origins) or {a['category_id'] for a in data['annotations']} != set(manifest['stages'][1]['active_classes']):
        raise ValueError(f'Incomplete held-out released data: {protocol}')
    spec = {k: prior[k] for k in ('manifest_sha256', 'ap_mode', 'visible_current_ids', 'checkpoints')}
    spec.update(images=len(order), origins=origins, sources=dict(Counter(origins.values())), exclusion_inputs=hashes,
                excluded_ids_sha256=sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
                eval_order_sha256=sha256('\n'.join(order).encode()).hexdigest(),
                annotation=str(OUTPUT / f'{protocol}-heldout.json'),
                annotation_sha256=sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest())
    return spec, data


def main():
    parent = read_json(PARENT / 'plan.json')
    result = read_json(PARENT / 'comparison.json')
    if (file_sha256(PARENT / 'comparison.json') != '664c0695c0daac32f162f3041fc09b7ef6e1645109a0e7867e984a04589e0518' or
            result['status'] != 'complete' or not result['follow_up'] or
            file_sha256(PARENT / 'plan.json') != result['plan_sha256'] or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_name() != parent['gpu'] or str(torch.__version__) != parent['torch_version'] or
            (LIMIT, OLD_POOL, DEDUP_IOU) != (parent['max_detections'], parent['old_pool'], parent['nms_iou'])):
        raise ValueError('Expected the three passing frozen endpoints, original rule and GPU environment')
    protocols, datasets = {}, {}
    for protocol in ('s-owodb', 'm-owodb'):
        print(f'Preparing independent {protocol} images', flush=True)
        protocols[protocol], datasets[protocol] = independent_data(protocol, parent['protocols'][protocol])
    plan = {k: parent[k] for k in ('gpu', 'torch_version', 'batch_size', 'max_detections', 'old_pool', 'nms_iou',
                                   'criterion', 'proxy_policy', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                device=os.environ.get('CUDA_VISIBLE_DEVICES'), selection_seed=SELECTION_SEED, protocols=protocols,
                parent_comparison_sha256=file_sha256(PARENT / 'comparison.json'), parent_plan_sha256=result['plan_sha256'],
                policy='Freeze native Current and unknown entries, refill only Previous. Evaluate all three endpoints '
                       'on fresh training-source images. Each must pass the parent gate. Report the source mix.')
    path = OUTPUT / 'plan.json'
    pin(path, plan)
    plan_hash = file_sha256(path)
    rows = {}
    for protocol, spec in protocols.items():
        annotation = Path(spec['annotation'])
        if not annotation.is_file():
            write_json(annotation, datasets[protocol])
        if file_sha256(annotation) != spec['annotation_sha256']:
            raise ValueError(f'Changed independent holdout: {annotation}')
        rows[protocol] = evaluate_protocol(protocol, SOURCES[protocol][0], None, spec, plan_hash,
                                           output=OUTPUT, old_only=True)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, protocols=rows,
               paper_comparable=False, follow_up=all(r['follow_up'] for group in rows.values() for r in group.values())))
    print('Independent comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
