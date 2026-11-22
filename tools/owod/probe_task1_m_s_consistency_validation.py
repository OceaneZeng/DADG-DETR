"""Check the fixed-budget small-box gate on fresh Task 1 diagnostic slices."""

from collections import Counter
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task1_consistency_budget import SMALL_AREA, SMALL_THRESHOLD, replay, summarize
from tools.owod.probe_task1_discovery_gates import frozen_miner
from tools.owod.probe_task1_paired_discovery import LOCKED, protocol_inputs
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import (
    OUTPUT as EXTENT, PREVIOUS_SLICES, SLICE_IMAGES, annotation_digest, cohorts,
    raw_evaluation_source, size_group,
)
from tools.owod.screening_data import source_ids, write_json


OUTPUT = Path('output/task1-m-s-consistency-validation')
SEEDS = {'first': 52, 'second': 53}
BATCH_SIZE = 2
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
NATIVE_THRESHOLD = .5
PRIOR_SLICES = (*PREVIOUS_SLICES,
                *[EXTENT / f'{protocol}-{fold}-annotations.json' for protocol in LOCKED for fold in SEEDS],
                Path('output/task1-m-s-small-box-geometry-replay/first-annotations.json'),
                Path('output/task1-m-s-small-box-geometry-replay/second-annotations.json'),
                Path('output/s-instance-task1-crop-proxy/fresh1024.json'))


def prepare():
    excluded = set()
    exclusions = {}
    prior = set(PRIOR_SLICES) | set(Path('output').glob('m-instance-task1-*/seed*-holdout1024*.json'))
    for path in sorted(prior):
        excluded.update(source_ids(read_json(path)))
        exclusions[path.as_posix()] = file_sha256(path)
    inputs = {protocol: protocol_inputs(protocol, excluded) for protocol in LOCKED}
    plan_protocols, assets = {}, {}
    for protocol, (_, root, manifest) in inputs.items():
        print(f'Registering {protocol} validation slices', flush=True)
        known = set(manifest['stages'][0]['classes'])
        raw = raw_evaluation_source(protocol, manifest, excluded)
        slices = {}
        for fold, data in cohorts(raw, known, excluded, seeds=SEEDS):
            names = source_ids(data)
            if len(names) != SLICE_IMAGES or names & excluded:
                raise ValueError(f'Overlapping or incomplete validation slice: {protocol}/{fold}')
            excluded.update(names)
            annotations = [ann for ann in data['annotations'] if not ann.get('difficult')]
            if not known <= {ann['category_id'] for ann in annotations}:
                raise ValueError(f'Missing released classes: {protocol}/{fold}')
            if any(not (root / image['file_name']).is_file() for image in data['images']):
                raise ValueError(f'Missing validation images: {protocol}/{fold}')
            counts = Counter(
                f'{"released" if ann["category_id"] in known else "proxy"}/{size_group(ann)}'
                for ann in annotations
            )
            if not counts['proxy/under_16_sq'] or not counts['proxy/16_to_32_sq']:
                raise ValueError(f'Missing small proxy targets: {protocol}/{fold}')
            path = OUTPUT / f'{protocol}-{fold}-annotations.json'
            assets[path] = data
            slices[fold] = dict(file=path.name, sha256=annotation_digest(data),
                                images=len(names), target_counts=counts)
        plan_protocols[protocol] = dict(**LOCKED[protocol], slices=slices)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                device=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), paper_comparable=False, batch_size=BATCH_SIZE,
                native_consistency=NATIVE_THRESHOLD, small_consistency=SMALL_THRESHOLD, small_area=SMALL_AREA,
                seeds=SEEDS, exclusion_files=exclusions, protocols=plan_protocols,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
                acceptance=dict(small_proxy='positive', proxy='nonnegative', released='nonpositive'))
    pin(OUTPUT / 'plan.json', plan)
    for path, data in assets.items():
        if not path.exists():
            write_json(path, data)
        if file_sha256(path) != annotation_digest(data):
            raise ValueError(f'Changed locked annotation file: {path}')
    return inputs, plan_protocols, file_sha256(OUTPUT / 'plan.json')


def paired_intervals(report):
    image_indices = {row['image_id']: index for index, row in enumerate(report['images'])}
    counts = {name: np.zeros((len(image_indices), 2), dtype=np.int64)
              for name in ('small_proxy', 'proxy', 'released')}
    for row in report['rows']:
        index = image_indices[row['image_id']]
        delta = int(row['candidate']) - int(row['native'])
        role = 'released' if row['released'] else 'proxy'
        counts[role][index] += (delta, 1)
        if role == 'proxy' and row['size'] != '32_sq_plus':
            counts['small_proxy'][index] += (delta, 1)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(image_indices), size=(BOOTSTRAP_REPLICATES, len(image_indices)))
    estimates = {}
    for name, values in counts.items():
        sums = values.sum(axis=0)
        bootstrap = values[samples].sum(axis=1)
        observed = bootstrap[:, 1] > 0
        rates = 100 * bootstrap[observed, 0] / bootstrap[observed, 1]
        estimates[name] = dict(delta_percentage_points=float(100 * sums[0] / sums[1]),
                               ci95_percentage_points=np.quantile(rates, [.025, .975]).tolist())
    return estimates


def passed(summary):
    groups = summary['groups']
    return (groups['small_proxy']['gained'] > groups['small_proxy']['lost']
            and groups['proxy']['gained'] >= groups['proxy']['lost']
            and groups['released']['gained'] <= groups['released']['lost'])


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    inputs, protocols, digest = prepare()
    print('Four independent validation slices locked', flush=True)
    results = {}
    for protocol, (_, root, manifest) in inputs.items():
        spec = protocols[protocol]
        model, criterion, ema = frozen_miner(Path(spec['checkpoint']['path']), manifest)
        if criterion.consistency_threshold != NATIVE_THRESHOLD:
            raise ValueError(f'Changed native consistency threshold: {protocol}')
        results[protocol] = {}
        for fold, info in spec['slices'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at fixed image budgets', flush=True)
                dataset, loader = make_loader(root, OUTPUT / info['file'], BATCH_SIZE, 2)
                write_json(dest, dict(plan_sha256=digest, **replay(model, criterion, ema, dataset, loader)))
            report = read_json(dest)
            if (report['plan_sha256'] != digest or len(report['images']) != info['images']
                    or len(report['rows']) != sum(info['target_counts'].values())
                    or any(len(row['native']) != len(row['candidate']) for row in report['images'])):
                raise ValueError(f'Changed replay budget or inputs: {protocol}/{fold}')
            summary = dict(**summarize(report), paired_by_image=paired_intervals(report))
            results[protocol][fold] = dict(**summary, passed=passed(summary))
        del model, criterion, ema
    accepted = all(row['passed'] for protocol in results.values() for row in protocol.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=results, accepted=accepted))


if __name__ == '__main__':
    main()
