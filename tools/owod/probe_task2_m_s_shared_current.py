"""Replay both Task 2 endpoints on identical held-out images and newly released targets."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_unknown_transition import measure
from tools.owod.probe_task1_m_s_consistency_validation import OUTPUT as TASK1_VALIDATION, PRIOR_SLICES
from tools.owod.probe_task1_paired_discovery import LOCKED, SCREENS, register, verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, annotation_digest, raw_evaluation_source
from tools.owod.screening_data import screening_files, select_images, source_ids, subset, write_json


OUTPUT = Path('output/task2-m-s-shared-current')
FOLDS = {'first': 62, 'second': 63}
IMAGES_PER_FOLD = 1024
POOL_IMAGES = 4096
POOL_SEED = 64
BATCH = 2
PHASES = {'increment': 3, 'finetune': 1}
ENDPOINTS = {
    'm-owodb': (
        Path('output/m-instance-decoder-bbox-lr-three-quarter'),
        'b71f7c1d42b7a35687cf0a783a6d2cb2840458ca9cfe1cb7f8cd2eac6ceb23fa',
        '8f7e06037be20dc04bc5d28a9173c643e456b3350e856d6b2220acea51379b1b',
    ),
    's-owodb': (
        Path('output/s-instance-task2-duration/3epochs'),
        '10125ef9a29f9e4b3fc854c99c266d5554fb77fc7102b97bd8c894f542926f2d',
        'ce7c483475383b87337f8ca9b2a84de590368450391e7ae184ccd81db84368a4',
    ),
}
ADDITIONAL_EXCLUSIONS = (
    Path('output/s-instance-task2-old-dedup-holdout/heldout.json'),
    Path('output/s-instance-task2-alternate-pool-fit/independent-annotations.json'),
    *(TASK1_VALIDATION / f'{protocol}-{fold}-annotations.json' for protocol in LOCKED for fold in FOLDS),
)
OUTCOMES = ('no_box', 'no_correct_class_output', 'unmatched_output', 'matched')


def endpoints(protocol):
    directory, *digests = ENDPOINTS[protocol]
    locked = LOCKED[protocol]
    source = locked['checkpoint']
    expected = dict(seed=42, owod_protocol=protocol, owod_stage=1, discovery_version=10,
                    discovery_scoring='focal', instance_mode='off', graph_routing='soft',
                    graph_localization='decoder-bbox', owod_max_detections=50, world_size=2, batch_size=2,
                    owod_manifest_sha256=locked['manifest_sha256'], owod_screening_sha256=locked['screen_sha256'],
                    decoder_bbox_lr_mult=.75)
    output = {}
    for (phase, epochs), digest in zip(PHASES.items(), digests, strict=True):
        checkpoint = directory / phase / 'checkpoint.pth'
        config = checkpoint.parent / 'run_config.json'
        identity = dict(expected, owod_phase=phase, epochs=epochs, pretrained=source['path'])
        actual = read_json(config)
        finished = read_json(checkpoint.parent / 'training_complete.json')
        inputs = read_json(checkpoint.parent / 'input.json')
        if (any(actual.get(key) != value for key, value in identity.items())
                or finished['last_epoch'] != epochs - 1 or inputs['--pretrained'] != source['sha256']
                or file_sha256(checkpoint) != digest):
            raise ValueError(f'Changed endpoint or parent chain: {checkpoint}')
        source = dict(path=str(checkpoint.resolve()), sha256=digest)
        output[phase] = dict(**source, config_sha256=file_sha256(config))
    return output


def collect_sources():
    blocked, exclusions = set(), {}
    for path in sorted(set((*PRIOR_SLICES, *ADDITIONAL_EXCLUSIONS))):
        blocked.update(source_ids(read_json(path)))
        exclusions[str(path)] = file_sha256(path)
    manifests, originals, specs, roots = {}, {}, {}, {}
    for protocol, path in MANIFESTS.items():
        spec = LOCKED[protocol]
        verified(path, spec['manifest_sha256'])
        manifest, files = stage_files(path, 1)
        screen_path = SCREENS[protocol]
        screen = verified(screen_path, spec['screen_sha256'])
        blocked.update(manifest['evaluation']['image_source_ids'])
        blocked.update(screen['heldout_source_ids'])
        for stage, labels in ((0, ('increment_train', 'full_val')), (1, ('increment_train', 'train', 'full_val'))):
            _, screening = screening_files(path, screen_path, stage)
            for label in labels:
                annotation = screening[label]
                blocked.update(source_ids(read_json(annotation)))
                exclusions[str(annotation)] = file_sha256(annotation)
        manifests[protocol] = manifest
        originals[protocol] = read_json(files['increment_train'])
        roots[protocol] = resolve_image_root(path, manifest)
        specs[protocol] = dict(manifest_sha256=spec['manifest_sha256'], screen_sha256=spec['screen_sha256'],
                               author_source_sha256=file_sha256(files['increment_train']),
                               previous=manifest['stages'][0]['classes'], current=manifest['stages'][1]['classes'],
                               ap_mode=manifest['evaluation']['ap_mode'], endpoints=endpoints(protocol))
    return manifests, originals, specs, roots, blocked, exclusions


def prepare(*, output=OUTPUT, fold_seeds=None, pool_seed=POOL_SEED, excluded_sources=()):
    manifests, originals, specs, roots, blocked, exclusions = collect_sources()
    fold_seeds = FOLDS if fold_seeds is None else fold_seeds
    blocked.update(excluded_sources)
    m, s = ('m-owodb', 's-owodb')
    shared = set(specs[m]['current']) & set(specs[s]['current'])
    if len(shared) != 15:
        raise ValueError('Changed shared Task 2 classes')
    eligible = (source_ids(originals[m]) & source_ids(originals[s])) - blocked
    registered = {p: register(originals[p], eligible, shared) for p in specs}
    mr, sr = (registered[p][1] for p in (m, s))
    matches = {key for key in mr.keys() & sr.keys() if mr[key] is not None and sr[key] is not None}
    names_with_matches = {key[0] for key in matches}
    # Only sample sources that have an exact shared target in both protocols.
    pool = dict(images=[im for name, im in registered[m][0].items() if name in names_with_matches],
                annotations=[mr[key] for key in sorted(matches)])
    candidate_names = select_images(pool, POOL_IMAGES, sorted(shared), pool_seed)
    full = {}
    coverage_pool = dict(images=[im for im in pool['images'] if im['owod_source_id'] in candidate_names],
                         annotations=[])
    image_ids = {im['owod_source_id']: im['id'] for im in coverage_pool['images']}
    coverage_classes = []
    for role, (protocol, manifest) in enumerate(manifests.items()):
        training_names = set(manifest['splits']['t1_train']) | set(manifest['splits']['t2_train'])
        full[protocol] = raw_evaluation_source(protocol, manifest, training_names - candidate_names)
        offset = 100 * role
        names_by_id = {im['id']: im['owod_source_id'] for im in full[protocol]['images']}
        active = specs[protocol]['previous'] + specs[protocol]['current']
        coverage_classes.extend(offset + c for c in active)
        coverage_pool['annotations'].extend(
            dict(image_id=image_ids[names_by_id[row['image_id']]], category_id=offset + row['category_id'],
                 bbox=row['bbox'])
            for row in full[protocol]['annotations'] if not row.get('difficult')
        )
    used, selections = set(), {}
    for fold, seed in fold_seeds.items():
        names = select_images(coverage_pool, IMAGES_PER_FOLD, coverage_classes, seed, excluded=used)
        selections[fold] = sorted(names)
        used.update(names)

    cohorts, assets = {}, {}
    for fold, order in selections.items():
        names = set(order)
        identity = sha256()
        for name in order:
            left, right = (registered[p][0][name] for p in (m, s))
            digest = file_sha256(roots[m] / left['file_name'])
            if ((left['width'], left['height']) != (right['width'], right['height'])
                    or digest != file_sha256(roots[s] / right['file_name'])):
                raise ValueError(f'Changed paired image: {name}')
            identity.update(f'{name}:{digest}\n'.encode())
        pair_rows, fold_specs = {}, {}
        common_keys = sorted(key for key in matches if key[0] in names)
        for protocol, spec in specs.items():
            active = set(spec['previous'] + spec['current'])
            payload = subset(full[protocol], names, active)
            _, raw_rows = register(payload, names, shared)
            present = {r['category_id'] for r in payload['annotations'] if not r.get('difficult')}
            unpaired = sum(raw_rows.get(key) is None for key in common_keys)
            if source_ids(payload) != names or present != active or unpaired:
                raise ValueError(f'Incomplete {protocol}/{fold}: missing classes={sorted(active - present)}, '
                                 f'unpaired targets={unpaired}')
            pair_rows[protocol] = [raw_rows[key]['id'] for key in common_keys]
            path = output / f'{protocol}-{fold}-annotations.json'
            assets[path] = payload
            fold_specs[protocol] = dict(file=path.name, sha256=annotation_digest(payload))
        pairs = [dict(image=key[0], category=key[1], voc_bbox=key[2],
                      annotation_ids={protocol: ids[i] for protocol, ids in pair_rows.items()})
                 for i, key in enumerate(common_keys)]
        pair_path = output / f'{fold}-paired-targets.json'
        assets[pair_path] = pairs
        cohorts[fold] = dict(order=order, image_bytes_sha256=identity.hexdigest(), targets=len(pairs),
                             paired_file=pair_path.name, paired_sha256=annotation_digest(pairs),
                             by_protocol=fold_specs)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocols=specs, shared_classes=sorted(shared), cohorts=cohorts, exclusions=exclusions,
                excluded_ids_sha256=sha256('\n'.join(sorted(blocked)).encode()).hexdigest(),
                seeds=fold_seeds, candidate_pool=dict(seed=pool_seed, images=POOL_IMAGES),
                batch_size=BATCH, paper_comparable=False, scoring='focal', max_detections=50,
                gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
    pin(output / 'plan.json', plan)
    for path, value in assets.items():
        if not path.exists():
            write_json(path, value)
        if file_sha256(path) != annotation_digest(value):
            raise ValueError(f'Changed frozen input: {path}')
    return plan, manifests, roots


def summarize(view, pairs, protocol):
    raw, emitted, matched = (set(view[key]) for key in ('raw', 'emitted', 'matched'))
    results = []
    for pair in pairs:
        target = pair['annotation_ids'][protocol]
        outcome = ('matched' if target in matched else 'unmatched_output' if target in emitted
                   else 'no_correct_class_output' if target in raw else 'no_box')
        results.append(outcome)
    counts = Counter(dict.fromkeys(OUTCOMES, 0))
    counts.update(results)
    return dict(metrics=view['metrics'], shared_counts=counts, outcomes=results)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    print('Checking both author splits, frozen endpoints and paired image identities', flush=True)
    plan, manifests, roots = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    print('Two held-out Task 2 folds locked', flush=True)
    results = {}
    for protocol, spec in plan['protocols'].items():
        active = spec['previous'] + spec['current']
        results[protocol] = {}
        for phase, endpoint in spec['endpoints'].items():
            model, _ = load_detector(None, Path(endpoint['path']), torch.device('cuda'))
            if set(model.known_ids) != set(active):
                raise ValueError(f'Changed active classes: {protocol}/{phase}')
            results[protocol][phase] = {}
            for fold, cohort in plan['cohorts'].items():
                annotation = cohort['by_protocol'][protocol]
                pairs = verified(OUTPUT / cohort['paired_file'], cohort['paired_sha256'])
                path = OUTPUT / f'{protocol}-{phase}-{fold}.json'
                if not path.exists():
                    print(f'Evaluating {protocol}/{phase}/{fold} at native focal Top-50', flush=True)
                    ds, loader = make_loader(roots[protocol], OUTPUT / annotation['file'], BATCH, 2)
                    evaluation = dict(manifests[protocol]['evaluation'], image_source_ids=cohort['order'])
                    view = measure(model, ds, loader, active, spec['previous'], spec['current'], evaluation, False)
                    write_json(path, dict(plan_sha256=digest, **summarize(view, pairs, protocol)))
                report = read_json(path)
                if report['plan_sha256'] != digest or len(report['outcomes']) != cohort['targets']:
                    raise ValueError(f'Changed endpoint replay: {path}')
                results[protocol][phase][fold] = {k: report[k] for k in ('metrics', 'shared_counts')}
            del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, protocols=results))


if __name__ == '__main__':
    main()
