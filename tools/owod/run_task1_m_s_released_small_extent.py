"""Train and evaluate a released-box extent-loss control independently on M and S."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.graph_local.instance_proposals import SMALL_BOX_EXTENT_WEIGHT
from models.owod_benchmark_eval import evaluate_author_benchmark, overlaps
from tools.owod import prepare_benchmarks, run_instance_pilot as pilot
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_small_box_extent import OUTPUT as EXTENT
from tools.owod.probe_task1_paired_discovery import LOCKED, OUTPUT as PAIRED, SCREENS, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.run_stronger_parent_study import PARENT_IMAGES, run_job
from tools.owod.screening_data import screening_files, select_images, source_ids, write_json


OUTPUT = pilot.ROOT / 'output/task1-m-s-released-small-extent'
MANIFESTS = {protocol: pilot.ROOT / 'data/owod' / protocol / 'split_manifest.json' for protocol in SCREENS}
DINO_SHA256 = '156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c'
GEOMETRY_PLAN_SHA256 = 'e5e87ab8755b0a5a46f92366e1c94eabd57194faf5992efdd07e8d9a7c62e123'
SEED = 42
EPOCHS = 3
SLICE_SEEDS = {'first': 50, 'second': 51}
SLICE_IMAGES = 1024
SMALL_PROXY_IMAGES_PER_SIZE = 192
ARMS = ('control', 'extent')
STAGES = ('dense', 'shortlist', 'selected', 'final')
GATES = dict(small_dense=.03, small_u_recall=.01, all_u_recall=0., known_ap50=-.005, wi=.01, a_ose=.02)
PREVIOUS_SLICES = (
    PAIRED / 'first-annotations.json',
    PAIRED / 'second-annotations.json',
    EXTENT / 'm-owodb-released-annotations.json',
    EXTENT / 's-owodb-released-annotations.json',
    Path('output/m-instance-task1-encoder-transfer/val-data.json'),
    Path('output/m-instance-task1-visible-classes/val-data.json'),
    Path('output/m-instance-task1-known-fraction/fresh1024.json'),
    Path('output/m-instance-task1-known-queries/fresh1024.json'),
    Path('output/m-instance-task1-unknown-budget/fresh1024.json'),
    Path('output/s-instance-task1-known-fraction/evaluation/fresh1024.json'),
    Path('output/s-instance-task1-released-seed/released2048.json'),
    Path('output/m-instance-task2-duration-holdout/heldout.json'),
    Path('output/s-instance-task2-duration/heldout.json'),
    Path('output/instance-task2-old-only-independent/m-owodb-heldout.json'),
    Path('output/instance-task2-old-only-independent/s-owodb-heldout.json'),
)


def annotation_digest(data):
    return sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest()


def size_group(ann):
    area = ann['bbox'][2] * ann['bbox'][3]
    return 'under_16_sq' if area < 16**2 else '16_to_32_sq' if area < 32**2 else '32_sq_plus'


def cohorts(raw, known, blocked, *, seeds=SLICE_SEEDS):
    known = set(known)
    eligible = [im for im in raw['images'] if im['owod_source_id'] not in blocked]
    by_id = {im['id']: im['owod_source_id'] for im in eligible}
    eligible_annotations = [a for a in raw['annotations'] if a['image_id'] in by_id and not a.get('difficult')]
    unknown_small = {size: {a['image_id'] for a in eligible_annotations
                            if a['category_id'] not in known and size_group(a) == size}
                     for size in ('under_16_sq', '16_to_32_sq')}
    known_by_class = {c: {by_id[a['image_id']] for a in eligible_annotations if a['category_id'] == c} for c in known}
    selected = set()
    for name, seed in seeds.items():
        chosen = set()
        for ids in unknown_small.values():
            pool = dict(images=[im for im in eligible if im['id'] in ids], annotations=[])
            chosen.update(select_images(pool, SMALL_PROXY_IMAGES_PER_SIZE, [], seed, excluded=selected | chosen))
        missing = sorted(c for c, images in known_by_class.items() if not images.intersection(chosen))
        chosen.update(select_images(dict(images=eligible, annotations=eligible_annotations),
                                    SLICE_IMAGES - len(chosen), missing, seed, excluded=selected | chosen))
        selected.update(chosen)
        images = [im for im in eligible if im['owod_source_id'] in chosen]
        ids = {im['id'] for im in images}
        yield name, dict(raw, images=images, annotations=[a for a in raw['annotations'] if a['image_id'] in ids])


def raw_evaluation_source(protocol, manifest, blocked):
    names = sorted((set(manifest['splits']['t1_train']) | set(manifest['splits']['t2_train'])) - blocked)
    image_root = resolve_image_root(MANIFESTS[protocol], manifest)
    allowed = manifest['stages'][1]['active_classes']
    if protocol == 's-owodb':
        images, boxes, actual = prepare_benchmarks.load_coco(image_root)
        expected = {Path(path).name: digest for path, digest in manifest['source_hashes'].items()}
        if {Path(path).name: digest for path, digest in actual.items()} != expected:
            raise ValueError('S evaluation source differs from its verified manifest')
        records = {name: (images[name], boxes[name]) for name in names}
    else:
        records = {name: prepare_benchmarks.read_voc(image_root.parent, name) for name in names}
    return prepare_benchmarks.subset(names, allowed, records)


def training_commands(protocol):
    root = OUTPUT / protocol
    original = pilot.command('off', 'task1', output=root, screen=SCREENS[protocol].resolve(),
                             manifest=MANIFESTS[protocol], epochs=EPOCHS, seed=SEED)
    original.remove('--screen-author-eval')
    original.append('--skip-eval')
    commands = {}
    for arm in ARMS:
        job = original.copy()
        job[job.index('--output_dir') + 1] = str(root / arm / 'task1')
        if arm == 'extent':
            job.append('--screen-small-box-extent')
        commands[arm] = job
    return commands


def prepare():
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if gpu_names != ['NVIDIA GeForce RTX 3090'] * 2 or len(set(devices)) != 2 or not all(d.isdecimal() for d in devices):
        raise ValueError('Expected two explicit 3090 devices for the paired Task 1 training')
    if file_sha256(pilot.WEIGHTS) != DINO_SHA256:
        raise ValueError('Changed DINO initialization')
    verified(EXTENT / 'plan.json', GEOMETRY_PLAN_SHA256)
    geometry = read_json(EXTENT / 'comparison.json')
    if (geometry['plan_sha256'] != GEOMETRY_PLAN_SHA256 or geometry['status'] != 'complete'
            or geometry['fixed_scale_direction'] is not None):
        raise ValueError('Missing completed heterogeneous extent evidence')
    excluded, sources = set(), {}
    for path in PREVIOUS_SLICES:
        digest = file_sha256(path)
        excluded.update(source_ids(read_json(path)))
        sources[path.as_posix()] = digest
    manifests, inputs, assets = {}, {}, {}
    for protocol, path in MANIFESTS.items():
        verified(path, LOCKED[protocol]['manifest_sha256'])
        screen_path = SCREENS[protocol]
        screen = verified(screen_path, LOCKED[protocol]['screen_sha256'])
        manifest, files = screening_files(path, screen_path, 0)
        training = read_json(files['increment_train'])
        if len(training['images']) != PARENT_IMAGES:
            raise ValueError(f'Changed Task 1 training budget: {protocol}')
        excluded.update(source_ids(training))
        excluded.update(screen['heldout_source_ids'])
        excluded.update(manifest['evaluation']['image_source_ids'])
        phase2 = screen['stages'][1]
        excluded.update(source_ids(verified(screen_path.parent / phase2['files']['increment_train'],
                                            phase2['sha256']['increment_train'])))
        manifests[protocol] = manifest
        inputs[protocol] = dict(manifest_sha256=LOCKED[protocol]['manifest_sha256'],
                                screen_sha256=LOCKED[protocol]['screen_sha256'],
                                train_sha256=file_sha256(files['increment_train']),
                                jobs=training_commands(protocol), slices={})

    for protocol, manifest in manifests.items():
        print(f'Preparing independent training-source validation: {protocol}', flush=True)
        fixed = excluded.copy()
        raw = raw_evaluation_source(protocol, manifest, fixed)
        image_root = resolve_image_root(MANIFESTS[protocol], manifest)
        known = set(manifest['stages'][0]['classes'])
        for name, data in cohorts(raw, known, fixed):
            names = source_ids(data)
            if len(names) != SLICE_IMAGES or names & excluded:
                raise ValueError(f'Overlapping or incomplete independent slice: {protocol}/{name}')
            excluded.update(names)
            for im in data['images']:
                if not (image_root / im['file_name']).is_file():
                    raise FileNotFoundError(im['file_name'])
            counts = Counter(('known' if a['category_id'] in known else 'proxy', size_group(a))
                             for a in data['annotations'] if not a.get('difficult'))
            if not known <= {a['category_id'] for a in data['annotations'] if not a.get('difficult')}:
                raise ValueError(f'Missing known ground truth: {protocol}/{name}')
            path = OUTPUT / f'{protocol}-{name}-annotations.json'
            assets[path] = data
            inputs[protocol]['slices'][name] = dict(
                file=path.name, sha256=annotation_digest(data), images=len(names),
                counts={f'{category}/{size}': count for (category, size), count in counts.items()})
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                paper_comparable=False, protocols=inputs, backbone_sha256=DINO_SHA256,
                geometry_plan_sha256=GEOMETRY_PLAN_SHA256, exclusion_files=sources,
                seed=SEED, slice_seeds=SLICE_SEEDS, train_images=PARENT_IMAGES, epochs=EPOCHS,
                devices=devices, gpus=gpu_names, torch_version=str(torch.__version__),
                per_gpu_batch_size=2, global_batch_size=4, validation_batch_size=2, max_detections=50,
                extent_weight=SMALL_BOX_EXTENT_WEIGHT, gates=GATES)
    pin(OUTPUT / 'plan.json', plan)
    for path, data in assets.items():
        if not path.exists():
            write_json(path, data)
        if file_sha256(path) != annotation_digest(data):
            raise ValueError(f'Changed validation slice: {path}')
    print('Paired Task 1 inputs and independent validation locked', flush=True)
    return plan, manifests


@torch.inference_mode()
def measure(model, manifest, image_root, path, batch, max_detections, *, observer=None):
    dataset, loader = make_loader(image_root, path, batch, 2)
    known = manifest['stages'][0]['classes']
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=max_detections)
    sequence = [im['owod_source_id'] for im in dataset.coco.dataset['images']]
    specification = {**manifest['evaluation'], 'image_source_ids': sequence}
    predictions, coverage = {}, {key: set() for key in STAGES}
    unknown = {a['id']: a for a in dataset.coco.anns.values()
               if a['category_id'] not in known and not a.get('difficult')}
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        detections = post(out, sizes)
        if observer is not None:
            observer(dataset, samples, targets, out, detections, sizes, post)
        for i, t in enumerate(targets):
            image = int(t['image_id'])
            height, width = t['orig_size'].tolist()
            predictions[image] = {key: detections[i][key].cpu() for key in ('scores', 'boxes', 'labels')}
            proxies = [a for a in dataset.coco.imgToAnns[image] if a['id'] in unknown]
            if not proxies:
                continue
            stages = stage_boxes(out, i)
            for key in coverage:
                coords = voc_coords(stages[key], width, height)
                coverage[key].update(a['id'] for a in proxies if
                                     (overlaps(np.asarray(a['voc_bbox']), coords) > specification['iou_threshold']).any())
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    metrics = evaluate_author_benchmark(predictions, dataset.coco, known, [], known, specification,
                                        include_matched_ids=True, include_wi_sources=True)
    matched = set(metrics.pop('matched_annotation_ids')[-1])
    wi_denominator = sum(item['total'] for item in metrics.pop('wi_sources').values())
    if metrics['unknown_gt'] != len(unknown) or metrics['u_recall'] != len(matched) / len(unknown):
        raise ValueError('Unique VOC unknown matches disagree with the diagnostic denominator')
    groups = {
        'all': set(unknown),
        'small': {i for i, a in unknown.items() if size_group(a) != '32_sq_plus'},
        **{size: {i for i, a in unknown.items() if size_group(a) == size}
           for size in ('under_16_sq', '16_to_32_sq', '32_sq_plus')},
    }
    return dict(metrics=metrics, wi_denominator=wi_denominator,
                groups={name: dict(gt=len(ids), unknown_tp=len(matched & ids),
                                   **{key: len(hits & ids) for key, hits in coverage.items()})
                        for name, ids in groups.items()},
                matched=sorted(matched), coverage={key: sorted(ids) for key, ids in coverage.items()})


def compare(control, extent):
    base_small, new_small = control['groups']['small'], extent['groups']['small']
    count = base_small['gt']
    if new_small['gt'] != count or count <= 0:
        raise ValueError('Unpaired small-object denominator')
    base, new = control['metrics'], extent['metrics']
    deltas = dict(small_dense=(new_small['dense'] - base_small['dense']) / count,
                  small_u_recall=(new_small['unknown_tp'] - base_small['unknown_tp']) / count,
                  all_u_recall=new['u_recall'] - base['u_recall'], known_ap50=new['known_ap50'] - base['known_ap50'],
                  wi=new['wi'] - base['wi'], a_ose=new['a_ose'] - base['a_ose'])
    gates = {key: deltas[key] >= GATES[key] for key in ('small_dense', 'small_u_recall', 'all_u_recall', 'known_ap50')}
    gates['wi'] = not (control['wi_denominator'] and extent['wi_denominator']) or deltas['wi'] <= GATES['wi']
    gates['a_ose'] = deltas['a_ose'] <= max(1, GATES['a_ose'] * base['a_ose'])
    return dict(deltas=deltas, gates=gates, passed=all(gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan, manifests = prepare()
    if args.plan_only:
        return
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for protocol, inputs in plan['protocols'].items():
        manifest = manifests[protocol]
        image_root = resolve_image_root(MANIFESTS[protocol], manifest)
        for arm, command in inputs['jobs'].items():
            print(f'Training {protocol}/{arm} (Task 1, {EPOCHS} epochs)', flush=True)
            checkpoint, saved = run_job(command, inputs['screen_sha256'])
            if (saved.owod_protocol != protocol or saved.backbone_weights_sha256 != plan['backbone_sha256']
                    or bool(saved.screen_small_box_extent) != (arm == 'extent')):
                raise ValueError(f'Unexpected Task 1 endpoint: {checkpoint}')
            endpoint = dict(plan_sha256=plan_hash, checkpoint_sha256=file_sha256(checkpoint))
            path = OUTPUT / f'{protocol}-{arm}.json'
            if not path.exists():
                model, _ = load_detector(None, checkpoint, torch.device('cuda'))
                model.eval()
                records = {}
                for name, spec in inputs['slices'].items():
                    annotations = OUTPUT / spec['file']
                    if file_sha256(annotations) != spec['sha256']:
                        raise ValueError(f'Validation input changed: {annotations}')
                    print(f'Evaluating {protocol}/{arm}/{name} with native focal Top-50', flush=True)
                    records[name] = measure(model, manifest, image_root, annotations,
                                            plan['validation_batch_size'], plan['max_detections'])
                del model
                write_json(path, dict(endpoint, slices=records))
            report = read_json(path)
            if any(report[key] != value for key, value in endpoint.items()):
                raise ValueError(f'Evaluation does not match the trained checkpoint: {path}')
            summaries[protocol, arm] = report['slices']
    deltas = {protocol: {name: compare(summaries[protocol, 'control'][name], summaries[protocol, 'extent'][name])
                         for name in SLICE_SEEDS}
              for protocol in plan['protocols']}
    passed = all(row['passed'] for slices in deltas.values() for row in slices.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
                                               protocols=deltas, shared_gate_passed=passed))
    print('Paired released small-box extent comparison saved', flush=True)


if __name__ == '__main__':
    main()
