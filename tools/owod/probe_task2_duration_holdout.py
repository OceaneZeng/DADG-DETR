"""Compare the four completed M Task 2 endpoints on fresh training-source images."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from tools.owod import prepare_benchmarks, run_decoder_bbox_seed_pair as replication
from tools.owod import run_stronger_parent_study as study
from tools.owod import run_task2_duration_pair as duration
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, resolve_manifest_path
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, select_images, source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-task2-duration-holdout'
IMAGES_PER_TASK = 512
SELECTION_SEED = 44


def evaluate_finetune_holdout(checkpoint, args, dataset, loader, spec, plan_hash):
    """Persist a single fixed-weight focal Top-50 evaluation beside its checkpoint."""
    digest = file_sha256(checkpoint)
    path = checkpoint.parent / 'holdout.json'
    if not path.exists():
        model, _ = load_detector(args, checkpoint, torch.device('cuda'))
        print(f'Evaluating {checkpoint.parent} on {len(dataset)} holdout images', flush=True)
        views = evaluate_views(model, loader, dataset.coco, {'top50': (args.owod_known_class_ids, 'focal', 50)},
                               spec, args.owod_previous_class_ids, args.owod_current_class_ids, 'cuda',
                               include_current_stats=True, print_freq=64)
        write_json(path, dict(view=views, plan_sha256=plan_hash, checkpoint_sha256=digest))
    report = read_json(path)
    if report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != digest:
        raise ValueError(f'Unexpected checkpoint or holdout evaluation: {path}')
    return report


def fresh_annotations():
    manifest, first = screening_files(MANIFEST, study.SCREEN, 0)
    _, second = screening_files(MANIFEST, study.SCREEN, 1)
    screen = read_json(study.SCREEN)
    replica = read_json(replication.SCREEN)
    if (replica['source_screen_sha256'] != file_sha256(study.SCREEN)
            or replica['heldout_source_ids'] != screen['heldout_source_ids']
            or any(a['sha256'] != b['sha256'] for a, b in zip(screen['stages'], replica['stages']))):
        raise ValueError('Seed 43 did not use the same training and validation images')
    excluded = set(manifest['evaluation']['image_source_ids']) | set(screen['heldout_source_ids'])
    for path in (first['increment_train'], second['increment_train'], second['train']):
        excluded.update(source_ids(read_json(path)))

    selected = []
    for stage in manifest['stages'][:2]:
        source = resolve_manifest_path(MANIFEST, stage['files']['increment_train'])
        names = select_images(read_json(source), IMAGES_PER_TASK, stage['classes'], SELECTION_SEED, excluded)
        selected.extend(sorted(names))
        excluded.update(names)
    # Phase training files hide other classes even after those labels have been
    # released. Score all 40 known classes on both sources using the author XML.
    image_root = resolve_image_root(MANIFEST, manifest)
    cache = {name: prepare_benchmarks.read_voc(image_root.parent, name) for name in selected}
    annotation = prepare_benchmarks.subset(selected, manifest['stages'][1]['active_classes'], cache)
    return manifest, image_root, annotation


def endpoints():
    paths = {}
    for seed, short, long, screen in (
        (42, duration.SOURCE, duration.OUTPUT, study.SCREEN),
        (43, replication.OUTPUT / duration.THREE_QUARTER,
         duration.OUTPUT.with_name(duration.OUTPUT.name + '-seed43'), replication.SCREEN),
    ):
        report = read_json(long / 'comparison.json')
        if report['status'] != 'complete' or report['seed'] != seed:
            raise ValueError(f'Missing completed duration pair for seed {seed}')
        for epochs, folder in ((3, short), (6, long)):
            checkpoint = folder / 'finetune/checkpoint.pth'
            digest = file_sha256(checkpoint)
            inc = folder / 'increment/checkpoint.pth'
            meta = load_checkpoint_metadata(checkpoint)
            args = meta['args']
            if (args.seed != seed or args.owod_stage != 1 or args.owod_phase != 'finetune'
                    or args.owod_manifest_sha256 != file_sha256(MANIFEST)
                    or args.owod_screening_sha256 != file_sha256(screen)
                    or args.decoder_bbox_lr_mult != .75 or args.discovery_scoring != 'focal'
                    or args.owod_max_detections != 50 or args.batch_size != 2
                    or args.epochs != 1 or meta['epoch'] != 0 or Path(args.pretrained) != inc
                    or read_json(folder / 'finetune/training_complete.json')['last_epoch'] != 0
                    or read_json(folder / 'increment/training_complete.json')['last_epoch'] != epochs - 1
                    or read_json(folder / 'finetune/input.json')['--pretrained'] != file_sha256(inc)):
                raise ValueError(f'Unexpected training endpoint: {checkpoint}')
            if epochs == 3 and read_json(checkpoint.parent / 'coverage.json')['checkpoint_sha256'] != digest:
                raise ValueError(f'Original reference checkpoint changed: {checkpoint}')
            paths[f'seed{seed}-{epochs}epochs'] = dict(path=str(checkpoint), sha256=digest,
                                                      source_result_sha256=file_sha256(long / 'comparison.json'))
    return paths


def main():
    print('Verifying source annotations and four completed checkpoints', flush=True)
    manifest, image_root, annotation = fresh_annotations()
    annotation_path = OUTPUT / 'heldout.json'
    encoded = (json.dumps(annotation, indent=2, sort_keys=True) + '\n').encode()
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                manifest_sha256=file_sha256(MANIFEST), checkpoints=endpoints(),
                annotation_sha256=hashlib.sha256(encoded).hexdigest(), images_per_task=IMAGES_PER_TASK,
                selection_seed=SELECTION_SEED, paper_comparable=False, gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), primary_endpoint='finetune current_ap50',
                annotation_policy='all 40 released categories in author XML for held-out training images')
    pin(OUTPUT / 'plan.json', plan)
    write_json(annotation_path, annotation)
    if file_sha256(annotation_path) != plan['annotation_sha256']:
        raise ValueError('Written holdout differs from its registered plan')
    image_ids = [image['owod_source_id'] for image in annotation['images']]
    spec = dict(manifest['evaluation'], image_source_ids=image_ids, known_only_screening=True)
    stage = manifest['stages'][1]
    views = {'focal_top50': (stage['active_classes'], 'focal', 50)}
    dataset, loader = make_loader(image_root, annotation_path, 2, 2)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    rows = {}
    for key, entry in plan['checkpoints'].items():
        row_path = OUTPUT / f'{key}.json'
        if not row_path.exists():
            checkpoint = Path(entry['path'])
            model, _ = load_detector(load_checkpoint_metadata(checkpoint)['args'], checkpoint, torch.device('cuda'))
            print(f'Evaluating {key} on {len(dataset)} images', flush=True)
            view = evaluate_views(model, loader, dataset.coco, views, spec, manifest['stages'][0]['classes'],
                                  stage['classes'], 'cuda', include_current_stats=True, print_freq=32)
            write_json(row_path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], view=view))
            del model
        row = read_json(row_path)
        if row['plan_sha256'] != plan_hash or row['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Evaluation differs from its registered plan: {row_path}')
        rows[key] = row['view']['focal_top50']['metrics']
    differences = {}
    for seed in (42, 43):
        short, long = (rows[f'seed{seed}-{epochs}epochs'] for epochs in (3, 6))
        differences[seed] = {name: 100 * (long[name] - short[name]) for name in short}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows,
                                               six_minus_three_pp=differences))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
