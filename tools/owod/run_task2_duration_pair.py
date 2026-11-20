"""Train the six-epoch M Task 2 treatment against the completed three-epoch arm."""
import argparse
import os
import subprocess

from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_decoder_bbox_lr_pair import CONTROL, THREE_QUARTER, treatment_commands
from tools.owod import run_decoder_bbox_seed_pair as replication
from tools.owod.run_schedule_pair import coverage
from tools.owod.screening_data import screening_files, select_images, subset, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata

import torch


SOURCE = pilot.ROOT / 'output' / f'm-instance-{THREE_QUARTER}'
OUTPUT = pilot.ROOT / 'output/m-instance-task2-six-epochs'


def compare_slices(checkpoint, args, manifest, slices):
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    variants = {'top50': (args.owod_known_class_ids, args.discovery_scoring, args.owod_max_detections)}
    rows = {}
    for name, annotation in slices.items():
        dataset, loader = make_loader(args.owod_image_root, annotation, args.batch_size, 2)
        ids = sorted(image['owod_source_id'] for image in dataset.coco.imgs.values())
        spec = dict(manifest['evaluation'], image_source_ids=ids, known_only_screening=True,
                    diagnostic_current_only=name == 'train')
        view = evaluate_views(model, loader, dataset.coco, variants, spec, args.owod_previous_class_ids,
                              args.owod_current_class_ids, 'cuda', include_current_stats=True)
        rows[name] = dict(metrics=view['top50']['metrics'], current_classes=view['top50']['current_class_stats'],
                          annotation_sha256=file_sha256(annotation), images=len(ids),
                          geometry=coverage(checkpoint, loader, include_ranks=True))
    return rows


def main(seed):
    if seed == 42:
        root, screen, source, output = study.OUTPUT, study.SCREEN, SOURCE, OUTPUT
        original = read_json(source / 'plan.json')
        reference_commands = original['commands']
        reference_folder = source
    else:
        root, screen = replication.OUTPUT, replication.SCREEN
        source = root / THREE_QUARTER
        output = OUTPUT.with_name(f'{OUTPUT.name}-seed{seed}')
        reference_folder = root
    reference_plan = read_json(root / 'plan.json')
    result = read_json(reference_folder / 'comparison.json')
    if seed == 43:
        reference_commands = reference_plan['arms'][THREE_QUARTER]
        original = dict(parent_sha256=result['task1']['checkpoint_sha256'],
                        gnn_sha256=result['common_gnn_sha256'], cache_sha256=result['common_cache_sha256'],
                        screen_sha256=read_json(source / 'increment/input.json')['screen_sha256'])
        if file_sha256(study.SCREEN) != reference_plan['source_screen_sha256']:
            raise ValueError('The original sampling screen has changed')
    _, base = study.study_commands(output=root, screen=screen, seed=seed, arms=(CONTROL,))
    standard = treatment_commands(base[CONTROL], source, .75)
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (result['status'] != 'complete' or result['seed'] != seed or reference_commands != standard
            or gpus != reference_plan['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != reference_plan['devices']
            or str(torch.__version__) != reference_plan['torch_version']):
        raise ValueError(f'Expected the completed seed-{seed} three-epoch arm and its original GPU configuration')
    parent = root / 'off/task1/checkpoint.pth'
    cache = root / 'retrieval' / f'instances-stage-1-{original["parent_sha256"]}.json'
    for path, digest in [(screen, original['screen_sha256']), (parent, original['parent_sha256']),
                         (root / 'off/gnn/gnn_stage0.pt', original['gnn_sha256']),
                         (cache, original['cache_sha256']), (pilot.MANIFEST, reference_plan['manifest_sha256']),
                         (pilot.WEIGHTS, reference_plan['weights_sha256'])]:
        if file_sha256(path) != digest:
            raise ValueError(f'Input differs from the three-epoch arm: {path}')

    commands = treatment_commands(base[CONTROL], output, .75)
    inc, ft = commands['increment'], commands['finetune']
    inc[inc.index('--epochs') + 1] = '6'
    # StepLR decays after the final update for both arms. Evaluate both at epoch three.
    inc[inc.index('--lr_drop') + 1] = '6'
    ft[ft.index('--discovery-pseudo-epoch-offset') + 1] = '6'
    identity = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    commands=commands, reference_plan_sha256=file_sha256(reference_folder / 'plan.json'),
                    source_result_sha256=file_sha256(reference_folder / 'comparison.json'), gpus=gpus, devices='0,1',
                    seed=seed, primary_endpoint='heldout finetune current_ap50', paper_comparable=False)
    pin(output / 'plan.json', identity)
    manifest, files = screening_files(pilot.MANIFEST, screen, 1)
    train = read_json(files['increment_train'])
    current = manifest['stages'][1]['classes']
    train_slice = output / 'train-slice.json'
    write_json(train_slice, subset(train, select_images(train, 128, current, 42), current))
    slices = {'train': train_slice, 'heldout': files['full_val']}
    if seed == 43:
        first = read_json(OUTPUT / 'comparison.json')
        original_slices = first['rows']['finetune']['six_epochs']['slices']
        if first['status'] != 'complete' or any(file_sha256(path) != original_slices[name]['annotation_sha256']
                                                 for name, path in slices.items()):
            raise ValueError('The replication requires the same diagnostic images and annotations as seed 42')
    pilot.run([inc[0], str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(pilot.MANIFEST.parent.parent)])
    rows = {}
    for phase in study.PHASES:
        new, args = study.run_job(commands[phase], original['screen_sha256'])
        ref = source / phase / 'checkpoint.pth'
        if file_sha256(ref) != read_json(source / phase / 'coverage.json')['checkpoint_sha256']:
            raise ValueError(f'Changed reference checkpoint: {ref}')
        rows[phase] = {}
        for key, path, config in [('six_epochs', new, args),
                                  ('three_epochs', ref, load_checkpoint_metadata(ref)['args'])]:
            rows[phase][key] = dict(author=pilot.metrics(path.parent, config.epochs - 1),
                                    slices=compare_slices(path, config, manifest, slices))
        write_json(output / 'progress.json', dict(status='partial', rows=rows))
    write_json(output / 'comparison.json', dict(status='complete', rows=rows, **identity))
    print('Comparison saved:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43), default=42)
    main(parser.parse_args().seed)
