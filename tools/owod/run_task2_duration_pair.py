"""Train the six-epoch M Task 2 treatment against the completed three-epoch arm."""
import os
from pathlib import Path
import subprocess

from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_decoder_bbox_lr_pair import CONTROL, THREE_QUARTER, treatment_commands
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


def main():
    original = read_json(SOURCE / 'plan.json')
    result = read_json(SOURCE / 'comparison.json')
    _, base = study.study_commands(arms=(CONTROL,))
    standard = treatment_commands(base[CONTROL], SOURCE, .75)
    reference_plan = read_json(study.OUTPUT / 'plan.json')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (result['status'] != 'complete' or result['seed'] != 42 or original['commands'] != standard
            or gpus != reference_plan['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != reference_plan['devices']
            or str(torch.__version__) != reference_plan['torch_version']):
        raise ValueError('Expected the completed seed-42 three-epoch arm and its original GPU configuration')
    parent = study.OUTPUT / 'off/task1/checkpoint.pth'
    cache = study.OUTPUT / 'retrieval' / f'instances-stage-1-{original["parent_sha256"]}.json'
    for path, digest in [(study.SCREEN, original['screen_sha256']), (parent, original['parent_sha256']),
                         (study.OUTPUT / 'off/gnn/gnn_stage0.pt', original['gnn_sha256']),
                         (cache, original['cache_sha256']), (pilot.MANIFEST, reference_plan['manifest_sha256']),
                         (pilot.WEIGHTS, reference_plan['weights_sha256'])]:
        if file_sha256(path) != digest:
            raise ValueError(f'Input differs from the three-epoch arm: {path}')

    commands = treatment_commands(base[CONTROL], OUTPUT, .75)
    inc, ft = commands['increment'], commands['finetune']
    inc[inc.index('--epochs') + 1] = '6'
    # StepLR decays after the final update for both arms. Evaluate both at epoch three.
    inc[inc.index('--lr_drop') + 1] = '6'
    ft[ft.index('--discovery-pseudo-epoch-offset') + 1] = '6'
    identity = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    commands=commands, reference_plan_sha256=file_sha256(SOURCE / 'plan.json'),
                    source_result_sha256=file_sha256(SOURCE / 'comparison.json'), gpus=gpus, devices='0,1',
                    seed=42, primary_endpoint='heldout finetune current_ap50', paper_comparable=False)
    pin(OUTPUT / 'plan.json', identity)
    manifest, files = screening_files(pilot.MANIFEST, study.SCREEN, 1)
    train = read_json(files['increment_train'])
    current = manifest['stages'][1]['classes']
    train_slice = OUTPUT / 'train-slice.json'
    write_json(train_slice, subset(train, select_images(train, 128, current, 42), current))
    slices = {'train': train_slice, 'heldout': files['full_val']}
    pilot.run([inc[0], str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(pilot.MANIFEST.parent.parent)])
    rows = {}
    for phase in study.PHASES:
        new, args = study.run_job(commands[phase], original['screen_sha256'])
        ref = SOURCE / phase / 'checkpoint.pth'
        if file_sha256(ref) != read_json(SOURCE / phase / 'coverage.json')['checkpoint_sha256']:
            raise ValueError(f'Changed reference checkpoint: {ref}')
        rows[phase] = {}
        for key, path, config in [('six_epochs', new, args),
                                  ('three_epochs', ref, load_checkpoint_metadata(ref)['args'])]:
            rows[phase][key] = dict(author=pilot.metrics(path.parent, config.epochs - 1),
                                    slices=compare_slices(path, config, manifest, slices))
        write_json(OUTPUT / 'progress.json', dict(status='partial', rows=rows))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', rows=rows, **identity))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
