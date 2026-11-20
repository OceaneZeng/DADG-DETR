"""Paired Task 2 image diversity experiment with the same optimizer and epoch budget."""
import os
import subprocess

import torch

from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, resolve_manifest_path
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_decoder_bbox_lr_pair import CONTROL, treatment_commands
from tools.owod.screening_data import screening_files, select_images, source_ids, subset, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = pilot.ROOT / 'output/m-instance-task2-exposure-pair'
SCREEN = OUTPUT / 'data/screening.json'
HELDOUT = holdout.OUTPUT / 'heldout.json'
MODES = ('repeat', 'alternate')
EXTRA_SEED = 45


def register_screen():
    manifest, first = screening_files(pilot.MANIFEST, study.SCREEN, 0)
    _, second = screening_files(pilot.MANIFEST, study.SCREEN, 1)
    screen = read_json(study.SCREEN)
    base = source_ids(read_json(second['increment_train']))
    evaluation = source_ids(read_json(HELDOUT))
    reserved = set(manifest['evaluation']['image_source_ids']) | set(screen['heldout_source_ids']) | evaluation
    prior = source_ids(read_json(first['increment_train'])) | source_ids(read_json(second['train']))
    if len(base) != pilot.TASK2_IMAGES or evaluation & (base | prior) or base & reserved:
        raise ValueError('Original training inputs overlap the evaluation or differ in size')
    stage = manifest['stages'][1]
    full = read_json(resolve_manifest_path(pilot.MANIFEST, stage['files']['increment_train']))
    extra = select_images(full, len(base), stage['classes'], EXTRA_SEED, reserved | prior | base)
    combined = subset(full, base | extra, stage['classes'])
    path = SCREEN.parent / 'increment_train.json'
    if path.exists() and read_json(path) != combined:
        raise ValueError('Registered expanded training inputs changed')
    write_json(path, combined)
    for record in screen['stages']:
        record['files'] = {k: str(study.SCREEN.parent / v) for k, v in record['files'].items()}
    screen['stages'][1]['files']['increment_train'] = str(path)
    screen['stages'][1]['sha256']['increment_train'] = file_sha256(path)
    screen.update(task2_train_images=len(base | extra), exposure_first_source_ids=sorted(base),
                  extra_selection_seed=EXTRA_SEED, independent_holdout_sha256=file_sha256(HELDOUT))
    if SCREEN.exists() and read_json(SCREEN) != screen:
        raise ValueError('Screen no longer matches the registered experiment')
    write_json(SCREEN, screen)
    return manifest


def main():
    task1, template = study.study_commands(output=OUTPUT, screen=SCREEN, arms=(CONTROL,))
    jobs = {}
    for mode in MODES:
        pair = treatment_commands(template[CONTROL], OUTPUT / mode, .75)
        for command in pair.values():
            command.extend(['--screen-exposure', mode])
        inc, ft = pair['increment'], pair['finetune']
        inc[inc.index('--epochs') + 1] = '6'
        inc[inc.index('--lr_drop') + 1] = '6'
        ft[ft.index('--discovery-pseudo-epoch-offset') + 1] = '6'
        jobs[mode] = pair
    ref = read_json(holdout.OUTPUT / 'comparison.json')
    ref_plan = read_json(holdout.OUTPUT / 'plan.json')
    if (ref['status'] != 'complete' or file_sha256(HELDOUT) != ref_plan['annotation_sha256']
            or file_sha256(holdout.OUTPUT / 'plan.json') != ref['plan_sha256']):
        raise ValueError('Expanded holdout differs from the completed duration probe')
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    original = read_json(study.OUTPUT / 'plan.json')
    if (devices != original['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != original['devices']
            or str(torch.__version__) != original['torch_version']):
        raise ValueError('Use the original two training GPUs and environment')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                manifest_sha256=file_sha256(pilot.MANIFEST), weights_sha256=file_sha256(pilot.WEIGHTS),
                source_screen_sha256=file_sha256(study.SCREEN), holdout_sha256=file_sha256(HELDOUT),
                task1=task1, jobs=jobs, seed=42, extra_selection_seed=EXTRA_SEED, devices=original['devices'],
                gpus=devices, torch_version=original['torch_version'], paper_comparable=False,
                primary_endpoint='1024-image holdout finetune Current AP50', increment_steps_per_arm=2304)
    pin(OUTPUT / 'plan.json', plan)
    manifest = register_screen()
    study.check_data_and_cuda()
    report = study.train_shared_parent_arms(task1, jobs, OUTPUT, SCREEN, 42)
    dataset, loader = make_loader(resolve_image_root(pilot.MANIFEST, manifest), HELDOUT, 2, 2)
    names = [img['owod_source_id'] for img in read_json(HELDOUT)['images']]
    spec = dict(manifest['evaluation'], image_source_ids=names, known_only_screening=True)
    stage = manifest['stages'][1]
    results = {}
    for mode in MODES:
        checkpoint = OUTPUT / mode / 'finetune/checkpoint.pth'
        model, _ = load_detector(load_checkpoint_metadata(checkpoint)['args'], checkpoint, torch.device('cuda'))
        view = evaluate_views(model, loader, dataset.coco, {'focal_top50': (stage['active_classes'], 'focal', 50)},
                              spec, manifest['stages'][0]['classes'], stage['classes'], 'cuda',
                              include_current_stats=True, print_freq=32)
        results[mode] = view['focal_top50']['metrics']
        write_json(OUTPUT / mode / 'holdout.json', dict(view=view, checkpoint_sha256=file_sha256(checkpoint),
                                                       holdout_sha256=plan['holdout_sha256']))
        del model
    delta = {k: 100 * (results['alternate'][k] - v) for k, v in results['repeat'].items()}
    report.update(status='complete', primary_endpoint=plan['primary_endpoint'], heldout=results,
                  alternate_minus_repeat_pp=delta, plan_sha256=file_sha256(OUTPUT / 'plan.json'))
    write_json(OUTPUT / 'comparison.json', report)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
