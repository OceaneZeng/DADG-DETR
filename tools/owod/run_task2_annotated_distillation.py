"""Compare released-label-supported distillation with the unchanged Task 2 finetune at two seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from tools.owod import probe_task2_distillation_selection as selection
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import screening_files, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata, validate_plan_arguments


OUTPUT = pilot.ROOT / 'output/m-instance-task2-annotated-distillation'


def register():
    source = read_json(selection.OUTPUT / 'plan.json')
    evidence = read_json(selection.OUTPUT / 'comparison.json')
    held_plan = read_json(holdout.OUTPUT / 'plan.json')
    held = read_json(holdout.OUTPUT / 'comparison.json')
    annotation = holdout.OUTPUT / 'heldout.json'
    if (evidence['status'] != 'complete' or evidence['plan_sha256'] != file_sha256(selection.OUTPUT / 'plan.json')
            or held['status'] != 'complete' or held['plan_sha256'] != file_sha256(holdout.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(pilot.MANIFEST)
            or held_plan['annotation_sha256'] != source['annotations']['holdout']
            or file_sha256(annotation) != held_plan['annotation_sha256']):
        raise ValueError('Changed diagnostic reference, manifest or held-out images')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    devices = os.environ['CUDA_VISIBLE_DEVICES']
    jobs = {}
    for seed in ('42', '43'):
        key = f'seed{seed}-6epochs'
        ref = held_plan['checkpoints'][key]
        checkpoint = Path(ref['path'])
        root = checkpoint.parent.parent
        recipe = read_json(root / 'plan.json')
        inputs = read_json(checkpoint.parent / 'input.json')
        original = inputs['command']
        teacher = source['teachers'][seed]
        if (file_sha256(checkpoint) != ref['sha256']
                or file_sha256(root / 'comparison.json') != ref['source_result_sha256']
                or recipe['commands']['finetune'] != original or recipe['seed'] != int(seed)
                or devices != recipe['devices'] or gpus != recipe['gpus'] or str(torch.__version__) != source['torch_version']
                or original[original.index('--pretrained') + 1] != teacher['path']
                or inputs['--pretrained'] != teacher['sha256'] or file_sha256(Path(teacher['path'])) != teacher['sha256']
                or file_sha256(Path(original[original.index('--graph-gnn-checkpoint') + 1])) != inputs['--graph-gnn-checkpoint']):
            raise ValueError(f'Original seed {seed} finetuning provenance differs')
        args = load_checkpoint_metadata(checkpoint)['args']
        validate_plan_arguments({'command': original}, args)
        screen = Path(args.owod_screening)
        _, files = screening_files(pilot.MANIFEST, screen, 1)
        if (file_sha256(files['train']) != source['annotations']['balanced']
                or inputs['screen_sha256'] != file_sha256(screen) or args.epochs != 1):
            raise ValueError('The paired finetunes require the original balanced training input and budget')
        arms = {}
        for name in ('annotated', 'control'):
            command = original.copy()
            command[command.index('--output_dir') + 1] = str(OUTPUT / f'seed{seed}' / name)
            if name == 'annotated':
                command.append('--screen-annotated-distill')
            arms[name] = command
        jobs[seed] = dict(commands=arms, screen_sha256=inputs['screen_sha256'], teacher=teacher,
                          historical_holdout=held['rows'][key], historical_checkpoint_sha256=ref['sha256'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                jobs=jobs, gpus=gpus, devices=devices, torch_version=str(torch.__version__),
                manifest_sha256=source['manifest_sha256'], annotations=source['annotations'],
                selection_result_sha256=file_sha256(selection.OUTPUT / 'comparison.json'),
                reference_holdout_plan_sha256=held['plan_sha256'], paper_comparable=False,
                primary_endpoint='1024-image finetune Previous AP50',
                criterion='Both seeds: Previous gain >= 0.5 pp, Current loss <= 0.5 pp and Known gain > 0')
    pin(OUTPUT / 'plan.json', plan)
    return plan, annotation


def main():
    plan, annotation = register()
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    pilot.run([sys.executable, str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(pilot.MANIFEST.parent.parent)])
    manifest = read_json(pilot.MANIFEST)
    images = [img['owod_source_id'] for img in read_json(annotation)['images']]
    spec = dict(manifest['evaluation'], image_source_ids=images, known_only_screening=True)
    dataset, loader = make_loader(resolve_image_root(pilot.MANIFEST, manifest), annotation, 2, 2)
    rows, deltas = {}, {}
    for seed, job in plan['jobs'].items():
        rows[seed] = {}
        for arm, command in job['commands'].items():
            checkpoint, args = study.run_job(command, job['screen_sha256'])
            report = holdout.evaluate_finetune_holdout(checkpoint, args, dataset, loader, spec, plan_hash)
            last = json.loads((checkpoint.parent / 'metrics.jsonl').read_text().splitlines()[-1])
            extra = last.get('train_distill_annotated_queries_unscaled', 0.)
            if last['epoch'] != 0 or (arm == 'annotated' and extra <= 0):
                raise ValueError(f'Missing finetune endpoint or annotated query signal: {checkpoint}')
            rows[seed][arm] = dict(heldout=report['view']['top50']['metrics'], checkpoint_sha256=report['checkpoint_sha256'],
                                   distill_queries=last['train_distill_queries'], extra_queries=extra)
            write_json(OUTPUT / 'progress.json', dict(status='partial', plan_sha256=plan_hash, rows=rows))
        baseline = rows[seed]['control']['heldout']
        deltas[seed] = {k: 100 * (rows[seed]['annotated']['heldout'][k] - v) for k, v in baseline.items()}
    passed = all(d['previous_ap50'] >= .5 and d['current_ap50'] >= -.5 and d['known_ap50'] > 0
                 for d in deltas.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows,
                                               annotated_minus_control_pp=deltas, passed_screen=passed))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
