"""Compare matched, disabled and shuffled historical instance alignment at two training seeds."""
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import torch

from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import run_instance_pilot as pilot
from tools.owod import run_stronger_parent_study as study
from tools.owod import run_task2_annotated_distillation as finetune_study
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_schedule_pair import endpoint
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata, validate_plan_arguments


OUTPUT = pilot.ROOT / 'output/m-instance-task2-history-alignment'


def seed_plan(seed, ref, devices, gpus):
    increment = Path(ref['teacher']['path'])
    root = increment.parent.parent
    recipe = read_json(root / 'plan.json')
    template = recipe['commands']
    if (file_sha256(increment) != ref['teacher']['sha256'] or recipe['seed'] != int(seed)
            or recipe['devices'] != devices or recipe['gpus'] != gpus):
        raise ValueError(f'Changed seed {seed} reference weights or device configuration')
    for phase, command in template.items():
        inputs = read_json(root / phase / 'input.json')
        if command != inputs['command'] or inputs['screen_sha256'] != ref['screen_sha256']:
            raise ValueError(f'Changed reference command or screen: {root / phase}')
    args = load_checkpoint_metadata(increment)['args']
    validate_plan_arguments({'command': template['increment']}, args)
    spec = args.graph_adapter_specs[-1]
    parent_hash = file_sha256(Path(args.pretrained))
    gnn_hash = file_sha256(Path(args.graph_gnn_checkpoint))
    cache = Path(args.instance_cache) / f'instances-stage-1-{parent_hash}.json'
    cache_hash = file_sha256(cache)
    evidence = read_json(cache)
    records = evidence['data']['records']
    counts = Counter(str(r['label']) for r in records)
    if (args.instance_mode != 'off' or args.epochs != 6 or args.decoder_bbox_lr_mult != .75
            or parent_hash != spec['source_checkpoint_sha256'] or gnn_hash != spec['gnn_sha256']
            or cache_hash != spec['instances']['cache_sha256'] or file_sha256(Path(args.owod_screening)) != ref['screen_sha256']
            or evidence['identity']['screen'] != ref['screen_sha256'] or evidence['identity']['seed'] != int(seed)
            or len({(r['image'], r['obj'], r['label']) for r in records}) != len(records)
            or len({r['slot'] for r in records}) < 2):
        raise ValueError(f'Expected a common verified parent, GNN and nontrivial retrieval cache: {cache}')
    commands = {}
    for arm in pilot.ARMS:
        phases = {}
        for phase in study.PHASES:
            command = template[phase].copy()
            command[command.index('--instance-mode') + 1] = arm
            folder = OUTPUT / f'seed{seed}' / arm
            command[command.index('--output_dir') + 1] = str(folder / phase)
            if phase == 'finetune':
                command[command.index('--pretrained') + 1] = str(folder / 'increment/checkpoint.pth')
            phases[phase] = command
        commands[arm] = phases
    return dict(commands=commands, screen_sha256=ref['screen_sha256'], parent_sha256=parent_hash,
                gnn_sha256=gnn_hash, cache_path=str(cache), cache_sha256=cache_hash,
                reference_plan_sha256=file_sha256(root / 'plan.json'), accepted_by_class=dict(counts))


def main():
    source = read_json(finetune_study.OUTPUT / 'plan.json')
    last = read_json(finetune_study.OUTPUT / 'comparison.json')
    annotation = holdout.OUTPUT / 'heldout.json'
    if (last['status'] != 'complete' or last['plan_sha256'] != file_sha256(finetune_study.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(pilot.MANIFEST)
            or source['annotations']['holdout'] != file_sha256(annotation)
            or str(torch.__version__) != source['torch_version']):
        raise ValueError('The completed paired study, M manifest and unchanged holdout are required')
    devices = os.environ['CUDA_VISIBLE_DEVICES']
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    jobs = {seed: seed_plan(seed, ref, devices, gpus) for seed, ref in source['jobs'].items()}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                jobs=jobs, devices=devices, gpus=gpus, torch_version=str(torch.__version__), batch_per_gpu=2,
                manifest_sha256=source['manifest_sha256'], annotations=source['annotations'], paper_comparable=False,
                reference_plan_sha256=last['plan_sha256'], primary_endpoint='1024-image holdout finetune Current AP50',
                criterion='Both seeds: matched Current >= 0.5 pp above each control; Previous >= -0.5 pp and Known > 0 pp versus off')
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    pilot.run([sys.executable, str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 'm-owodb', '--output-root', str(pilot.MANIFEST.parent.parent)])
    manifest = read_json(pilot.MANIFEST)
    names = [im['owod_source_id'] for im in read_json(annotation)['images']]
    eval_spec = dict(manifest['evaluation'], image_source_ids=names, known_only_screening=True)
    dataset, loader = make_loader(resolve_image_root(pilot.MANIFEST, manifest), annotation, 2, 2)
    rows, differences = {}, {}
    for seed, job in jobs.items():
        rows[seed] = {}
        for arm, commands in job['commands'].items():
            row = rows[seed][arm] = {}
            for phase in study.PHASES:
                checkpoint, args = study.run_job(commands[phase], job['screen_sha256'])
                spec = args.graph_adapter_specs[-1]
                retrieval = spec['alignment_summary']
                metrics = endpoint(checkpoint.parent, phase, epoch=args.epochs - 1)
                expected = sum(job['accepted_by_class'].values())
                if (spec['instances']['cache_sha256'] != job['cache_sha256']
                        or spec['source_checkpoint_sha256'] != job['parent_sha256'] or spec['gnn_sha256'] != job['gnn_sha256']
                        or retrieval['accepted'] != expected or retrieval['applied'] != (0 if arm == 'off' else expected)
                        or (retrieval['shuffled_across_slots'] > 0) != (arm == 'shuffled')
                        or (phase == 'increment' and (metrics['mean_alignment_count_per_rank_batch'] > 0) != (arm != 'off'))):
                    raise ValueError(f'Unexpected training exposure or different retrieval inputs: {checkpoint}')
                row[phase] = dict(author=metrics, retrieval=retrieval)
                write_json(OUTPUT / 'progress.json', dict(status='partial', rows=rows, plan_sha256=plan_hash))
            report = holdout.evaluate_finetune_holdout(checkpoint, args, dataset, loader, eval_spec, plan_hash)
            view = report['view']['top50']
            row['heldout'] = view['metrics']
            row['finetune_sha256'] = report['checkpoint_sha256']
            row['current_ap_by_retrieval'] = {}
            for group, supported in [('with_match', True), ('without_match', False)]:
                items = [s['ap50'] for c, s in view['current_class_stats'].items()
                         if (c in job['accepted_by_class']) == supported and s['gt'] > 0]
                row['current_ap_by_retrieval'][group] = sum(items) / len(items) if items else None
            write_json(OUTPUT / 'progress.json', dict(status='partial', rows=rows, plan_sha256=plan_hash))
        matched = rows[seed]['matched']['heldout']
        differences[seed] = {arm: {k: 100 * (v - rows[seed][arm]['heldout'][k]) for k, v in matched.items()}
                             for arm in ('off', 'shuffled')}
    passed = all(min(d['off']['current_ap50'], d['shuffled']['current_ap50']) >= .5
                 and d['off']['previous_ap50'] >= -.5 and d['off']['known_ap50'] > 0 for d in differences.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows,
                                               matched_minus_control_pp=differences, passed_screen=passed))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
