"""Screen the same-class refill rule on short chains using the formal model settings."""

import os
from pathlib import Path
import subprocess
import sys

from pycocotools.coco import COCO
import torch

from main import get_args_parser as training_parser
from models.graph_local.discovery import build_discovery_postprocessor
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_unknown_refill_independent import (
    MANIFESTS, OVERLAP_INDEPENDENT_RESULT, OVERLAP_OUTPUT, summarize_refill,
)
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_benchmark import build_plan, get_args_parser
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT, WEIGHTS, run
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.strict_setting import configure_run


OUTPUT = OVERLAP_OUTPUT.with_name('task2-m-s-formal-refill-screen')
COHORT_ROOT = OVERLAP_OUTPUT / 'cohorts'
INCREMENT_EPOCHS = 3
FINETUNE_EPOCHS = 1
DEVICE = torch.device('cuda')


def arguments(manifest, path, screen, gpu_ids):
    return ['--manifest', str(manifest), '--output-dir', str(path),
            '--recipe', 'ow-detr-paper', '--method', 'discovery',
            '--instance-mode', 'matched', '--graph-routing', 'hard',
            '--graph-localization', 'frozen', '--discovery-scoring', 'focal',
            '--backbone-weights', str(WEIGHTS), '--gpus', gpu_ids, '--nproc-per-node', '2',
            '--batch-size', '2', '--num-workers', '4', '--activation-checkpointing', '--seed', '42',
            '--owod-screening', str(screen), '--screen-increment-epochs', str(INCREMENT_EPOCHS),
            '--screen-finetune-epochs', str(FINETUNE_EPOCHS), '--screen-eval-interval', str(INCREMENT_EPOCHS),
            '--task2-protected-refill', '--task2-refill-same-class-overlap']


def prepare():
    reference = verified(OVERLAP_OUTPUT / 'comparison.json', OVERLAP_INDEPENDENT_RESULT)
    evidence = verified(OVERLAP_OUTPUT / 'plan.json', reference['plan_sha256'])
    cohorts = verified(COHORT_ROOT / 'plan.json', evidence['cohort_plan'])
    if not reference['accepted'] or evidence['min_same_class_iou'] != .70 or evidence['limit'] != 50:
        raise ValueError('Expected the passing same-class refill reference')
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpu_ids.split(',')) != 2 or gpu_names != ['NVIDIA GeForce RTX 3090'] * 2:
        raise ValueError('Use two visible RTX 3090s')
    jobs = {}
    for protocol, info in cohorts['protocols'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, info['manifest_sha256'])
        config = verified(Path(info['endpoints']['finetune']['path']).with_name('run_config.json'),
                          info['endpoints']['finetune']['config_sha256'])
        screen = Path(config['owod_screening'])
        identity = verified(screen, info['screen_sha256'])
        excluded = set(manifest['evaluation']['image_source_ids']) | set(identity['heldout_source_ids'])
        train_counts = []
        for stage in (0, 1):
            _, files = screening_files(path, screen, stage)
            loaded = {key: source_ids(read_json(data)) for key, data in files.items()}
            for names in loaded.values():
                excluded.update(names)
            train_counts.append(len(loaded['increment_train']))
        if train_counts != [12288, 1536]:
            raise ValueError(f'Changed training budget: {protocol}')
        arm_dir = OUTPUT / protocol
        options = arguments(path, arm_dir, screen, gpu_ids)
        argv = get_args_parser().parse_args(options)
        benchmark = build_plan(argv)
        if (benchmark.get('paper_comparable') is not False
                or [(r['stage'], r['phase'], r.get('epochs')) for r in benchmark['runs']] !=
                [(0,'increment',INCREMENT_EPOCHS), (0,'calibration',None),
                 (1,'increment',INCREMENT_EPOCHS), (1,'finetune',FINETUNE_EPOCHS)]):
            raise ValueError(f'Unexpected diagnostic phase chain: {protocol}')
        folds, seen = {}, set()
        for fold, cohort in cohorts['cohorts'].items():
            label = f'{protocol}-{fold}'
            images = set(cohort['order'])
            if len(images) != len(cohort['order']) or images & (excluded | seen):
                raise ValueError(f'Fold has duplicate or training/official test images: {label}')
            seen.update(images)
            released = cohort['by_protocol'][protocol]
            full = evidence['reports'][label]
            pairs = verified(COHORT_ROOT / cohort['paired_file'], cohort['paired_sha256'])
            released_rows = verified(COHORT_ROOT / released['file'], released['sha256'])
            full_rows = verified(OVERLAP_OUTPUT / full['file'], full['sha256'])
            if source_ids(released_rows) != images or source_ids(full_rows) != images:
                raise ValueError(f'Changed fold image provenance: {label}')
            folds[fold] = dict(released=released, full=full, order=cohort['order'],
                               shared=sorted({pair['annotation_ids'][protocol] for pair in pairs}))
        jobs[protocol] = dict(options=options, benchmark=benchmark, folds=folds, train_counts=train_counts,
                              previous=info['previous'], current=info['current'])
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                cohort_plan_sha256=evidence['cohort_plan'], reference_sha256=OVERLAP_INDEPENDENT_RESULT,
                jobs=jobs, gpu_ids=gpu_ids, device_names=gpu_names, seed=42, batch_size=2,
                limit=evidence['limit'], min_same_class_iou=evidence['min_same_class_iou'],
                protected_group='current', minimum_unknown_recall_gain=evidence['minimum_unknown_recall_gain'],
                refill_ap_tolerance=evidence['refill_ap_tolerance'], paper_comparable=False)
    pin(OUTPUT / 'plan.json', plan)
    return plan


def evaluate(protocol, job, directory, plan, plan_sha):
    path = MANIFESTS[protocol]
    manifest = verified(path, job['benchmark']['manifest_sha256'])
    root = resolve_image_root(path, manifest)
    checkpoint = directory / 'checkpoint.pth'
    checkpoint_sha = file_sha256(checkpoint)
    marker = read_json(directory / 'training_complete.json')
    if marker['last_epoch'] != FINETUNE_EPOCHS - 1:
        raise ValueError(f'Missing final finetune endpoint: {protocol}')
    command = job['benchmark']['runs'][-1]['command']
    start = command.index(str(ROOT / 'main.py')) + 1
    args = training_parser().parse_args(command[start:])
    configure_run(args)
    post = build_discovery_postprocessor(args)
    if post.min_same_class_iou != plan['min_same_class_iou'] or list(post.protected_ids) != job['current']:
        raise ValueError(f'Changed configured refill matching: {protocol}')
    model, _ = load_detector(None, checkpoint, DEVICE)
    model.eval()
    summaries = {}
    for fold, entry in job['folds'].items():
        label = f'{protocol}-{fold}'
        resume = OUTPUT / f'{label}.json'
        identity = dict(plan_sha256=plan_sha, checkpoint_sha256=checkpoint_sha)
        if not resume.exists():
            released = COHORT_ROOT / entry['released']['file']
            scored = OVERLAP_OUTPUT / entry['full']['file']
            if (file_sha256(released) != entry['released']['sha256']
                    or file_sha256(scored) != entry['full']['sha256']):
                raise ValueError(f'Changed diagnostic evaluation annotations: {label}')
            dataset, loader = make_loader(root, released, plan['batch_size'], 2)
            evaluation = dict(manifest['evaluation'], image_source_ids=entry['order'])
            print(f'Evaluating {protocol}/{fold} native and same-class refill', flush=True)
            report = measure(model, dataset, loader, job['previous'] + job['current'], job['previous'],
                             job['current'], evaluation, plan['limit'], evaluation_coco=COCO(str(scored)),
                             extra_posts=dict(unknown_refill=post), audit_query_counts=True,
                             include_perturbed=False, audit_previous_matches=True)
            write_json(resume, dict(**identity, summary=summarize_refill(report, set(entry['shared']), plan),
                                    report=report))
        result = read_json(resume)
        if any(result[key] != value for key, value in identity.items()):
            raise ValueError(f'Changed crossover evaluation: {label}')
        summaries[fold] = result['summary']
    return summaries


def main():
    plan = prepare()
    plan_sha = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        print(f'Training {protocol} formal-compatible short chain', flush=True)
        run([sys.executable, str(ROOT / 'tools/owod/run_benchmark.py'), *job['options'], '--resume'])
        directory = Path(job['benchmark']['runs'][-1]['output_dir'])
        results[protocol] = evaluate(protocol, job, directory, plan, plan_sha)
    write_json(OUTPUT / 'comparison.json',
               dict(status='complete', plan_sha256=plan_sha, protocols=results,
                    accepted=all(summary['passed'] for folds in results.values() for summary in folds.values())))


if __name__ == '__main__':
    main()
