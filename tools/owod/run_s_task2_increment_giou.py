"""Test a single increment-stage training change on the S three-epoch duration control."""
import argparse
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch

from tools.owod import run_instance_pilot as pilot
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_current_phase import OUTPUT as PHASE_SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import CHECKPOINT, MANIFEST, OUTPUT as DURATION, SCREEN
from tools.owod.run_schedule_pair import endpoint
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.screening_data import screening_files, write_json
from tools.owod.shared_task1 import validate_inputs
from tools.owod.validate_discovery import load_checkpoint_metadata, validate_plan_arguments


OUTPUT = DURATION.with_name('s-instance-task2-increment-giou')
PHASES = ('increment', 'finetune')


def epoch_positive_counts(path, rank_batches):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r['epoch'] for r in rows] != [0, 1, 2]:
        raise ValueError('Expected the original three increment epochs')
    totals = [r['train_discovery_known_count_unscaled'] * rank_batches for r in rows]
    if any(not isclose(total, round(total), abs_tol=1e-6) for total in totals):
        raise ValueError('Epoch averages do not reconstruct whole matched targets')
    return [round(total) for total in totals]


def prepare(output=OUTPUT, *, variant='giou'):
    phase_plan = read_json(PHASE_SOURCE / 'plan.json')
    evidence = read_json(PHASE_SOURCE / 'comparison.json')
    duration_plan = read_json(DURATION / 'plan.json')
    duration = read_json(DURATION / 'comparison.json')
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (file_sha256(PHASE_SOURCE / 'comparison.json') != 'f6acd34d144dc50f2ddca107e250a2c7d36e9e35fbeb7988878e2f10d2173aea'
            or file_sha256(PHASE_SOURCE / 'plan.json') != evidence['plan_sha256'] or evidence['next_focus'] != 'increment'
            or file_sha256(DURATION / 'comparison.json') != phase_plan['sources'][DURATION.name]
            or file_sha256(DURATION / 'plan.json') != duration['plan_sha256'] or devices != duration_plan['gpus']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != duration_plan['devices'] or str(torch.__version__) != duration_plan['torch_version']):
        raise ValueError('Changed phase evidence, control, or two-GPU environment')
    identity = validate_inputs(MANIFEST, SCREEN, CHECKPOINT, DURATION/'off/gnn/gnn_stage0.pt', pilot.WEIGHTS, 42)
    _, screening = screening_files(MANIFEST, SCREEN, 1)
    cache = DURATION/'retrieval'/f'instances-stage-1-{identity["checkpoint_sha256"]}.json'
    if identity != duration['identity'] or file_sha256(cache) != duration['common_cache_sha256']:
        raise ValueError('Changed shared Task 1 parent, GNN, images or instance cache')
    annotation = Path(phase_plan['annotation'])
    if file_sha256(annotation) != phase_plan['annotation_sha256']:
        raise ValueError('Changed release-only validation set')
    jobs, controls = {}, {}
    for phase in PHASES:
        folder = DURATION/'3epochs'/phase
        ref = phase_plan['checkpoints'][f'3epochs-{phase}']
        baseline = duration_plan['jobs']['3epochs'][phase]
        if (file_sha256(Path(ref['path'])) != ref['sha256'] or read_json(folder/'input.json')['command'] != baseline
                or read_json(folder/'training_complete.json')['last_epoch'] != (2 if phase == 'increment' else 0)):
            raise ValueError(f'Changed historical control: {phase}')
        args = load_checkpoint_metadata(Path(ref['path']))['args']
        validate_plan_arguments({'command': baseline}, args)
        if (args.giou_loss_coef, args.bbox_loss_coef, args.set_cost_giou, args.batch_size, args.seed,
                args.class_embed_lr_mult) != (2, 5, 2, 2, 42, 1):
            raise ValueError('Unexpected control loss, assignment cost, batch or seed')
        cmd = list(baseline)
        cmd[cmd.index('--output_dir')+1] = str(output/phase)
        if variant == 'proposal-quality':
            cmd[cmd.index('--graph-localization') + 1] = 'decoder-bbox-increment-quality'
        elif phase == 'increment' and variant != 'tail-positives':
            increment_flags = {'classifier-lr': ['--class_embed_lr_mult', '2'], 'giou': ['--giou_loss_coef', '4'],
                               'no-random-crop': ['--no-random-crop']}
            cmd.extend(increment_flags[variant])
        if phase == 'finetune':
            cmd[cmd.index('--pretrained')+1] = str(output/'increment/checkpoint.pth')
        jobs[phase], controls[phase] = cmd, ref
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                protocol='s-owodb', paper_comparable=False, seed=42, gpus=devices, devices=duration_plan['devices'],
                torch_version=str(torch.__version__), global_batch_size=4, per_gpu_batch_size=2, identity=identity,
                cache_sha256=duration['common_cache_sha256'], control=controls, jobs=jobs,
                source_phase_result_sha256=file_sha256(PHASE_SOURCE/'comparison.json'),
                source_duration_result_sha256=file_sha256(DURATION/'comparison.json'), annotation=str(annotation),
                annotation_sha256=phase_plan['annotation_sha256'], eval_order_sha256=phase_plan['eval_order_sha256'],
                primary='Finetune native focal Current AP50 on the same release-only 1024-image S duration holdout.',
                criterion='Advance if Current gains at least 1.00 pp, Previous drops at most 1.00 pp and Known rises. '
                          'Also require mean masked-fold U-Recall not to fall, mean WI and summed A-OSE not to rise. '
                          'Record whether incremental raw Current box coverage improves. No author-test model selection.')
    if variant == 'tail-positives':
        from tools.owod.s_task2_tail_positive import configure_tail
        configure_tail(plan, output, screening['increment_train'])
    if variant == 'proposal-quality':
        from tools.owod.s_task2_proposal_quality import add_evidence
        add_evidence(plan, CHECKPOINT)
    if variant == 'no-random-crop':
        coverage = DURATION.with_name('s-instance-task2-increment-match-coverage')
        result = read_json(coverage/'comparison.json')
        coverage_plan = read_json(coverage/'plan.json')
        ref = Path(controls['increment']['path']).parent
        config = read_json(ref/'run_config.json')
        rank_batches = coverage_plan['train_images'] // config['batch_size']
        counts = epoch_positive_counts(ref/'metrics.jsonl', rank_batches)
        if (file_sha256(coverage/'comparison.json') != 'fc28a6a71ca8e1ca7effbe3fc1f5e49da82fa3eeec1ca0d5b1d787556962977d'
                or result['supported'] or result['plan_sha256'] != file_sha256(coverage/'plan.json')
                or coverage_plan['checkpoints']['3epochs']['sha256'] != controls['increment']['sha256']
                or coverage_plan['screening_sha256'] != identity['screening_sha256']
                or config['lightweight'] or config['no_random_crop'] or config['no_augmentation']
                or config['world_size'] != 2 or rank_batches != 768 or counts != [4240, 4232, 4276]):
            raise ValueError('Changed frozen coverage, training augmentation or matched-target exposure')
        plan.update(source_coverage_sha256=file_sha256(coverage/'comparison.json'),
                    reference_epoch_metrics_sha256=file_sha256(ref/'metrics.jsonl'), reference_positive_counts=counts,
                    eligible_targets=result['arms']['3epochs']['gt'], rank_batches_per_epoch=rank_batches,
                    intervention='Disable the random crop branch only in increment. Retain flip and multi-scale resize, '
                                 'original balanced finetuning transforms, author data, batch, optimizer steps and inference.')
        plan['criterion'] += (' Additionally require each no-crop epoch to retain all eligible matched targets and '
                              'the increment endpoint to cover more held-out new-class boxes than the control.')
    if variant == 'classifier-lr':
        matched = DURATION.with_name('s-instance-task2-matched-exposure')
        reports = {matched:'b0500f0ed19822b5dd4b5de2f3ed42372b38069fdf495aa119a44cc02dc97bb4',
                   OUTPUT:'1ba2488652817ac039d22f05165f0a345a681840b397b6558406423b6f695f1a'}
        for folder, digest in reports.items():
            if (file_sha256(folder/'comparison.json') != digest
                    or read_json(folder/'comparison.json')['plan_sha256'] != file_sha256(folder/'plan.json')):
                raise ValueError(f'Changed evidence for classifier comparison: {folder}')
        plan.update(source_results={f.name:h for f,h in reports.items()},
                    intervention='Increment current-class classifier rows LR x2; finetune head LR x1. '
                                 'Same three-epoch control, GIoU coefficient 2 in both phases.')
    pin(output/'plan.json', plan)
    return plan, evidence


def evaluate_heldout(model, dataset, loader, manifest, order, *, proxies, fp_sources=False, rank_observer=None):
    previous, current = (manifest['stages'][i]['classes'] for i in (0, 1))
    variants = {'released': (manifest['stages'][1]['active_classes'], 'focal', 50)}
    if proxies:
        for fold, visible in [('proxy_even', current[::2]), ('proxy_odd', current[1::2])]:
            variants[fold] = (previous+visible, 'focal', 50)
    spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_fp_sources=fp_sources)
    return evaluate_views(model, loader, dataset.coco, variants, spec, previous, current, 'cuda',
                          include_current_stats=True, print_freq=64, rank_observer=rank_observer)


def evaluate(name, checkpoint, plan, dataset, loader, manifest, order, *, proxies, output=OUTPUT, fp_sources=False):
    path = output/f'{name}-heldout.json'
    digest = file_sha256(checkpoint)
    plan_hash = file_sha256(output/'plan.json')
    if not path.exists():
        print(f'Evaluating {name} on 1024 held-out images', flush=True)
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        views = evaluate_heldout(model, dataset, loader, manifest, order, proxies=proxies, fp_sources=fp_sources)
        write_json(path, dict(checkpoint_sha256=digest, plan_sha256=plan_hash, views=views))
    result = read_json(path)
    if result['checkpoint_sha256'] != digest or result['plan_sha256'] != plan_hash:
        raise ValueError(f'Changed evaluation inputs: {path}')
    return result['views']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--tail-positives', dest='variant', action='store_const', const='tail-positives',
                         help='Double increment matched focal positives for the bottom frequency quartile')
    options.add_argument('--classifier-lr', dest='variant', action='store_const', const='classifier-lr',
                         help='Double only the increment classifier LR')
    options.add_argument('--proposal-quality', dest='variant', action='store_const', const='proposal-quality',
                         help='Train the existing proposal quality head during increment')
    options.add_argument('--no-random-crop', dest='variant', action='store_const', const='no-random-crop',
                         help='Keep whole-image multi-scale training in increment, restore the original crop in finetuning')
    parser.set_defaults(variant='giou')
    variant = parser.parse_args().variant
    output = OUTPUT.with_name('s-instance-task2-increment-' + variant)
    print('Verifying S duration controls, original cache and stage evidence', flush=True)
    plan, evidence = prepare(output, variant=variant)
    pilot.run([sys.executable, str(pilot.ROOT/'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 's-owodb', '--output-root', str(MANIFEST.parent.parent)])
    annotation = Path(plan['annotation'])
    manifest = read_json(MANIFEST)
    order = [row['owod_source_id'] for row in read_json(annotation)['images']]
    if len(order) != 1024 or sha256('\n'.join(order).encode()).hexdigest() != plan['eval_order_sha256']:
        raise ValueError('Changed holdout order')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    report = dict(status='partial', plan_sha256=file_sha256(output/'plan.json'), paper_comparable=False, phases={})
    for phase in PHASES:
        path, args = run_job(plan['jobs'][phase], plan['identity']['screening_sha256'])
        adapter = args.graph_adapter_specs[-1]
        if (adapter['source_checkpoint_sha256'] != plan['identity']['checkpoint_sha256']
                or adapter['gnn_sha256'] != plan['identity']['gnn_sha256']
                or adapter['instances']['cache_sha256'] != plan['cache_sha256']
                or args.giou_loss_coef != (4 if phase == 'increment' and variant == 'giou' else 2)
                or args.class_embed_lr_mult != (2 if phase == 'increment' and variant == 'classifier-lr' else 1)):
            raise ValueError('Different training inputs or objective')
        views = evaluate('candidate-'+phase, path, plan, ds, loader, manifest, order, proxies=phase=='finetune', output=output)
        stats = views['released']['metrics']
        reference = evidence['arms']['3epochs'][phase]
        raw = views['raw_current_box_recall_iou50']
        report['phases'][phase] = dict(checkpoint_sha256=file_sha256(path), heldout=stats, raw_current=raw,
            delta_raw_hits=raw['hits']-(reference['gt']-reference['no_box']), author=endpoint(path.parent, phase, epoch=args.epochs-1),
            delta={k:stats[k]-reference[k] for k in ('current_ap50','previous_ap50','known_ap50')})
        if variant == 'tail-positives':
            expected = plan['tail_class_ids'] if phase == 'increment' else None
            if args.screen_tail_positive_ids != expected or args.no_random_crop:
                raise ValueError('Different weighted classes or augmentation at endpoint')
            if phase == 'increment':
                epochs = [json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
                counts = [r['train_current_positive_matches_unscaled'] for r in epochs]
                if [r['epoch'] for r in epochs] != [0, 1, 2] or min(counts) <= 0:
                    raise ValueError('Expected matched weighted positives in all three increment epochs')
                report['weighted_matches_per_rank_batch'] = counts
        if variant == 'no-random-crop':
            if args.no_random_crop != (phase == 'increment') or args.no_augmentation or args.lightweight:
                raise ValueError('Trained endpoint used a different augmentation setting')
            if phase == 'increment':
                counts = epoch_positive_counts(path.parent/'metrics.jsonl', plan['rank_batches_per_epoch'])
                report['positive_counts'] = counts
                report['exposure_pass'] = all(n == plan['eligible_targets'] for n in counts)
        if variant == 'proposal-quality':
            from tools.owod.s_task2_proposal_quality import compare_shortlist, quality_digest
            head_digest = quality_digest(path)
            report['phases'][phase]['quality_sha256'] = head_digest
            if phase == 'increment':
                if head_digest == plan['parent_quality_sha256']:
                    raise ValueError('Increment training did not change the proposal quality head')
                report['proposal_coverage'] = compare_shortlist(path, output, loader, manifest, plan)
            elif head_digest != report['phases']['increment']['quality_sha256']:
                raise ValueError('Balanced finetuning changed the frozen proposal quality head')
        write_json(output/'progress.json', report)
    control = evaluate('control-finetune', Path(plan['control']['finetune']['path']), plan, ds, loader, manifest, order,
                       proxies=True, output=output)
    if any(control['released']['metrics'][k] != evidence['arms']['3epochs']['finetune'][k] for k in ('previous_ap50','current_ap50','known_ap50')):
        raise ValueError('Historical native finetune AP did not replay')
    delta = report['phases']['finetune']['delta']
    folds, known_pass, open_pass = paired_gates(views, control, delta)
    report.update(status='complete', proxy_deltas=folds, known_pass=known_pass, open_pass=open_pass, follow_up=known_pass and open_pass)
    if variant == 'tail-positives':
        from tools.owod.s_task2_tail_positive import gained_classes
        tail = plan['tail_class_ids']
        gained = gained_classes(views['released']['current_class_stats'], control['released']['current_class_stats'], tail)
        ann = Path(plan['independent_annotation'])
        independent_order = [im['owod_source_id'] for im in read_json(ann)['images']]
        if (sha256('\n'.join(independent_order).encode()).hexdigest() != plan['independent_order_sha256']
                or set(independent_order).intersection(order) or file_sha256(ann) != plan['independent_annotation_sha256']):
            raise ValueError('Independent held-out records or order changed')
        ds2, loader2 = make_loader(resolve_image_root(MANIFEST, manifest), ann, 2, 2)
        paired = {}
        for arm, ckpt in [('control', Path(plan['control']['finetune']['path'])), ('candidate', path)]:
            paired[arm] = evaluate(arm+'-finetune-independent', ckpt, plan, ds2, loader2, manifest,
                                   independent_order, proxies=True, output=output)
        ref, new = paired['control']['released'], paired['candidate']['released']
        original = read_json(output.with_name('s-instance-task2-crop-ap-independent')/'control-finetune.json')['view']
        if any(ref['metrics'][k] != original['metrics'][k] for k in delta):
            raise ValueError('Independent control AP did not replay')
        gaps = {k:new['metrics'][k]-ref['metrics'][k] for k in delta}
        folds2, known2, open2 = paired_gates(paired['candidate'], paired['control'], gaps)
        gained2 = gained_classes(new['current_class_stats'], ref['current_class_stats'], tail)
        report.update(independent=dict(delta=gaps, proxy_deltas=folds2, known_pass=known2, open_pass=open2),
                      gained_tail_classes=dict(original=sorted(gained), independent=sorted(gained2), both=sorted(gained & gained2)))
        report['follow_up'] = report['follow_up'] and known2 and open2 and len(gained & gained2) >= 2
    if variant == 'no-random-crop':
        report['coverage_pass'] = report['phases']['increment']['delta_raw_hits'] > 0
        report['follow_up'] = report['follow_up'] and report['exposure_pass'] and report['coverage_pass']
    if variant == 'proposal-quality':
        shortlist = report['proposal_coverage']['shortlist']
        report['shortlist_pass'] = shortlist['candidate'] > shortlist['control']
        report['follow_up'] = report['follow_up'] and report['shortlist_pass']
    write_json(output/'comparison.json', report)
    print('Increment comparison saved:', output/'comparison.json', flush=True)


def paired_gates(views, control, delta):
    folds = {}
    for fold in ('proxy_even', 'proxy_odd'):
        old, new = control[fold]['metrics'], views[fold]['metrics']
        if old['unknown_gt'] != new['unknown_gt'] or old['unknown_gt'] <= 0:
            raise ValueError('Invalid proxy fold')
        folds[fold] = {k:new[k]-old[k] for k in ('u_recall','a_ose','wi')}
    known_pass = delta['current_ap50'] >= .01 and delta['previous_ap50'] >= -.01 and delta['known_ap50'] > 0
    open_pass = (mean(f['u_recall'] for f in folds.values()) >= 0 and sum(f['a_ose'] for f in folds.values()) <= 0
                 and mean(f['wi'] for f in folds.values()) <= 0)
    return folds, known_pass, open_pass


if __name__ == '__main__':
    main()
