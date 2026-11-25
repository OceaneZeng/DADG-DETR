"""Train the single local-saliency change on the locked Task 1 screen."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess

import torch
import numpy as np

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_local_saliency import OUTPUT as PREFLIGHT
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import TOP_KNOWN_SHA
from tools.owod.probe_task1_m_s_refinement_top_known import OUTPUT as REFERENCE, TopKnownPerQuery
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from models.owod_benchmark_eval import overlaps
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, OUTPUT as TRAINING, compare, prepare as prepare_refinement
from tools.owod.run_task1_m_s_released_small_extent import measure, size_group
from tools.owod.screening_data import write_json


OUTPUT = TRAINING.with_name('task1-m-s-refinement-saliency-training')
MATCHED_COUNT_OUTPUT = TRAINING.with_name('task1-m-s-refinement-saliency-matched-count')
QUALITY_OUTPUT = TRAINING.with_name('task1-m-s-refinement-small-box-quality')
DETACHED_OUTPUT = TRAINING.with_name('task1-m-s-refinement-detached-rank')
QUALITY_SHA = '8ac23ca0c1eb44c2b71ef8f73fd31f7f400585e5db13be76848ccd2182516292'
HOLDOUT_SHA = '15d7e2827d896d94d189e6655ed616d8f89fc3f8176d60bbb47865d181c149c1'
QUALITY_CHECKPOINTS = {
    'm-owodb': '9d5fe319f71c9ae8ecc445b024dca309293b2e45e2c10b6829e52b85581a6a04',
    's-owodb': '1cb87265dac80851d17f22d0e3cd4e92d170c686f7fa1f59a0f79c2c69de211f',
}
PREFLIGHT_SHA = '3be9cf68082bbc19224ce2123d7fea0a0ec808a1226d884fcf9a9c25be5c1056'
CAP_COMPARISON_SHA = '9a28ed0ade713fec7101ade0070aedb130341bf0b348e5b89f7f3c75b7b012ca'
GATE_FLAG = '--discovery-saliency-top-points'
COUNT_FLAG = '--discovery-saliency-match-count'
QUALITY_FLAG = '--screen-small-box-quality-rank'
ENCODER_PREFLIGHT_SHA = '4f0f41bab6bfbbc7377d3a64c5bf3482a57f11a5dbed89ff89014238b76f4406'
COMPETITION_SHA = 'b3801feb4573a0b6d9586d70f5479d8be541acea665678baaf5f76f974570f67'
MATCHED_COUNT_SHA = '8d8b5817191b7b491f52ff247eb87dbeb41f310b7ceb59ad4ebb45f37d038181'


def prepare(output, match_count):
    evidence = verified(PREFLIGHT / 'comparison.json', PREFLIGHT_SHA)
    if evidence['status'] != 'complete' or not evidence['passed']:
        raise ValueError('Both released training-source saliency checks must pass')
    preflight = verified(PREFLIGHT / 'plan.json', evidence['plan_sha256'])
    top = verified(REFERENCE / 'comparison.json', TOP_KNOWN_SHA)
    reference = verified(REFERENCE / 'plan.json', top['plan_sha256'])
    prior = verified(TRAINING / 'plan.json', reference['training_plan_sha256'])
    checked = prepare_refinement()
    if prior['jobs'] != checked['jobs'] or prior['gates'] != reference['gates']:
        raise ValueError('Prepared inputs differ from the locked Task 1 screen')
    cap_reports = None
    if match_count:
        cap = verified(OUTPUT / 'comparison.json', CAP_COMPARISON_SHA)
        cap_plan = verified(OUTPUT / 'plan.json', cap['plan_sha256'])
        if (cap['status'] != 'complete' or cap['passed'] or cap_plan['preflight_sha256'] != PREFLIGHT_SHA
                or cap_plan['gates'] != reference['gates']):
            raise ValueError('Expected the completed native-cap saliency comparison')
        cap_reports = cap['protocols']
    jobs = {}
    for protocol, source in checked['jobs'].items():
        command = list(source['command'])
        command[command.index('--output_dir') + 1] = str((output / protocol / 'increment').resolve())
        command.extend((GATE_FLAG, str(preflight['strongest_points'])))
        if match_count:
            expected = list(cap_plan['jobs'][protocol]['command'])
            expected[expected.index('--output_dir') + 1] = command[command.index('--output_dir') + 1]
            if command != expected:
                raise ValueError(f'Native-cap training command differs: {protocol}')
            command.append(COUNT_FLAG)
        control = reference['jobs'][protocol]['arms']['control']
        jobs[protocol] = dict(source, command=command, control_report=control['report'],
                              control_report_sha256=control['report_sha256'],
                              refinement_reference=top['protocols'][protocol])
        if cap_reports is not None:
            jobs[protocol]['cap_reference'] = cap_reports[protocol]
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                preflight_sha256=PREFLIGHT_SHA, training_plan_sha256=reference['training_plan_sha256'],
                top_known_sha256=TOP_KNOWN_SHA, jobs=jobs, limit=reference['limit'], batch_size=reference['batch_size'],
                device_ids=checked['device_ids'], devices=checked['devices'], epochs=checked['epochs'],
                seed=checked['seed'], strongest_points=preflight['strongest_points'],
                gates=reference['gates'], paper_comparable=False,
                start='Fresh shared DINO initialization with the original refinement architecture and schedule.',
                decision='Apply all original author-VOC screening gates on all four locked folds. Compare the score '
                         'and coverage differences to the refinement endpoint under the same TopKnown postprocessor.')
    if match_count:
        plan.update(match_count=True, cap_comparison_sha256=CAP_COMPARISON_SHA)
    return plan


def add_quality_arm(plan):
    from tools.owod.probe_task1_m_s_refinement_known_dedup import COMPETITION_OUTPUT

    preflight = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-encoder-rank-preflight')
    ranks = verified(preflight / 'comparison.json', ENCODER_PREFLIGHT_SHA)
    competition = verified(COMPETITION_OUTPUT / 'comparison.json', COMPETITION_SHA)
    matched = verified(MATCHED_COUNT_OUTPUT / 'comparison.json', MATCHED_COUNT_SHA)
    matched_plan = verified(MATCHED_COUNT_OUTPUT / 'plan.json', matched['plan_sha256'])
    if any(row['status'] != 'complete' for row in (ranks, competition, matched)):
        raise ValueError('Complete source diagnostics are required for the quality-rank arm')
    for protocol, job in plan['jobs'].items():
        command = job['command']
        prior = list(matched_plan['jobs'][protocol]['command'])
        prior[prior.index('--output_dir') + 1] = command[command.index('--output_dir') + 1]
        if command != prior:
            raise ValueError(f'Training command differs from matched-count control: {protocol}')
        command.append(QUALITY_FLAG)
        job['competition_reference'] = competition['protocols'][protocol]
    plan.update(small_box_quality_rank=True, encoder_rank_preflight_sha256=ENCODER_PREFLIGHT_SHA,
                competition_comparison_sha256=COMPETITION_SHA, matched_count_comparison_sha256=MATCHED_COUNT_SHA,
                decision='Apply all original gates with the locked known/unknown competition postprocessor. '
                         'Compare localization and VOC metrics to the matched-count endpoint under the same scoring.')


def prepare_detached():
    from tools.owod.screening_data import screening_files

    control = verified(QUALITY_OUTPUT / 'comparison.json', QUALITY_SHA)
    plan = verified(QUALITY_OUTPUT / 'plan.json', control['plan_sha256'])
    holdout = verified(TRAINING.with_name('task1-m-s-refinement-mined-seed-holdout') / 'comparison.json', HOLDOUT_SHA)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (control['status'] != 'complete' or holdout['status'] != 'complete' or holdout['passed']
            or not plan['small_box_quality_rank'] or devices != plan['devices']):
        raise ValueError('Expected the original quality-rank control, stopped frozen holdout and matched GPUs')
    for protocol, job in plan['jobs'].items():
        command = job['command']
        if QUALITY_FLAG not in command or '--pretrained' in command or '--resume' in command:
            raise ValueError(f'Expected a fresh quality-rank control: {protocol}')
        manifest = MANIFESTS[protocol]
        screen = Path(command[command.index('--owod-screening') + 1])
        backbone = Path(command[command.index('--backbone-weights') + 1])
        if (file_sha256(manifest) != job['manifest_sha256'] or file_sha256(screen) != job['screen_sha256']
                or file_sha256(backbone) != job['backbone_sha256']):
            raise ValueError(f'Changed paired training inputs: {protocol}')
        screening_files(manifest, screen, 0)
        checkpoint = QUALITY_OUTPUT / protocol / 'increment' / 'checkpoint.pth'
        if file_sha256(checkpoint) != QUALITY_CHECKPOINTS[protocol]:
            raise ValueError(f'Changed paired quality checkpoint: {protocol}')
        job['quality_reference_ap50'] = {fold: row['metrics']['known_ap50']
                                         for fold, row in control['protocols'][protocol].items()}
        job['quality_reference_sha256'] = QUALITY_CHECKPOINTS[protocol]
        for fold in job['folds'].values():
            verified_annotation(fold)
        command[command.index('--output_dir') + 1] = str((DETACHED_OUTPUT / protocol / 'increment').resolve())
        command.append('--screen-detach-rank-features')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                device_ids=os.environ['CUDA_VISIBLE_DEVICES'], detached_rank_features=True,
                quality_comparison_sha256=QUALITY_SHA, holdout_comparison_sha256=HOLDOUT_SHA,
                factor='Stop the auxiliary small-box ranking gradient at encoder memory. Retain the same quality '
                       'head and ranking loss. Native localization, quality and decoder supervision train jointly.',
                decision='Use fixed final checkpoints. Compare known AP50 and released small learned-query coverage '
                         'to the quality-rank control on each locked fold. Require nondecreasing Known AP50 '
                         'and learned-slot coverage on every fold, with a coverage gain on at least one fold '
                         'per protocol. Report withheld metrics and the original '
                         'four-fold gates as diagnostics. No label-guided reservation at inference.')
    return plan


def released_slot_observer(control, known, threshold, counts):
    def observe(dataset, samples, targets, candidate, detections, sizes, post):
        reference = control(samples.to('cuda'))
        for batch, target in enumerate(targets):
            height, width = target['orig_size'].tolist()
            truths = [np.asarray(a['voc_bbox']) for a in dataset.coco.imgToAnns[int(target['image_id'])]
                      if a['category_id'] in known and not a.get('difficult') and size_group(a) != '32_sq_plus']
            counts['released_small'] += len(truths)
            for name, outputs in (('control', reference), ('candidate', candidate)):
                boxes = outputs['pred_boxes'][batch]
                learned = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                active = outputs['geometry']['active'][batch]
                learned[:len(active)] = ~active
                coords = voc_coords(boxes[learned], width, height)
                counts[name] += sum((overlaps(truth, coords) > threshold).any().item() for truth in truths)
    return observe


def evaluate(checkpoint, protocol, job, plan, digest, output):
    checkpoint_sha = file_sha256(checkpoint)
    path = MANIFESTS[protocol]
    manifest = verified(path, job['manifest_sha256'])
    root = resolve_image_root(path, manifest)
    control = verified(Path(job['control_report']), job['control_report_sha256'])['folds']
    model, _ = load_detector(None, checkpoint, torch.device('cuda'))
    model.eval()
    post_type = TopKnownPerQuery
    if plan.get('small_box_quality_rank'):
        from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
        post_type = CompeteWithEmittedUnknown
    post = post_type(known_ids=manifest['stages'][0]['classes'], scoring='focal', max_detections=plan['limit'])
    quality_control = None
    if plan.get('detached_rank_features'):
        quality_control, _ = load_detector(None, QUALITY_OUTPUT / protocol / 'increment' / 'checkpoint.pth',
                                           torch.device('cuda'))
        quality_control.eval()
    folds = {}
    for fold, entry in job['folds'].items():
        dest = output / f'{protocol}-{fold}.json'
        if not dest.exists():
            annotations = verified_annotation(entry)
            print(f'Evaluating {protocol}/{fold} Task 1 endpoint', flush=True)
            counts = Counter()
            observer = (released_slot_observer(quality_control, manifest['stages'][0]['classes'],
                                               manifest['evaluation']['iou_threshold'], counts)
                        if quality_control is not None else None)
            report = measure(model, manifest, root, annotations, plan['batch_size'], plan['limit'],
                             observer=observer, postprocessor=post)
            outcome = compare(control[fold], report, plan['gates'])
            if observer is not None:
                reference_ap = job['quality_reference_ap50'][fold]
                if counts['released_small'] == 0:
                    raise ValueError(f'No released small targets: {protocol}/{fold}')
                outcome['released_check'] = dict(**counts, control_known_ap50=reference_ap,
                                                 candidate_known_ap50=report['metrics']['known_ap50'],
                                                 passed=counts['candidate'] >= counts['control']
                                                 and report['metrics']['known_ap50'] >= reference_ap)
            reference = job['refinement_reference'][fold]
            outcome.update(metrics=report['metrics'],
                           deltas_vs_refinement={key: report['metrics'][key] - before
                                                 for key, before in reference['metrics'].items()},
                           gate_deltas_vs_refinement={key: value - reference['deltas'][key]
                                                      for key, value in outcome['deltas'].items()})
            if 'cap_reference' in job:
                cap = job['cap_reference'][fold]
                outcome['gate_deltas_vs_cap'] = {key: value - cap['deltas'][key]
                                                 for key, value in outcome['deltas'].items()}
            if 'competition_reference' in job:
                matched = job['competition_reference'][fold]['deltas']
                outcome['gate_deltas_vs_same_scoring'] = {key: value - matched[key]
                                                          for key, value in outcome['deltas'].items()}
            write_json(dest, dict(plan_sha256=digest, checkpoint_sha256=checkpoint_sha, outcome=outcome, report=report))
        data = read_json(dest)
        if data['plan_sha256'] != digest or data['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError(f'Changed training plan or checkpoint: {dest}')
        folds[fold] = data['outcome']
    return folds


def verified_annotation(entry):
    path = Path(entry['path'])
    if file_sha256(path) != entry['sha256']:
        raise ValueError(f'Changed frozen validation annotations: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--match-count', action='store_true')
    modes.add_argument('--small-box-quality', action='store_true')
    modes.add_argument('--detached-rank', action='store_true')
    args = parser.parse_args()
    quality = args.small_box_quality or args.detached_rank
    match_count = args.match_count or quality
    output = MATCHED_COUNT_OUTPUT if match_count else OUTPUT
    if args.small_box_quality:
        output = QUALITY_OUTPUT
    if args.detached_rank:
        output = DETACHED_OUTPUT
        plan = prepare_detached()
    else:
        plan = prepare(output, match_count)
        if quality:
            add_quality_arm(plan)
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        print(f'Fitting {protocol} Task 1 {output.name}', flush=True)
        checkpoint, saved = run_job(job['command'], job['screen_sha256'])
        if (not saved.with_box_refine or saved.backbone_weights_sha256 != job['backbone_sha256']
                or saved.discovery_saliency_top_points != plan['strongest_points']
                or getattr(saved, 'discovery_saliency_match_count', False) != match_count
                or getattr(saved, 'screen_small_box_quality_rank', False) != quality
                or getattr(saved, 'screen_detach_rank_features', False) != args.detached_rank):
            raise ValueError(f'Unexpected training endpoint: {protocol}')
        protocols[protocol] = evaluate(checkpoint, protocol, job, plan, digest, output)
    passed = all(row['passed'] for folds in protocols.values() for row in folds.values())
    result = dict(status='complete', plan_sha256=digest, protocols=protocols, passed=passed)
    if args.detached_rank:
        result['diagnostic_four_fold_passed'] = passed
        checks = [[row['released_check'] for row in folds.values()] for folds in protocols.values()]
        result['passed'] = all(all(row['passed'] for row in pair)
                               and any(row['candidate'] > row['control'] for row in pair) for pair in checks)
    write_json(output / 'comparison.json', result)


if __name__ == '__main__':
    main()
