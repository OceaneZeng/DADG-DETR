"""Compare finetune step sizes from the same S Task 2 increment checkpoint."""
import argparse
from ast import literal_eval
from hashlib import sha256
from math import isclose
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch

from tools.owod import run_instance_pilot as pilot
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task2_quality_timing import OUTPUT as TIMING, SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import evaluate
from tools.owod.run_schedule_pair import endpoint
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.s_task2_proposal_quality import quality_digest
from tools.owod.screening_data import write_json
from tools.owod.shared_task1 import validate_inputs


OUTPUT = SOURCE.with_name('s-instance-task2-finetune-half-lr')
REFERENCES = {
    'candidate': '9bb8ee7e3c07de8144c5ff833410cb9324e7da5527d5ba2ac25e26d3bc195a8f',
    'control': '93c29044b3ccc63b976d2f7964eb389f5d274f47daeedc2796eb3f53138de21d',
}


def prepare(output=OUTPUT, *, restore_offsets=False, double_old_boxes=False, double_current_positives=False):
    source = read_json(SOURCE/'plan.json')
    fitted = read_json(SOURCE/'comparison.json')
    timing = read_json(TIMING/'comparison.json')
    trace = read_json(TIMING/'plan.json')
    timing_sha = file_sha256(TIMING/'comparison.json')
    command = source['jobs']['finetune'].copy()

    def value(flag):
        return command[command.index(flag) + 1]

    parent = Path(value('--pretrained'))
    identity = validate_inputs(MANIFEST, value('--owod-screening'), value('--screen-branch-checkpoint'),
                               value('--graph-gnn-checkpoint'), value('--backbone-weights'), source['seed'])
    inc = fitted['phases']['increment']
    cache = Path(value('--instance-cache')) / f'instances-stage-1-{identity["checkpoint_sha256"]}.json'
    if (timing_sha != 'a41d16624350d8e24892197c42fddfdce52937eede89dd3cbccf05754f45b4b0'
            or timing['plan_sha256'] != file_sha256(TIMING/'plan.json') or timing['timing']['supported'] != ['finetune']
            or trace['source_training_sha256'] != file_sha256(SOURCE/'comparison.json')
            or fitted['plan_sha256'] != file_sha256(SOURCE/'plan.json') or fitted['follow_up']
            or file_sha256(parent) != inc['checkpoint_sha256'] or quality_digest(parent) != inc['quality_sha256']
            or read_json(parent.parent/'training_complete.json')['last_epoch'] != 2
            or read_json(SOURCE/'finetune/input.json')['command'] != command or identity != source['identity']
            or file_sha256(cache) != source['cache_sha256']):
        raise ValueError('Changed stage evidence, shared inputs or finetune predecessor')

    manifest = read_json(MANIFEST)
    annotation = Path(source['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (file_sha256(annotation) != source['annotation_sha256'] or len(order) != 1024 or len(set(order)) != len(order)
            or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or devices != source['gpus'] or str(torch.__version__) != source['torch_version']
            or os.environ['CUDA_VISIBLE_DEVICES'] != source['devices']):
        raise ValueError('Changed holdout or training environment')

    refs = {}
    for arm, digest in REFERENCES.items():
        path = SOURCE/f'{arm}-finetune-heldout.json'
        row = read_json(path)
        expected = fitted['phases']['finetune']['checkpoint_sha256'] if arm == 'candidate' else source['control']['finetune']['sha256']
        if (file_sha256(path) != digest or row['plan_sha256'] != fitted['plan_sha256']
                or row['checkpoint_sha256'] != expected):
            raise ValueError(f'Changed reference evaluation: {arm}')
        refs[arm] = row['views']
    command[command.index('--output_dir') + 1] = str(output/'finetune')
    for flag in ('--lr', '--lr_backbone'):
        command[command.index(flag) + 1] = str(float(value(flag)) * .5)
    plan = {k: source[k] for k in ('identity', 'cache_sha256', 'annotation', 'annotation_sha256', 'eval_order_sha256',
                                    'protocol', 'seed', 'paper_comparable', 'devices', 'gpus', 'per_gpu_batch_size',
                                    'global_batch_size', 'torch_version')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                source_timing_sha256=timing_sha, source_training_sha256=trace['source_training_sha256'],
                increment_sha256=inc['checkpoint_sha256'], frozen_quality_sha256=inc['quality_sha256'],
                references=REFERENCES, command=command, lr_multiplier=.5,
                criterion='Against both the same-parent full-rate finetune and the original duration control: '
                          'Current +1.00 pp or more, Previous no worse than -1.00 pp, Known improves; mean proxy '
                          'U-Recall does not drop, mean WI and summed A-OSE do not rise. Use only release-source holdout.')
    if restore_offsets or double_old_boxes:
        swap = OUTPUT.with_name('s-instance-task2-step-classifier')
        trace = read_json(swap/'comparison.json')
        swap_plan = read_json(swap/'plan.json')
        half = read_json(OUTPUT/'comparison.json')
        ref = read_json(swap/'half-half.json')
        base_config = read_json(OUTPUT/'finetune/run_config.json')
        if (file_sha256(swap/'comparison.json') != '42d7ddb06f0fb64db34e0049f8487ca569ece4991fdcf9c96375a58ea1248352'
                or trace['plan_sha256'] != file_sha256(swap/'plan.json') or trace['current_effects']['supported'] != ['rest']
                or swap_plan['source_sha256'] != file_sha256(OUTPUT/'comparison.json') or half['follow_up']
                or half['plan_sha256'] != file_sha256(OUTPUT/'plan.json') or swap_plan['increment_sha256'] != inc['checkpoint_sha256']
                or ref['plan_sha256'] != trace['plan_sha256'] or ref['body_checkpoint'] != half['checkpoint_sha256']
                or ref['classifier_checkpoint'] != half['checkpoint_sha256'] or ref['views']['released']['metrics'] != half['heldout']
                or base_config['lr_linear_proj_names'] != ['reference_points', 'sampling_offsets']
                or base_config['lr_linear_proj_mult'] != .1 or base_config['decoder_bbox_lr_mult'] != .75):
            raise ValueError('Changed classifier study, half-rate reference or optimizer groups')
        refs['half'] = ref['views']
        old_loss = refs['candidate']['raw_old_box_recall_iou50']['hits'] - refs['half']['raw_old_box_recall_iou50']['hits']
        if old_loss != 162 or '--lr_linear_proj_mult' in command:
            raise ValueError('Changed old coverage deficit or projection-rate control')
        plan.update(source_classifier_sha256=file_sha256(swap/'comparison.json'), half_reference_sha256=file_sha256(swap/'half-half.json'),
                    old_hits_to_recover=old_loss//2)
        if restore_offsets:
            command.extend(['--lr_linear_proj_mult', '0.2'])
            plan['intervention'] = ('Restore the decoder sampling-offset group to full rate; '
                                    'all other groups keep their half rate. Compare all groups against the recorded optimizer.')
        else:
            box = OUTPUT.with_name('s-instance-task2-step-box')
            evidence = read_json(box/'comparison.json')
            box_plan = read_json(box/'plan.json')
            box_sha = file_sha256(box/'comparison.json')
            if (box_sha != '7584b96b66a9ba868ac0a47247441d937de7819456998741efc8baa587733aab'
                    or evidence['plan_sha256'] != file_sha256(box/'plan.json') or evidence['old_hit_effects']['supported'] != ['rest']
                    or box_plan['increment_sha256'] != inc['checkpoint_sha256']
                    or box_plan['endpoints']['half']['sha256'] != half['checkpoint_sha256']
                    or box_plan['endpoints']['full']['sha256'] != fitted['phases']['finetune']['checkpoint_sha256']):
                raise ValueError('Changed frozen box-head evidence')
            command.append('--screen-double-old-box-loss')
            plan.update(source_box_sha256=box_sha, intervention='Double matched previous-class GT L1 and GIoU terms '
                        'at all supervised decoder layers during balanced finetuning. Keep the original global box '
                        'normalization and the half-rate optimizer, matching costs, classification and discovery terms.')
        plan['criterion'] += (' Separately test whether at least 81 of the 162 lost old raw box hits return '
                              'versus the half-rate endpoint. Report Current and FP-family differences against that endpoint.')
    if double_current_positives:
        ranking = OUTPUT.with_name('s-instance-task2-fixed-score-ranking')
        rank = read_json(ranking/'comparison.json')
        rank_plan = read_json(ranking/'plan.json')
        half = read_json(OUTPUT/'comparison.json')
        half_plan = read_json(OUTPUT/'plan.json')
        ref_path = OUTPUT.with_name('s-instance-task2-step-classifier')/'half-half.json'
        ref = read_json(ref_path)
        expected = half_plan['command'].copy()
        expected[expected.index('--output_dir') + 1] = str(output/'finetune')
        if (file_sha256(ranking/'comparison.json') != 'e240842e19cb96bbc5872a20d0f1a3e3365b110fa8d9e2298e06aef7af7fbbd8'
                or rank['plan_sha256'] != file_sha256(ranking/'plan.json') or rank['supported']
                or file_sha256(ref_path) != 'ef52e7ac8c349de2a020742109374d63caf0919bce4a6e6fb8c42d5153f5f979'
                or ref['body_checkpoint'] != ref['classifier_checkpoint'] or ref['body_checkpoint'] != half['checkpoint_sha256']
                or ref['views']['released']['metrics'] != half['heldout'] or half['plan_sha256'] != file_sha256(OUTPUT/'plan.json')
                or rank_plan['endpoints']['half']['sha256'] != half['checkpoint_sha256']
                or half_plan['identity'] != identity or half_plan['increment_sha256'] != plan['increment_sha256']
                or command != expected):
            raise ValueError('Changed ranking evidence, half-rate command or reference endpoint')
        command.append('--screen-double-current-positives')
        refs['half'] = ref['views']
        plan.update(source_ranking_sha256=file_sha256(ranking/'comparison.json'), half_reference_sha256=file_sha256(ref_path),
                    intervention='Double only the focal loss for matched GT positive current-class entries in balanced '
                                 'finetuning, including auxiliary layers. Keep negative class terms and original normalization.')
        plan['criterion'] += ' Require the same gates against the half-rate reference as well. Report changes by FP family.'
    pin(output/'plan.json', plan)
    return plan, manifest, order, refs


def compare(views, ref):
    delta = {k: views['released']['metrics'][k] - ref['released']['metrics'][k]
             for k in ('previous_ap50', 'current_ap50', 'known_ap50')}
    folds = {}
    for fold in ('proxy_even', 'proxy_odd'):
        new, old = views[fold]['metrics'], ref[fold]['metrics']
        if new['unknown_gt'] <= 0 or new['unknown_gt'] != old['unknown_gt']:
            raise ValueError(f'Different proxy objects: {fold}')
        folds[fold] = {k: new[k] - old[k] for k in ('u_recall', 'wi', 'a_ose')}
    known = delta['current_ap50'] >= .01 and delta['previous_ap50'] >= -.01 and delta['known_ap50'] > 0
    opened = (mean(f['u_recall'] for f in folds.values()) >= 0 and mean(f['wi'] for f in folds.values()) <= 0
              and sum(f['a_ose'] for f in folds.values()) <= 0)
    return dict(delta=delta, proxy_deltas=folds, known_pass=known, open_pass=opened)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--restore-offsets', action='store_true', help='Restore only the decoder sampling-offset rate')
    variants.add_argument('--double-old-boxes', action='store_true', help='Double only the released old-GT localization terms')
    variants.add_argument('--double-current-positives', action='store_true', help='Double matched new-class focal positives')
    options = parser.parse_args()
    output = OUTPUT
    if options.restore_offsets:
        output = OUTPUT.with_name('s-instance-task2-finetune-offset-rate')
    elif options.double_old_boxes:
        output = OUTPUT.with_name('s-instance-task2-finetune-old-box-loss')
    elif options.double_current_positives:
        output = OUTPUT.with_name('s-instance-task2-finetune-current-positive')
    print('Verifying S timing evidence and matched finetune inputs', flush=True)
    plan, manifest, order, refs = prepare(output, restore_offsets=options.restore_offsets, double_old_boxes=options.double_old_boxes,
                                         double_current_positives=options.double_current_positives)
    pilot.run([sys.executable, str(pilot.ROOT/'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 's-owodb', '--output-root', str(MANIFEST.parent.parent)])
    checkpoint, args = run_job(plan['command'], plan['identity']['screening_sha256'])
    adapter = args.graph_adapter_specs[-1]
    if (not isclose(args.lr, 2.5e-6) or not isclose(args.lr_backbone, 2.5e-7)
            or adapter['instances']['cache_sha256'] != plan['cache_sha256']
            or adapter['source_checkpoint_sha256'] != plan['identity']['checkpoint_sha256']
            or adapter['gnn_sha256'] != plan['identity']['gnn_sha256']
            or quality_digest(checkpoint) != plan['frozen_quality_sha256']):
        raise ValueError('Unexpected effective rates, adapter inputs or frozen quality head')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), 2, 2)
    views = evaluate('candidate-finetune', checkpoint, plan, ds, loader, manifest, order, proxies=True,
                     output=output, fp_sources='half' in refs)
    comparisons = {arm: compare(views, ref) for arm, ref in refs.items()}
    report = dict(
        status='complete', paper_comparable=False, plan_sha256=file_sha256(output/'plan.json'),
        checkpoint_sha256=file_sha256(checkpoint), author=endpoint(checkpoint.parent, 'finetune', epoch=0),
        heldout=views['released']['metrics'], comparisons=comparisons,
        follow_up=all(comparisons[arm]['known_pass'] and comparisons[arm]['open_pass']
                      for arm in (refs if options.double_current_positives else REFERENCES)))
    if 'half' in refs:
        from tools.owod.probe_s_task2_current_fp_sources import summarize

        before = optimizer_groups(OUTPUT/'finetune/train.log')
        after = optimizer_groups(output/'finetune/train.log')
        if options.restore_offsets:
            expected = [(rate * 2 if size == 394752 else rate, size) for rate, size in before]
            if (after != expected or sum(size == 394752 for _, size in before) != 1
                    or not any(size == 394752 and isclose(rate, 3.75e-7) for rate, size in after)):
                raise ValueError('Expected only the nonempty sampling-offset group to regain full rate')
        else:
            import json
            epoch = json.loads((output/'finetune/metrics.jsonl').read_text().splitlines()[-1])
            metric = 'current_positive_matches' if options.double_current_positives else 'old_box_matches'
            enabled = args.screen_double_current_positives if options.double_current_positives else args.screen_double_old_box_loss
            counts = {k: v for k, v in epoch.items() if k.startswith('train_'+metric) and k.endswith('_unscaled')}
            if (after != before or not enabled or len(counts) != args.dec_layers
                    or min(counts.values()) <= 0 or epoch['epoch'] != 0):
                raise ValueError('Expected nonempty weighted matches at every layer and the unchanged half-rate optimizer')
            report['current_positive_matches_per_rank_batch' if options.double_current_positives else 'old_matches_per_rank_batch'] = counts
        old = views['raw_old_box_recall_iou50']
        base = refs['half']['raw_old_box_recall_iou50']
        if old['gt'] != base['gt']:
            raise ValueError('Old coverage uses a different object population')
        fp = summarize(views['released'], views['released'])
        ref_fp = summarize(refs['half']['released'], refs['half']['released'])
        report.update(old_hits_gained=old['hits']-base['hits'], optimizer_groups=after,
                      fp_deltas_vs_half={k: row['fp']-ref_fp[k]['fp'] for k, row in fp.items()})
        if 'old_hits_to_recover' in plan:
            report['old_coverage_pass'] = report['old_hits_gained'] >= plan['old_hits_to_recover']
    write_json(output/'comparison.json', report)
    print('Finetune learning-rate comparison saved', flush=True)


def optimizer_groups(path):
    prefix = 'Optimizer groups (learning rate, parameter count):'
    lines = [line for line in path.read_text().splitlines() if line.startswith(prefix)]
    return [(rate, size) for rate, size in literal_eval(lines[-1][len(prefix):]) if size]


if __name__ == '__main__':
    main()
