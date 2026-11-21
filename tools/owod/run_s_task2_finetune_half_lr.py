"""Test half the balanced-finetune learning rate from the same S Task 2 increment checkpoint."""
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


def prepare():
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
    command[command.index('--output_dir') + 1] = str(OUTPUT/'finetune')
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
    pin(OUTPUT/'plan.json', plan)
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
    print('Verifying S timing evidence and matched finetune inputs', flush=True)
    plan, manifest, order, refs = prepare()
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
    views = evaluate('candidate-finetune', checkpoint, plan, ds, loader, manifest, order, proxies=True, output=OUTPUT)
    comparisons = {arm: compare(views, ref) for arm, ref in refs.items()}
    write_json(OUTPUT/'comparison.json', dict(
        status='complete', paper_comparable=False, plan_sha256=file_sha256(OUTPUT/'plan.json'),
        checkpoint_sha256=file_sha256(checkpoint), author=endpoint(checkpoint.parent, 'finetune', epoch=0),
        heldout=views['released']['metrics'], comparisons=comparisons,
        follow_up=all(c['known_pass'] and c['open_pass'] for c in comparisons.values())))
    print('Finetune learning-rate comparison saved', flush=True)


if __name__ == '__main__':
    main()
