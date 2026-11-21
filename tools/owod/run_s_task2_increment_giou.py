"""Test a single increase in incremental matched-box GIoU weight on the S duration control."""
from hashlib import sha256
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


def prepare():
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
    screening_files(MANIFEST, SCREEN, 1)
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
        if (args.giou_loss_coef, args.bbox_loss_coef, args.set_cost_giou, args.batch_size, args.seed) != (2, 5, 2, 2, 42):
            raise ValueError('Unexpected control loss, assignment cost, batch or seed')
        cmd = list(baseline)
        cmd[cmd.index('--output_dir')+1] = str(OUTPUT/phase)
        if phase == 'increment':
            cmd.extend(['--giou_loss_coef', '4'])
        else:
            cmd[cmd.index('--pretrained')+1] = str(OUTPUT/'increment/checkpoint.pth')
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
    pin(OUTPUT/'plan.json', plan)
    return plan, evidence


def evaluate(name, checkpoint, plan, dataset, loader, manifest, order, *, proxies):
    path = OUTPUT/f'{name}-heldout.json'
    digest = file_sha256(checkpoint)
    plan_hash = file_sha256(OUTPUT/'plan.json')
    if not path.exists():
        print(f'Evaluating {name} on 1024 held-out images', flush=True)
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        previous, current = (manifest['stages'][i]['classes'] for i in (0,1))
        variants = {'released': (manifest['stages'][1]['active_classes'], 'focal', 50)}
        if proxies:
            for fold, visible in [('proxy_even', current[::2]), ('proxy_odd', current[1::2])]:
                variants[fold] = (previous+visible, 'focal', 50)
        views = evaluate_views(model, loader, dataset.coco, variants, dict(manifest['evaluation'], image_source_ids=order),
                               previous, current, 'cuda', include_current_stats=True, print_freq=64)
        write_json(path, dict(checkpoint_sha256=digest, plan_sha256=plan_hash, views=views))
    result = read_json(path)
    if result['checkpoint_sha256'] != digest or result['plan_sha256'] != plan_hash:
        raise ValueError(f'Changed evaluation inputs: {path}')
    return result['views']


def main():
    print('Verifying S duration controls, original cache and stage evidence', flush=True)
    plan, evidence = prepare()
    pilot.run([sys.executable, str(pilot.ROOT/'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 's-owodb', '--output-root', str(MANIFEST.parent.parent)])
    annotation = Path(plan['annotation'])
    manifest = read_json(MANIFEST)
    order = [row['owod_source_id'] for row in read_json(annotation)['images']]
    if len(order) != 1024 or sha256('\n'.join(order).encode()).hexdigest() != plan['eval_order_sha256']:
        raise ValueError('Changed holdout order')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    report = dict(status='partial', plan_sha256=file_sha256(OUTPUT/'plan.json'), paper_comparable=False, phases={})
    for phase in PHASES:
        path, args = run_job(plan['jobs'][phase], plan['identity']['screening_sha256'])
        adapter = args.graph_adapter_specs[-1]
        if (adapter['source_checkpoint_sha256'] != plan['identity']['checkpoint_sha256']
                or adapter['gnn_sha256'] != plan['identity']['gnn_sha256']
                or adapter['instances']['cache_sha256'] != plan['cache_sha256']
                or args.giou_loss_coef != (4 if phase == 'increment' else 2)):
            raise ValueError('Different training inputs or objective')
        views = evaluate('candidate-'+phase, path, plan, ds, loader, manifest, order, proxies=phase=='finetune')
        stats = views['released']['metrics']
        reference = evidence['arms']['3epochs'][phase]
        raw = views['raw_current_box_recall_iou50']
        report['phases'][phase] = dict(checkpoint_sha256=file_sha256(path), heldout=stats, raw_current=raw,
            delta_raw_hits=raw['hits']-(reference['gt']-reference['no_box']), author=endpoint(path.parent, phase, epoch=args.epochs-1),
            delta={k:stats[k]-reference[k] for k in ('current_ap50','previous_ap50','known_ap50')})
        write_json(OUTPUT/'progress.json', report)
    control = evaluate('control-finetune', Path(plan['control']['finetune']['path']), plan, ds, loader, manifest, order, proxies=True)
    if any(control['released']['metrics'][k] != evidence['arms']['3epochs']['finetune'][k] for k in ('previous_ap50','current_ap50','known_ap50')):
        raise ValueError('Historical native finetune AP did not replay')
    folds = {}
    for fold in ('proxy_even', 'proxy_odd'):
        old, new = control[fold]['metrics'], views[fold]['metrics']
        if old['unknown_gt'] != new['unknown_gt'] or old['unknown_gt'] <= 0:
            raise ValueError('Invalid proxy fold')
        folds[fold] = {k:new[k]-old[k] for k in ('u_recall','a_ose','wi')}
    delta = report['phases']['finetune']['delta']
    known_pass = delta['current_ap50'] >= .01 and delta['previous_ap50'] >= -.01 and delta['known_ap50'] > 0
    open_pass = (mean(f['u_recall'] for f in folds.values()) >= 0 and sum(f['a_ose'] for f in folds.values()) <= 0
                 and mean(f['wi'] for f in folds.values()) <= 0)
    report.update(status='complete', proxy_deltas=folds, known_pass=known_pass, open_pass=open_pass, follow_up=known_pass and open_pass)
    write_json(OUTPUT/'comparison.json', report)
    print('GIoU comparison saved:', OUTPUT/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
