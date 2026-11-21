"""Attribute native Current false positives on the frozen S Task 2 release-only holdout."""
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import OUTPUT as GIOU, PHASE_SOURCE
from tools.owod.screening_data import write_json


OUTPUT = GIOU.with_name('s-instance-task2-current-fp-sources')
FAMILIES = ('duplicate', 'other_released_class', 'partial_same_class', 'unresolved')


def prepare():
    report = read_json(GIOU/'comparison.json')
    source = read_json(GIOU/'plan.json')
    if (file_sha256(GIOU/'comparison.json') != '1ba2488652817ac039d22f05165f0a345a681840b397b6558406423b6f695f1a'
            or report['plan_sha256'] != file_sha256(GIOU/'plan.json') or report['follow_up']
            or source['source_phase_result_sha256'] != file_sha256(PHASE_SOURCE/'comparison.json')
            or file_sha256(MANIFEST) != source['identity']['manifest_sha256']):
        raise ValueError('Changed GIoU evidence or manifest')
    annotation = Path(source['annotation'])
    data, manifest = read_json(annotation), read_json(MANIFEST)
    order = [im['owod_source_id'] for im in data['images']]
    known = manifest['stages'][1]['active_classes']
    if (file_sha256(annotation) != source['annotation_sha256'] or len(order) != 1024
            or len(set(order)) != len(order) or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or {a['category_id'] for a in data['annotations']} != set(known)
            or manifest['evaluation']['ap_mode'] != 'voc2007_11point'):
        raise ValueError('Changed holdout, order or protocol')
    endpoints = {}
    for arm in ('control-increment', 'control-finetune', 'candidate-finetune'):
        group, phase = arm.split('-')
        if group == 'control':
            entry = source['control'][phase]
            path, digest = Path(entry['path']), entry['sha256']
            reference = PHASE_SOURCE/f'3epochs-{phase}.json'
        else:
            path, digest = GIOU/phase/'checkpoint.pth', report['phases'][phase]['checkpoint_sha256']
            reference = GIOU/'candidate-finetune-heldout.json'
        if (file_sha256(path) != digest or read_json(reference)['checkpoint_sha256'] != digest
                or read_json(reference)['plan_sha256'] != file_sha256(reference.parent/'plan.json')
                or read_json(path.parent/'training_complete.json')['last_epoch'] != (2 if phase == 'increment' else 0)):
            raise ValueError(f'Incomplete or changed frozen endpoint: {arm}')
        endpoints[arm] = dict(path=str(path), sha256=digest, reference=str(reference),
                              reference_sha256=file_sha256(reference))
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                seed=42, protocol='s-owodb', paper_comparable=False, device=os.environ['CUDA_VISIBLE_DEVICES'],
                gpu=torch.cuda.get_device_name(), batch_size=2, torch_version=str(torch.__version__),
                annotation=str(annotation), annotation_sha256=source['annotation_sha256'],
                manifest_sha256=file_sha256(MANIFEST), eval_order_sha256=source['eval_order_sha256'],
                source_sha256=file_sha256(GIOU/'comparison.json'), endpoints=endpoints, families=list(FAMILIES),
                scoring='focal', max_detections=50, ap_mode='voc2007_11point', iou_partial=.1,
                criterion='Prioritize a source only if it has the unique largest fixed-TP macro AP gain '
                          'and that gain is at least 1.00 pp in both control phases. Report GIoU finetune separately.')
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order


def summarize(view, baseline):
    stats = view['current_class_stats']
    for key in ('previous_ap50', 'current_ap50', 'known_ap50'):
        if view['metrics'][key] != baseline['metrics'][key]:
            raise ValueError(f'Native AP replay differs: {key}')
    if view['composition'] != baseline['composition']:
        raise ValueError('Prediction composition differs')
    for label, row in stats.items():
        source = row['fp_sources']
        if (any(row[k] != baseline['current_class_stats'][label][k] for k in ('gt', 'tp', 'fp', 'ap50'))
                or sum(f['fp'] for f in source['families'].values()) != row['fp']
                or sum(source['confused_with'].values()) != source['families']['other_released_class']['fp']):
            raise ValueError(f'Unreconciled FP attribution: {label}')
    return {family: dict(fp=sum(row['fp_sources']['families'][family]['fp'] for row in stats.values()),
                         gain=mean(row['fp_sources']['families'][family]['ap50_without']-row['ap50']
                                   for row in stats.values())) for family in FAMILIES}


def main():
    plan, manifest, order = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), 2, 2)
    previous, stage = manifest['stages'][:2]
    spec = dict(manifest['evaluation'], image_source_ids=order, known_only_screening=True,
                diagnostic_current_fp_sources=True)
    summary = {}
    for arm, entry in plan['endpoints'].items():
        out = OUTPUT/f'{arm}.json'
        if not out.exists():
            print(f'Evaluating {arm} on 1,024 held-out images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            views = evaluate_views(model, loader, dataset.coco, {'native': (stage['active_classes'], 'focal', 50)},
                                   spec, previous['classes'], stage['classes'], 'cuda', include_current_stats=True,
                                   print_freq=64)
            write_json(out, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'], view=views['native']))
            del model
        result = read_json(out)
        if result['plan_sha256'] != digest or result['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Changed replay identity: {arm}')
        ref = read_json(Path(entry['reference']))
        baseline = ref['view']['native'] if arm.startswith('control-') else ref['views']['released']
        summary[arm] = summarize(result['view'], baseline)
    supported = [f for f in FAMILIES if all(
        summary[arm][f]['gain'] >= .01 and all(summary[arm][f]['gain'] > summary[arm][g]['gain']
                                               for g in FAMILIES if g != f)
        for arm in ('control-increment', 'control-finetune'))]
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              arms=summary, priority=supported[0] if supported else None))
    print('Current FP sources saved:', OUTPUT/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
