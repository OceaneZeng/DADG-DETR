"""Trace the failed S proxy WI gate at the same frozen operating point."""
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task2_dedup_open_set import AuditedRefill, OUTPUT as SOURCE, SOURCES, inputs
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/s-instance-task2-dedup-wi-sources'
SOURCE_COMPARISON = '8e98021a8b491bb4b0c8113953c8dd1c87b08d4cde237b2a6ac44ece26d36398'


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Use one RTX 3090 for the frozen attribution')
    arm, protocol = '6epochs', 's-owodb'
    manifest_path, source, arms, ap_mode = SOURCES[protocol]
    parent = read_json(SOURCE / 'comparison.json')
    source_plan = read_json(SOURCE / 'plan.json')
    reference = read_json(SOURCE / f'{protocol}-{arm}.json')
    spec = inputs(protocol, manifest_path, source, arms, ap_mode)
    checkpoint = spec['checkpoints'][arm]
    if (file_sha256(SOURCE / 'comparison.json') != SOURCE_COMPARISON or
            parent['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or
            parent['protocols'][protocol][arm]['follow_up'] or spec != source_plan['protocols'][protocol] or
            reference['checkpoint_sha256'] != checkpoint['sha256'] or reference['plan_sha256'] != parent['plan_sha256']):
        raise ValueError('Source evidence differs from the failed S endpoint')
    manifest = read_json(manifest_path)
    previous, current = (manifest['stages'][i]['classes'] for i in (0, 1))
    annotation = Path(spec['annotation'])
    heldout = read_json(annotation)
    order = [image['owod_source_id'] for image in heldout['images']]
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
                source_comparison_sha256=SOURCE_COMPARISON, source_plan_sha256=parent['plan_sha256'],
                source_endpoint_sha256=file_sha256(SOURCE / f'{protocol}-{arm}.json'), protocol=protocol,
                arm=arm, seed=42, paper_comparable=False, inputs=spec, device=os.environ['CUDA_VISIBLE_DEVICES'],
                batch_size=2, gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                max_detections=source_plan['max_detections'], old_pool=source_plan['old_pool'],
                nms_iou=source_plan['nms_iou'], wi_recall=manifest['evaluation']['wi_recall'],
                objective='Recover both archived fold metrics exactly and account for every WI numerator hit '
                          'by predicted class and highest-IoU masked GT class, and denominator by predicted class. '
                          'Separate numerator change from denominator change. End at attribution; keep wider gate closed.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    dataset, loader = make_loader(resolve_image_root(manifest_path, manifest), annotation, 2, 2)
    model, _ = load_detector(None, Path(checkpoint['path']), torch.device('cuda'))
    variants, processors, auditors = {}, {}, {}
    for fold in ('proxy_odd', 'proxy_even'):
        known = previous + spec['visible_current_ids'][fold]
        auditor = AuditedRefill(known, previous)
        auditors[fold] = auditor
        for mode, processor in [('native', auditor.native), ('dedup', auditor)]:
            name = f'{fold}_{mode}'
            variants[name] = known, 'focal', plan['max_detections']
            processors[name] = processor
    views = evaluate_views(model, loader, dataset.coco, variants,
                           dict(manifest['evaluation'], image_source_ids=order), previous, current, 'cuda',
                           postprocessors=processors, print_freq=64, include_wi_sources=True)
    if any(auditor.checked_images != len(order) for auditor in auditors.values()):
        raise ValueError('Incomplete per-image unknown checks')
    for name in variants:
        view = views[name]
        if view['metrics'] != reference['views'][name]['metrics'] or view['composition'] != reference['views'][name]['composition']:
            raise ValueError(f'Failed to replay {name}')
        rows = view['wi_sources'].values()
        num, den = sum(row['open'] for row in rows), sum(row['total'] for row in rows)
        if (num / max(1, den) != view['metrics']['wi'] or sum(row['full_open'] for row in rows) != view['metrics']['a_ose']
                or any(sum(row['by_proxy'].values()) != row['open'] for row in rows)):
            raise ValueError(f'WI or A-OSE source counts differ in {name}')
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
               plan_sha256=file_sha256(plan_path), labels={r['id']: r['name'] for r in heldout['categories']},
               previous=previous, views=views))
    print('WI attribution saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
