"""Trace native proposal coverage for released S Task 2 classes at two frozen endpoints."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_unknown_transition import OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-proposal-stages')
SOURCE_DIGEST = '7d038ea0db94cce9c94223970c9b113553148e5b7e1d5d1c98707b4f441fb17c'
REFERENCES = {'increment': '632894a5c4f71fd60bd1886208b911854bad72c970f4099cb7718b2a9a62a49a',
              'finetune': 'faca3cd4e7e463bbceb5187ef07c0d525bb4b03d3a23881196b1b8d84ab5aedc'}


def prepare():
    src = read_json(SOURCE / 'plan.json')
    result = read_json(SOURCE / 'comparison.json')
    annotation = Path(src['annotation'])
    manifest = read_json(MANIFEST)
    data = read_json(annotation)
    order = [im['owod_source_id'] for im in data['images']]
    if (file_sha256(SOURCE / 'comparison.json') != SOURCE_DIGEST or result['prioritize_classification']
            or result['plan_sha256'] != file_sha256(SOURCE / 'plan.json')
            or file_sha256(MANIFEST) != src['manifest_sha256'] or file_sha256(annotation) != src['annotation_sha256']
            or sha256('\n'.join(order).encode()).hexdigest() != src['eval_order_sha256']
            or len(set(order)) != 1024 or len(order) != 1024 or set(order) & set(manifest['evaluation']['image_source_ids'])
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != src['gpu']
            or os.environ['CUDA_VISIBLE_DEVICES'] != src['device'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed transition evidence, data identity or replay environment')
    endpoints = {}
    for phase, digest in REFERENCES.items():
        ref_path = SOURCE / f'control-{phase}.json'
        ref = read_json(ref_path)
        entry = src['endpoints']['control-' + phase]
        if (file_sha256(ref_path) != digest or ref['plan_sha256'] != result['plan_sha256']
                or ref['checkpoint_sha256'] != entry['sha256'] or file_sha256(Path(entry['path'])) != entry['sha256']):
            raise ValueError(f'Changed frozen endpoint: {phase}')
        endpoints[phase] = entry
    plan = {k: src[k] for k in ('annotation', 'annotation_sha256', 'manifest_sha256', 'eval_order_sha256',
                                'protocol', 'paper_comparable', 'seed', 'device', 'gpu', 'batch_size', 'torch_version')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                endpoints=endpoints, source_result_sha256=SOURCE_DIGEST, references=REFERENCES,
                criterion='Prioritize a proposal-stage investigation only if the same disjoint group accounts '
                          'for strictly more than half of all final-uncovered Current GT at both control endpoints. '
                          'Report all groups by original box area and verify exact final coverage identities. '
                          'This trace selects no score threshold and makes no AP gain claim.')
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


@torch.inference_mode()
def trace(model, loader, annotations, threshold):
    hits = {}
    seen = set()
    model.eval()
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        for index, target in enumerate(targets):
            image_id = int(target['image_id'].item())
            if image_id in seen:
                raise ValueError('Repeated image')
            seen.add(image_id)
            anns = [a for a in loader.dataset.coco.imgToAnns[image_id] if a['id'] in annotations]
            h, w = target['orig_size'].tolist()
            for stage, boxes in stage_boxes(output, index).items():
                stage_hits = hits.setdefault(stage, set())
                if not anns:
                    continue
                pixels = box_cxcywh_to_xyxy(boxes) * boxes.new_tensor([w, h, w, h])
                # VOC stores one-based minima and rounds all four corners to one decimal.
                voc = pixels.cpu().numpy().astype(np.float64)
                voc[:, :2] += 1
                np.round(voc, decimals=1, out=voc)
                for ann in anns:
                    if np.any(overlaps(np.asarray(ann['voc_bbox']), voc) > threshold):
                        stage_hits.add(ann['id'])
        if step % 64 == 0 or step == len(loader):
            print(f'Traced {step}/{len(loader)} batches, {len(seen)} images', flush=True)
    if seen != set(loader.dataset.ids):
        raise ValueError('Incomplete release-only image replay')
    return {stage: sorted(ids) for stage, ids in hits.items()}


def summarize(trace_ids, annotations, reference):
    hits = {stage: set(ids) for stage, ids in trace_ids.items()}
    if (hits['final'] != set(reference['view']['raw']) or not hits['selected'] <= hits['shortlist'] <= hits['dense']
            or any(not ids <= annotations.keys() for ids in hits.values())):
        raise ValueError('Final VOC identities or nested proposal coverage differ')
    missing = annotations.keys() - hits['final']
    groups = dict(no_dense=missing-hits['dense'], outside_shortlist=missing & (hits['dense']-hits['shortlist']),
                  outside_selected=missing & (hits['shortlist']-hits['selected']), selected_then_missing=missing & hits['selected'])
    sizes = {size: {i for i, a in annotations.items() if lo <= a['area'] < hi}
             for size, lo, hi in [('small', 0, 32**2), ('medium', 32**2, 96**2), ('large', 96**2, float('inf'))]}
    sizes['all'] = set(annotations)
    return {size: dict(gt=len(ids), final_missing=len(ids & missing),
                       stage_covered={k: len(ids & v) for k, v in hits.items()},
                       missing_groups={k: len(ids & v) for k, v in groups.items()}) for size, ids in sizes.items()}


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), plan['batch_size'], 2)
    current = set(manifest['stages'][1]['classes'])
    annotations = {i: a for i, a in ds.coco.anns.items() if a['category_id'] in current and not a.get('difficult', 0)}
    phases = {}
    for phase, entry in plan['endpoints'].items():
        path = OUTPUT / f'{phase}.json'
        reference = read_json(SOURCE / f'control-{phase}.json')
        if not path.is_file():
            print(f'Tracing {phase} on {len(ds)} images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            ids = trace(model, loader, annotations, manifest['evaluation']['iou_threshold'])
            phases[phase] = summarize(ids, annotations, reference)
            write_json(path, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'], hits=ids))
            del model
        else:
            saved = read_json(path)
            if saved['plan_sha256'] != digest or saved['checkpoint_sha256'] != entry['sha256']:
                raise ValueError(f'Incompatible stage trace: {phase}')
            phases[phase] = summarize(saved['hits'], annotations, reference)
    totals = [rows['all'] for rows in phases.values()]
    priority = [g for g in totals[0]['missing_groups']
                if all(2*r['missing_groups'][g] > r['final_missing'] for r in totals)]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                                phases=phases, priority=priority))
    print('Current proposal-stage comparison saved', flush=True)


if __name__ == '__main__':
    main()
