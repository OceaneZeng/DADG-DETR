"""Replay a finest-level area cap at the original proposal scores and shortlist."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_geometry_stages import COHORTS, MANIFESTS, OUTPUT as GEOMETRY, stage_boxes
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = GEOMETRY.with_name('task1-m-s-tiny-extent-cap')
PARENT_PLAN = '85eb1388a7c6603e67cbd92da07f0ca8f38abc5c8401c56d810f5c5dd8b2fde3'
PARENT_RESULT = '7f64a192140adc0d8c094158f1d2caafe9718833bb970e9b3bbe222ba90220c8'
MAX_AREA = 16**2
STAGES = ('dense', 'shortlist')
SIZES = ('under_16_sq', '16_to_32_sq')


def capped_output(output, index, height, width):
    geometry = output['geometry']
    box_batch = geometry['boxes'].clone()
    sites = geometry['masks'][0][index].numel()
    sides = box_batch[index, :sites, 2:]
    area = sides.prod(dim=-1) * height * width
    factor = (MAX_AREA / area.clamp_min(1e-8)).sqrt().clamp(max=1.)
    sides.mul_(factor[:, None])
    return dict(output, geometry=dict(geometry, boxes=box_batch))


@torch.inference_mode()
def trace(model, loader, reference):
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        coco = loader.dataset.coco
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            anns = [ann for ann in coco.imgToAnns[image] if ann['id'] in reference]
            if not anns:
                continue
            h, w = target['orig_size'].tolist()
            arms = (stage_boxes(output, i), stage_boxes(capped_output(output, i, h, w), i))
            coords = [{key: voc_coords(arm[key], w, h) for key in STAGES} for arm in arms]
            for ann in anns:
                prior = reference[ann['id']]
                truth = np.asarray(ann['voc_bbox'])
                native, capped = [{key: bool((overlaps(truth, boxes[key]) > .5).any()) for key in STAGES}
                                  for boxes in coords]
                if prior['image_id'] != image or any(native[key] != prior['coverage'][key] for key in STAGES):
                    raise ValueError(f'Changed native proposal coverage for target {ann["id"]}')
                rows.append(dict(annotation_id=ann['id'], image_id=image, size=prior['size'],
                                 native=native, capped=capped))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(reference) or {row['annotation_id'] for row in rows} != set(reference):
        raise ValueError('Changed target set')
    return rows


def summarize(rows):
    output = {}
    for size in SIZES:
        group = [row for row in rows if row['size'] == size]
        output[size] = {}
        for stage in STAGES:
            hits = Counter((row['native'][stage], row['capped'][stage]) for row in group)
            both, gained, lost = hits[True, True], hits[False, True], hits[True, False]
            output[size][stage] = dict(targets=len(group), native=both+lost, capped=both+gained,
                                       gained=gained, lost=lost, delta_pp=100 * (gained-lost) / len(group))
    return output


def main():
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or gpu != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    geometry = verified(GEOMETRY / 'plan.json', PARENT_PLAN)
    result = verified(GEOMETRY / 'comparison.json', PARENT_RESULT)
    if (result['status'] != 'complete' or result['plan_sha256'] != PARENT_PLAN
            or result['shared_priority']['under_16_sq'] != 'dense'):
        raise ValueError('Expected the completed native geometry diagnosis')
    references = {label: file_sha256(GEOMETRY / f'{label}.json') for label in geometry['references']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, inputs=geometry['inputs'], references=references,
                parent_plan=PARENT_PLAN, parent_result=PARENT_RESULT,
                gpu=gpu, batch_size=geometry['batch_size'], max_area=MAX_AREA,
                minimum_dense_delta_pp=5., shortlist_rule='positive')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    slices = {}
    for protocol, spec in plan['inputs'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, spec['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        checkpoint = Path(spec['checkpoint']['path'])
        if file_sha256(checkpoint) != spec['checkpoint']['sha256']:
            raise ValueError(f'Changed checkpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        slices[protocol] = {}
        for fold, cohort in spec['slices'].items():
            label = f'{protocol}-{fold}'
            original = verified(GEOMETRY / f'{label}.json', references[label])
            if original['plan_sha256'] != PARENT_PLAN:
                raise ValueError(f'Changed parent reference: {label}')
            annotation = COHORTS / cohort['file']
            verified(annotation, cohort['sha256'])
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} native scores with capped geometry', flush=True)
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                reference = {row['annotation_id']: row for row in original['rows']}
                write_json(dest, dict(plan_sha256=digest, rows=trace(model, loader, reference)))
            report = read_json(dest)
            if report['plan_sha256'] != digest:
                raise ValueError(f'Changed area cap plan: {label}')
            summary = summarize(report['rows'])
            tiny = summary['under_16_sq']
            passed = tiny['dense']['delta_pp'] >= plan['minimum_dense_delta_pp'] and tiny['shortlist']['delta_pp'] > 0
            slices[protocol][fold] = dict(groups=summary, passed=passed)
        del model
    accepted = all(row['passed'] for fs in slices.values() for row in fs.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=slices, accepted=accepted))


if __name__ == '__main__':
    main()
