"""Find where native small-target localization first loses coverage."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_candidate_capacity import OUTPUT as CAPACITY
from tools.owod.probe_task1_m_s_consistency_validation import OUTPUT as COHORTS
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = CAPACITY.with_name('task1-m-s-geometry-stages')
PARENT_PLAN = 'e379584f4986a95d51b756591352d4c77dda666bc067bd64e2f42b73cb1d7b2a'
PARENT_RESULT = 'd885b06f10e9a00419ef99a03a229ae3f5f4a52f536fa666221500d3718686a8'
STAGES = ('dense', 'shortlist', 'selected', 'final')
SIZES = ('all_small', 'under_16_sq', '16_to_32_sq')


def small_reference(report):
    return {row['annotation_id']: row for row in report['rows']
            if not row['released'] and row['size'] in SIZES}


@torch.inference_mode()
def trace(model, loader, reference):
    rows = []
    coco = loader.dataset.coco
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        for i, target in enumerate(targets):
            image_id = int(target['image_id'])
            anns = [ann for ann in coco.imgToAnns[image_id] if ann['id'] in reference]
            if not anns:
                continue
            height, width = target['orig_size'].tolist()
            proposal_stages = stage_boxes(output, i)
            boxes = {key: voc_coords(proposal_stages[key], width, height) for key in STAGES}
            for ann in anns:
                prior = reference[ann['id']]
                truth = np.asarray(ann['voc_bbox'])
                flags = {key: bool((overlaps(truth, value) > .5).any()) for key, value in boxes.items()}
                expected_final = prior['native'] or prior['capacity']['raw']
                if prior['image_id'] != image_id or flags['final'] != expected_final:
                    raise ValueError(f'Changed final decoder coverage for target {ann["id"]}')
                first_gap = next((key for key in STAGES if not flags[key]), None)
                rows.append(dict(annotation_id=ann['id'], image_id=image_id, size=prior['size'],
                                 coverage=flags, first_gap=first_gap))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(reference) or {row['annotation_id'] for row in rows} != set(reference):
        raise ValueError('Changed small-target identities')
    return rows


def summarize(rows):
    groups = {size: dict(targets=0, covered=Counter(dict.fromkeys(STAGES, 0)),
                        final_miss_first_gap=Counter(dict.fromkeys(STAGES, 0)),
                        recovered_after_first_gap=Counter(dict.fromkeys(STAGES[:-1], 0)))
              for size in SIZES}
    for row in rows:
        for size in ('all_small', row['size']):
            group = groups[size]
            group['targets'] += 1
            group['covered'].update(row['coverage'])
            if not row['coverage']['final']:
                group['final_miss_first_gap'][row['first_gap']] += 1
            elif row['first_gap'] is not None:
                group['recovered_after_first_gap'][row['first_gap']] += 1
    return groups


def main():
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or gpu != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    parent = verified(CAPACITY / 'plan.json', PARENT_PLAN)
    result = verified(CAPACITY / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete' or result['plan_sha256'] != PARENT_PLAN or result['rank_headroom']:
        raise ValueError('Expected the completed candidate-capacity diagnosis')
    references = {label: file_sha256(CAPACITY / f'{label}.json') for label in parent['frozen_run_files']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, parent_plan=PARENT_PLAN, parent_result=PARENT_RESULT,
                references=references, inputs=parent['inputs'], batch_size=parent['batch_size'], gpu=gpu,
                coverage_iou_strict=.5, shared_majority_min_folds=3)
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
            source = verified(CAPACITY / f'{label}.json', references[label])
            if source['plan_sha256'] != PARENT_PLAN:
                raise ValueError(f'Changed capacity reference: {label}')
            annotation = COHORTS / cohort['file']
            verified(annotation, cohort['sha256'])
            destination = OUTPUT / f'{label}.json'
            if not destination.exists():
                print(f'Evaluating {protocol}/{fold} proposal and decoder geometry', flush=True)
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                write_json(destination, dict(plan_sha256=digest, rows=trace(model, loader, small_reference(source))))
            report = read_json(destination)
            if report['plan_sha256'] != digest:
                raise ValueError(f'Changed geometry-stage run plan: {label}')
            slices[protocol][fold] = summarize(report['rows'])
        del model
    majority = {size: Counter(dict.fromkeys(STAGES, 0)) for size in SIZES}
    for fs in slices.values():
        for fold in fs.values():
            for size, group in fold.items():
                gaps = group['final_miss_first_gap']
                majority[size].update({key: 2 * count > gaps.total() for key, count in gaps.items()})
    priority = {size: next((key for key, n in counts.items() if n >= plan['shared_majority_min_folds']), None)
                for size, counts in majority.items()}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=slices, majority_folds=majority, shared_priority=priority))


if __name__ == '__main__':
    main()
