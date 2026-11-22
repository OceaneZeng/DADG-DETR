"""Replay a scale-dependent consistency gate at fixed native pseudo-label counts."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task1_view_consistency import (
    DATA, OUTPUT as SOURCE, PROTOCOL, SOURCE_REPORTS,
)
from tools.owod.probe_s_task1_pseudo_coverage import OUTPUT as COVERAGE
from tools.owod.probe_task1_discovery_gates import candidate_batches, frozen_miner
from tools.owod.probe_task1_paired_discovery import LOCKED, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, size_group
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-task1-consistency-budget')
SOURCE_PLAN = 'd195a3906b6f4207f35abebb5dec8c863f9d658ea9624279208f462e40e5a38c'
SOURCE_RESULT = 'afd453e319797a901072802105b3edb706b268c0dce59d6a039734330938317c'
SMALL_AREA = 32**2
SMALL_THRESHOLD = .25


@torch.inference_mode()
def replay(model, criterion, ema, dataset, loader, reference=None, *, audit_capacity=False):
    prior_rows = {row['annotation_id']: row for row in reference['rows']} if reference is not None else None
    prior_counts = {row['image_id']: row['queries']['selected'] for row in reference['images']} if reference is not None else None
    known = set(model.known_ids)
    rows, images = [], []
    for step, (targets, output, trace, mined) in enumerate(candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            image_id = int(target['image_id'])
            height, width = target['orig_size'].tolist()
            boxes = output['pred_boxes'][i]
            native = torch.where(mined['pseudo'][i])[0]
            quota = len(native)
            if prior_counts is not None and quota != prior_counts[image_id]:
                raise ValueError(f'Changed original budget on image {image_id}')
            gate = trace[i]
            small = boxes[:, 2:].prod(-1) * (height * width) < SMALL_AREA
            qualified = gate['eligible'] & gate['salient'] & gate['other_salient']
            relaxed = qualified & (gate['consistent'] | (small & (mined['consistency'][i] >= SMALL_THRESHOLD)))
            quality = mined['saliency'][i] * mined['consistency'][i] * mined['semantic_weights'][i]
            if 'geometry_quality' in output:
                quality = quality * output['geometry_quality'][i].detach()
            baseline = select_distinct_boxes(boxes, gate['confident'], quality, quota)
            if not torch.equal(baseline.sort().values, native):
                raise ValueError(f'Changed native ranking on image {image_id}')
            candidate = select_distinct_boxes(boxes, relaxed, quality, quota)
            if len(candidate) < quota:
                # Greedy deduplication may exhaust the expanded pool earlier.
                # Retain the original feasible set on that image.
                candidate = baseline
                fallback = True
            else:
                fallback = False
            native_ids, candidate_ids = native.tolist(), candidate.tolist()
            images.append(dict(image_id=image_id, native=native_ids, candidate=candidate_ids,
                               expanded=int((relaxed & ~gate['confident']).sum()), fallback=fallback))
            native_boxes = voc_coords(boxes[native], width, height)
            candidate_boxes = voc_coords(boxes[candidate], width, height)
            if audit_capacity:
                all_boxes = voc_coords(boxes, width, height)
                native_pool = gate['confident'].cpu().numpy()
                relaxed_pool = relaxed.cpu().numpy()
            for ann in dataset.coco.imgToAnns[image_id]:
                if ann.get('difficult'):
                    continue
                truth = np.asarray(ann['voc_bbox'])
                before = bool((overlaps(truth, native_boxes) > .5).any())
                after = bool((overlaps(truth, candidate_boxes) > .5).any())
                released = ann['category_id'] in known
                size = size_group(ann)
                if prior_rows is not None:
                    prior = prior_rows[ann['id']]
                    if (before, released, size) != (prior['coverage']['selected'], prior['released'], prior['size']):
                        raise ValueError(f'Changed original target coverage for {ann["id"]}')
                row = dict(annotation_id=ann['id'], released=released, size=size, native=before, candidate=after)
                if reference is None:
                    row['image_id'] = image_id
                if audit_capacity and not before and not released and size != '32_sq_plus':
                    correct = overlaps(truth, all_boxes) > .5
                    row['capacity'] = dict(raw=bool(correct.any()),
                                           native_pool=bool((correct & native_pool).any()),
                                           relaxed_pool=bool((correct & relaxed_pool).any()),
                                           positive_quota=bool(quota))
                rows.append(row)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    expected_rows = (prior_rows if prior_rows is not None else
                     {ann['id'] for ann in dataset.coco.anns.values() if not ann.get('difficult')})
    expected_images = prior_counts if prior_counts is not None else dataset.coco.imgs
    if ({row['annotation_id'] for row in rows} != set(expected_rows) or len(rows) != len(expected_rows)
            or {row['image_id'] for row in images} != set(expected_images) or len(images) != len(expected_images)):
        raise ValueError('Incomplete locked replay')
    return dict(rows=rows, images=images)


def summarize(report):
    groups = {role: Counter(dict(targets=0, native=0, candidate=0, gained=0, lost=0))
              for role in ('proxy', 'small_proxy', 'released')}
    for row in report['rows']:
        role = 'released' if row['released'] else 'proxy'
        labels = (role, 'small_proxy') if role == 'proxy' and row['size'] != '32_sq_plus' else (role,)
        for label in labels:
            before, after = row['native'], row['candidate']
            groups[label].update(targets=1, native=before, candidate=after, gained=after and not before,
                                 lost=before and not after)
    changes = [len(set(row['candidate']) - set(row['native'])) for row in report['images']]
    return dict(groups=groups, changed_images=sum(n > 0 for n in changes),
                replacements=sum(changes), fallback_images=sum(row['fallback'] for row in report['images']),
                native_budget=sum(len(row['native']) for row in report['images']),
                expanded_candidates=sum(row['expanded'] for row in report['images']))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE/'plan.json', SOURCE_PLAN)
    comparison = verified(SOURCE/'comparison.json', SOURCE_RESULT)
    checkpoint = prior['checkpoint']
    path = Path(checkpoint['path'])
    if (comparison['status'] != 'complete' or comparison['plan_sha256'] != SOURCE_PLAN
            or file_sha256(path) != checkpoint['sha256'] or prior['consistency_threshold'] != .5):
        raise ValueError('Changed agreement study')
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                parent_plan=SOURCE_PLAN, parent_comparison=SOURCE_RESULT, native_reports=SOURCE_REPORTS,
                checkpoint=checkpoint, slices=prior['inputs'], batch_size=prior['batch_size'],
                paper_comparable=False, small_area=SMALL_AREA, small_threshold=SMALL_THRESHOLD)
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    manifest_path = MANIFESTS[PROTOCOL]
    manifest = verified(manifest_path, LOCKED[PROTOCOL]['manifest_sha256'])
    root = resolve_image_root(manifest_path, manifest)
    model, criterion, ema = frozen_miner(path, manifest)
    results = {}
    for fold, spec in plan['slices'].items():
        annotation = DATA/spec['file']
        verified(annotation, spec['sha256'])
        reference = verified(COVERAGE/f'parent12-{fold}.json', SOURCE_REPORTS[fold])
        dest = OUTPUT/f'{fold}.json'
        if not dest.exists():
            ds, loader = make_loader(root, annotation, plan['batch_size'], 2)
            print(f'Evaluating {fold} with fixed per-image pseudo-label budgets', flush=True)
            write_json(dest, dict(plan_sha256=digest, **replay(model, criterion, ema, ds, loader, reference)))
        report = read_json(dest)
        if (report['plan_sha256'] != digest or
                any(len(r['native']) != len(r['candidate']) for r in report['images'])):
            raise ValueError(f'Unpaired report {fold}')
        results[fold] = summarize(report)
    accepted = all(x['groups']['small_proxy']['candidate'] > x['groups']['small_proxy']['native']
                   and x['groups']['proxy']['candidate'] >= x['groups']['proxy']['native']
                   and x['groups']['released']['candidate'] <= x['groups']['released']['native']
                   for x in results.values())
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=results, accepted=accepted))


if __name__ == '__main__':
    main()
