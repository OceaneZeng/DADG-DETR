"""Compare area-capped boxes at a fixed per-image shortlist budget."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_geometry_stages import COHORTS, MANIFESTS
from tools.owod.probe_task1_m_s_tiny_extent_cap import MAX_AREA, OUTPUT as CAP, capped_output
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import size_group
from tools.owod.screening_data import write_json


OUTPUT = CAP.with_name('task1-m-s-tiny-shortlist')
PARENT_PLAN = '3e21f6b3843f91e6beee126cbc19fc6d735e885d840240e7b13bcd16cc658309'
PARENT_RESULT = 'c9a0a8693ecc2b46060ba096cb62e48d00a3f12bcfd4b814faba05b94c21bb06'
ARMS = ('native', 'same_cell', 'capped')
GROUPS = ('under_16_sq', '16_to_32_sq', '32_sq_plus', 'released')
MIN_DELTA_PP = 1.
TAIL_DIVISOR = 20


def shortlist(module, output, i, height, width):
    geo = output['geometry']
    ids = torch.where(geo['valid'][i])[0]
    count = min(len(ids), geo['chosen'].shape[1] * module.shortlist_multiplier)
    native = ids[geo['logits'][i, ids].topk(count).indices]
    finest = geo['masks'][0][i].numel()
    boxes = geo['boxes'][i]
    candidate = ids[ids < finest]
    area = boxes[candidate, 2:].prod(dim=-1) * height * width
    candidate = candidate[(area > MAX_AREA) & ~torch.isin(candidate, native)]
    capped = capped_output(output, i, height, width)['geometry']['boxes'][i]
    scores = module.logits(geo['memory'][i, candidate], capped[candidate])
    replaced = min(count // TAIL_DIVISOR, len(candidate))
    added = candidate[scores.topk(replaced).indices]
    retained = native[:count - replaced]
    same_cell = torch.cat((boxes[retained], boxes[added]))
    mixed = torch.cat((boxes[retained], capped[added]))
    if len(same_cell) != count or len(mixed) != count:
        raise ValueError('Changed image shortlist budget')
    return (boxes[native], same_cell, mixed), count, added.tolist()


@torch.inference_mode()
def trace(model, loader, reference):
    module = model.detr.transformer.instance_proposals
    known = set(model.known_ids)
    coco = loader.dataset.coco
    rows, images = [], []
    observed_reference = set()
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        for i, target in enumerate(targets):
            image = int(target['image_id'])
            height, width = target['orig_size'].tolist()
            arms, budget, added = shortlist(module, output, i, height, width)
            coords = [voc_coords(boxes, width, height) for boxes in arms]
            images.append(dict(image_id=image, budget=budget, added=added))
            for ann in coco.imgToAnns[image]:
                if ann.get('difficult'):
                    continue
                truth = np.asarray(ann['voc_bbox'])
                hits = {key: bool((overlaps(truth, boxes) > .5).any()) for key, boxes in zip(ARMS, coords)}
                group = 'released' if ann['category_id'] in known else size_group(ann)
                if group in GROUPS[:2]:
                    prior = reference[ann['id']]
                    if group != prior['size'] or hits['native'] != prior['native']['shortlist']:
                        raise ValueError(f'Changed native shortlist for target {ann["id"]}')
                    observed_reference.add(ann['id'])
                rows.append(dict(annotation_id=ann['id'], image_id=image, group=group, **hits))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed_reference != set(reference) or len(images) != len(coco.imgs):
        raise ValueError('Changed cohort coverage')
    return dict(rows=rows, images=images)


def summarize(report):
    counts = {name: Counter(dict.fromkeys(('targets', *ARMS, 'gained', 'lost'), 0)) for name in GROUPS}
    for row in report['rows']:
        counts[row['group']].update(targets=1, **{key: row[key] for key in ARMS},
                                    gained=row['capped'] and not row['native'],
                                    lost=row['native'] and not row['capped'])
    return counts


def passes(groups):
    tiny = groups['under_16_sq']
    return (100 * (tiny['capped'] - tiny['native']) >= MIN_DELTA_PP * tiny['targets']
            and tiny['capped'] > tiny['same_cell']
            and all(groups[key]['capped'] >= groups[key]['native'] for key in ('16_to_32_sq', 'released')))


def main():
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or gpu != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    parent = verified(CAP / 'plan.json', PARENT_PLAN)
    result = verified(CAP / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete' or result['plan_sha256'] != PARENT_PLAN or result['accepted']:
        raise ValueError('Expected the completed original shortlist replay')
    references = {label: file_sha256(CAP / f'{label}.json') for label in parent['references']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, inputs=parent['inputs'], references=references,
                parent_plan=PARENT_PLAN, parent_result=PARENT_RESULT, max_area=MAX_AREA,
                gpu=gpu, batch_size=parent['batch_size'], tail_divisor=TAIL_DIVISOR, minimum_tiny_delta_pp=MIN_DELTA_PP)
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
        if set(model.known_ids) != set(manifest['stages'][0]['classes']):
            raise ValueError(f'Changed known classes: {protocol}')
        slices[protocol] = {}
        for fold, cohort in spec['slices'].items():
            label = f'{protocol}-{fold}'
            prior = verified(CAP / f'{label}.json', references[label])
            if prior['plan_sha256'] != PARENT_PLAN:
                raise ValueError(f'Changed geometry reference: {label}')
            annotation = COHORTS / cohort['file']
            verified(annotation, cohort['sha256'])
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at the native shortlist budget', flush=True)
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                reference = {row['annotation_id']: row for row in prior['rows']}
                write_json(dest, dict(plan_sha256=digest, **trace(model, loader, reference)))
            report = read_json(dest)
            if report['plan_sha256'] != digest:
                raise ValueError(f'Changed equal-budget plan: {label}')
            groups = summarize(report)
            slices[protocol][fold] = dict(groups=groups, passed=passes(groups),
                                          replacements=sum(len(row['added']) for row in report['images']))
        del model
    accepted = all(row['passed'] for fs in slices.values() for row in fs.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=slices, accepted=accepted))


if __name__ == '__main__':
    main()
