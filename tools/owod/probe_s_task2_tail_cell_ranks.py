"""Trace the quality rank and training-sampler coverage of missed S Task 2 proposals."""
from functools import partial
import os
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch

from models.graph_local.instance_proposals import nearby_anchor_ids
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_shortlist_depth import OUTPUT as PARENT
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = PARENT.with_name('s-instance-task2-tail-cell-ranks')
REFERENCE_HASH = {
    'original': 'cfa23bb140b83a073a62892a54f4278e25a1e239d438c3683e978d073e56099b',
    'independent': 'a6e597fe38384ad2d69f5166e95f2616b5e4c658ab6611864e3c4ccb79f6faed',
}
CRITERION = ('For the baseline final-missed tail targets with dense but no top-300 coverage, report '
             'the rank of the highest-scoring correct cell and whether the 16 native supervision anchors '
             'contain any correct cell. If a strict majority lack sampler coverage in both slices, investigate '
             'positive sampling. If a strict majority have coverage in both, investigate quality ranking '
             'against competing cells. Otherwise record a mixed result. No label-guided inference.')


def prepare():
    src = read_json(PARENT/'plan.json')
    result = read_json(PARENT/'comparison.json')
    cp = src['checkpoint']
    if (file_sha256(PARENT/'comparison.json') != 'f9564f1fb40c83ee1dff759c9b3421964f808467e86d11ade29b5736ca89286a'
            or result['status'] != 'complete' or result['follow_up']
            or file_sha256(PARENT/'plan.json') != result['plan_sha256']
            or file_sha256(Path(cp['path'])) != cp['sha256'] or file_sha256(MANIFEST) != src['manifest_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed parent comparison, checkpoint, protocol or runtime')
    for name, entry in src['slices'].items():
        if (file_sha256(Path(entry['annotation'])) != entry['annotation_sha256']
                or file_sha256(PARENT/f'{name}-native.json') != REFERENCE_HASH[name]):
            raise ValueError(f'Changed annotations or native reference for {name}')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version', 'tail_class_ids')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan_sha256=result['plan_sha256'], parent_comparison_sha256=file_sha256(PARENT/'comparison.json'),
                native_sha256=REFERENCE_HASH, criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan, result


def observe(rows, dataset, wanted, threshold, outputs, targets, batch_images, score_tables):
    geo = outputs['geometry']
    for i, image_id in batch_images.items():
        annotations = valid_annotations(dataset, image_id)
        selected = [(j, ann) for j, ann in enumerate(annotations) if ann['id'] in wanted]
        if not selected:
            continue
        target = targets[i]
        if target['labels'].tolist() != [ann['category_id'] for ann in annotations]:
            raise ValueError(f'Cannot pair transformed annotations on image {image_id}')
        ids = torch.where(geo['valid'][i])[0]
        quality = geo['logits'][i, ids]
        scores = quality.cpu().numpy()
        near = nearby_anchor_ids(target['boxes'].to(ids.device), geo['anchors'][i], ids)
        near_indices = torch.searchsorted(ids, near).cpu().numpy()
        height, width = target['orig_size'].tolist()
        boxes = box_cxcywh_to_xyxy(geo['boxes'][i, ids])
        voc = (boxes * boxes.new_tensor([width, height, width, height])).cpu().numpy().astype(np.float64)
        voc[:, :2] += 1
        np.round(voc, decimals=1, out=voc)
        native_pool = quality.topk(min(300, len(ids))).indices.cpu().numpy()
        expanded_pool = quality.topk(min(600, len(ids))).indices.cpu().numpy()
        eligible = [ann for ann in annotations if not ann.get('difficult')]
        # Each high-scoring cell counts once, even if it covers multiple released objects.
        covers_released = np.zeros(len(native_pool), dtype=bool)
        for ann in eligible:
            covers_released |= overlaps(np.asarray(ann['voc_bbox']), voc[native_pool]) > threshold
        for j, ann in selected:
            ious = overlaps(np.asarray(ann['voc_bbox']), voc)
            good = ious > threshold
            if not good.any() or good[native_pool].any():
                raise ValueError(f'Dense coverage or top-300 exclusion changed for {ann["id"]}')
            best_score = scores[good].max()
            sample_ious = ious[near_indices[j]]
            rows.append(dict(annotation_id=ann['id'], category_id=ann['category_id'],
                             image=dataset.coco.imgs[image_id]['owod_source_id'], valid_cells=len(ids),
                             accurate_cells=int(good.sum()), first_rank=1+int((scores > best_score).sum()),
                             last_tied_rank=int((scores >= best_score).sum()),
                             sampled_correct=int((sample_ious > threshold).sum()), sampled_best_iou=float(sample_ious.max()),
                             dense_best_iou=float(ious.max()), in_expanded=bool(good[expanded_pool].any()),
                             top300_cells=len(native_pool), top300_covering_released=int(covers_released.sum())))


def main():
    plan, parent = prepare()
    plan_hash = file_sha256(OUTPUT/'plan.json')
    manifest = read_json(MANIFEST)
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    summary = {}
    for name, entry in plan['slices'].items():
        src = parent['slices'][name]
        wanted = set(src['missing_before_shortlist'])
        path = OUTPUT/f'{name}.json'
        if not path.exists():
            ann = Path(entry['annotation'])
            ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
            order = [im['owod_source_id'] for im in read_json(ann)['images']]
            rows = []
            print(f'Evaluating {name} on {len(ds)} images, {len(wanted)} missed targets', flush=True)
            observer = partial(observe, rows, ds, wanted, manifest['evaluation']['iou_threshold'])
            views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, rank_observer=observer)
            write_json(path, dict(plan_sha256=plan_hash, views=views, rows=rows))
        evidence = read_json(path)
        rows = evidence['rows']
        if (evidence['plan_sha256'] != plan_hash or evidence['views'] != read_json(PARENT/f'{name}-native.json')['views']
                or len(rows) != len(wanted) or {r['annotation_id'] for r in rows} != wanted
                or {r['annotation_id'] for r in rows if r['in_expanded']} != set(src['recovered_in_shortlist'])):
            raise ValueError(f'Native benchmark or traced identities differ on {name}')
        unsampled = sum(r['sampled_correct'] == 0 for r in rows)
        by_image = {r['image']: r for r in rows}
        summary[name] = dict(targets=len(rows), without_accurate_sample=unsampled,
                             with_accurate_sample=len(rows)-unsampled, median_first_rank=median(r['first_rank'] for r in rows),
                             images=len(by_image), top300_cells=sum(r['top300_cells'] for r in by_image.values()),
                             top300_covering_released=sum(r['top300_covering_released'] for r in by_image.values()))
    decision = 'mixed'
    if all(s['without_accurate_sample'] * 2 > s['targets'] for s in summary.values()):
        decision = 'positive_sampling'
    elif all(s['with_accurate_sample'] * 2 > s['targets'] for s in summary.values()):
        decision = 'quality_ranking'
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
                                               slices=summary, decision=decision))
    print('Dense cell rank and sampler comparison saved', flush=True)


if __name__ == '__main__':
    main()
