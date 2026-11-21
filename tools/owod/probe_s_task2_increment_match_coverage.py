"""Replay new-class target assignment on the original S Task 2 increment images."""
from collections import Counter
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from models.matcher import HungarianMatcher
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION, SCREEN
from tools.owod.screening_data import screening_files, source_ids, write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = DURATION.with_name('s-instance-task2-increment-match-coverage')


def summarize_image(qualified, native, geometry):
    """Binary LSAP maximizes the number of distinct target/query pairs above the threshold."""
    q, g = linear_sum_assignment(qualified.astype(np.int8), maximize=True)
    best = int(qualified[q, g].sum())
    hits = []
    for queries, targets in (native, geometry):
        picked = np.zeros(qualified.shape[1], dtype=bool)
        picked[targets] = qualified[queries, targets]
        hits.append(picked)
    raw = qualified.any(axis=0)
    if max(int(h.sum()) for h in hits) > best or best > int(raw.sum()):
        raise ValueError('Assignment exceeds the feasible one-to-one geometry bound')
    return raw, best, hits[0], hits[1]


@torch.inference_mode()
def replay(model, matcher, loader, current):
    model.eval()
    geometry_matcher = HungarianMatcher(cost_class=0, cost_bbox=matcher.cost_bbox, cost_giou=matcher.cost_giou)
    classes = {label: Counter(images=0, gt=0, raw_hits=0, native_hits=0, geometry_hits=0) for label in current}
    totals = Counter()
    seen = set()
    for batch, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        targets = move_targets(targets, 'cuda')
        native = matcher(outputs, targets)
        geometry = geometry_matcher(outputs, targets)
        for idx, target in enumerate(targets):
            image_id = int(target['image_id'].item())
            if image_id in seen:
                raise ValueError('Repeated training image in frozen replay')
            seen.add(image_id)
            labels = target['labels'].cpu().numpy()
            iou, _ = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target['boxes']))
            raw, best, actual, geo = summarize_image((iou > .5).cpu().numpy(),
                tuple(x.numpy() for x in native[idx]), tuple(x.numpy() for x in geometry[idx]))
            totals.update(gt=len(labels), raw_hits=int(raw.sum()), attainable_hits=best,
                          native_hits=int(actual.sum()), geometry_hits=int(geo.sum()))
            for label in np.unique(labels):
                keep = labels == label
                classes[int(label)].update(images=1, gt=int(keep.sum()), raw_hits=int(raw[keep].sum()),
                    native_hits=int(actual[keep].sum()), geometry_hits=int(geo[keep].sum()))
        if batch % 96 == 0 or batch == len(loader):
            print(f'Matching {batch}/{len(loader)} batches, {len(seen)} training images', flush=True)
    if seen != set(loader.dataset.ids):
        raise ValueError('Incomplete training image coverage')
    return dict(images=len(seen), totals=dict(totals), classes=classes)


def main():
    print('Checking original increment supervision and trained endpoints',flush=True)
    absent = DURATION.with_name('s-instance-task2-absent-class-errors')
    result = read_json(absent/'comparison.json')
    prior = read_json(absent/'plan.json')
    duration = read_json(DURATION/'comparison.json')
    training = read_json(DURATION/'plan.json')
    if (file_sha256(absent/'comparison.json') != 'd8937b9614b7f1c1c9ee0965c8702915999684f1d689c977d046203561a8dd8c'
            or result['plan_sha256'] != file_sha256(absent/'plan.json') or result['supported']
            or file_sha256(DURATION/'comparison.json') != '4f698b6c4c608cb835f063960416ba546af06e557356f623555b4158eee9172f'
            or duration['plan_sha256'] != file_sha256(DURATION/'plan.json') or file_sha256(SCREEN) != training['screen_sha256']
            or file_sha256(MANIFEST) != prior['manifest_sha256'] or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != prior['gpu'] or str(torch.__version__) != prior['torch_version']
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0'):
        raise ValueError('Changed parent evidence, training split or frozen replay environment')
    manifest, files = screening_files(MANIFEST, SCREEN, 1)
    annotation = files['increment_train']
    data = read_json(annotation)
    ids = source_ids(data)
    current = manifest['stages'][1]['classes']
    if (len(ids) != len(data['images']) or len(ids) != training['train_images']
            or ids.intersection(training['origins']) or ids.intersection(manifest['evaluation']['image_source_ids'])
            or set(a['category_id'] for a in data['annotations']) != set(current)):
        raise ValueError('Expected original, release-only increment training images')
    checkpoints = {}
    for arm, epoch in [('3epochs',2), ('6epochs',5)]:
        cp = DURATION/arm/'increment/checkpoint.pth'
        digest = duration['arms'][arm]['increment']['checkpoint_sha256']
        cfg = read_json(cp.parent/'run_config.json')
        if (file_sha256(cp) != digest or read_json(cp.parent/'training_complete.json')['last_epoch'] != epoch
                or cfg['owod_screening_sha256'] != training['screen_sha256'] or cfg['owod_stage'] != 1
                or cfg['owod_phase'] != 'increment' or cfg['owod_current_class_ids'] != current):
            raise ValueError(f'Changed increment endpoint: {arm}')
        checkpoints[arm] = dict(path=str(cp),sha256=digest,
            costs=[cfg[k] for k in ('set_cost_class','set_cost_bbox','set_cost_giou')])
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_result_sha256=file_sha256(absent/'comparison.json'), source_duration_sha256=file_sha256(DURATION/'comparison.json'),
                manifest_sha256=file_sha256(MANIFEST), screening_sha256=file_sha256(SCREEN), annotation=str(annotation),
                annotation_sha256=file_sha256(annotation), checkpoints=checkpoints, classes=current, train_images=len(ids),
                protocol=prior['protocol'], paper_comparable=False, seed=42, gpu=prior['gpu'], device='0', batch_size=2,
                torch_version=prior['torch_version'], transform='deterministic validation resize of increment training images',
                criterion='At both frozen endpoints, require a nonzero deficit and at least half the native '
                          'GT misses at strict continuous IoU >0.5 to be recoverable by a one-to-one assignment '
                          'on the same boxes. To prioritize the classification cost, geometry-only Hungarian matching '
                          'must additionally recover at least half that attainable improvement at both endpoints.')
    pin(OUTPUT/'plan.json',plan)
    digest = file_sha256(OUTPUT/'plan.json')
    _, loader = make_loader(resolve_image_root(MANIFEST,manifest),annotation,2,2)
    summaries = {}
    for arm, cp in checkpoints.items():
        path = OUTPUT/(arm+'.json')
        if not path.exists():
            print(f'Replaying {arm} on {len(ids)} original increment images',flush=True)
            model, criterion = load_detector(None,Path(cp['path']),torch.device('cuda'))
            matcher = criterion.matcher
            if [matcher.cost_class,matcher.cost_bbox,matcher.cost_giou] != cp['costs']:
                raise ValueError('Frozen matcher differs from the training configuration')
            evidence = replay(model,matcher,loader,current)
            write_json(path,dict(plan_sha256=digest,checkpoint_sha256=cp['sha256'],**evidence))
            del model, criterion
        row = read_json(path)
        totals = row['totals']
        if (row['plan_sha256'] != digest or row['checkpoint_sha256'] != cp['sha256'] or row['images'] != len(ids)
                or any(totals[k] != sum(c[k] for c in row['classes'].values()) for k in ('gt','native_hits','raw_hits','geometry_hits'))):
            raise ValueError(f'Incomplete per-class matching audit: {arm}')
        missed = totals['gt'] - totals['native_hits']
        possible = totals['attainable_hits'] - totals['native_hits']
        gain = totals['geometry_hits'] - totals['native_hits']
        summaries[arm] = dict(**totals,missed=missed,attainable_gain=possible,geometry_gain=gain,
                              assignment_pass=missed>0 and possible*2 >= missed,
                              class_cost_pass=possible>0 and gain*2 >= possible)
    counts = [read_json(OUTPUT/(arm+'.json'))['classes'] for arm in checkpoints]
    if any((c['gt'],c['images']) != (counts[1][k]['gt'],counts[1][k]['images']) for k,c in counts[0].items()):
        raise ValueError('Different class exposure in replay')
    write_json(OUTPUT/'comparison.json',dict(status='complete',paper_comparable=False,plan_sha256=digest,arms=summaries,
               supported=all(v['assignment_pass'] and v['class_cost_pass'] for v in summaries.values())))
    print('Increment match coverage comparison saved',flush=True)


if __name__ == '__main__':
    main()
