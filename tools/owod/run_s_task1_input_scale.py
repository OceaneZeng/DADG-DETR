"""Paired S Task 1 input-size diagnosis with the native focal Top-50."""
from collections import Counter
from pathlib import Path
import os
import subprocess
import time

import torch

from datasets import transforms as T
from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import probe_task1_known_queries as independent
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = PARENT.with_name('s-instance-task1-input-scale')
SIZES = {'standard': (800, 1333), 'larger': (1000, 1666)}


def valid_annotations(dataset, image_id):
    image = dataset.coco.imgs[image_id]
    annotations = []
    for ann in dataset.coco.imgToAnns[image_id]:
        x, y, w, h = ann['bbox']
        if (not ann.get('iscrowd') and min(x + w, image['width']) > max(x, 0)
                and min(y + h, image['height']) > max(y, 0)):
            annotations.append(ann)
    return annotations


def summarize(dataset, rows, known_ids, proxy_ids):
    totals = Counter()
    for key, row in rows.items():
        annotations = valid_annotations(dataset, int(key))
        hidden = [a for a in annotations if a['category_id'] in proxy_ids]
        known = [a['category_id'] for a in annotations if a['category_id'] in known_ids]
        assert len(hidden) == row['counts']['gt']
        best = row['hidden_gt']
        covered = set(best)
        emitted = {best[q] for q in row['picks']['native']}
        correct = set()
        for position in row['known_positions']:
            query, label = divmod(position, len(known_ids) + 1)
            index = row['known_gt'][query]
            if index >= 0 and known[index] == known_ids[label]:
                correct.add(index)
        totals.update(known_gt=len(known), known_correct=len(correct), unknown_quota=row['counts']['quota'],
                      native_hits=row['counts']['native_hits'], known_only=row['counts']['native_known_only'])
        for i, ann in enumerate(hidden):
            area = ann['area']
            size = 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'
            totals[size + '_gt'] += 1
            totals[size + '_covered'] += int(i in covered)
            totals[size + '_native'] += int(i in emitted)
    return totals


def main():
    previous = read_json(PARENT / 'comparison.json')
    evaluation = PARENT / 'evaluation'
    old_plan = read_json(evaluation / 'plan.json')
    source_plan = read_json(PARENT / 'plan.json')
    job = old_plan['jobs']['42']
    checkpoint = Path(job['parent'])
    data = evaluation / 'fresh1024.json'
    reference_path = evaluation / 'seed42-fresh1024.json'
    reference = read_json(reference_path)
    if (previous['status'] != 'complete' or previous['passes_screen']
            or not file_sha256(PARENT / 'plan.json') == previous['plan_sha256'] == old_plan['plan_sha256']
            or not file_sha256(evaluation / 'plan.json') == reference['plan_sha256'] == previous['evaluation_plan_sha256']
            or not file_sha256(checkpoint) == job['parent_sha256'] == previous['parent_sha256']
            or file_sha256(data) != old_plan['fresh_sha256'] or file_sha256(MANIFEST) != source_plan['manifest_sha256']
            or torch.cuda.get_device_name() != source_plan['gpus'][0] or str(torch.__version__) != source_plan['torch_version']):
        raise ValueError('Expected the completed S seed 42 diagnosis and its frozen inputs')
    plan = dict(seed=42, protocol='s-owodb', paper_comparable=False, job=job, sizes={k: list(v) for k, v in SIZES.items()},
                batch_size=old_plan['batch_size'], devices=os.environ['CUDA_VISIBLE_DEVICES'],
                gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                manifest_sha256=source_plan['manifest_sha256'], fresh_sha256=old_plan['fresh_sha256'],
                parent_plan_sha256=old_plan['plan_sha256'], parent_result_sha256=file_sha256(PARENT / 'comparison.json'),
                reference_sha256=file_sha256(reference_path),
                criterion='Explore an independent slice only if small best-target coverage rises >=5 pp, '
                          'total native proxy hits increase and correctly classified known GT hits do not fall. '
                          'Fixed native joint Top-50. Unknown quota can vary with resolution. No training.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    root = resolve_image_root(MANIFEST, manifest)
    known = manifest['stages'][0]['classes']
    proxy = manifest['stages'][1]['classes']
    sources = {int(im): r['source'] for im, r in reference['images'].items()}
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal',
                                unknown_threshold=args.unknown_threshold)
    rows = {}
    for arm, (short_side, cap) in SIZES.items():
        dataset, loader = make_loader(root, data, plan['batch_size'], 2)
        resize, normalize = dataset._transforms.transforms
        if resize.sizes != [800] or resize.max_size != 1333:
            raise ValueError('Unexpected parent validation transform')
        dataset._transforms = T.Compose([T.RandomResize([short_side], max_size=cap), normalize])
        path = OUTPUT / f'{arm}.json'
        if not path.exists():
            print(f'{arm}: evaluating short side {short_side}, long side cap {cap}', flush=True)
            start = time.monotonic()
            result = independent.evaluate(model, post, loader, torch.tensor(known, device='cuda'),
                                          torch.tensor(proxy, device='cuda'), sources)
            write_json(path, dict(plan_sha256=digest, seconds=time.monotonic() - start, **result))
        saved = read_json(path)
        if saved['plan_sha256'] != digest or saved['images'].keys() != reference['images'].keys():
            raise ValueError(f'Incomplete or changed {arm} endpoint')
        if arm == 'standard':
            for im, row in saved['images'].items():
                parent = reference['images'][im]
                if (row['picks']['native'] != parent['picks']['native'] or
                        any(row[k] != parent[k] for k in ('hidden_gt', 'known_gt', 'known_positions'))):
                    raise ValueError(f'Baseline predictions differ on image {im}')
        rows[arm] = dict(counts=summarize(dataset, saved['images'], known, proxy), seconds=saved['seconds'])
    base, large = (rows[k]['counts'] for k in SIZES)
    passes = (100 * (large['small_covered'] - base['small_covered']) >= 5 * base['small_gt']
              and large['native_hits'] > base['native_hits'] and large['known_correct'] >= base['known_correct'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                                follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
