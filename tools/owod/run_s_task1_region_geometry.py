"""Compare pixel-region and translated proposals at a fixed 100-box budget."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import os
import subprocess

import cv2
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import run_s_task1_crop_proxy as probe
from tools.owod import run_s_task1_crop_timeline as timeline
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.prepare_cat_proposals import selective_search
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = timeline.OUTPUT.with_name('s-instance-task1-region-geometry')
ARMS = ('native', 'retained', 'hybrid', 'shifted', 'region100')


def prepare():
    base = read_json(timeline.OUTPUT / 'plan.json')
    result = read_json(timeline.OUTPUT / 'comparison.json')
    reference = timeline.OUTPUT / 'epoch12.json'
    data = probe.OUTPUT / 'fresh1024.json'
    checkpoint = Path(base['checkpoints']['12']['path'])
    if (result['status'] != 'complete' or result['late_localization_gate']
            or result['plan_sha256'] != file_sha256(timeline.OUTPUT / 'plan.json')
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(data) != base['annotation_sha256']
            or file_sha256(checkpoint) != base['checkpoints']['12']['sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed final snapshot, slice, parent decision or environment')
    plan = {k: base[k] for k in ('seed', 'protocol', 'gpu', 'torch_version', 'batch_size', 'annotation_sha256',
                                'selected_sha256', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], job=base['checkpoints']['12'],
                parent_plan_sha256=result['plan_sha256'], parent_result_sha256=file_sha256(timeline.OUTPUT / 'comparison.json'),
                reference_sha256=file_sha256(reference), opencv=cv2.__version__, workers=4,
                region=dict(mode='fast', max_proposals=2000, min_area=16, small_area_cap=2048, replace=25),
                rule='Use image pixels only for selective search. Take first 25 distinct boxes of area 16..2048 '
                     'from the native engine order, and retain the highest focal-scored 100 minus that count '
                     'full-frame decoder queries. Compare native 100 and same hybrid with identical regions '
                     'translated halfway across each valid origin range. Separately report coverage with '
                     'up to the first 100 small regions. GT used after candidate sets are fixed. Any-box IoU >=.5, '
                     'pure geometry before decoding or class scoring.',
                criterion='Advance to a decoder test only if the hybrid small proxy coverage rate exceeds both '
                          'native and shifted by >=5 pp and preserves >=95% of the native-covered identities '
                          'in each of small known and all known. Exploratory only.')
    pin(OUTPUT / 'plan.json', plan)
    ref = read_json(reference)
    if ref['plan_sha256'] != result['plan_sha256'] or ref['checkpoint_sha256'] != plan['job']['sha256']:
        raise ValueError('Unpaired geometry reference')
    return plan, read_json(MANIFEST), data, ref['images']


@torch.inference_mode()
def native_rows(model, post, loader, known, proxy, reference):
    rows = {}
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        probabilities, _, _ = post.candidate_scores(out)
        for i, t in enumerate(targets):
            key = str(t['image_id'].item())
            info = loader.dataset.coco.imgs[int(key)]
            boxes = out['pred_boxes'][i]
            if len(boxes) != 100:
                raise ValueError('Expected 100 native decoder boxes')
            order = torch.argsort(probabilities[i].amax(-1), descending=True, stable=True)
            anns = valid_annotations(loader.dataset, int(key))
            if t['labels'].tolist() != [a['category_id'] for a in anns]:
                raise ValueError('Changed annotation order')
            gt = t['boxes'].cuda()
            groups = dict(known=[j for j, a in enumerate(anns) if a['category_id'] in known],
                          proxy=[j for j, a in enumerate(anns) if a['category_id'] in proxy])
            groups.update({g + '_small': [j for j in indices if anns[j]['area'] < 32**2]
                           for g, indices in list(groups.items())})
            for g in ('known', 'proxy'):
                idx = groups[g + '_small']
                actual = probe.covered(boxes, gt[idx], [anns[j]['id'] for j in idx])
                if actual != reference[key]['small'][g]['raw_native']:
                    raise ValueError(f'Changed native {g} geometry on {key}')
            rows[key] = dict(source=reference[key]['source'], file=info['file_name'], width=info['width'], height=info['height'],
                             boxes=boxes[order].tolist(), truth=t['boxes'].tolist(), ann_ids=[a['id'] for a in anns], groups=groups)
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'{datetime.now().isoformat(timespec="seconds")} native {step}/{len(loader)} batches', flush=True)
    if rows.keys() != reference.keys():
        raise ValueError('Incomplete native endpoint')
    return rows


def region_boxes(item, root, settings):
    key, row = item
    cv2.setRNGSeed(int(key) % 2147483647)
    boxes = selective_search(root / row['file'], settings['mode'], settings['max_proposals'], settings['min_area'])
    small = [b for b in boxes if b[2] * b[3] <= settings['small_area_cap']]
    return key, len(boxes), len(small), small[:100]


def normalized(boxes, width, height):
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    b[:, :2] += .5 * b[:, 2:]
    return b / b.new_tensor([width, height, width, height])


def compare(row, proposals, limit):
    w, h = row['width'], row['height']
    selected = proposals[:limit]
    translated = []
    for x, y, bw, bh in selected:
        nx, ny = w - bw + 1, h - bh + 1
        translated.append([(x + nx // 2) % nx, (y + ny // 2) % ny, bw, bh])
    native = torch.tensor(row['boxes'])
    retained = native[:100 - len(selected)]
    candidates = dict(native=native, retained=retained, hybrid=torch.cat((retained, normalized(selected, w, h))),
                      shifted=torch.cat((retained, normalized(translated, w, h))), region100=normalized(proposals, w, h))
    if len(candidates['hybrid']) != 100 or len(candidates['shifted']) != 100:
        raise ValueError('Hybrid or shifted proposal budget mismatch')
    gt = torch.tensor(row['truth'], dtype=torch.float32).reshape(-1, 4)
    coverage = {}
    for g, indexes in row['groups'].items():
        ids = [row['ann_ids'][j] for j in indexes]
        coverage[g] = {arm: probe.covered(boxes, gt[indexes], ids) for arm, boxes in candidates.items()}
    return dict(source=row['source'], inserted=len(selected), regions=proposals, coverage=coverage)


def counts(rows, native):
    totals = {g: Counter() for g in ('known', 'proxy', 'known_small', 'proxy_small')}
    for key, row in rows:
        for g, v in row['coverage'].items():
            totals[g]['gt'] += len(native[key]['groups'][g])
            totals[g].update({arm: len(v[arm]) for arm in ARMS})
            totals[g]['native_preserved'] += len(set(v['native']).intersection(v['hybrid']))
    return totals


def main():
    plan, manifest, data, ref = prepare()
    root = resolve_image_root(MANIFEST, manifest)
    digest = file_sha256(OUTPUT / 'plan.json')
    checkpoint = Path(plan['job']['path'])
    native_path = OUTPUT / 'native.json'
    if not native_path.exists():
        model, criterion = load_detector(load_checkpoint_metadata(checkpoint)['args'], checkpoint, torch.device('cuda'))
        model.eval()
        known, proxy = [s['classes'] for s in manifest['stages'][:2]]
        post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
        _, loader = make_loader(root, data, plan['batch_size'], 2)
        rows = native_rows(model, post, loader, known, proxy, ref)
        write_json(native_path, dict(plan_sha256=digest, images=rows))
        del model, criterion, rows
        torch.cuda.empty_cache()
    saved = read_json(native_path)
    if saved['plan_sha256'] != digest or saved['images'].keys() != ref.keys():
        raise ValueError('Incompatible native endpoint')
    native = saved['images']
    paired_path = OUTPUT / 'paired.json'
    paired = read_json(paired_path) if paired_path.exists() else dict(plan_sha256=digest, native_sha256=file_sha256(native_path), images={})
    if paired['plan_sha256'] != digest or paired['native_sha256'] != file_sha256(native_path) or not paired['images'].keys() <= native.keys():
        raise ValueError('Incompatible region checkpoint')
    rows = paired['images']
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    items = [(k, v) for k, v in native.items() if k not in rows]
    with ThreadPoolExecutor(max_workers=plan['workers']) as executor:
        stream = executor.map(lambda item: region_boxes(item, root, plan['region']), items)
        for key, raw_count, small_count, proposals in stream:
            result = compare(native[key], proposals, plan['region']['replace'])
            result.update(total_raw=raw_count, total_small=small_count)
            rows[key] = result
            n = len(rows)
            if n == 1 or n % 16 == 0 or n == len(native):
                print(f'{datetime.now().isoformat(timespec="seconds")} regions {n}/{len(native)} images', flush=True)
            if n % 128 == 0 or n == len(native):
                write_json(paired_path, paired)
    if rows.keys() != native.keys():
        raise ValueError('Incomplete region study')
    totals = counts(rows.items(), native)
    sources = {s: counts(((k, r) for k, r in rows.items() if r['source'] == s), native) for s in probe.SOURCES}
    v = totals['proxy_small']
    gain = 100 * (v['hybrid'] - max(v['native'], v['shifted'])) >= 5 * v['gt']
    retention = all(100 * totals[g]['native_preserved'] >= 95 * totals[g]['native'] for g in ('known_small', 'known'))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals, by_source=sources,
                                                follow_up=bool(gain and retention), paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
