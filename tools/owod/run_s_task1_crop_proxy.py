"""Verify the fixed crop refill on fresh S Task 1 / Task 2 training-source images."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import run_s_task1_crop_refill as crop
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_encoder_transfer import identities
from tools.owod.probe_task1_known_queries import SOURCES, fresh_origins
from tools.owod.probe_task1_unknown_factors import best_target
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT, proxy_data
from tools.owod.run_s_task1_released_seed import OUTPUT as RELEASED
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = crop.OUTPUT.with_name('s-instance-task1-crop-proxy')
ARMS = ('native', 'full', 'crops')


def prepare():
    base = read_json(crop.OUTPUT / 'plan.json')
    result = read_json(crop.OUTPUT / 'comparison.json')
    released = read_json(RELEASED / 'plan.json')
    screen = PARENT / 'data/screening.json'
    data_path = RELEASED / 'released2048.json'
    checkpoint = Path(base['job']['parent'])
    if (result['status'] != 'complete' or not result['follow_up']
            or result['plan_sha256'] != file_sha256(crop.OUTPUT / 'plan.json')
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(checkpoint) != base['job']['parent_sha256']
            or file_sha256(data_path) != base['released_sha256'] or file_sha256(screen) != released['screen_sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed parent endpoint, model, data or environment')
    manifest, files = screening_files(MANIFEST, screen, 0)
    blocked = set(manifest['evaluation']['image_source_ids']) | set(read_json(screen)['heldout_source_ids'])
    blocked.update(source_ids(read_json(files['increment_train'])))
    blocked.update(read_json(PARENT / 'selection/inputs.json')['origins'])
    if identities(blocked) != released['exclusions_sha256']:
        raise ValueError('Previous exclusions changed')
    blocked.update(source_ids(read_json(data_path)))
    origins = fresh_origins(manifest, blocked, 51)
    if set(origins) & blocked or Counter(origins.values()) != {s: 512 for s in SOURCES}:
        raise ValueError('Proxy sources are not an independent 512/512 slice')
    # Identity selection is finished before accessing the combined annotations.
    data = proxy_data(manifest, origins)
    if source_ids(data) != set(origins) or len(data['images']) != 1024:
        raise ValueError('Incomplete independent slice')
    data_hash = sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    plan = {k: base[k] for k in ('job', 'seed', 'gpu', 'torch_version', 'batch_size', 'protocol',
                                 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], parent_plan_sha256=result['plan_sha256'],
                parent_result_sha256=file_sha256(crop.OUTPUT / 'comparison.json'), parent_released_sha256=base['released_sha256'],
                selection_seed=51, excluded_sha256=identities(blocked), selected_sha256=identities(origins),
                per_source=512, screen_sha256=file_sha256(screen), annotation_sha256=data_hash,
                rule='Apply the pinned parent native, unused full-frame and 4-crop Top-50 refill unchanged. '
                     'Use fixed slice annotation IDs for posthoc paired unknown-proxy and known hits. '
                     'Report small proxy pool coverage, including the subset predicted unknown.',
                criterion='Consider an independent seed only if small unknown-proxy hits exceed both controls and '
                          'paired positive vs negative small-hit images favor crops over native at one-sided exact '
                          'sign P<=.05. Require total proxy and known correct hits >=native and the fraction of '
                          'unknown outputs covering only known GT <=native. Report each source and all losses.')
    pin(OUTPUT / 'plan.json', plan)
    annotation = OUTPUT / 'fresh1024.json'
    if not annotation.exists():
        write_json(annotation, data)
    if file_sha256(annotation) != data_hash:
        raise ValueError('Changed proxy annotations')
    pin(OUTPUT / 'selection' / 'origins.json', origins)
    return plan, manifest, checkpoint, annotation, {im['id']: origins[im['owod_source_id']] for im in data['images']}


def measure(boxes, labels, truth, anns, known, proxy):
    unknown = labels == 91
    hits, matches = {}, {}
    for group, classes, active in [('known', known, ~unknown), ('proxy', proxy, unknown)]:
        indices = [j for j, a in enumerate(anns) if a['category_id'] in classes]
        assigned = best_target(boxes, truth[indices])
        matches[group] = assigned >= 0
        hits[group] = sorted({anns[indices[j]]['id'] for j, label in zip(assigned[active].tolist(), labels[active].tolist())
                              if j >= 0 and (group == 'proxy' or label == anns[indices[j]]['category_id'])})
    hits.update(unknown_outputs=int(unknown.sum()), unknown_only_known=int((unknown & matches['known'] & ~matches['proxy']).sum()))
    return hits


def covered(boxes, truth, ids):
    if not len(boxes) or not ids:
        return []
    hits = (overlap(boxes, truth) >= .5).any(0).tolist()
    return [j for j, hit in zip(ids, hits) if hit]


@torch.inference_mode()
def evaluate(model, post, loader, known, proxy, origins):
    rows = {}
    for step, (samples, targets) in enumerate(loader, 1):
        keys = [int(t['image_id'].item()) for t in targets]
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        views = crop.image_views(model, post, loader.dataset, samples, keys, sizes)
        for i, (selected, labels, full, crops) in enumerate(views):
            key = keys[i]
            pools = dict(full=full, crops=crops)
            outputs = {'native': (selected, labels, {})}
            for arm, pool in pools.items():
                outputs[arm] = crop.refill(pool, selected, labels, sizes[i])
            # Only the frozen predictions above are used to choose the refill.
            anns = valid_annotations(loader.dataset, key)
            truth = targets[i]['boxes'].cuda()
            if [a['category_id'] for a in anns] != targets[i]['labels'].tolist():
                raise ValueError('Changed annotation order')
            idx = [j for j, a in enumerate(anns) if a['category_id'] in proxy and a['area'] < 32**2]
            small = [anns[j]['id'] for j in idx]
            records = {}
            for arm, (boxes, predicted, record) in outputs.items():
                if len(boxes) != 50 or len(predicted) != 50:
                    raise ValueError('Changed output budget')
                record.update(measure(boxes, predicted, truth, anns, known, proxy))
                if arm in pools:
                    del record['pool_covers'], record['winner_covers']
                    pool_boxes, _, pool_labels, _ = pools[arm]
                    eligible = crop.eligible_cells(pool_boxes, selected, sizes[i])
                    if len(eligible) != record['pool']:
                        raise ValueError('Changed refill pool')
                    record['pool_small'] = covered(pool_boxes[eligible], truth[idx], small)
                    record['unknown_pool_small'] = covered(pool_boxes[eligible[pool_labels[eligible] == 91]], truth[idx], small)
                    record['winner_small'] = covered(boxes[-1:], truth[idx], small) if record['candidate'] else []
                records[arm] = record
            rows[key] = dict(source=origins[key], small_proxy_ids=small, outcomes=records)
        if step == 1 or step % 32 == 0 or step == len(loader):
            print(f'{datetime.now().isoformat(timespec="seconds")} proxy {step}/{len(loader)} batches', flush=True)
    if rows.keys() != origins.keys():
        raise ValueError('Incomplete paired proxy endpoint')
    return rows


def summarize(rows):
    totals = {arm: Counter() for arm in ARMS}
    for row in rows:
        small = set(row['small_proxy_ids'])
        native = len(small.intersection(row['outcomes']['native']['proxy']))
        for arm, record in row['outcomes'].items():
            n = len(small.intersection(record['proxy']))
            totals[arm].update(images=1, known=len(record['known']), proxy=len(record['proxy']), small_proxy=n,
                               positive_images=int(n > native), negative_images=int(n < native),
                               unknown_outputs=record['unknown_outputs'], unknown_only_known=record['unknown_only_known'])
    return totals


def main():
    plan, manifest, checkpoint, annotation, origins = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    model, _ = load_detector(load_checkpoint_metadata(checkpoint)['args'], checkpoint, torch.device('cuda'))
    model.eval()
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    _, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, plan['batch_size'], 2)
    rows = evaluate(model, post, loader, known, proxy, origins)
    write_json(OUTPUT / 'paired.json', dict(plan_sha256=digest, images=rows))
    totals = summarize(rows.values())
    groups = {s: summarize(row for row in rows.values() if row['source'] == s) for s in SOURCES}
    c, n = totals['crops'], totals['native']
    discordant = c['positive_images'] + c['negative_images']
    p = sum(comb(discordant, j) for j in range(c['positive_images'], discordant + 1)) / 2**discordant
    passes = (c['small_proxy'] > max(n['small_proxy'], totals['full']['small_proxy']) and p <= .05
              and c['proxy'] >= n['proxy'] and c['known'] >= n['known'] and min(n['unknown_outputs'], c['unknown_outputs']) > 0
              and c['unknown_only_known'] * n['unknown_outputs'] <= n['unknown_only_known'] * c['unknown_outputs'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals, by_source=groups,
                                                sign_p=p, follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
