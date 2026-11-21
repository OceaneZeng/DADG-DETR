"""Trace raw and filtered small-object coverage at three fixed training snapshots."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import run_s_task1_crop_refill as crop
from tools.owod import run_s_task1_crop_proxy as probe
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = probe.OUTPUT.with_name('s-instance-task1-crop-timeline')


def prepare():
    base = read_json(probe.OUTPUT / 'plan.json')
    result = read_json(probe.OUTPUT / 'comparison.json')
    final = Path(base['job']['parent'])
    data = probe.OUTPUT / 'fresh1024.json'
    ref = probe.OUTPUT / 'paired.json'
    if (result['status'] != 'complete' or result['follow_up'] or result['plan_sha256'] != file_sha256(probe.OUTPUT / 'plan.json')
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(data) != base['annotation_sha256']
            or file_sha256(final) != base['job']['parent_sha256'] or torch.cuda.get_device_name() != base['gpu']
            or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed parent model, endpoint, slice or environment')
    if read_json(final.parent / 'training_complete.json')['last_epoch'] != 11:
        raise ValueError('Incomplete parent training chain')
    snapshots = {}
    for epoch, path in [(5, final.with_name('checkpoint0004.pth')), (10, final.with_name('checkpoint0009.pth')), (12, final)]:
        saved = load_checkpoint_metadata(path)
        args = saved['args']
        if (saved['epoch'] != epoch - 1 or args.epochs != 12 or args.seed != base['seed']
                or args.owod_stage != 0 or args.owod_phase != 'increment' or Path(args.output_dir).resolve() != final.parent.resolve()
                or args.owod_protocol != base['protocol'] or args.discovery_version != 10 or args.discovery_scoring != 'focal'):
            raise ValueError(f'Unexpected training metadata for epoch {epoch}')
        snapshots[str(epoch)] = dict(path=str(path), sha256=file_sha256(path))
    plan = {k: base[k] for k in ('seed', 'protocol', 'gpu', 'torch_version', 'batch_size', 'annotation_sha256',
                                 'selected_sha256', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], checkpoints=snapshots, parent_plan_sha256=result['plan_sha256'],
                parent_result_sha256=file_sha256(probe.OUTPUT / 'comparison.json'), reference_sha256=file_sha256(ref),
                parent_training_input_sha256=file_sha256(final.parent / 'input.json'),
                rule='Same parent run at completed epochs 5, 10, 12; same 1024 training-source images. For small known '
                     'and small proxy GT separately trace all native queries, all four crops before any filtering, '
                     'quadrant ownership, area limit, exclusion against native top 49, and compatible class label. '
                     'Reconcile epoch 12 with the archived endpoint. Descriptive only, never select an earlier checkpoint.',
                criterion='Test late localization loss in a controlled training study only if raw four-crop coverage '
                          'of small proxy GT at epoch 12 falls >=5 percentage points vs epoch 5, while same-size known '
                          'raw coverage is flat or higher. Report epoch 10, paired IDs and source strata.')
    pin(OUTPUT / 'plan.json', plan)
    return plan, data, ref


def trace(native_boxes, query_count, all_crops, selected, native_labels, size, truth, anns, known, proxy):
    b, _, labels, ids = all_crops
    if len(b) != 4 * query_count:
        raise ValueError('Incomplete raw crop inventory')
    owned = crop.owned_centers(b, ids // query_count)
    pool = tuple(v[owned] for v in all_crops)
    wb, _, wl, _ = pool
    area = wb[:, 2:].prod(-1) * size.prod()
    sized = (area > 0) & (area <= 2048)
    eligible = crop.eligible_cells(wb, selected, size)
    boxes, preds, replacement = crop.refill(pool, selected, native_labels, size)
    measured = dict(native=probe.measure(selected, native_labels, truth, anns, known, proxy),
                    crops=probe.measure(boxes, preds, truth, anns, known, proxy))
    groups = {}
    for group, classes in [('known', known), ('proxy', proxy)]:
        indexes = [j for j, a in enumerate(anns) if a['category_id'] in classes and a['area'] < 32**2]
        target = truth[indexes]
        gt = [anns[j]['id'] for j in indexes]
        required = labels.new_tensor([anns[j]['category_id'] if group == 'known' else 91 for j in indexes])
        edges = overlap(b, target) >= .5
        own = edges[owned]
        legal = own[eligible]
        stages = dict(raw_native=(overlap(native_boxes, target) >= .5).any(0), raw_crops=edges.any(0),
                      owned=own.any(0), sized=own[sized].any(0), eligible=legal.any(0),
                      label_supported=(legal & (wl[eligible, None] == required[None, :])).any(0))
        groups[group] = {k: [j for j, hit in zip(gt, v.tolist()) if hit] for k, v in stages.items()}
        groups[group]['gt'] = gt
    return dict(small=groups, outcomes=measured, candidate=replacement['candidate'])


@torch.inference_mode()
def evaluate(epoch, model, post, loader, known, proxy, reference):
    rows = {}
    for step, (samples, targets) in enumerate(loader, 1):
        keys = [int(t['image_id'].item()) for t in targets]
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = model(samples.to('cuda'))
        dets = post(native, sizes, include_top_indices=True)
        raw = crop.crop_predictions(model, post, loader.dataset, keys, sizes, include_unowned=True)
        for i, (key, target, det) in enumerate(zip(keys, targets, dets)):
            boxes = native['pred_boxes'][i]
            queries = det['top_indices'] // (len(known) + 1)
            anns = valid_annotations(loader.dataset, key)
            if len(queries) != 50 or target['labels'].tolist() != [a['category_id'] for a in anns]:
                raise ValueError('Unexpected output budget or annotation order')
            row = trace(boxes, len(boxes), raw[i], boxes[queries], det['labels'], sizes[i],
                        target['boxes'].cuda(), anns, known, proxy)
            prior = reference[str(key)]
            row['source'] = prior['source']
            if row['small']['proxy']['gt'] != prior['small_proxy_ids']:
                raise ValueError('Unpaired small proxy population')
            if epoch == 12:
                expected = prior['outcomes']['crops']
                if (row['small']['proxy']['eligible'] != expected['pool_small']
                        or row['small']['proxy']['label_supported'] != expected['unknown_pool_small']
                        or row['candidate'] != expected['candidate']
                        or any(v != prior['outcomes'][arm][k] for arm, rec in row['outcomes'].items() for k, v in rec.items())):
                    raise ValueError(f'Changed final epoch predictions on {key}')
            rows[str(key)] = row
        if step == 1 or step % 32 == 0 or step == len(loader):
            print(f'{datetime.now().isoformat(timespec="seconds")} epoch {epoch} proxy {step}/{len(loader)} batches', flush=True)
    if rows.keys() != reference.keys():
        raise ValueError('Incomplete snapshot trace')
    return rows


def summarize(images):
    groups = {k: Counter() for k in ('known', 'proxy')}
    for row in images:
        for group, values in row['small'].items():
            counts = groups[group]
            counts.update({k: len(ids) for k, ids in values.items()})
            for arm, record in row['outcomes'].items():
                counts[arm + '_emitted'] += len(set(values['gt']).intersection(record[group]))
    return groups


def main():
    plan, data, ref = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    reference = read_json(ref)
    if reference['plan_sha256'] != plan['parent_plan_sha256']:
        raise ValueError('Unpaired previous run')
    manifest = read_json(MANIFEST)
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    _, loader = make_loader(resolve_image_root(MANIFEST, manifest), data, plan['batch_size'], 2)
    totals, by_source = {}, {}
    for epoch in (12, 5, 10):
        spec = plan['checkpoints'][str(epoch)]
        path = OUTPUT / f'epoch{epoch}.json'
        if not path.exists():
            checkpoint = Path(spec['path'])
            model, _criterion = load_detector(load_checkpoint_metadata(checkpoint)['args'], checkpoint, torch.device('cuda'))
            model.eval()
            rows = evaluate(epoch, model, post, loader, known, proxy, reference['images'])
            write_json(path, dict(plan_sha256=digest, checkpoint_sha256=spec['sha256'], images=rows))
            del model, _criterion
        packet = read_json(path)
        if (packet['plan_sha256'] != digest or packet['checkpoint_sha256'] != spec['sha256']
                or packet['images'].keys() != reference['images'].keys()):
            raise ValueError(f'Incompatible saved snapshot {epoch}')
        records = packet['images'].values()
        totals[epoch] = summarize(records)
        by_source[epoch] = {s: summarize(r for r in records if r['source'] == s) for s in probe.SOURCES}
    early, late = totals[5], totals[12]
    direction = (100 * (early['proxy']['raw_crops'] - late['proxy']['raw_crops']) >= 5 * early['proxy']['gt']
                 and late['known']['raw_crops'] >= early['known']['raw_crops'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals, by_source=by_source,
                                                late_localization_gate=direction, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
