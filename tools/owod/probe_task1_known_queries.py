"""Confirm known-query exclusion on previously unused training-source images."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import prepare_benchmarks
from tools.owod import probe_task1_unknown_factors as factors
from tools.owod import probe_task1_visible_classes as visible
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_task1_encoder_transfer import identities
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-task1-known-queries'
SOURCES = ('t1_train', 't2_train')


def prepare():
    parent = read_json(visible.OUTPUT / 'plan.json')
    end = read_json(visible.OUTPUT / 'comparison.json')
    factor_plan = read_json(factors.OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    prior = visible.OUTPUT / 'val-data.json'
    if (end['status'] != 'complete' or end['plan_sha256'] != file_sha256(visible.OUTPUT / 'plan.json')
            or file_sha256(factors.OUTPUT / 'plan.json') != parent['parent_plan_sha256']
            or parent['jobs'] != factor_plan['jobs'] or file_sha256(prior) != parent['val_sha256']
            or file_sha256(MANIFEST) != parent['manifest_sha256'] or torch.cuda.get_device_name() != parent['gpu']
            or str(torch.__version__) != parent['torch_version']):
        raise ValueError('Changed released-only study, checkpoints or inference environment')
    blocked = visible.blocked_sources(factor_plan, manifest)
    if identities(blocked) != parent['excluded_names_sha256']:
        raise ValueError('Changed image exclusions')
    blocked.update(source_ids(read_json(prior)))
    origins = {}
    for source in SOURCES:
        pool = set(manifest['splits'][source]) - blocked
        chosen = sorted(pool, key=lambda s: hashlib.sha256(f'47:{s}'.encode()).digest())[:512]
        if len(chosen) != 512:
            raise ValueError(f'Incomplete independent pool for {source}')
        origins.update(dict.fromkeys(chosen, source))
        blocked.update(chosen)
    root = resolve_image_root(MANIFEST, manifest)
    xml = {s: prepare_benchmarks.read_voc(root.parent, s) for s in origins}
    data = prepare_benchmarks.subset(origins, manifest['stages'][1]['active_classes'], xml)
    encoded = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
    plan = {k: parent[k] for k in ('jobs', 'gpu', 'batch_size', 'torch_version', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=visible.OUTPUT.name, parent_plan_sha256=end['plan_sha256'],
                parent_result_sha256=file_sha256(visible.OUTPUT / 'comparison.json'),
                selection_seed=47, per_source=512, fresh_sha256=hashlib.sha256(encoded).hexdigest(),
                source_order=list(SOURCES), rule='Lock native focal Top-50 known entries and unknown quota. '
                'Compare native, objectness, and objectness with already-emitted known query IDs ranked last. '
                'Use stable query-index ties within the objectness queue; retain exact counts in every image.',
                criterion='Each seed: objectness has positive proxy hit gain and excess known-only overlaps '
                'versus native. Query exclusion retains >=half that gain and eliminates >=half the excess. '
                'Report both sources. Labels from the first 40 released classes enter scoring only.')
    pin(OUTPUT / 'plan.json', plan)
    path = OUTPUT / 'fresh1024.json'
    if not path.exists():
        write_json(path, data)
    if file_sha256(path) != plan['fresh_sha256']:
        raise ValueError('Changed independent evaluation data')
    return plan, manifest, {im['id']: origins[im['owod_source_id']] for im in data['images']}


@torch.inference_mode()
def evaluate(model, post, loader, known, proxy, sources):
    images, groups = {}, {s: Counter() for s in SOURCES}
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = post(out, sizes, include_top_indices=True)
        obj = out['pred_objectness'].sigmoid()
        product = obj * out['pred_novelty'].sigmoid()
        order = obj.argsort(dim=1, descending=True, stable=True)
        for i, target in enumerate(targets):
            im = int(target['image_id'].item())
            d = native[i]
            is_unknown = d['labels'] == 91
            positions = d['top_indices']
            known_positions = positions[~is_unknown]
            queries = positions // (len(known) + 1)
            original = queries[is_unknown]
            quota = len(original)
            ranked = order[i]
            emitted = torch.isin(ranked, queries[~is_unknown])
            distinct = torch.cat((ranked[~emitted], ranked[emitted]))[:quota]
            picks = dict(native=original, objectness=ranked[:quota], separate=distinct)
            if (len(positions) != 50 or len(distinct.unique()) != quota or im in images
                    or not torch.equal(d['scores'][is_unknown], product[i, original])):
                raise ValueError('Changed native scores or output budget')
            # All three lists are fixed before the released annotations are read.
            truth, labels = target['boxes'].cuda(), target['labels'].cuda()
            hidden = truth[torch.isin(labels, proxy)]
            h = factors.best_target(out['pred_boxes'][i], hidden)
            v = factors.best_target(out['pred_boxes'][i], truth[torch.isin(labels, known)])
            counts = dict(images=1, gt=len(hidden), known=len(known_positions), quota=quota)
            for arm, indices in picks.items():
                hit, seen = h[indices], v[indices]
                counts[arm + '_hits'] = len(hit[hit >= 0].unique())
                counts[arm + '_known_only'] = int(((hit < 0) & (seen >= 0)).sum())
            source = sources[im]
            groups[source].update(counts)
            images[im] = dict(source=source, counts=counts, known_positions=known_positions.tolist(),
                              known_scores=d['scores'][~is_unknown].tolist(), objectness=obj[i].tolist(),
                              product=product[i].tolist(), hidden_gt=h.tolist(), known_gt=v.tolist(),
                              picks={a: q.tolist() for a, q in picks.items()})
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches', flush=True)
    if set(images) != set(sources):
        raise ValueError('Incomplete independent slice')
    return dict(images=images, sources=groups)


def main():
    plan, manifest, sources = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    known = torch.tensor(manifest['stages'][0]['classes'], device='cuda')
    proxy = torch.tensor(manifest['stages'][1]['classes'], device='cuda')
    results, passed = {}, {}
    for seed, job in plan['jobs'].items():
        path = OUTPUT / f'seed{seed}-fresh1024.json'
        if not path.exists():
            checkpoint = Path(job['parent'])
            args = load_checkpoint_metadata(checkpoint)['args']
            if (args.owod_stage != 0 or args.discovery_scoring != 'focal' or args.owod_max_detections != 50
                    or list(args.owod_known_class_ids) != known.tolist()):
                raise ValueError('Unexpected Task 1 detector')
            model, _ = load_detector(args, checkpoint, torch.device('cuda'))
            model.eval()
            post = DiscoveryPostProcess(known_ids=args.owod_known_class_ids, max_detections=50, scoring='focal',
                                        unknown_threshold=args.unknown_threshold)
            _, loader = make_loader(root, OUTPUT / 'fresh1024.json', plan['batch_size'], 2)
            print(f'Seed {seed}: confirming known-query exclusion on fresh1024', flush=True)
            result = evaluate(model, post, loader, known, proxy, sources)
            write_json(path, dict(plan_sha256=digest, **result))
            del model, result
        saved = read_json(path)
        if (saved['plan_sha256'] != digest or len(saved['images']) != 1024
                or any(saved['sources'][s]['images'] != 512 for s in SOURCES)):
            raise ValueError('Incomplete independent endpoint')
        pooled = Counter()
        for counts in saved['sources'].values():
            pooled.update(counts)
        gain = pooled['objectness_hits'] - pooled['native_hits']
        cost = pooled['objectness_known_only'] - pooled['native_known_only']
        passed[seed] = (gain > 0 and cost > 0 and 2 * (pooled['separate_hits'] - pooled['native_hits']) >= gain
                        and 2 * (pooled['objectness_known_only'] - pooled['separate_known_only']) >= cost)
        results[seed] = dict(sources=saved['sources'], pooled=pooled)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=results,
                                                passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
