"""Confirm emitted known labels across views at a matched replacement count."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.graph_local.discovery_candidates import flip_samples
from tools.owod import probe_task1_known_queries as independent
from tools.owod import probe_task1_visible_classes as visible
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-task1-known-confirmation'


def prepare():
    src = read_json(visible.OUTPUT / 'plan.json')
    digest = file_sha256(visible.OUTPUT / 'plan.json')
    end = read_json(visible.OUTPUT / 'comparison.json')
    last = read_json(independent.OUTPUT / 'plan.json')
    result = read_json(independent.OUTPUT / 'comparison.json')
    if (end['status'] != 'complete' or end['plan_sha256'] != digest or last['parent_plan_sha256'] != digest
            or last['jobs'] != src['jobs'] or result['status'] != 'complete'
            or result['plan_sha256'] != file_sha256(independent.OUTPUT / 'plan.json')
            or src['manifest_sha256'] != file_sha256(MANIFEST) or src['gpu'] != torch.cuda.get_device_name()
            or src['torch_version'] != str(torch.__version__) or file_sha256(visible.OUTPUT / 'val-data.json') != src['val_sha256']):
        raise ValueError('Changed frozen study, released-only validation data or environment')
    refs = {}
    for seed, job in src['jobs'].items():
        path = visible.OUTPUT / f'seed{seed}-released.json'
        saved = read_json(path)
        if (file_sha256(Path(job['parent'])) != job['parent_sha256'] or saved['plan_sha256'] != digest
                or saved['folds'] != end['rows'][seed]['folds'] or len(saved['images']) != src['val_images']):
            raise ValueError(f'Changed original-view predictions for seed {seed}')
        refs[seed] = file_sha256(path)
    plan = {key: src[key] for key in ('jobs', 'gpu', 'torch_version', 'batch_size', 'held_folds', 'val_images',
                                       'val_sha256', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=visible.OUTPUT.name, parent_plan_sha256=digest, reference_sha256=refs,
                previous_run=independent.OUTPUT.name, previous_result_sha256=file_sha256(independent.OUTPUT / 'comparison.json'),
                rule='Each masked fold preserves native known entries and unknown quota. Start from top objectness '
                'queries. Remove a known-emitting query only when a flipped Top-50 entry has the same visible '
                'label and unflipped IoU >=0.5. Fill from the never-claimed queue after the objectness cutoff. '
                'The matched control removes the same number of original claimed queries by highest emitted '
                'known score and receives identical replacements. All-claims removal is the hard reference.',
                criterion='Both seeds: positive excess objectness hits and visible-only overlaps over native. '
                'Mirror retains >=half the hit gain and removes >=half the excess overlaps, beats hard on hits, '
                'and weakly dominates the matched control on hits (up) and visible-only outputs (down), '
                'with at least one strict improvement. Four released-only folds; no Task 2 labels.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


def choose(ref, order, flipped, iou, stride):
    positions = torch.tensor(ref['known_positions'], dtype=torch.long)
    q = positions // stride
    agree = (positions % stride)[:, None] == (flipped % stride)[None, :]
    supported = set(q[((iou[q[:, None], (flipped // stride)[None, :]] >= .5) & agree).any(1)].tolist())
    best = {}
    for query, score in zip(q.tolist(), ref['known_scores']):
        best[query] = max(score, best.get(query, 0.))
    quota = ref['counts']['quota']
    original = order[:quota]
    claimed = set(original).intersection(best)
    mirror = claimed & supported
    matched = set(sorted(claimed, key=lambda j: (-best[j], j))[:len(mirror)])
    supply = [j for j in order[quota:] if j not in best]
    arms = {'objectness': original}
    for name, removed in [('hard', claimed), ('mirror', mirror), ('matched', matched)]:
        picks = [j for j in original if j not in removed] + supply[:len(removed)]
        if len(picks) != quota:
            raise ValueError('Insufficient unclaimed replacements')
        arms[name] = picks
    return arms, len(mirror)


@torch.inference_mode()
def evaluate(model, loader, classes, folds, reference):
    rows, totals = {}, {str(i): Counter() for i in range(len(folds))}
    posts = [DiscoveryPostProcess(known_ids=[c for c in classes if c not in held], max_detections=50,
                                  scoring='focal') for held in folds]
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to('cuda')
        first, flipped = model(samples), model(flip_samples(samples))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        obj = first['pred_objectness'].sigmoid()
        original_boxes = first['pred_boxes'].cpu()
        restored = flipped['pred_boxes'].cpu().clone()
        restored[..., 0] = 1 - restored[..., 0]
        overlaps = [overlap(a, b) for a, b in zip(original_boxes, restored)]
        keys = [str(t['image_id'].item()) for t in targets]
        for i, key in enumerate(keys):
            if key in rows or obj[i].tolist() != reference[key]['objectness']:
                raise ValueError(f'Unpaired original objectness on image {key}')
            rows[key] = dict(boxes=original_boxes[i].tolist(), restored_boxes=restored[i].tolist(), folds={})
        orders = obj.argsort(dim=1, descending=True, stable=True).tolist()
        for fold, post in enumerate(posts):
            f = str(fold)
            a = post(first, sizes, include_top_indices=True)
            b = post(flipped, sizes, include_top_indices=True)
            for i, key in enumerate(keys):
                ref = reference[key]['folds'][f]
                keep = a[i]['labels'] != 91
                if (a[i]['top_indices'][keep].tolist() != ref['known_positions']
                        or a[i]['scores'][keep].tolist() != ref['known_scores']):
                    raise ValueError(f'Known detections changed on image {key} fold {f}')
                flip_positions = b[i]['top_indices'][b[i]['labels'] != 91].cpu()
                arms, n = choose(ref, orders[i], flip_positions, overlaps[i], len(post.known_ids) + 1)
                h, v = ref['hidden_gt'], ref['visible_gt']
                counts = {k: ref['counts'][k] for k in ('images', 'gt', 'known', 'quota', 'native_hits', 'native_visible_only')}
                counts['mirror_removed'] = n
                for name, picks in arms.items():
                    counts[name + '_hits'] = len({h[j] for j in picks} - {-1})
                    counts[name + '_visible_only'] = sum(v[j] >= 0 and h[j] < 0 for j in picks)
                if arms['objectness'] != ref['picks']['objectness']:
                    raise ValueError('Objectness baseline changed')
                totals[f].update(counts)
                rows[key]['folds'][f] = dict(counts=counts, flipped_positions=flip_positions.tolist(), picks=arms)
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} two-view batches', flush=True)
    if set(rows) != set(reference):
        raise ValueError('Incomplete released-only evaluation')
    return dict(images=rows, folds=totals)


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    root = resolve_image_root(MANIFEST, manifest)
    classes = manifest['stages'][0]['classes']
    results, passed = {}, {}
    for seed, job in plan['jobs'].items():
        path = OUTPUT / f'seed{seed}-released.json'
        if not path.exists():
            ckpt = Path(job['parent'])
            args = load_checkpoint_metadata(ckpt)['args']
            if args.owod_stage != 0 or args.discovery_scoring != 'focal' or list(args.owod_known_class_ids) != classes:
                raise ValueError('Expected frozen Task 1 focal checkpoint')
            model, _ = load_detector(args, ckpt, torch.device('cuda'))
            model.eval()
            _, loader = make_loader(root, visible.OUTPUT / 'val-data.json', plan['batch_size'], 2)
            ref = read_json(visible.OUTPUT / path.name)['images']
            print(f'Seed {seed}: evaluating class confirmation on released folds', flush=True)
            data = evaluate(model, loader, classes, plan['held_folds'], ref)
            write_json(path, dict(plan_sha256=digest, **data))
            del data, model, ref
        saved = read_json(path)
        if saved['plan_sha256'] != digest or len(saved['images']) != plan['val_images'] or len(saved['folds']) != 4:
            raise ValueError('Incomplete two-view endpoint')
        p = Counter()
        for counts in saved['folds'].values():
            p.update(counts)
        gain = p['objectness_hits'] - p['native_hits']
        cost = p['objectness_visible_only'] - p['native_visible_only']
        extra_hits = p['mirror_hits'] - p['matched_hits']
        overlaps_saved = p['matched_visible_only'] - p['mirror_visible_only']
        passed[seed] = (gain > 0 and cost > 0 and 2 * (p['mirror_hits'] - p['native_hits']) >= gain
                        and 2 * (p['objectness_visible_only'] - p['mirror_visible_only']) >= cost
                        and p['mirror_hits'] > p['hard_hits']
                        and min(extra_hits, overlaps_saved) >= 0 and extra_hits + overlaps_saved > 0)
        results[seed] = dict(folds=saved['folds'], pooled=p)
        del saved
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=results,
                                                passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
