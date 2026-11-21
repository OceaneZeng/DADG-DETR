"""Measure the cost of known-class overlap when ranking unknowns by objectness."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import probe_task1_encoder_transfer as transfer
from tools.owod import probe_task1_unknown_factors as factors
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, source_ids, subset, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-task1-visible-classes'


def blocked_sources(parent, manifest):
    blocked = set(manifest['evaluation']['image_source_ids'])
    for path, expected in [(factors.SLICES['fresh1024'], parent['fresh_sha256']),
                           (factors.SLICES['reference1024'], parent['holdout_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError(f'Changed prior evaluation slice {path}')
        blocked.update(source_ids(read_json(path)))
    prior_val = source_ids(read_json(transfer.OUTPUT / 'val-data.json'))
    if transfer.identities(prior_val) != read_json(transfer.OUTPUT / 'plan.json')['val_names_sha256']:
        raise ValueError('Changed prior released-only validation slice')
    blocked.update(prior_val)
    for job in parent['jobs'].values():
        screen = Path(job['screen'])
        if file_sha256(screen) != job['screen_sha256'] or file_sha256(Path(job['parent'])) != job['parent_sha256']:
            raise ValueError('Changed parent training input or checkpoint')
        for stage in (0, 1):
            _, files = screening_files(MANIFEST, screen, stage)
            for path in set(files.values()):
                blocked.update(source_ids(read_json(path)))
    return blocked


def prepare():
    parent = read_json(factors.OUTPUT / 'plan.json')
    result = read_json(factors.OUTPUT / 'comparison.json')
    parent_hash = file_sha256(factors.OUTPUT / 'plan.json')
    manifest, official = stage_files(MANIFEST, 0)
    classes = manifest['stages'][0]['classes']
    if (result['status'] != 'complete' or result['plan_sha256'] != parent_hash
            or not result['released_only_followup']['objectness'] or file_sha256(MANIFEST) != parent['manifest_sha256']
            or parent['gpu'] != torch.cuda.get_device_name() or parent['torch_version'] != str(torch.__version__)):
        raise ValueError('Expected completed factor diagnosis with matching inputs and environment')
    blocked = blocked_sources(parent, manifest)
    data = read_json(official['increment_train'])
    names = set(sorted(source_ids(data) - blocked,
                       key=lambda name: hashlib.sha256(f'46:{name}'.encode()).digest())[:1024])
    if len(names) != 1024 or len(classes) != 20:
        raise ValueError('Incomplete released-only slice')
    val = subset(data, names, classes)
    encoded = (json.dumps(val, indent=2, sort_keys=True) + '\n').encode()
    plan = {key: parent[key] for key in ('jobs', 'batch_size', 'gpu', 'torch_version', 'manifest_sha256',
                                         'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=factors.OUTPUT.name, parent_plan_sha256=parent_hash,
                parent_result_sha256=file_sha256(factors.OUTPUT / 'comparison.json'),
                selection_seed=46, excluded_names_sha256=transfer.identities(blocked),
                val_images=len(names), val_names_sha256=transfer.identities(names),
                val_sha256=hashlib.sha256(encoded).hexdigest(), held_folds=[classes[i::4] for i in range(4)],
                rule='Detector trained on all 20 released Task 1 classes. Mask five classifier columns per fold, '
                'using native joint Top-50 over 15 visible classes plus unknown. Preserve the known entries and '
                'unknown count; rank unknowns by native product, objectness, or objectness times one minus '
                'the largest visible-class sigmoid. Each GT is held in exactly one fold; no NMS or fitting.',
                criterion='In both seeds pooled across folds, objectness must improve hidden-GT hits over native '
                'and increase visible-only overlaps. Adjusted scores must retain at least half the extra hits '
                'and eliminate at least half the extra visible-only outputs. Report all folds separately.')
    pin(OUTPUT / 'plan.json', plan)
    path = OUTPUT / 'val-data.json'
    if not path.exists():
        write_json(path, val)
    if file_sha256(path) != plan['val_sha256']:
        raise ValueError('Changed released-only validation annotations')
    return plan, manifest


@torch.inference_mode()
def evaluate(model, loader, classes, folds):
    rows = {}
    totals = {str(i): Counter() for i in range(len(folds))}
    visible = [[c for c in classes if c not in held] for held in folds]
    posts = [DiscoveryPostProcess(known_ids=ids, max_detections=50, scoring='focal') for ids in visible]
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        obj = out['pred_objectness'].sigmoid()
        nov = out['pred_novelty'].sigmoid()
        for i, target in enumerate(targets):
            key = str(target['image_id'].item())
            if key in rows:
                raise ValueError(f'Duplicate image {key}')
            rows[key] = dict(objectness=obj[i].tolist(), novelty=nov[i].tolist(), folds={})
        for fold, (ids, post) in enumerate(zip(visible, posts)):
            detections = post(out, sizes, include_top_indices=True)
            strongest = out['pred_logits'][..., ids].amax(-1).sigmoid()
            adjusted = obj * (1 - strongest)
            for i, target in enumerate(targets):
                key = str(target['image_id'].item())
                d = detections[i]
                is_unknown = d['labels'] == 91
                positions = d['top_indices']
                native = positions[is_unknown] // (len(ids) + 1)
                quota = len(native)
                picks = dict(native=native, objectness=torch.argsort(obj[i], descending=True, stable=True)[:quota],
                             adjusted=torch.argsort(adjusted[i], descending=True, stable=True)[:quota])
                if len(positions) != 50 or not torch.equal(d['scores'][is_unknown], (obj[i] * nov[i])[native]):
                    raise ValueError('Changed masked native Top-50')
                # Use annotations only after all three rankings have been fixed.
                labels, truth = target['labels'].cuda(), target['boxes'].cuda()
                seen = torch.isin(labels, labels.new_tensor(ids))
                hidden_gt = factors.best_target(out['pred_boxes'][i], truth[~seen])
                visible_gt = factors.best_target(out['pred_boxes'][i], truth[seen])
                counts = dict(images=1, gt=int((~seen).sum()), quota=quota, known=len(positions)-quota)
                for arm, queries in picks.items():
                    h, v = hidden_gt[queries], visible_gt[queries]
                    counts[arm + '_hits'] = len(h[h >= 0].unique())
                    counts[arm + '_visible_only'] = int(((v >= 0) & (h < 0)).sum())
                totals[str(fold)].update(counts)
                rows[key]['folds'][str(fold)] = dict(counts=counts, visible_max=strongest[i].tolist(),
                                                    known_positions=positions[~is_unknown].tolist(),
                                                    known_scores=d['scores'][~is_unknown].tolist(),
                                                    hidden_gt=hidden_gt.tolist(), visible_gt=visible_gt.tolist(),
                                                    picks={arm: q.tolist() for arm, q in picks.items()})
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches across four folds', flush=True)
    if set(rows) != {str(im) for im in loader.dataset.ids}:
        raise ValueError('Incomplete independent evaluation')
    return dict(images=rows, folds=totals)


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    classes = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFEST, manifest)
    results, passed = {}, {}
    for seed, job in plan['jobs'].items():
        path = OUTPUT / f'seed{seed}-released.json'
        if not path.exists():
            checkpoint = Path(job['parent'])
            args = load_checkpoint_metadata(checkpoint)['args']
            if (args.owod_stage != 0 or args.discovery_scoring != 'focal' or args.owod_max_detections != 50
                    or list(args.owod_known_class_ids) != classes):
                raise ValueError('Unexpected Task 1 detector')
            model, _ = load_detector(args, checkpoint, torch.device('cuda'))
            model.eval()
            _, loader = make_loader(root, OUTPUT / 'val-data.json', plan['batch_size'], 2)
            print(f'Seed {seed}: scoring four visible-class folds', flush=True)
            result = evaluate(model, loader, classes, plan['held_folds'])
            write_json(path, dict(plan_sha256=digest, **result))
            del model, result
        saved = read_json(path)
        if (saved['plan_sha256'] != digest or len(saved['images']) != plan['val_images']
                or len(saved['folds']) != 4 or any(s['images'] != plan['val_images'] for s in saved['folds'].values())):
            raise ValueError(f'Incomplete released-only evaluation for seed {seed}')
        pooled = Counter()
        for counts in saved['folds'].values():
            pooled.update(counts)
        gain = pooled['objectness_hits'] - pooled['native_hits']
        cost = pooled['objectness_visible_only'] - pooled['native_visible_only']
        passed[seed] = (gain > 0 and cost > 0 and 2 * (pooled['adjusted_hits'] - pooled['native_hits']) >= gain
                        and 2 * (pooled['objectness_visible_only'] - pooled['adjusted_visible_only']) >= cost)
        results[seed] = dict(folds=saved['folds'], pooled=pooled)
        del saved
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=results,
                                                passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
