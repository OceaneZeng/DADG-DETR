"""Test redistribution of a fixed Current refill budget across visible classes in one image."""
from collections import Counter
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import evaluate_author_benchmark
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_refill_capacity import HASHES, OUTPUT as SOURCE, SOURCE as SHARED
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import paired_gates
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-refill-allocation')
CRITERION = ('For label-free pooling require on both slices released Current AP50 >= native +1 pp and above '
             'same-class refill, Known above native, Previous and proxy U-Recall unchanged, no worse mean WI '
             'or summed A-OSE, and a reduction of at least 10 pp in the fraction of refill slots lacking '
             'same-class nondifficult GT vs fixed quotas. Investigate learned image-level class presence only '
             'if the released-only tag-assisted reference gains >=1 pp Current above both native and pooling '
             'on each slice. Tag assistance is diagnostic only.')


def prepare():
    src = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != '9e1c6a71dcc60328f8045e653c6b803d105fe11b1df975c57d62750280b5fdc3'
            or result['status'] != 'complete' or result['ranking_opportunity'] or file_sha256(SOURCE/'plan.json') != result['plan_sha256']
            or file_sha256(MANIFEST) != src['manifest_sha256'] or file_sha256(Path(src['checkpoint']['path'])) != src['checkpoint']['sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed quota evidence, source checkpoint, protocol or runtime')
    for name, entry in src['slices'].items():
        if file_sha256(Path(entry['annotation'])) != entry['annotation_sha256'] or file_sha256(SHARED/f'{name}.json') != HASHES[name]:
            raise ValueError(f'Changed native and same-class reference on {name}')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version', 'references')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=file_sha256(SOURCE/'comparison.json'), criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan


def allocate(native, scores, categories, previous, current, present):
    cols = len(categories)
    ids = np.asarray(native)
    current_col = np.isin(categories, current)
    old = np.isin(categories[ids % cols], previous)
    old_max = np.full(len(scores), -np.inf)
    np.maximum.at(old_max, ids[old]//cols, scores.flat[ids[old]])
    veto = current_col & (scores < old_max[:, None])
    keep, drop = ids[~veto.flat[ids]], ids[veto.flat[ids]]
    allowed = np.broadcast_to(current_col, scores.shape) & ~veto
    allowed.flat[ids] = False
    candidates = np.flatnonzero(allowed)
    candidates = candidates[np.argsort(-scores.flat[candidates], kind='stable')]
    slots = len(drop)
    selected = {'native': ids}
    audits = {}
    budget = Counter(drop % cols)
    same = [candidates[candidates % cols == c][:n] for c, n in budget.items()]
    additions = dict(same=np.concatenate(same) if same else drop, pooled=candidates[:slots])
    if present is not None:
        tagged = np.isin(categories[candidates % cols], list(present))
        additions['tagged'] = np.concatenate((candidates[tagged], candidates[~tagged]))[:slots]
    for arm, added in additions.items():
        merged = np.concatenate((keep, np.sort(added)))
        merged = merged[np.argsort(-scores.flat[merged], kind='stable')]
        protected = ~current_col[merged % cols]
        if (len(added) != slots or not np.array_equal(merged[protected], ids[~current_col[ids % cols]])
                or not set(keep).issubset(merged) or len(set(merged)) != len(ids)):
            raise ValueError('Retained outputs, protected sequence or total Current budget changed')
        selected[arm] = merged
        if present is not None:
            audits[arm] = dict(slots=slots, absent=int((~np.isin(categories[added % cols], list(present))).sum()),
                               moved=int(sum((Counter(added % cols)-budget).values())))
    return selected, audits


@torch.inference_mode()
def evaluate(model, ds, loader, manifest, order, reference, parent_summary):
    previous, current = (s['classes'] for s in manifest['stages'][:2])
    folds = dict(released=manifest['stages'][1]['active_classes'], proxy_even=previous+current[::2], proxy_odd=previous+current[1::2])
    post = {k: DiscoveryPostProcess(known_ids=ids, scoring='focal', max_detections=50) for k, ids in folds.items()}
    predictions = {k: {arm: {} for arm in (('native','same','pooled','tagged') if k=='released' else ('native','same','pooled'))}
                   for k in folds}
    records = {}
    model.eval()
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        h, w = sizes.unbind(1)
        boxes = (box_cxcywh_to_xyxy(out['pred_boxes'])*torch.stack((w,h,w,h), 1)[:,None]).cpu()
        for fold, processor in post.items():
            native = processor(out, sizes, include_top_indices=True)
            scores, cats, _ = processor.candidate_scores(out)
            scores, cats = scores.cpu(), cats.cpu()
            for i, target in enumerate(targets):
                image_id = int(target['image_id'])
                present = None
                if fold == 'released':
                    present = {a['category_id'] for a in ds.coco.imgToAnns.get(image_id, []) if not a.get('difficult')}
                selections, audit = allocate(native[i]['top_indices'].tolist(), scores[i].numpy(), cats.numpy(),
                                             previous, current, present)
                if fold == 'released':
                    records[ds.coco.imgs[image_id]['owod_source_id']] = audit
                for arm, indices in selections.items():
                    pos = torch.from_numpy(indices)
                    predictions[fold][arm][image_id] = dict(boxes=boxes[i, pos//len(cats)], labels=cats[pos%len(cats)],
                                                            scores=scores[i].flatten()[pos])
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(records)} images', flush=True)
    spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_fp_sources=True)
    views = {}
    for fold, known in folds.items():
        views[fold] = {}
        active = [c for c in current if c in known]
        for arm in tuple(predictions[fold]):
            print(f'Computing author metrics for {fold}-{arm}', flush=True)
            met = evaluate_author_benchmark(predictions[fold].pop(arm), ds.coco, known, previous, active, spec,
                                            include_current_stats=True)
            stats = {str(k): v for k, v in met.pop('current_class_stats').items()}
            # JSON reference also uses strings for nested confused-with class IDs.
            for row in stats.values():
                src = row['fp_sources']
                src['confused_with'] = {str(k): v for k, v in src['confused_with'].items()}
            if arm in ('native','same'):
                ref = reference['views'][fold+('_native' if arm=='native' else '_refill')]
                if met != ref['metrics'] or stats != ref['current_class_stats']:
                    raise ValueError(f'Reference metrics changed for {fold}-{arm}')
            base = reference['views'][fold+'_native']['metrics']
            if met['previous_ap50'] != base['previous_ap50'] or met.get('u_recall') != base.get('u_recall'):
                raise ValueError('Protected AP or U-Recall differs')
            views[fold][arm] = dict(metrics=met, current_class_stats=stats)
    totals = {arm: dict(sum_counts(records, arm)) for arm in ('same','pooled','tagged')}
    if (len(records)!=1024 or totals['same']['slots']!=parent_summary['slots']
            or totals['same']['absent']!=parent_summary['slots_without_class_gt'] or totals['same']['moved']):
        raise ValueError('Original slot and image-label statistics changed')
    return dict(views=views, images=records, totals=totals)


def sum_counts(records, arm):
    counts = Counter()
    for row in records.values():
        counts.update(row[arm])
    return counts


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    manifest = read_json(MANIFEST)
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    parent = read_json(SOURCE/'comparison.json')['slices']
    summaries = {}
    for name, entry in plan['slices'].items():
        path = OUTPUT/f'{name}.json'
        if not path.exists():
            ann = Path(entry['annotation'])
            ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
            order = [im['owod_source_id'] for im in read_json(ann)['images']]
            print(f'Evaluating Current slot allocation on {name}, {len(ds)} images', flush=True)
            data = evaluate(model, ds, loader, manifest, order, read_json(SHARED/f'{name}.json'), parent[name])
            write_json(path, dict(plan_sha256=digest, **data))
        data = read_json(path)
        if data['plan_sha256'] != digest:
            raise ValueError('Changed allocation endpoint identity')
        views, counts = data['views'], data['totals']
        native, pooled = ({k:v[arm] for k,v in views.items()} for arm in ('native','pooled'))
        delta = {k: pooled['released']['metrics'][k]-native['released']['metrics'][k]
                 for k in ('current_ap50','previous_ap50','known_ap50')}
        costs, known_pass, open_pass = paired_gates(pooled, native, delta)
        ap = {a:r['metrics']['current_ap50'] for a,r in views['released'].items()}
        absence_drop = (counts['same']['absent']-counts['pooled']['absent'])/counts['same']['slots']
        summaries[name] = dict(current_ap50=ap, delta=delta, proxy_deltas=costs, absence_drop=absence_drop,
                               pooled_pass=known_pass and open_pass and ap['pooled']>ap['same'] and absence_drop>=.1,
                               tag_opportunity=ap['tagged']-max(ap['native'],ap['pooled'])>=.01)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                               slices=summaries, pooled_pass=all(v['pooled_pass'] for v in summaries.values()),
                                               tag_opportunity=all(v['tag_opportunity'] for v in summaries.values())))
    print('Current refill allocation comparison saved', flush=True)


if __name__ == '__main__':
    main()
