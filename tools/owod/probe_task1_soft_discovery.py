"""Measure actual fixed-budget proposals when multiple view gates become continuous evidence."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task1-soft-discovery'
ARMS = ('original', 'hard_rescore', 'soft')


def selections(output, mined, trace, index):
    original = mined['pseudo'][index]
    quota = int(original.sum())
    choices = {'original': original}
    fallback = {}
    score = (mined['saliency'][index] * trace['matched_saliency']).sqrt()
    score = score * mined['consistency'][index] * mined['semantic_weights'][index] * output['geometry_quality'][index]
    soft = trace['eligible'] & (mined['matches'][index] >= 0) & ~mined['background'][index]
    for arm, pool in [('hard_rescore', trace['confident']), ('soft', soft)]:
        indices = select_distinct_boxes(output['pred_boxes'][index], pool & (score > 0), score, quota)
        fallback[arm] = len(indices) != quota
        choices[arm] = original if fallback[arm] else torch.zeros_like(original).scatter_(0, indices, True)
    return choices, fallback


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, current):
    image_rows = {}
    totals = {arm: Counter() for arm in ARMS}
    new_ids = torch.tensor(current, device='cuda')
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        for index, target in enumerate(targets):
            choices, fallback = selections(output, mined, traces[index], index)
            chosen = {arm: mask.cpu() for arm, mask in choices.items()}
            quota = int(chosen['original'].sum())
            image_id = int(target['image_id'].item())
            if image_id in image_rows or any(int(mask.sum()) != quota for mask in chosen.values()):
                raise ValueError(f'Repeated image or unequal proposal counts: {image_id}')
            truth = target['boxes'][torch.isin(target['labels'], new_ids)]
            boxes = output['pred_boxes'][index]
            edges = (box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0] >= .5).cpu()
            union = torch.stack(list(chosen.values())).any(0).nonzero().flatten()
            row = dict(gt=len(truth), quota=quota, query_ids={}, hits={}, fallback=fallback,
                       boxes_cxcywh={int(q): coords for q, coords in zip(union, boxes[union].tolist())})
            for arm, mask in chosen.items():
                row['query_ids'][arm] = mask.nonzero().flatten().tolist()
                stats = gates.measure_boxes(edges, mask, quota)
                hits = row['hits'][arm] = stats['one_to_one']
                diff = hits - row['hits']['original']
                totals[arm].update(gt=len(truth), images=1, proposals=quota, hits=hits,
                                   covered_gt=stats['covered_gt'], positive_queries=stats['positive_queries'],
                                   new_queries=int((mask & ~chosen['original']).sum()),
                                   image_gains=max(diff, 0), image_losses=max(-diff, 0),
                                   fallback_images=int(fallback.get(arm, False)))
            image_rows[image_id] = row
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(image_rows)} images', flush=True)
    if set(image_rows) != set(loader.dataset.ids):
        raise ValueError('Incomplete slice')
    return dict(totals=totals, images=image_rows)


def main():
    ref = read_json(gates.OUTPUT / 'comparison.json')
    source = read_json(gates.OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    large = holdout.OUTPUT / 'heldout.json'
    if (ref['status'] != 'complete' or ref['plan_sha256'] != file_sha256(gates.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(MANIFEST) or source['holdout_sha256'] != file_sha256(large)
            or source['torch_version'] != str(torch.__version__) or source['gpu'] != torch.cuda.get_device_name()):
        raise ValueError('Completed gate audit and its original inputs are required')
    for job in source['jobs'].values():
        for name in ('parent', 'small'):
            if file_sha256(Path(job[name])) != job[name + '_sha256']:
                raise ValueError(f'Changed input: {job[name]}')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=ref['plan_sha256'], source_result_sha256=file_sha256(gates.OUTPUT / 'comparison.json'),
                jobs=source['jobs'], manifest_sha256=source['manifest_sha256'], holdout_sha256=source['holdout_sha256'],
                gpu=source['gpu'], torch_version=source['torch_version'], batch_size=source['batch_size'],
                paper_comparable=False, score='Geometric mean of two saliencies times original consistency, semantic weight and geometry quality.',
                rule='Same per-image count and IoU suppression for all arms. Soft requires a view match and positive score, and excludes known objects, invalid geometry and explicit background. Reuse original on underfill.',
                criterion='Both seeds: soft minus original >= 5 pp on holdout1024, soft beats hard_rescore there, no regression versus original on reference128.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in source['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, anno in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                dataset, loader = make_loader(root, anno, plan['batch_size'], 2)
                print(f'Seed {seed}, {label}: paired proposal selection, {len(dataset)} images', flush=True)
                result = evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'])
                write_json(path, dict(plan_sha256=digest, **result))
            saved = read_json(path)
            totals = saved['totals']
            actual = totals['original']
            previous = ref['rows'][seed][label]['all']
            selected = previous['selected']
            if (saved['plan_sha256'] != digest or actual['images'] != previous['sample']['images']
                    or actual['gt'] != previous['sample']['gt'] or actual['hits'] != selected['one_to_one']
                    or actual['proposals'] != selected['predictions'] or actual['covered_gt'] != selected['covered_gt']
                    or actual['positive_queries'] != selected['positive_queries']):
                raise ValueError(f'Original proposal selection changed for {seed} {label}')
            rows[seed][label] = totals
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        passed[seed] = (100 * (big['soft']['hits'] - big['original']['hits']) >= 5 * big['original']['gt']
                        and big['soft']['hits'] > big['hard_rescore']['hits']
                        and small['soft']['hits'] >= small['original']['hits'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
