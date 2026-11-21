"""Separate local box ordering from image-wide retrieval for the frozen dense rankers."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_encoder_transfer as transfer
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import source_ids, subset, write_json


OUTPUT = ROOT / 'output/m-instance-task1-encoder-neighborhood'
PARENT = transfer.OUTPUT


def prepare():
    src = read_json(PARENT / 'plan.json')
    end = read_json(PARENT / 'comparison.json')
    digest = file_sha256(PARENT / 'plan.json')
    manifest, files = stage_files(MANIFEST, 0)
    if (end['status'] != 'complete' or end['plan_sha256'] != digest
            or src['manifest_sha256'] != file_sha256(MANIFEST) or src['torch_version'] != str(torch.__version__)
            or src['gpu'] != torch.cuda.get_device_name()):
        raise ValueError('Expected the completed released-class transfer study and original environment')
    val = read_json(PARENT / 'val-data.json')
    classes = manifest['stages'][0]['classes']
    names = source_ids(val)
    groups = dict(held=src['held_classes'], fitted=classes[2::4])
    if (transfer.identities(names) != src['val_names_sha256'] or len(names) != src['val_images']
            or val != subset(read_json(files['increment_train']), names, classes)
            or not set(groups['fitted']) <= set(src['fit_classes']) or len(groups['fitted']) != 5):
        raise ValueError('Changed diagnostic classes or released evaluation data')
    inputs = {}
    for seed, job in src['jobs'].items():
        val_path = PARENT / f'seed{seed}-val.json'
        result = read_json(val_path)
        fit_path = PARENT / f'seed{seed}-fit.json'
        if (file_sha256(Path(job['parent'])) != job['parent_sha256'] or result['plan_sha256'] != digest
                or result['counts'] != end['rows'][seed] or read_json(fit_path)['plan_sha256'] != digest
                or result['fit_sha256'] != end['fit_sha256'][seed] or file_sha256(fit_path) != result['fit_sha256']):
            raise ValueError(f'Changed fitted direction or reference for seed {seed}')
        inputs[seed] = dict(fit_sha256=result['fit_sha256'], val_sha256=file_sha256(val_path))
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=PARENT.name, source_plan_sha256=digest, source_result_sha256=file_sha256(PARENT / 'comparison.json'),
                references=inputs, val_sha256=file_sha256(PARENT / 'val-data.json'), groups=groups, jobs=src['jobs'],
                batch_size=src['batch_size'], val_images=len(names), val_names_sha256=src['val_names_sha256'],
                gpu=src['gpu'], torch_version=src['torch_version'], manifest_sha256=src['manifest_sha256'],
                paper_comparable=False, rule='Same frozen scores and images. Assign cells to the highest-IoU target '
                'in each five-class group. Compare within-object top-1 and positive/near-miss pair ordering; '
                'partition all cells ahead of the first image-wide hit by max IoU below .1 or [.1,.5).',
                criterion='For both seeds on held classes, local top-1 must improve >=5 pp on paired objects '
                'and the majority of contrast top-1 misses on reachable images must have more low-IoU than '
                'near cells ahead. Then consider a label-free region proposal followed by local scoring only.')
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


def rank_details(scores, order, quality, owner):
    pos = quality >= .5
    near = (quality >= .1) & ~pos
    correct = pos[order].nonzero().flatten()
    row = dict(first_hit=None, near_total=int(near.sum()), low_total=int((quality < .1).sum()),
               near_ahead=0, low_ahead=0, objects=[])
    if len(correct):
        ahead = order[:int(correct[0])]
        row.update(first_hit=len(ahead) + 1, near_ahead=int(near[ahead].sum()),
                   low_ahead=int((quality[ahead] < .1).sum()))
    # Using the same ownership and eligibility as the fit makes each object one observation.
    for obj in owner[pos].unique():
        owned = owner == obj
        positives, negatives = scores[pos & owned], scores[near & owned]
        if not len(negatives):
            continue
        negatives = negatives.sort().values
        below = torch.searchsorted(negatives, positives)
        at_most = torch.searchsorted(negatives, positives, right=True)
        twice_wins = int((below + at_most).sum())
        local_order = order[(owned & (pos | near))[order]]
        row['objects'].append(dict(gt=int(obj), positive=len(positives), near=len(negatives),
                                   top1=int(pos[local_order[0]]), twice_wins=twice_wins))
    return row


@torch.inference_mode()
def evaluate(model, ema, loader, direction, excluded, prior):
    totals, images = Counter(), {}
    for im, ids, base, boxes, memory, truth in transfer.candidates(model, ema, loader, excluded):
        overlap = paired.overlap(boxes, truth)
        quality = base.new_zeros(len(ids))
        owner = torch.full_like(ids, -1)
        if len(truth) and len(ids):
            quality, owner = overlap.max(1)
        row = dict(gt=len(truth), cells=len(ids), reachable=int((quality >= .5).any()))
        totals.update(row)
        ref = prior[str(im)] if prior is not None else None
        if ref is not None and any(ref[k] != v for k, v in row.items()):
            raise ValueError(f'Changed residual pool at {im}')
        for arm, scores in [('original', base), ('contrast', memory @ direction)]:
            order = torch.argsort(scores, descending=True, stable=True)
            top = order[:10]
            if ref is not None and (ids[top].tolist() != ref[arm]['ids'] or
                    [e.nonzero().flatten().tolist() for e in (overlap[top] >= .5)] != ref[arm]['gt_ids']):
                raise ValueError(f'Changed {arm} ranking at {im}')
            v = row[arm] = rank_details(scores, order, quality, owner)
            rank = v['first_hit']
            items = v['objects']
            stats = dict(top1=int(rank == 1), top10=int(rank is not None and rank <= 10),
                         missed=int(rank is not None and rank > 1), low_majority=int(v['low_ahead'] > v['near_ahead']),
                         paired=len(items), local_top1=sum(o['top1'] for o in items),
                         object_auc_sum=sum(o['twice_wins'] / (2 * o['near'] * o['positive']) for o in items))
            totals.update({arm + '_' + k: value for k, value in stats.items()})
        images[im] = row
    if set(images) != set(loader.dataset.ids) or totals['original_paired'] != totals['contrast_paired']:
        raise ValueError('Incomplete paired diagnostic')
    return dict(images=images, counts=totals)


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, support = {}, {}
    for seed, job in plan['jobs'].items():
        model, _, ema = gates.frozen_miner(Path(job['parent']), manifest)
        fit = read_json(PARENT / f'seed{seed}-fit.json')
        direction = torch.tensor(fit['direction'], device='cuda', dtype=torch.float32)
        prior = read_json(PARENT / f'seed{seed}-val.json')['images']
        rows[seed] = {}
        for group, classes in plan['groups'].items():
            path = OUTPUT / f'seed{seed}-{group}.json'
            if not path.exists():
                excluded = [c for c in model.known_ids if c not in classes]
                _, loader = make_loader(root, PARENT / 'val-data.json', plan['batch_size'], 2)
                print(f'Seed {seed}, {group}: evaluating local and global ranks', flush=True)
                result = evaluate(model, ema, loader, direction, excluded, prior if group == 'held' else None)
                write_json(path, dict(plan_sha256=digest, group=group, **result))
            saved = read_json(path)
            if saved['plan_sha256'] != digest or saved['group'] != group or set(saved['images']) != set(prior):
                raise ValueError(f'Unpaired diagnostic: {seed}, {group}')
            rows[seed][group] = saved['counts']
        c = rows[seed]['held']
        support[seed] = (c['contrast_paired'] > 0
                         and 100 * (c['contrast_local_top1'] - c['original_local_top1']) >= 5 * c['contrast_paired']
                         and 2 * c['contrast_low_majority'] > c['contrast_missed'])
        del model, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                             supports_local_by_seed=support, supports_local=all(support.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
