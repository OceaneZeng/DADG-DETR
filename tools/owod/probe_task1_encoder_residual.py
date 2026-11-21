"""Try one direct encoder proposal outside the final decoder boxes at fixed image budgets."""
from collections import Counter
from pathlib import Path

import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_encoder_counts as counts
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = ROOT / 'output/m-instance-task1-encoder-residual'


def mirror_indices(masks, row):
    parts = []
    offset = 0
    for mask in masks:
        height, width = mask.shape[-2:]
        valid_width = (~mask[row, 0]).sum()
        x = (valid_width - 1 - torch.arange(width, device=mask.device)).clamp_min(0)
        grid = torch.arange(height, device=mask.device)[:, None] * width + x
        parts.append(offset + grid.flatten())
        offset += height * width
    return torch.cat(parts)


def residual_scores(geometry, other, row, known, decoder):
    boxes = geometry['boxes'][row]
    mirror = mirror_indices(geometry['masks'], row)
    valid = geometry['valid'][row]
    if not torch.equal(valid, other['valid'][row]) or not other['valid'][row, mirror[valid]].all():
        raise ValueError('Unpaired feature-map padding between views')
    flipped = other['boxes'][row, mirror].clone()
    flipped[:, 0] = 1 - flipped[:, 0]
    a, b = box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(flipped)
    intersection = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(1)
    joint = boxes[:, 2:].prod(1) + flipped[:, 2:].prod(1) - intersection
    agreement = intersection / joint.clamp_min(1e-8)
    score = agreement * (geometry['logits'][row].sigmoid() * other['logits'][row, mirror].sigmoid()).sqrt()
    eligible = valid & (agreement >= .5) & (boxes[:, 2:] > .005).all(1) & (boxes[:, 2:].prod(1) < .85)
    # Compare only eligible locations and bound the temporary cell-by-query matrix.
    for ids in torch.where(eligible)[0].split(1024):
        keep = (paired.overlap(boxes[ids], decoder) < .5).all(1)
        if len(known):
            keep &= (paired.overlap(boxes[ids], known) <= .3).all(1)
        eligible[ids] = keep
    return score, eligible


def matching(edges):
    return gates.measure_boxes(edges, torch.ones(len(edges), dtype=torch.bool), len(edges))['one_to_one']


def compare_pool(cell_ids, score, encoder_boxes, truth, edges, original, missing):
    # Choose before reading evaluation overlaps. Empty output quotas stay empty.
    choice = int(cell_ids[score[cell_ids].argmax()]) if len(cell_ids) and len(edges) else None
    candidate_edges = (paired.overlap(encoder_boxes[cell_ids], truth) >= .5)
    reachable = candidate_edges.any(0).cpu()
    hits = upper = original
    hit_ids = []
    if choice is not None:
        actual = (paired.overlap(encoder_boxes[choice:choice + 1], truth) >= .5).cpu()
        hits = matching(torch.cat((edges[:-1], actual)))
        # A single extra box may match at most one GT. The union row represents
        # exactly the possible matches of that slot. Retaining the donor is allowed.
        upper = max(original, matching(torch.cat((edges[:-1], reachable[None]))))
        hit_ids = actual[0].nonzero().flatten().tolist()
    if hits > upper:
        raise ValueError('Selected residual exceeded the one-swap upper bound')
    return dict(choice=choice, cells=len(cell_ids), hits=hits, upper=upper, hits_gt=hit_ids,
                reachable_gt=reachable.nonzero().flatten().tolist(),
                reachable_missed_by_decoder=int((reachable & missing).sum()))


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, new_ids, previous):
    known_ids = torch.tensor(model.known_ids, device='cuda')
    new_ids = torch.tensor(new_ids, device='cuda')
    other = {}

    def capture(module, inputs, output):
        state = output[-1]
        other.update({k: state[k] for k in ('boxes', 'valid', 'logits')})

    hook = ema.detr.transformer.instance_proposals.register_forward_hook(capture)
    images, totals = {}, Counter()
    try:
        for step, (targets, out, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
            geometry = out['geometry']
            for i, target in enumerate(targets):
                im = int(target['image_id'].item())
                ref = previous[str(im)]
                decoder = out['pred_boxes'][i]
                soft_score, soft_pool = soft.continuous_evidence(out, mined, traces[i], i)
                ids = select_distinct_boxes(decoder, soft_pool, soft_score, int(mined['pseudo'][i].sum()))
                seeds = geometry['chosen'][i, geometry['active'][i]]
                if len(ids) != ref['quota'] or seeds.tolist() != ref['encoder_ids']:
                    raise ValueError(f'Changed reference queries or encoder selection on {im}')
                known = target['boxes'][torch.isin(target['labels'], known_ids)]
                score, legal = residual_scores(geometry, other, i, known, decoder)
                pools = dict(seed=seeds[legal[seeds]], dense=torch.where(legal)[0])
                truth = target['boxes'][torch.isin(target['labels'], new_ids)]
                edges = (paired.overlap(decoder[ids], truth) >= .5).cpu()
                original = matching(edges)
                missing = ~(paired.overlap(decoder, truth) >= .5).any(0)
                for layer in out['aux_outputs']:
                    missing &= ~(paired.overlap(layer['pred_boxes'][i], truth) >= .5).any(0)
                missing = missing.cpu()
                if original != ref['soft_hits'] or len(truth) - int(missing.sum()) != ref['raw_any_layer']:
                    raise ValueError(f'Changed final or all-layer coverage at {im}')
                base = dict(gt=len(truth), quota=len(ids), original=original, hard=ref['hard_hits'])
                totals.update(base)
                row = dict(**base, query_ids=ids.tolist(), query_gt=[e.nonzero().flatten().tolist() for e in edges])
                for arm, cell_ids in pools.items():
                    v = compare_pool(cell_ids, score, geometry['boxes'][i], truth, edges, original, missing)
                    row[arm] = v
                    for k in ('cells', 'hits', 'upper', 'reachable_missed_by_decoder'):
                        totals[arm + '_' + k] += v[k]
                    totals[arm + '_swaps'] += int(v['choice'] is not None)
                images[im] = row
            if step == 1 or step % 64 == 0 or step == len(loader):
                print(f'Evaluation {step}/{len(loader)} batches, {len(images)} images', flush=True)
    finally:
        hook.remove()
    if set(images) != set(loader.dataset.ids):
        raise ValueError('Incomplete residual proposal evaluation')
    return dict(counts=totals, images=images)


def main():
    parent = counts.OUTPUT
    plan, manifest, report, large = paired.paired_plan(
        parent, 'Use the original global encoder and frozen EMA. Keep all but the lowest-ranked soft proposal. '
        'Try the highest-ranked eligible encoder seed or dense cell as one replacement. Require mirrored-cell '
        'IoU >= .5, valid geometry, known GT IoU <= .3 and IoU < .5 to every final student decoder box. Rank by '
        'view IoU times geometric mean encoder quality. Keep donor if no eligible cell; hold zero quotas fixed.',
        'Both seeds: dense exceeds unchanged soft and seed-only on holdout1024, gains >=5 pp over original hard '
        'there and does not regress versus unchanged soft on reference128. Report best one-swap-or-keep ceiling.',
        arms=('global',))
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, annotation in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                src = read_json(parent / f'seed{seed}-{label}-global.json')
                print(f'Seed {seed}, {label}: evaluating direct encoder residuals', flush=True)
                result = evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'], src['images'])
                write_json(path, dict(plan_sha256=digest, **result))
            saved = read_json(path)
            c, ref = saved['counts'], report['rows'][seed][label]['global']
            if (saved['plan_sha256'] != digest or len(saved['images']) != ref['images'] or c['gt'] != ref['gt']
                    or c['quota'] != ref['quota'] or c['original'] != ref['soft_hits'] or c['hard'] != ref['hard_hits']):
                raise ValueError(f'Unpaired encoder residual endpoint: {seed}, {label}')
            rows[seed][label] = c
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        passed[seed] = (big['dense_hits'] > max(big['original'], big['seed_hits'])
                        and 100 * (big['dense_hits'] - big['hard']) >= 5 * big['gt']
                        and small['dense_hits'] >= small['original'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
