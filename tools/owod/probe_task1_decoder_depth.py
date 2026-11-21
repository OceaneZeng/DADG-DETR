"""Measure extra new-object localization available at earlier decoder layers, before training a selector."""
from collections import Counter, defaultdict
from pathlib import Path

import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task1-decoder-depth'
SCOPES = ('selected', 'soft_pool')


def image_counts(boxes, known, truth, pool, ids):
    layers, queries, _ = boxes.shape
    flat = boxes.reshape(-1, 4)
    size = flat[:, 2:]
    legal = torch.isfinite(flat).all(1) & (size > .005).all(1) & (size.prod(1) < .85)
    if len(known):
        legal &= (paired.overlap(flat, known) <= .3).all(1)
    legal = legal.reshape(layers, queries)
    if (pool & ~legal[-1]).any():
        raise ValueError('Final pool contains an invalid or released-overlap coordinate')
    edges = (paired.overlap(flat, truth) >= .5).reshape(layers, queries, len(truth))
    raw = edges.any(1)
    row = dict(gt=len(truth), quota=len(ids), pool_queries=int(pool.sum()), raw_final=int(raw[-1].sum()),
               raw_earlier_only=int((raw[:-1].any(0) & ~raw[-1]).sum()))
    for number, covered in enumerate(raw, 1):
        row[f'layer{number}_raw_covered'] = int(covered.sum())
        row[f'layer{number}_raw_lost_by_final'] = int((covered & ~raw[-1]).sum())
    allowed = edges & legal[:, :, None]
    selected = torch.zeros_like(pool)
    selected[ids] = True
    # Union by query, before matching: one query can contribute only one box.
    union = allowed.any(0)
    for scope, mask in [('selected', selected), ('soft_pool', pool)]:
        for key, candidate_edges in [('final', allowed[-1]), ('union', union)]:
            row[f'{scope}_{key}'] = gates.measure_boxes(candidate_edges, mask, len(ids))['same_quota']
    return row


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, new_ids, previous):
    known_ids = torch.tensor(model.known_ids, device='cuda')
    new_ids = torch.tensor(new_ids, device='cuda')
    rows, counts = {}, Counter()
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        aux = output.get('aux_outputs')
        if not aux or len(aux) + 1 != model.detr.transformer.decoder.num_layers:
            raise ValueError('All auxiliary decoder coordinates are required')
        coords = torch.stack([a['pred_boxes'] for a in aux] + [output['pred_boxes']]).cpu()
        for i, target in enumerate(targets):
            image = int(target['image_id'].item())
            ref = previous[str(image)]
            score, pool = soft.continuous_evidence(output, mined, traces[i], i)
            ids = select_distinct_boxes(output['pred_boxes'][i], pool, score, int(mined['pseudo'][i].sum())).cpu()
            if ids.tolist() != ref['query_ids']:
                raise ValueError(f'Unpaired final-layer selection at {image}')
            known = target['boxes'][torch.isin(target['labels'], known_ids)].cpu()
            truth = target['boxes'][torch.isin(target['labels'], new_ids)].cpu()
            row = image_counts(coords[:, i], known, truth, pool.cpu(), ids)
            if row['gt'] != ref['gt'] or row['selected_final'] != ref['hits']['student']:
                raise ValueError(f'Final coordinates did not reproduce the frozen reference at {image}')
            rows[image] = row
            counts.update(row)
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluation {step}/{len(loader)} batches, {len(coords)} layers, {len(rows)} images', flush=True)
    if set(rows) != set(loader.dataset.ids):
        raise ValueError('Incomplete decoder diagnostic')
    return dict(counts=counts, images=rows)


def main():
    parent = paired.OUTPUT
    plan, manifest, report, large = paired.paired_plan(
        parent, 'Count raw per-layer GT coverage, and GT seen earlier but missed by every final query. Within the '
        'final selected identities and the final soft pool, compare last-layer and union-of-depth matching '
        'ceilings under the original image quota, one coordinate per query. Keep legal geometry and known-GT '
        'exclusion; the ceilings relax mutual NMS and use held-out GT only for measurement.',
        'A scope merits a cross-depth selector only if its union adds >= 5 pp over its own last-layer ceiling '
        'for both seeds on holdout1024. Otherwise stop depth-only coordinate selection for that scope.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passes = {}, defaultdict(dict)
    for seed, job in plan['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, annotation in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                print(f'Seed {seed}, {label}: measuring decoder localization depth', flush=True)
                measured = evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'],
                                    read_json(parent / path.name)['images'])
                write_json(path, dict(plan_sha256=digest, **measured))
            saved = read_json(path)
            c, ref = saved['counts'], report['rows'][seed][label]
            if (saved['plan_sha256'] != digest or len(saved['images']) != ref['images'] or c['gt'] != ref['gt']
                    or c['selected_final'] != ref['student'] or c['quota'] != ref['proposals']):
                raise ValueError(f'Unexpected decoder endpoint: {seed}, {label}')
            rows[seed][label] = c
        big = rows[seed]['holdout1024']
        for scope in SCOPES:
            passes[scope][seed] = 100 * (big[scope + '_union'] - big[scope + '_final']) >= 5 * big['gt']
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
               passes_by_scope_and_seed=passes, supported_scopes=[s for s in SCOPES if all(passes[s].values())]))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
