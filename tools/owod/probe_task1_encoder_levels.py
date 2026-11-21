"""Compare global and within-feature-level proposal priorities through the frozen decoder."""
from collections import Counter
from pathlib import Path

import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from tools.owod import probe_task1_decoder_depth as depth
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task1-encoder-levels'


def covered(boxes, truth):
    return (paired.overlap(boxes, truth) >= .5).any(0)


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, new_ids, previous, control, *, teacher_budget=None):
    known_ids = torch.tensor(model.known_ids, device='cuda')
    new_ids = torch.tensor(new_ids, device='cuda')
    rows, total = {}, Counter()
    for step, (targets, out, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        geometry = out['geometry']
        lengths = [mask[0].numel() for mask in geometry['masks']]
        for i, target in enumerate(targets):
            image = int(target['image_id'].item())
            ref = previous[str(image)]
            quota = ref['quota']
            boxes = out['pred_boxes'][i]
            score, pool = soft.continuous_evidence(out, mined, traces[i], i)
            selected = select_distinct_boxes(boxes, pool, score, quota)
            active = geometry['chosen'][i, geometry['active'][i]]
            dense_boxes = geometry['boxes'][i]
            valid = geometry['valid'][i]
            # The frozen detector and both rankings have finished before scoring.
            known = target['boxes'][torch.isin(target['labels'], known_ids)]
            truth = target['boxes'][torch.isin(target['labels'], new_ids)]
            dense = covered(dense_boxes[valid], truth)
            chosen = covered(dense_boxes[active], truth)
            final = covered(boxes, truth)
            any_layer = final.clone()
            for layer in out['aux_outputs']:
                any_layer |= covered(layer['pred_boxes'][i], truth)
            edges = (paired.overlap(boxes, truth) >= .5).cpu()
            mask = torch.zeros(len(boxes), dtype=torch.bool)
            mask[selected.cpu()] = True
            hits = gates.measure_boxes(edges, mask, quota)['one_to_one']
            hard = mined['pseudo'][i].cpu()
            counts = dict(images=1, gt=len(truth), known_gt=len(known), known_final=int(covered(boxes, known).sum()),
                          quota=quota, proposals=len(selected), active=len(active), queries=len(boxes),
                          dense=int(dense.sum()), chosen=int(chosen.sum()), raw_final=int(final.sum()),
                          raw_any_layer=int(any_layer.sum()), dense_missed_by_decoder=int((dense & ~any_layer).sum()),
                          chosen_missed_by_decoder=int((chosen & ~any_layer).sum()), soft_hits=hits,
                          hard_hits=gates.measure_boxes(edges, hard, int(hard.sum()))['one_to_one'])
            start = 0
            for j, length in enumerate(lengths):
                counts[f'level{j + 1}_seeds'] = int(((active >= start) & (active < start + length)).sum())
                start += length
            if control is None:
                if (counts['raw_final'] != ref['raw_final'] or counts['raw_any_layer'] != ref['raw_final'] + ref['raw_earlier_only']
                        or hits != ref['selected_final'] or len(selected) != quota or int(hard.sum()) != quota):
                    raise ValueError(f'Global control differs from the registered depth reference at {image}')
            else:
                base = control[str(image)]
                if any(counts[k] != base[k] for k in ('gt', 'known_gt', 'dense', 'queries', 'quota')):
                    raise ValueError(f'Unexpected change before proposal selection at {image}')
                counts['budget_mismatch_images'] = int(len(active) != base['active'] or len(selected) != quota)
                counts['replaced_encoder_ids'] = len(set(active.tolist()) - set(base['encoder_ids']))
            if teacher_budget is not None:
                student_counts = geometry['natural_counts'][i]
                teacher_counts, teacher_active = teacher_budget[i]
                if len(active) != min(student_counts) or teacher_active != min(teacher_counts):
                    raise ValueError(f'Encoder did not apply the shared count in both views at {image}')
                counts.update(student_global=student_counts[0], student_within=student_counts[1],
                              teacher_global=teacher_counts[0], teacher_within=teacher_counts[1],
                              teacher_active=teacher_active)
            total.update(counts)
            rows[image] = dict(**counts, encoder_ids=active.tolist())
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluation {step}/{len(loader)} batches, {len(rows)} images', flush=True)
    if set(rows) != set(loader.dataset.ids):
        raise ValueError('Incomplete encoder comparison')
    return dict(counts=total, images=rows)


def main():
    parent = depth.OUTPUT
    plan, manifest, _, large = paired.paired_plan(
        parent, 'Global encoder quality ranking versus within-feature-level rank (raw quality breaks ties). '
        'Use the same priority for the shortlist and NMS. Fixed query budget, frozen student and EMA, '
        'unchanged quality values for decoder scoring. Score both arms at original per-image soft quota.',
        'Both seeds: large slice improves raw new coverage and fixed-budget soft hits over global, '
        'soft hits exceed the global hard control by >= 5 pp, known raw coverage drops by <= 1 pp, '
        'and small slice soft hits do not decrease. Both slices must match per-image active and output counts.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, annotation in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            src = read_json(parent / f'seed{seed}-{label}.json')['images']
            rows[seed][label] = {}
            control = None
            for arm in ('global', 'within_level'):
                for network in (model, ema):
                    network.detr.transformer.instance_proposals.rank_within_level = arm == 'within_level'
                path = OUTPUT / f'seed{seed}-{label}-{arm}.json'
                if not path.exists():
                    _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                    print(f'Seed {seed}, {label}, {arm}: encoder proposal selection', flush=True)
                    result = evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'], src, control)
                    write_json(path, dict(plan_sha256=digest, arm=arm, **result))
                saved = read_json(path)
                if saved['arm'] != arm or saved['plan_sha256'] != digest or set(saved['images']) != set(src):
                    raise ValueError(f'Unpaired encoder result: {seed}, {label}, {arm}')
                rows[seed][label][arm] = saved['counts']
                if arm == 'global':
                    control = saved['images']
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        a, b = big['global'], big['within_level']
        passed[seed] = (b['raw_final'] > a['raw_final'] and b['soft_hits'] > a['soft_hits']
                        and 100 * (b['soft_hits'] - a['hard_hits']) >= 5 * a['gt']
                        and 100 * (a['known_final'] - b['known_final']) <= a['known_gt']
                        and small['within_level']['soft_hits'] >= small['global']['soft_hits']
                        and b['budget_mismatch_images'] == small['within_level']['budget_mismatch_images'] == 0)
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
