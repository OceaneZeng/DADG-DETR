"""Compare student, flipped-teacher and midpoint coordinates for the same selected queries."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery_candidates import select_distinct_boxes
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_quality_views as views
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task1-view-boxes'


def overlap(a, b):
    return box_iou(box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b))[0]


def replace_coordinates(student, proposed, known):
    size = proposed[:, 2:]
    valid = torch.isfinite(proposed).all(1) & (size > .005).all(1) & (size.prod(1) < .85)
    if len(known):
        valid &= (overlap(proposed, known) <= .3).all(1)
    # Queries arrive in their original score order. Each accepted change must
    # respect NMS with both the already updated and the remaining student boxes.
    old_conflict = (overlap(proposed, student) > .7).fill_diagonal_(False)
    new_conflict = (overlap(proposed, proposed) > .7).fill_diagonal_(False)
    accepted = torch.zeros_like(valid)
    for j in valid.nonzero().flatten().tolist():
        accepted[j] = ~torch.where(accepted, new_conflict[j], old_conflict[j]).any()
    return torch.where(accepted[:, None], proposed, student), valid, accepted


@torch.inference_mode()
def evaluate(model, criterion, ema, loader, new_ids, previous):
    known_ids = torch.tensor(model.known_ids, device='cuda')
    new_ids = torch.tensor(new_ids, device='cuda')
    rows, totals = {}, Counter()
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            image = int(target['image_id'].item())
            ref = previous[str(image)]
            score, pool = soft.continuous_evidence(output, mined, traces[i], i)
            boxes = output['pred_boxes'][i]
            idx = select_distinct_boxes(boxes, pool, score, int(mined['pseudo'][i].sum()))
            ids = idx.tolist()
            if sorted(ids) != ref['ids']['soft'] or len(ids) != ref['quota']:
                raise ValueError(f'Cannot reproduce fixed query identities on {image}')
            partner = mined['matches'][i, idx]
            if (partner < 0).any():
                raise ValueError(f'Unmatched teacher on {image}')
            student = boxes[idx].cpu()
            teacher = mined['teacher_boxes'][i, partner].cpu()
            known = target['boxes'][torch.isin(target['labels'], known_ids)].cpu()
            midpoint = (student + teacher) * .5
            alternatives = {'teacher': teacher, 'midpoint': midpoint}
            actual = {'student': student}
            legal, accepted = {}, {}
            for arm, proposed in alternatives.items():
                actual[arm], legal[arm], accepted[arm] = replace_coordinates(student, proposed, known)
            # The later released boxes enter only after deployed coordinates are fixed.
            truth = target['boxes'][torch.isin(target['labels'], new_ids)].cpu()
            mask = torch.ones(len(ids), dtype=torch.bool)
            edges = {arm: overlap(coords, truth) >= .5 for arm, coords in actual.items()}
            oracle = edges['student'].clone()
            for arm, proposed in alternatives.items():
                oracle |= legal[arm][:, None] & (overlap(proposed, truth) >= .5)
            edges['upper_bound'] = oracle
            hits = {a: gates.measure_boxes(e, mask, len(ids))['one_to_one'] for a, e in edges.items()}
            if hits['student'] != ref['hits']['soft'] or hits['upper_bound'] < max(hits.values()):
                raise ValueError(f'Unexpected reference matching or upper bound on {image}')
            rows[image] = dict(gt=len(truth), query_ids=ids, partner_ids=partner.tolist(), hits=hits,
                               accepted={a: v.tolist() for a, v in accepted.items()})
            totals.update(images=1, gt=len(truth), proposals=len(ids), original=ref['hits']['original'], **hits)
            for arm in alternatives:
                totals[arm + '_accepted'] += int(accepted[arm].sum())
                delta = hits[arm] - hits['student']
                totals[arm + '_gains'] += max(0, delta)
                totals[arm + '_losses'] += max(0, -delta)
        if step % 64 == 0 or step == len(loader):
            print(f'Evaluation {step}/{len(loader)} batches, {len(rows)} images', flush=True)
    if set(rows) != set(loader.dataset.ids):
        raise ValueError('Incomplete slice')
    return dict(counts=totals, images=rows)


def paired_plan(parent, rule, criterion, *, arms=()):
    source = read_json(parent / 'plan.json')
    report = read_json(parent / 'comparison.json')
    manifest = read_json(MANIFEST)
    large = holdout.OUTPUT / 'heldout.json'
    digest = file_sha256(parent / 'plan.json')
    if (report['status'] != 'complete' or report['plan_sha256'] != digest
            or file_sha256(MANIFEST) != source['manifest_sha256'] or file_sha256(large) != source['holdout_sha256']
            or torch.cuda.get_device_name() != source['gpu'] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Original paired study and identical detector environment required')
    references = {}
    for seed, job in source['jobs'].items():
        for name in ('parent', 'small'):
            if file_sha256(Path(job[name])) != job[name + '_sha256']:
                raise ValueError(f'Changed {name} for seed {seed}')
        for label in views.SLICES:
            for arm in arms or (None,):
                suffix = f'-{arm}' if arm else ''
                path = parent / f'seed{seed}-{label}{suffix}.json'
                ref = read_json(path)
                counts = report['rows'][seed][label]
                if ref['plan_sha256'] != digest or ref['counts'] != (counts[arm] if arm else counts):
                    raise ValueError(f'Changed paired reference {path}')
                references[path.name] = file_sha256(path)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=parent.name, source_plan_sha256=digest, source_result_sha256=file_sha256(parent / 'comparison.json'),
                reference_sha256=references, jobs=source['jobs'], gpu=source['gpu'], torch_version=source['torch_version'],
                manifest_sha256=source['manifest_sha256'], holdout_sha256=source['holdout_sha256'], batch_size=2,
                rule=rule, criterion=criterion, paper_comparable=False)
    return plan, manifest, report, large


def main():
    parent = views.OUTPUT
    plan, manifest, report, large = paired_plan(
        parent, 'Keep the soft-selected queries. Compare student, unflipped teacher and arithmetic midpoint. '
        'Retain student if proposed geometry, known overlap or within-image suppression fails.',
        'Both seeds: midpoint beats student on holdout1024, gains at least 5 pp over original hard '
        'selection there, and matches or beats student on reference128. Teacher is a separate control. '
        'Upper bound uses any of the three valid coordinates per query and relaxes mutual suppression.')
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    image_root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, annotation in zip(views.SLICES, (Path(job['small']), large)):
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                _, loader = make_loader(image_root, annotation, plan['batch_size'], 2)
                print(f'Seed {seed}, {label}: evaluating fixed-query boxes', flush=True)
                result = evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'],
                                  read_json(parent / path.name)['images'])
                write_json(path, dict(plan_sha256=plan_hash, **result))
            saved = read_json(path)
            counts, ref = saved['counts'], report['rows'][seed][label]
            if (saved['plan_sha256'] != plan_hash or counts['gt'] != ref['soft']['gt']
                    or counts['proposals'] != ref['soft']['proposals'] or counts['student'] != ref['soft']['hits']
                    or counts['original'] != ref['original']['hits']):
                raise ValueError(f'Unpaired coordinate comparison for {seed}, {label}')
            rows[seed][label] = counts
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        passed[seed] = (big['midpoint'] > big['student'] and 100 * (big['midpoint'] - big['original']) >= 5 * big['gt']
                        and small['midpoint'] >= small['student'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
