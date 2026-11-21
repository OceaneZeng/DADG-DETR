"""Read out released classes within frozen slots at three points in the retrieval path."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import probe_task2_history_retrieval as retrieval
from tools.owod import probe_task2_slot_naming as naming
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task2-slot-features'
VIEWS = ('decoder', 'projection', 'history_mix')


def unit_mean(vectors):
    return F.normalize(F.normalize(torch.stack(vectors), dim=1).mean(0), dim=0)


def prototypes(hits, view, min_images):
    images = defaultdict(lambda: defaultdict(list))
    for image, slot, label, vectors in hits:
        images[slot, label][image].append(vectors[view])
    by_slot = defaultdict(dict)
    for (slot, label), by_image in sorted(images.items()):
        if len(by_image) >= min_images:
            by_slot[slot][label] = unit_mean([unit_mean(vs) for vs in by_image.values()])
    return {slot: (list(classes), torch.stack(list(classes.values()))) for slot, classes in by_slot.items()
            if len(classes) >= 2}


def read_features(train, test, voted, votes, total_gt, min_images):
    centers = {view: prototypes(train, i, min_images) for i, view in enumerate(VIEWS)}
    counts = {view: Counter(correct=0, rescued=0, lost=0, distinguished=0) for view in VIEWS}
    by_class = defaultdict(Counter)
    by_slot = defaultdict(Counter)
    selective = Counter(accepted=0, correct=0)
    for _, slot, actual, vectors in test:
        base = voted[slot]
        was_correct = base == actual
        by_slot[slot][actual] += 1
        by_class[actual]['named'] += 1
        by_class[actual]['vote'] += int(was_correct)
        support = votes[slot]
        if support[base] >= min_images and 3 * support[base] >= 2 * sum(support.values()):
            selective['accepted'] += 1
            selective['correct'] += int(was_correct)
        for i, view in enumerate(VIEWS):
            prediction = base
            if slot in centers[view]:
                labels, matrix = centers[view][slot]
                best = torch.topk(matrix @ F.normalize(vectors[i], dim=0), 2)
                if best.values[0] - best.values[1] > 1e-6:
                    prediction = labels[int(best.indices[0])]
                    counts[view]['distinguished'] += 1
            correct = prediction == actual
            counts[view]['correct'] += int(correct)
            counts[view]['rescued'] += int(correct and not was_correct)
            counts[view]['lost'] += int(was_correct and not correct)
            by_class[actual][view] += int(correct)
    original_vote = sum(row['vote'] for row in by_class.values())
    for row in counts.values():
        if row['correct'] != original_vote + row['rescued'] - row['lost']:
            raise ValueError('Unbalanced paired gains and losses')
        row['passed'] = (row['correct'] > original_vote and 3 * row['correct'] >= 2 * len(test)
                         and 20 * row['correct'] >= total_gt)
    selective['passed'] = (selective['accepted'] > 0 and 3 * selective['correct'] >= 2 * selective['accepted']
                           and 20 * selective['correct'] >= total_gt)
    return dict(gt=total_gt, named=len(test), vote_correct=original_vote, views=counts, selective_vote=selective,
                by_class=by_class, holdout_slot_class_counts=by_slot,
                fixed_slot_ceiling_correct=sum(max(cs.values()) for cs in by_slot.values()))


def measure_seed(seed, entry, loader, holdout_ids, ref, current, image_root):
    model, bank, original, calibration, ids = naming.seed_inputs(seed, entry, holdout_ids, current, image_root)
    train, test = [], []
    print(f'Seed {seed}: calibration on {len(ids)} images', flush=True)
    cal = retrieval.measure(model, calibration, current, bank, original, ids, named_features=train)
    min_images = bank.settings['independent_images']
    voted, votes = naming.name_from_votes([(i, s, c) for i, s, c, _ in train], original, min_images)
    if (json.loads(json.dumps(votes)) != ref['calibration_votes'] or json.loads(json.dumps(voted)) != ref['vote_names']
            or cal['totals'] != ref['calibration_totals']):
        raise ValueError(f'Calibration or paired vote reference changed for seed {seed}')
    print(f'Seed {seed}: measuring within-slot features on {len(holdout_ids)} holdout images', flush=True)
    hold = retrieval.measure(model, loader, current, bank, original, holdout_ids, named_features=test)
    result = read_features(train, test, voted, votes, hold['totals']['gt'], min_images)
    if (any(result[k] != ref[k] for k in ('named', 'gt', 'vote_correct', 'fixed_slot_ceiling_correct'))
            or hold['totals']['correct'] != ref['original_correct']):
        raise ValueError(f'Holdout hits differ from the completed slot-naming comparison for seed {seed}')
    return dict(calibration_images=len(ids), holdout_images=len(holdout_ids), **result)


def main():
    source = read_json(naming.OUTPUT / 'plan.json')
    ref = read_json(naming.OUTPUT / 'comparison.json')
    annotation = holdout.OUTPUT / 'heldout.json'
    if (ref['status'] != 'complete' or ref['plan_sha256'] != file_sha256(naming.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(MANIFEST) or source['holdout_sha256'] != file_sha256(annotation)
            or source['gpu'] != torch.cuda.get_device_name() or source['torch_version'] != str(torch.__version__)):
        raise ValueError('The completed slot-naming comparison and original environment are required')
    for entry in source['entries'].values():
        for name, digest in [('parent', 'parent_sha256'), ('cache_path', 'cache_sha256'), ('screen', 'screen_sha256')]:
            if file_sha256(Path(entry[name])) != entry[digest]:
                raise ValueError(f'Changed registered input: {entry[name]}')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=ref['plan_sha256'], source_result_sha256=file_sha256(naming.OUTPUT / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], holdout_sha256=source['holdout_sha256'], entries=source['entries'],
                gpu=source['gpu'], torch_version=source['torch_version'], batch_size=2, paper_comparable=False,
                views=list(VIEWS), rule='Within-slot nearest cosine class prototype, at least two independent images, equal image weights. Fall back to vote with fewer than two classes or top scores within 1e-6.',
                criterion='Same view beats vote on both seeds, precision >= 2/3 and correct coverage >= 5%; report selective vote separately at 2/3 calibration support.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    current = sorted(manifest['stages'][1]['classes'])
    root = resolve_image_root(MANIFEST, manifest)
    dataset, loader = make_loader(root, annotation, 2, 2)
    rows = {}
    for seed, entry in source['entries'].items():
        path = OUTPUT / f'seed{seed}.json'
        if not path.is_file():
            result = measure_seed(seed, entry, loader, set(dataset.ids), ref['rows'][seed], current, root)
            write_json(path, dict(plan_sha256=digest, **result))
        row = read_json(path)
        if row['plan_sha256'] != digest or row['holdout_images'] != len(dataset):
            raise ValueError(f'Unexpected seed result: {path}')
        rows[seed] = row
    passed = [view for view in VIEWS if all(r['views'][view]['passed'] for r in rows.values())]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows, passed_views=passed,
                                               selective_vote_passed=all(r['selective_vote']['passed'] for r in rows.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
