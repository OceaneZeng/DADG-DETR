"""Compare two training-source slot naming rules on the exact same frozen holdout hits."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from models.graph_local.instance_transfer import sample_loader
from tools.owod import probe_task2_history_retrieval as retrieval
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task2-slot-naming'


def name_from_votes(hits, original, min_images):
    votes = {slot: Counter() for slot in original}
    for _, slot, label in set(hits):
        if slot in original:
            votes[slot][label] += 1
    proposed = original.copy()
    for slot, counts in votes.items():
        top = counts.most_common(2)
        if top and top[0][1] >= min_images and (len(top) == 1 or top[0][1] > top[1][1]):
            proposed[slot] = top[0][0]
    return proposed, votes


def compare(hits, original, proposed, total_gt, current):
    labels_per_slot = defaultdict(Counter)
    by_class = {c: dict(named=0, original=0, vote=0) for c in current}
    for _, slot, label in hits:
        if slot in original:
            labels_per_slot[slot][label] += 1
            by_class[label]['named'] += 1
            by_class[label]['original'] += int(original[slot] == label)
            by_class[label]['vote'] += int(proposed[slot] == label)
    named = sum(c['named'] for c in by_class.values())
    original_correct = sum(c['original'] for c in by_class.values())
    vote_correct = sum(c['vote'] for c in by_class.values())
    # Descriptive upper bound at fixed coverage. Never export an oracle slot-to-label map.
    ceiling = sum(max(counts.values()) for counts in labels_per_slot.values())
    return dict(named=named, gt=total_gt, original_correct=original_correct, vote_correct=vote_correct,
                fixed_slot_ceiling_correct=ceiling, by_class=by_class,
                passes_vote_screen=(vote_correct > original_correct and 3 * vote_correct >= 2 * named
                                    and 20 * vote_correct >= total_gt))


def measure_seed(seed, entry, holdout_loader, holdout_ids, ref, current, image_root, digest):
    bank, original = retrieval.restore_bank(entry, holdout_ids, current)
    args = SimpleNamespace(owod_manifest=str(MANIFEST), owod_screening=entry['screen'], owod_image_root=str(image_root),
                           seed=int(seed), batch_size=2, num_workers=2)
    calibration = sample_loader(args, 1, 'increment_train', bank.settings['query_images'])
    subset = calibration.dataset
    calibration_ids = {subset.dataset.ids[i] for i in subset.indices}
    if len(calibration_ids) != bank.settings['query_images'] or calibration_ids & holdout_ids:
        raise ValueError(f'Incomplete calibration set or holdout overlap for seed {seed}')
    model, _ = load_detector(None, Path(entry['parent']), torch.device('cuda'))
    model.eval()
    print(f'Seed {seed}: calibration on {len(calibration_ids)} images', flush=True)
    calibration_hits = []
    calibration_row = retrieval.measure(model, calibration, current, bank, original, calibration_ids,
                                        slot_hits=calibration_hits)
    proposed, votes = name_from_votes(calibration_hits, original, bank.settings['independent_images'])
    print(f'Seed {seed}: comparing fixed slot labels on {len(holdout_ids)} holdout images', flush=True)
    holdout_hits = []
    baseline = retrieval.measure(model, holdout_loader, current, bank, original, holdout_ids, slot_hits=holdout_hits)
    # Saved JSON uses string class IDs for the confusion matrix and class counts.
    saved_form = json.loads(json.dumps(baseline))
    if any(saved_form[k] != ref[k] for k in ('images', 'by_class', 'confusion')):
        raise ValueError(f'Original per-class holdout retrieval counts changed for seed {seed}')
    result = compare(holdout_hits, original, proposed, baseline['totals']['gt'], current)
    if (result['original_correct'] != ref['totals']['correct'] or result['named'] != ref['totals']['named']):
        raise ValueError(f'Paired objects changed for seed {seed}')
    return dict(plan_sha256=digest, calibration_images=len(calibration_ids), calibration_totals=calibration_row['totals'],
                calibration_votes=votes, original_names=original, vote_names=proposed, holdout_images=baseline['images'],
                **result)


def main():
    source = read_json(retrieval.OUTPUT / 'plan.json')
    reference = read_json(retrieval.OUTPUT / 'comparison.json')
    alignment_plan = retrieval.alignment.OUTPUT / 'plan.json'
    holdout = retrieval.holdout.OUTPUT / 'heldout.json'
    if (reference['status'] != 'complete' or reference['plan_sha256'] != file_sha256(retrieval.OUTPUT / 'plan.json')
            or file_sha256(alignment_plan) != source['source_plan_sha256']
            or file_sha256(MANIFEST) != source['manifest_sha256'] or file_sha256(holdout) != source['annotation_sha256']
            or source['torch_version'] != str(torch.__version__) or source['gpu'] != torch.cuda.get_device_name()):
        raise ValueError('The completed retrieval diagnostic and its verified inputs are required')
    entries = {}
    jobs = read_json(alignment_plan)['jobs']
    for seed, entry in source['entries'].items():
        cmd = jobs[seed]['commands']['matched']['increment']
        screen = cmd[cmd.index('--owod-screening') + 1]
        if (file_sha256(Path(screen)) != jobs[seed]['screen_sha256'] or file_sha256(Path(entry['parent'])) != entry['parent_sha256']
                or file_sha256(Path(entry['cache_path'])) != entry['cache_sha256']):
            raise ValueError(f'Changed calibration inputs for seed {seed}')
        entries[seed] = dict(**entry, screen=screen, screen_sha256=jobs[seed]['screen_sha256'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=reference['plan_sha256'], source_result_sha256=file_sha256(retrieval.OUTPUT / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], holdout_sha256=source['annotation_sha256'], entries=entries,
                gpu=source['gpu'], torch_version=source['torch_version'], batch_size=2, paper_comparable=False,
                rule='Registered named slots only; one vote per image, slot and class. Keep original on a tie or insufficient support.',
                criterion='Both seeds on identical hits: improve accuracy, accuracy >= 2/3 and correct coverage >= 5% of new GT')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    image_root = resolve_image_root(MANIFEST, read_json(MANIFEST))
    dataset, loader = make_loader(image_root, holdout, plan['batch_size'], 2)
    rows = {}
    for seed, entry in entries.items():
        path = OUTPUT / f'seed{seed}.json'
        if not path.is_file():
            row = measure_seed(seed, entry, loader, set(dataset.ids), reference['rows'][seed],
                               source['current_classes'], image_root, digest)
            write_json(path, row)
        row = read_json(path)
        if row['plan_sha256'] != digest or row['holdout_images'] != len(dataset):
            raise ValueError(f'Unexpected seed result: {path}')
        rows[seed] = row
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passes_vote_screen=all(r['passes_vote_screen'] for r in rows.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
