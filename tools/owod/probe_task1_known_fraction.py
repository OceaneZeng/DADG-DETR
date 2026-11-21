"""Transfer a released-only removal fraction to a new single-view proxy slice."""
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from tools.owod import probe_task1_known_confirmation as confirmation
from tools.owod import probe_task1_known_queries as independent
from tools.owod import probe_task1_unknown_factors as factors
from tools.owod import probe_task1_visible_classes as visible
from tools.owod.probe_task1_encoder_transfer import identities
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import source_ids, write_json


OUTPUT = ROOT / 'output/m-instance-task1-known-fraction'


def released_counts(plan, end):
    counts, hashes = {}, {}
    for seed in plan['jobs']:
        path = confirmation.OUTPUT / f'seed{seed}-released.json'
        data = read_json(path)
        if (data['plan_sha256'] != end['plan_sha256'] or data['folds'] != end['rows'][seed]['folds']
                or len(data['images']) != plan['val_images']):
            raise ValueError('Changed released calibration endpoint')
        confirmed = claimed = 0
        for row in data['images'].values():
            for fold in row['folds'].values():
                picks = fold['picks']
                original = set(picks['objectness'])
                claimed += len(original - set(picks['hard']))
                confirmed += len(original - set(picks['mirror']))
        if confirmed != end['rows'][seed]['pooled']['mirror_removed'] or not 0 < confirmed <= claimed:
            raise ValueError('Unexpected released-only removal count')
        counts[seed] = dict(confirmed=confirmed, claimed=claimed)
        hashes[seed] = file_sha256(path)
    return counts, hashes


def prepare():
    src = read_json(confirmation.OUTPUT / 'plan.json')
    end = read_json(confirmation.OUTPUT / 'comparison.json')
    previous = read_json(independent.OUTPUT / 'plan.json')
    released = read_json(visible.OUTPUT / 'plan.json')
    older = read_json(factors.OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    if (end['status'] != 'complete' or end['plan_sha256'] != file_sha256(confirmation.OUTPUT / 'plan.json')
            or src['parent_plan_sha256'] != file_sha256(visible.OUTPUT / 'plan.json')
            or src['previous_result_sha256'] != file_sha256(independent.OUTPUT / 'comparison.json')
            or read_json(independent.OUTPUT / 'comparison.json')['plan_sha256'] != file_sha256(independent.OUTPUT / 'plan.json')
            or released['parent_plan_sha256'] != file_sha256(factors.OUTPUT / 'plan.json')
            or not (src['jobs'] == released['jobs'] == older['jobs'] == previous['jobs'])
            or src['manifest_sha256'] != file_sha256(MANIFEST) or src['gpu'] != torch.cuda.get_device_name()
            or src['torch_version'] != str(torch.__version__)):
        raise ValueError('Expected verified study chain and matching detector environment')
    counts, hashes = released_counts(src, end)
    numerator = sum(c['confirmed'] for c in counts.values())
    denominator = sum(c['claimed'] for c in counts.values())
    blocked = visible.blocked_sources(older, manifest)
    if identities(blocked) != released['excluded_names_sha256']:
        raise ValueError('Changed original exclusion list')
    for path, digest in [(visible.OUTPUT / 'val-data.json', src['val_sha256']),
                         (independent.OUTPUT / 'fresh1024.json', previous['fresh_sha256'])]:
        if file_sha256(path) != digest:
            raise ValueError(f'Changed prior image list {path}')
        blocked.update(source_ids(read_json(path)))
    data, sources = independent.fresh_subset(manifest, blocked, 48)
    encoded = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
    plan = {k: src[k] for k in ('jobs', 'gpu', 'batch_size', 'torch_version', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=confirmation.OUTPUT.name, parent_plan_sha256=end['plan_sha256'],
                parent_result_sha256=file_sha256(confirmation.OUTPUT / 'comparison.json'),
                calibration_sha256=hashes, calibration_counts=counts, numerator=numerator, denominator=denominator,
                selection_seed=48, per_source=512, fresh_sha256=hashlib.sha256(encoded).hexdigest(),
                source_order=list(independent.SOURCES), excluded_names_sha256=identities(blocked),
                rule='On original objectness Top-Q conflicts with known Top-50 entries, remove the shared '
                'released-only confirmed/claimed proportion, rounded half up per image. Rank conflicts by highest '
                'emitted known score, breaking ties by query index. The matched control ranks them by objectness. '
                'Both receive the same highest-objectness never-claimed replacements. Preserve all known entries.',
                criterion='Each seed: confidence retains >=half the positive excess proxy hits, removes >=half '
                'the positive excess known-only overlaps over native, and beats hard removal on hits. Report '
                'separately whether confidence weakly dominates matched objectness on both outcomes with at '
                'least one strict improvement. No tuning using proxy labels.')
    pin(OUTPUT / 'plan.json', plan)
    path = OUTPUT / 'fresh1024.json'
    if not path.exists():
        write_json(path, data)
    if file_sha256(path) != plan['fresh_sha256']:
        raise ValueError('Changed new image slice')
    return plan, manifest, sources


def fraction_picks(ranked, quota, positions, scores, stride, *, numerator, denominator):
    order = ranked.tolist()
    best = {}
    for q, score in zip((positions // stride).tolist(), scores.tolist()):
        best[q] = max(score, best.get(q, 0.))
    original = order[:quota]
    claimed = [q for q in original if q in best]
    n = (2 * numerator * len(claimed) + denominator) // (2 * denominator)
    supply = [q for q in order[quota:] if q not in best][:n]
    if len(supply) != n:
        raise ValueError('Insufficient unclaimed replacements')
    high = sorted(claimed, key=lambda q: (-best[q], q))[:n]
    arms = {}
    for arm, removed in [('confidence', set(high)), ('matched', set(claimed[:n]))]:
        arms[arm] = ranked.new_tensor([q for q in original if q not in removed] + supply)
    return arms


def comparison(results, plan_sha256):
    passed, ranking = {}, {}
    for seed, row in results.items():
        p = row['pooled']
        gain = p['objectness_hits'] - p['native_hits']
        cost = p['objectness_known_only'] - p['native_known_only']
        hit = p['confidence_hits']
        overlaps = p['confidence_known_only']
        passed[seed] = (gain > 0 and cost > 0 and 2 * (hit - p['native_hits']) >= gain
                        and 2 * (p['objectness_known_only'] - overlaps) >= cost and hit > p['separate_hits'])
        dh = hit - p['matched_hits']
        dc = p['matched_known_only'] - overlaps
        ranking[seed] = min(dh, dc) >= 0 and dh + dc > 0
    return dict(status='complete', plan_sha256=plan_sha256, rows=results, passed_by_seed=passed,
                passes_screen=all(passed.values()), ranking_by_seed=ranking)


def main():
    plan, manifest, sources = prepare()
    choose = partial(fraction_picks, numerator=plan['numerator'], denominator=plan['denominator'])
    results = independent.run_endpoints(OUTPUT, plan, manifest, sources, extra_selection=choose)
    write_json(OUTPUT / 'comparison.json', comparison(results, file_sha256(OUTPUT / 'plan.json')))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
