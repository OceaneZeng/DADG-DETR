"""Trace the coverage limits of a frozen Task 1 pseudo-label pool."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task1_consistency_budget import replay
from tools.owod.probe_task1_discovery_gates import frozen_miner
from tools.owod.probe_task1_m_s_consistency_validation import OUTPUT as SOURCE
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-candidate-capacity')
PARENT_PLAN = '4cc7d20a34045d725da445344d82ec79fac7e0e292cbd1c57ca0ed4db36b223b'
PARENT_RESULT = '06ce1c5c72366f99cb7175c9871c8fb353a96b13e57b6d6a7eb046e33779e883'
UPPER_BOUND_MINIMUM_PP = 1.


def summarize(report):
    groups = {name: Counter(dict(targets=0, selected=0, raw=0, native_pool=0, relaxed_pool=0,
                                native_positive_quota=0, relaxed_positive_quota=0))
              for name in ('all_small', 'under_16_sq', '16_to_32_sq')}
    for row in report['rows']:
        if row['released'] or row['size'] == '32_sq_plus':
            continue
        for name in ('all_small', row['size']):
            group = groups[name]
            group.update(targets=1, selected=row['native'])
            if row['native']:
                continue
            cap = row['capacity']
            group.update(raw=cap['raw'], native_pool=cap['native_pool'], relaxed_pool=cap['relaxed_pool'],
                         native_positive_quota=cap['native_pool'] and cap['positive_quota'],
                         relaxed_positive_quota=cap['relaxed_pool'] and cap['positive_quota'])
    return dict(groups=groups, optimistic_extra_pp={
        name: 100 * groups['all_small'][name] / groups['all_small']['targets']
        for name in ('native_positive_quota', 'relaxed_positive_quota')
    })


def verify_replay(report, reference):
    original_rows = [{k: v for k, v in row.items() if k != 'capacity'} for row in report['rows']]
    if original_rows != reference['rows'] or report['images'] != reference['images']:
        raise ValueError('Changed fixed-budget replay')


def main():
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or gpu != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    original = verified(SOURCE / 'plan.json', PARENT_PLAN)
    previous = verified(SOURCE / 'comparison.json', PARENT_RESULT)
    if (previous['status'] != 'complete' or previous['accepted'] or
            previous['plan_sha256'] != PARENT_PLAN):
        raise ValueError('Expected the completed fixed-budget threshold study')
    references = {}
    for protocol, spec in original['protocols'].items():
        for fold, cohort in spec['slices'].items():
            label = f'{protocol}-{fold}'
            verified(SOURCE / cohort['file'], cohort['sha256'])
            reference = SOURCE / f'{label}.json'
            references[label] = file_sha256(reference)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, parent_plan=PARENT_PLAN, parent_comparison=PARENT_RESULT,
                inputs=original['protocols'], frozen_run_files=references,
                gpu=gpu, batch_size=original['batch_size'], upper_bound_minimum_pp=UPPER_BOUND_MINIMUM_PP)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, spec in plan['inputs'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, spec['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        weights = Path(spec['checkpoint']['path'])
        if file_sha256(weights) != spec['checkpoint']['sha256']:
            raise ValueError(f'Changed frozen checkpoint: {protocol}')
        model, criterion, ema = frozen_miner(weights, manifest)
        if criterion.consistency_threshold != original['native_consistency']:
            raise ValueError(f'Changed native threshold: {protocol}')
        results[protocol] = {}
        for fold, cohort in spec['slices'].items():
            label = f'{protocol}-{fold}'
            original_report = verified(SOURCE / f'{label}.json', references[label])
            if original_report['plan_sha256'] != PARENT_PLAN:
                raise ValueError(f'Changed reference plan for {label}')
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} candidate coverage', flush=True)
                dataset, loader = make_loader(root, SOURCE / cohort['file'], plan['batch_size'], 2)
                report = replay(model, criterion, ema, dataset, loader, audit_capacity=True)
                verify_replay(report, original_report)
                write_json(dest, dict(plan_sha256=digest, **report))
            report = read_json(dest)
            if report['plan_sha256'] != digest:
                raise ValueError(f'Changed candidate coverage plan for {label}')
            verify_replay(report, original_report)
            results[protocol][fold] = summarize(report)
        del model, criterion, ema
    rank_headroom = all(row['optimistic_extra_pp']['relaxed_positive_quota'] >= UPPER_BOUND_MINIMUM_PP
                        for protocol in results.values() for row in protocol.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               paper_comparable=False, slices=results, rank_headroom=rank_headroom))


if __name__ == '__main__':
    main()
