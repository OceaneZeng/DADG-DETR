"""Swap old-class classifier rows between the two frozen Task 2 finetuning arms."""

from pathlib import Path
import subprocess

import torch

from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_previous_hard_negatives import (
    MANIFESTS, METRICS, OUTPUT as PARENT, compare, evaluate,
)
from tools.owod.screening_data import write_json


OUTPUT = PARENT.with_name('task2-m-s-previous-hard-negative-head')
PARENT_SHA = 'e9e036c003d35656857dc488c842a1a1733815f022bd193a2d9aec6fbb8342e6'
REFERENCE_SHAS = {
    'm-owodb': '037dec508eaf3ff59539b85fe2160eeee8216f7e9f92ece92300ab2eca2e6b54',
    's-owodb': '454ba79b5183afdebec24dcfd48aa594f923f4be8d9d0c6dd5ecbacebd796109',
}
MIN_SHARE = .5


def summarize(control, candidate, restored, transferred):
    arms = dict(control=control, candidate=candidate, restored=restored, transferred=transferred)
    scores = {arm: {k: row['views']['native']['metrics'][k] for k in METRICS} for arm, row in arms.items()}
    previous = {arm: row['previous_ap50'] for arm, row in scores.items()}
    decline = previous['control'] - previous['candidate']
    fractions = None
    if decline > 0:
        fractions = dict(recovered=(previous['restored'] - previous['candidate']) / decline,
                         transferred=(previous['control'] - previous['transferred']) / decline)
    return dict(metrics=scores, previous_ap50_decline=decline, fractions=fractions,
                old_rows_explain_most=None if fractions is None else min(fractions.values()) >= MIN_SHARE,
                restored_vs_control=compare(control, restored),
                restoration_vs_candidate=compare(candidate, restored),
                transferred_vs_control=compare(control, transferred))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    result = verified(PARENT / 'comparison.json', PARENT_SHA)
    parent = verified(PARENT / 'plan.json', result['plan_sha256'])
    if result['status'] != 'complete' or result['follow_up']:
        raise ValueError('Expected the rejected paired training screen')
    sources = {}
    for protocol, job in parent['jobs'].items():
        reference = PARENT / f'{protocol}.json'
        row = verified(reference, REFERENCE_SHAS[protocol])
        candidate = PARENT / protocol / 'finetune/checkpoint.pth'
        control = Path(job['endpoints']['finetune']['path'])
        if (row['plan_sha256'] != result['plan_sha256'] or row['comparisons'] != result['protocols'][protocol]
                or file_sha256(candidate) != row['candidate_sha256']
                or file_sha256(control) != row['control_sha256']
                or row['control_sha256'] != job['endpoints']['finetune']['sha256']):
            raise ValueError(f'Changed frozen endpoints: {protocol}')
        sources[protocol] = dict(reference_sha256=REFERENCE_SHAS[protocol],
                                 candidate_sha256=row['candidate_sha256'], control_sha256=row['control_sha256'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=PARENT_SHA, parent_plan_sha256=result['plan_sha256'],
                sources=sources, min_share=MIN_SHARE, paper_comparable=False, max_detections=50,
                evaluation='same four frozen training-source folds and original scoring',
                intervention='swap only final linear classifier previous-class rows and biases in both directions')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for protocol, job in parent['jobs'].items():
        marker = OUTPUT / f'{protocol}.json'
        if not marker.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            root = resolve_image_root(manifest_path, manifest)
            reference = verified(PARENT / marker.name, sources[protocol]['reference_sha256'])
            control = Path(job['endpoints']['finetune']['path'])
            candidate = PARENT / protocol / 'finetune/checkpoint.pth'
            print(f'{protocol}: restore the control previous-class rows in the candidate', flush=True)
            restored = evaluate(candidate, root, manifest, job, parent, previous_source=control)
            print(f'{protocol}: transfer the candidate previous-class rows to the control', flush=True)
            transferred = evaluate(control, root, manifest, job, parent, previous_source=candidate)
            folds = {fold: summarize(reference['control'][fold], reference['candidate'][fold],
                                     restored[fold], transferred[fold]) for fold in job['folds']}
            write_json(marker, dict(plan_sha256=digest, folds=folds, restored=restored, transferred=transferred))
        report = read_json(marker)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed result: {protocol}')
        summaries[protocol] = report['folds']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=summaries))


if __name__ == '__main__':
    main()
