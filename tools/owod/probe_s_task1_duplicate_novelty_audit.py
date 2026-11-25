"""Check whether the frozen novelty-head candidate suppressed its targeted duplicates."""

import subprocess
from pathlib import Path

import torch

from tools.owod.probe_s_task1_known_overlap_assignment import OUTPUT as ASSIGNMENT_OUTPUT, evaluate
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_duplicate_novelty_rank import AUDIT_SHA, OUTPUT as CANDIDATE_OUTPUT
from tools.owod.run_s_task1_known_score_order import PROTOCOL
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = CANDIDATE_OUTPUT.with_name('task1-s-duplicate-novelty-audit')
CANDIDATE_SHA = '0c08f35f83efb71a11477861d5efa651ea03559216ca79cdb0663a48d1c1aec9'
RELEASED_SHA = '0376e8b1fd450855b906b9f819b2036478fd9d7aa28a175bbc830d52013539d8'


def prepare():
    candidate = verified(CANDIDATE_OUTPUT / 'comparison.json', CANDIDATE_SHA)
    training = verified(CANDIDATE_OUTPUT / 'plan.json', candidate['plan_sha256'])
    released = verified(CANDIDATE_OUTPUT / 'released.json', RELEASED_SHA)
    audit = verified(ASSIGNMENT_OUTPUT / 'comparison.json', AUDIT_SHA)
    source = verified(ASSIGNMENT_OUTPUT / 'plan.json', audit['plan_sha256'])
    checkpoint = CANDIDATE_OUTPUT / PROTOCOL / 'increment/checkpoint.pth'
    shared = {key: source[key] for key in ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit')}
    if (candidate['status'] != 'complete' or candidate['passed'] or audit['status'] != 'complete'
            or training['assignment_audit_sha256'] != AUDIT_SHA
            or any(training[key] != value for key, value in shared.items())
            or source['overlap'] != training['duplicate_rank_min_iou']
            or released['plan_sha256'] != candidate['plan_sha256']
            or released['checkpoint_sha256'] != candidate['checkpoint_sha256']
            or file_sha256(checkpoint) != candidate['checkpoint_sha256']):
        raise ValueError('Expected the pinned candidate and half-gradient assignment audit')
    plan = dict(**shared, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, overlap=source['overlap'], assignment_audit_sha256=AUDIT_SHA,
                control=source['arms']['midpoint'], candidate_comparison_sha256=CANDIDATE_SHA,
                candidate=dict(checkpoint=str(checkpoint), sha256=candidate['checkpoint_sha256'],
                               expected_ap=candidate['known_ap50'],
                               expected_unknown=released['report']['emitted']['composition']['unknown']),
                decision='Both the violation fraction among eligible duplicates and their emitted unknown count must fall.')
    pin(OUTPUT / 'plan.json', plan)
    return plan, audit['arms']['midpoint']


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen assignment replay')
    plan, control = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    path = Path(plan['holdout'])
    if file_sha256(path) != plan['holdout_sha256']:
        raise ValueError('Changed released-only evaluation inputs')
    print('Evaluating S Task 1 duplicate novelty candidate by released assignment', flush=True)
    candidate = evaluate(plan['candidate'], manifest, path, plan)
    if candidate['counts']['emitted_unknown'] != plan['candidate']['expected_unknown']:
        raise ValueError('Native unknown count differs from the archived evaluation')
    rates = {}
    for name, row in (('control', control), ('candidate', candidate)):
        counts = row['counts']
        if counts['eligible_duplicates'] <= 0:
            raise ValueError('Expected eligible unmatched duplicate queries')
        rates[name] = counts['duplicate_violations'] / counts['eligible_duplicates']
    candidate_counts, control_counts = candidate['counts'], control['counts']
    gates = {
        'violation_rate': rates['candidate'] < rates['control'],
        'emitted_duplicates': candidate_counts['emitted_duplicate_over_known'] < control_counts['emitted_duplicate_over_known'],
    }
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, control=control,
                                               candidate=candidate, violation_rates=rates, gates=gates,
                                               supported=all(gates.values())))


if __name__ == '__main__':
    main()
