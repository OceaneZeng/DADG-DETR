"""Pair small released targets by learned and proposal query coverage on frozen S endpoints."""

import subprocess
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_duplicate_novelty_audit import OUTPUT as AUDIT, RELEASED_SHA
from tools.owod.probe_s_task1_half_rank_quota import RETRY
from tools.owod.probe_s_task1_rank_independent import PROTOCOL, evaluate
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_duplicate_novelty_rank import OUTPUT as CANDIDATE
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = AUDIT.with_name('task1-s-duplicate-novelty-geometry')
AUDIT_SHA = '3f19f667fa3d714bea9757ae8b2659a5128a358dfb9190da874b2fa6b0f4e4fc'
HALF_RELEASED_SHA = '12ebbd778f68d63b28fcdaa7c79b3356658596d0cbad0e5be377194550e2a2f1'
SIZE_BANDS = ('under_16_sq', '16_to_32_sq')


def prepare():
    audit = verified(AUDIT / 'comparison.json', AUDIT_SHA)
    source = verified(AUDIT / 'plan.json', audit['plan_sha256'])
    if audit['status'] != 'complete' or not audit['supported']:
        raise ValueError('Expected the completed frozen assignment audit')
    arms = {}
    for name, path, digest in (('control', RETRY / 'independent.json', HALF_RELEASED_SHA),
                               ('candidate', CANDIDATE / 'released.json', RELEASED_SHA)):
        endpoint = verified(path, digest)
        arm = source[name]
        report = endpoint['report']
        if (endpoint['checkpoint_sha256'] != arm['sha256']
                or file_sha256(Path(arm['checkpoint'])) != arm['sha256']
                or report['emitted']['metrics']['known_ap50'] != audit[name]['known_ap50']):
            raise ValueError(f'Changed frozen endpoint: {name}')
        arms[name] = dict(**arm, reference_sha256=digest,
                          expected_small=report['released_small_learned_query_coverage'],
                          expected_raw=report['raw_current_box_recall_iou50'])
    if any(arms[name]['expected_small'] != {'covered': covered, 'released': 2922}
           for name, covered in (('control', 939), ('candidate', 793))):
        raise ValueError('Expected the original small-target denominators and learned-query hits')
    shared = {key: source[key] for key in ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit')}
    plan = dict(**shared, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                assignment_sha256=AUDIT_SHA, arms=arms, paper_comparable=False,
                decision='Use the majority of lost learned hits: entirely uncovered, proposal-covered, or tie.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


def summarize(control, candidate, ids):
    old_learned = {i for i in ids if control[i]['learned']}
    new_learned = {i for i in ids if candidate[i]['learned']}
    old_proposal = {i for i in ids if control[i]['proposal']}
    new_proposal = {i for i in ids if candidate[i]['proposal']}
    lost = old_learned - new_learned
    gained = new_learned - old_learned
    return dict(gt=len(ids), control_learned=len(old_learned), candidate_learned=len(new_learned),
                control_proposal=len(old_proposal), candidate_proposal=len(new_proposal),
                control_any=len(old_learned | old_proposal), candidate_any=len(new_learned | new_proposal),
                retained_learned=len(old_learned & new_learned), lost_learned=len(lost), gained_learned=len(gained),
                lost_retained_by_candidate_proposal=len(lost & new_proposal),
                lost_uncovered_by_candidate=len(lost - new_proposal))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen paired replay')
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    annotations = Path(plan['holdout'])
    if file_sha256(annotations) != plan['holdout_sha256']:
        raise ValueError('Changed released-only evaluation inputs')
    traces = {}
    for name, arm in plan['arms'].items():
        dest = OUTPUT / f'{name}.json'
        binding = dict(plan_sha256=digest, checkpoint_sha256=arm['sha256'])
        if not dest.is_file():
            print(f'Evaluating S Task 1 {name} query origins on released small targets', flush=True)
            model, _ = load_detector(None, Path(arm['checkpoint']), torch.device('cuda'))
            report = evaluate(model, manifest, annotations, plan, trace_query_origin=True)
            del model
            if (report['emitted']['metrics']['known_ap50'] != arm['expected_ap']
                    or report['released_small_learned_query_coverage'] != arm['expected_small']
                    or report['raw_current_box_recall_iou50'] != arm['expected_raw']):
                raise ValueError(f'Frozen inference did not reproduce the released holdout: {name}')
            write_json(dest, dict(**binding, targets=report['released_small_query_origin']))
        record = read_json(dest)
        if any(record[key] != value for key, value in binding.items()):
            raise ValueError(f'Changed result binding: {name}')
        traces[name] = record['targets']
    control, candidate = traces['control'], traces['candidate']
    if control.keys() != candidate.keys() or any(
            (old['image_id'], old['size']) != (candidate[i]['image_id'], candidate[i]['size'])
            for i, old in control.items()):
        raise ValueError('Expected the same released annotations on both endpoints')
    groups = {'all': set(control),
              **{size: {i for i, row in control.items() if row['size'] == size} for size in SIZE_BANDS}}
    counts = {name: summarize(control, candidate, ids) for name, ids in groups.items()}
    total = counts['all']
    if (total['gt'] != 2922 or total['control_learned'] != 939 or total['candidate_learned'] != 793
            or any(total[key] != sum(counts[band][key] for band in SIZE_BANDS) for key in total)):
        raise ValueError('Paired query origin counts disagree with the published coverage')
    comparison = total['lost_uncovered_by_candidate'] - total['lost_retained_by_candidate_proposal']
    decision = 'all_query_geometry_loss' if comparison > 0 else 'proposal_retains_majority' if comparison < 0 else 'tie'
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
                                               groups=counts, decision=decision))


if __name__ == '__main__':
    main()
