"""Screen novelty-head ranking on unmatched duplicates in the fresh S Task 1 recipe."""

from models.graph_local.discovery import DUPLICATE_KNOWN_IOU
from tools.owod.probe_s_task1_known_overlap_assignment import OUTPUT as AUDIT_OUTPUT
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_score_order import (
    OUTPUT as MATCHED_OUTPUT, RANK_WEIGHT, evaluate_endpoint, prepare as prepare_midpoint_plan,
)
from tools.owod.run_stronger_parent_study import run_job


OUTPUT = MATCHED_OUTPUT.with_name('task1-s-duplicate-novelty-rank')
AUDIT_SHA = 'a338ad9836f93ea69b6e1ec61a82150319961a232b8eb399221720075b73aea8'
RANK_FLAG = '--discovery-duplicate-rank-coef'


def prepare():
    plan, manifest = prepare_midpoint_plan(OUTPUT, RANK_FLAG)
    audit = verified(AUDIT_OUTPUT / 'comparison.json', AUDIT_SHA)
    audit_plan = verified(AUDIT_OUTPUT / 'plan.json', audit['plan_sha256'])
    if (audit['status'] != 'complete' or audit_plan['overlap'] != DUPLICATE_KNOWN_IOU
            or audit_plan['arms']['midpoint']['comparison_sha256'] != plan['midpoint_sha256']
            or any(audit_plan[key] != plan[key] for key in ('holdout_sha256', 'manifest_sha256', 'batch_size', 'limit'))
            or plan['limit'] != 50):
        raise ValueError('Expected the fixed assignment audit and original Top-50 endpoint')
    for arm in audit['arms'].values():
        counts = arm['counts']
        if 2 * counts['emitted_duplicate_over_known'] <= counts['emitted_unknown_over_known']:
            raise ValueError('The duplicate majority prerequisite did not pass')
    from main import get_args_parser
    command = plan['command']
    args = get_args_parser().parse_args(command[command.index('--coco_path'):])
    if args.discovery_warmup != 1 or args.discovery_pseudo_epoch_offset != 0:
        raise ValueError('Expected one warmup epoch before duplicate novelty ranking')
    plan.update(duplicate_rank_coef=RANK_WEIGHT, duplicate_rank_min_iou=DUPLICATE_KNOWN_IOU,
                duplicate_rank_start_epoch=args.discovery_warmup, duplicate_rank_gradient='novelty_head_only',
                duplicate_rank_normalization='global_eligible_query_count', assignment_audit_sha256=AUDIT_SHA)
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    print('Fitting S Task 1 with head-only novelty ranking on unmatched released-box duplicates', flush=True)
    checkpoint, saved = run_job(plan['command'], plan['screen_sha256'])
    if saved.discovery_duplicate_rank_coef != RANK_WEIGHT or saved.backbone_weights_sha256 != plan['backbone_sha256']:
        raise ValueError('Unexpected duplicate-novelty-rank endpoint')
    evaluate_endpoint(OUTPUT, checkpoint, manifest, plan, digest)


if __name__ == '__main__':
    main()
