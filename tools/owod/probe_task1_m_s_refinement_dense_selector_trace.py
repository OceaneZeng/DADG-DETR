"""Attribute gains and losses of the frozen released local full-box selector."""

import argparse
from collections import Counter
import subprocess

import torch

from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_dense_selector import (
    INTERACTION_OUTPUT, OUTPUT as SELECTOR_OUTPUT, SELECTOR_SHA, TRACE_OUTPUT, selector_transform,
)
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_protected_residual import protect_known
from tools.owod.probe_task1_m_s_refinement_trajectory import hits, run_released_probe
from tools.owod.probe_task1_paired_discovery import verified
from util.box_ops import box_cxcywh_to_xyxy


INTERACTION_SHA = '5fe885d49b4dcde4ddb38aa2ebae9e8bc6ed1ca894d17b79afcda06b2e087d14'
INTERACTION_TRACE_OUTPUT = INTERACTION_OUTPUT.with_name('task1-m-s-refinement-dense-local-interaction-trace')


def trace_image(outputs, revised, target, batch, trace, small_area, stats):
    final = outputs['pred_boxes'][batch]
    truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
    small = target['source_box_area'].cuda() < small_area
    native, after = hits(final, truth), hits(revised[batch], truth)
    covered, rescored = native.any(dim=0), after.any(dim=0)
    misses = small & ~covered
    gained, lost = rescored & ~covered, covered & ~rescored
    stats['geometry'].update(gt=int(small.sum()), final=int((small & covered).sum()),
                             revised=int((small & rescored).sum()), gained=int((small & gained).sum()),
                             lost=int((small & lost).sum()), all_gained=int(gained.sum()), all_lost=int(lost.sum()))
    changed = (final != revised[batch]).any(dim=1)
    stats['changes'].update(images=int(changed.any()), queries=int(changed.sum()))
    if trace is None:
        stats['geometry'].update(veto_revised=int((small & rescored).sum()), veto_gained=0, veto_lost=0)
        return
    queries, cells, scores, winner = trace
    boxes = outputs['geometry']['boxes'][batch, cells]
    candidate_hits = hits(boxes.flatten(0, 1), truth).reshape(*cells.shape, len(truth))
    rows = torch.arange(len(queries), device=queries.device)
    best = candidate_hits[rows, winner]
    safe = ~native[queries].any(dim=1)
    available = candidate_hits[:, :, misses].any(dim=(1, 2)) & safe
    best_positive = best[:, misses].any(dim=1) & safe
    permit = scores > 0
    actual = changed[queries]
    breaks_hit = (native[queries] & ~best).any(dim=1)
    counts = stats['queries']
    counts.update(eligible=len(queries), available=int(available.sum()),
                  rank_error=int((available & ~best_positive).sum()),
                  keep_threshold=int((best_positive & ~permit).sum()),
                  before_rollback=int((best_positive & permit).sum()),
                  after_rollback=int((best_positive & actual).sum()),
                  selected=int(permit.sum()), changed=int(actual.sum()),
                  changed_without_positive=int((actual & ~best_positive).sum()),
                  changed_with_native_hit=int((actual & ~safe).sum()),
                  changed_breaking_native_hit=int((actual & breaks_hit).sum()))
    kept = after.clone()
    damage = actual & breaks_hit
    kept[queries[damage]] = native[queries[damage]]
    veto_covered = kept.any(dim=0)
    stats['geometry'].update(veto_revised=int((small & veto_covered).sum()),
                             veto_gained=int((small & veto_covered & ~covered).sum()),
                             veto_lost=int((small & covered & ~veto_covered).sum()))


@torch.inference_mode()
def measure(model, loader, known, plan, params):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    stats = {name: Counter() for name in ('geometry', 'queries', 'changes')}
    protection = Counter(restored_known_queries=0, reverted_images=0)
    observed = 0
    trace = {}

    def observe(batch, queries, cells, scores, winner):
        trace[batch] = queries, cells, scores, winner

    transform = protect_known(selector_transform(params, post, known, torch.device('cuda'), observe=observe),
                              post, known, protection)
    for step, (samples, targets) in enumerate(loader, 1):
        trace.clear()
        outputs = model(samples.to('cuda'))
        revised = transform(outputs)
        for batch, target in enumerate(targets):
            trace_image(outputs, revised, target, batch, trace.get(batch), plan['small_area'], stats)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete local selector attribution replay')
    stats['protection'] = protection
    return dict(groups=stats, coverage={key: stats['geometry'][key] for key in ('gt', 'final')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen local selector attribution')
    source, digest, output = SELECTOR_OUTPUT, SELECTOR_SHA, TRACE_OUTPUT
    if args.interaction:
        source, digest, output = INTERACTION_OUTPUT, INTERACTION_SHA, INTERACTION_TRACE_OUTPUT
    selector = verified(source / 'comparison.json', digest)
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(source / 'plan.json', selector['plan_sha256'])
    if selector['status'] != 'complete' or selector['passed'] or not quarter['passed']:
        raise ValueError('Expected the stopped local selector and its quarter-cap comparison')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_selector_sha256=plan.get('selector_sha256'), selector_sha256=digest,
                factor='Attribute viable same-query box corrections to candidate rank, the original keep threshold, '
                       'and native known output protection. Count changes that break existing query overlap.',
                advance='Estimate the geometric effect if an oracle vetoed only damaging moves. Use the fixed replay '
                        'for diagnosis; any new rule is selected using the released fitting sources.')

    def verify_measure(model, loader, known, plan):
        protocol = next(name for name, job in plan['jobs'].items() if job['classes'] == known)
        reference = selector['protocols'][protocol]
        result = measure(model, loader, known, plan, reference['selector'])
        observed = result['groups']
        if observed['changes'] != reference['changes'] or observed['protection'] != reference['protection']:
            raise ValueError(f'Changed frozen selector decisions: {protocol}')
        if any(observed['geometry'][key] != reference['groups']['small'][key]
               for key in ('gt', 'final', 'revised', 'gained', 'lost')):
            raise ValueError(f'Changed frozen released small geometry: {protocol}')
        return result

    run_released_probe(output, plan, quarter, verify_measure)


if __name__ == '__main__':
    main()
