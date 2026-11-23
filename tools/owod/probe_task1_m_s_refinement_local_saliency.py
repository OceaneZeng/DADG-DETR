"""Check a single saliency-gate change on the locked released training replay."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery_candidates import flip_samples, mine_candidates, saliency_in_boxes, select_distinct_boxes
from tools.owod.probe_task1_discovery_gates import frozen_miner
from tools.owod.probe_task1_m_s_refinement_aose_gates import OUTPUT as ERROR_TRACE
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import OUTPUT as SOURCE, training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SOURCE.with_name('task1-m-s-refinement-local-saliency')
SOURCE_SHA = '297aabdcb8e37daebd7864445b8c14b3fb22efb54d023a6e950ac07768e7d6e8'
TRACE_SHA = 'f57c8d70c8f226ae69130907549fe9e8ca011e4407f3b2e859fe36226ffa60dc'
STRONGEST_POINTS = 5


def prepare():
    origin = verified(SOURCE / 'comparison.json', SOURCE_SHA)
    previous = verified(SOURCE / 'plan.json', origin['plan_sha256'])
    verified(ERROR_TRACE / 'comparison.json', TRACE_SHA)
    expected = {}
    for protocol in previous['jobs']:
        reference = read_json(SOURCE / f'{protocol}.json')
        if reference['plan_sha256'] != origin['plan_sha256'] or reference['groups'] != origin['protocols'][protocol]:
            raise ValueError(f'Changed released baseline: {protocol}')
        expected[protocol] = dict(sample_order_sha256=reference['sample_order_sha256'],
                                  pseudo=origin['protocols'][protocol]['pseudo'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                training_replay_sha256=SOURCE_SHA, aose_attribution_sha256=TRACE_SHA,
                jobs=previous['jobs'], expected=expected, images=previous['images'], seed=previous['seed'],
                batch_size=previous['batch_size'], strongest_points=STRONGEST_POINTS, paper_comparable=False,
                intervention='Use the mean of the strongest 5 of the same 9 points for both saliency predicates. '
                             'Keep original candidate ranking, consistency, known exclusion, geometry, dedup and cap.',
                proxy='Hide the smallest transformed released box per nonempty image from matching and exclusion. '
                      'Use that box only for scoring both candidate sets. Do not use hidden original classes.',
                advance='In both protocols require a net gain of known-dominant selected queries on the intact replay '
                        'and positive net masked-proxy coverage at IoU > 0.5 with the native top_k.')


def strongest(view):
    return saliency_in_boxes(view['attention_feature'], view['pred_boxes'], view['image_sizes'],
                             view['padded_size'], strongest_points=STRONGEST_POINTS)


def replay_selector(outputs, teacher, targets, criterion, known, student_top, teacher_top):
    trace = []
    original = mine_candidates(outputs, targets, criterion.matcher(outputs, targets), known,
                               teacher, criterion.top_k, criterion.saliency_threshold,
                               criterion.consistency_threshold, gate_trace=trace)
    matches = original['matches']
    threshold = criterion.saliency_threshold
    both_views = (student_top >= threshold) & (matches >= 0) & (teacher_top.gather(1, matches.clamp_min(0)) >= threshold)
    quality = original['saliency'] * original['consistency'] * original['semantic_weights']
    if 'geometry_quality' in outputs:
        quality *= outputs['geometry_quality']
    candidate = torch.zeros_like(original['pseudo'])
    quota_matched = torch.zeros_like(candidate)
    for batch, gate in enumerate(trace):
        expanded = gate['eligible'] & gate['consistent'] & both_views[batch]
        if (gate['confident'] & ~expanded).any():
            raise ValueError('Strongest-points saliency unexpectedly discarded an eligible candidate')
        boxes = outputs['pred_boxes'][batch]
        chosen = select_distinct_boxes(boxes, expanded, quality[batch], criterion.top_k)
        candidate[batch, chosen] = True
        original_count = int(original['pseudo'][batch].sum())
        if len(chosen) < original_count:
            quota_matched[batch] = original['pseudo'][batch]
        else:
            quota_matched[batch, chosen[:original_count]] = True
    return original['pseudo'], candidate, quota_matched


def hidden_proxy(targets):
    proxies, released = [], []
    for target in targets:
        boxes = target['boxes']
        keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        if len(boxes):
            index = boxes[:, 2:].prod(-1).argmin().reshape(1)
            keep[index] = False
        proxies.append(boxes[~keep])
        released.append({key: value[keep] for key, value in target.items()})
    return released, proxies


@torch.inference_mode()
def measure(model, criterion, ema, loader):
    exposure = {key: Counter() for key in ('native_cap', 'native_count')}
    proxy_hits = {key: Counter() for key in ('native_cap', 'native_count')}
    images, proxies = 0, 0
    known = model.known_ids
    for step, (images_batch, targets) in enumerate(loader, 1):
        samples = images_batch.to('cuda')
        targets = [{key: target[key].cuda() for key in ('labels', 'boxes')} for target in targets]
        output = model(samples)
        teacher = ema.forward_teacher(flip_samples(samples))
        student_top, teacher_top = strongest(output), strongest(teacher)
        reference, candidate, equal_count = replay_selector(output, teacher, targets, criterion, known,
                                                           student_top, teacher_top)
        max_known = output['pred_logits'][..., known].sigmoid().amax(-1)
        # Match the baseline's subtraction before applying the tie rule.
        conflict = (output['pred_novelty'].sigmoid() - max_known) <= 0
        for mode, selected in (('native_cap', candidate), ('native_count', equal_count)):
            added, removed = selected & ~reference, reference & ~selected
            exposure[mode].update(
                baseline=int(reference.sum()), candidate=int(selected.sum()),
                baseline_conflicts=int((reference & conflict).sum()),
                candidate_conflicts=int((selected & conflict).sum()),
                added=int(added.sum()), removed=int(removed.sum()),
                added_conflicts=int((added & conflict).sum()), removed_conflicts=int((removed & conflict).sum()),
                baseline_zero_images=int((~reference.any(1)).sum()), candidate_zero_images=int((~selected.any(1)).sum()))

        masked, proxy_boxes = hidden_proxy(targets)
        reference, candidate, equal_count = replay_selector(output, teacher, masked, criterion, known,
                                                           student_top, teacher_top)
        for batch, proxy in enumerate(proxy_boxes):
            if not len(proxy):
                continue
            proxies += 1
            hits = box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][batch]), box_cxcywh_to_xyxy(proxy))[0][:, 0] > .5
            baseline_hit = bool((hits & reference[batch]).any())
            for mode, selected in (('native_cap', candidate), ('native_count', equal_count)):
                candidate_hit = bool((hits & selected[batch]).any())
                proxy_hits[mode].update(baseline=baseline_hit, candidate=candidate_hit,
                                        gained=candidate_hit and not baseline_hit,
                                        lost=baseline_hit and not candidate_hit)
        images += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    return dict(images=images, proxies=proxies, exposure=exposure, proxy_hits=proxy_hits)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 matching the source replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed refined endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(path, manifest),
                                            job, plan)
            expected = plan['expected'][protocol]
            if order != expected['sample_order_sha256']:
                raise ValueError(f'Changed training replay image sequence: {protocol}')
            model, criterion, ema = frozen_miner(checkpoint, manifest)
            print(f'Evaluating {protocol} released training saliency', flush=True)
            report = measure(model, criterion, ema, loader)
            exposure, proxy = report['exposure']['native_cap'], report['proxy_hits']['native_cap']
            if (report['images'] != plan['images'] or exposure['baseline'] != expected['pseudo']['queries']
                    or exposure['baseline_conflicts'] != expected['pseudo']['conflicts']):
                raise ValueError(f'Changed native released training pseudo assignments: {protocol}')
            report['passed'] = (exposure['added_conflicts'] > exposure['removed_conflicts'] and proxy['gained'] > proxy['lost'])
            write_json(dest, dict(plan_sha256=digest, sample_order_sha256=order, **report))
            del model, criterion, ema
        results[protocol] = read_json(dest)
        if results[protocol]['plan_sha256'] != digest:
            raise ValueError(f'Changed diagnostic plan: {dest}')
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                               passed=all(row['passed'] for row in results.values())))


if __name__ == '__main__':
    main()
