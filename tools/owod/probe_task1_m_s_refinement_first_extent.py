"""Measure first-layer sampling from native selected box extents, with frozen weights."""

import argparse
from collections import Counter
from contextlib import ExitStack
import subprocess

import torch

from tools.owod.probe_task1_m_s_later_sampling import sampler_hook
from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure
from tools.owod.probe_task1_m_s_refinement_dense_quality_veto import OUTPUT as VETO_OUTPUT
from tools.owod.probe_task1_m_s_refinement_trajectory import hits, run_released_probe
from tools.owod.probe_task1_paired_discovery import verified
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = VETO_OUTPUT.with_name('task1-m-s-refinement-first-sampling-extent')
REGRESSION_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-first-regression-trace')
VETO_SHA = '347156e7f779408f60fefd0f40217240377ed0acea52e45d9b1f704c6139539e'
EXTENT_SHA = 'de74c42587e16ca7c82038fa42ce28eccfc405ff7d2d2a009254bb1bca958ca7'


def resample(model, samples, native):
    decoder = model.detr.transformer.decoder
    selected = native['geometry']
    slots = torch.where(selected['active'])
    if not len(slots[0]):
        return native
    boxes = selected['boxes'][slots[0], selected['chosen'][slots]]
    ratios = {}

    def capture(_decoder, inputs):
        ratios['value'] = inputs[5]

    with ExitStack() as stack:
        stack.enter_context(decoder.register_forward_pre_hook(capture))
        stack.enter_context(decoder.layers[0].register_forward_pre_hook(
            sampler_hook(stack, slots, boxes, ratios, use_extent=True, keep_center=True)))
        changed = model(samples)
    for key in ('active', 'chosen'):
        if not torch.equal(selected[key], changed['geometry'][key]):
            raise ValueError(f'Sampling replay changed native encoder selection: {key}')
    return changed


def observe(native, revised, targets, plan, counts):
    geometry = native['geometry']
    for batch, target in enumerate(targets):
        small = target['source_box_area'].cuda() < plan['small_area']
        truth = box_cxcywh_to_xyxy(target['boxes'].cuda()[small])
        slots = torch.where(geometry['active'][batch])[0]
        cells = geometry['chosen'][batch, slots]
        seeds = hits(geometry['boxes'][batch, cells], truth)
        seed_covered = seeds.any(dim=0)
        counts['selected_small'] += int(seed_covered.sum())
        for arm, outputs in (('native', native), ('extent', revised)):
            first = hits(outputs['aux_outputs'][0]['pred_boxes'][batch], truth)
            final = hits(outputs['pred_boxes'][batch], truth)
            counts.update({
                f'{arm}_seed_first': int((first[slots] & seeds).any(dim=0).sum()),
                f'{arm}_seed_final': int((final[slots] & seeds).any(dim=0).sum()),
                f'{arm}_any_first': int(first.any(dim=0).sum()),
            })


@torch.inference_mode()
def trace_regression(model, loader, plan):
    counts = Counter()
    coverage = Counter()
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        geometry = outputs['geometry']
        for batch, target in enumerate(targets):
            small = target['source_box_area'].cuda() < plan['small_area']
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda()[small])
            slots = torch.where(geometry['active'][batch])[0]
            cells = geometry['chosen'][batch, slots]
            seed = geometry['boxes'][batch, cells]
            first = outputs['aux_outputs'][0]['pred_boxes'][batch, slots]
            final = hits(outputs['pred_boxes'][batch], truth)
            seed_hit = hits(seed, truth)
            first_hit = hits(first, truth)
            center = torch.cat((seed[:, :2], first[:, 2:]), dim=1)
            extent = torch.cat((first[:, :2], seed[:, 2:]), dim=1)
            center_ok, extent_ok = ((hits(boxes, truth) & seed_hit).any(dim=0) for boxes in (center, extent))
            selected = seed_hit.any(dim=0)
            early = (seed_hit & first_hit).any(dim=0)
            late = (seed_hit & final[slots]).any(dim=0)
            final_any = final.any(dim=0)
            lost = selected & ~early
            lost_to_end = lost & ~final_any
            counts.update(selected=int(selected.sum()), first_same_slot=int(early.sum()),
                          final_same_slot=int(late.sum()), selected_final_missed=int((selected & ~final_any).sum()),
                          first_lost=int(lost.sum()), first_lost_recovered_same_slot=int((lost & late).sum()),
                          first_lost_final_missed=int(lost_to_end.sum()),
                          seed_center_only=int((lost & center_ok & ~extent_ok).sum()),
                          seed_extent_only=int((lost & extent_ok & ~center_ok).sum()),
                          either_component=int((lost & center_ok & extent_ok).sum()),
                          both_needed=int((lost & ~(center_ok | extent_ok)).sum()),
                          final_miss_center=int((lost_to_end & center_ok).sum()),
                          final_miss_extent=int((lost_to_end & extent_ok).sum()))
            coverage.update(gt=len(truth), final=int(final_any.sum()))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released first-layer regression replay')
    return dict(groups={'regression': counts}, coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression-trace', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for released first-layer sampling')
    veto = verified(VETO_OUTPUT / 'comparison.json', VETO_SHA)
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(PROTECTED_FIT_OUTPUT / 'plan.json', quarter['plan_sha256'])
    if veto['status'] != 'complete' or veto['passed'] or not quarter['passed']:
        raise ValueError('Expected completed post-decoder veto and released reference')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), veto_sha256=VETO_SHA,
                factor='Retain native selected cells, query embeddings and 2D reference centers. At decoder layer 1 '
                       'only, use the selected encoder box extent to scale cross-attention offsets on injected slots. '
                       'Keep the 2D box regression reference and all other layer rules.',
                advance='Compare same-query seed survival at first and final layers, net small dense coverage and '
                        'actual Top-50 correct-class matching before considering training.')

    if args.regression_trace:
        sampling = verified(OUTPUT / 'comparison.json', EXTENT_SHA)
        sampling_plan = verified(OUTPUT / 'plan.json', sampling['plan_sha256'])
        keys = ('jobs', 'images', 'batch_size', 'seed', 'small_area')
        if (sampling['status'] != 'complete' or any(row['groups']['passed'] for row in sampling['protocols'].values())
                or any(plan[key] != sampling_plan[key] for key in keys)):
            raise ValueError('Expected both failed sampling arms on the same released replay')
        plan.update(sampling_sha256=EXTENT_SHA,
                    factor='For natively accurate selected seeds lost in decoder layer 1, restore the seed center, '
                           'extent, or both in the corresponding query output for released geometry attribution.',
                    advance='Count first-layer recoveries and their overlap with lasting misses. Existing seed '
                            'coverage and slow recovery are distinct from improvement of final output.')

        def verify_trace(model, loader, known, plan):
            protocol = next(name for name, job in plan['jobs'].items() if job['classes'] == known)
            report = trace_regression(model, loader, plan)
            row = report['groups']['regression']
            native = sampling['protocols'][protocol]['groups']['sampling']
            if (row['selected'], row['first_same_slot'], row['final_same_slot']) != (
                    native['selected_small'], native['native_seed_first'], native['native_seed_final']):
                raise ValueError(f'Changed native selected seed hits: {protocol}')
            return report

        run_released_probe(REGRESSION_OUTPUT, plan, quarter, verify_trace)
        return

    def replay(model, loader, known, plan):
        counts = Counter()

        def prediction(samples, native, targets):
            revised = resample(model, samples, native)
            observe(native, revised, targets, plan, counts)
            return revised

        report = measure(model, known, loader, plan, replay=prediction)
        protocol = next(name for name, job in plan['jobs'].items() if job['classes'] == known)
        groups = report['groups']
        reference = quarter['protocols'][protocol]['groups']
        if any(row['native_known_tp'] != reference[name]['native_known_tp'] for name, row in groups.items()):
            raise ValueError(f'Changed native class matches: {protocol}')
        small = groups['small']
        sufficient = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                      and all(row['revised_known_tp'] >= row['native_known_tp'] for row in groups.values()))
        groups.update(sampling=counts, changes=report['changes'], passed=sufficient)
        return dict(groups=groups, coverage={key: small[key] for key in ('gt', 'final')})

    run_released_probe(OUTPUT, plan, quarter, replay)


if __name__ == '__main__':
    main()
