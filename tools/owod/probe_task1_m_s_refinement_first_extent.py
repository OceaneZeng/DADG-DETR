"""Measure first-layer sampling from native selected box extents, with frozen weights."""

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
VETO_SHA = '347156e7f779408f60fefd0f40217240377ed0acea52e45d9b1f704c6139539e'


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


def main():
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
