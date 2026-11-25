"""Trace usable earlier box geometry on the fixed released Task 1 replay."""

from collections import Counter
from pathlib import Path
import subprocess

from scipy.optimize import linear_sum_assignment
import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_range import FIT_PAIRING_OUTPUT
from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = FIT_PAIRING_OUTPUT.with_name('task1-m-s-refinement-released-trajectory')
PAIRING_SHA = '772696e1673d5a7f55bf581ac2c968b8c7d5038f3f80564c84679ba14c55d450'
STAGES = ('selected_seed', 'decoder_1', 'decoder_2', 'decoder_3', 'decoder_4', 'decoder_5')
GROUPS = ('under_16_sq', '16_to_32_sq', 'all_small')


def hits(boxes, truth):
    return box_iou(box_cxcywh_to_xyxy(boxes), truth)[0] > .5


def matched_targets(edges):
    values = edges.cpu().numpy()
    queries, targets = linear_sum_assignment(values, maximize=True)
    return int(values[queries, targets].sum())


def trace_image(final, layers, geometry, batch, target, scored, columns, small_area, groups, coverage):
    truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
    source_area = target['source_box_area'].cuda()
    native = hits(final, truth)
    covered = native.any(dim=0)
    small = source_area < small_area
    coverage.update(gt=int(small.sum()), final=int((small & covered).sum()))
    misses = small & ~covered
    if not misses.any():
        return

    slots = torch.where(geometry['active'][batch])[0]
    cells = geometry['chosen'][batch, slots]
    seed = torch.zeros_like(native)
    seed[slots] = hits(geometry['boxes'][batch, cells], truth)
    earlier = torch.stack([seed, *(hits(layer, truth) for layer in layers)])
    safe = ~(native[None] & ~earlier).any(dim=-1)
    indices = scored['top_indices'] // columns
    emitted_unknown = scored['labels'] == 91
    queries = torch.arange(len(final), device=final.device)
    movable = ~torch.isin(queries, indices[~emitted_unknown])
    unknown = torch.isin(queries, indices[emitted_unknown])
    injected = torch.isin(queries, slots)

    for name, mask in zip(GROUPS, (source_area < 16**2, source_area >= 16**2, small), strict=True):
        wanted = misses & mask
        alternatives = earlier[:, :, wanted]
        safe_alternatives = alternatives & (safe & movable[None])[:, :, None]
        stage_counts = groups[name]
        stage_counts['misses'] += int(wanted.sum())
        for pos, stage in enumerate(STAGES):
            stage_counts[f'{stage}_targets'] += int(alternatives[pos].any(dim=0).sum())
            stage_counts[f'{stage}_safe_targets'] += int(safe_alternatives[pos].any(dim=0).sum())
        safe_any = safe_alternatives.any(dim=0)
        stage_counts['any_box_targets'] += int(alternatives.any(dim=(0, 1)).sum())
        stage_counts['safe_targets'] += int(safe_any.any(dim=0).sum())
        stage_counts['safe_injected_targets'] += int(safe_any[injected].any(dim=0).sum())
        stage_counts['safe_learned_targets'] += int(safe_any[~injected].any(dim=0).sum())
        stage_counts['safe_matched'] += matched_targets(safe_any)
        stage_counts['safe_emitted_unknown_matched'] += matched_targets(safe_any[unknown])


@torch.inference_mode()
def measure(model, loader, known, plan):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    groups = {name: Counter() for name in GROUPS}
    coverage = Counter()
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        layers = [layer['pred_boxes'] for layer in outputs['aux_outputs']]
        if len(layers) != 5:
            raise ValueError('Expected all five earlier boxes from the locked six-layer decoder')
        scored = post(outputs, final.new_ones((len(targets), 2)), include_top_indices=True)
        for batch, (target, row) in enumerate(zip(targets, scored, strict=True)):
            trace_image(final[batch], [layer[batch] for layer in layers], outputs['geometry'], batch, target, row,
                        len(known) + 1, plan['small_area'], groups, coverage)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released trajectory replay')
    return dict(groups=groups, coverage=coverage)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released trajectory replay')
    pairing = verified(FIT_PAIRING_OUTPUT / 'comparison.json', PAIRING_SHA)
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(PROTECTED_FIT_OUTPUT / 'plan.json', quarter['plan_sha256'])
    if pairing['status'] != 'complete' or quarter['status'] != 'complete' or not quarter['passed']:
        raise ValueError('Expected completed fit pairing and passed quarter-cap preflight')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                fit_pairing_sha256=PAIRING_SHA, quarter_fit_sha256=QUARTER_FIT_SHA,
                factor='Count native selected proposals and earlier decoder boxes that recover a released small miss '
                        'on the same query while preserving its existing hits and native known output boxes.',
                advance='Check one-to-one geometric headroom before fitting any box choice. Candidate edges use released '
                        'labels. Native emitted unknown counts are geometry bounds before any rescoring or suppression.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    device = torch.device('cuda')
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            print(f'Tracing {protocol} released decoder geometry', flush=True)
            report = measure(model, loader, job['classes'], plan)
            reference = quarter['protocols'][protocol]['groups']['small']
            if any(report['coverage'][key] != reference[key] for key in ('gt', 'final')):
                raise ValueError(f'Changed native small box geometry: {protocol}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed released trajectory plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('groups', 'coverage')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
