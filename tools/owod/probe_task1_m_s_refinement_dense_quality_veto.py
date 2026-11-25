"""Calibrate a frozen localization-quality veto outside the Task 1 fit and replay cohorts."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_residual import PROTECTED_FIT_OUTPUT, QUARTER_FIT_SHA
from tools.owod.probe_task1_m_s_refinement_center_selector import source_ids
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure
from tools.owod.probe_task1_m_s_refinement_dense_selector import OUTPUT as SELECTOR_OUTPUT, SELECTOR_SHA, selector_transform
from tools.owod.probe_task1_m_s_refinement_dense_selector_trace import INTERACTION_TRACE_OUTPUT
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_protected_residual import protect_known
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_m_s_refinement_trajectory import hits
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SELECTOR_OUTPUT.with_name('task1-m-s-refinement-dense-quality-veto')
INTERACTION_TRACE_SHA = '31653ab29c5b47beb4dff000bfca050053ddd7c76cf82ccda4923f29acc8a05d'
CALIBRATION_IMAGES = 1536


def quality_delta(outputs, revised, score):
    return score(outputs['geometry'], revised).sigmoid() - outputs['geometry_quality']


@torch.inference_mode()
def calibrate(model, loader, transform, score, plan):
    beneficial, harmful = [], []
    observed = changed_total = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        revised = transform(outputs)
        changed = (final != revised).any(dim=2)
        delta = quality_delta(outputs, revised, score)
        for batch, target in enumerate(targets):
            moves = changed[batch]
            if not moves.any():
                continue
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            native = hits(final[batch], truth)
            adjusted = hits(revised[batch, moves], truth)
            misses = (target['source_box_area'].cuda() < plan['small_area']) & ~native.any(dim=0)
            damage = (native[moves] & ~adjusted).any(dim=1)
            useful = adjusted[:, misses].any(dim=1) & ~damage
            values = delta[batch, moves]
            harmful.append(values[damage].cpu())
            beneficial.append(values[useful].cpu())
            changed_total += len(values)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Calibrating veto {step}/{len(loader)} batches', flush=True)
    if observed != plan['calibration_images'] or not beneficial:
        raise ValueError('Incomplete released quality-veto calibration')
    beneficial, harmful = torch.cat(beneficial), torch.cat(harmful)
    counts = dict(moved=changed_total, useful=len(beneficial), harmful=len(harmful))
    if not len(beneficial) or not len(harmful):
        return dict(counts, threshold=None, advances=False)
    if not (beneficial.isfinite().all() and harmful.isfinite().all()):
        raise ValueError('Nonfinite quality-delta observations')
    # The lowest cutoff that vetoes two thirds of damaging changes keeps the most useful changes.
    threshold = float(harmful.kthvalue((2 * len(harmful) + 2) // 3).values)
    retained = int((beneficial > threshold).sum())
    return dict(counts, threshold=threshold, useful_retained=retained, harmful_vetoed=int((harmful <= threshold).sum()),
                useful_positive_delta=int((beneficial > 0).sum()), harmful_positive_delta=int((harmful > 0).sum()),
                advances=2 * retained >= len(beneficial))


def veto_transform(transform, score, cutoff):
    def apply(outputs):
        revised = transform(outputs)
        delta = quality_delta(outputs, revised, score)
        return torch.where((delta > cutoff)[..., None], revised, outputs['pred_boxes'])
    return apply


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the released quality veto')
    source = verified(SELECTOR_OUTPUT / 'comparison.json', SELECTOR_SHA)
    trace = verified(INTERACTION_TRACE_OUTPUT / 'comparison.json', INTERACTION_TRACE_SHA)
    quarter = verified(PROTECTED_FIT_OUTPUT / 'comparison.json', QUARTER_FIT_SHA)
    plan = verified(SELECTOR_OUTPUT / 'plan.json', source['plan_sha256'])
    if source['status'] != 'complete' or source['passed'] or trace['status'] != 'complete' or not quarter['passed']:
        raise ValueError('Expected completed released local-selection diagnostics')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                selector_sha256=SELECTOR_SHA, interaction_trace_sha256=INTERACTION_TRACE_SHA,
                calibration_images=CALIBRATION_IMAGES, calibration_offset=plan['fit_images'] + plan['images'],
                factor='Keep the original local chooser. Rescore both box geometries using the same frozen refined '
                       'quality head, and allow a proposed replacement only above a separately calibrated quality gain.',
                advance='A calibration cutoff vetoes at least two thirds of damaging moves and retains at least '
                        'half the safe small-target gains. Replay qualifying protocols with the locked released gates.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            inputs = (checkpoint, Path(job['training']), resolve_image_root(path, manifest), job)
            replay, order = training_loader(*inputs, plan, source_box_area=True)
            fit, fit_order = training_loader(*inputs, dict(plan, images=plan['fit_images']), source_box_area=True,
                                            offset=plan['images'])
            loader, calibration_order = training_loader(*inputs, dict(plan, images=CALIBRATION_IMAGES),
                                                         source_box_area=True, offset=plan['calibration_offset'])
            calibration_ids = source_ids(loader)
            held_out = set(source_ids(fit)) | set(source_ids(replay))
            if (order != job['expected_order'] or fit_order != plan['fit_order_sha256'][protocol]
                    or len(set(calibration_ids)) != CALIBRATION_IMAGES or set(calibration_ids) & held_out):
                raise ValueError(f'Changed released calibration partition: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
            transform = selector_transform(source['protocols'][protocol]['selector'], post, known, device)
            score = model.detr.transformer.instance_proposals.refined_quality
            print(f'Calibrating {protocol} frozen local-box veto', flush=True)
            result = calibrate(model, loader, transform, score, plan)
            report = {}
            passed = False
            if result['advances']:
                print(f'Evaluating {protocol} frozen local-box veto', flush=True)
                counts = Counter(restored_known_queries=0, reverted_images=0)
                guarded = protect_known(veto_transform(transform, score, result['threshold']), post, known, counts)
                report = measure(model, known, replay, plan, box_transform=guarded)
                report['protection'] = counts
                reference = quarter['protocols'][protocol]['groups']
                for name, group in report['groups'].items():
                    if any(group[key] != reference[name][key] for key in ('gt', 'final', 'native_known_tp')):
                        raise ValueError(f'Changed released baseline: {protocol}/{name}')
                small = report['groups']['small']
                passed = (small['revised'] - small['final'] >= int(protocol == 's-owodb')
                          and all(row['revised_known_tp'] >= row['native_known_tp'] for row in report['groups'].values()))
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, calibration_order_sha256=calibration_order,
                                  calibration=result, passed=passed, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed released veto plan: {dest}')
        protocols[protocol] = {key: value for key, value in report.items() if key != 'plan_sha256'}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
