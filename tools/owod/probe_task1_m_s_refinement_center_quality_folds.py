"""Evaluate the frozen center selector on the four locked Task 1 folds."""

from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import QUALITY_OUTPUT, center_transform
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import COMPETITION_OUTPUT, COMPETITION_SHA
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, compare
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.run_task1_m_s_refinement_saliency_training import verified_annotation
from tools.owod.screening_data import write_json


OUTPUT = QUALITY_OUTPUT.with_name('task1-m-s-refinement-center-quality-folds')
PREFLIGHT_SHA = '6db34143914d3cd53ee9361d4ff472633618812942494b443e2510af7f40d573'


def prepare():
    released = verified(QUALITY_OUTPUT / 'comparison.json', PREFLIGHT_SHA)
    released_plan = verified(QUALITY_OUTPUT / 'plan.json', released['plan_sha256'])
    reference = verified(COMPETITION_OUTPUT / 'comparison.json', COMPETITION_SHA)
    plan = verified(COMPETITION_OUTPUT / 'plan.json', reference['plan_sha256'])
    if released['status'] != 'complete' or not released['passed'] or reference['status'] != 'complete':
        raise ValueError('Expected passed released preflight and locked competition reference')
    for protocol, job in plan['jobs'].items():
        source = released_plan['jobs'][protocol]
        keys = ('checkpoint', 'checkpoint_sha256', 'manifest_sha256')
        if any(job[key] != source[key] for key in keys):
            raise ValueError(f'Changed released endpoint: {protocol}')
        job['selector'] = released['protocols'][protocol]['selector']
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                preflight_sha256=PREFLIGHT_SHA, competition_sha256=COMPETITION_SHA,
                factor='Use the frozen released selector to move accepted learned query centers after decoding. '
                       'Keep the locked known/unknown competition and original joint VOC gates.')
    return plan, reference


@torch.inference_mode()
def evaluate(ref, model, protocol, job, plan, digest):
    manifest_path = MANIFESTS[protocol]
    manifest = verified(manifest_path, job['manifest_sha256'])
    image_root = resolve_image_root(manifest_path, manifest)
    control = verified(Path(job['control_report']), job['control_report_sha256'])['folds']
    known = manifest['stages'][0]['classes']
    quality = model.detr.transformer.instance_proposals.refined_quality
    transform = center_transform(job['selector'], known, torch.device('cuda'), quality)
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])

    def predict(samples):
        outputs = model(samples)
        return dict(outputs, pred_boxes=transform(outputs))

    folds = {}
    for fold, entry in job['folds'].items():
        dest = OUTPUT / f'{protocol}-{fold}.json'
        if not dest.exists():
            annotations = verified_annotation(entry)
            print(f'Evaluating {protocol}/{fold} frozen center selector', flush=True)
            candidate = measure(predict, manifest, image_root, annotations, plan['batch_size'], plan['limit'],
                                postprocessor=post)
            matched_count = verified(Path(entry['report']), entry['report_sha256'])['report']
            if any(ids != matched_count['coverage'][stage] for stage, ids in candidate['coverage'].items()
                   if stage != 'final'):
                raise ValueError(f'Changed upstream geometry: {protocol}/{fold}')
            reference = ref[fold]
            small = candidate['groups']['small']
            outcome = compare(control[fold], candidate, plan['gates'])
            outcome.update(
                metrics=candidate['metrics'],
                delta_vs_same_scoring={key: candidate['metrics'][key] - value
                                       for key, value in reference['candidate'].items()},
                small_final=[matched_count['groups']['small']['final'], small['final']],
                small_unknown=[reference['small_unknown'][1], small['unknown_tp']],
            )
            write_json(dest, dict(plan_sha256=digest, checkpoint_sha256=job['checkpoint_sha256'],
                                  outcome=outcome, report=candidate))
        report = read_json(dest)
        if report['plan_sha256'] != digest or report['checkpoint_sha256'] != job['checkpoint_sha256']:
            raise ValueError(f'Changed frozen fold plan: {dest}')
        folds[fold] = report['outcome']
    return folds


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the locked Task 1 replay')
    plan, reference = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        checkpoint = Path(job['checkpoint'])
        if file_sha256(checkpoint) != job['checkpoint_sha256']:
            raise ValueError(f'Changed matched-count endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        protocols[protocol] = evaluate(reference['protocols'][protocol], model, protocol, job, plan, digest)
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for folds in protocols.values()
                                                          for row in folds.values())))


if __name__ == '__main__':
    main()
