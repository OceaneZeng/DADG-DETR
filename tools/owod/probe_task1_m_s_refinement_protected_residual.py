"""Preflight residual geometry while retaining native known Top-50 emissions."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_residual import OUTPUT as RESIDUAL_OUTPUT, residual_transform
from tools.owod.probe_task1_m_s_refinement_center_residual_trace import OUTPUT as TRACE_OUTPUT, RESIDUAL_SHA
from tools.owod.probe_task1_m_s_refinement_decoder_box_preflight import measure
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = RESIDUAL_OUTPUT.with_name('task1-m-s-refinement-protected-residual')
TRACE_SHA = 'bc850b427bc678afca2f958e0cce11265dd65a12c58b9f714762a26d08b51e06'


def protect_known(residual, post, known, counts):
    columns = len(known) + 1

    def transform(outputs):
        original = outputs['pred_boxes']
        revised = residual(outputs)
        unit = original.new_ones((len(original), 2))
        baseline = post(outputs, unit, include_top_indices=True)
        protected = [row['top_indices'][row['labels'] != 91] for row in baseline]
        for batch, identities in enumerate(protected):
            queries = identities // columns
            counts['restored_known_queries'] += int((original[batch, queries] != revised[batch, queries]).any(dim=1).sum())
            revised[batch, queries] = original[batch, queries]
        candidate = post(dict(outputs, pred_boxes=revised), unit, include_top_indices=True)
        for batch, identities in enumerate(protected):
            if not torch.isin(identities, candidate[batch]['top_indices']).all():
                revised[batch] = original[batch]
                counts['reverted_images'] += 1
        return revised

    return transform


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the protected residual')
    source = verified(RESIDUAL_OUTPUT / 'comparison.json', RESIDUAL_SHA)
    evidence = verified(TRACE_OUTPUT / 'comparison.json', TRACE_SHA)
    plan = verified(RESIDUAL_OUTPUT / 'plan.json', source['plan_sha256'])
    if source['status'] != 'complete' or source['passed'] or evidence['status'] != 'complete':
        raise ValueError('Expected the stopped residual and its completed trace')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                residual_comparison_sha256=RESIDUAL_SHA, trace_sha256=TRACE_SHA,
                factor='Keep the boxes of all known queries emitted by native Top-50. Revert that image '
                       'when the locked postprocessor displaces a protected known query and class.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
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
            known = job['classes']
            frozen = source['protocols'][protocol]
            sampler = model.detr.transformer.instance_proposals.refined_features
            residual = residual_transform(frozen['residual'], sampler, device)
            counts = Counter(restored_known_queries=0, reverted_images=0)
            post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
            transform = protect_known(residual, post, known, counts)
            print(f'Evaluating {protocol} protected center residual', flush=True)
            report = measure(model, known, loader, plan, box_transform=transform)
            for group, row in report['groups'].items():
                expected = frozen['groups'][group]
                if any(row[key] != expected[key] for key in ('gt', 'final', 'native_known_tp')):
                    raise ValueError(f'Changed native released metrics: {protocol}/{group}')
                if row['revised_known_tp'] < row['native_known_tp']:
                    raise ValueError(f'A protected known match was lost: {protocol}/{group}')
            small = report['groups']['small']
            passed = small['revised'] - small['final'] >= int(protocol == 's-owodb')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, protection=counts, passed=passed, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed protected residual plan: {dest}')
        protocols[protocol] = {key: report[key] for key in ('protection', 'groups', 'changes', 'passed')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
