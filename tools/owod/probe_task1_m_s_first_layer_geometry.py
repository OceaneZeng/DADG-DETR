"""Attribute first-layer localization loss on fixed Task 1 injected queries."""

from functools import partial
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_prescreen import register
from tools.owod.probe_task1_m_s_prescreen_replacement import OUTPUT as REPLACEMENT, replay
from tools.owod.probe_task1_m_s_query_trace import OUTPUT as SOURCE, SOURCE_PLAN as REPLACEMENT_PLAN, observe
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-first-layer-geometry')
SOURCE_PLAN = 'd76c8f859e7a9f8ff0c224099751c21931fc206f89a6b6856af294e3ced0e2d0'
SOURCE_RESULT = '27a3a841a6837abc46775e810b3fc68b2b0646dfba9d90debdf713b0ea4d44d4'
DONORS = ('gt', 'encoder')
MODES = ('center', 'extent', 'both')
MARGIN = .10


def attribute(boxes):
    gt, encoder, first = (np.asarray(boxes[k]) for k in ('gt', 'encoder', 'first'))
    center = (first[:2] + first[2:]) / 2
    half = (first[2:] - first[:2]) / 2
    result = {}
    for label, donor in zip(DONORS, (gt, encoder), strict=True):
        donor_center = (donor[:2] + donor[2:]) / 2
        donor_half = (donor[2:] - donor[:2]) / 2
        candidates = [np.concatenate((donor_center - half, donor_center + half)),
                      np.concatenate((center - donor_half, center + donor_half)),
                      donor]
        result[label] = dict(zip(MODES, overlaps(gt, candidates).tolist(), strict=True))
    return result


def summarize(rows):
    missed = [row for row in rows if row['first_iou'] <= .5]
    recovered = {donor: {mode: sum(r['corrected_ious'][donor][mode] > .5 for r in missed)
                         for mode in MODES} for donor in DONORS}
    return dict(injected=len(rows), missed=len(missed), recovered=recovered,
                median_initial_center_error=median(r['reference_distance_gt_size'] for r in missed),
                median_first_center_error=median(r['first_center_distance_gt_size'] for r in missed),
                median_first_wh_ratio=[median(r['first_wh_ratio'][axis] for r in missed) for axis in range(2)])


def priority(slices, donor):
    differences = [(s['recovered'][donor]['center'] - s['recovered'][donor]['extent']) / s['missed']
                   for s in slices]
    if all(d >= MARGIN for d in differences):
        return 'center'
    if all(d <= -MARGIN for d in differences):
        return 'extent'
    return 'mixed'


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE / 'plan.json', SOURCE_PLAN)
    result = verified(SOURCE / 'comparison.json', SOURCE_RESULT)
    roots, inputs, cases, _ = register()
    if (inputs != prior['inputs'] or str(torch.__version__) != prior['torch_version']
            or result['plan_sha256'] != SOURCE_PLAN or result['status'] != 'complete'):
        raise ValueError('Changed query trace inputs')
    trace_hashes = {label: file_sha256(SOURCE / f'{label}.json') for label in inputs}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=SOURCE_PLAN, source_result=SOURCE_RESULT, trace_hashes=trace_hashes,
                paper_comparable=False, inputs=inputs, margin=MARGIN)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for label, spec in inputs.items():
        path = OUTPUT / f'{label}.json'
        source = verified(SOURCE / f'{label}.json', trace_hashes[label])
        if source['plan_sha256'] != SOURCE_PLAN or len(source['rows']) != result['slices'][label]['injected']:
            raise ValueError(f'Changed source trace for {label}')
        if not path.exists():
            protocol = 'm-owodb' if label.startswith('m-owodb-') else 's-owodb'
            known = read_json(MANIFESTS[protocol])['stages'][0]['classes']
            model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
            model.eval()
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            rows, boxes = [], {}
            print(f'Evaluating {label} for first-layer geometry', flush=True)
            coverage = replay(model, ds, loader, known, cases[label],
                              observer=partial(observe, rows, known, geometry=boxes))
            native = read_json(REPLACEMENT / f'{label}.json')
            if rows != source['rows'] or native.pop('plan_sha256') != REPLACEMENT_PLAN or coverage != native:
                raise ValueError(f'Changed injected queries or replacement coverage for {label}')
            evidence = []
            for row in rows:
                ann_id = row['annotation_id']
                measurements = boxes[ann_id]
                evidence.append(dict(annotation_id=ann_id, first_iou=row['layer_ious'][0],
                                     reference_distance_gt_size=row['reference_distance_gt_size'],
                                     first_center_distance_gt_size=row['first_center_distance_gt_size'],
                                     first_wh_ratio=row['first_wh_ratio'], boxes=measurements,
                                     corrected_ious=attribute(measurements)))
            write_json(path, dict(plan_sha256=digest, rows=evidence))
            del model
        report = read_json(path)
        if report['plan_sha256'] != digest or [r['annotation_id'] for r in report['rows']] != [
                r['annotation_id'] for r in source['rows']]:
            raise ValueError(f'Changed saved geometry for {label}')
        results[label] = summarize(report['rows'])
    groups = dict(medium=[x for key, x in results.items() if not key.startswith('s-extent-')],
                  small=[x for key, x in results.items() if key.startswith('s-extent-')])
    decisions = {group: {donor: priority(slices, donor) for donor in DONORS} for group, slices in groups.items()}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                               slices=results, decisions=decisions))


if __name__ == '__main__':
    main()
