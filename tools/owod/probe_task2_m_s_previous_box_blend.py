"""Screen a fixed penultimate/final box blend for previous-class predictions."""

import json
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_low_iou_box_head import ARMS, OUTPUT as CROSS, TRACE_REPORT_SHAS
from tools.owod.probe_task2_m_s_previous_hard_negative_misses import (
    BOX_COMPARISON_SHA, BOX_REPORT_SHAS, BOX_TRACE, match_ids,
)
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_previous_hard_negatives import BOX_OUTPUT, FULL, MANIFESTS, SOURCE, compare
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = CROSS.with_name('task2-m-s-previous-box-layer-blend')
CROSS_SHA = 'd78f80fb817819dfb35f9934fdd755fa3b979fee5b290e8d7c8943a37949175a'
WEIGHT = .5
MIN_PREVIOUS_GAIN = .0005
DEVICE = torch.device('cuda')


class PreviousBoxBlend(DiscoveryPostProcess):
    def __init__(self, known, previous, limit):
        super().__init__(known_ids=known, scoring='focal', max_detections=limit)
        self.previous = previous

    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        if len(outputs['aux_outputs']) != 5:
            raise ValueError('Expected the penultimate output of six decoder layers')
        results = super().forward(outputs, target_sizes, include_top_indices=True)
        penultimate = box_cxcywh_to_xyxy(outputs['aux_outputs'][-1]['pred_boxes'])
        heights, widths = target_sizes.unbind(1)
        scale = torch.stack([widths, heights, widths, heights], -1)
        previous = torch.as_tensor(self.previous, device=penultimate.device)
        for result, boxes, factors in zip(results, penultimate, scale, strict=True):
            indices = result['top_indices'] if include_top_indices else result.pop('top_indices')
            queries = indices // (len(self.known_ids) + 1)
            old = torch.isin(result['labels'], previous)
            result['boxes'][old] = torch.lerp(result['boxes'][old], boxes[queries[old]] * factors, WEIGHT)
        return results


def arm_report(report, view):
    return dict(views={'native': report['views'][view]},
                previous_matches={'native': report['previous_matches'][view]})


def main():
    cross = verified(CROSS / 'comparison.json', CROSS_SHA)
    source = verified(BOX_OUTPUT / 'comparison.json', BOX_COMPARISON_SHA)
    parent = verified(BOX_OUTPUT / 'plan.json', source['plan_sha256'])
    if cross['status'] != 'complete' or cross['crossover_passed'] or parent['max_detections'] != 50:
        raise ValueError('Expected the completed native crossover below its geometry gate')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                cross_sha256=CROSS_SHA, source_plan_sha256=source['plan_sha256'],
                source_report_shas=BOX_REPORT_SHAS, trace_report_shas=TRACE_REPORT_SHAS,
                eval_batch=parent['eval_batch'], max_detections=parent['max_detections'],
                penultimate_weight=WEIGHT, min_previous_gain=MIN_PREVIOUS_GAIN, paper_comparable=False,
                decision='The control gains at least 0.05 pp Previous AP50 in each fold, changes no Current AP50 '
                         'and gains a net old-class VOC match. The weighted candidate is diagnostic.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in parent['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        reference = verified(BOX_OUTPUT / f'{protocol}.json', BOX_REPORT_SHAS[protocol])
        previous, current = job['previous'], job['current']
        post = PreviousBoxBlend(previous + current, previous, plan['max_detections'])
        results[protocol] = {}
        for arm in ARMS:
            checkpoint = (Path(job['endpoints']['finetune']['path']) if arm == 'control'
                          else BOX_OUTPUT / protocol / 'finetune/checkpoint.pth')
            if file_sha256(checkpoint) != reference[f'{arm}_sha256']:
                raise ValueError(f'Changed native checkpoint: {protocol}/{arm}')
            model, _ = load_detector(None, checkpoint, DEVICE)
            model.eval()
            results[protocol][arm] = {}
            for fold, cohort in job['folds'].items():
                label = f'{protocol}-{fold}'
                marker = OUTPUT / f'{label}-{arm}.json'
                if not marker.exists():
                    released = SOURCE / cohort['released']['file']
                    scored = FULL / cohort['scored']['file']
                    if (file_sha256(released) != cohort['released']['sha256']
                            or file_sha256(scored) != cohort['scored']['sha256']):
                        raise ValueError(f'Changed evaluation annotations: {label}')
                    trace = verified(BOX_TRACE / f'{label}.json', TRACE_REPORT_SHAS[label])
                    tracked = {r['annotation'] for r in trace['rows'] if r['trace_outcome'] == 'decoder_6'}
                    dataset, loader = make_loader(root, released, plan['eval_batch'], 2)
                    spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                    print(f'Evaluating {protocol}/{fold} {arm}', flush=True)
                    report = measure(model, dataset, loader, previous + current, previous, current, spec,
                                     plan['max_detections'], evaluation_coco=COCO(str(scored)),
                                     extra_posts={'blended': post}, include_perturbed=False, audit_previous_matches=True)
                    native, blend = (arm_report(report, view) for view in ('native', 'blended'))
                    expected = reference[arm][fold]
                    if json.loads(json.dumps(native)) != arm_report(expected, 'native') or any(
                            item['native'] != item['blended'] for item in report['per_image'].values()):
                        raise ValueError(f'Changed native replay or output composition: {label}/{arm}')
                    if native['views']['native']['matched'] != blend['views']['native']['matched']:
                        raise ValueError(f'Blend changed Current VOC matches: {label}/{arm}')
                    summary = compare(native, blend)
                    old = summary['top50_voc_matches']['previous']
                    delta = summary['delta']
                    summary['passed'] = (delta['previous_ap50'] >= MIN_PREVIOUS_GAIN and
                                         delta['current_ap50'] == 0 and old['gained'] > old['lost'])
                    summary['tracked'] = dict(count=len(tracked),
                                              native_matched=len(tracked & match_ids(native)),
                                              blended_matched=len(tracked & match_ids(blend)))
                    summary['metrics'] = {name: r['views']['native']['metrics'] for name, r in
                                          (('native', native), ('blended', blend))}
                    write_json(marker, dict(plan_sha256=digest, summary=summary))
                row = read_json(marker)
                if row['plan_sha256'] != digest:
                    raise ValueError(f'Changed result: {label}/{arm}')
                results[protocol][arm][fold] = row['summary']
            del model
    follow_up = all(r['passed'] for arms in results.values() for r in arms['control'].values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                                follow_up=follow_up))


if __name__ == '__main__':
    main()
