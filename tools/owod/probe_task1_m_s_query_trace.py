"""Follow each injected encoder cell and its assigned target through Task 1."""

from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_prescreen import register
from tools.owod.probe_task1_m_s_prescreen_replacement import OUTPUT as SOURCE, replay
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-query-trace')
SOURCE_PLAN = '8cf9cfb5743c2cec5f1d14712052f0c04f06e59b514e28690d624093ec2ded96'
SOURCE_RESULT = '3f54c9f6d9b3dc89f88a2ea82f6778c6d72f6316ff67e6a532e77ef0762e5e95'


def observe(rows, known, out, detections, targets, objects, cells, assignments):
    geo = out['geometry']
    layers = [layer['pred_boxes'] for layer in out['aux_outputs']] + [out['pred_boxes']]
    if len(layers) != 6:
        raise ValueError('Expected six decoder layers')
    for i, ids in cells.items():
        slots = torch.where(geo['active'][i])[0][-len(ids):]
        if not torch.equal(ids, geo['chosen'][i, slots]):
            raise ValueError('Injected assignments no longer match decoder slots')
        h, w = targets[i]['orig_size'].tolist()
        selected = geo['boxes'][i, ids]
        source_boxes = voc_coords(selected, w, h)
        decoded = [voc_coords(layer[i, slots], w, h) for layer in layers]
        target_by_id = {a['id']: a for a in objects[i]}
        probs_obj = out['pred_objectness'][i].sigmoid()
        probs_novel = out['pred_novelty'][i].sigmoid()
        unknown = probs_obj * probs_novel
        det = detections[i]
        unknown_slots = set((det['top_indices'][det['labels'] == 91] // (len(known) + 1)).tolist())
        for j, (slot, ann_id) in enumerate(zip(slots.tolist(), assignments[i], strict=True)):
            gt = np.asarray(target_by_id[ann_id]['voc_bbox'])
            start = float(overlaps(gt, source_boxes[j:j+1])[0])
            if start <= .5:
                raise ValueError(f'Injected cell lost its initial overlap for target {ann_id}')
            center = (gt[:2] + gt[2:]) / 2
            gt_wh = np.maximum(gt[2:] - gt[:2] + 1, 1)
            reference = selected[j, :2].cpu().numpy() * [w, h]
            center_distance = float(np.linalg.norm((reference + .5 - center) / gt_wh))
            first_box = decoded[0][j]
            first_distance = float(np.linalg.norm(((first_box[:2] + first_box[2:]) / 2 - center) / gt_wh))
            rows.append(dict(annotation_id=ann_id, image_id=int(targets[i]['image_id']), slot=slot,
                             cell=int(ids[j]), source_iou=start, initial_reference_xy=reference.tolist(),
                             reference_distance_gt_size=center_distance,
                             first_center_distance_gt_size=first_distance,
                             first_wh_ratio=((first_box[2:] - first_box[:2] + 1) / gt_wh).tolist(),
                             layer_ious=[float(overlaps(gt, coords[j:j+1])[0]) for coords in decoded],
                             objectness=float(probs_obj[slot]), novelty=float(probs_novel[slot]),
                             unknown_score=float(unknown[slot]),
                             unknown_rank=int((unknown > unknown[slot]).sum()) + 1,
                             unknown_top50=slot in unknown_slots,
                             top50_joint_cutoff=float(det['scores'][-1])))


def summarize(rows):
    hits = [sum(r['layer_ious'][layer] > .5 for r in rows) for layer in range(6)]
    final = [row for row in rows if row['layer_ious'][-1] > .5]
    final_scored = [r for r in final if r['unknown_top50']]
    return dict(injected=len(rows), layer_hits=hits,
                median_source_iou=median(r['source_iou'] for r in rows),
                median_layer_ious=[median(r['layer_ious'][i] for r in rows) for i in range(6)],
                median_reference_distance=median(r['reference_distance_gt_size'] for r in rows),
                median_first_center_distance=median(r['first_center_distance_gt_size'] for r in rows),
                median_first_wh_ratio=[median(r['first_wh_ratio'][axis] for r in rows) for axis in range(2)],
                final_unknown_top50=len(final_scored),
                any_unknown_top50=sum(r['unknown_top50'] for r in rows),
                median_objectness=median(r['objectness'] for r in rows),
                median_novelty=median(r['novelty'] for r in rows),
                median_unknown_rank=median(r['unknown_rank'] for r in rows),
                localized_scores=[{k: r[k] for k in ('annotation_id','objectness','novelty','unknown_score',
                                                     'unknown_top50','top50_joint_cutoff')} for r in final])


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE / 'plan.json', SOURCE_PLAN)
    summary = verified(SOURCE / 'comparison.json', SOURCE_RESULT)
    roots, inputs, cases, _ = register()
    if inputs != prior['inputs'] or summary['plan_sha256'] != SOURCE_PLAN or summary['status'] != 'complete':
        raise ValueError('Changed replacement inputs')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                replacement_plan_sha256=SOURCE_PLAN, replacement_comparison_sha256=SOURCE_RESULT,
                paper_comparable=False, inputs=inputs, torch_version=str(torch.__version__))
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for label, spec in inputs.items():
        path = OUTPUT / f'{label}.json'
        if not path.exists():
            protocol = 'm-owodb' if label.startswith('m-owodb-') else 's-owodb'
            known = read_json(MANIFESTS[protocol])['stages'][0]['classes']
            model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
            model.eval()
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            rows = []
            print(f'Evaluating {label} with decoder query tracing', flush=True)
            observed = replay(model, ds, loader, known, cases[label],
                              observer=lambda *args: observe(rows, known, *args))
            baseline = read_json(SOURCE / f'{label}.json')
            if baseline.pop('plan_sha256') != SOURCE_PLAN or observed != baseline:
                raise ValueError(f'Replacement coverage or counts changed for {label}')
            write_json(path, dict(plan_sha256=digest, rows=rows))
            del model
        report = read_json(path)
        if (report['plan_sha256'] != digest or len(report['rows']) != summary['slices'][label]['replacements']
                or len({r['annotation_id'] for r in report['rows']}) != len(report['rows'])):
            raise ValueError(f'Incomplete query trace for {label}')
        results[label] = summarize(report['rows'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
                                               plan_sha256=digest, slices=results))


if __name__ == '__main__':
    main()
