"""Ablate the real first box head on frozen, injected Task 1 queries."""

from functools import partial
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_first_layer_geometry import OUTPUT as SOURCE
from tools.owod.probe_task1_m_s_prescreen import register
from tools.owod.probe_task1_m_s_prescreen_replacement import (
    OUTPUT as REPLACEMENT, replay, replace_cells,
)
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json
from util.misc import inverse_sigmoid


OUTPUT = SOURCE.with_name('task1-m-s-first-layer-forward')
PRIOR_PLAN = '32962c3322458514aa4f9896cafcb63a09bf22add4abde0646bc5089eb6289dc'
PRIOR_RESULT = 'c3baa7d9fbfdc4dcb7fd677b9ac81d56f0426fc7b7a1637b4828786166d27d9d'
MODES = ('center', 'extent', 'both')


def patch_first_box(slots, encoder_extents, mode):
    calls = 0

    def hook(_module, _inputs, deltas):
        nonlocal calls
        calls += 1
        if calls != 1:
            return None
        changed = deltas.clone()
        if mode in ('center', 'both'):
            changed[..., :2][slots] = 0
        if mode in ('extent', 'both'):
            changed[..., 2:][slots] = inverse_sigmoid(encoder_extents)
        return changed

    return hook


def observe(rows, model, post, samples, baseline, detections, targets, objects, cells, assignments):
    detr = model.detr
    geo = baseline['geometry']
    mask = torch.zeros_like(baseline['pred_boxes'][..., 0], dtype=torch.bool)
    assigned_slots = {}
    widths = []
    for i, ids in cells.items():
        indices = torch.where(geo['active'][i])[0][-len(ids):]
        if not torch.equal(geo['chosen'][i, indices], ids):
            raise ValueError('Replacement query positions changed')
        mask[i, indices] = True
        assigned_slots[i] = indices
    for i in sorted(assigned_slots):
        widths.append(geo['boxes'][i, cells[i], 2:])
    extents = torch.cat(widths)
    expected_first = baseline['aux_outputs'][0]['pred_boxes']
    results = {'native': expected_first}
    sizes = torch.stack([t['orig_size'] for t in targets]).to(expected_first.device)
    proposal = detr.transformer.instance_proposals
    for mode in MODES:
        with proposal.register_forward_hook(partial(replace_cells, replacements=cells)), (
            detr.bbox_embed[0].register_forward_hook(patch_first_box(mask, extents, mode))
        ):
            out = model(samples)
        first = out['aux_outputs'][0]['pred_boxes']
        if not torch.equal(first[~mask], expected_first[~mask]):
            raise ValueError('First box head changed outside replacement positions')
        untouched = []
        for layer, native in zip(out['aux_outputs'][1:], baseline['aux_outputs'][1:], strict=True):
            untouched.append(torch.equal(layer['pred_boxes'], native['pred_boxes']))
        for k in ('pred_boxes', 'pred_logits', 'pred_objectness', 'pred_novelty'):
            untouched.append(torch.equal(out[k], baseline[k]))
        postprocessed = post(out, sizes)
        for current, native in zip(postprocessed, detections, strict=True):
            untouched.extend(torch.equal(current[k], native[k]) for k in ('scores', 'labels', 'boxes'))
        if not all(untouched):
            raise ValueError(f'Unexpected downstream change with iterative refinement disabled: {mode}')
        results[mode] = first
    for i, indices in assigned_slots.items():
        h, w = targets[i]['orig_size'].tolist()
        decoded = {mode: voc_coords(boxes[i, indices], w, h) for mode, boxes in results.items()}
        references = {ann['id']: np.asarray(ann['voc_bbox']) for ann in objects[i]}
        for j, ann_id in enumerate(assignments[i]):
            gt = references[ann_id]
            rows.append(dict(annotation_id=ann_id, image_id=int(targets[i]['image_id']),
                             first_iou={mode: float(overlaps(gt, boxes[j:j+1])[0])
                                        for mode, boxes in decoded.items()}))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE / 'plan.json', PRIOR_PLAN)
    result = verified(SOURCE / 'comparison.json', PRIOR_RESULT)
    roots, inputs, cases, _ = register()
    if inputs != prior['inputs'] or result['plan_sha256'] != PRIOR_PLAN or result['status'] != 'complete':
        raise ValueError('Changed geometry attribution inputs')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=PRIOR_PLAN, source_result=PRIOR_RESULT, paper_comparable=False,
                inputs=inputs, modes=list(MODES), iterative_refinement=False)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for label, spec in inputs.items():
        path = OUTPUT / f'{label}.json'
        native = read_json(REPLACEMENT / f'{label}.json')
        prior_rows = read_json(SOURCE / f'{label}.json')['rows']
        if not path.exists():
            protocol = 'm-owodb' if label.startswith('m-owodb-') else 's-owodb'
            known = read_json(MANIFESTS[protocol])['stages'][0]['classes']
            model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
            model.eval()
            if model.detr.transformer.decoder.bbox_embed is not None:
                raise ValueError('Expected original decoder with iterative refinement disabled')
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=50)
            rows = []
            print(f'Evaluating {label} with first-layer box-head interventions', flush=True)
            replayed = replay(model, ds, loader, known, cases[label],
                              batch_observer=partial(observe, rows, model, post))
            if any(replayed[k] != native[k] for k in ('rows', 'query_counts')):
                raise ValueError(f'Changed native replacement coverage in {label}')
            write_json(path, dict(plan_sha256=digest, rows=rows))
            del model
        report = read_json(path)
        rows = report['rows']
        if report['plan_sha256'] != digest or len(rows) != len(prior_rows):
            raise ValueError(f'Changed head intervention report in {label}')
        for row, previous in zip(rows, prior_rows, strict=True):
            baseline = previous['first_iou']
            if row['annotation_id'] != previous['annotation_id'] or row['first_iou']['native'] != baseline:
                raise ValueError(f'Changed native head or target pairing in {label}')
        summaries[label] = dict(injected=len(rows),
                                first={mode: sum(row['first_iou'][mode] > .5 for row in rows)
                                       for mode in ('native', *MODES)},
                                native_final=sum(row['replacement']['final'] for row in native['rows']),
                                native_unknown=sum(row['replacement']['unknown'] for row in native['rows']),
                                downstream_identical=True)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
                                               plan_sha256=digest, slices=summaries))


if __name__ == '__main__':
    main()
