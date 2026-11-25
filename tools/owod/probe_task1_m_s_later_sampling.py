"""Measure later decoder sampling on the locked Task 1 injection cohorts."""

from contextlib import ExitStack
from functools import partial
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_first_layer_forward import OUTPUT as SOURCE
from tools.owod.probe_task1_m_s_prescreen import register
from tools.owod.probe_task1_m_s_prescreen_replacement import (
    OUTPUT as REPLACEMENT, coverage, replay, replace_cells,
)
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-later-sampling')
SOURCE_PLAN = 'eef0b6a2b33bb0ec46bab483e1bc6c242bdea634651715cb6a49735eda513d96'
SOURCE_RESULT = 'cd7d6eeeb72079e59de4b9ecf260cd9666a7c55362b598661ba97371694a7f1c'
MODES = ('native', 'first_center', 'encoder_box')
GAIN = .05


def sampler_hook(stack, slots, boxes, valid_ratios, use_extent, *, keep_center=False):
    """Apply later sampling changes only at the injected query positions."""
    def change_layer(_layer, inputs):
        refs, shapes = inputs[2], inputs[4]
        ratios = valid_ratios['value'][slots[0]]
        if refs.shape[-1] != 2:
            raise ValueError('Expected the native two-dimensional sampling references')
        if use_extent:
            # For 2D sampling with denominator (W,H), this scale yields
            # the native 4D branch displacement offset * (w,h) / (2 * points).
            wh = boxes[:, None, 2:] * ratios
            normalizer = shapes[:, [1, 0]].to(wh)
            scale = wh * normalizer / (2 * _layer.cross_attn.n_points)

            def rescale(_module, _inputs, output):
                attn = _layer.cross_attn
                offsets = output.reshape(*output.shape[:2], attn.n_heads, attn.n_levels, attn.n_points, 2)
                offsets = offsets.clone()
                offsets[slots] *= scale[:, None, :, None, :]
                return offsets.reshape_as(output)

            stack.enter_context(_layer.cross_attn.sampling_offsets.register_forward_hook(rescale))
        if keep_center:
            return
        # The layer sees references already scaled by valid image ratios.
        changed = refs.clone()
        changed[slots] = boxes[:, None, :2] * ratios
        return (*inputs[:2], changed, *inputs[3:])

    return change_layer


def run_variant(model, samples, cells, slots, boxes, use_extent):
    decoder = model.detr.transformer.decoder
    proposal = model.detr.transformer.instance_proposals
    with ExitStack() as stack:
        stack.enter_context(proposal.register_forward_hook(partial(replace_cells, replacements=cells)))
        ratios = {}

        def capture(_module, inputs):
            ratios['value'] = inputs[5]

        stack.enter_context(decoder.register_forward_pre_hook(capture))
        for layer in decoder.layers[1:]:
            stack.enter_context(layer.register_forward_pre_hook(
                sampler_hook(stack, slots, boxes, ratios, use_extent)))
        return model(samples)


def observe(rows, model, post, samples, baseline, detections, targets, objects, cells, assignments):
    geo = baseline['geometry']
    first = baseline['aux_outputs'][0]['pred_boxes']
    positions = {i: torch.where(geo['active'][i])[0][-len(ids):] for i, ids in cells.items()}
    slots = (torch.cat([q.new_full(q.shape, i) for i, q in positions.items()]),
             torch.cat(list(positions.values())))
    encoder = geo['boxes'][slots[0], geo['chosen'][slots]]
    first_boxes = first[slots]
    sizes = torch.stack([t['orig_size'] for t in targets]).to(first.device)
    variants = {'native': (baseline, detections)}
    for label, boxes in (('first_center', first_boxes), ('encoder_box', encoder)):
        changed = run_variant(model, samples, cells, slots, boxes, label == 'encoder_box')
        if (not torch.equal(first, changed['aux_outputs'][0]['pred_boxes'])
                or not torch.equal(geo['active'], changed['geometry']['active'])
                or not torch.equal(geo['chosen'], changed['geometry']['chosen'])):
            raise ValueError(f'Changed first-layer outputs or query slot identities in {label}')
        variants[label] = (changed, post(changed, sizes))
    for i, assigned in positions.items():
        h, w = targets[i]['orig_size'].tolist()
        by_id = {ann['id']: ann for ann in objects[i]}
        measured = {}
        for label, (output, detected) in variants.items():
            layers = [layer['pred_boxes'] for layer in output['aux_outputs']] + [output['pred_boxes']]
            measured[label] = (coverage(output, detected, i, targets[i], objects[i]),
                               [voc_coords(layer[i, assigned], w, h) for layer in layers])
        for j, ann_id in enumerate(assignments[i]):
            gt = np.asarray(by_id[ann_id]['voc_bbox'])
            rows.append(dict(annotation_id=ann_id, image_id=int(targets[i]['image_id']),
                             modes={label: dict(layer_ious=[float(overlaps(gt, layer[j:j+1])[0]) for layer in layers],
                                                **result[ann_id])
                                    for label, (result, layers) in measured.items()}))


def summarize(rows):
    views = {mode: dict(layer_hits=[sum(row['modes'][mode]['layer_ious'][i] > .5 for row in rows)
                                   for i in range(6)],
                        final=sum(row['modes'][mode]['final'] for row in rows),
                        unknown=sum(row['modes'][mode]['unknown'] for row in rows))
             for mode in MODES}
    return dict(assigned=len(rows), modes=views,
                passes={mode: all((views[mode][metric] - views['native'][metric]) / len(rows) >= GAIN
                                  for metric in ('final', 'unknown')) for mode in MODES[1:]})


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE / 'plan.json', SOURCE_PLAN)
    result = verified(SOURCE / 'comparison.json', SOURCE_RESULT)
    roots, inputs, cases, _ = register()
    if inputs != prior['inputs'] or result['plan_sha256'] != SOURCE_PLAN or result['status'] != 'complete':
        raise ValueError('Changed first-layer forward evidence')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=SOURCE_PLAN, source_result=SOURCE_RESULT, inputs=inputs,
                paper_comparable=False, modes=list(MODES), gate_gain=GAIN)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for label, spec in inputs.items():
        path = OUTPUT / f'{label}.json'
        reference = read_json(SOURCE / f'{label}.json')['rows']
        if not path.exists():
            protocol = 'm-owodb' if label.startswith('m-owodb-') else 's-owodb'
            known = read_json(MANIFESTS[protocol])['stages'][0]['classes']
            model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
            model.eval()
            if model.detr.transformer.decoder.bbox_embed is not None:
                raise ValueError('Expected frozen decoder with iterative refinement disabled')
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=50)
            rows = []
            print(f'Evaluating {label} with later sampling interventions', flush=True)
            replayed = replay(model, ds, loader, known, cases[label],
                              batch_observer=partial(observe, rows, model, post))
            native = read_json(REPLACEMENT / f'{label}.json')
            if any(replayed[k] != native[k] for k in ('rows', 'query_counts')):
                raise ValueError(f'Changed baseline replacement coverage in {label}')
            write_json(path, dict(plan_sha256=digest, rows=rows))
            del model
        report = read_json(path)
        rows = report['rows']
        if report['plan_sha256'] != digest or len(rows) != len(reference):
            raise ValueError(f'Changed report for {label}')
        for row, expected in zip(rows, reference, strict=True):
            if (row['annotation_id'] != expected['annotation_id']
                    or row['modes']['native']['layer_ious'][0] != expected['first_iou']['native']):
                raise ValueError(f'Changed target or first-layer geometry in {label}')
        summaries[label] = summarize(rows)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
                                               plan_sha256=digest, slices=summaries))


if __name__ == '__main__':
    main()
