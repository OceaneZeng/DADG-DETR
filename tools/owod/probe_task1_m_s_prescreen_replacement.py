"""Measure label-guided Task 1 replacements at the native encoder query budget."""

from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_prescreen import OUTPUT as SOURCE, register
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-prescreen-replacement')
RANK_PLAN = 'dc0c336142599f65119b1e80ef3bc91a1d079cf6398e4d406a55a152b3926679'
RANK_RESULT = '432020813db5e42742b3f6a50b26b08b76fb7d2ffdffeefa95fb3d6c20344a0a'
MAX_REPLACEMENTS = 2
MIN_GAIN = .10


def replace_cells(module, inputs, output, replacements):
    memory = inputs[0]
    *mixed, geometry = output
    active = geometry['active']
    chosen = geometry['chosen'].clone()
    changed = torch.zeros_like(active)
    for i, cells in replacements.items():
        slots = torch.where(active[i])[0][-len(cells):]
        chosen[i, slots] = cells
        changed[i, slots] = True
    batch = torch.arange(len(memory), device=memory.device)[:, None]
    boxes = geometry['boxes'][batch, chosen].detach()
    new_values = (module.position(boxes), memory[batch, chosen], boxes[..., :2])
    quota = chosen.shape[1]
    tensors = tuple(torch.cat((torch.where(changed[..., None], replacement, value[:, :quota]),
                               value[:, quota:]), dim=1)
                    for value, replacement in zip(mixed, new_values))
    return (*tensors, dict(geometry, chosen=chosen))


def emit(out, targets, post):
    sizes = torch.stack([t['orig_size'] for t in targets]).to(out['pred_boxes'].device)
    return post(out, sizes)


def coverage(out, detections, i, target, objects):
    geom = out['geometry']
    h, w = target['orig_size'].tolist()
    selected = geom['chosen'][i, geom['active'][i]]
    boxes = voc_coords(geom['boxes'][i, selected], w, h)
    final = voc_coords(out['pred_boxes'][i], w, h)
    det = detections[i]
    unknown = np.asarray([voc_prediction_box(b) for b in det['boxes'][det['labels'] == 91].tolist()])
    return {ann['id']: {k: bool((overlaps(np.asarray(ann['voc_bbox']), coords) > .5).any())
                         for k, coords in (('selected', boxes), ('final', final), ('unknown', unknown))}
            for ann in objects}


def choose(out, i, target, objects, before, *, assignments=None):
    geom = out['geometry']
    ids = torch.where(geom['valid'][i])[0]
    active = geom['chosen'][i, geom['active'][i]]
    if not len(active):
        return ids[:0]
    h, w = target['orig_size'].tolist()
    boxes = voc_coords(geom['boxes'][i, ids], w, h)
    unavailable = np.isin(ids.cpu().numpy(), active.cpu().numpy())
    proposals = []
    for ann in objects:
        if before[ann['id']]['selected']:
            continue
        iou = overlaps(np.asarray(ann['voc_bbox']), boxes)
        order = np.flatnonzero((iou > .5) & ~unavailable)
        for index in order:
            proposals.append((-iou[index], ann['id'], int(index)))
    assigned, used, chosen = set(), set(), []
    for _, ann, index in sorted(proposals):
        if ann in assigned or index in used:
            continue
        assigned.add(ann)
        used.add(index)
        chosen.append(index)
        if assignments is not None:
            assignments.append(ann)
        if len(chosen) == min(MAX_REPLACEMENTS, len(active)):
            break
    return ids[chosen]


@torch.inference_mode()
def replay(model, dataset, loader, known, cases, *, observer=None, batch_observer=None):
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=50)
    module = model.detr.transformer.instance_proposals
    rows, counts = [], []
    for batch, (samples, targets) in enumerate(loader, 1):
        objects = {i: [ann for ann in dataset.coco.imgToAnns[int(t['image_id'])] if ann['id'] in cases]
                   for i, t in enumerate(targets)}
        objects = {i: v for i, v in objects.items() if v}
        if objects:
            samples = samples.to('cuda')
            native = model(samples)
            detections = emit(native, targets, post)
            before = {i: coverage(native, detections, i, targets[i], v) for i, v in objects.items()}
            assignments = {i: [] for i in objects}
            cells = {i: choose(native, i, targets[i], v, before[i], assignments=assignments[i])
                     for i, v in objects.items()}
            cells = {i: indices for i, indices in cells.items() if len(indices)}
            if cells:
                with module.register_forward_hook(lambda m, x, y: replace_cells(m, x, y, cells)):
                    changed = model(samples)
                if not torch.equal(changed['geometry']['active'], native['geometry']['active']):
                    raise ValueError('Changed actual encoder query counts')
                selected = changed['geometry']['chosen']
                for i, indices in cells.items():
                    if not torch.isin(indices, selected[i, changed['geometry']['active'][i]]).all():
                        raise ValueError('Replacement did not reach decoder query construction')
                sizes = torch.stack([t['orig_size'] for t in targets]).to(changed['pred_boxes'].device)
                alternate = post(changed, sizes, include_top_indices=observer is not None)
                if observer is not None:
                    observer(changed, alternate, targets, objects, cells, assignments)
                if batch_observer is not None:
                    batch_observer(samples, changed, alternate, targets, objects, cells, assignments)
            else:
                changed, alternate = native, detections
            for i, v in objects.items():
                active = native['geometry']['active'][i]
                counts.append(dict(image_id=int(targets[i]['image_id']), native=int(active.sum()),
                                   replacements=len(cells[i]) if i in cells else 0))
                after = coverage(changed, alternate, i, targets[i], v)
                for ann, previous in before[i].items():
                    if any(previous[k] != cases[ann][k] for k in ('selected', 'final')) or previous['unknown']:
                        raise ValueError(f'Baseline coverage changed for target {ann}')
                    rows.append(dict(annotation_id=ann, native=previous, replacement=after[ann]))
        if batch % 64 == 0 or batch == len(loader):
            print(f'Inference {batch}/{len(loader)} batches', flush=True)
    if len(rows) != len(cases) or {r['annotation_id'] for r in rows} != set(cases):
        raise ValueError('Changed cohort identities')
    return dict(rows=rows, query_counts=counts)


def summarize(report):
    rows = report['rows']
    views = {view: {k: sum(r[view][k] for r in rows) for k in ('selected', 'final', 'unknown')}
             for view in ('native', 'replacement')}
    new = views['replacement']
    return dict(targets=len(rows), images=len(report['query_counts']),
                replacements=sum(row['replacements'] for row in report['query_counts']),
                **views, passed=((new['final'] - views['native']['final']) / len(rows) >= MIN_GAIN
                                 and (new['unknown'] - views['native']['unknown']) / len(rows) >= MIN_GAIN))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    prior = verified(SOURCE / 'plan.json', RANK_PLAN)
    result = verified(SOURCE / 'comparison.json', RANK_RESULT)
    if result['status'] != 'complete' or result['medium_priority'] != 'quality_ranking':
        raise ValueError('Missing quality-rank evidence')
    roots, inputs, cases, _ = register()
    if inputs != prior['inputs'] or str(torch.__version__) != prior['torch_version']:
        raise ValueError('Changed frozen replay inputs')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                rank_plan_sha256=RANK_PLAN, rank_comparison_sha256=RANK_RESULT, paper_comparable=False,
                inputs=inputs, max_replacements=MAX_REPLACEMENTS, minimum_gain=MIN_GAIN)
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for label, spec in inputs.items():
        path = OUTPUT / f'{label}.json'
        if not path.is_file():
            protocol = 'm-owodb' if label.startswith('m-owodb-') else 's-owodb'
            known = read_json(MANIFESTS[protocol])['stages'][0]['classes']
            model, _ = load_detector(None, Path(spec['checkpoint']['path']), torch.device('cuda'))
            model.eval()
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            print(f'Evaluating {label} with at most two fixed-budget replacements per image', flush=True)
            write_json(path, dict(plan_sha256=plan_hash, **replay(model, ds, loader, known, cases[label])))
            del model
        report = read_json(path)
        if report['plan_sha256'] != plan_hash or {r['annotation_id'] for r in report['rows']} != set(cases[label]):
            raise ValueError(f'Changed saved evidence for {label}')
        summaries[label] = summarize(report)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False,
                                               plan_sha256=plan_hash, slices=summaries))


if __name__ == '__main__':
    main()
