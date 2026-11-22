"""Trace native quality ranks for the fixed Task 1 prescreen cohorts."""

from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch

from models.graph_local.instance_proposals import nearby_anchor_ids
from models.owod_benchmark_eval import overlaps
from tools.owod import prepare_benchmarks
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_small_box_geometry import SOURCE_FILES as PAIRED_HASHES
from tools.owod.probe_task1_paired_discovery import OUTPUT as PAIRED, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, OUTPUT as EXTENT, size_group
from tools.owod.screening_data import write_json


OUTPUT = PAIRED.with_name('task1-m-s-prescreen-ranks')
PAIRED_PLAN = '50279971f12e4e79ba0a604f1c82c8e8db870cc3576893e6c97d47f904962ebb'
EXTENT_PLAN = 'fd056b5878413cdb35917a3b5c3e3c9fefbcde55cad22ab89c0cff08a6cf3691'
EXTENT_REPORTS = {
    'control': '7fb8d49704bd8a15cb345ca9ff97ee55cd0199fbd18c5863485e9ccb0d9b7ca6',
    'extent': 'bf3b3e5ca26d9f793c6ca5791f9c5451ff1920c995dd19704b0c03db7245beb7',
}
SLICES = ('first', 'second')
STAGES = ('dense', 'shortlist', 'selected', 'final')


def released_boxes(objects, known):
    return [np.asarray(ann['voc_bbox']) for ann in objects if ann['category_id'] in known
            and not ann.get('difficult') and not ann.get('iscrowd')]


def register():
    paired = verified(PAIRED / 'plan.json', PAIRED_PLAN)
    extent = verified(EXTENT / 'plan.json', EXTENT_PLAN)
    reports = {arm: verified(EXTENT / f's-owodb-{arm}.json', digest) for arm, digest in EXTENT_REPORTS.items()}
    if any(report['plan_sha256'] != EXTENT_PLAN for report in reports.values()):
        raise ValueError('Changed S extent provenance')
    protocols = paired['parents']
    manifests = {p: verified(MANIFESTS[p], row['manifest_sha256']) for p, row in protocols.items()}
    roots = {p: resolve_image_root(MANIFESTS[p], m) for p, m in manifests.items()}
    _, s_objects, hashes = prepare_benchmarks.load_coco(roots['s-owodb'])
    expected = {Path(path).name: digest for path, digest in manifests['s-owodb']['source_hashes'].items()}
    if {Path(path).name: digest for path, digest in hashes.items()} != expected:
        raise ValueError('Changed S released annotations')
    inputs, cases, competition = {}, {}, {}
    for name in SLICES:
        path = PAIRED / f'{name}-annotations.json'
        payload = verified(path, PAIRED_HASHES[path.name])
        rows = {}
        for protocol in protocols:
            report_path = PAIRED / f'{protocol}-{name}.json'
            report = verified(report_path, PAIRED_HASHES[report_path.name])
            if report['plan_sha256'] != PAIRED_PLAN:
                raise ValueError('Changed paired discovery report')
            rows[protocol] = {r['annotation_id']: r for r in report['rows']}
        shared = set.intersection(*({i for i, row in group.items() if row['missed_group'] == 'outside_shortlist'}
                                    for group in rows.values()))
        targets = [a for a in payload['annotations'] if a['id'] in shared and
                   32**2 <= a['bbox'][2] * a['bbox'][3] < 96**2]
        if len(targets) != {'first': 187, 'second': 372}[name]:
            raise ValueError('Changed shared medium cohort')
        image_ids = {a['image_id'] for a in targets}
        for protocol, row in protocols.items():
            key = f'{protocol}-{name}'
            inputs[key] = dict(checkpoint=row['checkpoint'], annotation=str(path), annotation_sha256=PAIRED_HASHES[path.name],
                               batch_size=paired['batch_size'], image_protocol='m-owodb', targets=len(targets))
            cases[key] = {a['id']: rows[protocol][a['id']]['coverage'] for a in targets}
            known = set(paired['known'][protocol])
            competition[key] = {}
            for im in payload['images']:
                if im['id'] not in image_ids:
                    continue
                stem = im['owod_source_id']
                objects = (prepare_benchmarks.read_voc(roots[protocol].parent, stem)[1]
                           if protocol == 'm-owodb' else s_objects[stem])
                competition[key][im['id']] = released_boxes(objects, known)

        spec = extent['protocols']['s-owodb']['slices'][name]
        path = EXTENT / spec['file']
        payload = verified(path, spec['sha256'])
        old, new = (reports[arm]['slices'][name]['coverage'] for arm in ('control', 'extent'))
        new = {stage: set(new[stage]) for stage in STAGES}
        gained = new['dense'] - set(old['dense'])
        targets = [a for a in payload['annotations'] if a['id'] in gained and size_group(a) != '32_sq_plus']
        if len(targets) != {'first': 223, 'second': 237}[name]:
            raise ValueError('Changed S extent small cohort')
        key = f's-extent-{name}'
        inputs[key] = dict(checkpoint=dict(path=str(EXTENT / 's-owodb/extent/task1/checkpoint.pth'),
                                            sha256=reports['extent']['checkpoint_sha256']),
                           annotation=str(path), annotation_sha256=spec['sha256'],
                           batch_size=extent['validation_batch_size'], image_protocol='s-owodb', targets=len(targets))
        cases[key] = {a['id']: {stage: a['id'] in new[stage] for stage in STAGES} for a in targets}
        known = set(paired['known']['s-owodb'])
        competition[key] = {im['id']: [] for im in payload['images']}
        for ann in payload['annotations']:
            competition[key][ann['image_id']].extend(released_boxes([ann], known))
    for case in inputs.values():
        checkpoint = case['checkpoint']
        if file_sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError('Changed frozen checkpoint')
    return roots, inputs, cases, competition


@torch.inference_mode()
def trace(model, dataset, loader, cases, released):
    rows, images = [], {}
    for step, (samples, targets) in enumerate(loader, 1):
        wanted = [(i, int(t['image_id'])) for i, t in enumerate(targets)
                  if any(a['id'] in cases for a in dataset.coco.imgToAnns[int(t['image_id'])])]
        if wanted:
            out = model(samples.to('cuda'))
            geo = out['geometry']
            for i, image in wanted:
                t = targets[i]
                annotations = valid_annotations(dataset, image)
                if t['labels'].tolist() != [a['category_id'] for a in annotations]:
                    raise ValueError(f'Changed transformed target order for image {image}')
                selected = [(j, ann) for j, ann in enumerate(annotations) if ann['id'] in cases]
                ids = torch.where(geo['valid'][i])[0]
                scores = geo['logits'][i, ids]
                near = nearby_anchor_ids(t['boxes'][[j for j, _ in selected]].to(ids.device), geo['anchors'][i], ids)
                near = torch.searchsorted(ids, near).cpu().numpy()
                h, w = t['orig_size'].tolist()
                stages = {k: voc_coords(b, w, h) for k, b in stage_boxes(out, i).items() if k in STAGES}
                top = stages['shortlist']
                covered = np.zeros(len(top), dtype=bool)
                for gt in released[image]:
                    covered |= overlaps(gt, top) > .5
                images[image] = dict(native_pool=len(top), covering_released=int(covered.sum()))
                score_values = scores.cpu().numpy()
                for nearby, (_, ann) in zip(near, selected):
                    gt = np.asarray(ann['voc_bbox'])
                    iou = overlaps(gt, stages['dense'])
                    accurate = iou > .5
                    actual = {k: bool((overlaps(gt, boxes) > .5).any()) for k, boxes in stages.items()}
                    if actual != cases[ann['id']] or not accurate.any():
                        raise ValueError(f'Changed native coverage for target {ann["id"]}: {actual}')
                    highest = score_values[accurate].max()
                    rows.append(dict(annotation_id=ann['id'], image_id=image, coverage=actual,
                                     rank=1 + int((score_values > highest).sum()),
                                     last_tied_rank=int((score_values >= highest).sum()),
                                     valid_cells=len(ids), accurate_cells=int(accurate.sum()),
                                     sampled_correct=int(accurate[nearby].sum()),
                                     best_sample_iou=float(iou[nearby].max())))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if len(rows) != len(cases) or {row['annotation_id'] for row in rows} != set(cases):
        raise ValueError('Replay did not cover the registered target identities')
    return dict(rows=rows, images=images)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one verified RTX 3090 for the frozen replay')
    roots, inputs, cases, competition = register()
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plans=[PAIRED_PLAN, EXTENT_PLAN], inputs=inputs, paper_comparable=False,
                torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name())
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    summaries = {}
    for key, spec in inputs.items():
        path = OUTPUT / f'{key}.json'
        if not path.exists():
            checkpoint = Path(spec['checkpoint']['path'])
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            ds, loader = make_loader(roots[spec['image_protocol']], Path(spec['annotation']), spec['batch_size'], 2)
            print(f'Evaluating {key} on the original batches', flush=True)
            result = trace(model, ds, loader, cases[key], competition[key])
            write_json(path, dict(plan_sha256=plan_hash, **result))
            del model
        report = read_json(path)
        rows = report['rows']
        if (report['plan_sha256'] != plan_hash or len(rows) != len(cases[key])
                or {r['annotation_id']: r['coverage'] for r in rows} != cases[key]):
            raise ValueError(f'Changed replay report for {key}')
        summaries[key] = dict(targets=len(rows), sampled=sum(r['sampled_correct'] > 0 for r in rows),
                              median_rank=median(r['rank'] for r in rows),
                              pool_cells=sum(im['native_pool'] for im in report['images'].values()),
                              pool_covering_released=sum(im['covering_released'] for im in report['images'].values()))
    medium = [s for key, s in summaries.items() if not key.startswith('s-extent-')]
    decision = ('quality_ranking' if all(2 * s['sampled'] > s['targets'] for s in medium)
                else 'sampling' if all(2 * s['sampled'] < s['targets'] for s in medium) else 'mixed')
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
                                               slices=summaries, medium_priority=decision))


if __name__ == '__main__':
    main()
