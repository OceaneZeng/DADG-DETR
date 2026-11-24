"""Trace selected encoder proposal slots through the fixed Task 1 decoder endpoints."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import MATCHED_COUNT_OUTPUT, QUALITY_OUTPUT
from tools.owod.run_task1_m_s_released_small_extent import size_group
from tools.owod.screening_data import write_json


OUTPUT = QUALITY_OUTPUT.with_name('task1-m-s-refinement-selected-slot-trace')
ARMS = {
    'matched_count': (MATCHED_COUNT_OUTPUT, '8d8b5817191b7b491f52ff247eb87dbeb41f310b7ceb59ad4ebb45f37d038181'),
    'quality': (QUALITY_OUTPUT, '8ac23ca0c1eb44c2b71ef8f73fd31f7f400585e5db13be76848ccd2182516292'),
}


def prepare():
    plans = {}
    for arm, (root, digest) in ARMS.items():
        comparison = verified(root / 'comparison.json', digest)
        if comparison['status'] != 'complete':
            raise ValueError(f'Expected completed endpoint: {arm}')
        plans[arm] = verified(root / 'plan.json', comparison['plan_sha256'])
    jobs = {}
    for protocol, source in plans['quality']['jobs'].items():
        peer = plans['matched_count']['jobs'][protocol]
        if any(source[key] != peer[key] for key in ('folds', 'manifest_sha256', 'backbone_sha256')):
            raise ValueError(f'Changed fixed cohort: {protocol}')
        arms = {}
        for arm, (root, _) in ARMS.items():
            checkpoint = root / protocol / 'increment' / 'checkpoint.pth'
            digest = file_sha256(checkpoint)
            reports = {}
            for fold in source['folds']:
                path = root / f'{protocol}-{fold}.json'
                if read_json(path)['checkpoint_sha256'] != digest:
                    raise ValueError(f'Changed endpoint: {arm}/{protocol}/{fold}')
                reports[fold] = dict(path=str(path), sha256=file_sha256(path))
            arms[arm] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, reports=reports)
        jobs[protocol] = dict(manifest_sha256=source['manifest_sha256'], folds=source['folds'], arms=arms)
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                jobs=jobs, batch_size=plans['quality']['batch_size'], paper_comparable=False,
                cohort='Small withheld targets with native selected or final coverage in either frozen arm. '
                       'Released small targets in those same replayed images are scored separately.')


@torch.inference_mode()
def trace(model, dataset, loader, known, cohort, reference, threshold):
    rows = []
    observed = set()
    for step, (samples, targets) in enumerate(loader, 1):
        wanted = []
        for batch, target in enumerate(targets):
            image_id = int(target['image_id'])
            anns = dataset.coco.imgToAnns[image_id]
            if any(ann['id'] in cohort for ann in anns):
                wanted.append((batch, target, anns))
        if wanted:
            outputs = model(samples.to('cuda'))
            geo = outputs['geometry']
            layers = [layer['pred_boxes'] for layer in outputs['aux_outputs']] + [outputs['pred_boxes']]
            if len(layers) != 6:
                raise ValueError('Expected six frozen decoder layers')
            for batch, target, anns in wanted:
                slots = torch.where(geo['active'][batch])[0]
                cells = geo['chosen'][batch, slots]
                height, width = target['orig_size'].tolist()
                seed = voc_coords(geo['boxes'][batch, cells], width, height)
                boxes = [voc_coords(stage[batch], width, height) for stage in layers]
                slot_ids, cell_ids = slots.tolist(), cells.tolist()
                for ann in anns:
                    released = ann['category_id'] in known
                    if not (ann['id'] in cohort or released and not ann.get('difficult')
                            and size_group(ann) != '32_sq_plus'):
                        continue
                    truth = np.asarray(ann['voc_bbox'])
                    seed_hits = np.flatnonzero(overlaps(truth, seed) > threshold)
                    layer_hits = [set(np.flatnonzero(overlaps(truth, stage) > threshold).tolist()) for stage in boxes]
                    ann_id = ann['id']
                    if not released:
                        observed.add(ann_id)
                        if bool(len(seed_hits)) != (ann_id in reference['selected']) or bool(layer_hits[-1]) != (ann_id in reference['final']):
                            raise ValueError(f'Native coverage differs at annotation {ann_id}')
                    rows.append(dict(annotation=ann_id, image=int(target['image_id']), released=released,
                                     seed_slots=[dict(query=slot_ids[i], cell=cell_ids[i],
                                                      covered=[slot_ids[i] in hits for hits in layer_hits])
                                                 for i in seed_hits],
                                     any_final=bool(layer_hits[-1])))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != cohort:
        raise ValueError('Frozen replay missed registered annotations')
    return rows


def summarize(rows):
    groups = {}
    for released, name in ((False, 'withheld'), (True, 'released')):
        counts = Counter()
        for row in rows:
            if row['released'] != released:
                continue
            seeds = row['seed_slots']
            counts.update(targets=1, selected=bool(seeds), final=row['any_final'],
                          final_within_selected=any(seed['covered'][-1] for seed in seeds),
                          final_only_other_slots=bool(seeds and row['any_final']
                                                      and not any(seed['covered'][-1] for seed in seeds)))
            if seeds:
                counts.update({f'decoder_{i+1}_within_selected': any(seed['covered'][i] for seed in seeds)
                               for i in range(6)})
        groups[name] = counts
    return groups


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, job['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        protocol_results = {}
        for fold, spec in job['folds'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                annotations = Path(spec['path'])
                payload = verified(annotations, spec['sha256'])
                small = {ann['id'] for ann in payload['annotations'] if size_group(ann) != '32_sq_plus'}
                archive = {arm: verified(Path(entry['reports'][fold]['path']), entry['reports'][fold]['sha256'])['report']
                           for arm, entry in job['arms'].items()}
                cohort = small & set().union(*(set(report['coverage'][key]) for report in archive.values()
                                               for key in ('selected', 'final')))
                if not cohort:
                    raise ValueError(f'Empty frozen trace cohort: {protocol}/{fold}')
                summaries, evidence = {}, {}
                for arm, entry in job['arms'].items():
                    model, _ = load_detector(None, Path(entry['checkpoint']), torch.device('cuda'))
                    model.eval()
                    dataset, loader = make_loader(root, annotations, plan['batch_size'], 2)
                    reference = {key: set(archive[arm]['coverage'][key]) for key in ('selected', 'final')}
                    print(f'Evaluating {protocol}/{fold}/{arm} selected query slots', flush=True)
                    rows = trace(model, dataset, loader, set(manifest['stages'][0]['classes']),
                                 cohort, reference, manifest['evaluation']['iou_threshold'])
                    evidence[arm], summaries[arm] = rows, summarize(rows)
                    del model
                write_json(dest, dict(plan_sha256=digest, targets=len(cohort), summaries=summaries, events=evidence))
            result = read_json(dest)
            if result['plan_sha256'] != digest:
                raise ValueError(f'Changed frozen replay plan: {dest}')
            protocol_results[fold] = result['summaries']
        results[protocol] = protocol_results
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results))


if __name__ == '__main__':
    main()
