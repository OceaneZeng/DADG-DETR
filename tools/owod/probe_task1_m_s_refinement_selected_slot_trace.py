"""Trace selected encoder proposal slots through the fixed Task 1 decoder endpoints."""

import argparse
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
ATTRIBUTION_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-final-slot-attribution')
TRACE_COMPARISON_SHA = '789ca52079fe7bd80a5590fb9c96933d34fc32ec30fbaefa7dc61e68b442152e'
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
def trace(model, dataset, loader, known, cohort, reference, threshold, attribute_final=False):
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
                seed_index = {query: i for i, query in enumerate(slot_ids)}
                for ann in anns:
                    released = ann['category_id'] in known
                    if not (ann['id'] in cohort or released and not ann.get('difficult')
                            and size_group(ann) != '32_sq_plus'):
                        continue
                    truth = np.asarray(ann['voc_bbox'])
                    seed_iou = overlaps(truth, seed)
                    seed_hits = np.flatnonzero(seed_iou > threshold)
                    layer_iou = [overlaps(truth, stage) for stage in boxes]
                    layer_hits = [set(np.flatnonzero(iou > threshold).tolist()) for iou in layer_iou]
                    ann_id = ann['id']
                    if not released:
                        observed.add(ann_id)
                        if bool(len(seed_hits)) != (ann_id in reference['selected']) or bool(layer_hits[-1]) != (ann_id in reference['final']):
                            raise ValueError(f'Native coverage differs at annotation {ann_id}')
                    row = dict(annotation=ann_id, image=int(target['image_id']), released=released,
                               seed_slots=[dict(query=slot_ids[i], cell=cell_ids[i],
                                                covered=[slot_ids[i] in hits for hits in layer_hits])
                                           for i in seed_hits],
                               any_final=bool(layer_hits[-1]))
                    if attribute_final:
                        final_hits = []
                        for query in sorted(layer_hits[-1]):
                            index = seed_index.get(query)
                            hit = dict(query=query, final_iou=float(layer_iou[-1][query]))
                            if index is None:
                                hit.update(source='learned', cell=None, seed_iou=None)
                            else:
                                iou = float(seed_iou[index])
                                source = 'injected_qualified' if iou > threshold else 'injected_unqualified'
                                hit.update(source=source, cell=cell_ids[index], seed_iou=iou)
                            final_hits.append(hit)
                        row['final_hits'] = final_hits
                    rows.append(row)
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


def pair_final_hits(evidence):
    def by_target(rows):
        return {(row['image'], row['annotation'], row['released']): row for row in rows}

    matched = by_target(evidence['matched_count'])
    quality = by_target(evidence['quality'])
    if (matched.keys() != quality.keys() or len(matched) != len(evidence['matched_count'])
            or len(quality) != len(evidence['quality'])):
        raise ValueError('Final slot attribution requires unique paired targets')
    summary, pairs = {}, []
    for key in sorted(matched):
        image, annotation, released = key
        before = sorted({hit['source'] for hit in matched[key]['final_hits']})
        after = sorted({hit['source'] for hit in quality[key]['final_hits']})
        if before:
            change = 'retained' if after else 'lost'
        else:
            change = 'gained' if after else 'uncovered'
        group = summary.setdefault('released' if released else 'withheld', {})
        transition = f'{"+".join(before) or "none"} -> {"+".join(after) or "none"}'
        group.setdefault(change, Counter())[transition] += 1
        pairs.append(dict(image=image, annotation=annotation, released=released, change=change,
                          matched_count_sources=before, quality_sources=after))
    return summary, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-slot-attribution', action='store_true')
    attribute_final = parser.parse_args().final_slot_attribution
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    output = ATTRIBUTION_OUTPUT if attribute_final else OUTPUT
    if attribute_final:
        parent = verified(OUTPUT / 'comparison.json', TRACE_COMPARISON_SHA)
        parent_plan = verified(OUTPUT / 'plan.json', parent['plan_sha256'])
        if any(plan[key] != parent_plan[key] for key in ('jobs', 'batch_size', 'cohort')):
            raise ValueError('Changed frozen selected-slot trace inputs')
        plan.update(trace_comparison_sha256=TRACE_COMPARISON_SHA,
                    attribution='Every strict final-layer VOC IoU hit, paired by image and annotation. '
                                'Retain all slot sources if more than one hits the same target.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, job['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        protocol_results = {}
        for fold, spec in job['folds'].items():
            dest = output / f'{protocol}-{fold}.json'
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
                    print(f'Evaluating {protocol}/{fold}/{arm} query slots', flush=True)
                    rows = trace(model, dataset, loader, set(manifest['stages'][0]['classes']),
                                 cohort, reference, manifest['evaluation']['iou_threshold'], attribute_final)
                    evidence[arm], summaries[arm] = rows, summarize(rows)
                    del model
                result = dict(plan_sha256=digest, targets=len(cohort), summaries=summaries, events=evidence)
                if attribute_final:
                    if summaries != parent['protocols'][protocol][fold]:
                        raise ValueError(f'Selected-slot coverage differs on {protocol}/{fold}')
                    result['attribution'], result['pairs'] = pair_final_hits(evidence)
                write_json(dest, result)
            result = read_json(dest)
            if result['plan_sha256'] != digest:
                raise ValueError(f'Changed frozen replay plan: {dest}')
            protocol_results[fold] = result['summaries']
            if attribute_final:
                protocol_results[fold] = dict(arms=result['summaries'], attribution=result['attribution'])
        results[protocol] = protocol_results
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results))


if __name__ == '__main__':
    main()
