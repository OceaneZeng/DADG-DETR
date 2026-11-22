"""Cross the frozen shared box heads of matched Task 2 M/S finetunes."""

from collections import defaultdict
from math import isclose
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_previous_hard_negative_misses import (
    BOX_COMPARISON_SHA, BOX_REPORT_SHAS, BOX_TRACE, checked_top_indices, voc_output_boxes,
)
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_previous_hard_negatives import BOX_OUTPUT, FULL, MANIFESTS, SOURCE, compare
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = BOX_TRACE.with_name('task2-m-s-current-low-iou-box-head')
TRACE_SHA = '51e8f5a4132a7eb828fa69c8cf97dc7e4a33d94e051f5bc375e306625663bcfc'
TRACE_REPORT_SHAS = {
    'm-owodb-first': 'b210dfcae96f5ead894d032e065c90fdddac4e0dd872ffe29f722ccecc4c0640',
    'm-owodb-second': 'a5bf17be374fdfdc01f848ce84c9f459874ba1fa4922706872d12aa2ee399760',
    's-owodb-first': 'a9213b4320e9af6896084d9de6a01a0fde13240c3c2769dd95ebec45c904ed34',
    's-owodb-second': 'cbe220b6f3f25a1e53876f148c6eda4cd7565df549fd64660d8090a49aaeace8',
}
ARMS = ('control', 'candidate')
DEVICE = torch.device('cuda')
COUNTS = ('target_count', 'candidate_recovered', 'control_lost', 'candidate_top50_recovered', 'control_top50_lost')


def load_arms(protocol, job, report):
    paths = dict(control=Path(job['endpoints']['finetune']['path']),
                 candidate=BOX_OUTPUT / protocol / 'finetune/checkpoint.pth')
    models, heads, states = {}, {}, {}
    for arm, path in paths.items():
        if file_sha256(path) != report[f'{arm}_sha256']:
            raise ValueError(f'Changed {protocol} checkpoint: {arm}')
        model, _ = load_detector(None, path, DEVICE)
        models[arm] = model.eval()
        detr = model.detr
        if detr.transformer.decoder.bbox_embed is not None or len({id(m) for m in detr.bbox_embed}) != 1:
            raise ValueError(f'Expected a shared readout outside decoder refinement: {protocol}/{arm}')
        heads[arm] = detr.bbox_embed[-1]
        states[arm] = {key: value.detach().clone() for key, value in heads[arm].state_dict().items()}
    if all(torch.equal(value, states['control'][key]) for key, value in states['candidate'].items()):
        raise ValueError(f'Box heads are identical: {protocol}')
    return models, heads, states


def batch_boxes(model, head, native, swapped, samples, sizes, post):
    base = model(samples)
    head.load_state_dict(swapped, strict=True)
    try:
        crossed = model(samples)
    finally:
        head.load_state_dict(native, strict=True)
    for key in ('pred_logits', 'pred_objectness', 'pred_novelty'):
        if not torch.equal(base[key], crossed[key]):
            raise ValueError(f'Box head swap changed {key}')
    return (post(base, sizes, include_top_indices=True),
            box_cxcywh_to_xyxy(base['pred_boxes']), box_cxcywh_to_xyxy(crossed['pred_boxes']))


@torch.inference_mode()
def replay(models, heads, states, dataset, loader, reference, wanted, job, limit, threshold):
    previous, current = job['previous'], job['current']
    known = previous + current
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=limit)
    wanted_by_image = defaultdict(list)
    for key, row in wanted.items():
        ann = dataset.coco.anns[key]
        if ann.get('difficult') or ann['category_id'] not in previous or ann['image_id'] != row['image']:
            raise ValueError(f'Changed pinned target: {key}')
        wanted_by_image[ann['image_id']].append(ann)
    rows = {}
    for step, (samples, targets) in enumerate(loader, 1):
        images = [int(target['image_id']) for target in targets]
        if any(image in wanted_by_image for image in images):
            samples = samples.to(DEVICE)
            sizes = torch.stack([target['orig_size'] for target in targets]).to(DEVICE)
            for arm, donor in (ARMS, ARMS[::-1]):
                selected, native, crossed = batch_boxes(
                    models[arm], heads[arm], states[arm], states[donor], samples, sizes, post)
                for i, image in enumerate(images):
                    if image not in wanted_by_image:
                        continue
                    checked_top_indices(selected[i], reference[arm], image, previous, current)
                    queries = (selected[i]['top_indices'] // (len(known) + 1)).tolist()
                    labels = selected[i]['labels'].tolist()
                    boxes = [voc_output_boxes(tensor[i], targets[i]) for tensor in (native, crossed)]
                    for ann in wanted_by_image[image]:
                        key = ann['id']
                        ious = [overlaps(np.asarray(ann['voc_bbox']), predictions) for predictions in boxes]
                        before, after = (float(values.max()) for values in ious)
                        baseline = wanted[key][f'{arm}_ious'][-1]
                        was_covered = arm == 'control'
                        if not isclose(before, baseline, abs_tol=1e-7) or (before > threshold) != was_covered:
                            raise ValueError(f'Changed frozen geometry: {arm}/{key}')
                        indices = [q for q, label in zip(queries, labels, strict=True) if label == ann['category_id']]
                        top50_before, top50_after = (bool((values[indices] > threshold).any()) for values in ious)
                        if top50_before != was_covered:
                            raise ValueError(f'Changed frozen Top-{limit} coverage: {arm}/{key}')
                        row = rows.setdefault(key, dict(annotation=key, image=image, category=ann['category_id']))
                        row[arm] = dict(native_max_iou=before, crossed_max_iou=after,
                                        crossed_covered=after > threshold, crossed_top50_covered=top50_after)
        if step % 128 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    if rows.keys() != wanted.keys():
        raise ValueError('Incomplete crossover replay')
    return list(rows.values())


def full_evaluation(protocol, context, plan, digest):
    job, manifest, image_root, report = context
    models, heads, states = load_arms(protocol, job, report)
    previous, current = job['previous'], job['current']
    results = {}
    for arm, donor in (ARMS, ARMS[::-1]):
        heads[arm].load_state_dict(states[donor], strict=True)
        results[arm] = {}
        for fold, cohort in job['folds'].items():
            marker = OUTPUT / f'{protocol}-{fold}-{arm}-ap.json'
            if not marker.exists():
                released = SOURCE / cohort['released']['file']
                scored = FULL / cohort['scored']['file']
                for annotation, item in ((released, cohort['released']), (scored, cohort['scored'])):
                    if file_sha256(annotation) != item['sha256']:
                        raise ValueError(f'Changed evaluation annotations: {annotation}')
                dataset, loader = make_loader(image_root, released, plan['eval_batch'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                print(f'Evaluating {protocol}/{fold} {arm} with {donor} head', flush=True)
                crossed = measure(models[arm], dataset, loader, previous + current, previous, current, spec,
                                  plan['max_detections'], evaluation_coco=COCO(str(scored)),
                                  include_perturbed=False, audit_previous_matches=True)
                native = report[arm][fold]
                if crossed['per_image'] != native['per_image']:
                    raise ValueError(f'Box swap changed the full-fold output composition: {protocol}/{fold}/{arm}')
                write_json(marker, dict(plan_sha256=digest, comparison=compare(native, crossed),
                                        metrics=crossed['views']['native']['metrics']))
            result = verified_result(marker, digest)
            results[arm][fold] = result['comparison']
    return results


def verified_result(marker, digest):
    result = read_json(marker)
    if result['plan_sha256'] != digest:
        raise ValueError(f'Changed crossover result: {marker}')
    return result


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    trace = verified(BOX_TRACE / 'comparison.json', TRACE_SHA)
    trace_plan = verified(BOX_TRACE / 'plan.json', trace['plan_sha256'])
    source = verified(BOX_OUTPUT / 'comparison.json', BOX_COMPARISON_SHA)
    parent = verified(BOX_OUTPUT / 'plan.json', source['plan_sha256'])
    if trace['status'] != 'complete' or trace['preferred'] != ['decoder_6'] or parent['max_detections'] != 50:
        raise ValueError('Expected the completed native final-layer geometry trace')
    count = sum(row['decoder_6'] for row in trace['pooled'].values())
    pinned = {}
    for label, sha in TRACE_REPORT_SHAS.items():
        report = verified(BOX_TRACE / f'{label}.json', sha)
        if report['plan_sha256'] != trace['plan_sha256']:
            raise ValueError(f'Changed trace plan for {label}')
        pinned[label] = {row['annotation']: row for row in report['rows'] if row['trace_outcome'] == 'decoder_6'}
    if count != 91 or sum(map(len, pinned.values())) != count:
        raise ValueError('Changed number of final-layer misses')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                trace_sha256=TRACE_SHA, trace_plan_sha256=trace['plan_sha256'], trace_report_shas=TRACE_REPORT_SHAS,
                source_plan_sha256=source['plan_sha256'], source_report_shas=BOX_REPORT_SHAS, targets=count,
                eval_batch=trace_plan['eval_batch'], max_detections=trace_plan['max_detections'],
                paper_comparable=False,
                decision='In each protocol at least half must recover in the candidate and lose coverage in the control.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results, contexts = {}, {}
    for protocol, job in parent['jobs'].items():
        report = verified(BOX_OUTPUT / f'{protocol}.json', BOX_REPORT_SHAS[protocol])
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        image_root = resolve_image_root(path, manifest)
        contexts[protocol] = (job, manifest, image_root, report)
        models, heads, states = load_arms(protocol, job, report)
        results[protocol] = {}
        for fold, cohort in job['folds'].items():
            label = f'{protocol}-{fold}'
            marker = OUTPUT / f'{label}.json'
            wanted = pinned[label]
            if len(wanted) != trace['protocols'][protocol][fold]['decoder_6']:
                raise ValueError(f'Changed pinned coverage count: {label}')
            if not marker.exists():
                released = SOURCE / cohort['released']['file']
                if file_sha256(released) != cohort['released']['sha256']:
                    raise ValueError(f'Changed input annotations: {label}')
                dataset, loader = make_loader(image_root, released, plan['eval_batch'], 2)
                print(f'Evaluating {protocol}/{fold}', flush=True)
                reference = {arm: report[arm][fold] for arm in ARMS}
                rows = replay(models, heads, states, dataset, loader, reference, wanted, job, plan['max_detections'],
                              manifest['evaluation']['iou_threshold'])
                counts = dict(target_count=len(rows),
                              candidate_recovered=sum(row['candidate']['crossed_covered'] for row in rows),
                              control_lost=sum(not row['control']['crossed_covered'] for row in rows),
                              candidate_top50_recovered=sum(row['candidate']['crossed_top50_covered'] for row in rows),
                              control_top50_lost=sum(not row['control']['crossed_top50_covered'] for row in rows))
                write_json(marker, dict(plan_sha256=digest, counts=counts, rows=rows))
            result = verified_result(marker, digest)
            if {row['annotation'] for row in result['rows']} != wanted.keys():
                raise ValueError(f'Changed crossover annotation IDs: {label}')
            results[protocol][fold] = result['counts']
        del models, heads, states
    pooled = {protocol: {key: sum(row[key] for row in folds.values()) for key in COUNTS}
              for protocol, folds in results.items()}
    passed = all(2 * min(row['candidate_recovered'], row['control_lost']) >= row['target_count']
                 for row in pooled.values())
    full_ap = {protocol: full_evaluation(protocol, ctx, plan, digest) for protocol, ctx in contexts.items()} if passed else {}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results, pooled=pooled,
                                                crossover_passed=passed, full_ap=full_ap))


if __name__ == '__main__':
    main()
