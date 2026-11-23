"""Measure cross-category known-box suppression at the frozen matched-count endpoints."""

from pathlib import Path
import subprocess

import torch
from torchvision.ops import nms

from models.owod_benchmark_eval import evaluate_author_benchmark
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_aose import Attribution, METRICS
from tools.owod.probe_task1_m_s_refinement_top_known import TopKnownPerQuery
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, compare
from tools.owod.run_task1_m_s_refinement_saliency_training import MATCHED_COUNT_OUTPUT as SOURCE
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('task1-m-s-refinement-known-dedup')
SOURCE_SHA = '8d8b5817191b7b491f52ff247eb87dbeb41f310b7ceb59ad4ebb45f37d038181'
ENDPOINTS_SHA = '9b6b6b2f0099f36a191604a43322b57bf2f299b765a7c9ce646f186cbf4280ae'
OVERLAP = .7


class DeduplicateKnown(TopKnownPerQuery):
    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        known = scores[..., :len(self.known_ids)]
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        for geometry, items in zip(boxes, known, strict=True):
            chosen = nms(geometry.float(), items.amax(-1).float(), OVERLAP)
            keep = torch.zeros(len(geometry), dtype=torch.bool, device=geometry.device)
            keep[chosen] = True
            items.masked_fill_(~keep[:, None], -torch.inf)
        return scores, categories, unknown


def prepare():
    result = verified(SOURCE / 'comparison.json', SOURCE_SHA)
    source = verified(SOURCE / 'plan.json', result['plan_sha256'])
    endpoints = SOURCE / 'endpoint-sha256.txt'
    if result['status'] != 'complete' or result['passed'] or not source['match_count']:
        raise ValueError('Expected the completed matched-count screen')
    if file_sha256(endpoints) != ENDPOINTS_SHA:
        raise ValueError('Changed checkpoint digest list')
    hashes = {name: digest for digest, name in (line.split() for line in endpoints.read_text().splitlines())}
    jobs = {}
    for protocol, job in source['jobs'].items():
        relative = f'{protocol}/increment/checkpoint.pth'
        checkpoint = SOURCE / relative
        if file_sha256(checkpoint) != hashes[relative]:
            raise ValueError(f'Changed matched-count checkpoint: {protocol}')
        folds = {}
        for fold, entry in job['folds'].items():
            report = SOURCE / f'{protocol}-{fold}.json'
            baseline = read_json(report)
            if (baseline['plan_sha256'] != result['plan_sha256'] or baseline['checkpoint_sha256'] != hashes[relative]
                    or baseline['outcome'] != result['protocols'][protocol][fold]):
                raise ValueError(f'Changed matched-count fold: {protocol}/{fold}')
            folds[fold] = dict(entry, report=str(report), report_sha256=file_sha256(report))
        jobs[protocol] = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashes[relative], folds=folds,
                              manifest_sha256=job['manifest_sha256'], control_report=job['control_report'],
                              control_report_sha256=job['control_report_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_comparison_sha256=SOURCE_SHA, source_plan_sha256=result['plan_sha256'],
                endpoint_list_sha256=ENDPOINTS_SHA, batch_size=source['batch_size'], limit=source['limit'],
                gates=source['gates'], overlap=OVERLAP, paper_comparable=False, jobs=jobs,
                factor='Apply cross-category NMS to known outputs at IoU 0.7 before the common Top-50. '
                       'Unknown scores retain their original competition for the same output slots.')


def evaluate(model, manifest, root, annotations, plan, baseline, control):
    known = manifest['stages'][0]['classes']
    threshold = manifest['evaluation']['iou_threshold']
    settings = dict(known_ids=known, scoring='focal', max_detections=plan['limit'])
    post = DeduplicateKnown(**settings)
    native = TopKnownPerQuery(**settings)
    before, after = Attribution(known, threshold), Attribution(known, threshold)
    native_predictions, coco = {}, None

    def observe(dataset, _samples, targets, outputs, candidates, sizes, _post):
        nonlocal coco
        coco = dataset.coco
        originals = native(outputs, sizes)
        for target, original, candidate in zip(targets, originals, candidates, strict=True):
            image_id = int(target['image_id'])
            anns = coco.imgToAnns[image_id]
            before.add(image_id, original, anns)
            after.add(image_id, candidate, anns)
            native_predictions[image_id] = {key: original[key].cpu() for key in ('scores', 'boxes', 'labels')}

    candidate = measure(model, manifest, root, annotations, plan['batch_size'], plan['limit'],
                        observer=observe, postprocessor=post)
    spec = dict(manifest['evaluation'], image_source_ids=[row['owod_source_id'] for row in coco.dataset['images']])
    native_metrics = evaluate_author_benchmark(native_predictions, coco, known, [], known, spec)
    for key in METRICS:
        if native_metrics[key] != baseline['metrics'][key]:
            raise ValueError(f'Native Top-50 does not reconstruct matched-count baseline metric: {key}')
    first, second = before.result(), after.result()
    if first['a_ose'] != native_metrics['a_ose'] or second['a_ose'] != candidate['metrics']['a_ose']:
        raise ValueError('Detection trace does not reproduce author A-OSE')
    if candidate['coverage'] != baseline['coverage']:
        raise ValueError('Postprocessing changed frozen query geometry')
    summary = compare(control, candidate, plan['gates'])
    summary.update(native={key: native_metrics[key] for key in METRICS},
                   candidate={key: candidate['metrics'][key] for key in METRICS},
                   delta={key: candidate['metrics'][key] - native_metrics[key] for key in METRICS},
                   small_unknown=(baseline['groups']['small']['unknown_tp'], candidate['groups']['small']['unknown_tp']),
                   native_aose_trace=first, candidate_aose_trace=second)
    return summary


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        control = verified(Path(job['control_report']), job['control_report_sha256'])['folds']
        model, _ = load_detector(None, Path(job['checkpoint']), torch.device('cuda'))
        model.eval()
        results[protocol] = {}
        for fold, entry in job['folds'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                annotations = Path(entry['path'])
                if file_sha256(annotations) != entry['sha256']:
                    raise ValueError(f'Changed frozen validation annotations: {protocol}/{fold}')
                baseline = verified(Path(entry['report']), entry['report_sha256'])['report']
                print(f'Evaluating {protocol}/{fold} cross-category known deduplication', flush=True)
                result = evaluate(model, manifest, root, annotations, plan, baseline, control[fold])
                write_json(dest, dict(plan_sha256=digest, outcome=result))
            result = read_json(dest)
            if result['plan_sha256'] != digest:
                raise ValueError(f'Changed frozen replay plan: {dest}')
            results[protocol][fold] = result['outcome']
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
                                               passed=all(row['passed'] for folds in results.values() for row in folds.values())))


if __name__ == '__main__':
    main()
