"""Stress-test Task 2 Current scores at the native focal Top-50 budget."""

from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import evaluate_author_benchmark
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_score_margin import OUTPUT as MARGINS, PARENT_PLAN, PARENT_RESULT
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-current-score')
MARGIN_RESULT = 'f7db72ce7f691be1999440c4115b943b2313edb035a236b288fb2bbf1bc25eec'
MULTIPLIER = 1.15
AP_TOLERANCE = .005


class CurrentScore(DiscoveryPostProcess):
    def __init__(self, known, current, max_detections, multiplier):
        super().__init__(known_ids=known, max_detections=max_detections, scoring='focal')
        self.current_positions = [i for i, category in enumerate(known) if category in current]
        self.multiplier = multiplier

    def candidate_scores(self, outputs):
        probabilities, categories, unknown = super().candidate_scores(outputs)
        probabilities[..., self.current_positions] *= self.multiplier
        return probabilities, categories, unknown


@torch.inference_mode()
def measure(model, dataset, loader, known, previous, current, spec, limit):
    posts = dict(native=DiscoveryPostProcess(known_ids=known, max_detections=limit, scoring='focal'),
                 perturbed=CurrentScore(known, set(current), limit, MULTIPLIER))
    predictions = {name: {} for name in posts}
    per_image = {}
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        for name, post in posts.items():
            for target, pred in zip(targets, post(output, sizes), strict=True):
                image = int(target['image_id'])
                if image in predictions[name] or len(pred['labels']) != limit:
                    raise ValueError(f'Invalid per-image output budget: {image}/{name}')
                predictions[name][image] = {k: pred[k].cpu() for k in ('scores', 'boxes', 'labels')}
                labels = pred['labels'].tolist()
                counts = dict(previous=sum(c in previous for c in labels), current=sum(c in current for c in labels),
                              unknown=labels.count(91))
                if sum(counts.values()) != limit:
                    raise ValueError(f'Unexpected predicted category: {image}/{name}')
                per_image.setdefault(image, {})[name] = counts
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    views = {}
    for name, values in predictions.items():
        metrics = evaluate_author_benchmark(values, dataset.coco, known, previous, current, spec,
                                            include_current_stats=True, include_matched_ids=True)
        by_class = metrics.pop('matched_annotation_ids')
        matched = {i for category in current for i in by_class[category]}
        if len(matched) != sum(c['tp'] for c in metrics['current_class_stats'].values()):
            raise ValueError(f'Changed Current VOC matching: {name}')
        views[name] = dict(metrics=metrics, matched=sorted(matched))
    return dict(views=views, per_image=per_image)


def summarize(report, shared_ids):
    native, perturbed = (report['views'][key] for key in ('native', 'perturbed'))
    before, after = (set(view['matched']) & shared_ids for view in (native, perturbed))
    delta = {key: perturbed['metrics'][key] - native['metrics'][key]
             for key in ('previous_ap50', 'current_ap50', 'known_ap50')}
    passed = (delta['current_ap50'] >= AP_TOLERANCE and delta['previous_ap50'] >= -AP_TOLERANCE
              and delta['known_ap50'] >= 0.)
    totals = {view: {role: sum(image[view][role] for image in report['per_image'].values())
                     for role in ('previous', 'current', 'unknown')} for view in report['views']}
    return dict(deltas=delta, shared=dict(native=len(before), perturbed=len(after),
                                          gained=len(after-before), lost=len(before-after)),
                outputs=totals, passed=passed)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    source = verified(SOURCE / 'plan.json', PARENT_PLAN)
    reference = verified(SOURCE / 'comparison.json', PARENT_RESULT)
    margins = verified(MARGINS / 'comparison.json', MARGIN_RESULT)
    if any(v['status'] != 'complete' for v in (reference, margins)):
        raise ValueError('Missing completed source reports')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan=PARENT_PLAN, parent_result=PARENT_RESULT, margin_result=MARGIN_RESULT,
                paper_comparable=False, multiplier=MULTIPLIER, max_detections=source['max_detections'],
                ap_tolerance=AP_TOLERANCE, batch_size=source['batch_size'])
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, info in source['protocols'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, info['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        previous, current = info['previous'], info['current']
        known = previous + current
        checkpoint = Path(info['endpoints']['finetune']['path'])
        if file_sha256(checkpoint) != info['endpoints']['finetune']['sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        if set(model.known_ids) != set(known):
            raise ValueError(f'Changed active classes: {protocol}')
        protocols[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            annotation = cohort['by_protocol'][protocol]
            path = SOURCE / annotation['file']
            verified(path, annotation['sha256'])
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at a shared Top-50 budget', flush=True)
                dataset, loader = make_loader(root, path, plan['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                result = measure(model, dataset, loader, known, previous, current, spec, plan['max_detections'])
                write_json(dest, dict(plan_sha256=digest, **result))
            report = read_json(dest)
            baseline = reference['protocols'][protocol]['finetune'][fold]
            if (report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order'])
                    or report['views']['native']['metrics'] != baseline['metrics']):
                raise ValueError(f'Changed native benchmark: {protocol}/{fold}')
            shared_ids = {p['annotation_ids'][protocol] for p in pairs}
            summary = summarize(report, shared_ids)
            if summary['shared']['native'] != baseline['shared_counts']['matched']:
                raise ValueError(f'Changed native shared-class outcomes: {protocol}/{fold}')
            protocols[protocol][fold] = summary
        del model
    accepted = all(row['passed'] for folds in protocols.values() for row in folds.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=protocols, accepted=accepted, paper_comparable=False))


if __name__ == '__main__':
    main()
