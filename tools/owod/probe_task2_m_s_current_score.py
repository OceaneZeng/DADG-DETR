"""Stress-test Task 2 Current scores at the native focal Top-50 budget."""

import argparse
from pathlib import Path
from statistics import mean
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
CURRENT_RESULT = '4c5f5e4b320aebad7ba3f152b125b0548bb964073732a95ba1d298087d454d52'
FAMILIES = ('duplicate', 'other_released_class', 'partial_same_class', 'unresolved')
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
def measure(model, dataset, loader, known, previous, current, spec, limit, *, evaluation_coco=None,
            extra_posts=None, restore_scores_for=None, audit_query_counts=False):
    posts = dict(native=DiscoveryPostProcess(known_ids=known, max_detections=limit, scoring='focal'),
                 perturbed=CurrentScore(known, set(current), limit, MULTIPLIER))
    if extra_posts is not None:
        posts.update(extra_posts)
    predictions = {name: {} for name in posts}
    if restore_scores_for is not None:
        predictions['restored'] = {}
    per_image = {}
    queries_per_image = {}
    for step, (samples, targets) in enumerate(loader, 1):
        output = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        if restore_scores_for is not None:
            native_scores, _, _ = posts['native'].candidate_scores(output)
            native_scores = native_scores.flatten(1)
        for name, post in posts.items():
            restore = name == restore_scores_for
            audit = audit_query_counts and name in ('native', 'unique')
            selected = post(output, sizes, include_top_indices=restore or audit)
            for index, (target, pred) in enumerate(zip(targets, selected, strict=True)):
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
                if audit:
                    categories = len(known) + int(len(known) < 80)
                    queries = pred['top_indices'] // categories
                    count = queries.unique().numel()
                    if name == 'unique' and count != limit:
                        raise ValueError(f'Repeated query in unique Top-50: {image}')
                    queries_per_image.setdefault(image, {})[name] = count
                if restore:
                    scores = native_scores[index].gather(0, pred['top_indices']).cpu()
                    predictions['restored'][image] = dict(predictions[name][image], scores=scores)
                    per_image[image]['restored'] = counts.copy()
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    views = {}
    coco = dataset.coco if evaluation_coco is None else evaluation_coco
    for name, values in predictions.items():
        metrics = evaluate_author_benchmark(values, coco, known, previous, current, spec,
                                            include_current_stats=True, include_matched_ids=True)
        by_class = metrics.pop('matched_annotation_ids')
        matched = {i for category in current for i in by_class[category]}
        if len(matched) != sum(c['tp'] for c in metrics['current_class_stats'].values()):
            raise ValueError(f'Changed Current VOC matching: {name}')
        views[name] = dict(metrics=metrics, matched=sorted(matched))
    report = dict(views=views, per_image=per_image)
    if audit_query_counts:
        report['queries_per_image'] = queries_per_image
    return report


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


def summarize_false_positives(report, prior):
    views = {name: value['metrics']['current_class_stats'] for name, value in report['views'].items()}
    for name, view in report['views'].items():
        reference = prior['views'][name]
        stats = {label: {k: v for k, v in row.items() if k != 'fp_sources'}
                 for label, row in views[name].items()}
        if (dict(view['metrics'], current_class_stats=stats) != reference['metrics']
                or view['matched'] != reference['matched'] or report['per_image'] != prior['per_image']):
            raise ValueError(f'Changed native or perturbed output: {name}')
        if any(sum(entry['fp'] for entry in row['fp_sources']['families'].values()) != row['fp']
               for row in views[name].values()):
            raise ValueError(f'Incomplete false-positive attribution: {name}')
    results = {}
    for name in FAMILIES:
        counts, gains = {}, {}
        for view, stats in views.items():
            counts[view] = sum(row['fp_sources']['families'][name]['fp'] for row in stats.values())
            gains[view] = mean(row['fp_sources']['families'][name]['ap50_without'] - row['ap50']
                               for row in stats.values())
        results[name] = dict(counts=counts, delta=counts['perturbed'] - counts['native'],
                             ap50_gain_without_family=gains)
    additional = sum(row['delta'] for row in results.values())
    priority = next((name for name, row in results.items()
                     if 2 * row['delta'] > additional and row['ap50_gain_without_family']['perturbed'] >= .005), None)
    return dict(families=results, additional_false_positives=additional, priority=priority)


def main(fp_sources=False):
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
    output = OUTPUT
    prior_reports = {}
    if fp_sources:
        completed = verified(OUTPUT / 'comparison.json', CURRENT_RESULT)
        if completed['status'] != 'complete' or completed['plan_sha256'] != file_sha256(OUTPUT / 'plan.json'):
            raise ValueError('Changed score perturbation evidence')
        for protocol in source['protocols']:
            for fold in source['cohorts']:
                name = f'{protocol}-{fold}.json'
                prior_reports[name] = file_sha256(OUTPUT / name)
        output = OUTPUT.with_name('task2-m-s-current-fp')
        plan.update(current_score_result=CURRENT_RESULT, prior_reports=prior_reports, fp_families=FAMILIES,
                    minimum_ap50_gain=.005)
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
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
            dest = output / f'{protocol}-{fold}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} at a shared Top-50 budget', flush=True)
                dataset, loader = make_loader(root, path, plan['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                if fp_sources:
                    spec['diagnostic_current_fp_sources'] = True
                result = measure(model, dataset, loader, known, previous, current, spec, plan['max_detections'])
                write_json(dest, dict(plan_sha256=digest, **result))
            report = read_json(dest)
            baseline = reference['protocols'][protocol]['finetune'][fold]
            if report['plan_sha256'] != digest or len(report['per_image']) != len(cohort['order']):
                raise ValueError(f'Changed image set or plan: {protocol}/{fold}')
            if fp_sources:
                prior = verified(OUTPUT / dest.name, prior_reports[dest.name])
                protocols[protocol][fold] = summarize_false_positives(report, prior)
                continue
            if report['views']['native']['metrics'] != baseline['metrics']:
                raise ValueError(f'Changed native benchmark: {protocol}/{fold}')
            shared_ids = {p['annotation_ids'][protocol] for p in pairs}
            summary = summarize(report, shared_ids)
            if summary['shared']['native'] != baseline['shared_counts']['matched']:
                raise ValueError(f'Changed native shared-class outcomes: {protocol}/{fold}')
            protocols[protocol][fold] = summary
        del model
    folds = [row for protocol in protocols.values() for row in protocol.values()]
    completion = dict(status='complete', plan_sha256=digest, protocols=protocols, paper_comparable=False)
    if fp_sources:
        completion['shared_priority'] = next((name for name in FAMILIES if all(
            row['priority'] == name for row in folds)), None)
    else:
        completion['accepted'] = all(row['passed'] for row in folds)
    write_json(output / 'comparison.json', completion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp-sources', action='store_true')
    main(parser.parse_args().fp_sources)
