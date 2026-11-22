"""Test increasing only the leading Current score in each query."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_fp_open_set import OPEN_KEYS, OPEN_RESULT, OUTPUT as FULL
from tools.owod.probe_task2_m_s_current_score import AP_TOLERANCE, MULTIPLIER, measure, summarize
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-current-top-class')
NEAR = SOURCE.with_name('task2-m-s-current-fp-near-target')
NEAR_RESULT = '8a841a56fb07c60072bbf5c5efe277ee35f05a0bfa3d8767325d327b39e1b72e'
TOP_RESULT = '62428d324a76a3a711a5efdb601a8d8185167738107f7b3176138505b41634eb'
ORDER_RESULT = 'f9eb597a03348486df77666d7e8dd7ceb9272feeabfa447d52b552708c28d42d'


class LeadingCurrentScore(DiscoveryPostProcess):
    def __init__(self, known, current, limit):
        super().__init__(known_ids=known, max_detections=limit, scoring='focal')
        self.current = tuple(current)

    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        leading = scores[..., :len(self.known_ids)].argmax(dim=-1, keepdim=True)
        eligible = torch.isin(categories[leading], categories.new_tensor(self.current))
        original = scores.gather(-1, leading)
        scores.scatter_(-1, leading, torch.where(eligible, original * MULTIPLIER, original))
        return scores, categories, unknown


class UniqueQueryScore(DiscoveryPostProcess):
    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        winning_scores, winners = scores.max(dim=-1, keepdim=True)
        unique = torch.full_like(scores, -torch.inf).scatter_(-1, winners, winning_scores)
        return unique, categories, unknown


def summarize_views(report, prior, shared):
    for arm in ('native', 'perturbed'):
        baseline = prior['views'][arm]
        view = report['views'][arm]
        stats = {label: {k: v for k, v in row.items() if k != 'fp_sources'}
                 for label, row in baseline['metrics']['current_class_stats'].items()}
        if (view['metrics'] != dict(baseline['metrics'], current_class_stats=stats)
                or view['matched'] != baseline['matched']
                or any(views[arm] != prior['per_image'][image][arm] for image, views in report['per_image'].items())):
            raise ValueError(f'Changed reference Top-50 or VOC metrics: {arm}')
    metrics = {name: value['metrics'] for name, value in report['views'].items()}
    fp = {name: sum(x['fp'] for x in row['current_class_stats'].values()) for name, row in metrics.items()}
    native = report['views']['native']
    comparisons = {}
    for name in report['views']:
        if name == 'native':
            continue
        pair = dict(views=dict(native=native, perturbed=report['views'][name]),
                    per_image={key: dict(native=value['native'], perturbed=value[name])
                               for key, value in report['per_image'].items()})
        comparisons[name] = summarize(pair, shared)
    selected = comparisons['selective']
    return dict(arms=comparisons, current_fp=fp,
                open_metrics={name: {k: row[k] for k in OPEN_KEYS} for name, row in metrics.items()},
                passed=selected['passed'] and fp['selective'] < fp['perturbed'])


def main(native_order=False, unique_query=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    previous = verified(FULL / 'comparison.json', OPEN_RESULT)
    near = verified(NEAR / 'comparison.json', NEAR_RESULT)
    if previous['status'] != 'complete' or near['status'] != 'complete':
        raise ValueError('Missing completed full-annotation or geometry study')
    base = verified(FULL / 'plan.json', previous['plan_sha256'])
    source = verified(SOURCE / 'plan.json', base['parent_plan'])
    reports = {name: dict(info, reference_sha256=file_sha256(FULL / f'{name}.json'))
               for name, info in base['reports'].items()}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan=previous['plan_sha256'], parent_result=OPEN_RESULT, near_result=NEAR_RESULT,
                batch_size=base['batch_size'], limit=base['max_detections'], multiplier=MULTIPLIER,
                ap_tolerance=AP_TOLERANCE, reports=reports, paper_comparable=False)
    output = OUTPUT
    if native_order:
        top = verified(OUTPUT / 'comparison.json', TOP_RESULT)
        if top['status'] != 'complete' or top['plan_sha256'] != file_sha256(OUTPUT / 'plan.json'):
            raise ValueError('Changed selective-score comparison')
        output = OUTPUT.with_name('task2-m-s-current-native-order')
        for name, info in reports.items():
            info['selective_sha256'] = file_sha256(OUTPUT / f'{name}.json')
        plan.update(selective_plan=top['plan_sha256'], selective_result=TOP_RESULT, minimum_recovery_fraction=.5)
    if unique_query:
        order_path = OUTPUT.with_name('task2-m-s-current-native-order')
        order = verified(order_path / 'comparison.json', ORDER_RESULT)
        if order['status'] != 'complete' or order['plan_sha256'] != file_sha256(order_path / 'plan.json'):
            raise ValueError('Changed native-order comparison')
        output = OUTPUT.with_name('task2-m-s-unique-query')
        plan.update(native_order_plan=order['plan_sha256'], native_order_result=ORDER_RESULT,
                    maximum_outputs_per_query=1)
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    results = {}
    for protocol, entry in source['protocols'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, entry['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        checkpoint = Path(entry['endpoints']['finetune']['path'])
        if file_sha256(checkpoint) != entry['endpoints']['finetune']['sha256']:
            raise ValueError(f'Changed frozen checkpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        known = entry['previous'] + entry['current']
        if set(model.known_ids) != set(known):
            raise ValueError(f'Changed active classes: {protocol}')
        results[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            info = reports[label]
            full = FULL / info['file']
            verified(full, info['sha256'])
            annotation = cohort['by_protocol'][protocol]
            released = SOURCE / annotation['file']
            verified(released, annotation['sha256'])
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            path = output / f'{label}.json'
            if not path.exists():
                print(f'Evaluating {protocol}/{fold} at the fixed query budget', flush=True)
                dataset, loader = make_loader(root, released, plan['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                post = LeadingCurrentScore(known, entry['current'], plan['limit'])
                extra = dict(selective=post)
                if unique_query:
                    extra['unique'] = UniqueQueryScore(known_ids=known, max_detections=plan['limit'], scoring='focal')
                report = measure(model, dataset, loader, known, entry['previous'], entry['current'], spec,
                                 plan['limit'], evaluation_coco=COCO(str(full)), extra_posts=extra,
                                 restore_scores_for='selective' if native_order else None,
                                 audit_query_counts=unique_query)
                write_json(path, dict(plan_sha256=digest, **report))
            report = read_json(path)
            prior = verified(FULL / path.name, info['reference_sha256'])
            if report['plan_sha256'] != digest or report['per_image'].keys() != prior['per_image'].keys():
                raise ValueError(f'Changed replay plan or image set: {label}')
            shared = {pair['annotation_ids'][protocol] for pair in pairs}
            summary = summarize_views(report, prior, shared)
            if native_order:
                original = verified(OUTPUT / path.name, info['selective_sha256'])
                if (original['plan_sha256'] != plan['selective_plan']
                        or any(report['views'][arm] != view for arm, view in original['views'].items())
                        or any(any(arms[arm] != count for arm, count in original['per_image'][img].items())
                               or arms['restored'] != arms['selective']
                               for img, arms in report['per_image'].items())):
                    raise ValueError(f'Changed selective score replay: {label}')
                selective = summary['arms']['selective']['deltas']['current_ap50']
                restored = summary['arms']['restored']['deltas']['current_ap50']
                summary['ranking'] = dict(current_ap50_restored_minus_selective=restored-selective,
                                          recovered_half=selective < 0 and restored-selective >=
                                          -plan['minimum_recovery_fraction']*selective)
                summary['passed'] = (summary['arms']['restored']['passed']
                                     and summary['current_fp']['restored'] < summary['current_fp']['perturbed'])
            if unique_query:
                counts = report['queries_per_image']
                if counts.keys() != report['per_image'].keys() or any(
                        row['unique'] != plan['limit'] or not 0 < row['native'] <= plan['limit']
                        for row in counts.values()):
                    raise ValueError(f'Changed distinct-query accounting: {label}')
                summary['query_budget'] = dict(
                    native_repeated_slots=sum(plan['limit'] - row['native'] for row in counts.values()),
                    native_images_with_repeats=sum(row['native'] < plan['limit'] for row in counts.values()),
                    unique_repeated_slots=0)
                open_metrics = summary['open_metrics']
                summary['passed'] = (summary['arms']['unique']['passed'] and
                                     open_metrics['unique']['u_recall'] >= open_metrics['native']['u_recall'])
            results[protocol][fold] = summary
        del model
    accepted = all(r['passed'] for folds in results.values() for r in folds.values())
    result = dict(status='complete', plan_sha256=digest, protocols=results, accepted=accepted, paper_comparable=False)
    if native_order:
        result['m_ranking_consistent'] = all(fold['ranking']['recovered_half'] for fold in results['m-owodb'].values())
    write_json(output / 'comparison.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--native-order', action='store_true')
    mode.add_argument('--unique-query', action='store_true')
    args = parser.parse_args()
    main(args.native_order, args.unique_query)
