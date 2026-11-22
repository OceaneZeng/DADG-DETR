"""Check confidence-order sensitivity on the four independent refill folds."""

from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_current_top_class import UnknownRefillScore
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS
from tools.owod.probe_task2_m_s_unknown_refill_independent import OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-refill-score-order')
PARENT_RESULT = '8e51e2e4b4cdedf5c32a273fba893e3c9a6d17b5638aa27c77e2539424b9a5d3'
ORDERS = ('stable_quantized', 'stable_raw')
KEYS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')


def summarize(report, reference):
    if any(report[key] != reference[key] for key in ('views', 'per_image', 'queries_per_image')):
        raise ValueError('Changed author-order replay')
    author = {name: view['metrics'] for name, view in report['views'].items()}
    rows = {}
    for order, arms in dict(author=author, **report['score_order_views']).items():
        native, refill = (arms[k] for k in ('native', 'unknown_refill'))
        rows[order] = dict(
            deltas={key: refill[key] - native[key] for key in KEYS},
            zebra={name: arm['current_class_stats'].get('24') for name, arm in arms.items()},
        )
    return rows


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    result = verified(SOURCE / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete':
        raise ValueError('Missing completed independent refill')
    parent = verified(SOURCE / 'plan.json', result['plan_sha256'])
    cohort_dir = SOURCE / 'cohorts'
    cohorts = verified(cohort_dir / 'plan.json', parent['cohort_plan'])
    references = {name: file_sha256(SOURCE / f'{name}.json') for name in parent['reports']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan=result['plan_sha256'], parent_result=PARENT_RESULT,
                cohort_plan=parent['cohort_plan'], references=references, orders=ORDERS,
                minimum_recovery_fraction=.5, paper_comparable=False)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    reports = {}
    for protocol, info in cohorts['protocols'].items():
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        checkpoint = Path(info['endpoints']['finetune']['path'])
        if file_sha256(checkpoint) != info['endpoints']['finetune']['sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        known = info['previous'] + info['current']
        post = UnknownRefillScore(known_ids=known, max_detections=parent['limit'], scoring='focal')
        reports[protocol] = {}
        for fold, cohort in cohorts['cohorts'].items():
            label = f'{protocol}-{fold}'
            reference = verified(SOURCE / f'{label}.json', references[label])
            if reference['plan_sha256'] != result['plan_sha256']:
                raise ValueError(f'Changed independent source: {label}')
            annotation = cohort['by_protocol'][protocol]
            path = cohort_dir / annotation['file']
            verified(path, annotation['sha256'])
            full = parent['reports'][label]
            full_path = SOURCE / full['file']
            verified(full_path, full['sha256'])
            dest = OUTPUT / f'{label}.json'
            if not dest.exists():
                print(f'Evaluating {protocol}/{fold} with three score orders', flush=True)
                root = resolve_image_root(MANIFESTS[protocol], manifest)
                dataset, loader = make_loader(root, path, parent['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                replay = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 parent['limit'], evaluation_coco=COCO(str(full_path)),
                                 extra_posts=dict(unknown_refill=post), audit_query_counts=True,
                                 include_perturbed=False, diagnostic_score_orders=ORDERS)
                write_json(dest, dict(plan_sha256=digest, **replay))
            replay = read_json(dest)
            if replay['plan_sha256'] != digest:
                raise ValueError(f'Changed score-order replay: {label}')
            reports[protocol][fold] = summarize(replay, reference)
        del model
    focus = reports['m-owodb']['second']
    author = focus['author']['deltas']['current_ap50']
    sensitive = author < 0 and all(
        focus[mode]['deltas']['current_ap50'] - author >= -plan['minimum_recovery_fraction'] * author
        for mode in ORDERS)
    write_json(OUTPUT / 'comparison.json', dict(
        status='complete', plan_sha256=digest, protocols=reports, ranking_sensitive=sensitive,
        paper_comparable=False))


if __name__ == '__main__':
    main()
