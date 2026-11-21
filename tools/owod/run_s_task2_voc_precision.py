"""Decompose the S Task 2 old-class VOC AP gap at a fixed detection budget."""
from hashlib import sha256
from math import isclose
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_paired_targets import OUTPUT as PAIRED
from tools.owod.run_s_task2_score_competition import ARMS, MANIFEST, OUTPUT as SOURCE
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-instance-task2-voc-precision')
VIEW = 'known_class_only'
RANKING_GATE_PP = .75


def summarize(stats):
    ap = mean(row['ap50'] for row in stats.values())
    ceiling = mean(row['recall_ceiling_ap50'] for row in stats.values())
    counts = {key: sum(row[key] for row in stats.values()) for key in ('gt', 'tp', 'fp', 'fp_duplicates')}
    return dict(previous_ap50=ap, recall_ceiling_ap50=ceiling, ranking_penalty=ceiling-ap, **counts)


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for frozen inference')
    scoring = read_json(SOURCE / 'comparison.json')
    source = read_json(SOURCE / 'plan.json')
    paired = read_json(PAIRED / 'comparison.json')
    annotation, manifest = Path(source['annotation']), read_json(MANIFEST)
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if (scoring['status'] != 'complete' or scoring['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or
            paired['status'] != 'complete' or paired['next_focus'] != 'inspect_precision' or
            paired['plan_sha256'] != file_sha256(PAIRED / 'plan.json') or
            read_json(PAIRED / 'plan.json')['source_result_sha256'] != file_sha256(SOURCE / 'comparison.json') or
            file_sha256(annotation) != source['annotation_sha256'] or
            sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256'] or
            file_sha256(MANIFEST) != source['manifest_sha256'] or len(order) != 1024 or len(set(order)) != len(order)):
        raise ValueError('Changed S scoring or paired-target inputs')
    checkpoints = source['checkpoints']
    for entry in checkpoints.values():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Changed frozen checkpoint: {entry["path"]}')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', paper_comparable=False, seed=source['seed'], images=1024, device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__), batch_size=2, max_detections=50,
                annotation=str(annotation), annotation_sha256=source['annotation_sha256'], eval_order_sha256=source['eval_order_sha256'],
                source_scoring_sha256=file_sha256(SOURCE / 'comparison.json'), paired_sha256=file_sha256(PAIRED / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], checkpoints=checkpoints,
                criterion='Prioritize false-positive ranking if mean Previous ranking penalty grows by at least 0.75 pp.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    plan_hash = file_sha256(plan_path)
    previous = manifest['stages'][0]['active_classes']
    stage = manifest['stages'][1]
    variants = {VIEW: (stage['active_classes'], 'focal_class_only', 50)}
    spec = dict(manifest['evaluation'], image_source_ids=order, known_only_screening=True)
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    arms, per_class = {}, {}
    for arm in ARMS:
        entry = checkpoints[arm]
        path = OUTPUT / f'{arm}.json'
        if not path.is_file():
            print(f'Evaluating precision for {arm}', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            evaluation = evaluate_views(model, loader, dataset.coco, variants, spec, previous, stage['classes'], 'cuda',
                                        exclude_unknown=(VIEW,), include_previous_stats=True, print_freq=64)[VIEW]
            write_json(path, dict(checkpoint_sha256=entry['sha256'], plan_sha256=plan_hash, evaluation=evaluation))
            del model
        report = read_json(path)
        view = report['evaluation']
        stats = view['previous_class_stats']
        if (report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != entry['sha256'] or
                view['metrics'] != scoring['arms'][arm][VIEW] or {int(label) for label in stats} != set(previous)):
            raise ValueError(f'Precision report does not reproduce the archived scoring view: {path}')
        arms[arm], per_class[arm] = summarize(stats), stats
        if not isclose(arms[arm]['previous_ap50'], view['metrics']['previous_ap50'], abs_tol=1e-12):
            raise ValueError(f'Per-class VOC mean differs from the endpoint: {path}')
    short, long = ARMS
    deltas = {key: 100*(arms[long][key]-arms[short][key]) for key in
              ('previous_ap50', 'recall_ceiling_ap50', 'ranking_penalty')}
    for row in arms.values():
        row['fp_nonduplicate'] = row['fp'] - row['fp_duplicates']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
               arms=arms, deltas_pp=deltas, per_class=per_class, rank_follow_up=deltas['ranking_penalty'] >= RANKING_GATE_PP))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
