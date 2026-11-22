"""Compare class-only and native focal ranks on the same newly covered S Task 2 targets."""

from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_ffn_target_transfer import MANIFEST, OUTPUT as SOURCE
from tools.owod.probe_s_task2_tail_query_ranking import record_targets
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-instance-task2-ffn-score-components')
REPLAY_SHA256 = 'aa0f985f009e5f4cc7142e5b168f6f49406576e21fcf1e5287417292217fd71b'
CRITERION = (
    'On both slices, class-only within-class Top-5 coverage among newly geometrically covered '
    'native misses must improve by at least 10 percentage points over focal. Count gains and '
    'losses on paired annotation identities, and require at least two common classes to gain '
    'targets. No fitting, score rule adoption, or AP gain inference.'
)


def prepare():
    source = read_json(SOURCE / 'plan.json')
    comparison_file = SOURCE / 'comparison.json'
    result = read_json(comparison_file)
    source_plan_sha256 = file_sha256(SOURCE / 'plan.json')
    if (file_sha256(comparison_file) != REPLAY_SHA256 or result['status'] != 'complete'
            or not result['current_prioritize_selection'] or result['plan_sha256'] != source_plan_sha256
            or file_sha256(MANIFEST) != source['manifest_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != source['device'] or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpu'] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed evidence, benchmark or replay environment')
    for endpoint in source['checkpoints'].values():
        if file_sha256(Path(endpoint['path'])) != endpoint['sha256']:
            raise ValueError('Changed frozen checkpoint')
    annotations = {}
    for name, entry in source['slices'].items():
        path = Path(entry['annotation'])
        order = [image['owod_source_id'] for image in read_json(path)['images']]
        if (len(order) != 1024 or file_sha256(path) != entry['annotation_sha256']
                or sha256('\n'.join(order).encode()).hexdigest() != entry['order_sha256']):
            raise ValueError(f'Changed validation slice: {name}')
        reports = {}
        for arm, endpoint in source['checkpoints'].items():
            report = SOURCE / f'{name}-{arm}.json'
            payload = read_json(report)
            if (payload['plan_sha256'] != source_plan_sha256
                    or payload['checkpoint_sha256'] != endpoint['sha256']):
                raise ValueError(f'Changed target replay: {report}')
            reports[arm] = dict(path=str(report), sha256=file_sha256(report))
        annotations[name] = dict(annotation=str(path), order_sha256=entry['order_sha256'],
                                 annotation_sha256=entry['annotation_sha256'], reports=reports,
                                 expected=result['slices'][name]['current_gained_box_only'])
    plan = {key: source[key] for key in ('protocol', 'seed', 'paper_comparable', 'device', 'gpu',
                                       'batch_size', 'torch_version', 'manifest_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256=REPLAY_SHA256, candidate=source['checkpoints']['candidate'],
                slices=annotations, criterion=CRITERION)
    pin(OUTPUT / 'plan.json', plan)
    return plan


def select_targets(entry, classes, threshold):
    before = read_json(Path(entry['reports']['control']['path']))
    after = read_json(Path(entry['reports']['candidate']['path']))
    left = {row['annotation_id']: row for row in before['rows']}
    selected = {row['annotation_id']: row for row in after['rows']
                if (row['category'] in classes and not row['native_hit']
                    and left[row['annotation_id']]['best_iou'] <= threshold < row['best_iou'])}
    if len(selected) != entry['expected']:
        raise ValueError('The fixed target cohort disagrees with the paired replay')
    return selected, after['views']['released']['metrics']


def replay(name, entry, plan, model, manifest, classes, threshold, digest):
    path = OUTPUT / f'{name}.json'
    targets, metrics = select_targets(entry, classes, threshold)
    if not path.exists():
        ann = Path(entry['annotation'])
        dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
        order = [image['owod_source_id'] for image in read_json(ann)['images']]
        rows = []
        observer = partial(record_targets, rows, dataset.coco, set(classes), threshold, readout_ids=classes)
        print(f'Evaluating {name} on {len(order)} images', flush=True)
        views = evaluate_heldout(model, dataset, loader, manifest, order, proxies=False, rank_observer=observer)
        if views['released']['metrics'] != metrics:
            raise ValueError(f'Native VOC metrics changed: {name}')
        selected = [row for row in rows if row['annotation_id'] in targets]
        for row in selected:
            ref = targets[row['annotation_id']]
            if any(row[k] != ref[k] for k in ('category', 'image', 'best_iou', 'native_hit', 'within_class_rank')):
                raise ValueError(f'Native target replay changed: {row["annotation_id"]}')
        if {row['annotation_id'] for row in selected} != targets.keys() or len(selected) != len(targets):
            raise ValueError('Incomplete class-only replay')
        rows = [{k: row[k] for k in ('annotation_id', 'category', 'image', 'best_iou',
                                    'within_class_rank', 'class_only_rank')} for row in selected]
        write_json(path, dict(plan_sha256=digest, native_metrics=metrics, rows=rows))
    report = read_json(path)
    if report['plan_sha256'] != digest or report['native_metrics'] != metrics:
        raise ValueError(f'Changed replay report: {name}')
    return report['rows']


def summarize(rows):
    focal = sum(row['within_class_rank'] <= 5 for row in rows)
    class_only = sum(row['class_only_rank'] <= 5 for row in rows)
    improved = [row for row in rows if row['class_only_rank'] <= 5 < row['within_class_rank']]
    lost = sum(row['within_class_rank'] <= 5 < row['class_only_rank'] for row in rows)
    return dict(targets=len(rows), native_top5=focal, class_only_top5=class_only, gained=len(improved), lost=lost,
                improved_classes=sorted({row['category'] for row in improved}),
                margin_pass=10 * (class_only - focal) >= len(rows))


def main():
    plan = prepare()
    manifest = read_json(MANIFEST)
    classes = tuple(manifest['stages'][1]['classes'])
    threshold = manifest['evaluation']['iou_threshold']
    digest = file_sha256(OUTPUT / 'plan.json')
    model, _ = load_detector(None, Path(plan['candidate']['path']), torch.device('cuda'))
    result = {name: summarize(replay(name, entry, plan, model, manifest, classes, threshold, digest))
              for name, entry in plan['slices'].items()}
    common = set.intersection(*(set(row['improved_classes']) for row in result.values()))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, paper_comparable=False,
               slices=result, improved_in_both=sorted(common),
               supported=len(common) >= 2 and all(row['margin_pass'] for row in result.values())))
    print('Frozen score-component comparison saved', flush=True)


if __name__ == '__main__':
    main()
