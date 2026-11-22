"""Separate geometry changes from native Top-50 changes for paired S Task 2 targets."""

from collections import Counter
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_tail_query_ranking import record_targets
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import write_json


PARENT = DURATION.with_name('s-instance-task2-increment-encoder-ffn')
OUTPUT = DURATION.with_name('s-instance-task2-ffn-target-transfer')
SOURCE_SHA256 = 'adcbf7210c91029e3c782192ffda7cc2794bbdfe33de8c37b8eaad4fac88f9b5'
CRITERION = (
    'Prioritize native selection for newly covered Current targets only if strictly more than half '
    'of the newly geometrically covered targets lack a correct-class Top-50 match on both slices. '
    'Prioritize geometry for lost Previous targets only if strictly more than half of the old-class '
    'Top-50 losses also lose all-query geometry on both slices. These are per-GT coverages, '
    'separate from one-to-one VOC true positives. No fitting or score threshold selection.'
)


def prepare():
    source = read_json(PARENT / 'plan.json')
    result = read_json(PARENT / 'comparison.json')
    manifest = read_json(MANIFEST)
    if (file_sha256(PARENT / 'comparison.json') != SOURCE_SHA256
            or result['status'] != 'complete' or result['follow_up']
            or result['plan_sha256'] != file_sha256(PARENT / 'plan.json')
            or file_sha256(MANIFEST) != source['identity']['manifest_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpus'][0]
            or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed FFN evidence, protocol or replay environment')

    checkpoints = {
        'control': source['control']['finetune'],
        'candidate': dict(path=str(PARENT / 'finetune/checkpoint.pth'),
                          sha256=result['phases']['finetune']['checkpoint_sha256']),
    }
    for checkpoint in checkpoints.values():
        path = Path(checkpoint['path'])
        if (file_sha256(path) != checkpoint['sha256']
                or read_json(path.parent / 'training_complete.json')['last_epoch'] != 0):
            raise ValueError(f'Changed balanced-finetune endpoint: {path}')

    slices = {}
    excluded = set(manifest['evaluation']['image_source_ids'])
    for name, prefix, order_field, suffix in (
        ('original', '', 'eval_order_sha256', ''),
        ('independent', 'independent_', 'independent_order_sha256', '-independent'),
    ):
        annotation = Path(source[prefix + 'annotation'])
        order = [image['owod_source_id'] for image in read_json(annotation)['images']]
        if (file_sha256(annotation) != source[prefix + 'annotation_sha256']
                or len(order) != 1024 or len(set(order)) != 1024 or excluded.intersection(order)
                or sha256('\n'.join(order).encode()).hexdigest() != source[order_field]):
            raise ValueError(f'Changed evaluation images: {name}')
        excluded.update(order)
        references = {}
        for arm, checkpoint in checkpoints.items():
            path = PARENT / f'{arm}-finetune{suffix}-heldout.json'
            reference = read_json(path)
            if (reference['plan_sha256'] != result['plan_sha256']
                    or reference['checkpoint_sha256'] != checkpoint['sha256']):
                raise ValueError(f'Unpaired evaluation: {path}')
            references[arm] = dict(path=str(path), sha256=file_sha256(path))
        slices[name] = dict(annotation=str(annotation), annotation_sha256=file_sha256(annotation),
                            order_sha256=source[order_field], references=references)

    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol=source['protocol'], seed=source['seed'], paper_comparable=False,
                checkpoint_source_sha256=SOURCE_SHA256, manifest_sha256=file_sha256(MANIFEST),
                gpu=torch.cuda.get_device_name(), device='0', batch_size=2, torch_version=source['torch_version'],
                checkpoints=checkpoints, slices=slices, criterion=CRITERION)
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


def stage(row, threshold):
    if row['native_hit']:
        return 'top50'
    if row['best_iou'] > threshold:
        return 'raw_only'
    return 'no_box'


def paired_transitions(control, candidate, classes, threshold):
    left, right = ({row['annotation_id']: row for row in arm} for arm in (control, candidate))
    if len(left) != len(control) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError('The paired annotation identities differ')
    counts = Counter()
    for identity, before in left.items():
        after = right[identity]
        if (before['image'], before['category']) != (after['image'], after['category']):
            raise ValueError(f'Changed image or class for annotation {identity}')
        if before['category'] in classes:
            counts[stage(before, threshold), stage(after, threshold)] += 1
    return counts


def verify_replay(report, reference, classes, threshold):
    for name, view in report['views'].items():
        if view != reference['views'][name]:
            raise ValueError(f'Frozen evaluation changed for {name}')
    for group, category_ids in classes.items():
        subjects = [row for row in report['rows'] if row['category'] in category_ids]
        observed = report['views'][f'raw_{group}_box_recall_iou50']
        hits = sum(row['best_iou'] > threshold for row in subjects)
        if (len(subjects), hits) != (observed['gt'], observed['hits']):
            raise ValueError(f'Frozen annotation coverage changed for {group}')


def replay(name, endpoint, dataset, loader, manifest, order, classes, threshold, plan_hash, ref):
    path = OUTPUT / f'{name}.json'
    if not path.exists():
        print(f'Evaluating {name} on {len(order)} images', flush=True)
        model, _ = load_detector(None, Path(endpoint['path']), torch.device('cuda'))
        rows = []
        observe = partial(record_targets, rows, dataset.coco, set.union(*classes.values()), threshold)
        views = evaluate_heldout(model, dataset, loader, manifest, order, proxies=False, rank_observer=observe)
        report = dict(plan_sha256=plan_hash, checkpoint_sha256=endpoint['sha256'], rows=rows, views=views)
        verify_replay(report, ref, classes, threshold)
        write_json(path, report)
    report = read_json(path)
    if report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != endpoint['sha256']:
        raise ValueError(f'Changed replay inputs: {name}')
    verify_replay(report, ref, classes, threshold)
    return report['rows']


def main():
    plan, manifest = prepare()
    threshold = manifest['evaluation']['iou_threshold']
    classes = {name: set(stage['classes']) for name, stage in zip(('old', 'current'), manifest['stages'][:2])}
    image_root = resolve_image_root(MANIFEST, manifest)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    comparisons = {}
    for slice_name, item in plan['slices'].items():
        path = Path(item['annotation'])
        dataset, loader = make_loader(image_root, path, plan['batch_size'], 2)
        order = [image['owod_source_id'] for image in read_json(path)['images']]
        rows = {}
        for arm, checkpoint in plan['checkpoints'].items():
            ref = read_json(Path(item['references'][arm]['path']))
            rows[arm] = replay(f'{slice_name}-{arm}', checkpoint, dataset, loader, manifest, order,
                               classes, threshold, plan_hash, ref)

        counts = {group: paired_transitions(rows['control'], rows['candidate'], ids, threshold)
                  for group, ids in classes.items()}
        gained_only = counts['current']['no_box', 'raw_only']
        gained_ranked = counts['current']['no_box', 'top50']
        lost_box = counts['old']['top50', 'no_box']
        lost_rank = counts['old']['top50', 'raw_only']
        comparisons[slice_name] = dict(
            transitions={k: {f'{x}->{y}': n for (x, y), n in matrix.items()} for k, matrix in counts.items()},
            current_gained_box_only=gained_only, current_gained_ranked=gained_ranked,
            old_lost_box=lost_box, old_lost_rank=lost_rank,
            current_prioritize_selection=gained_only > gained_ranked,
            old_prioritize_geometry=lost_box > lost_rank,
        )
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
               slices=comparisons, current_prioritize_selection=all(x['current_prioritize_selection'] for x in comparisons.values()),
               old_prioritize_geometry=all(x['old_prioritize_geometry'] for x in comparisons.values())))
    print('Paired target transfer comparison saved', flush=True)


if __name__ == '__main__':
    main()
