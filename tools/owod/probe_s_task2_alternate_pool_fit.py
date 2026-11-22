"""Measure whether exposure to B improves B itself at the frozen S increment endpoint."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.probe_s_task2_current_exposure import evaluate_groups
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_fraction import proxy_data
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION, SCREEN
from tools.owod.screening_data import screening_files, source_ids, subset, write_json


SOURCE = DURATION.with_name('s-instance-task2-increment-alternate-images')
OUTPUT = DURATION.with_name('s-instance-task2-alternate-pool-fit')


def prepare():
    training = read_json(SOURCE/'plan.json')
    report = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != 'aa33f4b9c055193775ecd3535ce4198ef654a47a4085ca4f01e0664d7e5e892b'
            or report['plan_sha256'] != file_sha256(SOURCE/'plan.json') or report['status'] != 'complete'
            or report['follow_up'] or file_sha256(MANIFEST) != training['identity']['manifest_sha256']
            or file_sha256(SCREEN) != training['identity']['screening_sha256']
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != training['gpus'][0] or str(torch.__version__) != training['torch_version']):
        raise ValueError('Changed paired training evidence or evaluation environment')
    manifest, files = screening_files(MANIFEST, SCREEN, 1)
    second = SOURCE/'pool-b.json'
    if file_sha256(second) != training['pool_b_sha256'] or file_sha256(files['increment_train']) != training['pool_a_file_sha256']:
        raise ValueError('Changed historical SGD images')
    groups = {'A': sorted(source_ids(read_json(files['increment_train']))), 'B': read_json(second)}
    heldout = read_json(Path(training['independent_annotation']))
    groups['independent'] = [im['owod_source_id'] for im in heldout['images']]
    original = Path(training['annotation'])
    occupied = set(manifest['evaluation']['image_source_ids']) | source_ids(read_json(original))
    distinct = set().union(*map(set, groups.values()))
    if (file_sha256(original) != training['annotation_sha256']
            or file_sha256(Path(training['independent_annotation'])) != training['independent_annotation_sha256']
            or list(map(len, groups.values())) != [1536, 1536, 1024]
            or len(distinct) != 4096 or distinct & occupied):
        raise ValueError('Changed cohort sizes, identity or separation')
    stage = manifest['stages'][1]
    data = proxy_data(manifest, groups['A'] + groups['B'])
    cohorts = {k: subset(data, set(groups[k]), stage['active_classes']) for k in ('A', 'B')}
    cohorts['independent'] = heldout
    for k, payload in cohorts.items():
        current = {ann['category_id'] for ann in payload['annotations'] if not ann.get('difficult', 0)}
        if [im['owod_source_id'] for im in payload['images']] != groups[k] or not set(stage['classes']) <= current:
            raise ValueError(f'Changed annotation order or missing current classes in {k}')
        if k in ('A', 'B'):
            counts = Counter(str(a['category_id']) for a in payload['annotations']
                             if a['category_id'] in stage['classes'] and not a.get('difficult') and not a.get('iscrowd')
                             and min(a['bbox'][2:]) > 0)
            if counts != training['non_difficult_gt_by_class'][k]:
                raise ValueError(f'Current-class annotations differ from SGD pool {k}')
    checkpoints = dict(repeat=training['control']['increment'],
                       alternate=dict(path=str(SOURCE/'increment/checkpoint.pth'), sha256=report['phases']['increment']['checkpoint_sha256']))
    for entry in checkpoints.values():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Frozen increment checkpoint differs from completed run')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), protocol='s-owodb',
                seed=42, paper_comparable=False, gpu=torch.cuda.get_device_name(), device='0', batch_size=2,
                torch_version=str(torch.__version__), checkpoints=checkpoints, manifest_sha256=file_sha256(MANIFEST),
                groups=groups, common_classes=stage['classes'], source_sha256=file_sha256(SOURCE/'comparison.json'),
                source_plan_sha256=report['plan_sha256'], original_increment_delta=report['phases']['increment']['delta'],
                annotations={k:sha256((json.dumps(v, indent=2, sort_keys=True)+'\n').encode()).hexdigest()
                             for k, v in cohorts.items()},
                criterion='Flag a B-specific response only if the 21-class increment Current AP50 improvement on B '
                          'is at least 1 pp and exceeds the improvements on A, original holdout and independent holdout '
                          'by at least 1 pp each. Flag loss on A separately at -1 pp. Otherwise do not attribute '
                          'the final holdout failure to a demonstrated B-specific gain. Report Previous and Known, '
                          'class counts and all-query Current box coverage by cohort. This probe cannot advance the training arm.')
    pin(OUTPUT/'plan.json', plan)
    for group, payload in cohorts.items():
        path = OUTPUT/f'{group}-annotations.json'
        if not path.exists():
            write_json(path, payload)
        if file_sha256(path) != plan['annotations'][group]:
            raise ValueError(f'Changed registered annotation: {group}')
    return manifest, plan


def main():
    print('Checking frozen increment checkpoints, training pools and independent evaluation', flush=True)
    manifest, plan = prepare()
    arms = evaluate_groups(OUTPUT, manifest, plan)
    deltas = {group:{k:row[k]-arms['repeat'][group][k] for k in ('current_ap50', 'previous_ap50', 'known_ap50')}
              for group, row in arms['alternate'].items()}
    deltas['original'] = plan['original_increment_delta']
    on_b = deltas['B']['current_ap50']
    contrasts = {k:on_b-deltas[k]['current_ap50'] for k in ('A', 'original', 'independent')}
    write_json(OUTPUT/'comparison.json',dict(status='complete', paper_comparable=False, plan_sha256=file_sha256(OUTPUT/'plan.json'),
               arms=arms, alternate_minus_repeat=deltas, b_gain_minus_other_gain=contrasts,
               b_specific=on_b >= .01 and min(contrasts.values()) >= .01,
               loss_on_a=deltas['A']['current_ap50'] <= -.01))
    print('Frozen SGD pool comparison saved', flush=True)


if __name__ == '__main__':
    main()
