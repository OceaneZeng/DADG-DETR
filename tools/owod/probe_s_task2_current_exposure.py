"""Compare S Task 2 native Current AP by image exposure at frozen increment endpoints."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_current_fp_sources import OUTPUT as FP_SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task1_known_fraction import proxy_data
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION, SCREEN
from tools.owod.screening_data import screening_files, source_ids, subset, write_json


OUTPUT = DURATION.with_name('s-instance-task2-current-exposure')
SELECTION_SEED = 61
COUNT = 489


def prepare():
    source = read_json(FP_SOURCE/'comparison.json')
    previous = read_json(FP_SOURCE/'plan.json')
    duration = read_json(DURATION/'comparison.json')
    training = read_json(DURATION/'plan.json')
    if (file_sha256(FP_SOURCE/'comparison.json') != '3cadc2000b49a660a93d17bc329cdb1a380eefd00f9e97f0969d7ca999722ec8'
            or source['priority'] is not None or source['plan_sha256'] != file_sha256(FP_SOURCE/'plan.json')
            or file_sha256(DURATION/'comparison.json') != '4f698b6c4c608cb835f063960416ba546af06e557356f623555b4158eee9172f'
            or duration['plan_sha256'] != file_sha256(DURATION/'plan.json')
            or file_sha256(MANIFEST) != previous['manifest_sha256'] or file_sha256(SCREEN) != training['screen_sha256']
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != previous['gpu']
            or str(torch.__version__) != previous['torch_version']):
        raise ValueError('Changed inputs, checkpoints or frozen evaluation environment')
    manifest, first = screening_files(MANIFEST, SCREEN, 0)
    _, second = screening_files(MANIFEST, SCREEN, 1)
    first_ids = source_ids(read_json(first['increment_train']))
    second_ids = source_ids(read_json(second['increment_train']))
    unseen = {k for k, v in training['origins'].items() if v == 't2_train'}
    pools = dict(both_tasks=first_ids & second_ids, task2_only=second_ids-first_ids, heldout=unseen)
    all_ids = set.union(*pools.values())
    if ([len(v) for v in pools.values()] != [1047, COUNT, 512] or unseen & (first_ids | second_ids)
            or unseen & source_ids(read_json(second['train'])) or all_ids - set(manifest['splits']['t2_train'])
            or all_ids & set(manifest['evaluation']['image_source_ids'])):
        raise ValueError('Unexpected Task 2 exposure membership')
    groups = {k: sorted(v, key=lambda x: sha256(f'{SELECTION_SEED}:{x}'.encode()).digest())[:COUNT]
              for k, v in pools.items()}
    data = proxy_data(manifest, [name for group in groups.values() for name in group])
    known, current = manifest['stages'][1]['active_classes'], manifest['stages'][1]['classes']
    cohorts, positives = {}, {}
    for group, names in groups.items():
        cohort = subset(data, set(names), known)
        if [im['owod_source_id'] for im in cohort['images']] != names:
            raise ValueError(f'Changed ordered identities: {group}')
        cohorts[group] = cohort
        positives[group] = Counter(a['category_id'] for a in cohort['annotations'] if not a.get('difficult', 0))
    supported = [c for c in current if all(counts[c] >= 5 for counts in positives.values())]
    checkpoints = {}
    for epochs in (3, 6):
        arm = f'{epochs}epochs'
        path = DURATION/arm/'increment/checkpoint.pth'
        digest = duration['arms'][arm]['increment']['checkpoint_sha256']
        if file_sha256(path) != digest or read_json(path.parent/'training_complete.json')['last_epoch'] != epochs-1:
            raise ValueError(f'Changed increment endpoint: {arm}')
        checkpoints[arm] = dict(path=str(path), sha256=digest)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256=file_sha256(FP_SOURCE/'comparison.json'), duration_sha256=file_sha256(DURATION/'comparison.json'),
                manifest_sha256=file_sha256(MANIFEST), screening_sha256=file_sha256(SCREEN),
                protocol='s-owodb', seed=42, selection_seed=SELECTION_SEED, device=os.environ['CUDA_VISIBLE_DEVICES'],
                gpu=torch.cuda.get_device_name(), batch_size=2, torch_version=str(torch.__version__), paper_comparable=False,
                groups=groups, eligible={k:len(v) for k,v in pools.items()}, common_classes=supported,
                checkpoints=checkpoints, ap_mode=manifest['evaluation']['ap_mode'], scoring='focal', max_detections=50,
                annotations={k:sha256((json.dumps(v, indent=2, sort_keys=True)+'\n').encode()).hexdigest()
                             for k,v in cohorts.items()},
                criterion='Compare task2_only minus heldout native Current AP on classes with >=5 non-difficult GT '
                          'in every cohort. Require at least 11 supported classes. At both durations: gap >=3 pp '
                          'prioritizes generalization, or absolute gap <=1 pp and task2_only AP <5% prioritizes '
                          'training fit. Otherwise retain an inconclusive or duration-dependent finding.')
    pin(OUTPUT/'plan.json', plan)
    for group, data in cohorts.items():
        path = OUTPUT/f'{group}-annotations.json'
        if not path.exists():
            write_json(path, data)
        if file_sha256(path) != plan['annotations'][group]:
            raise ValueError(f'Changed released annotations: {group}')
    return manifest, plan


def main():
    print('Checking exposure identities and rebuilding released annotations', flush=True)
    manifest, plan = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    supported = plan['common_classes']
    old, stage = manifest['stages'][:2]
    summary, gaps = {}, {}
    for arm, entry in plan['checkpoints'].items():
        model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
        rows = summary[arm] = {}
        for group, names in plan['groups'].items():
            path = OUTPUT/f'{arm}-{group}.json'
            if not path.exists():
                print(f'Evaluating {arm} {group}: {len(names)} images', flush=True)
                ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), OUTPUT/f'{group}-annotations.json', 2, 2)
                views = evaluate_views(model, loader, ds.coco, {'native': (stage['active_classes'], 'focal', 50)},
                                       dict(manifest['evaluation'], image_source_ids=names, known_only_screening=True),
                                       old['classes'], stage['classes'], 'cuda', include_current_stats=True, print_freq=64)
                write_json(path, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'],
                                      annotation_sha256=plan['annotations'][group], views=views))
            v = read_json(path)
            if (v['plan_sha256'], v['checkpoint_sha256'], v['annotation_sha256']) != (
                    digest, entry['sha256'], plan['annotations'][group]):
                raise ValueError(f'Changed exposure result: {arm}/{group}')
            native = v['views']['native']
            stats = native['current_class_stats']
            rows[group] = dict(**native['metrics'], raw_current=v['views']['raw_current_box_recall_iou50'],
                               common_current_ap50=mean(stats[str(k)]['ap50'] for k in supported) if supported else None,
                               voc_tp=sum(c['tp'] for c in stats.values()), voc_fp=sum(c['fp'] for c in stats.values()))
        del model
        gaps[arm] = rows['task2_only']['common_current_ap50']-rows['heldout']['common_current_ap50'] if supported else None
    focus = 'inconclusive'
    if len(supported) >= 11:
        if all(gap >= .03 for gap in gaps.values()):
            focus = 'generalization'
        elif all(abs(gap) <= .01 and summary[arm]['task2_only']['common_current_ap50'] < .05 for arm,gap in gaps.items()):
            focus = 'training_fit'
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              arms=summary, common_classes=supported, gaps=gaps, focus=focus))
    print('Current exposure comparison saved:', OUTPUT/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
