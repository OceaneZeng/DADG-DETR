"""Validate the locked S Task 2 old-class refill rule on another training-source split."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.probe_task1_known_queries import fresh_origins
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION, PARENT, PREVIOUS_PROXY, RANKING, SCREEN, proxy_data
from tools.owod.run_s_task2_old_dedup import DEDUP_IOU, LIMIT, OLD_POOL, OUTPUT as SOURCE, evaluate_pair
from tools.owod.screening_data import screening_files, source_ids, write_json


OUTPUT = SOURCE.with_name('s-instance-task2-old-dedup-holdout')
SELECTION_SEED = 59


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for the independent holdout')
    source = read_json(SOURCE / 'plan.json')
    report = read_json(SOURCE / 'comparison.json')
    duration = read_json(DURATION / 'plan.json')
    duration_result = read_json(DURATION / 'comparison.json')
    first_holdout = Path(source['annotation'])
    if (report['status'] != 'complete' or not report['follow_up'] or
            report['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or
            duration_result['plan_sha256'] != file_sha256(DURATION / 'plan.json') or
            source['nms_iou'] != DEDUP_IOU or source['max_detections'] != LIMIT or source['old_pool'] != OLD_POOL or
            file_sha256(MANIFEST) != source['manifest_sha256'] or file_sha256(SCREEN) != duration['screen_sha256'] or
            file_sha256(first_holdout) != duration['annotation_sha256'] or
            source['annotation_sha256'] != duration['annotation_sha256']):
        raise ValueError('Changed screening provenance or duplicate suppression rule')
    for arm, entry in source['checkpoints'].items():
        if (file_sha256(Path(entry['path'])) != entry['sha256'] or
                entry['sha256'] != duration_result['arms'][arm]['finetune']['checkpoint_sha256']):
            raise ValueError(f'Changed independent-test weight for {arm}')
    print('Selecting the independent S duration holdout', flush=True)
    manifest, first = screening_files(MANIFEST, SCREEN, 0)
    _, second = screening_files(MANIFEST, SCREEN, 1)
    excluded = set(manifest['evaluation']['image_source_ids']) | set(read_json(SCREEN)['heldout_source_ids'])
    for path in (first['increment_train'], second['increment_train'], second['train']):
        excluded.update(source_ids(read_json(path)))
    excluded.update(read_json(PARENT / 'selection/inputs.json')['origins'])
    excluded.update(read_json(PREVIOUS_PROXY / 'selection/origins.json'))
    excluded.update(read_json(RANKING / 'plan.json')['split'])
    excluded.update(source_ids(read_json(first_holdout)))
    origins = fresh_origins(manifest, excluded, SELECTION_SEED)
    if set(origins) & excluded or Counter(origins.values()) != {'t1_train': 512, 't2_train': 512}:
        raise ValueError('Independent holdout overlap or incorrect source balance')
    data = proxy_data(manifest, origins)
    order = [r['owod_source_id'] for r in data['images']]
    if order != list(origins) or {a['category_id'] for a in data['annotations']} != set(manifest['stages'][1]['active_classes']):
        raise ValueError('Incomplete release-only S holdout')
    annotation = OUTPUT / 'heldout.json'
    plan = {k: source[k] for k in ('protocol', 'paper_comparable', 'seed', 'images', 'batch_size', 'checkpoints',
                                   'manifest_sha256', 'max_detections', 'old_pool', 'nms_iou', 'unknown_policy', 'criterion')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=report['plan_sha256'], source_result_sha256=file_sha256(SOURCE / 'comparison.json'),
                duration_plan_sha256=duration_result['plan_sha256'], selection_seed=SELECTION_SEED, origins=origins,
                excluded_ids_sha256=sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
                eval_order_sha256=sha256('\n'.join(order).encode()).hexdigest(), annotation=str(annotation),
                annotation_sha256=sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest(),
                gpu=torch.cuda.get_device_name(), device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_version=str(torch.__version__))
    pin(OUTPUT / 'plan.json', plan)
    if not annotation.exists():
        write_json(annotation, data)
    if file_sha256(annotation) != plan['annotation_sha256']:
        raise ValueError('Changed independent holdout annotation')
    evaluate_pair(OUTPUT, plan, manifest, annotation, order)


if __name__ == '__main__':
    main()
