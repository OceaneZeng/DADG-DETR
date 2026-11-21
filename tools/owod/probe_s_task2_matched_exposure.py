"""Pair seen and unseen S Task 2 images using released annotations only."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_instance_pilot import ROOT
from tools.owod.screening_data import subset, write_json


OUTPUT = ROOT/'output/s-instance-task2-matched-exposure'
SEED = 67
SEEN = {'no_old': 'task2_only', 'with_old': 'both_tasks'}


def hash_order(key):
    return sha256(f'{SEED}:{key}'.encode()).digest()


def image_bins(data, old, current):
    objects = defaultdict(list)
    for ann in data['annotations']:
        if not ann.get('difficult', 0):
            objects[ann['image_id']].append(ann)
    bins = defaultdict(list)
    for image in data['images']:
        anns = objects[image['id']]
        anchor = max((a for a in anns if a['category_id'] in current),
                     key=lambda a: (a['area'], -a['category_id'], -a['id']), default=None)
        if anchor is None:
            continue
        has_old = any(a['category_id'] in old for a in anns)
        size = int(anchor['area'] >= 32**2) + int(anchor['area'] >= 96**2)
        bins[has_old, anchor['category_id'], size].append(image['owod_source_id'])
    return bins


def match_images(datasets, old, current):
    bins = {g: image_bins(data, old, current) for g, data in datasets.items()}
    groups, cohorts = {}, {}
    for stratum, source_group in SEEN.items():
        pairs = []
        for key, seen_ids in bins[source_group].items():
            if key[0] != (stratum == 'with_old'):
                raise ValueError('Unexpected previous-class annotation in the training cohort')
            unseen_ids = bins['heldout'].get(key, ())
            pairs.extend(zip(sorted(seen_ids, key=hash_order), sorted(unseen_ids, key=hash_order)))
        pairs.sort(key=lambda p: hash_order('|'.join(p)))
        for side, source, index in [('seen', source_group, 0), ('unseen', 'heldout', 1)]:
            name = f'{stratum}-{side}'
            ids = [p[index] for p in pairs]
            cohort = subset(datasets[source], set(ids), old | current)
            images = {im['owod_source_id']: im for im in cohort['images']}
            cohort['images'] = [images[i] for i in ids]
            groups[name], cohorts[name] = ids, cohort
    return groups, cohorts


def prepare():
    import torch
    from tools.owod.probe_s_task2_current_exposure import MANIFEST, OUTPUT as SOURCE
    from tools.owod.run_cross_attention_study import pin

    prior, result = read_json(SOURCE/'plan.json'), read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != '8e206bb89cf9b4320b1e4d5121a925bbdb5cccf682622f09b08e9d8279f81049'
            or result['focus'] != 'inconclusive' or result['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(MANIFEST) != prior['manifest_sha256'] or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != prior['gpu'] or str(torch.__version__) != prior['torch_version']):
        raise ValueError('Changed source study or evaluation environment')
    manifest = read_json(MANIFEST)
    datasets = {}
    for g, digest in prior['annotations'].items():
        path = SOURCE/f'{g}-annotations.json'
        if file_sha256(path) != digest:
            raise ValueError(f'Changed source annotations: {g}')
        datasets[g] = read_json(path)
    for entry in prior['checkpoints'].values():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Frozen checkpoint changed')
    old, current = (s['classes'] for s in manifest['stages'][:2])
    groups, cohorts = match_images(datasets, set(old), set(current))
    unique_ids = set.union(*(set(ids) for ids in groups.values()))
    if (len(unique_ids) != sum(map(len, groups.values())) or unique_ids & set(manifest['evaluation']['image_source_ids'])
            or any(len(groups[f'{s}-{side}']) != n for s, n in [('no_old', 236), ('with_old', 232)]
                   for side in ('seen', 'unseen'))):
        raise ValueError('Changed matched cohorts or reused image identity')
    counts = [Counter(a['category_id'] for a in group['annotations'] if not a.get('difficult', 0))
              for group in cohorts.values()]
    supported = [k for k in current if all(c[k] >= 5 for c in counts)]
    plan = {k:prior[k] for k in ('manifest_sha256', 'protocol', 'seed', 'gpu', 'batch_size', 'torch_version',
                                  'paper_comparable', 'checkpoints', 'ap_mode', 'scoring', 'max_detections')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                device=os.environ['CUDA_VISIBLE_DEVICES'], selection_seed=SEED, source_sha256=file_sha256(SOURCE/'comparison.json'),
                source_plan_sha256=result['plan_sha256'], groups=groups, common_classes=supported,
                pairing='Paired by index within each stratum. Match previous-class presence and largest '
                        'non-difficult current GT class and pixel area bin (<32^2, 32^2 to <96^2, >=96^2).',
                annotations={k:sha256((json.dumps(v, indent=2, sort_keys=True)+'\n').encode()).hexdigest()
                             for k,v in cohorts.items()},
                criterion='Primary no_old seen minus unseen common Current AP. Keep >=5 GT per class in every group '
                          'and >=11 classes. At both durations: gap >=3 pp flags a residual gap, or abs(gap)<=1 pp '
                          'and seen AP<5% flags limited fit. Otherwise inconclusive. Report with_old separately.')
    pin(OUTPUT/'plan.json', plan)
    for g, cohort in cohorts.items():
        path = OUTPUT/f'{g}-annotations.json'
        if not path.exists():
            write_json(path, cohort)
        if file_sha256(path) != plan['annotations'][g]:
            raise ValueError(f'Matched annotations changed: {g}')
    return manifest, plan


def main():
    from tools.owod.probe_s_task2_current_exposure import evaluate_groups

    manifest, plan = prepare()
    rows = evaluate_groups(OUTPUT, manifest, plan)
    gaps = {}
    for arm, groups in rows.items():
        gaps[arm] = {s:groups[f'{s}-seen']['common_current_ap50']-groups[f'{s}-unseen']['common_current_ap50']
                     if plan['common_classes'] else None for s in SEEN}
    focus = 'inconclusive'
    if len(plan['common_classes']) >= 11:
        if all(v['no_old'] >= .03 for v in gaps.values()):
            focus = 'residual_gap'
        elif all(abs(v['no_old']) <= .01 and rows[a]['no_old-seen']['common_current_ap50'] < .05
                 for a, v in gaps.items()):
            focus = 'training_fit'
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, arms=rows, gaps=gaps,
                                              focus=focus, plan_sha256=file_sha256(OUTPUT/'plan.json')))
    print('Matched exposure comparison saved:', OUTPUT/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
