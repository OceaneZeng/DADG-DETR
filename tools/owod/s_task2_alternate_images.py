"""Build a held-out-safe second SGD pool for the S three-epoch duration control."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from tools.owod.protocol import file_sha256, read_json, stage_files
from tools.owod.screening_data import source_ids


def configure_pool(plan, output, manifest_path, screen_path, screening):
    source = output.with_name('s-instance-task2-refill-allocation')
    evidence = read_json(source/'comparison.json')
    allocation = read_json(source/'plan.json')
    if (file_sha256(source/'comparison.json') != '28725d614179b998d59008bc852cc5d28d73bf238c62b581cf447807f69d2fac'
            or evidence['plan_sha256'] != file_sha256(source/'plan.json')
            or evidence['pooled_pass'] or evidence['tag_opportunity']
            or allocation['checkpoint']['sha256'] != plan['control']['finetune']['sha256']
            or allocation['slices']['original']['annotation_sha256'] != plan['annotation_sha256']):
        raise ValueError('Changed control, heldout or allocation evidence')

    manifest, official = stage_files(manifest_path, 1)
    screen = read_json(screen_path)
    full = read_json(official['increment_train'])
    previous = read_json(stage_files(manifest_path, 0)[1]['increment_train'])
    previous_sources = source_ids(previous)
    previous_train = source_ids(read_json(screen_path.parent / screen['stages'][0]['files']['increment_train']))
    memory = source_ids(read_json(screening['train']))
    base = source_ids(read_json(screening['increment_train']))
    reserved = set(manifest['evaluation']['image_source_ids']) | set(screen['heldout_source_ids'])
    holdouts = []
    for entry in allocation['slices'].values():
        path = Path(entry['annotation'])
        names = source_ids(read_json(path))
        if file_sha256(path) != entry['annotation_sha256'] or len(names) != 1024:
            raise ValueError('Changed release-source evaluation')
        holdouts.append(names)
        reserved.update(names)
    if len(base) != 1536 or base & reserved or holdouts[0] & holdouts[1]:
        raise ValueError('Original SGD or held-out images changed')

    def stratum(name):
        return name in previous_train, name in previous_sources, name in memory

    counts = Counter(stratum(name) for name in base)
    candidates = defaultdict(list)
    for name in source_ids(full) - base - reserved:
        candidates[stratum(name)].append(name)
    selected = set()
    for key, count in counts.items():
        ordered = sorted(candidates[key], key=lambda name: sha256(f'45:{name}'.encode()).hexdigest())
        selected.update(ordered[:count])
    if Counter(stratum(name) for name in selected) != counts:
        raise ValueError('Insufficient held-out-safe images to match historical exposure and memory')

    image_names = {im['id']: im['owod_source_id'] for im in full['images']}
    labels = {arm: Counter() for arm in ('A', 'B')}
    for ann in full['annotations']:
        if ann.get('difficult') or ann.get('iscrowd') or min(ann['bbox'][2:]) <= 0:
            continue
        name = image_names[ann['image_id']]
        if name in base:
            labels['A'][ann['category_id']] += 1
        elif name in selected:
            labels['B'][ann['category_id']] += 1
    active = set(manifest['stages'][1]['classes'])
    if any(set(counts_by_class) != active for counts_by_class in labels.values()):
        raise ValueError('Both pools must cover every released current class')
    second = sorted(selected)
    digest = sha256((json.dumps(second, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    path = output/'pool-b.json'
    if path.exists() and file_sha256(path) != digest:
        raise ValueError('Existing second pool differs from registered selection')
    for command in plan['jobs'].values():
        if command[command.index('--instance-mode') + 1] != 'off':
            raise ValueError('Instance alignment must be off on both SGD pools')
        command.extend(['--screen-sgd-pool', str(path), digest])
    independent = allocation['slices']['independent']
    plan.update(pool_b_sha256=digest, pool_a_file_sha256=file_sha256(screening['increment_train']),
                extra_selection_seed=45, pool_order=['A', 'B', 'A'], increment_steps_per_epoch=384,
                non_difficult_gt_by_class=labels, stratum_counts=[
                    dict(seen_in_task1=k[0], in_author_task1=k[1], in_balanced_memory=k[2], images=v)
                    for k,v in sorted(counts.items())], source_allocation_sha256=file_sha256(source/'comparison.json'),
                independent_annotation=independent['annotation'], independent_annotation_sha256=independent['annotation_sha256'],
                independent_order_sha256=independent['order_sha256'],
                intervention='At equal steps replace epoch two of three with disjoint S author increment images. '
                             'Match Task 1 training, author Task 1 and balanced memory membership. '
                             'Keep graph calibration on A, instance alignment off and the original balanced finetune.',
                primary='Native focal finetune Current AP50 on each fixed 1024-image release-source holdout.')
    plan['criterion'] += ' Require every known and proxy gate independently on both held-out slices.'
    return second
