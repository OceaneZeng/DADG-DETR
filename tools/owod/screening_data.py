"""Training-only diagnostic subsets; never a replacement benchmark protocol."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.owod.protocol import file_sha256, read_json, stage_files


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_ids(payload):
    return {image['owod_source_id'] for image in payload['images']}


def select_images(payload, count, classes, seed, excluded=()):
    """Deterministic class coverage followed by seeded hash order, no test GT."""
    images = {image['id']: image['owod_source_id'] for image in payload['images']}
    labels = defaultdict(set)
    for ann in payload['annotations']:
        if not ann.get('difficult', 0) and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
            labels[images[ann['image_id']]].add(ann['category_id'])
    eligible = set(images.values()) - set(excluded)
    order = sorted(eligible, key=lambda name: hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest())
    if count < len(classes) or count > len(order):
        raise ValueError(f'Screen image budget {count} must be between {len(classes)} and {len(order)}')
    selected = set()
    for category in classes:
        if any(category in labels[name] for name in selected):
            continue
        candidate = next((name for name in order if category in labels[name]), None)
        if candidate is None:
            raise ValueError(f'No eligible screening image for class {category}')
        selected.add(candidate)
    for name in order:
        if len(selected) == count:
            break
        selected.add(name)
    return selected


def subset(payload, names, classes):
    images = [image for image in payload['images'] if image['owod_source_id'] in names]
    ids = {image['id'] for image in images}
    annotations = [ann for ann in payload['annotations']
                   if ann['image_id'] in ids and ann['category_id'] in classes]
    return {**payload, 'images': images, 'annotations': annotations}


def merge_training(payloads):
    images, annotations = {}, {}
    for payload in payloads:
        for image in payload['images']:
            old = images.setdefault(image['id'], image)
            if old != image:
                raise ValueError('Inconsistent image identities across author tasks')
        for ann in payload['annotations']:
            key = (ann['image_id'], ann['owod_source_annotation_id'])
            record = {k: v for k, v in ann.items() if k != 'id'}
            old = annotations.setdefault(key, record)
            if old != record:
                raise ValueError('Inconsistent source annotation across author tasks')
    return {'categories': payloads[0]['categories'], 'images': list(images.values()),
            'annotations': [dict(ann, id=i + 1) for i, ann in enumerate(annotations.values())]}


def prepare_screen(manifest_path, output, train_images=2000, val_images=400, seed=42, *, task2_images=None):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, first = stage_files(manifest_path, 0)
    _, second = stage_files(manifest_path, 1)
    inputs = [read_json(files['increment_train']) for files in (first, second)]
    memory = read_json(second['train'])
    combined = merge_training(inputs)
    # Validation is globally disjoint from every training/calibration/promotion
    # phase. Excluding released memory here keeps Task 2 finetuning intact.
    heldout = select_images(combined, val_images, manifest['stages'][1]['active_classes'], seed,
                            source_ids(memory))
    budgets = (train_images, train_images if task2_images is None else task2_images)
    selected = [select_images(data, budgets[stage], manifest['stages'][stage]['classes'], seed, heldout)
                for stage, data in enumerate(inputs)]
    spec_path = output / 'screening.json'
    settings = dict(kind='training_only_screen', version=1, paper_comparable=False,
                    parent_manifest_sha256=file_sha256(manifest_path), protocol=manifest['protocol'],
                    seed=seed, train_images=train_images, val_images=val_images)
    if task2_images is not None:
        settings["task2_train_images"] = task2_images
    if spec_path.exists():
        saved = read_json(spec_path)
        if (saved.get('task2_train_images') != task2_images or
                any(saved.get(key) != value for key, value in settings.items())):
            raise ValueError('Screen settings changed; use a new screening output directory')
        for stage in range(2):
            screening_files(manifest_path, spec_path, stage)
        return spec_path
    if output.exists() and any(output.iterdir()):
        raise ValueError('Screen data output must be empty (or contain a verified screening.json)')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.screen-prepare-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'data'
        stages = []
        for stage, data in enumerate(inputs):
            classes = manifest['stages'][stage]
            train = subset(data, selected[stage], classes['classes'])
            validation = subset(combined, heldout, classes['active_classes'])
            payloads = {'increment_train': train, 'train': train if stage == 0 else memory,
                        'full_val': validation}
            record = {'files': {}, 'sha256': {}}
            for key, payload in payloads.items():
                relative = Path(f'stage_{stage}') / f'{key}.json'
                write_json(staging / relative, payload)
                record['files'][key] = relative.as_posix()
                record['sha256'][key] = file_sha256(staging / relative)
            stages.append(record)
        staged_spec = staging / 'screening.json'
        write_json(staged_spec, {**settings, 'stages': stages,
                              'heldout_source_ids': sorted(heldout),
                              'selection': 'class coverage then SHA256(seed:source_id); train sources only',
                              'unknown_metrics': 'unavailable: no future/test annotations used'})
        for stage in range(2):
            screening_files(manifest_path, staged_spec, stage)
        if output.exists():
            output.rmdir()  # Only an empty destination is permitted above.
        staging.replace(output)
    return spec_path


def screening_files(manifest_path, spec_path, stage):
    """Verify original schema-3 inputs and diagnostic file integrity on each use."""
    manifest, official = stage_files(Path(manifest_path), stage)
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    if (spec.get('kind') != 'training_only_screen' or spec.get('version') != 1
            or spec.get('paper_comparable') is not False
            or spec.get('parent_manifest_sha256') != file_sha256(Path(manifest_path))
            or spec.get('protocol') != manifest['protocol'] or stage not in (0, 1)):
        raise ValueError('Invalid screening provenance; only verified Tasks 1 and 2 are supported')
    record = spec['stages'][stage]
    files = {key: spec_path.parent / value for key, value in record['files'].items()}
    if set(files) != {'increment_train', 'train', 'full_val'}:
        raise ValueError('Screen is missing phase files')
    for key, path in files.items():
        if file_sha256(path) != record['sha256'][key]:
            raise ValueError(f'Screen annotation changed: {path}')
    heldout = set(spec['heldout_source_ids'])
    active = set(manifest['stages'][stage]['active_classes'])
    for key in ('increment_train', 'train'):
        data, original = read_json(files[key]), read_json(official[key])
        names = source_ids(data)
        allowed = set(manifest['stages'][stage]['classes']) if key == 'increment_train' else active
        if (names & heldout or not names <= source_ids(original)
                or {ann['category_id'] for ann in data['annotations']} != allowed
                or data != subset(original, names, allowed)):
            raise ValueError('Screen training changed source records, leaked holdout, or lost class coverage')
        if stage == 1 and key == 'train' and data != original:
            raise ValueError('Screen must retain the entire released balanced memory')
    validation = read_json(files['full_val'])
    if (source_ids(validation) != heldout
            or {ann['category_id'] for ann in validation['annotations']} != active):
        raise ValueError('Screen validation identity/class coverage mismatch')
    # Verify held-out records against the two released training sources, never test GT.
    sources = [read_json(stage_files(Path(manifest_path), i)[1]['increment_train']) for i in range(2)]
    if validation != subset(merge_training(sources), heldout, active):
        raise ValueError('Screen validation does not match released training records')
    return manifest, files


def experiment_files(args, stage):
    spec = getattr(args, 'owod_screening', '')
    if spec:
        return screening_files(Path(args.owod_manifest), Path(spec), stage)
    return stage_files(Path(args.owod_manifest), stage)
