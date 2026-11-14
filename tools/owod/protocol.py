"""Strict author-release validation for M-OWODB and S-OWODB.

Old custom manifests and pilot bypasses are retired. Cross-task image overlap
is legal in the authors' lists; repeated annotation records are not.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from tools.owod.benchmark_spec import (COCO_IDS, NAME_TO_ID, ALIASES,
                                       official_split, specification, split_digest)

PROTOCOLS = {'m-owodb': 'M-OWODB', 's-owodb': 'S-OWODB'}
SOURCE_PLACEHOLDERS = ('<', '>', 'replace', 'placeholder', 'todo', '替换')
SCHEMA_VERSION = 3
RETIRED_MESSAGE = ('Custom/unverified OWOD settings are retired. Prepare the author '
                   'M/S-OWODB release with tools/owod/prepare_benchmarks.py; no pilot bypass is allowed.')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def annotation_class_ids(payload: Mapping) -> set[int]:
    return {int(item['category_id']) for item in payload.get('annotations', [])}


def annotation_image_ids(payload: Mapping) -> set[int]:
    return {int(item['id']) for item in payload.get('images', [])}


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def resolve_image_root(manifest_path: Path, manifest: Mapping, override: Path | None = None) -> Path:
    """Resolve the raw image root, optionally using a machine-local override."""
    if override is not None:
        return Path(override).resolve()
    value = manifest.get('image_root')
    if not value:
        raise ValueError('Author manifest must record image_root or receive an image-root override')
    return resolve_manifest_path(manifest_path, str(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_reference(source_reference: str) -> str:
    reference = str(source_reference).strip()
    if not reference or any(token in reference.lower() for token in SOURCE_PLACEHOLDERS):
        raise ValueError('source_reference is missing or still a placeholder')
    return reference


def reject_retired_local_split(annotation_root: Path) -> None:
    legacy_path = annotation_root / 'split_manifest.json'
    if legacy_path.is_file() and read_json(legacy_path).get('schema_version') != SCHEMA_VERSION:
        raise ValueError(RETIRED_MESSAGE)


def _find_stage_file(stage_dir: Path, names: Iterable[str]) -> Path:
    for name in names:
        path = stage_dir / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'Missing author-release annotation in {stage_dir}: {list(names)}')


def _validate_payload(payload: dict, allowed: set[int], split: list[str], label: str) -> None:
    images = payload.get('images', [])
    ids = [int(image['id']) for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{label}: duplicate image IDs')
    actual = Counter(str(image.get('owod_source_id', Path(image['file_name']).stem))
                     for image in images)
    if actual != Counter(list(dict.fromkeys(split))):
        raise ValueError(f'{label}: images do not match the exact official split membership')
    cats = {int(c['id']): ALIASES.get(c['name'], c['name']) for c in payload['categories']}
    if cats != {value: name for name, value in NAME_TO_ID.items()}:
        raise ValueError(f'{label}: categories must use the canonical 80 COCO IDs and names')
    observed = annotation_class_ids(payload)
    if not observed.issubset(allowed):
        raise ValueError(f'{label}: disallowed/future class labels {sorted(observed - allowed)}')
    if observed != allowed:
        raise ValueError(f'{label}: missing protocol category annotations')
    image_ids = set(ids)
    ann_ids, objects = set(), set()
    for ann in payload.get('annotations', []):
        # Different author objects can share coordinates after integer conversion.
        # Identify source records rather than merging geometrically equal objects.
        key = (ann['image_id'], ann.get('owod_source_annotation_id',
               (ann['category_id'], tuple(ann['bbox']), ann.get('difficult', 0))))
        if ann['id'] in ann_ids or key in objects:
            raise ValueError(f'{label}: duplicate annotation records')
        ann_ids.add(ann['id'])
        objects.add(key)
        if ann['image_id'] not in image_ids:
            raise ValueError(f'{label}: annotation references a missing image')
        if len(ann['bbox']) != 4 or min(ann['bbox'][2:]) <= 0:
            raise ValueError(f'{label}: invalid bounding box')
        if 'voc_bbox' not in ann or len(ann['voc_bbox']) != 4:
            raise ValueError(f'{label}: raw author VOC coordinates are required')


def build_official_manifest(annotation_root: Path, protocol: str, source_reference: str,
                            allow_image_overlap: bool = False) -> dict:
    if allow_image_overlap:
        raise ValueError('The old overlap override is retired; official overlap is validated by identity')
    spec = specification(protocol)
    annotation_root = annotation_root.resolve()
    source_reference = validate_source_reference(source_reference)
    reject_retired_local_split(annotation_root)
    records, prior_ids = [], set()
    order = [c for task in spec['task_class_ids'] for c in task]
    split_names = [spec['test_split']]
    split_names.extend(f't{task}_train' for task in range(1, 5))
    split_names.extend(f't{task}_ft' for task in range(2, 5))
    splits = {name: official_split(protocol, name) for name in split_names}
    full_split = splits[spec['test_split']]
    for index, current in enumerate(spec['task_class_ids']):
        active = [c for task in spec['task_class_ids'][:index + 1] for c in task]
        stage_dir = annotation_root / f'stage_{index}'
        files = {
            'increment_train': _find_stage_file(stage_dir, ('instances_increment_train2017.json',)),
            'train': _find_stage_file(stage_dir, ('instances_train2017.json',)),
            'known_val': _find_stage_file(stage_dir, ('instances_val2017.json',)),
            'full_val': _find_stage_file(stage_dir, ('instances_val2017_full.json',)),
        }
        train_split = splits[f't{index + 1}_train']
        ft_split = train_split if index == 0 else splits[f't{index + 1}_ft']
        payloads = {key: read_json(path) for key, path in files.items()}
        for key, allowed, names in [('increment_train', set(current), train_split),
                                    ('train', set(active), ft_split),
                                    ('known_val', set(active), full_split),
                                    ('full_val', set(COCO_IDS), full_split)]:
            _validate_payload(payloads[key], allowed, names, f'stage_{index}/{key}')
        def objects(payload, allowed):
            return Counter((a['image_id'], a['category_id'], tuple(a['voc_bbox']), a.get('difficult', 0))
                           for a in payload['annotations'] if a['category_id'] in allowed)
        if objects(payloads['known_val'], set(active)) != objects(payloads['full_val'], set(active)):
            raise ValueError(f'stage_{index}: known validation is not the visible subset of full labels')
        if annotation_class_ids(payloads['increment_train']) != set(current):
            raise ValueError(f'stage_{index}: missing current training categories')
        if annotation_class_ids(payloads['train']) != set(active):
            raise ValueError(f'stage_{index}: balanced finetuning omits known categories')
        if index:
            counts = Counter(a['category_id'] for a in payloads['train']['annotations'])
            if any(counts[c] < 50 for c in active):
                raise ValueError(f'stage_{index}: released replay must retain at least 50 instances per known class')
        image_ids = annotation_image_ids(payloads['increment_train'])
        records.append({
            'index': index, 'classes': current, 'active_classes': active,
            'unknown_classes': [c for c in order if c not in active],
            'increment_image_count': len(train_split), 'train_image_count': len(ft_split),
            'cross_task_image_overlap': len(image_ids & prior_ids),
            'files': {key: str(path) for key, path in files.items()},
            'sha256': {key: file_sha256(path) for key, path in files.items()},
        })
        prior_ids.update(image_ids)
    if len({record['sha256']['full_val'] for record in records}) != 1:
        raise ValueError('All tasks must evaluate the same full-label author test annotations')
    return {
        'schema_version': SCHEMA_VERSION, 'benchmark': PROTOCOLS[protocol],
        'protocol': protocol, 'official_annotations': True, 'paper_comparable': True,
        'validation_mode': 'author_release_strict', 'source_reference': source_reference,
        'annotation_root': str(annotation_root), 'category_order_source_ids': order,
        'evaluation': {**spec, 'image_source_ids': full_split,
                       'duplicate_test_entries': len(full_split) - len(set(full_split))},
        # Keep the exact author ordering in the manifest itself so the
        # generated protocol remains self-contained.
        'splits': splits,
        'split_hashes': {name: split_digest(values) for name, values in splits.items()},
        'stages': records,
        'integrity': {'category_count': 80, 'stage_count': 4,
                      'task_class_counts': list(map(len, spec['task_class_ids'])),
                      'official_test_entries': len(full_split),
                      'unique_test_images': len(set(full_split))},
    }


def stage_files(manifest_path: Path, stage: int,
                allow_unverified: bool = False) -> tuple[dict, dict[str, Path]]:
    if allow_unverified:
        raise ValueError(RETIRED_MESSAGE)
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if (manifest.get('schema_version') != SCHEMA_VERSION or
            not manifest.get('official_annotations') or not manifest.get('paper_comparable')):
        raise ValueError(RETIRED_MESSAGE)
    validate_source_reference(manifest.get('source_reference', ''))
    spec = specification(manifest['protocol'])
    if any(manifest.get('evaluation', {}).get(key) != value for key, value in spec.items()):
        raise ValueError('Manifest evaluation does not match the author protocol')
    splits = manifest.get('splits')
    if not isinstance(splits, dict) or spec['test_split'] not in splits:
        raise ValueError('Manifest is missing embedded author splits')
    if manifest['evaluation'].get('image_source_ids') != splits[spec['test_split']]:
        raise ValueError('Manifest test sequence differs from its embedded split')
    stages = manifest.get('stages', [])
    order = [c for task in spec['task_class_ids'] for c in task]
    if len(stages) != 4 or manifest.get('category_order_source_ids') != order:
        raise ValueError('Manifest category order is not the author-defined order')
    for i, record in enumerate(stages):
        active = [c for task in spec['task_class_ids'][:i + 1] for c in task]
        if (record['classes'] != spec['task_class_ids'][i] or record['active_classes'] != active
                or record['unknown_classes'] != [c for c in order if c not in active]):
            raise ValueError(f'stage_{i}: category partition differs from the author definition')
    if not 0 <= stage < 4:
        raise ValueError('stage must be in [0, 4)')
    record = stages[stage]
    files = {key: resolve_manifest_path(manifest_path, value) for key, value in record['files'].items()}
    if set(files) != {'increment_train', 'train', 'known_val', 'full_val'}:
        raise ValueError('Manifest stage is missing required files')
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != record['sha256'].get(key):
            raise ValueError(f'Annotation changed after registration: {path}')
        names = (splits[f't{stage + 1}_train'] if key == 'increment_train'
                 else splits[f't{stage + 1}_ft'] if key == 'train' and stage
                 else splits['t1_train'] if key == 'train'
                 else manifest['evaluation']['image_source_ids'])
        allowed = (set(record['classes']) if key == 'increment_train' else set(COCO_IDS)
                   if key == 'full_val' else set(record['active_classes']))
        _validate_payload(read_json(path), allowed, names, f'stage_{stage}/{key}')
    return manifest, files
