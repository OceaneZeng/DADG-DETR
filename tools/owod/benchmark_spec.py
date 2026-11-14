"""Author-defined OWOD specifications loaded from the released manifests.

The schema-3 manifests under ``data/owod/<protocol>/`` are the single local
source of the M-OWODB and S-OWODB protocol.  They contain the author class
order, evaluation rules, and the exact ordered image lists used by every
training/evaluation phase.  Keeping this loader independent of generated
stage annotations makes ``plan`` usable from a clean checkout as well.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = PROJECT_ROOT / 'data' / 'owod'
PROTOCOLS = ('m-owodb', 's-owodb')
COCO_NAMES = (
    'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,'
    'fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
    'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,'
    'skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,'
    'surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,'
    'apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,'
    'potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,'
    'microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,'
    'hair drier,toothbrush'
).split(',')
COCO_IDS = (list(range(1, 12)) + [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
NAME_TO_ID = dict(zip(COCO_NAMES, COCO_IDS))
ALIASES = dict(zip(['aeroplane', 'diningtable', 'motorbike', 'pottedplant',
                   'sofa', 'tvmonitor'],
                  ['airplane', 'dining table', 'motorcycle', 'potted plant',
                   'couch', 'tv']))
def manifest_path(protocol: str) -> Path:
    """Return the checked-in schema-3 manifest for ``protocol``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unsupported OWOD benchmark: {protocol}')
    return MANIFEST_ROOT / protocol / 'split_manifest.json'


def split_digest(values: list[str]) -> str:
    """Hash an ordered split independent of the source file line ending."""
    payload = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=len(PROTOCOLS))
def load_manifest(protocol: str) -> dict:
    """Load one canonical protocol manifest and validate its basic shape."""
    path = manifest_path(protocol)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing canonical OWOD manifest: {path}. '
            'Prepare data/owod/<protocol>/split_manifest.json first.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 3 or manifest.get('protocol') != protocol
            or not manifest.get('official_annotations')
            or not manifest.get('paper_comparable')):
        raise ValueError(f'Invalid canonical OWOD manifest: {path}')
    evaluation = manifest.get('evaluation', {})
    required = ('protocol', 'task_class_ids', 'ap_mode', 'wi_recall',
                'iou_threshold', 'test_split', 'source_paper', 'image_source_ids')
    missing = [key for key in required if key not in evaluation]
    if missing:
        raise ValueError(f'Canonical manifest is missing evaluation fields: {missing}')
    splits = manifest.get('splits')
    if not isinstance(splits, dict):
        raise ValueError(
            f'Canonical manifest has no embedded ordered splits: {path}. '
            'Regenerate the schema-3 manifest before using this benchmark.')
    split_hashes = manifest.get('split_hashes')
    if not isinstance(split_hashes, dict):
        raise ValueError(f'Canonical manifest has no embedded split hashes: {path}')
    for name, values in splits.items():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f'Invalid embedded split {protocol}/{name}')
        if split_hashes.get(name) != split_digest(values):
            raise ValueError(f'Embedded split was modified: {protocol}/{name}')
    if evaluation['test_split'] not in splits:
        raise ValueError(f'Canonical manifest is missing test split {evaluation["test_split"]!r}')
    if splits[evaluation['test_split']] != evaluation['image_source_ids']:
        raise ValueError('Manifest test sequence differs from its embedded split')
    return manifest


def specification(protocol: str) -> dict:
    """Return the author protocol definition recorded in its manifest."""
    evaluation = load_manifest(protocol)['evaluation']
    result = {key: evaluation[key] for key in (
        'protocol', 'task_class_ids', 'ap_mode', 'wi_recall',
        'iou_threshold', 'test_split', 'source_paper')}
    result['task_class_ids'] = [list(task) for task in result['task_class_ids']]
    return result


def official_split(protocol: str, name: str) -> list[str]:
    """Return an exact ordered author split, preserving duplicate entries."""
    splits = load_manifest(protocol)['splits']
    try:
        values = splits[name]
    except KeyError as error:
        raise KeyError(f'Unknown {protocol} author split: {name}') from error
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f'Invalid embedded split {protocol}/{name}')
    return list(values)


def categories() -> list[dict]:
    return [{'id': i, 'name': n} for i, n in zip(COCO_IDS, COCO_NAMES)]
