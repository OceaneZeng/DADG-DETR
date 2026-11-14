#!/usr/bin/env python
"""Serialize the authors' released splits into Tree-DETR COCO JSON, without re-splitting."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.benchmark_spec import (ALIASES, NAME_TO_ID, categories,
                                      load_manifest, official_split, specification)
from tools.owod.protocol import build_official_manifest, file_sha256


def image_id(stem):
    return int('2021' + stem.replace('_', ''))


def read_voc(root, stem):
    path = root / 'Annotations' / (stem + '.xml')
    tree = ET.parse(path).getroot()
    size = tree.find('size')
    image = {'id': image_id(stem), 'owod_source_id': stem, 'file_name': stem + '.jpg',
             'width': int(size.findtext('width')), 'height': int(size.findtext('height'))}
    annotations = []
    for object_index, obj in enumerate(tree.findall('object')):
        name = obj.findtext('name')
        name = ALIASES.get(name, name)
        if name not in NAME_TO_ID:
            raise ValueError(f'Unknown class in author annotation {path}: {name}')
        box = [float(obj.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        xmin, ymin, xmax, ymax = box
        annotations.append({'category_id': NAME_TO_ID[name], 'voc_bbox': box,
                            'owod_source_annotation_id': f'{stem}:{object_index}',
                            'bbox': [xmin - 1, ymin - 1, xmax - xmin + 1, ymax - ymin + 1],
                            'difficult': int(obj.findtext('difficult', '0')), 'iscrowd': 0})
    return image, annotations


def load_coco(root):
    images, annotations, source_hashes = {}, defaultdict(list), {}
    for subset in ('train2017', 'val2017'):
        path = root / 'annotations' / ('instances_' + subset + '.json')
        payload = json.loads(path.read_text(encoding='utf-8'))
        source_hashes[str(path.resolve())] = file_sha256(path)
        for image in payload['images']:
            stem = Path(image['file_name']).stem
            if stem in images:
                raise ValueError(f'COCO image occurs in both train and validation: {stem}')
            images[stem] = {**image, 'id': image_id(stem), 'owod_source_id': stem,
                            'file_name': subset + '/' + image['file_name']}
        for ann in payload['annotations']:
            stem = f"{ann['image_id']:012d}"
            x, y, w, h = ann['bbox']
            # Exact conversion in OW-DETR/datasets/coco2voc.py, including crowd
            # annotations (the author's XML converter sets difficult=0 for all).
            box = [int(x + 1), int(y + 1), int(x + w + 1), int(y + h + 1)]
            annotations[stem].append({'category_id': ann['category_id'], 'voc_bbox': box,
                                      'owod_source_annotation_id': ann['id'],
                                      'bbox': [box[0] - 1, box[1] - 1,
                                               box[2] - box[0] + 1, box[3] - box[1] + 1],
                                      'difficult': 0, 'iscrowd': 0})
    return images, annotations, source_hashes


def subset(names, allowed, cache):
    images, annotations = [], []
    for stem in dict.fromkeys(names):
        image, objects = cache[stem]
        images.append(dict(image))
        for obj in objects:
            if obj['category_id'] in allowed:
                ann = {**obj, 'id': len(annotations) + 1, 'image_id': image['id']}
                ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                annotations.append(ann)
    return {'images': images, 'annotations': annotations, 'categories': categories()}


def prepare(protocol, data_root, output, check_images=True):
    spec = specification(protocol)
    names = set(official_split(protocol, spec['test_split']))
    for task in range(1, 5):
        names.update(official_split(protocol, f't{task}_train'))
        if task > 1:
            names.update(official_split(protocol, f't{task}_ft'))
    cache, source_hashes = {}, {}
    image_root = data_root / 'JPEGImages' if protocol == 'm-owodb' else data_root
    missing = []
    if protocol == 'm-owodb':
        for stem in sorted(names):
            annotation = data_root / 'Annotations' / (stem + '.xml')
            if not annotation.is_file() or (check_images and not (image_root / (stem + '.jpg')).is_file()):
                missing.append(stem)
        if missing:
            raise FileNotFoundError(f'M-OWODB author VOC-style release is incomplete: {len(missing)} '
                                    f'images/XML missing; examples={missing[:8]}. '
                                    'COCO alone cannot replace the VOC2007/2012 Task 1 data.')
        digest = __import__('hashlib').sha256()
        for stem in sorted(names):
            cache[stem] = read_voc(data_root, stem)
            digest.update(stem.encode())
            digest.update((data_root / 'Annotations' / (stem + '.xml')).read_bytes())
        source_hashes['author_xml_collection_sha256'] = digest.hexdigest()
    else:
        images, annotations, source_hashes = load_coco(data_root)
        for stem in sorted(names):
            if stem not in images or (check_images and not (data_root / images[stem]['file_name']).is_file()):
                missing.append(stem)
            else:
                cache[stem] = (images[stem], annotations[stem])
        if missing:
            raise FileNotFoundError(f'S-OWODB COCO release is incomplete: {len(missing)} missing; '
                                    f'examples={missing[:8]}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a clean output directory, refusing to replace existing settings: {output}')
    test = official_split(protocol, spec['test_split'])
    active = set()
    for stage, current in enumerate(spec['task_class_ids']):
        active.update(current)
        increment = official_split(protocol, f't{stage + 1}_train')
        finetune = increment if stage == 0 else official_split(protocol, f't{stage + 1}_ft')
        stage_dir = output / f'stage_{stage}'
        stage_dir.mkdir(parents=True, exist_ok=True)
        for filename, selected, labels in [
            ('instances_increment_train2017.json', increment, set(current)),
            ('instances_train2017.json', finetune, active),
            ('instances_val2017.json', test, active),
            ('instances_val2017_full.json', test, set(NAME_TO_ID.values())),
        ]:
            (stage_dir / filename).write_text(json.dumps(subset(selected, labels, cache)) + '\n', encoding='utf-8')
    # Provenance and split hashes are carried by the canonical manifest.
    sources = load_manifest(protocol)['split_source']
    reference = sources['repository'] + '/tree/' + sources['commit']
    manifest = build_official_manifest(output, protocol, reference)
    manifest.update({'image_root': str(image_root.resolve()), 'source_hashes': source_hashes,
                     'images_verified': check_images, 'split_source': sources})
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, choices=('m-owodb', 's-owodb'))
    parser.add_argument('--data-root', required=True, type=Path,
                        help='M: author release with Annotations/JPEGImages; S: COCO2017 root')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--annotations-only', action='store_true',
                        help='Prepare annotations now; training still requires the real images')
    args = parser.parse_args(argv)
    manifest = prepare(args.protocol, args.data_root.resolve(), args.output.resolve(),
                       check_images=not args.annotations_only)
    print(json.dumps(manifest['integrity'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
