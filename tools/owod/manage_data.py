#!/usr/bin/env python
"""One data entry point for the pinned author M/S-OWODB protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.benchmark_spec import COCO_IDS, COCO_NAMES, official_split, specification
from tools.owod.prepare_benchmarks import prepare
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files

PROTOCOLS = ('m-owodb', 's-owodb')


def split_plan(protocol):
    spec = specification(protocol)
    tasks, active = [], []
    names = dict(zip(COCO_IDS, COCO_NAMES))
    for task, current in enumerate(spec['task_class_ids'], 1):
        active += current
        increment = official_split(protocol, f't{task}_train')
        finetune = official_split(protocol, f't{task}_ft') if task > 1 else []
        tasks.append({'task': task, 'new_classes': [names[c] for c in current],
                      'new_class_ids': current, 'known_class_ids': list(active),
                      'increment_images': len(increment), 'finetune_images': len(finetune)})
    test = official_split(protocol, spec['test_split'])
    return {'protocol': protocol, 'source_paper': spec['source_paper'],
            'tasks': tasks, 'test_entries': len(test), 'test_unique_images': len(set(test)),
            'split_policy': 'exact author ImageSets; no random split',
            'training_policy': 'current labels for increment; known labels for released finetuning',
            'evaluation_policy': 'full labels; future classes merged into unknown'}


def required_stems(protocol):
    spec = specification(protocol)
    stems = set(official_split(protocol, spec['test_split']))
    for task in range(1, 5):
        stems.update(official_split(protocol, f't{task}_train'))
        if task > 1:
            stems.update(official_split(protocol, f't{task}_ft'))
    return sorted(stems)


def check_source(protocol, source):
    """Check the entire released image coverage without writing derived data."""
    source = Path(source).resolve()
    result = {'protocol': protocol, 'source_root': str(source), 'ready': False,
              'missing_count': 0, 'missing_examples': []}
    stems = required_stems(protocol)
    result['required_unique_images'] = len(stems)
    def missing(path):
        result['missing_count'] += 1
        if len(result['missing_examples']) < 8:
            result['missing_examples'].append(str(path))
    if protocol == 'm-owodb':
        required_dirs = [source / 'Annotations', source / 'JPEGImages']
        absent = [path for path in required_dirs if not path.is_dir()]
        if absent:
            for path in absent:
                missing(path)
            return result
        for stem in stems:
            for path in (source / 'Annotations' / (stem + '.xml'),
                         source / 'JPEGImages' / (stem + '.jpg')):
                if not path.is_file() or not path.stat().st_size:
                    missing(path)
    else:
        required_files = [source / 'annotations' / ('instances_' + subset + '.json')
                          for subset in ('train2017', 'val2017')]
        absent = [path for path in required_files if not path.is_file()]
        if absent:
            for path in absent:
                missing(path)
            return result
        images = {}
        for subset, path in zip(('train2017', 'val2017'), required_files):
            payload = read_json(path)
            for image in payload['images']:
                stem = Path(image['file_name']).stem
                if stem in images:
                    raise ValueError(f'COCO train/val image collision: {stem}')
                images[stem] = source / subset / image['file_name']
        for stem in stems:
            path = images.get(stem)
            if path is None or not path.is_file() or not path.stat().st_size:
                missing(path or ('COCO annotation missing image ' + stem))
    result['ready'] = result['missing_count'] == 0
    return result


def verify_dataset(directory, protocol, source=None, decode_images=False):
    """Validate every phase hash and every image; optionally decode JPEG files."""
    directory = Path(directory).resolve()
    manifest_path = directory / 'split_manifest.json'
    image_paths, tasks = set(), []
    manifest = None
    for stage in range(4):
        manifest, files = stage_files(manifest_path, stage)
        if manifest['protocol'] != protocol:
            raise ValueError(f'Expected {protocol}, received {manifest["protocol"]}')
        expected = None
        if source is not None:
            expected = Path(source).resolve()
            if protocol == 'm-owodb':
                expected /= 'JPEGImages'
        image_root = resolve_image_root(manifest_path, manifest, expected)
        for key in ('increment_train', 'train', 'full_val'):
            payload = read_json(files[key])
            image_paths.update(image_root / image['file_name'] for image in payload['images'])
        tasks.append({'task': stage + 1, 'files_verified': len(files)})
    missing = [str(path) for path in sorted(image_paths)
               if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing/empty images; examples={missing[:8]}')
    if decode_images:
        from PIL import Image
        for path in sorted(image_paths):
            with Image.open(path) as image:
                image.verify()
    # Check raw annotation provenance, not just the derived annotation hashes.
    for name, expected_hash in manifest.get('source_hashes', {}).items():
        if name == 'author_xml_collection_sha256':
            digest = hashlib.sha256()
            for stem in required_stems(protocol):
                digest.update(stem.encode())
                digest.update((image_root.parent / 'Annotations' / (stem + '.xml')).read_bytes())
            actual = digest.hexdigest()
        else:
            source_path = Path(name)
            if not source_path.is_absolute():
                source_path = (manifest_path.parent / source_path).resolve()
            # Prepared manifests may have been created on another machine.
            # When a raw source override is supplied, resolve the two COCO
            # annotation names against that local root instead of trusting an
            # absolute path embedded in the provenance record.
            if source is not None and protocol == 's-owodb':
                source_path = Path(source).resolve() / 'annotations' / source_path.name
            actual = file_sha256(source_path)
        if actual != expected_hash:
            raise ValueError(f'Raw source annotation changed: {name}')
    return {'protocol': protocol, 'ready': True, 'manifest': str(manifest_path),
            'manifest_sha256': file_sha256(manifest_path), 'tasks': tasks,
            'unique_image_files_verified': len(image_paths), 'images_decoded': decode_images}


def prepare_dataset(protocol, source, directory, decode_images=False):
    """Publish only a fully validated dataset; reuse unchanged prepared data."""
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if (directory / 'split_manifest.json').is_file():
        report = verify_dataset(directory, protocol, source, decode_images)
        report['action'] = 'reused'
        return report
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to replace incomplete/old data: {directory}; use a new output root')
    if source == directory or directory in source.parents or source in directory.parents:
        raise ValueError('Raw source and output directories must be separate, without nesting')
    check = check_source(protocol, source)
    if not check['ready']:
        raise FileNotFoundError(json.dumps(check, ensure_ascii=False))
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.prepare-' + protocol + '-', dir=directory.parent))
    if staging.resolve().parent != directory.parent:
        raise ValueError('Temporary preparation directory must stay inside the output parent')
    try:
        prepare(protocol, source, staging)
        verify_dataset(staging, protocol, source, decode_images)
        manifest_path = staging / 'split_manifest.json'
        manifest = read_json(manifest_path)
        # Relative stage paths survive the atomic rename to the final directory.
        manifest['annotation_root'] = str(directory)
        for stage in manifest['stages']:
            stage['files'] = {key: Path(path).relative_to(staging).as_posix()
                              for key, path in stage['files'].items()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if directory.exists():
            directory.rmdir()  # only an empty destination is allowed
        os.rename(staging, directory)
    finally:
        if staging.exists():
            if staging.resolve().parent != directory.parent or not staging.name.startswith('.prepare-'):
                raise ValueError('Refusing to clean a directory outside the preparation workspace')
            shutil.rmtree(staging)  # only the temporary directory created above
    report = verify_dataset(directory, protocol, source, decode_images=False)
    report['images_decoded'] = decode_images
    report['action'] = 'prepared'
    return report


def prepare_fit_data(manifest_path, screening_path, output, images=160, seed=42):
    """Prepare disposable training-only overfit data; never register a new protocol."""
    from tools.owod.screening_data import screening_files, select_images, subset, write_json
    manifest, files = screening_files(Path(manifest_path), Path(screening_path), 1)
    screen = read_json(Path(screening_path))
    if seed != screen['seed']:
        raise ValueError('Tiny-fit seed must match the parent screening data')
    source = read_json(files['increment_train'])
    classes = manifest['stages'][1]['classes']
    names = select_images(source, images, classes, seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty tiny-fit data directory')
    path = output / 'train.json'
    write_json(path, subset(source, names, classes))
    write_json(output / 'diagnostic_data.json', {
        'kind': 'tiny_fit_train_only', 'paper_comparable': False,
        'parent_manifest_sha256': file_sha256(Path(manifest_path)),
        'screening_sha256': file_sha256(Path(screening_path)),
        'source_annotation_sha256': file_sha256(files['increment_train']),
        'annotation_sha256': file_sha256(path), 'seed': seed, 'images': images,
        'source_ids': sorted(names), 'evaluation': 'training fit only; not generalization'})
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'check', 'prepare', 'verify', 'prepare-screen', 'verify-screen'))
    parser.add_argument('--manifest', type=Path, help='Parent schema-3 manifest for diagnostic screening')
    parser.add_argument('--screen-output', type=Path, help='Separate generated diagnostic data directory')
    parser.add_argument('--train-images', type=int, default=2000)
    parser.add_argument('--val-images', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark', choices=(*PROTOCOLS, 'both'), default='both')
    parser.add_argument('--config', type=Path, help='JSON paths, relative to the project root')
    parser.add_argument('--ore-root', type=Path, help='Original ORE release (Annotations/JPEGImages)')
    parser.add_argument('--coco-root', type=Path, help='COCO2017 (annotations/train2017/val2017)')
    parser.add_argument('--output-root', type=Path, help='Default: data/owod; one subdirectory per benchmark')
    parser.add_argument('--decode-images', action='store_true', help='Additionally verify image decoding')
    parser.add_argument('--report', type=Path, help='Save the JSON result as well as printing it')
    args = parser.parse_args(argv)
    if args.command in ('prepare-screen', 'verify-screen'):
        if not args.manifest or not args.screen_output:
            parser.error('Screen data requires --manifest and --screen-output')
        from tools.owod.screening_data import prepare_screen, screening_files
        if args.command == 'prepare-screen':
            manifest = read_json(args.manifest)
            verification = verify_dataset(args.manifest.resolve().parent, manifest['protocol'])
            if verification.get('ready') is False:
                raise ValueError('Parent dataset verification failed')
            path = prepare_screen(args.manifest, args.screen_output, args.train_images, args.val_images, args.seed)
        else:
            path = args.screen_output / 'screening.json'
        for stage in range(2):
            screening_files(args.manifest, path, stage)
        print(json.dumps({'ready': True, 'paper_comparable': False, 'screening': str(path)}))
        return 0
    config = read_json(args.config.resolve()) if args.config else {}
    def root(value):
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    sources = {'m-owodb': root(args.ore_root or config.get('ore_root', 'data/raw/ore')),
               's-owodb': root(args.coco_root or config.get('coco_root', 'data/raw/coco'))}
    output = root(args.output_root or config.get('output_root', 'data/owod'))
    protocols = PROTOCOLS if args.benchmark == 'both' else (args.benchmark,)
    reports = []
    failed = False
    for protocol in protocols:
        try:
            if args.command == 'plan':
                report = split_plan(protocol)
            elif args.command == 'check':
                report = check_source(protocol, sources[protocol])
            elif args.command == 'prepare':
                report = prepare_dataset(protocol, sources[protocol], output / protocol, args.decode_images)
            else:
                report = verify_dataset(output / protocol, protocol, sources[protocol], args.decode_images)
            failed |= report.get('ready') is False
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            report = {'protocol': protocol, 'ready': False, 'error': str(error)}
            failed = True
        reports.append(report)
    result = {'command': args.command, 'output_root': str(output), 'benchmarks': reports}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n', encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
