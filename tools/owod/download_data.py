"""Resume official VOC/COCO/ORE sources and DINO weights, then use manage_data.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
COCO = 'https://s3.amazonaws.com/images.cocodataset.org/'
DINO = 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth'
ORE = 'https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k'
# Official Oxford HTTPS sources and archive MD5s, as listed by torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py
VOC = (
    ('voc2007', 'VOCtrainval_06-Nov-2007.tar', 'c52e279531787c972589f7e41ab4ae64'),
    ('voc2007', 'VOCtest_06-Nov-2007.tar', 'b6e924de25625d8de591ea690078ad9f'),
    ('voc2012', 'VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd'),
)


def download_voc(url, target, expected_md5, proxy=None):
    """Verify release identity; reuse manually downloaded archives without overwriting."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + '.lock', timeout=1):
        record = target.with_suffix(target.suffix + '.source.json')
        if not target.is_file() or record.is_file():
            _download(url, target, proxy)
        digest = hashlib.md5()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_md5:
            raise ValueError(f'VOC archive MD5 mismatch (file preserved): {target}')
        record.write_text(json.dumps({'url': url, 'bytes': target.stat().st_size,
                                     'sha256': sha256(target), 'md5': expected_md5}, indent=2))
    return target


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, proxy=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + '.lock', timeout=1):
        return _download(url, target, proxy)


def _download(url, target, proxy=None):
    record = target.with_suffix(target.suffix + '.source.json')
    if target.is_file() and record.is_file():
        saved = json.loads(record.read_text())
        if saved['url'] == url and saved['sha256'] == sha256(target):
            return target
        raise ValueError(f'Download provenance mismatch: {target}')
    partial = target.with_suffix(target.suffix + '.part')
    partial_record = target.with_suffix(target.suffix + '.part.json')
    if partial_record.is_file():
        pending = json.loads(partial_record.read_text())
        if pending['url'] != url:
            raise ValueError(f'Partial download belongs to another source: {target}')
        if partial.is_file() and partial.stat().st_size == pending['bytes']:
            partial.replace(target)
            record.write_text(json.dumps({'url': url, 'bytes': pending['bytes'],
                                          'sha256': sha256(target)}, indent=2))
            partial_record.unlink()
            return target
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': proxy, 'https': proxy} if proxy else {}))
    offset = partial.stat().st_size if partial.exists() else 0
    failures = 0
    while True:
        requested_offset = offset
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{offset + 8 * 1024 * 1024 - 1}',
                                                           'User-Agent': 'Tree-DETR-data/1.0'})
            with opener.open(request, timeout=30) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status == 206:
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise ValueError(f'Incorrect resumed range: {content_range}')
                    total = int(content_range.rsplit('/', 1)[1])
                elif response.status == 200 and offset == 0:
                    total = int(response.headers['Content-Length'])
                else:
                    raise ValueError('Server refused safe range resume')
                partial_record.write_text(json.dumps({'url': url, 'bytes': total}))
                print(f'{target.name}: {offset}/{total} bytes', flush=True)
                with partial.open('ab') as output:
                    while block := response.read1(256 * 1024):
                        output.write(block)
                        offset += len(block)
            if offset == total:
                break
            if offset > total:
                raise IOError(f'Incorrect response size: {offset}/{total}')
            if offset == requested_offset:
                raise IOError('Download made no progress')
            failures = 0
        except Exception:
            offset = partial.stat().st_size if partial.exists() else 0
            failures = 0 if offset > requested_offset else failures + 1
            if failures >= 3:
                raise
    partial.replace(target)
    record.write_text(json.dumps({'url': url, 'bytes': offset,
                                  'sha256': sha256(target)}, indent=2))
    partial_record.unlink(missing_ok=True)
    return target


def extract(archive, root, prefixes=None):
    root = root.resolve()
    marker = archive.with_suffix(archive.suffix + '.extracted.json')
    identity = {'archive_sha256': sha256(archive), 'root': str(root)}
    if marker.is_file() and json.loads(marker.read_text()) == identity:
        return
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            relative = Path(info.filename.replace('\\', '/'))
            if prefixes is not None:
                parts = relative.parts
                starts = [i for i, part in enumerate(parts) if part in prefixes]
                if not starts:
                    continue
                relative = Path(*parts[starts[0]:])
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise ValueError('Archive member escapes data root')
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # ZIP reads validate the CRC; use temporary paths for interrupted extraction.
            temporary = destination.with_suffix(destination.suffix + '.extracting')
            with source.open(info) as src, temporary.open('wb') as out:
                import shutil
                shutil.copyfileobj(src, out, 1024 * 1024)
            temporary.replace(destination)
    marker.write_text(json.dumps(identity))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('m-owodb', 's-owodb', 'both', 'weights', 'voc'), default='both')
    parser.add_argument('--m-source', choices=('author-release', 'reconstruct'), default='author-release',
                        help='M input: original ORE Drive release (default), or official VOC + COCO archives; '
                             'reconstruct requires prepare_voc.py then prepare_ore.py after download')
    parser.add_argument('--root', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--proxy')
    args = parser.parse_args(argv)
    jobs = [] if args.benchmark == 'voc' else [(DINO, args.weights, None, None)]
    needs_m = args.benchmark in ('m-owodb', 'both')
    reconstruct = needs_m and args.m_source == 'reconstruct'
    if args.benchmark == 'voc' or reconstruct:
        for release, name, md5 in VOC:
            jobs.append((f'https://thor.robots.ox.ac.uk/pascal/VOC/{release}/{name}',
                         args.root / 'archives' / name, None, md5))
    if args.benchmark in ('s-owodb', 'both') or reconstruct:
        for name, folder in [('train2017.zip', 'zips/'), ('val2017.zip', 'zips/'),
                             ('annotations_trainval2017.zip', 'annotations/')]:
            jobs.append((COCO + folder + name, args.root / 'archives' / name, args.root / 'coco', None))
    errors = []
    def work(job):
        url, target, destination, md5 = job
        archive = (download_voc(url, target, md5, args.proxy) if md5
                   else download(url, target, args.proxy))
        if destination:
            extract(archive, destination)
        print(f'Completed: {target}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
                print(f'Download failed: {error}', file=sys.stderr, flush=True)
    if needs_m and not reconstruct:
        try:
            import gdown
            listing = gdown.download_folder(ORE, skip_download=True, proxy=args.proxy, quiet=True)
            selected = [entry for entry in listing if entry.path.lower().endswith('.zip')
                        and any(key in entry.path.lower() for key in ('annotation', 'jpeg', 'image'))]
            if not selected:
                raise RuntimeError('ORE folder has no identifiable author image/annotation ZIPs')
            for entry in selected:
                target = args.root / 'archives' / Path(entry.path).name
                result = gdown.download(id=entry.id, output=str(target), proxy=args.proxy, resume=True)
                if not result:
                    raise RuntimeError(f'ORE download failed: {entry.path}')
                extract(target, args.root / 'ore', ('Annotations', 'JPEGImages'))
        except Exception as error:
            errors.append(f'ORE: {error}')
            print(errors[-1], file=sys.stderr, flush=True)
    if not errors and reconstruct:
        print('Downloads ready. Next run prepare_voc.py --downloads <root>/archives, '
              'then prepare_ore.py --coco-root <root>/coco --ore-root <root>/ore; '
              'finally manage_data.py prepare/verify --benchmark m-owodb. '
              'No training manifest has been generated.', flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
