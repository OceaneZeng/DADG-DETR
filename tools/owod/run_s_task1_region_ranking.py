"""Rank fixed selective-search regions using only released Task 1 box supervision."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import subprocess

import cv2
import numpy as np

from tools.owod.prepare_cat_proposals import selective_search
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_fraction import MANIFEST
from tools.owod.run_s_task1_region_geometry import OUTPUT as PARENT
from tools.owod.run_s_task1_released_seed import OUTPUT as RELEASED
from tools.owod.screening_data import write_json


OUTPUT = PARENT.with_name('s-instance-task1-region-ranking')
FEATURES = ['log_area', 'log_aspect', 'absolute_log_aspect', 'mean_border_jump',
            'minimum_border_jump', 'border_minus_interior', 'inside_ring_contrast']
GROUP = 64


def prepare():
    parent = read_json(PARENT / 'plan.json')
    result = read_json(PARENT / 'comparison.json')
    released = read_json(RELEASED / 'plan.json')
    data_path = RELEASED / 'released2048.json'
    if (result['status'] != 'complete' or result['follow_up'] or
            result['plan_sha256'] != file_sha256(PARENT / 'plan.json') or
            file_sha256(MANIFEST) != released['manifest_sha256'] or
            released['manifest_sha256'] != parent['manifest_sha256'] or
            file_sha256(data_path) != released['data_sha256'] or cv2.__version__ != parent['opencv']):
        raise ValueError('Incompatible region study, released slice or OpenCV')
    data = read_json(data_path)
    manifest = read_json(MANIFEST)
    ids = {im['owod_source_id'] for im in data['images']}
    allowed = set(manifest['splits']['t1_train']) - set(manifest['evaluation']['image_source_ids'])
    if (len(ids) != 2048 or not ids <= allowed or len(data['images']) != 2048 or
            {a['category_id'] for a in data['annotations']} - set(manifest['stages'][0]['classes'])):
        raise ValueError('Expected 2048 released-only Task 1 training-source images')
    imgs = sorted(data['images'], key=lambda im: sha256(('52:' + im['owod_source_id']).encode()).digest())
    plan = dict(protocol='s-owodb', paper_comparable=False, selection_seed=52, region=parent['region'],
                opencv=cv2.__version__, numpy=np.__version__, workers=4, features=FEATURES,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan_sha256=result['plan_sha256'], parent_result_sha256=file_sha256(PARENT / 'comparison.json'),
                released_plan_sha256=file_sha256(RELEASED / 'plan.json'), data_sha256=released['data_sha256'],
                manifest_sha256=parent['manifest_sha256'], split=[im['owod_source_id'] for im in imgs],
                fit_images=1024, validation_images=1024, limit=25, pool=100, penalty=.01,
                rule='First 1024 hash-ordered images fit equal-class-weight ridge on released small boxes. '
                     'Positive if max IoU >=.5, local nonmatch if .1 <= max IoU <.5, other regions ignored in fitting. '
                     'On the next 1024 images rank the entire region pool before scoring hits. '
                     'Compare engine order, seeded shuffle, shape only, shape plus pixel evidence, and full pool.',
                criterion='Advance only to a fresh proxy evaluation if the validation pool covers >=50 small known '
                          'identities and pixel top25 covers >=60% of pool-covered identities, >=15 pp more '
                          'than each of engine and shuffle, and >=5 pp more than shape alone (same denominator).')
    pin(OUTPUT / 'plan.json', plan)
    return plan, imgs, data['annotations'], resolve_image_root(MANIFEST, manifest)


def pixel_features(photo, boxes):
    if not len(boxes):
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    lab = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    dx = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
    dy = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
    dx = np.pad(dx, ((0, 0), (1, 0)))
    dy = np.pad(dy, ((1, 0), (0, 0)))
    col = np.pad(dx.cumsum(axis=0), ((1, 0), (0, 1)))
    row = np.pad(dy.cumsum(axis=1), ((0, 1), (1, 0)))
    x, y, bw, bh = boxes.T
    x2, y2 = x + bw, y + bh
    borders = np.stack([(col[y2, x] - col[y, x]) / bh, (col[y2, x2] - col[y, x2]) / bh,
                        (row[y, x2] - row[y, x]) / bw, (row[y2, x2] - row[y2, x]) / bw])
    def rect(table, left, top, right, bottom):
        return table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]
    area = bw * bh
    energy = cv2.integral(.5 * (dx + dy), sdepth=cv2.CV_64F)
    interior = rect(energy, x, y, x2, y2) / area
    colors = cv2.integral(lab, sdepth=cv2.CV_64F)
    inside = rect(colors, x, y, x2, y2)
    l, t, r, b = np.maximum(x-2, 0), np.maximum(y-2, 0), np.minimum(x2+2, w), np.minimum(y2+2, h)
    ring_area = (r-l)*(b-t) - area
    ring_mean = (rect(colors, l, t, r, b) - inside) / np.maximum(1, ring_area[:, None])
    contrast = np.linalg.norm(inside / area[:, None] - ring_mean, axis=1)
    contrast[ring_area == 0] = 0.
    aspect = np.log(bw / bh)
    return np.column_stack((np.log(area), aspect, np.abs(aspect), borders.mean(0), borders.min(0),
                            borders.mean(0) - interior, contrast)).astype(np.float32)


def describe(im, root, plan):
    key = str(im['id'])
    path = root / im['file_name']
    cv2.setRNGSeed(int(key) % 2147483647)
    settings = plan['region']
    raw = selective_search(path, settings['mode'], settings['max_proposals'], settings['min_area'])
    boxes = np.asarray([q for q in raw if q[2]*q[3] <= settings['small_area_cap']][:plan['pool']],
                       dtype=np.int32).reshape(-1, 4)
    photo = cv2.imread(str(path))
    if photo is None or photo.shape[:2] != (im['height'], im['width']):
        raise ValueError(f'Changed image {key}')
    return key, np.column_stack((boxes, pixel_features(photo, boxes))).astype(np.float32)


def features(imgs, root, plan):
    digest = file_sha256(OUTPUT / 'plan.json')
    directory = OUTPUT / 'features' / digest
    directory.mkdir(parents=True, exist_ok=True)
    all_rows, chunks = {}, {}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        for i in range(0, len(imgs), GROUP):
            batch = imgs[i:i+GROUP]
            dest = directory / f'{i:04d}.npz'
            if not dest.is_file():
                rows = dict(pool.map(lambda im: describe(im, root, plan), batch))
                tmp = dest.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, **rows)
                tmp.replace(dest)
            with np.load(dest, allow_pickle=False) as z:
                if set(z.files) != {str(im['id']) for im in batch}:
                    raise ValueError('Mismatched image IDs in feature checkpoint')
                all_rows.update({k: z[k] for k in z.files})
            chunks[dest.relative_to(OUTPUT).as_posix()] = file_sha256(dest)
            print(f'{datetime.now().isoformat(timespec="seconds")} regions {len(all_rows)}/{len(imgs)} images', flush=True)
    return all_rows, chunks


def targets(im, annotations):
    truth, ids = [], []
    for ann in annotations:
        if ann.get('iscrowd') or ann['area'] >= 32**2:
            continue
        x, y, w, h = ann['bbox']
        box = [max(0, x), max(0, y), min(im['width'], x+w), min(im['height'], y+h)]
        if box[2] > box[0] and box[3] > box[1]:
            truth.append(box)
            ids.append(ann['id'])
    return np.asarray(truth, dtype=np.float64).reshape(-1, 4), np.asarray(ids, dtype=np.int64)


def ious(boxes, truth):
    p = boxes[:, :2]
    q = p + boxes[:, 2:]
    inter = np.maximum(0., np.minimum(q[:, None], truth[:, 2:]) - np.maximum(p[:, None], truth[:, :2])).prod(2)
    return inter / (boxes[:, 2:].prod(1)[:, None] + (truth[:, 2:] - truth[:, :2]).prod(1) - inter)


def fit(imgs, rows, groups, penalty):
    features, labels = [], []
    for im in imgs:
        table = rows[str(im['id'])]
        truth, _ = targets(im, groups[im['id']])
        if not len(truth) or not len(table):
            continue
        best = ious(table[:, :4], truth).max(1)
        keep = best >= .1
        features.append(table[keep, 4:])
        labels.append(best[keep] >= .5)
    x, y = np.concatenate(features).astype(np.float64), np.concatenate(labels)
    if not y.any() or y.all():
        raise ValueError('Missing positive or local nonmatching training regions')
    weights = np.where(y, .5 / y.sum(), .5 / (~y).sum())
    mu = weights @ x
    scale = np.sqrt(np.maximum(weights @ ((x-mu)**2), 1.e-8))
    models = {}
    for name, dim in [('shape', 3), ('pixel', len(FEATURES))]:
        z = ((x-mu)/scale)[:, :dim]
        gram = z.T @ (weights[:, None] * z) + penalty * np.eye(dim)
        coef = np.linalg.solve(gram, z.T @ (weights * (y.astype(float) - .5)))
        models[name] = dict(mean=mu[:dim].tolist(), scale=scale[:dim].tolist(), coef=coef.tolist())
    return dict(positive=int(y.sum()), local_nonmatch=int((~y).sum()), models=models)


def validate(imgs, rows, groups, trained, limit):
    totals, records = Counter(), {}
    for im in imgs:
        key = str(im['id'])
        table = rows[key]
        ids = np.arange(len(table))
        arms = dict(pool=ids, engine=ids[:limit], shuffle=np.random.default_rng(int(key) + 52).permutation(ids)[:limit])
        for name, model in trained['models'].items():
            f = table[:, 4:4+len(model['coef'])]
            scores = ((f-model['mean'])/model['scale']) @ model['coef']
            arms[name] = np.argsort(-scores, kind='stable')[:limit]
        truth, ann_ids = targets(im, groups[im['id']])
        hit = ious(table[:, :4], truth) >= .5
        hits = {name: ann_ids[hit[indices].any(0)].tolist() for name, indices in arms.items()}
        totals.update({name: len(value) for name, value in hits.items()})
        totals['gt'] += len(ann_ids)
        records[key] = dict(gt=ann_ids.tolist(), hits=hits, chosen={k: v.tolist() for k, v in arms.items() if k != 'pool'})
    return totals, records


def main():
    plan, imgs, anns, root = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    print(f'{datetime.now().isoformat(timespec="seconds")} extracting released-image regions for {len(imgs)} images', flush=True)
    rows, chunks = features(imgs, root, plan)
    grouped = defaultdict(list)
    for ann in anns:
        grouped[ann['image_id']].append(ann)
    split = plan['fit_images']
    learned = fit(imgs[:split], rows, grouped, plan['penalty'])
    write_json(OUTPUT / 'model.json', dict(plan_sha256=digest, **learned))
    totals, records = validate(imgs[split:], rows, grouped, learned, plan['limit'])
    n = totals['pool']
    passed = (n >= 50 and 100*totals['pixel'] >= 60*n and
              100*(totals['pixel'] - max(totals['engine'], totals['shuffle'])) >= 15*n and
              100*(totals['pixel'] - totals['shape']) >= 5*n)
    write_json(OUTPUT / 'paired.json', dict(plan_sha256=digest, images=records))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, paper_comparable=False,
               rows=totals, feature_chunks=chunks, model_sha256=file_sha256(OUTPUT / 'model.json'),
               paired_sha256=file_sha256(OUTPUT / 'paired.json'), follow_up=bool(passed)))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
