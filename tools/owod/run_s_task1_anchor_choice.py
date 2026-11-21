"""Learn which finest anchor to use from released Task 1 boxes only."""
from collections import Counter
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import run_s_task1_finest_anchor as fixed
from tools.owod import run_s_task1_proposal_stages as stages
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = fixed.OUTPUT.with_name('s-instance-task1-anchor-choice')
FULL, HALF = .05, .025


@torch.inference_mode()
def fit_selector(model, loader, known, plan):
    head = model.detr.transformer.instance_proposals.offset
    dim = head.in_features
    gram = torch.zeros(dim + 2, dim + 2, device='cuda', dtype=torch.float64)
    counts = Counter()
    for step, (samples, targets) in enumerate(loader, 1):
        geo = model(samples.to('cuda'))['geometry']
        masks = geo['masks'][0].flatten(1)
        cells = masks.shape[1]
        features = geo['memory'][:, :cells]
        anchors = geo['anchors'][:, :cells].clone()
        anchors[..., 2:] = HALF
        half = (anchors.clamp(.001, .999).logit() + head(features)).sigmoid()
        for i, target in enumerate(targets):
            if not set(target['labels'].tolist()).issubset(known):
                raise ValueError('Unreleased labels in fit set')
            truth = target['boxes'].cuda()
            counts.update(images=1, gt=len(truth))
            if not len(truth):
                continue
            valid = ~masks[i]
            a = overlap(geo['boxes'][i, :cells][valid], truth)
            b = overlap(half[i, valid], truth)
            best, objects = torch.maximum(a, b).max(1)
            ids = torch.where(best >= plan['min_iou'])[0]
            if not len(ids):
                continue
            objects = objects[ids]
            delta = b[ids, objects] - a[ids, objects]
            weight = objects.bincount(minlength=len(truth))[objects].float().reciprocal()
            x = torch.cat((features[i, valid][ids], torch.ones_like(delta[:, None]), delta[:, None]), dim=1)
            gram += (x.T @ (weight[:, None] * x)).double()
            counts.update(cells=len(ids), represented_gt=objects.unique().numel())
        if step == 1 or step % 256 == 0 or step == len(loader):
            print(f'Fit {step}/{len(loader)} batches, {counts["represented_gt"]} represented GT', flush=True)
    moments = gram.cpu() / gram[-2, -2].item()
    mean = moments[:-2, -2]
    covariance = moments[:-2, :-2] - torch.outer(mean, mean)
    scale = covariance.diag().clamp_min(1e-12).sqrt()
    system = covariance / torch.outer(scale, scale) + plan['ridge'] * torch.eye(dim, dtype=moments.dtype)
    slopes = torch.linalg.solve(system, (moments[:-2, -1] - mean * moments[-2, -1]) / scale) / scale
    intercept = float(moments[-2, -1] - slopes @ mean)
    if not torch.isfinite(slopes).all() or abs(gram[-2, -2].item() - counts['represented_gt']) >= 1:
        raise ValueError('Invalid object-weighted ridge fit')
    return dict(slopes=slopes.tolist(), intercept=intercept, counts=counts)


class AnchorChoice:
    def __init__(self, fit, shuffle, seed):
        self.slopes = torch.tensor(fit['slopes'], device='cuda', dtype=torch.float32)
        self.intercept = fit['intercept']
        self.shuffle = shuffle
        self.rng = torch.Generator(device='cuda').manual_seed(seed)
        self.half_counts = []

    def sides(self, features, mask, xy):
        valid = ~mask.flatten(1)
        half = ((features @ self.slopes + self.intercept) > 0) & valid
        self.half_counts.extend(half.sum(1).tolist())
        if self.shuffle:
            for i, row in enumerate(valid):
                ids = torch.where(row)[0]
                perm = torch.randperm(len(ids), generator=self.rng, device=ids.device)
                half[i, ids] = half[i, ids[perm]]
        return torch.where(half[..., None], xy.new_tensor(HALF), xy.new_tensor(FULL)).expand_as(xy)


def main():
    base = read_json(fixed.OUTPUT / 'plan.json')
    result = read_json(fixed.OUTPUT / 'comparison.json')
    screen = PARENT / 'data/screening.json'
    manifest, files = screening_files(MANIFEST, screen, 0)
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    data = PARENT / 'evaluation/fresh1024.json'
    checkpoint = Path(base['job']['parent'])
    train = read_json(files['increment_train'])
    names = source_ids(train)
    forbidden = set(manifest['evaluation']['image_source_ids']) | set(read_json(PARENT / 'selection/inputs.json')['origins'])
    parent_input = read_json(PARENT / 'off/task1/input.json')
    if (result['status'] != 'complete' or result['follow_up'] or file_sha256(fixed.OUTPUT / 'plan.json') != result['plan_sha256']
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(screen) != parent_input['screen_sha256']
            or file_sha256(checkpoint) != base['job']['parent_sha256'] or file_sha256(data) != base['fresh_sha256']
            or len(names) != len(train['images']) or len(names) != 12288 or names & forbidden
            or {a['category_id'] for a in train['annotations']} != set(known)
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Unexpected endpoint, calibration split or inference environment')
    plan = {k: base[k] for k in ('seed', 'protocol', 'job', 'batch_size', 'gpu', 'torch_version',
                                 'fresh_sha256', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], parent_plan_sha256=result['plan_sha256'],
                parent_result_sha256=file_sha256(fixed.OUTPUT / 'comparison.json'), train_images=len(names),
                train_sha256=file_sha256(files['increment_train']), screen_sha256=file_sha256(screen),
                min_iou=.1, ridge=.01, shuffle_seed=42, sizes=[FULL, HALF],
                selection='Fit the IoU difference (half minus standard) on frozen finest-level features. '
                          'Assign each nearby cell to its best released box over both sizes, unit total weight '
                          'per represented box. Choose half when predicted difference is positive. Compare '
                          'a seeded shuffle within each image at the same count; one anchor per location.',
                criterion='Continue only if small proxy dense coverage gains >=5 pp vs standard, final small '
                          'coverage and small emitted hits both increase vs standard, and total emitted proxy '
                          'and correct known hits do not fall vs standard. Also require strictly more small '
                          'dense coverage and small emitted hits than the count-matched shuffle.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    fit_path = OUTPUT / 'fit.json'
    if not fit_path.exists():
        _, loader = make_loader(root, files['increment_train'], plan['batch_size'], 2)
        write_json(fit_path, dict(plan_sha256=digest, **fit_selector(model, loader, set(known), plan)))
    fit = read_json(fit_path)
    if fit['plan_sha256'] != digest or fit['counts']['images'] != len(names):
        raise ValueError('Incomplete fit on released images')
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    rows, counts = dict(result['rows']), {}
    standard = read_json(fixed.OUTPUT / 'standard.json')
    if standard['plan_sha256'] != result['plan_sha256'] or standard['rows'] != rows['standard']:
        raise ValueError('Changed fixed-anchor reference')
    for arm in ('learned', 'shuffled'):
        path = OUTPUT / f'{arm}.json'
        if not path.exists():
            selector = AnchorChoice(fit, arm == 'shuffled', plan['shuffle_seed'])
            model.detr.transformer.instance_proposals.finest_anchor_sides = selector.sides
            _, loader = make_loader(root, data, plan['batch_size'], 2)
            print(f'Evaluating {arm} anchor choices', flush=True)
            images, totals = stages.evaluate(model, post, loader, known, proxy)
            if len(selector.half_counts) != len(loader.dataset):
                raise ValueError('Incomplete per-image anchor allocation')
            write_json(path, dict(plan_sha256=digest, fit_sha256=file_sha256(fit_path), images=images, rows=totals,
                                  half_counts=dict(zip(loader.dataset.ids, selector.half_counts))))
        saved = read_json(path)
        if (saved['plan_sha256'] != digest or saved['fit_sha256'] != file_sha256(fit_path)
                or saved['images'].keys() != standard['images'].keys() or saved['half_counts'].keys() != saved['images'].keys()):
            raise ValueError(f'Incomplete {arm} result')
        for key, ref in standard['images'].items():
            im = saved['images'][key]
            if im['annotation_ids'] != ref['annotation_ids'] or im['valid_cells'] != ref['valid_cells']:
                raise ValueError(f'Changed paired population on {key}')
        rows[arm], counts[arm] = saved['rows'], saved['half_counts']
    if counts['learned'] != counts['shuffled']:
        raise ValueError('Different number of half-size anchors in paired images')
    a, b, c = [rows[k]['proxy_small'] for k in ('standard', 'learned', 'shuffled')]
    passes = (100 * (b['dense'] - a['dense']) >= 5 * a['gt'] and b['final'] > a['final']
              and b['emitted_best'] > max(a['emitted_best'], c['emitted_best']) and b['dense'] > c['dense'])
    for group in ('known_', 'proxy_'):
        learned = sum(v['emitted_best'] for k, v in rows['learned'].items() if k.startswith(group))
        standard_hits = sum(v['emitted_best'] for k, v in rows['standard'].items() if k.startswith(group))
        passes = passes and learned >= standard_hits
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                                follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
