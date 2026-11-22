"""Test same-class refill for Current detections sharing a query with a stronger Previous output."""
import os
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_quality_query_count import OUTPUT as SOURCE
from tools.owod.probe_s_task2_shortlist_depth import OUTPUT as NATIVE
from tools.owod.probe_task2_dedup_open_set import protected_predictions
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import paired_gates
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-shared-query-refill')
CRITERION = ('On each slice reduce Current false positives assigned to Previous classes by at least 25% '
             'on the full released view and increase Current AP50 by at least 1 pp. Known must improve. '
             'Require unchanged Previous and proxy U-Recall, no rise in mean proxy WI or summed A-OSE. '
             'Preserve each image/class quota and the native Previous and unknown query IDs, boxes and scores.')


class SharedQueryRefill(DiscoveryPostProcess):
    def __init__(self, known, previous):
        super().__init__(known_ids=known, scoring='focal', max_detections=50)
        self.previous = previous
        self.checked = 0
        self.changed_images = 0
        self.replacements = 0

    @torch.no_grad()
    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        native = super().forward(outputs, target_sizes, include_top_indices=True)
        scores, labels, unknown = self.candidate_scores(outputs)
        positions = torch.stack([row['top_indices'] for row in native])
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.flatten(1).scatter_(1, positions, True)
        previous = torch.isin(labels, labels.new_tensor(self.previous))
        current = ~previous & (labels != 91)
        old_max = scores.masked_fill(~(selected & previous), -torch.inf).amax(dim=2, keepdim=True)
        conflict = current & (scores < old_max)
        removed = selected & conflict
        need = removed.sum(dim=1, keepdim=True)
        eligible = ~selected & current & ~conflict
        if (eligible.sum(dim=1, keepdim=True) < need).any():
            raise ValueError('Insufficient candidates for same-class refill')
        order = scores.masked_fill(~eligible, -torch.inf).argsort(dim=1, descending=True, stable=True)
        first = torch.arange(scores.shape[1], device=scores.device)[None, :, None] < need
        added = torch.zeros_like(selected).scatter_(1, order, first.expand_as(order))
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        height, width = target_sizes.unbind(1)
        boxes = boxes * torch.stack((width, height, width, height), 1)[:, None]
        results = []
        for i, ids in enumerate(positions):
            kept = ids[~removed[i].flatten()[ids]]
            promoted = torch.where(added[i].flatten())[0]
            # Stable ordering preserves the native sequence for tied retained scores.
            merged = torch.cat((kept, promoted))
            ids = merged[scores[i].flatten()[merged].argsort(descending=True, stable=True)]
            queries = ids // len(labels)
            u = unknown[i, queries]
            row = dict(top_indices=ids, scores=scores[i].flatten()[ids], labels=labels[ids % len(labels)],
                       boxes=boxes[i, queries], unknown_scores=u, unknown_mask=u >= self.unknown_threshold)
            if not torch.equal(row['labels'].sort().values, native[i]['labels'].sort().values):
                raise ValueError('Per-class budget changed')
            self.replacements += len(promoted)
            self.changed_images += bool(len(promoted))
            results.append(row)
        before = protected_predictions(native, [*self.previous, 91])
        after = protected_predictions(results, [*self.previous, 91])
        if not all(torch.equal(a, b) for a, b in zip(before, after)):
            raise ValueError('Protected Previous or unknown predictions changed')
        self.checked += len(native)
        return results


def prepare():
    src = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != '2341ef4173905bc100b08598f1953ba34bfa83a1605588f0bb458d1e339738de'
            or result['status'] != 'complete' or result['count_supported'] or result['follow_up']
            or file_sha256(SOURCE/'plan.json') != result['plan_sha256'] or file_sha256(MANIFEST) != src['manifest_sha256']
            or file_sha256(Path(src['checkpoint']['path'])) != src['checkpoint']['sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed diagnostic evidence, protocol, original checkpoint or runtime')
    for name, entry in src['slices'].items():
        if (file_sha256(Path(entry['annotation'])) != entry['annotation_sha256']
                or file_sha256(NATIVE/f'{name}-native.json') != src['native_sha256'][name]):
            raise ValueError(f'Changed released holdout or native reference: {name}')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version', 'native_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=file_sha256(SOURCE/'comparison.json'), criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan


def previous_confusions(view, previous):
    return sum(n for stats in view['current_class_stats'].values()
               for cls, n in stats['fp_sources']['confused_with'].items() if int(cls) in previous)


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    manifest = read_json(MANIFEST)
    prev, cur = (s['classes'] for s in manifest['stages'][:2])
    folds = dict(released=manifest['stages'][1]['active_classes'], proxy_even=prev+cur[::2], proxy_odd=prev+cur[1::2])
    variants = {f'{fold}_{arm}': (known, 'focal', 50) for fold, known in folds.items() for arm in ('native', 'refill')}
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    summaries = {}
    for name, entry in plan['slices'].items():
        path = OUTPUT/f'{name}.json'
        if not path.exists():
            ann = Path(entry['annotation'])
            ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
            order = [im['owod_source_id'] for im in read_json(ann)['images']]
            processors = {f'{fold}_refill': SharedQueryRefill(known, prev) for fold, known in folds.items()}
            spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_fp_sources=True)
            print(f'Evaluating {name}, native and same-class refill on {len(ds)} images', flush=True)
            views = evaluate_views(model, loader, ds.coco, variants, spec, prev, cur, 'cuda',
                                   include_current_stats=True, postprocessors=processors, print_freq=64)
            audits = {k: dict(images=m.checked, changed=m.changed_images, replacements=m.replacements)
                      for k, m in processors.items()}
            write_json(path, dict(plan_sha256=digest, views=views, audits=audits))
        data = read_json(path)
        if data['plan_sha256'] != digest or any(a['images'] != 1024 for a in data['audits'].values()):
            raise ValueError(f'Incomplete per-image audit on {name}')
        reference = read_json(NATIVE/f'{name}-native.json')['views']
        views = data['views']
        native = {f: views[f'{f}_native'] for f in folds}
        refill = {f: views[f'{f}_refill'] for f in folds}
        for f, view in native.items():
            stats = {k: {m: x for m, x in v.items() if m != 'fp_sources'} for k, v in view['current_class_stats'].items()}
            if (dict(view, current_class_stats=stats) != reference[f]
                    or view['metrics']['previous_ap50'] != refill[f]['metrics']['previous_ap50']
                    or view['metrics'].get('u_recall') != refill[f]['metrics'].get('u_recall')):
                raise ValueError(f'Native replay or protected metrics changed on {name}/{f}')
        before, after = (previous_confusions(v['released'], prev) for v in (native, refill))
        delta = {k: refill['released']['metrics'][k]-native['released']['metrics'][k]
                 for k in ('previous_ap50', 'current_ap50', 'known_ap50')}
        costs, known_pass, open_pass = paired_gates(refill, native, delta)
        confusion_pass = before > 0 and 4*after <= 3*before
        summaries[name] = dict(previous_confusions=[before, after], delta=delta, proxy_deltas=costs,
                               confusion_pass=confusion_pass, known_pass=known_pass, open_pass=open_pass,
                               follow_up=confusion_pass and known_pass and open_pass)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                               slices=summaries, follow_up=all(s['follow_up'] for s in summaries.values())))
    print('Shared-query refill comparison saved', flush=True)


if __name__ == '__main__':
    main()
