"""Measure the objectness factor in VOC ranking with native detection membership fixed."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_old_box_competition import OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-instance-task2-fixed-score-ranking')
SOURCE_HASH = '294545d2a1f62c3e9ad7ab7b838b1b36d2b3ab4d9a1c080f0afe03711ba05b10'


class FixedDetectionClassScores(DiscoveryPostProcess):
    """Select by native focal scores, then rescore just the selected known detections."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        results = super().forward(outputs, target_sizes, include_top_indices=True)
        columns = len(self.known_ids) + int(len(self.known_ids) < 80)
        for i, result in enumerate(results):
            indices = result['top_indices']
            labels = result['labels']
            keep = labels != 91
            queries = indices[keep] // columns
            scores = outputs['pred_logits'][i, queries, labels[keep]].sigmoid()
            objectness = outputs['pred_objectness'][i, queries].sigmoid()
            if not torch.allclose(result['scores'][keep], scores * objectness, atol=1e-7, rtol=1e-6):
                raise ValueError('Selected scores do not reconstruct the native focal product')
            result['scores'][keep] = scores
            if not include_top_indices:
                del result['top_indices']
        return results


def main():
    print('Checking frozen competition endpoints and released holdout', flush=True)
    source = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(source['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if (file_sha256(SOURCE/'comparison.json') != SOURCE_HASH or result['status'] != 'complete'
            or result['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or any(effect['supported'] for effect in result['effects'].values())
            or file_sha256(annotation) != source['annotation_sha256'] or file_sha256(MANIFEST) != source['manifest_sha256']
            or len(order) != 1024 or len(set(order)) != 1024 or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or os.environ['CUDA_VISIBLE_DEVICES'] != source['device'] or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpu'] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed selection study, data or evaluation environment')
    refs = {}
    for arm, entry in source['endpoints'].items():
        row = read_json(SOURCE/(arm+'.json'))
        if (row['plan_sha256'] != result['plan_sha256'] or row['checkpoint_sha256'] != entry['sha256']
                or file_sha256(Path(entry['path'])) != entry['sha256']):
            raise ValueError(f'Changed frozen endpoint: {arm}')
        refs[arm] = row['views']
    plan = {k: source[k] for k in ('annotation', 'annotation_sha256', 'eval_order_sha256', 'manifest_sha256',
                                    'protocol', 'paper_comparable', 'seed', 'increment_sha256', 'endpoints',
                                    'batch_size', 'torch_version', 'device', 'gpu', 'max_detections')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_comparison_sha256=SOURCE_HASH, source_plan_sha256=result['plan_sha256'],
                criterion='Prioritize objectness in known-score ranking only if class-only rescoring of the native '
                          'Top-50 raises Current AP50 by at least 1.00 pp at both frozen endpoints. Keep unknown scores, '
                          'query identities, labels and boxes. Report Previous and Known changes. No inference rule selection.')
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, plan['batch_size'], 2)
    old, current = [s['classes'] for s in manifest['stages'][:2]]
    known = manifest['stages'][1]['active_classes']
    views = {'native': (known, 'focal', 50), 'fixed_class_only': (known, 'focal_class_only', 50)}
    processor = FixedDetectionClassScores(known_ids=known, scoring='focal', max_detections=50).cuda()
    spec = dict(manifest['evaluation'], image_source_ids=order)
    deltas = {}
    for arm, entry in plan['endpoints'].items():
        path = OUTPUT/(arm+'.json')
        if not path.exists():
            print(f'Evaluating {arm} on {len(ds)} images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            values = evaluate_views(model, loader, ds.coco, views, spec, old, current, 'cuda',
                                    postprocessors={'fixed_class_only': processor}, include_previous_stats=True,
                                    include_current_stats=True, print_freq=64)
            write_json(path, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'], views=values))
            del model
        row = read_json(path)
        values = row['views']
        native, fixed = values['native'], values['fixed_class_only']
        if (row['plan_sha256'] != digest or row['checkpoint_sha256'] != entry['sha256']
                or native != refs[arm]['native']
                or any(native['composition'][k] != fixed['composition'][k] for k in ('detections', 'unknown', 'current', 'old'))
                or any(values[k] != refs[arm][k] for k in ('raw_old_box_recall_iou50', 'raw_current_box_recall_iou50'))):
            raise ValueError(f'Changed native detections or raw coverage: {arm}')
        deltas[arm] = {k: fixed['metrics'][k] - native['metrics'][k] for k in ('current_ap50', 'previous_ap50', 'known_ap50')}
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
        deltas=deltas, supported=all(d['current_ap50'] >= .01 for d in deltas.values())))
    print('Fixed detection ranking comparison saved', flush=True)


if __name__ == '__main__':
    main()
