"""Test frozen linear ranks with original per-image per-class detection budgets."""

import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_ffn_target_transfer import MANIFEST, PARENT
from tools.owod.probe_s_task2_frozen_query_readout import OUTPUT as SOURCE, capture_classifier
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-readout-fixed-slots')
SOURCE_SHA256 = '91b87eade6ae65400fbd79ae08e358e867f39faa37a0335f1f662bb1ed27293d'
CRITERION = (
    'On both slices require Current AP50 +1 pp versus native and a positive gain versus shuffled; '
    'Previous AP50 drop at most 1 pp and Known AP50 rises versus native. Require the two-fold mean '
    'proxy U-Recall not to fall, mean WI and summed A-OSE not to rise. Same per-image per-class slots '
    'and inherited scores; old and unknown boxes are preserved.'
)
MODES = {'native': None, 'readout': 'weights', 'shuffled': 'control'}


class FixedSlotReadout(DiscoveryPostProcess):
    def __init__(self, visible, classes, matrix, capture):
        super().__init__(known_ids=visible, max_detections=50, scoring='focal')
        self.classes = classes
        self.matrix = matrix
        self.capture = capture

    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        if include_top_indices:
            raise ValueError('The fixed-slot diagnostic does not report source query indices')
        predictions = super().forward(outputs, target_sizes)
        scores = self.capture(outputs).to(self.matrix.dtype) @ self.matrix
        pixel_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        h, w = target_sizes.unbind(1)
        pixel_boxes = pixel_boxes * torch.stack([w, h, w, h], -1)[:, None]
        for i, prediction in enumerate(predictions):
            labels = prediction['labels']
            for cls in set(labels.tolist()).intersection(self.classes):
                slots = torch.nonzero(labels == cls).flatten()
                order = torch.argsort(scores[i, :, self.classes[cls]], descending=True, stable=True)
                prediction['boxes'][slots] = pixel_boxes[i, order[:len(slots)]]
        return predictions


def prepare():
    source_plan = read_json(SOURCE / 'plan.json')
    result = read_json(SOURCE / 'comparison.json')
    fit = read_json(SOURCE / 'readout.json')
    manifest = read_json(MANIFEST)
    source_hash = file_sha256(SOURCE / 'plan.json')
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (file_sha256(SOURCE / 'comparison.json') != SOURCE_SHA256 or result['supported']
            or result['status'] != 'complete' or result['plan_sha256'] != source_hash
            or fit['plan_sha256'] != source_hash or file_sha256(MANIFEST) != source_plan['manifest_sha256']
            or not device.isdecimal() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source_plan['gpu'] or str(torch.__version__) != source_plan['torch_version']):
        raise ValueError('Changed fitted readout, benchmark or evaluation environment')
    checkpoint = source_plan['checkpoint']
    if file_sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
        raise ValueError('Changed frozen detection model')
    slices = {}
    for name, suffix in (('original', ''), ('independent', '-independent')):
        cohort = source_plan['cohorts'][name]
        path = Path(cohort['annotation'])
        ref_path = PARENT / f'control-finetune{suffix}-heldout.json'
        reference = read_json(ref_path)
        if (file_sha256(path) != cohort['annotation_sha256']
                or reference['checkpoint_sha256'] != checkpoint['sha256']
                or reference['plan_sha256'] != file_sha256(PARENT / 'plan.json')):
            raise ValueError(f'Changed paired evaluation references: {name}')
        slices[name] = dict(annotation=str(path), annotation_sha256=cohort['annotation_sha256'],
                            reference=str(ref_path), reference_sha256=file_sha256(ref_path))
    plan = {key: source_plan[key] for key in ('protocol', 'seed', 'paper_comparable', 'gpu',
                                             'batch_size', 'torch_version', 'manifest_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                device=device, checkpoint=checkpoint, readout_sha256=file_sha256(SOURCE / 'readout.json'),
                source_sha256=SOURCE_SHA256, slices=slices, criterion=CRITERION)
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest, fit['heads']


def evaluate(model, manifest, plan, heads, name, digest):
    path = OUTPUT / f'{name}.json'
    reference = read_json(Path(plan['slices'][name]['reference']))['views']
    old, current = (stage['classes'] for stage in manifest['stages'][:2])
    released = manifest['stages'][1]['active_classes']
    allowed = dict(released=released, proxy_even=old + current[::2], proxy_odd=old + current[1::2])
    if not path.exists():
        ann = Path(plan['slices'][name]['annotation'])
        dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
        order = [image['owod_source_id'] for image in read_json(ann)['images']]
        spec = dict(manifest['evaluation'], image_source_ids=order)
        variants = {f'{mode}/{view}': (ids, 'focal', 50) for mode in MODES for view, ids in allowed.items()}
        fitted = [c for c in current if 'weights' in heads[str(c)]]
        columns = {c: i for i, c in enumerate(fitted)}
        with capture_classifier(model) as capture:
            processors = {}
            for mode, field in MODES.items():
                if field is not None:
                    matrix = torch.tensor([heads[str(c)][field] for c in fitted],
                                          dtype=torch.float64, device='cuda').T.contiguous()
                    for view, ids in allowed.items():
                        processors[f'{mode}/{view}'] = FixedSlotReadout(ids, columns, matrix, capture)
            print(f'Evaluating {name} on {len(order)} images', flush=True)
            views = evaluate_views(model, loader, dataset.coco, variants, spec, old, current, 'cuda',
                                   postprocessors=processors, include_current_stats=True, print_freq=64)
        for view in allowed:
            baseline = views[f'native/{view}']
            if baseline['metrics'] != reference[view]['metrics'] or baseline['composition'] != reference[view]['composition']:
                raise ValueError(f'Native results changed: {name}/{view}')
            for mode in ('readout', 'shuffled'):
                data = views[f'{mode}/{view}']
                if (data['composition'] != baseline['composition']
                        or any(data['metrics'][k] != baseline['metrics'][k]
                               for k in ('previous_ap50', 'unknown_predictions'))):
                    raise ValueError(f'Fixed slots or preserved groups changed: {name}/{mode}/{view}')
        write_json(path, dict(plan_sha256=digest, views=views))
    row = read_json(path)
    if row['plan_sha256'] != digest:
        raise ValueError(f'Changed fixed-slot evaluation: {name}')
    return row['views']


def summarize(views):
    known = {mode: views[f'{mode}/released']['metrics'] for mode in MODES}
    fields = ('current_ap50', 'previous_ap50', 'known_ap50')
    delta = {key: known['readout'][key] - known['native'][key] for key in fields}
    versus_shuffled = known['readout']['current_ap50'] - known['shuffled']['current_ap50']
    open_metrics = {}
    for mode in MODES:
        folds = [views[f'{mode}/{view}']['metrics'] for view in ('proxy_even', 'proxy_odd')]
        open_metrics[mode] = dict(u_recall=mean(f['u_recall'] for f in folds), wi=mean(f['wi'] for f in folds),
                                  a_ose=sum(f['a_ose'] for f in folds))
    baseline, readout = (open_metrics[mode] for mode in ('native', 'readout'))
    pass_known = delta['current_ap50'] >= .01 and delta['previous_ap50'] >= -.01 and delta['known_ap50'] > 0
    pass_open = readout['u_recall'] >= baseline['u_recall'] and readout['wi'] <= baseline['wi'] and readout['a_ose'] <= baseline['a_ose']
    return dict(known=known, delta=delta, current_gain_vs_shuffled=versus_shuffled, open_proxy=open_metrics,
                known_pass=pass_known, open_pass=pass_open, supported=pass_known and pass_open and versus_shuffled > 0)


def main():
    plan, manifest, heads = prepare()
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    slices = {name: summarize(evaluate(model, manifest, plan, heads, name, plan_hash))
              for name in plan['slices']}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
               slices=slices, supported=all(row['supported'] for row in slices.values())))
    print('Fixed-slot readout comparison saved', flush=True)


if __name__ == '__main__':
    main()
