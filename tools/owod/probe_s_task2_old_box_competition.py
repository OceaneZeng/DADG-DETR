"""Isolate old/new competition for a shared Top-50 at the matched old-box-loss endpoints."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_finetune_half_lr import OUTPUT as HALF
from tools.owod.screening_data import write_json


WEIGHTED = HALF.with_name('s-instance-task2-finetune-old-box-loss')
OUTPUT = HALF.with_name('s-instance-task2-old-box-competition')
VIEWS = ('native', 'without_current', 'without_previous')


class MaskCandidates(DiscoveryPostProcess):
    """Mask columns before Top-K, keeping every other native query score and index."""

    def __init__(self, known, excluded):
        super().__init__(known_ids=known, max_detections=50, scoring='focal')
        self.register_buffer('excluded', torch.tensor(excluded, dtype=torch.long), persistent=False)

    def candidate_scores(self, outputs):
        scores, classes, unknown = super().candidate_scores(outputs)
        return scores.masked_fill(torch.isin(classes, self.excluded), -torch.inf), classes, unknown


def prepare():
    source = read_json(WEIGHTED/'plan.json')
    study = read_json(WEIGHTED/'comparison.json')
    result_sha = file_sha256(WEIGHTED/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(source['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if (result_sha != '9b18cc3d98031240d54f9daae5ed31e34df07436158b341f2105e0fb42a63554'
            or study['plan_sha256'] != file_sha256(WEIGHTED/'plan.json') or study['follow_up'] or study['old_coverage_pass']
            or file_sha256(annotation) != source['annotation_sha256'] or file_sha256(MANIFEST) != source['identity']['manifest_sha256']
            or len(order) != 1024 or len(set(order)) != len(order) or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpus'][0] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed old-box study, held-out data or frozen evaluation environment')
    inputs, references = {}, {}
    for arm, folder, ref_sha in (
            ('half', HALF, '4212a7809cd7f2f03e61be49aad4c5e158095f1e8625df365691142d5a64e4de'),
            ('weighted', WEIGHTED, 'fcdf03dfff5af87ddfe2f32d541f1b54380a79f7deaaa8414ffc405f7e38a505')):
        checkpoint = folder/'finetune/checkpoint.pth'
        ref = folder/'candidate-finetune-heldout.json'
        row = read_json(ref)
        result = read_json(folder/'comparison.json')
        if (row['plan_sha256'] != result['plan_sha256'] or row['plan_sha256'] != file_sha256(folder/'plan.json')
                or file_sha256(ref) != ref_sha or row['checkpoint_sha256'] != result['checkpoint_sha256']
                or file_sha256(checkpoint) != row['checkpoint_sha256'] or row['views']['released']['metrics'] != result['heldout']
                or read_json(checkpoint.parent/'input.json')['--pretrained'] != source['increment_sha256']
                or read_json(checkpoint.parent/'training_complete.json')['last_epoch'] != 0):
            raise ValueError(f'Changed trained endpoint or native reference: {arm}')
        inputs[arm] = dict(path=str(checkpoint), sha256=row['checkpoint_sha256'], reference=str(ref), reference_sha256=ref_sha)
        references[arm] = row['views']
    deficits = {}
    for group in ('previous', 'current'):
        key = group+'_ap50'
        deficit = references['half']['released']['metrics'][key] - references['weighted']['released']['metrics'][key]
        if deficit <= 0 or deficit != -study['comparisons']['half']['delta'][key]:
            raise ValueError(f'Native {group} gap does not match the paired loss study')
        deficits[group] = deficit

    plan = {k: source[k] for k in ('annotation','annotation_sha256','eval_order_sha256','seed','protocol',
                                    'paper_comparable','torch_version','increment_sha256')}
    plan.update(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256=result_sha, endpoints=inputs, device='0', gpu=torch.cuda.get_device_name(), batch_size=2,
                manifest_sha256=file_sha256(MANIFEST), native_deficits=deficits, scoring='focal', max_detections=50,
                criterion='Primary: excluding current candidates before native Top-50 must close at least half '
                          'the native weighted-minus-half Previous AP50 deficit to support current-class competition. '
                          'Secondary: apply the reciprocal test to Current AP50 excluding previous candidates. Keep '
                          'unknown eligible, all released GT, fixed per-query scores and box geometry. No rule selection.')
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order, references


def main():
    print('Checking both endpoints and the released holdout', flush=True)
    plan, manifest, order, refs = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), plan['batch_size'], 2)
    old, current = [manifest['stages'][i]['classes'] for i in (0, 1)]
    known = manifest['stages'][1]['active_classes']
    variants = {name: (known,'focal',50) for name in VIEWS}
    processors = {v: MaskCandidates(known, excluded).cuda() for v, excluded in
                  [('without_current', current), ('without_previous', old)]}
    spec = dict(manifest['evaluation'], image_source_ids=order)
    cells = {}
    for arm, entry in plan['endpoints'].items():
        path = OUTPUT/(arm+'.json')
        if not path.exists():
            print(f'Evaluating {arm} on {len(ds)} images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            views = evaluate_views(model, loader, ds.coco, variants, spec, old, current, 'cuda', postprocessors=processors,
                                   include_previous_stats=True, include_current_stats=True, print_freq=64)
            write_json(path, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'], views=views))
            del model
        row = read_json(path)
        views = row['views']
        if (row['plan_sha256'] != digest or row['checkpoint_sha256'] != entry['sha256']
                or any(views['native'][k] != refs[arm]['released'][k] for k in ('metrics','composition'))
                or any(views[k] != refs[arm][k] for k in ('raw_old_box_recall_iou50','raw_current_box_recall_iou50'))
                or any(views[v]['composition']['detections'] != 50*len(order) for v in VIEWS)
                or views['without_current']['composition']['current'] or views['without_previous']['composition']['old']):
            raise ValueError(f'Changed native scores, boxes or masked candidate budget: {arm}')
        cells[arm] = {}
        for group, view in [('previous','without_current'), ('current','without_previous')]:
            cells[arm][group] = {}
            for name in ('native', view):
                stats = views[name][group+'_class_stats']
                cells[arm][group][name] = dict(ap50=views[name]['metrics'][group+'_ap50'],
                    **{k: sum(c[k] for c in stats.values()) for k in ('gt','tp','fp')})
    effects = {}
    for group, view in [('previous','without_current'), ('current','without_previous')]:
        baseline = -plan['native_deficits'][group]
        restricted = cells['weighted'][group][view]['ap50'] - cells['half'][group][view]['ap50']
        closure = restricted - baseline
        threshold = -baseline/2
        effects[group] = dict(native_delta=baseline, restricted_delta=restricted, closure=closure,
                              threshold=threshold, supported=closure >= threshold)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              cells=cells, effects=effects))
    print('Inter-class competition comparison saved', flush=True)


if __name__ == '__main__':
    main()
