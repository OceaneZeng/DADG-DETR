"""Restore only Current classifier rows at the frozen Task 2 finetune endpoint."""

from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_fp_open_set import OUTPUT as FULL
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_score_margin import OUTPUT as MARGINS
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-current-head-restore')
MARGIN_RESULT = 'f7db72ce7f691be1999440c4115b943b2313edb035a236b288fb2bbf1bc25eec'
FULL_RESULT = '5626099bf497f1467272789a293eec0e0164a7597bc7a779973b68f326f0ae38'
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')
AP_MARGIN = .005
MIN_RECOVERY = .25


def restore_head(info):
    models = {}
    for phase, endpoint in info['endpoints'].items():
        path = Path(endpoint['path'])
        if file_sha256(path) != endpoint['sha256']:
            raise ValueError(f'Changed frozen endpoint: {path}')
        models[phase], _ = load_detector(None, path, torch.device('cuda'))
    original, restored = (models[phase].detr.class_embed[-1] for phase in ('increment', 'finetune'))
    if not isinstance(original, torch.nn.Linear) or not isinstance(restored, torch.nn.Linear):
        raise ValueError('Expected linear final classification heads')
    classes = torch.tensor(info['current'], device=restored.weight.device)
    with torch.no_grad():
        restored.weight.index_copy_(0, classes, original.weight.index_select(0, classes))
        restored.bias.index_copy_(0, classes, original.bias.index_select(0, classes))
    return models['finetune'].eval()


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    margins = verified(MARGINS / 'comparison.json', MARGIN_RESULT)
    margin_plan = verified(MARGINS / 'plan.json', margins['plan_sha256'])
    baseline = verified(FULL / 'comparison.json', FULL_RESULT)
    full_plan = verified(FULL / 'plan.json', baseline['plan_sha256'])
    source = verified(SOURCE / 'plan.json', margin_plan['parent_plan'])
    if (margins['status'] != 'complete' or baseline['status'] != 'complete'
            or full_plan['parent_plan'] != margin_plan['parent_plan']):
        raise ValueError('Changed matched cohorts or missing completed comparison')
    references = {label: file_sha256(FULL / f'{label}.json') for label in full_plan['reports']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=margin_plan['parent_plan'], margin_result=MARGIN_RESULT, full_result=FULL_RESULT,
                references=references, ap_margin=AP_MARGIN, minimum_recovery=MIN_RECOVERY,
                paper_comparable=False)
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    digest = file_sha256(plan_path)
    results = {}
    for protocol, info in source['protocols'].items():
        model = restore_head(info)
        known = info['previous'] + info['current']
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        results[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            entry = cohort['by_protocol'][protocol]
            annotation = SOURCE / entry['file']
            verified(annotation, entry['sha256'])
            full = full_plan['reports'][label]
            full_path = FULL / full['file']
            if file_sha256(full_path) != full['sha256']:
                raise ValueError(f'Changed full annotations: {label}')
            path = OUTPUT / f'{label}.json'
            if not path.exists():
                print(f'Evaluating {protocol}/{fold} with restored Current classifier rows', flush=True)
                dataset, loader = make_loader(root, annotation, source['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                report = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 source['max_detections'], evaluation_coco=COCO(str(full_path)),
                                 include_perturbed=False)
                write_json(path, dict(plan_sha256=digest, **report))
            report = read_json(path)
            reference = verified(FULL / f'{label}.json', references[label])
            if (report['plan_sha256'] != digest or reference['plan_sha256'] != baseline['plan_sha256']
                    or report['per_image'].keys() != reference['per_image'].keys()):
                raise ValueError(f'Changed matched evaluation: {label}')
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            wanted = margins['protocols'][protocol][fold]['rows']
            targets = {pairs[row['target']]['annotation_ids'][protocol] for row in wanted}
            native, restored = (row['views']['native'] for row in (reference, report))
            if len(targets) != len(wanted) or targets.intersection(native['matched']):
                raise ValueError(f'Changed dropped Current targets: {label}')
            deltas = {key: restored['metrics'][key] - native['metrics'][key] for key in METRICS}
            recovered = sorted(targets.intersection(restored['matched']))
            passed = (len(recovered) >= MIN_RECOVERY * len(targets)
                      and deltas['current_ap50'] >= AP_MARGIN and deltas['previous_ap50'] >= -AP_MARGIN
                      and deltas['known_ap50'] >= 0 and deltas['a_ose'] <= 0)
            results[protocol][fold] = dict(deltas=deltas, targets=len(targets), recovered=recovered, passed=passed)
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=results, paper_comparable=False))


if __name__ == '__main__':
    main()
