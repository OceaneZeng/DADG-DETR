"""Compare native boxes with a fixed blend on active encoder queries."""

from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_shortlist_replacement import TRACE_OUTPUT
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task2_m_s_previous_hard_negatives import FULL, GATES, MANIFESTS, METRICS, OUTPUT as TRAIN, SOURCE, compare
from tools.owod.screening_data import write_json


OUTPUT = TRAIN.with_name('task2-m-s-native-box-blend')
TRACE_SHA = 'afd7d7bbeb1bd32b061db74594cd25b92d405ee2de3e575512f5f1ac759bb484'
INPUT_SHA = '9b44fe67f57833d067c1534381695ee09cd93ed98691cf123ba942d518fa7c83'
ENCODER_WEIGHT = .5


class NativeBoxBlend(DiscoveryPostProcess):
    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        geometry = outputs['geometry']
        chosen, active = geometry['chosen'], geometry['active']
        quota = chosen.shape[1]
        encoder = geometry['boxes'].gather(1, chosen[..., None].expand(-1, -1, 4))
        final = outputs['pred_boxes']
        boxes = torch.where(active[..., None], torch.lerp(final[:, :quota], encoder, ENCODER_WEIGHT),
                            final[:, :quota])
        blended = dict(outputs, pred_boxes=torch.cat((boxes, final[:, quota:]), dim=1))
        return super().forward(blended, target_sizes, include_top_indices=include_top_indices)


def prepare():
    trace = verified(TRACE_OUTPUT / 'comparison.json', TRACE_SHA)
    source = verified(TRAIN / 'plan.json', INPUT_SHA)
    if not trace['first_layer_majority'] or trace['headroom_supported']:
        raise ValueError('Expected the completed frozen query trace')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    jobs = {}
    for protocol, entry in source['jobs'].items():
        checkpoint = entry['endpoints']['increment']
        if file_sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError(f'Changed increment checkpoint: {protocol}')
        jobs[protocol] = {k: entry[k] for k in ('folds', 'previous', 'current', 'manifest_sha256')}
        jobs[protocol]['checkpoint'] = checkpoint
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                trace_sha256=TRACE_SHA, source_plan_sha256=INPUT_SHA, jobs=jobs, batch=2,
                max_detections=source['max_detections'], encoder_weight=ENCODER_WEIGHT, gates=GATES, paper_comparable=False,
                scoring='native focal scores with original labels and Top-50 selection',
                decision='Every fold uses the previous/current/known AP50 and current matched-count gates from the paired training comparison.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        image_root = resolve_image_root(path, manifest)
        model, _ = load_detector(None, Path(job['checkpoint']['path']), torch.device('cuda'))
        model.eval()
        previous, current = job['previous'], job['current']
        limit = plan['max_detections']
        blended = NativeBoxBlend(previous + current, max_detections=limit, scoring='focal')
        results[protocol] = {}
        for fold, cohort in job['folds'].items():
            marker = OUTPUT / f'{protocol}-{fold}.json'
            if not marker.exists():
                print(f'Evaluating {protocol}/{fold} native and blended boxes', flush=True)
                annotation = SOURCE / cohort['released']['file']
                scored = FULL / cohort['scored']['file']
                if file_sha256(annotation) != cohort['released']['sha256'] or file_sha256(scored) != cohort['scored']['sha256']:
                    raise ValueError(f'Changed validation annotations: {protocol}/{fold}')
                dataset, loader = make_loader(image_root, annotation, plan['batch'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                report = measure(model, dataset, loader, previous + current, previous, current, spec, limit,
                                 evaluation_coco=COCO(str(scored)), extra_posts={'blended': blended},
                                 include_perturbed=False, audit_previous_matches=True)
                if any(item['native'] != item['blended'] for item in report['per_image'].values()):
                    raise ValueError(f'Changed the native output budget: {protocol}/{fold}')
                arms = {name: dict(views={'native': report['views'][name]},
                                   previous_matches={'native': report['previous_matches'][name]})
                        for name in ('native', 'blended')}
                summary = compare(arms['native'], arms['blended'])
                summary['metrics'] = {arm: {key: view['metrics'][key] for key in METRICS}
                                      for arm, view in report['views'].items()}
                write_json(marker, dict(plan_sha256=digest, summary=summary, report=report))
            row = read_json(marker)
            if row['plan_sha256'] != digest:
                raise ValueError(f'Changed experiment report: {protocol}/{fold}')
            results[protocol][fold] = row['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results,
               follow_up=all(item['passed'] for folds in results.values() for item in folds.values())))


if __name__ == '__main__':
    main()
