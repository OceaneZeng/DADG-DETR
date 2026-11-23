"""Screen a single known label per query on frozen Task 1 refinement endpoints."""

from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_aose import (
    Attribution, METRICS, OUTPUT as ATTRIBUTION, SOURCE,
)
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, compare
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-refinement-top-known')
ATTRIBUTION_SHA = '3c8c5816b40926a26060b22a964aed9bd1c7ece938a171424dc73d0897d9ea08'


class TopKnownPerQuery(DiscoveryPostProcess):
    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        known = scores[..., :len(self.known_ids)]
        values, indices = known.max(dim=-1, keepdim=True)
        known.fill_(-torch.inf).scatter_(-1, indices, values)
        return scores, categories, unknown


def prepare():
    result = verified(ATTRIBUTION / 'comparison.json', ATTRIBUTION_SHA)
    if result['status'] != 'complete':
        raise ValueError('Expected the completed A-OSE diagnosis')
    origin = verified(ATTRIBUTION / 'plan.json', result['plan_sha256'])
    training = verified(SOURCE / 'plan.json', origin['source_plan_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                attribution_sha256=ATTRIBUTION_SHA, attribution_plan_sha256=result['plan_sha256'],
                training_plan_sha256=origin['source_plan_sha256'], jobs=origin['jobs'], gates=training['gates'],
                limit=origin['limit'], batch_size=origin['batch_size'], paper_comparable=False,
                factor='Preserve the highest-scoring known category and the original unknown score per query.')


def evaluate(model, manifest, root, annotations, plan, control, reference):
    known_ids = manifest['stages'][0]['classes']
    trace = Attribution(known_ids, manifest['evaluation']['iou_threshold'])
    post = TopKnownPerQuery(known_ids=known_ids, scoring='focal', max_detections=plan['limit'])

    def observe(dataset, _samples, targets, _outputs, detections, _sizes, _post):
        for target, detection in zip(targets, detections, strict=True):
            image = int(target['image_id'])
            trace.add(image, detection, dataset.coco.imgToAnns[image])

    candidate = measure(model, manifest, root, annotations, plan['batch_size'], plan['limit'],
                        observer=observe, postprocessor=post)
    accounting = trace.result()
    if candidate['metrics']['a_ose'] != accounting['a_ose']:
        raise ValueError('Candidate trace does not reconstruct author A-OSE')
    for size, values in candidate['groups'].items():
        if any(values[key] != reference['groups'][size][key] for key in ('gt', 'dense', 'shortlist', 'selected', 'final')):
            raise ValueError('Changed frozen candidate geometry')
    summary = compare(control, candidate, plan['gates'])
    summary.update(metrics={key: candidate['metrics'][key] for key in METRICS}, trace=accounting,
                   deltas_vs_refinement={key: candidate['metrics'][key] - reference['metrics'][key] for key in METRICS})
    return dict(summary=summary, candidate=candidate)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        reports = {arm: verified(Path(entry['report']), entry['report_sha256'])['folds']
                   for arm, entry in job['arms'].items()}
        frozen = job['arms']['refinement']
        checkpoint = Path(frozen['checkpoint'])
        if file_sha256(checkpoint) != frozen['checkpoint_sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, _ = load_detector(None, checkpoint, torch.device('cuda'))
        model.eval()
        protocols[protocol] = {}
        for fold, entry in job['folds'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                annotations = Path(entry['path'])
                if file_sha256(annotations) != entry['sha256']:
                    raise ValueError(f'Changed validation annotations: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold} one known label per query', flush=True)
                data = evaluate(model, manifest, root, annotations, plan,
                                reports['control'][fold], reports['refinement'][fold])
                write_json(dest, dict(plan_sha256=digest, **data))
            data = read_json(dest)
            if data['plan_sha256'] != digest:
                raise ValueError(f'Changed frozen diagnostic plan: {dest}')
            protocols[protocol][fold] = data['summary']
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for folds in protocols.values() for row in folds.values())))


if __name__ == '__main__':
    main()
