"""Cross frozen IoU score and explicit two-view evidence at fixed per-image quotas."""
from pathlib import Path
import subprocess

import torch

from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_labeled_quality as quality
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task1-quality-views'
SLICES = ('reference128', 'holdout1024')


def main():
    parent = quality.OUTPUT
    source = read_json(parent / 'plan.json')
    report = read_json(parent / 'comparison.json')
    source_digest = file_sha256(parent / 'plan.json')
    manifest = read_json(MANIFEST)
    large = holdout.OUTPUT / 'heldout.json'
    if (report['status'] != 'complete' or report['plan_sha256'] != source_digest
            or source['manifest_sha256'] != file_sha256(MANIFEST) or source['holdout_sha256'] != file_sha256(large)
            or source['gpu'] != torch.cuda.get_device_name() or source['torch_version'] != str(torch.__version__)):
        raise ValueError('Completed paired quality study and its original inputs are required')
    artifacts = {}
    for seed, job in source['jobs'].items():
        for key in ('small', 'parent'):
            if file_sha256(Path(job[key])) != job[key + '_sha256']:
                raise ValueError(f'Changed seed {seed} {key}')
        fit_path = parent / f'seed{seed}-fit.json'
        if read_json(fit_path)['plan_sha256'] != source_digest:
            raise ValueError(f'Unexpected calibration for seed {seed}')
        artifacts[fit_path.name] = file_sha256(fit_path)
        for label in SLICES:
            path = parent / f'seed{seed}-{label}.json'
            prev = read_json(path)
            if (prev['plan_sha256'] != source_digest or prev['fit_sha256'] != artifacts[fit_path.name]
                    or prev['counts'] != report['rows'][seed][label]):
                raise ValueError(f'Inconsistent reference for seed {seed}, {label}')
            artifacts[path.name] = file_sha256(path)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=parent.name, source_plan_sha256=source_digest,
                source_result_sha256=file_sha256(parent / 'comparison.json'), artifacts=artifacts,
                jobs=source['jobs'], batch_size=source['batch_size'], gpu=source['gpu'], torch_version=source['torch_version'],
                manifest_sha256=source['manifest_sha256'], holdout_sha256=source['holdout_sha256'],
                paper_comparable=False, rule='Cross geometry versus fitted IoU with versus without the explicit '
                'two-view saliency and consistency multiplier. Same semantic weight, soft pool, quota and NMS.',
                criterion='For each seed fitted_with_views must beat soft and fitted on holdout1024, gain at least '
                '5 pp over original there, and match or beat original on reference128.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    image_root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        fit_row = read_json(parent / f'seed{seed}-fit.json')
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, annotation in zip(SLICES, (Path(job['small']), large)):
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                previous = read_json(parent / path.name)
                _, loader = make_loader(image_root, annotation, plan['batch_size'], 2)
                print(f'Seed {seed}, {label}: evaluating paired rankings', flush=True)
                result = quality.evaluate(model, criterion, ema, loader, fit_row, manifest['stages'][1]['classes'],
                                          previous['images'], view_ablation=True)
                write_json(path, dict(plan_sha256=digest, **result))
            saved = read_json(path)
            if saved['plan_sha256'] != digest or any(
                    saved['counts'][a] != report['rows'][seed][label][a] for a in ('original', 'soft', 'fitted')):
                raise ValueError(f'Unpaired result for seed {seed}, {label}')
            rows[seed][label] = saved['counts']
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        hits = big['fitted_with_views']['hits']
        passed[seed] = (hits > max(big[a]['hits'] for a in ('soft', 'fitted'))
                        and 100 * (hits - big['original']['hits']) >= 5 * big['original']['gt']
                        and small['fitted_with_views']['hits'] >= small['original']['hits'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
