"""Transfer the locked S Task 2 old-class refill rule to four M endpoints."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.probe_task2_duration_holdout import OUTPUT as DURATION_HOLDOUT
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.run_s_task2_old_dedup import DEDUP_IOU, LIMIT, OLD_POOL, evaluate_endpoints, passes_refill_gate
from tools.owod.run_s_task2_old_dedup_holdout import OUTPUT as S_HOLDOUT
from tools.owod.screening_data import write_json


OUTPUT = DURATION_HOLDOUT.with_name('m-instance-task2-old-dedup')
SEEDS = (42, 43)


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for frozen inference')

    print('Verifying the locked S rule and four M Task 2 checkpoints', flush=True)
    s_plan = read_json(S_HOLDOUT / 'plan.json')
    s_result = read_json(S_HOLDOUT / 'comparison.json')
    s_plan_hash = file_sha256(S_HOLDOUT / 'plan.json')
    if (s_plan['protocol'] != 's-owodb' or s_result['status'] != 'complete' or not s_result['follow_up'] or
            s_result['plan_sha256'] != s_plan_hash or
            (s_plan['max_detections'], s_plan['old_pool'], s_plan['nms_iou']) != (LIMIT, OLD_POOL, DEDUP_IOU)):
        raise ValueError('Missing validation of the locked S refill rule')

    source = read_json(DURATION_HOLDOUT / 'plan.json')
    reference = read_json(DURATION_HOLDOUT / 'comparison.json')
    annotation = DURATION_HOLDOUT / 'heldout.json'
    manifest, holdout = read_json(MANIFEST), read_json(annotation)
    source_hash = file_sha256(DURATION_HOLDOUT / 'plan.json')
    manifest_hash, annotation_hash = file_sha256(MANIFEST), file_sha256(annotation)
    specification = manifest['evaluation']
    order = [image['owod_source_id'] for image in holdout['images']]
    checkpoints = source['checkpoints']
    arms = {f'seed{seed}-{epochs}epochs' for seed in SEEDS for epochs in (3, 6)}
    if (reference['status'] != 'complete' or reference['plan_sha256'] != source_hash or
            set(reference['rows']) != arms or set(checkpoints) != arms or manifest_hash != source['manifest_sha256'] or
            annotation_hash != source['annotation_sha256'] or len(order) != 1024 or len(set(order)) != len(order) or
            set(order).intersection(specification['image_source_ids']) or
            (specification['protocol'], specification['ap_mode']) != ('m-owodb', 'integral') or
            {a['category_id'] for a in holdout['annotations']} != set(manifest['stages'][1]['active_classes'])):
        raise ValueError('Changed M duration holdout or unexpected VOC evaluation inputs')
    for arm, entry in checkpoints.items():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Changed M Task 2 checkpoint for {arm}')

    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='m-owodb', ap_mode='integral', paper_comparable=False, seeds=list(SEEDS), images=len(order),
                batch_size=2, gpu=torch.cuda.get_device_name(), device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_version=str(torch.__version__), manifest_sha256=manifest_hash, checkpoints=checkpoints,
                annotation=str(annotation), annotation_sha256=annotation_hash,
                eval_order_sha256=sha256('\n'.join(order).encode()).hexdigest(),
                duration_plan_sha256=source_hash, duration_result_sha256=file_sha256(DURATION_HOLDOUT / 'comparison.json'),
                s_plan_sha256=s_plan_hash, s_result_sha256=file_sha256(S_HOLDOUT / 'comparison.json'),
                max_detections=LIMIT, old_pool=OLD_POOL, nms_iou=DEDUP_IOU, unknown_policy=s_plan['unknown_policy'],
                criterion='Both 6-epoch arms: Previous >= +0.75 pp, Current >= -0.25 pp, Known > 0 pp, '
                          'and fewer old-class duplicate FPs; report both 3-epoch arms and residual duration gaps.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    rows, deltas = evaluate_endpoints(OUTPUT, plan, MANIFEST, manifest, annotation, order,
                                      native_reference=reference['rows'])
    duration_gaps, seed_pass = {}, {}
    for seed in SEEDS:
        short, long = (f'seed{seed}-{epochs}epochs' for epochs in (3, 6))
        seed_pass[seed] = passes_refill_gate(deltas[long])
        duration_gaps[seed] = {}
        for variant in ('native', 'dedup'):
            metrics = rows[long][variant]['metrics']
            duration_gaps[seed][variant] = {key: 100 * (value - rows[short][variant]['metrics'][key])
                                            for key, value in metrics.items()}
    result = OUTPUT / 'comparison.json'
    write_json(result, dict(status='complete', plan_sha256=file_sha256(plan_path), paper_comparable=False,
                            arms=rows, dedup_minus_native=deltas, six_minus_three_pp=duration_gaps,
                            seed_pass=seed_pass, follow_up=all(seed_pass.values())))
    print('Comparison saved:', result, flush=True)


if __name__ == '__main__':
    main()
