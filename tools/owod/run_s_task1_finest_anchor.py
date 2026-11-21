"""Change only the finest-level encoder anchor size on a frozen Task 1 model."""
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import run_s_task1_proposal_stages as stages
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import OUTPUT as SCALE
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = stages.OUTPUT.with_name('s-instance-task1-finest-anchor')


def main():
    base = read_json(stages.OUTPUT / 'plan.json')
    parent = read_json(stages.OUTPUT / 'comparison.json')
    trace = read_json(stages.OUTPUT / 'stage-trace.json')
    data = PARENT / 'evaluation/fresh1024.json'
    native = SCALE / 'standard.json'
    checkpoint = Path(base['job']['parent'])
    if (parent['status'] != 'complete' or not file_sha256(stages.OUTPUT / 'plan.json') == parent['plan_sha256'] == trace['plan_sha256']
            or file_sha256(checkpoint) != base['job']['parent_sha256'] or file_sha256(native) != base['reference_sha256']
            or file_sha256(data) != base['fresh_sha256'] or file_sha256(MANIFEST) != base['manifest_sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed diagnostic endpoint or paired inference environment')
    plan = {k: base[k] for k in ('seed', 'protocol', 'job', 'batch_size', 'gpu', 'torch_version',
                                 'fresh_sha256', 'manifest_sha256', 'reference_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], sizes={'standard': .05, 'half': .025},
                parent_plan_sha256=parent['plan_sha256'], parent_result_sha256=file_sha256(stages.OUTPUT / 'comparison.json'),
                stage_trace_sha256=file_sha256(stages.OUTPUT / 'stage-trace.json'),
                criterion='Only consider independent replication if small proxy dense IoU>=.5 coverage increases '
                          'by >=5 pp, small final coverage and emitted small proxy hits both rise, and total '
                          'proxy and correctly classified known hits do not fall. Keep native Top-50 and image size. '
                          'Stop after both arms. No fitting or threshold selection.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    root = resolve_image_root(MANIFEST, manifest)
    baseline = read_json(native)['images']
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    totals = {}
    for arm, side in plan['sizes'].items():
        model.detr.transformer.instance_proposals.finest_anchor_size = side
        path = OUTPUT / f'{arm}.json'
        if not path.exists():
            _, loader = make_loader(root, data, plan['batch_size'], 2)
            print(f'{arm}: evaluating finest anchor side {side}', flush=True)
            images, rows = stages.evaluate(model, post, loader, known, proxy, baseline if arm == 'standard' else None)
            write_json(path, dict(plan_sha256=digest, images=images, rows=rows))
        saved = read_json(path)
        if saved['plan_sha256'] != digest or saved['images'].keys() != trace['images'].keys():
            raise ValueError(f'Incomplete {arm} endpoint')
        for im, original in trace['images'].items():
            actual = saved['images'][im]
            if actual['annotation_ids'] != original['annotation_ids'] or (arm == 'standard' and actual['stages'] != original['stages']):
                raise ValueError(f'Unpaired {arm} proposal trace on {im}')
        totals[arm] = saved['rows']
    a, b = totals['standard'], totals['half']
    if a != parent['rows']:
        raise ValueError('Standard anchor does not reproduce all registered counts')
    def emitted(rows, group):
        return sum(rows[group + '_' + size]['emitted_best'] for size in ('small', 'medium', 'large'))
    prev, alt = a['proxy_small'], b['proxy_small']
    passes = (100 * (alt['dense'] - prev['dense']) >= 5 * prev['gt'] and alt['final'] > prev['final']
              and alt['emitted_best'] > prev['emitted_best'] and emitted(b, 'proxy') >= emitted(a, 'proxy')
              and emitted(b, 'known') >= emitted(a, 'known'))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals,
                                                follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
