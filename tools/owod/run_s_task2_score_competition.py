"""Measure how unknown candidates and objectness affect the S Task 2 duration gap."""
from hashlib import sha256
from math import isclose
import os
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as SOURCE
from tools.owod.screening_data import source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = SOURCE.with_name('s-instance-task2-score-competition-replay')
ARMS = ('3epochs', '6epochs')
VIEWS = {'native': 'focal', 'known_only': 'focal', 'known_class_only': 'focal_class_only'}
EXCLUDE_UNKNOWN = ('known_only', 'known_class_only')


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for the frozen inference comparison')
    print('Verifying the S duration holdout and finetune checkpoints', flush=True)
    source_plan_path = SOURCE / 'plan.json'
    source_result_path = SOURCE / 'comparison.json'
    annotation = SOURCE / 'heldout.json'
    source_plan, source_result = read_json(source_plan_path), read_json(source_result_path)
    manifest = read_json(MANIFEST)
    manifest_hash = file_sha256(MANIFEST)
    data = read_json(annotation)
    if (source_result['status'] != 'complete' or source_result['follow_up'] or
            source_result['plan_sha256'] != file_sha256(source_plan_path) or
            manifest_hash != source_plan['manifest_sha256'] or manifest['protocol'] != 's-owodb' or
            manifest['evaluation']['ap_mode'] != 'voc2007_11point' or
            file_sha256(annotation) != source_plan['annotation_sha256'] or
            len(data['images']) != 1024 or source_ids(data) != set(source_plan['origins'])):
        raise ValueError('Expected the pinned S duration experiment and its 1024-image release-only holdout')
    stage = manifest['stages'][1]
    known, previous, current = stage['active_classes'], manifest['stages'][0]['active_classes'], stage['classes']
    variants = {name: (known, scoring, 50) for name, scoring in VIEWS.items()}
    # The holdout arrays retain the original draw order. The plan's origins map
    # is sorted on disk, which changes the ordering of quantized VOC score ties.
    eval_ids = [image['owod_source_id'] for image in data['images']]
    inputs = {}
    args_by_arm = {}
    for arm in ARMS:
        folder = SOURCE / arm / 'finetune'
        checkpoint = folder / 'checkpoint.pth'
        digest = file_sha256(checkpoint)
        meta = load_checkpoint_metadata(checkpoint)
        args = meta['args']
        frozen = source_result['arms'][arm]
        if (digest != frozen['finetune']['checkpoint_sha256'] or
                read_json(folder / 'input.json')['--pretrained'] != frozen['increment']['checkpoint_sha256'] or
                read_json(folder / 'training_complete.json')['last_epoch'] != 0 or meta['epoch'] != 0 or
                args.seed != source_plan['seed'] or args.owod_stage != 1 or args.owod_phase != 'finetune' or
                args.owod_manifest_sha256 != manifest_hash or args.owod_screening_sha256 != source_plan['screen_sha256'] or
                args.discovery_version != 10 or args.discovery_scoring != 'focal' or args.owod_max_detections != 50):
            raise ValueError(f'Unexpected S duration checkpoint: {checkpoint}')
        inputs[arm] = dict(path=str(checkpoint), sha256=digest, baseline=frozen['heldout'])
        args_by_arm[arm] = args
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', seed=source_plan['seed'], paper_comparable=False, images=1024,
                annotation=str(annotation), annotation_sha256=source_plan['annotation_sha256'],
                eval_order_sha256=sha256('\n'.join(eval_ids).encode()).hexdigest(),
                manifest_sha256=manifest_hash, source_plan_sha256=source_result['plan_sha256'],
                source_result_sha256=file_sha256(source_result_path), checkpoints=inputs,
                device=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), batch_size=2, views=VIEWS, excluded_unknown=list(EXCLUDE_UNKNOWN),
                max_detections=50, primary='Previous AP50 gap closure, known_only vs native',
                criterion='At least 0.75 pp of six-minus-three Previous gap closure after excluding unknown candidates.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    plan_hash = file_sha256(plan_path)

    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    spec = dict(manifest['evaluation'], image_source_ids=eval_ids, known_only_screening=True)
    rows = {}
    for arm, entry in inputs.items():
        path = OUTPUT / f'{arm}.json'
        if not path.is_file():
            print(f'Evaluating {arm} fixed weights on {len(dataset)} images', flush=True)
            model, _ = load_detector(args_by_arm[arm], SOURCE / arm / 'finetune/checkpoint.pth', torch.device('cuda'))
            view = evaluate_views(model, loader, dataset.coco, variants, spec, previous, current, 'cuda',
                                  include_current_stats=True, exclude_unknown=EXCLUDE_UNKNOWN, print_freq=64)
            write_json(path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], view=view))
            del model
        report = read_json(path)
        view = report['view']
        if report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Scoring report identity differs from the registered plan: {path}')
        native = view['native']['metrics']
        if any(not isclose(native[key], value, abs_tol=1e-5) for key, value in entry['baseline'].items()):
            raise ValueError(f'{arm} native scores {native} differ from the original endpoint {entry["baseline"]}')
        if any(view[key]['composition']['unknown'] or view[key]['composition']['detections'] != 50*len(dataset)
               for key in EXCLUDE_UNKNOWN):
            raise ValueError(f'{arm} known-only view did not keep exactly 50 known detections per image')
        rows[arm] = {key: view[key]['metrics'] for key in VIEWS}
    deltas = {view: {key: 100*(rows['6epochs'][view][key]-value) for key, value in rows['3epochs'][view].items()}
              for view in VIEWS}
    closure = deltas['known_only']['previous_ap50'] - deltas['native']['previous_ap50']
    summary = dict(status='complete', plan_sha256=plan_hash, paper_comparable=False, arms=rows,
                   six_minus_three_pp=deltas, previous_gap_closure_pp=closure, follow_up=closure >= .75)
    write_json(OUTPUT / 'comparison.json', summary)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
