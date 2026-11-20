"""Cross increment and balanced-finetune classifiers on the registered M Task 2 holdout."""
import copy
import json
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_decoder_bbox_lr_pair import THREE_QUARTER
from tools.owod.run_decoder_bbox_seed_pair import OUTPUT as SEED43
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-finetune-classifier-probe'
PHASES = ('increment', 'finetune')


def run_seed(folder, seed):
    inputs = {}
    models = {}
    heads = {}
    paths = {phase: folder / phase / 'checkpoint.pth' for phase in PHASES}
    for phase, path in paths.items():
        digest = file_sha256(path)
        if digest != read_json(path.parent / 'coverage.json')['checkpoint_sha256']:
            raise ValueError(f'Checkpoint differs from the evaluated endpoint: {path}')
        payload = load_checkpoint_metadata(path)
        args = payload['args']
        if (args.owod_stage != 1 or args.owod_phase != phase or args.seed != seed
                or args.owod_protocol != 'm-owodb' or args.decoder_bbox_lr_mult != .75
                or args.owod_manifest_sha256 != file_sha256(MANIFEST)
                or args.owod_screening_sha256 != file_sha256(Path(args.owod_screening))
                or args.two_stage or args.owod_max_detections != 50 or args.discovery_scoring != 'focal'
                or payload['epoch'] != args.epochs - 1):
            raise ValueError(f'Unexpected diagnostic configuration: {path}')
        inputs[phase] = dict(checkpoint=str(path), sha256=digest, screen=args.owod_screening_sha256)
        models[phase], _ = load_detector(args, path, torch.device('cuda'))
        heads[phase] = copy.deepcopy(models[phase].detr.class_embed[-1].state_dict())

    finetune_input = read_json(folder / 'finetune/input.json')
    if (Path(args.pretrained) != paths['increment'] or finetune_input['--pretrained'] != inputs['increment']['sha256']
            or inputs['increment']['screen'] != inputs['finetune']['screen']):
        raise ValueError(f'Finetune is not the paired successor: {folder}')
    screen = Path(args.owod_screening)
    manifest, files = screening_files(MANIFEST, screen, 1)
    source_ids = read_json(screen)['heldout_source_ids']
    known, current = manifest['stages'][1]['active_classes'], manifest['stages'][1]['classes']
    previous = manifest['stages'][0]['active_classes']
    spec = {**manifest['evaluation'], 'image_source_ids': source_ids, 'known_only_screening': True}
    dataset, loader = make_loader(args.owod_image_root, files['full_val'], args.batch_size, 2)
    rows = {}
    for body, model in models.items():
        rows[body] = {}
        for head in PHASES:
            model.detr.class_embed[-1].load_state_dict(heads[head])
            view = evaluate_views(model, loader, dataset.coco, {'focal_top50': (known, 'focal', 50)},
                                  spec, previous, current, 'cuda', include_current_stats=True)
            rows[body][head] = view
            print(seed, 'body', body, 'head', head, view['focal_top50']['metrics'], flush=True)
    return dict(inputs=inputs, manifest_sha256=args.owod_manifest_sha256,
                holdout_sha256=file_sha256(files['full_val']), images=len(dataset), rows=rows)


def main():
    sources = {42: ROOT / 'output' / f'm-instance-{THREE_QUARTER}', 43: SEED43 / THREE_QUARTER}
    identity = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                    checkpoints={str(seed): {phase: read_json(folder / phase / 'coverage.json')['checkpoint_sha256']
                                             for phase in PHASES} for seed, folder in sources.items()},
                    primary_endpoint='heldout_current_ap50', paper_comparable=False)
    pin(OUTPUT / 'plan.json', identity)
    result = OUTPUT / 'comparison.json'
    if result.is_file():
        previous = read_json(result)
        if previous['status'] != 'complete' or any(previous[k] != v for k, v in identity.items()):
            raise ValueError(f'Unexpected existing classifier comparison: {result}')
        print('Already complete:', result, flush=True)
        return
    rows = {}
    for seed, folder in sources.items():
        rows[seed] = run_seed(folder, seed)
    if rows[42]['holdout_sha256'] != rows[43]['holdout_sha256']:
        raise ValueError('Seeds used different validation annotations')
    write_json(result, dict(status='complete', **identity, by_seed=rows))
    print(json.dumps({'completed_seeds': list(rows), 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
