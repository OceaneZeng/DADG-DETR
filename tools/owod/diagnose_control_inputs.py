#!/usr/bin/env python
"""Read-only input-response probes on each retained M-OWODB control epoch."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def differences(first, second):
    return {key: {'mean_abs': float((value - second[key]).abs().mean()),
                  'max_abs': float((value - second[key]).abs().max())}
            for key, value in first.items()}


def probe_inputs(model, dataset, device, indices=(0, 7, 15, 23)):
    """Hold geometry fixed; compare real pixels with mean color and another image.

    Only released training images are used. Encoder traces sample every 64th
    spatial token, matching the preceding v8 audit. These are not AP metrics.
    """
    import torch
    import torch.nn.functional as F
    from util.misc import NestedTensor, collate_fn

    if len(dataset) < 2 or not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError('Input probe requires distinct training images at all requested indices')
    traces = {}
    handles = []
    detector = getattr(model, 'detr', model)

    def capture(key):
        def hook(module, args, output):
            traces[key] = output[:, ::64].detach().cpu().clone()
        return hook

    def forward(samples):
        traces.clear()
        output = model(samples)
        return ({key: output[key].detach().cpu() for key in ('pred_boxes', 'pred_logits')},
                dict(traces))

    training = model.training
    rows = []
    try:
        for index, layer in enumerate(detector.transformer.encoder.layers, 1):
            handles.append(layer.norm2.register_forward_hook(capture(f'encoder_block_{index}')))
        model.eval()
        with torch.inference_mode():
            for index in indices:
                other_index = (index + 1) % len(dataset)
                samples, targets = collate_fn([dataset[index]])
                other, other_targets = collate_fn([dataset[other_index]])
                samples = samples.to(device)
                real, real_traces = forward(samples)
                neutral, neutral_traces = forward(
                    NestedTensor(torch.zeros_like(samples.tensors), samples.mask))
                resized = F.interpolate(other.tensors.to(device), size=samples.tensors.shape[-2:],
                                        mode='bilinear', align_corners=False)
                pair, pair_traces = forward(NestedTensor(resized, samples.mask))
                rows.append({'index': index, 'other_index': other_index,
                    'image_id': int(targets[0]['image_id']),
                    'other_image_id': int(other_targets[0]['image_id']),
                    'mean_color': {'outputs': differences(real, neutral),
                                   'encoder': differences(real_traces, neutral_traces)},
                    'different_real_image': {'outputs': differences(real, pair),
                                             'encoder': differences(real_traces, pair_traces)}})
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    summary = {}
    for comparison in ('mean_color', 'different_real_image'):
        summary[comparison] = {
            group: {key: sum(row[comparison][group][key]['mean_abs'] for row in rows) / len(rows)
                    for key in rows[0][comparison][group]}
            for group in ('outputs', 'encoder')}
    return {'summary_mean_abs': summary, 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--completed-epochs', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.completed_epochs < 1:
        raise ValueError('completed-epochs must be positive')

    import torch
    from main import load_local_checkpoint
    from tools.owod.gnn_calibration import build_calibration_dataset, load_detector
    from tools.owod.protocol import file_sha256, stage_files

    torch.set_num_threads(2)
    directory = args.run_dir.resolve()
    checkpoints = [directory / f'checkpoint{epoch:04}.pth' for epoch in range(args.completed_epochs)]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing retained control epoch: {checkpoint}')
    dataset = None
    identity = None
    report = {'note': 'Read-only training-image probes; fixed dimensions/mask, eval mode. '
                      'Mean color is zero in normalized pixels. No optimizer, EMA or prototype updates. '
                      'Encoder norms are sampled every 64th spatial token. Not author AP/U-Recall.',
              'epochs': []}
    for epoch, checkpoint in enumerate(checkpoints):
        digest = file_sha256(checkpoint)
        payload = load_local_checkpoint(checkpoint)
        saved = payload['args']
        if (payload.get('epoch') != epoch or epoch >= saved.epochs
                or saved.owod_stage != 0 or saved.owod_phase != 'increment'
                or getattr(saved, 'owod_discovery', False)
                or getattr(saved, 'graph_routed_adapter', False)
                or getattr(saved, 'paper_baseline', None) or getattr(saved, 'with_tree', False)
                or getattr(saved, 'owod_screening', '')
                or getattr(saved, 'owod_screening_sha256', None)):
            raise ValueError('Input probe requires full-data Task 1 supervised control epoch checkpoints')
        del payload
        manifest_path = Path(saved.owod_manifest)
        manifest, files = stage_files(manifest_path, 0)
        if manifest['protocol'] != 'm-owodb':
            raise ValueError('This input probe is registered for M-OWODB control')
        current_identity = {'manifest_sha256': file_sha256(manifest_path),
                            'backbone_weights_sha256': file_sha256(Path(saved.backbone_weights)),
                            'train_ann_sha256': file_sha256(files['increment_train'])}
        if (current_identity['manifest_sha256'] != saved.owod_manifest_sha256
                or current_identity['backbone_weights_sha256'] != saved.backbone_weights_sha256
                or current_identity['train_ann_sha256'] != file_sha256(Path(saved.train_ann))
                or list(saved.owod_known_class_ids) != manifest['stages'][0]['active_classes']):
            raise ValueError('Control checkpoint data, class order or backbone input changed')
        if identity is not None and current_identity != identity:
            raise ValueError('Control epoch checkpoints must use the same registered inputs')
        identity = current_identity
        if dataset is None:
            dataset = build_calibration_dataset(Path(manifest['image_root']), files['increment_train'])
        model, criterion = load_detector(saved, checkpoint, torch.device(args.device))
        model.eval()
        result = probe_inputs(model, dataset, args.device)
        if file_sha256(checkpoint) != digest:
            raise RuntimeError('Checkpoint changed during read-only input probe')
        report['epochs'].append({'epoch': epoch, 'completed_epochs': epoch + 1,
                                 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                                 'inputs': identity, **result})
        means = result['summary_mean_abs']
        last_encoder = list(means['mean_color']['encoder'])[-1]
        print(f'Control input probe: completed_epochs={epoch + 1} '
              f'mean_color_box_delta={means["mean_color"]["outputs"]["pred_boxes"]:.8g} '
              f'mean_color_logits_delta={means["mean_color"]["outputs"]["pred_logits"]:.8g} '
              f'mean_color_encoder_last_delta={means["mean_color"]["encoder"][last_encoder]:.8g} '
              f'real_pair_box_delta={means["different_real_image"]["outputs"]["pred_boxes"]:.8g} '
              f'real_pair_logits_delta={means["different_real_image"]["outputs"]["pred_logits"]:.8g} '
              f'real_pair_encoder_last_delta={means["different_real_image"]["encoder"][last_encoder]:.8g}',
              flush=True)
        del model, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir = directory / 'input_diagnostics' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output_dir.mkdir(parents=True)
    destination = output_dir / 'summary.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Control input-response report: {destination}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
