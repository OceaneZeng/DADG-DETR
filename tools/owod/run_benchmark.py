#!/usr/bin/env python
"""Run official four-task increments followed by released balanced finetuning."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, read_json, stage_files


def build_plan(args):
    manifest, _ = stage_files(args.manifest, 0)
    screening = getattr(args, 'owod_screening', None)
    shared = getattr(args, 'screen_shared_task1', None)
    shared_gnn = getattr(args, 'screen_shared_gnn', None)
    no_distill = getattr(args, 'screen_no_distill', False)
    task1_only = getattr(args, 'task1_only', False)
    if task1_only and (screening or shared or shared_gnn):
        raise ValueError('task1-only is for fresh full-data Task 1, not screening/shared-parent runs')
    if no_distill and not (screening and shared and shared_gnn and args.method == 'discovery'):
        raise ValueError('screen-no-distill requires shared-parent discovery screening')
    if bool(shared) != bool(shared_gnn) or (shared and not screening):
        raise ValueError('Shared Task 1/GNN inputs require screening and both artifact paths')
    if args.recipe == 'ore-paper':
        if manifest['protocol'] != 'm-owodb':
            raise ValueError('ORE 2021 does not specify S-OWODB')
        if args.finetune_epochs is None:
            raise ValueError('ORE specifies Nex=50 but not a standalone finetuning epoch count. '
                             'Supply --finetune-epochs from your documented author training recipe; '
                             'this launcher will not invent that value.')
        epochs, ft_epochs, lr, ft_lr = 8, args.finetune_epochs, .01, .01
    else:
        epochs, ft_epochs, lr, ft_lr = 50, 20, 2e-4, 2e-5
        if args.finetune_epochs is not None and args.finetune_epochs != 20:
            raise ValueError('OW-DETR Appendix D requires 20 finetuning epochs')
        if not args.backbone_weights:
            raise ValueError('Original OW-DETR recipe requires DINO --backbone-weights')
    if manifest['protocol'] == 's-owodb' and not args.backbone_weights:
        raise ValueError('S-OWODB original recipe requires DINO backbone pretraining')
    if args.backbone_weights and not args.backbone_weights.is_file():
        raise FileNotFoundError(args.backbone_weights)
    image_root = Path(manifest['image_root'])
    method = getattr(args, 'method', 'control')
    if screening:
        from tools.owod.screening_data import screening_files
        screening_files(args.manifest, screening, 0)
        screening_files(args.manifest, screening, 1)
        if args.recipe != 'ow-detr-paper' or method != 'discovery':
            raise ValueError('Screening supports the paired discovery variants only')
        epochs, ft_epochs = args.screen_increment_epochs, args.screen_finetune_epochs
        if epochs < 2 or ft_epochs < 1 or args.screen_eval_interval < 1:
            raise ValueError('Screening requires >=2 increment epochs, >=1 finetune epoch/eval interval')
    shared_identity = None
    if shared:
        from tools.owod.shared_task1 import validate_inputs
        shared_identity = validate_inputs(args.manifest, screening, shared, shared_gnn,
                                          args.backbone_weights, args.seed)
    runs, previous = [], shared
    for stage in (range(1) if task1_only else range(1, 2) if shared else range(2 if screening else 4)):
        gnn_checkpoint = None
        if shared:
            gnn_checkpoint = shared_gnn
        elif method != 'control' and stage > 0:
            calibration_dir = args.output_dir / manifest['protocol'] / f'seed_{args.seed}' / 'gnn' / f'stage_{stage - 1}'
            gnn_checkpoint = calibration_dir / f'gnn_stage{stage - 1}.pt'
            runs.append({'kind': 'calibration', 'stage': stage - 1, 'phase': 'calibration',
                         'output_dir': str(calibration_dir), 'command': [
                             sys.executable, str(ROOT / 'tools/owod/calibrate_interference_gnn.py'),
                             '--manifest', str(args.manifest), '--stage', str(stage - 1),
                             '--checkpoint', str(previous), '--output-dir', str(calibration_dir),
                             '--seed', str(args.seed), '--num_workers', str(args.num_workers),
                             '--last-decoder-layers', str(args.graph_adapter_last_n),
                             '--probe-rank', str(args.graph_adapter_rank)]})
            if screening:
                runs[-1]['command'] += ['--owod-screening', str(screening)]
        _, files = stage_files(args.manifest, stage)
        for key in ('increment_train', 'train', 'full_val'):
            payload = read_json(files[key])
            missing = [image['file_name'] for image in payload['images']
                       if not (image_root / image['file_name']).is_file()]
            if missing:
                raise FileNotFoundError(f'{key}: {len(missing)} images missing; examples={missing[:5]}')
        for phase in (['increment'] if stage == 0 else ['increment', 'finetune']):
            directory = args.output_dir / manifest['protocol'] / f'seed_{args.seed}' / f'stage_{stage}' / phase
            n_epochs = epochs if phase == 'increment' else ft_epochs
            n_lr = lr if phase == 'increment' else ft_lr
            command = [sys.executable]
            if args.nproc_per_node > 1:
                command += ['-m', 'torch.distributed.run', '--standalone',
                            '--nproc_per_node', str(args.nproc_per_node)]
            command += [str(ROOT / 'main.py'), '--coco_path', str(image_root),
                        '--owod-manifest', str(args.manifest), '--owod-stage', str(stage),
                        '--owod-phase', phase, '--output_dir', str(directory),
                        '--owod-recipe', args.recipe,
                        '--num_classes', '91', '--backbone', 'resnet50',
                        '--num_queries', '100', '--dim_feedforward', '1024',
                        '--epochs', str(n_epochs), '--lr', str(n_lr),
                        '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
                        '--seed', str(args.seed), '--lr_drop', str(max(1, n_epochs)),
                        '--eval_interval', '1']
            if getattr(args, 'activation_checkpointing', False):
                command += ['--activation-checkpointing']
            if screening:
                command += ['--owod-screening', str(screening)]
                command[command.index('--eval_interval') + 1] = str(args.screen_eval_interval)
            if shared:
                command += ['--screen-branch-checkpoint', str(shared)]
            if args.recipe == 'ore-paper':
                if args.batch_size * args.nproc_per_node != 8:
                    raise ValueError('ORE Appendix F requires effective batch size 8')
                command += ['--sgd', '--lr_backbone', str(n_lr)]
            else:
                command += ['--weight_decay', '0.0001', '--lr_backbone', str(n_lr / 10)]
            if args.backbone_weights:
                command += ['--backbone-weights', str(args.backbone_weights)]
            if previous:
                command += ['--pretrained', str(previous)]
            if method != 'control':
                command += ['--graph-routed-adapter', '--graph-adapter-rank', str(args.graph_adapter_rank),
                            '--graph-adapter-last-n', str(args.graph_adapter_last_n),
                            '--graph-risk-beta', str(args.graph_risk_beta),
                            '--graph-route-loss-coef', str(args.graph_route_loss_coef),
                            '--graph-routing', getattr(args, 'graph_routing', 'hard'),
                            '--graph-localization', getattr(args, 'graph_localization', 'frozen'),
                            '--graph-sketch-max-images', str(args.graph_sketch_max_images)]
                if gnn_checkpoint:
                    command += ['--graph-gnn-checkpoint', str(gnn_checkpoint)]
                if stage > 0 and not no_distill:
                    # Hold preservation constant across discovery and its adapter baseline.
                    record = manifest['stages'][stage]
                    old = [c for c in manifest['category_order_source_ids']
                           if c in set(record['active_classes']) - set(record['classes'])]
                    command += ['--old-class-distillation', '--distill-old-class-ids', *map(str, old)]
            if method.startswith('discovery'):
                if no_distill:
                    command += ['--screen-no-distill']
                command += ['--owod-discovery', '--discovery-scoring', getattr(args, 'discovery_scoring', 'hierarchical')]
                if getattr(args, 'discovery_policy', 'v5') == 'v6':
                    command += ['--discovery-policy', 'v6']
                if method == 'discovery-scratch':
                    command += ['--discovery-no-transfer']
                if method == 'discovery-no-message':
                    command += ['--discovery-no-message']
            runs.append({'stage': stage, 'phase': phase, 'command': command,
                         'output_dir': str(directory), 'epochs': n_epochs})
            previous = directory / 'checkpoint.pth'
    result = {'protocol': manifest['protocol'], 'manifest_sha256': file_sha256(args.manifest),
            'backbone_weights_sha256': file_sha256(args.backbone_weights) if args.backbone_weights else None,
            'recipe': args.recipe, 'recipe_source': ('ORE Appendix F; separate finetuning schedule supplied explicitly'
                                                    if args.recipe == 'ore-paper' else 'OW-DETR Appendix D'),
            'model': ('Open-world discovery and prototype promotion (project method)' if method.startswith('discovery')
                      else 'Graph risk-routed class adapters (project method)' if method == 'graph-adapter'
                      else 'Local Deformable DETR control (not an ORE/OW-DETR method reproduction)'),
            'runs': runs}
    if screening:
        result.update(screening_sha256=file_sha256(screening), paper_comparable=False,
                      schedule_source='User-authorized diagnostic budget; not paper training schedule')
    if shared_identity:
        result['shared_task1'] = shared_identity
    if task1_only:
        result['scope'] = 'full-data Task 1 only; no automatic Task 2 transition'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--task1-only', action='store_true',
                        help='Train only full-data Task 1 and stop before any increment/calibration')
    parser.add_argument('--owod-screening', type=Path)
    parser.add_argument('--screen-shared-task1', type=Path)
    parser.add_argument('--screen-shared-gnn', type=Path)
    parser.add_argument('--screen-no-distill', action='store_true')
    parser.add_argument('--screen-increment-epochs', type=int, default=8)
    parser.add_argument('--screen-finetune-epochs', type=int, default=4)
    parser.add_argument('--screen-eval-interval', type=int, default=4)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--recipe', required=True, choices=('ore-paper', 'ow-detr-paper'))
    parser.add_argument('--finetune-epochs', type=int)
    parser.add_argument('--backbone-weights', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--method', choices=('control', 'graph-adapter', 'discovery', 'discovery-scratch',
                                            'discovery-no-message'), default='control')
    parser.add_argument('--discovery-policy', choices=('v5', 'v6'), default='v5')
    parser.add_argument('--graph-adapter-rank', type=int, default=8)
    parser.add_argument('--graph-adapter-last-n', type=int, default=2)
    parser.add_argument('--graph-risk-beta', type=float, default=2.0)
    parser.add_argument('--graph-route-loss-coef', type=float, default=1.0)
    parser.add_argument('--graph-routing', choices=('hard', 'soft'), default='hard')
    parser.add_argument('--graph-localization', choices=('frozen', 'bbox-last'), default='frozen')
    parser.add_argument('--discovery-scoring', choices=('hierarchical', 'focal'), default='hierarchical')
    parser.add_argument('--graph-sketch-max-images', type=int, default=12)
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1 or args.batch_size < 1 or (args.finetune_epochs is not None and args.finetune_epochs < 1):
        raise ValueError('Process count, batch size and finetuning epochs must be positive')
    args.manifest = args.manifest.resolve()
    if args.owod_screening:
        args.owod_screening = args.owod_screening.resolve()
    for field in ('screen_shared_task1', 'screen_shared_gnn'):
        if getattr(args, field):
            setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    if args.backbone_weights:
        args.backbone_weights = args.backbone_weights.resolve()
    plan = build_plan(args)
    if args.dry_run:
        for run in plan['runs']:
            print(shlex.join(run['command']))
        return 0
    root = args.output_dir / plan['protocol'] / f'seed_{args.seed}'
    plan_path = root / 'benchmark_plan.json'
    if plan_path.is_file():
        if not args.resume or read_json(plan_path) != plan:
            raise ValueError('Output already has a plan; resume only with unchanged input and settings')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Use a clean output directory')
    root.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'models/ops') + os.pathsep + str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    if args.gpus:
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    for run in plan['runs']:
        directory = Path(run['output_dir'])
        if run.get('kind') == 'calibration':
            # Content-addressed calibration caches are verified by its runner.
            # Re-run on resume so changed/partial artifacts cannot be trusted.
            result = subprocess.run(run['command'], cwd=ROOT, env=env)
            if result.returncode:
                return result.returncode
            continue
        marker = directory / 'training_complete.json'
        command = list(run['command'])
        if args.resume and marker.is_file():
            completed = read_json(marker)
            if completed.get('last_epoch') == run['epochs'] - 1 and (directory / 'checkpoint.pth').is_file():
                continue
            raise ValueError(f'Invalid completion marker: {marker}')
        if args.resume and (directory / 'checkpoint.pth').is_file():
            command += ['--resume', str(directory / 'checkpoint.pth')]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
