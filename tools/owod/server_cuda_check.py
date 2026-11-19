"""Run with torchrun on the two selected 3090s before a long experiment."""
import os
import sys
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/ops'))
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import torch.distributed as dist
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema, update_ema
    from models.graph_local.discovery_candidates import flip_samples
    from models.graph_local.routed_adapter import append_stage, configure_trainable, mask_inactive_adapter_gradients
    from util.misc import nested_tensor_from_tensor_list
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Preflight expects exactly two visible GPUs/ranks')
    torch.cuda.set_device(rank)
    if '3090' not in torch.cuda.get_device_name(rank):
        raise RuntimeError('Selected GPU is not an RTX 3090; check CUDA_VISIBLE_DEVICES')
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    try:
        args = get_args_parser().parse_args([])
        args.no_pretrained_backbone = True
        if '--instances' in sys.argv[1:]:
            args.instance_mode = 'matched'
            args.owod_stage = 1
        args.hidden_dim, args.dim_feedforward = 64, 128
        args.enc_layers, args.dec_layers, args.num_queries = 1, 2, 8
        args.graph_adapter_last_n = 1
        args.num_classes = 91
        args.owod_known_class_ids = [1, 2, 3]
        args.owod_max_detections = 8
        args.graph_routed_adapter = True
        args.activation_checkpointing = True
        model, criterion, _ = build(args)
        model.cuda(); criterion.cuda()
        append_stage(model, dict(stage=1, class_ids=[3], old_ids=[1, 2], rank=8,
                                last_n=1, risk_beta=2., risk=[[.2, .4]], routing='soft'))
        localization = 'decoder-bbox' if '--decoder-bbox' in sys.argv[1:] else 'bbox-last'
        configure_trainable(model, [3], [1, 2, 3], localization=localization)
        model.train_discovery()
        criterion.weight_dict['loss_adapter_route'] = 1.
        criterion.epoch = args.discovery_warmup
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        teacher = make_ema(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
        samples = nested_tensor_from_tensor_list([torch.rand(3, 128, 128, device='cuda')])
        targets = [dict(labels=torch.tensor([3], device='cuda'),
                        boxes=torch.tensor([[.5, .5, .3, .3]], device='cuda'))]
        if getattr(args, 'instance_mode', 'legacy') == 'matched':
            targets[0].update(image_id=torch.tensor([rank], device='cuda'),
                              source_objects=torch.tensor([0], device='cuda'))
            # Rank one intentionally has no target. Both ranks must reduce the alignment weight.
            criterion.instance_targets = {(0, 0, 3): ([1.] * 64, .8)}
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = ddp(samples)
            with torch.no_grad():
                outputs['discovery_teacher'] = teacher(flip_samples(samples))
            losses = criterion(outputs, targets)
            loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite preflight loss')
            loss.backward()
            mask_inactive_adapter_gradients(model)
            optimizer.step()
            model.observe_training(criterion.evidence)
            update_ema(teacher, model)
        count = model.memory.counts[3].clone()
        dist.all_reduce(count)
        if count != 8:  # 2 ranks x 2 updates, synchronized support on both ranks
            raise RuntimeError('Distributed prototype synchronization failed')
        print(f'Rank {rank}: {torch.cuda.get_device_name(rank)}, torch={torch.__version__}, '
              f'CUDA runtime={torch.version.cuda}; CUDA extension/DDP/EMA/memory passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
