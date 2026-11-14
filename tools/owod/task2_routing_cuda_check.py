"""Two-rank, two-step synthetic hard/soft check; no data, saved weights or AP."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def main():
    import gc
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema, update_ema
    from models.graph_local.discovery_candidates import flip_samples
    from models.graph_local.routed_adapter import append_stage, configure_trainable, mask_inactive_adapter_gradients
    from tools.owod.train_task2_diagnostic import make_optimizer, capture_rank_state, restore_rank_state
    from util.misc import nested_tensor_from_tensor_list
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    try:
        for routing in ('hard', 'soft'):
            torch.manual_seed(42)
            args = get_args_parser().parse_args([])
            args.skip_backbone_initialization = True
            args.backbone = 'resnet18'
            args.hidden_dim, args.dim_feedforward = 32, 64
            args.enc_layers = args.dec_layers = args.graph_adapter_last_n = 1
            args.num_queries = args.owod_max_detections = 3
            args.discovery_version, args.discovery_warmup = 7, 0
            args.owod_known_class_ids, args.owod_previous_class_ids = [1], []
            model, criterion, _ = build(args)
            model.cuda(); criterion.cuda()
            old_teacher = make_ema(model)
            append_stage(model, dict(stage=1, class_ids=[3], old_ids=[1], rank=2, last_n=1,
                                    risk_beta=2., risk=[[.2]], routing=routing))
            model.known_ids = criterion.known_ids = criterion.base.active_class_ids = [1, 3]
            criterion.old_ids = [1]
            hooks = configure_trainable(model, [3], [1, 3])
            model.train_discovery()
            criterion.weight_dict['loss_adapter_route'] = 1.
            criterion.train()
            network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            teacher = make_ema(model)
            optimizer = make_optimizer(model, args)
            images = nested_tensor_from_tensor_list([torch.rand(3, 96, 96, device='cuda')])
            targets = [dict(labels=torch.tensor([3], device='cuda'),
                            boxes=torch.tensor([[.5, .5, .3, .3]], device='cuda'))]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs = network(images)
                with torch.no_grad():
                    outputs['discovery_old_teacher'] = old_teacher(images)
                    outputs['discovery_teacher'] = teacher(flip_samples(images))
                losses = criterion(outputs, targets)
                total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
                if not torch.isfinite(total):
                    raise ValueError('Nonfinite CUDA smoke loss')
                total.backward()
                mask_inactive_adapter_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step()
                model.observe_training(criterion.evidence)
                update_ema(teacher, model)
            states = [None] * 2
            generator = torch.Generator().manual_seed(42 + rank)
            dist.all_gather_object(states, capture_rank_state(generator))
            expected = torch.rand(3, device='cuda')
            restore_rank_state(states[rank], generator)
            assert torch.equal(expected, torch.rand(3, device='cuda'))
            counts = [torch.empty_like(model.memory.counts) for _ in range(2)]
            dist.all_gather(counts, model.memory.counts)
            assert torch.equal(counts[0], counts[1])
            for hook in hooks:
                hook.remove()
            print(f'Rank {rank}: v7 {routing}, CUDA backward/NCCL/EMA/memory/rank RNG passed', flush=True)
            del network, model, criterion, teacher, old_teacher, optimizer, outputs, losses, total
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
