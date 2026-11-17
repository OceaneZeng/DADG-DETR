"""Two-rank, two-step synthetic hard/soft check; no data, saved weights or AP."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--localization-comparison', action='store_true')
    comparison.add_argument('--unknown-comparison', action='store_true')
    opt = parser.parse_args()
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
        variants = (('soft', 'frozen', 'dense'), ('soft', 'frozen', 'cross-view-protected')) if opt.unknown_comparison else (
            (('soft', 'frozen', 'dense'), ('soft', 'bbox-last', 'dense')) if opt.localization_comparison else
            (('hard', 'frozen', 'dense'), ('soft', 'frozen', 'dense')))
        for routing, localization, negative_policy in variants:
            torch.manual_seed(42)
            args = get_args_parser().parse_args([])
            args.skip_backbone_initialization = True
            args.backbone = 'resnet18'
            args.hidden_dim, args.dim_feedforward = 32, 64
            args.enc_layers = args.dec_layers = args.graph_adapter_last_n = 1
            args.num_queries = args.owod_max_detections = 3
            args.discovery_version, args.discovery_warmup = 7, 0
            args.discovery_negative_policy = negative_policy
            args.owod_known_class_ids, args.owod_previous_class_ids = [1], []
            model, criterion, _ = build(args)
            model.cuda(); criterion.cuda()
            old_teacher = make_ema(model)
            append_stage(model, dict(stage=1, class_ids=[3], old_ids=[1], rank=2, last_n=1,
                                    risk_beta=2., risk=[[.2]], routing=routing))
            model.known_ids = criterion.known_ids = criterion.base.active_class_ids = [1, 3]
            criterion.old_ids = [1]
            hooks = configure_trainable(model, [3], [1, 3], localization=localization)
            coordinates = list(model.detr.bbox_embed[0].layers[-1].parameters())
            coordinate_before = [p.detach().clone() for p in coordinates]
            old_rows = model.detr.class_embed[0].weight[1].detach().clone()
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
                assert criterion.negative_policy == negative_policy
                assert 0 <= losses['discovery_protected_count'] <= args.discovery_top_k
                if negative_policy == 'dense':
                    assert losses['discovery_protected_count'] == 0
                total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
                if not torch.isfinite(total):
                    raise ValueError('Nonfinite CUDA smoke loss')
                total.backward()
                mask_inactive_adapter_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step()
                model.observe_training(criterion.evidence)
                update_ema(teacher, model)
            changed = any(not torch.equal(p, old) for p, old in zip(coordinates, coordinate_before))
            assert changed == (localization == 'bbox-last'), 'Unexpected coordinate update policy'
            assert torch.equal(old_rows, model.detr.class_embed[0].weight[1])
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
            print(f'Rank {rank}: v7 {routing}/{localization}/{negative_policy}, '
                  'CUDA backward/coordinate policy/old rows/NCCL/EMA/memory/rank RNG passed', flush=True)
            del network, model, criterion, teacher, old_teacher, optimizer, outputs, losses, total
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
