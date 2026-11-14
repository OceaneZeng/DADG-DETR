"""Full-data, paired v7 hard/soft Task 2 diagnostic, with read-only reference."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.protocol import file_sha256
from tools.owod.task2_parent_provenance import verify_parent_inputs, verify_reference
from tools.owod.train_task2_diagnostic import diagnostic_identity, dump, run_policy

EPOCHS = 8
EVALUATION_EPOCHS = (3, 8)


def routing_identity(manifest, checkpoint, gnn, proof, reference, world_size, gpu_selection):
    identity = diagnostic_identity(manifest, checkpoint, gnn, world_size, epochs=EPOCHS)
    identity.update(suite='sowodb-task2-v7-routing-v2', policies=[7],
                    evaluation_epochs=list(EVALUATION_EPOCHS),
                    variants=['hard', 'soft'], localization='frozen',
                    parent_provenance=proof, reference=reference, probe_images_per_class=12,
                    cuda_visible_devices=gpu_selection)
    return identity


def paired_report(hard, soft, *, epoch=EPOCHS):
    if epoch not in EVALUATION_EPOCHS:
        raise ValueError('Report requires a predeclared evaluation epoch')
    metrics = ('owod_previous_ap50', 'owod_current_ap50', 'owod_known_ap50', 'owod_u_recall',
               'owod_unknown_precision', 'owod_unknown_predictions', 'owod_wi', 'owod_a_ose')
    return dict(diagnostic_only=True, epochs=EPOCHS, evaluation_epoch=epoch,
                evaluation_epochs=list(EVALUATION_EPOCHS), interim=epoch < EPOCHS, finetune=False,
                ap_convention='VOC2007 11-point AP50; AP/recall/precision in [0,1]',
                hard={k: hard[k] for k in metrics if k in hard},
                soft={k: soft[k] for k in metrics if k in soft},
                soft_minus_hard={k: soft[k] - hard[k] for k in metrics if k in hard and k in soft},
                note=f'Predeclared epoch {epoch} results; final comparison is epoch {EPOCHS}. '
                     'Increment-only diagnostic, '
                     'not formal Task 2 or evidence of successful Task 1 unknown-to-new transfer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/owod/s-owodb/split_manifest.json'))
    parent = Path('output/tree-detr-s-v5-full/s-owodb/seed_42')
    parser.add_argument('--checkpoint', type=Path, default=parent / 'stage_0/increment/checkpoint0049.pth')
    parser.add_argument('--gnn', type=Path, default=parent / 'gnn/stage_0/gnn_stage0.pt')
    previous = Path('output/task2-v5-v7-diagnostic-3epochs-2gpu/results')
    parser.add_argument('--reference-checkpoint', type=Path, default=previous / 'v7/checkpoint.pth')
    parser.add_argument('--reference-plan', type=Path, default=previous / 'diagnostic_plan.json')
    parser.add_argument('--output', type=Path, default=Path('output/sowodb-task2-v7-routing-8epochs-2gpu/results'))
    parser.add_argument('--preflight', action='store_true', help='CPU-only provenance checks; writes no experiment state')
    parser.add_argument('--resume', action='store_true')
    opt = parser.parse_args()
    for field in ('manifest', 'checkpoint', 'gnn', 'reference_checkpoint', 'reference_plan', 'output'):
        setattr(opt, field, getattr(opt, field).resolve())
    import torch
    torch.set_num_threads(2)
    proof = verify_parent_inputs(opt.manifest, opt.checkpoint, opt.gnn)
    reference = verify_reference(opt.reference_checkpoint, opt.reference_plan, proof)
    if opt.preflight:
        print(json.dumps(dict(parent_provenance=proof, reference=reference,
                              variants=['hard', 'soft'], epochs=EPOCHS,
                              evaluation_epochs=list(EVALUATION_EPOCHS), effective_batch_size=2), indent=2))
        return
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.rank = int(os.environ.get('RANK', 0))
    opt.gpu = int(os.environ.get('LOCAL_RANK', 0))
    if opt.world_size != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        parser.error('This paired experiment requires two GPUs via torchrun, batch 1 per rank')
    torch.cuda.set_device(opt.gpu)
    from datetime import timedelta
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    # LOCAL_SIZE is consumed by the existing COCO dataset helpers.
    os.environ['LOCAL_SIZE'] = '2'
    identity = routing_identity(opt.manifest, opt.checkpoint, opt.gnn, proof, reference,
                                opt.world_size, os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    import torchvision
    import MultiScaleDeformableAttention as extension
    identity['runtime'] = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                               cuda=torch.version.cuda, extension_sha256=file_sha256(Path(extension.__file__)),
                               devices=[dict(name=torch.cuda.get_device_name(i),
                                             capability=torch.cuda.get_device_capability(i)) for i in range(2)])
    # JSON roundtrip keeps tuple-valued runtime fields stable on resume.
    identity = json.loads(json.dumps(identity))
    opt.output.mkdir(parents=True, exist_ok=True)
    plan_path = opt.output / 'diagnostic_plan.json'
    if opt.rank == 0:
        if plan_path.exists():
            if not opt.resume or json.loads(plan_path.read_text()) != identity:
                raise ValueError('Resume requires exactly the same code, inputs, GPU selection and plan')
        elif opt.resume or any(opt.output.iterdir()):
            raise ValueError('First run requires an empty result directory, without --resume')
        dump(plan_path, identity)
        dump(opt.output / 'parent_provenance.json', proof)
    torch.distributed.barrier()
    from tools.owod.task2_learning_probe import evaluate_reference
    evaluate_reference(opt, identity)
    gc.collect()
    torch.cuda.empty_cache()
    for variant in ('hard', 'soft'):
        print(f'v7 Task 2 paired experiment: {variant}, {EPOCHS} increment epochs, frozen localization', flush=True)
        run_policy(opt, identity, 7, variant=variant, provenance=proof)
    for path, expected in [(opt.checkpoint, proof['checkpoint_sha256']), (opt.gnn, proof['gnn_sha256']),
                           (opt.reference_checkpoint, reference['checkpoint_sha256']),
                           (opt.manifest, proof['canonical_manifest_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError(f'Input changed during experiment: {path}')
    if opt.rank == 0:
        comparisons = {}
        for epoch in EVALUATION_EPOCHS:
            metrics = [json.loads((opt.output / variant / f'metrics_epoch_{epoch}.json').read_text())
                       for variant in ('hard', 'soft')]
            comparisons[epoch] = paired_report(*metrics, epoch=epoch)
            dump(opt.output / f'comparison_epoch_{epoch}.json', comparisons[epoch])
        report = comparisons[EPOCHS]
        report['change_from_epoch_3_to_8'] = {
            variant: {key: value - comparisons[3][variant][key]
                      for key, value in report[variant].items() if key in comparisons[3][variant]}
            for variant in ('hard', 'soft')}
        dump(opt.output / 'comparison.json', report)
        print(json.dumps(report, indent=2), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
