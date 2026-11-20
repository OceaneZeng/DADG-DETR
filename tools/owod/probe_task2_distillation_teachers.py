"""Measure the two frozen distillation teachers on identical M Task 2 images."""
from pathlib import Path
import subprocess

import torch

from engine import teacher_prediction
from models.graph_local.distillation import select_distillation_queries
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import run_decoder_bbox_seed_pair as replica
from tools.owod import run_stronger_parent_study as study
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task2-distillation-teachers'


def teacher_pair(reference, seed, screen, previous):
    endpoint = reference['checkpoints'][f'seed{seed}-6epochs']
    finetune = Path(endpoint['path'])
    if file_sha256(finetune) != endpoint['sha256']:
        raise ValueError(f'Changed finetune checkpoint: {finetune}')
    args = load_checkpoint_metadata(finetune)['args']
    selection = dict(threshold=args.distill_score_threshold, cap=args.distill_max_queries)
    checkpoints = {}
    path = finetune
    for name, stage, epochs in [('task2_increment', 1, 6), ('task1', 0, study.PARENT_EPOCHS)]:
        if not args.old_class_distillation or list(args.distill_old_class_ids) != list(previous):
            raise ValueError(f'Unexpected distillation classes in {path}')
        inputs = read_json(path.parent / 'input.json')
        path = Path(args.pretrained)
        digest = file_sha256(path)
        metadata = load_checkpoint_metadata(path)
        args = metadata['args']
        if (digest != inputs['--pretrained'] or args.seed != seed or args.owod_stage != stage
                or args.owod_phase != 'increment' or metadata['epoch'] != epochs - 1
                or read_json(path.parent / 'training_complete.json')['last_epoch'] != epochs - 1
                or args.owod_manifest_sha256 != reference['manifest_sha256']
                or args.owod_screening_sha256 != file_sha256(screen)
                or args.distill_score_threshold != selection['threshold']
                or args.distill_max_queries != selection['cap']):
            raise ValueError(f'Unexpected teacher checkpoint: {path}')
        checkpoints[name] = dict(path=str(path), sha256=digest)
    return dict(checkpoints=checkpoints, selection=selection, finetune_sha256=endpoint['sha256'])


@torch.inference_mode()
def measure(model, loader, old_ids, current_ids, selection, score_boxes):
    images = selected_count = empty = 0
    old_gt = covered_gt = correct_queries = new_only = 0
    for step, (samples, targets) in enumerate(loader, 1):
        output = teacher_prediction(model, samples.to('cuda'), 'distillation')
        confidence, slots = output['pred_logits'][..., old_ids].sigmoid().max(-1)
        selected = select_distillation_queries(confidence, selection['threshold'], selection['cap'])
        images += len(targets)
        selected_count += int(selected.sum())
        empty += int((~selected.any(dim=1)).sum())
        if score_boxes:
            for index, target in enumerate(targets):
                labels = target['labels'].cuda()
                old = torch.isin(labels, old_ids)
                current = torch.isin(labels, current_ids)
                keep = selected[index]
                boxes = box_cxcywh_to_xyxy(output['pred_boxes'][index, keep])
                overlaps = box_iou(boxes, box_cxcywh_to_xyxy(target['boxes'].cuda()))[0] > .5
                old_overlap = overlaps[:, old]
                correct = old_overlap & (old_ids[slots[index, keep]][:, None] == labels[old])
                old_gt += int(old.sum())
                covered_gt += int(correct.any(dim=0).sum())
                correct_queries += int(correct.any(dim=1).sum())
                new_only += int((overlaps[:, current].any(dim=1) & ~old_overlap.any(dim=1)).sum())
        if step % 100 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {images} images', flush=True)
    row = dict(images=images, selected=selected_count, images_without_selection=empty,
               selected_per_image=selected_count / images)
    if score_boxes:
        row.update(old_gt=old_gt, covered_old_gt=covered_gt, correct_old_queries=correct_queries,
                   new_only_queries=new_only, old_gt_coverage=covered_gt / old_gt,
                   correct_fraction=correct_queries / selected_count if selected_count else None,
                   new_only_fraction=new_only / selected_count if selected_count else None)
    return row


def main():
    manifest, files = screening_files(MANIFEST, study.SCREEN, 1)
    previous, current = (manifest['stages'][i]['classes'] for i in (0, 1))
    reference = read_json(holdout.OUTPUT / 'plan.json')
    result = read_json(holdout.OUTPUT / 'comparison.json')
    slices = dict(increment=files['increment_train'], balanced=files['train'],
                  holdout=holdout.OUTPUT / 'heldout.json')
    annotations = {key: file_sha256(path) for key, path in slices.items()}
    _, replica_files = screening_files(MANIFEST, replica.SCREEN, 1)
    if (result['status'] != 'complete' or result['plan_sha256'] != file_sha256(holdout.OUTPUT / 'plan.json')
            or reference['manifest_sha256'] != file_sha256(MANIFEST)
            or annotations['holdout'] != reference['annotation_sha256']
            or any(file_sha256(files[k]) != file_sha256(replica_files[k]) for k in files)):
        raise ValueError('Expected the registered two-seed holdout and identical Task 2 phase inputs')
    reserved = source_ids(read_json(slices['holdout']))
    train = source_ids(read_json(slices['increment'])) | source_ids(read_json(slices['balanced']))
    if reserved & train or (reserved | train) & set(manifest['evaluation']['image_source_ids']):
        raise ValueError('Unexpected training, holdout or author test overlap')
    seeds = {str(seed): teacher_pair(reference, seed, screen, previous)
             for seed, screen in [(42, study.SCREEN), (43, replica.SCREEN)]}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                manifest_sha256=file_sha256(MANIFEST), reference_plan_sha256=result['plan_sha256'],
                annotations=annotations, slices={k: str(v) for k, v in slices.items()}, seeds=seeds,
                batch_size=2, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                primary_endpoint='balanced old GT covered by at least one selected correct old-class query, IoU > 0.5',
                paper_comparable=False)
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    loaders = {k: make_loader(resolve_image_root(MANIFEST, manifest), p, 2, 2)[1] for k, p in slices.items()}
    old_ids = torch.tensor(sorted(previous), device='cuda')
    current_ids = torch.tensor(current, device='cuda')
    rows = {}
    for seed, pair in seeds.items():
        for name, entry in pair['checkpoints'].items():
            key = f'seed{seed}-{name}'
            path = OUTPUT / f'{key}.json'
            if not path.is_file():
                model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
                model.eval()
                views = {}
                for label, loader in loaders.items():
                    print(f'{key} on {label}: {len(loader.dataset)} images', flush=True)
                    # Increment training labels contain current classes only.
                    views[label] = measure(model, loader, old_ids, current_ids, pair['selection'], label != 'increment')
                write_json(path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], views=views))
                del model
            row = read_json(path)
            if row['plan_sha256'] != plan_hash or row['checkpoint_sha256'] != entry['sha256']:
                raise ValueError(f'Changed teacher or evaluation plan: {path}')
            rows[key] = row['views']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
