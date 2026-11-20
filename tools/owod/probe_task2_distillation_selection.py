"""Measure old targets lost at the teacher confidence cut and one extra query per empty image."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from engine import teacher_prediction
from models.graph_local.distillation import select_distillation_queries
from tools.owod import probe_task2_distillation_teachers as teachers
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task2-distillation-selection'
SLICES = ('balanced', 'holdout')


@torch.inference_mode()
def measure(model, loader, old, current, selection):
    counts = Counter()
    for step, (samples, targets) in enumerate(loader, 1):
        output = teacher_prediction(model, samples.to('cuda'), 'distillation')
        logits = output['pred_logits']
        old_logits = logits[..., old]
        old_conf, old_slots = old_logits.sigmoid().max(-1)
        original = select_distillation_queries(old_conf, selection['threshold'], selection['cap'])
        prefers_old = old_logits.amax(-1) > logits[..., current].amax(-1)
        # Use at most one query and only on images empty under the training rule.
        best = old_conf.masked_fill(~prefers_old, -1).argmax(dim=1, keepdim=True)
        extra = torch.zeros_like(original).scatter_(1, best, True) & prefers_old & ~original.any(1, keepdim=True)
        for i, target in enumerate(targets):
            labels = target['labels'].to(old.device)
            is_old = torch.isin(labels, old)
            is_current = torch.isin(labels, current)
            overlap = box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][i]),
                              box_cxcywh_to_xyxy(target['boxes'].to(old.device)))[0] > .5
            old_overlap = overlap[:, is_old]
            correct = old_overlap & (old[old_slots[i]][:, None] == labels[is_old])
            selected_hits = correct[original[i]]
            above_cut = old_conf[i] >= selection['threshold']
            below_cut_only = correct.any(0) & ~correct[above_cut].any(0)
            counts.update(images=1, old_gt=int(is_old.sum()), selected=int(original[i].sum()),
                          correct_old_queries=int(selected_hits.any(1).sum()),
                          covered_old_gt=int(selected_hits.any(0).sum()),
                          images_without_selection=int(~original[i].any()),
                          localized_old_gt=int(old_overlap.any(0).sum()),
                          old_argmax_gt=int(correct.any(0).sum()),
                          known_argmax_gt=int(correct[prefers_old[i]].any(0).sum()),
                          below_cut_only_gt=int(below_cut_only.sum()),
                          extra_queries=int(extra[i].sum()), extra_correct=int(correct[extra[i]].any(1).sum()),
                          extra_covered_gt=int(correct[extra[i]].any(0).sum()),
                          extra_new_only=int((overlap[extra[i]][:, is_current].any(1)
                                              & ~old_overlap[extra[i]].any(1)).sum()))
        if step % 100 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {counts["images"]} images', flush=True)
    return dict(counts)


def main():
    source = read_json(teachers.OUTPUT / 'plan.json')
    results = read_json(teachers.OUTPUT / 'comparison.json')
    annotations = {name: Path(source['slices'][name]) for name in SLICES}
    if (results['status'] != 'complete' or results['plan_sha256'] != file_sha256(teachers.OUTPUT / 'plan.json')
            or file_sha256(MANIFEST) != source['manifest_sha256']
            or any(file_sha256(path) != source['annotations'][name] for name, path in annotations.items())
            or torch.cuda.get_device_name() != source['gpu'] or str(torch.__version__) != source['torch_version']):
        raise ValueError('The completed two-teacher comparison and its original inputs are required')
    for pair in source['seeds'].values():
        entry = pair['checkpoints']['task2_increment']
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Changed increment teacher: {entry["path"]}')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=results['plan_sha256'], source_result_sha256=file_sha256(teachers.OUTPUT / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], annotations={k: source['annotations'][k] for k in SLICES},
                teachers={s: p['checkpoints']['task2_increment'] for s, p in source['seeds'].items()},
                selections={s: p['selection'] for s, p in source['seeds'].items()}, gpu=source['gpu'],
                torch_version=source['torch_version'], batch_size=source['batch_size'], paper_comparable=False,
                rule='Only on empty images, add the highest old-confidence query whose best old logit exceeds its best current logit',
                training_gate='Both seeds on both slices: extra coverage >= 5 pp, >= 2 correct queries per incorrect query, and extra new-only fraction <= original new-only fraction')
    pin(OUTPUT / 'plan.json', plan)
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    old, current = (torch.tensor(sorted(stage['classes']), device='cuda') for stage in manifest['stages'][:2])
    loaders = {k: make_loader(resolve_image_root(MANIFEST, manifest), p, plan['batch_size'], 2)[1]
               for k, p in annotations.items()}
    rows = {}
    for seed, entry in plan['teachers'].items():
        model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
        model.eval()
        rows[seed] = {}
        for name, loader in loaders.items():
            path = OUTPUT / f'seed{seed}-{name}.json'
            if not path.is_file():
                print(f'Seed {seed}: {name} ({len(loader.dataset)} images)', flush=True)
                counts = measure(model, loader, old, current, plan['selections'][seed])
                write_json(path, dict(plan_sha256=plan_hash, counts=counts))
            row = read_json(path)
            counts = row['counts']
            ref = results['rows'][f'seed{seed}-task2_increment'][name]
            if (row['plan_sha256'] != plan_hash or any(counts[k] != ref[k] for k in
                    ('images', 'old_gt', 'selected', 'correct_old_queries', 'covered_old_gt', 'images_without_selection'))):
                raise ValueError(f'Original selection differs from the recorded teacher evaluation: {path}')
            n = counts['extra_queries']
            passed = (n > 0 and 20 * counts['extra_covered_gt'] >= counts['old_gt']
                      and 3 * counts['extra_correct'] >= 2 * n
                      and counts['extra_new_only'] * ref['selected'] <= n * ref['new_only_queries'])
            rows[seed][name] = dict(counts=counts, passes_training_gate=passed)
        del model
    ready = all(row['passes_training_gate'] for views in rows.values() for row in views.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, rows=rows,
                                               run_paired_finetune=ready))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
