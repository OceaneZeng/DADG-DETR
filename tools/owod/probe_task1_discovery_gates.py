"""Audit frozen Task 1 candidate gates and their fixed-count matching headroom."""
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

from scipy.optimize import linear_sum_assignment
import torch

from main import load_local_checkpoint
from models.graph_local.discovery import make_ema
from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import probe_task2_slot_features as features
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task1-discovery-gates'
GUARDS = ('student_saliency', 'consistency', 'teacher_saliency')


def masks(trace, selected):
    valid, eligible, student, consistent, teacher = (trace[k].cpu() for k in
                                                      ('valid', 'eligible', 'salient', 'consistent', 'other_salient'))
    return dict(raw=torch.ones_like(valid), valid=valid, eligible=eligible, student_salient=eligible & student,
                consistent=eligible & student & consistent, confident=trace['confident'].cpu(), selected=selected.cpu(),
                without_student_saliency=eligible & consistent & teacher, without_consistency=eligible & student & teacher,
                without_teacher_saliency=eligible & student & consistent)


def measure_boxes(overlap, mask, quota):
    edges = overlap[mask]
    matched = 0
    if edges.numel() and edges.any():
        values = edges.numpy().astype('int8')
        q, g = linear_sum_assignment(-values)
        matched = int(values[q, g].sum())
    return dict(predictions=int(mask.sum()), covered_gt=int(edges.any(0).sum()),
                positive_queries=int(edges.any(1).sum()), one_to_one=matched, same_quota=min(matched, quota))


def candidate_batches(model, criterion, ema, loader, *, include_teacher=False):
    known_ids = torch.tensor(model.known_ids, device='cuda')
    for images_batch, targets in loader:
        samples = images_batch.to('cuda')
        targets = move_targets(targets, 'cuda')
        known = []
        for target in targets:
            released = torch.isin(target['labels'], known_ids)
            known.append({k: target[k][released] for k in ('labels', 'boxes')})
        output = model(samples)
        trace = []
        teacher = ema.forward_teacher(flip_samples(samples))
        mined = mine_candidates(output, known, criterion.matcher(output, known), model.known_ids,
                                teacher, criterion.top_k,
                                criterion.saliency_threshold, criterion.consistency_threshold, gate_trace=trace)
        batch = targets, output, trace, mined
        yield (*batch, teacher) if include_teacher else batch


@torch.inference_mode()
def measure(model, criterion, ema, loader, new_classes, source_of):
    counters = defaultdict(lambda: defaultdict(Counter))
    images, new_ids = set(), torch.tensor(new_classes, device='cuda')
    for step, (targets, output, trace, mined) in enumerate(candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            image = int(target['image_id'].item())
            if image in images:
                raise ValueError('Repeated image in diagnostic slice')
            images.add(image)
            truth = target['boxes'][torch.isin(target['labels'], new_ids)]
            overlaps = (box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][i]), box_cxcywh_to_xyxy(truth))[0] >= .5).cpu()
            selected = mined['pseudo'][i]
            quota = int(selected.sum())
            stages = masks(trace[i], selected)
            if (stages['selected'] & ~stages['confident']).any():
                raise ValueError('Selected candidate outside its recorded gate')
            groups = [counters['all'], counters[source_of[image]]]
            for group in groups:
                group['sample'].update(images=1, gt=len(truth), zero_quota_images=int(quota == 0),
                                       gt_on_zero_quota_images=len(truth) if quota == 0 else 0)
            for name, mask in stages.items():
                stats = measure_boxes(overlaps, mask, quota)
                for group in groups:
                    group[name].update(stats)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(images)} images', flush=True)
    if images != set(source_of):
        raise ValueError('Incomplete slice')
    return counters


def matching_headroom(row):
    stats = row['all']
    gt, full = stats['sample']['gt'], stats['confident']['same_quota']
    delta = {name: stats['without_' + name]['same_quota'] - full for name in GUARDS}
    delta['ranking'] = full - stats['selected']['one_to_one']
    return {name: 100 * value / gt for name, value in delta.items()}


def frozen_miner(checkpoint, manifest):
    model, criterion = load_detector(None, checkpoint, torch.device('cuda'))
    model.eval()
    ema = make_ema(model)
    ema.load_state_dict(load_local_checkpoint(checkpoint)['discovery_ema'], strict=True)
    if set(model.known_ids) != set(manifest['stages'][0]['classes']):
        raise ValueError(f'Unexpected Task 1 known classes: {checkpoint}')
    return model, criterion, ema


def main():
    last = read_json(features.OUTPUT / 'comparison.json')
    source = read_json(features.OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    large = holdout.OUTPUT / 'heldout.json'
    if (last['status'] != 'complete' or last['plan_sha256'] != file_sha256(features.OUTPUT / 'plan.json')
            or file_sha256(large) != source['holdout_sha256'] or file_sha256(MANIFEST) != source['manifest_sha256']
            or str(torch.__version__) != source['torch_version'] or torch.cuda.get_device_name() != source['gpu']):
        raise ValueError('Completed source comparison and its original inputs are required')
    jobs = {}
    for seed, entry in source['entries'].items():
        parent, screen = Path(entry['parent']), Path(entry['screen'])
        audit = parent.parent / 'candidate_coverage.json'
        if (file_sha256(parent) != entry['parent_sha256'] or file_sha256(screen) != entry['screen_sha256']
                or read_json(audit)['checkpoint_sha256'] != entry['parent_sha256']):
            raise ValueError(f'Changed Task 1 parent or calibration source for seed {seed}')
        _, files = screening_files(MANIFEST, screen, 1)
        jobs[seed] = dict(parent=str(parent), parent_sha256=entry['parent_sha256'], screen_sha256=entry['screen_sha256'],
                          reference=str(audit), reference_sha256=file_sha256(audit),
                          small=str(files['full_val']), small_sha256=file_sha256(files['full_val']))
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=last['plan_sha256'], source_result_sha256=file_sha256(features.OUTPUT / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], holdout_sha256=source['holdout_sha256'], jobs=jobs,
                torch_version=source['torch_version'], gpu=source['gpu'], batch_size=2, paper_comparable=False,
                decision='Both seeds on the 1024-image holdout: prioritize ranking headroom >= 5 pp, otherwise report any single guard whose removal adds >= 5 pp over the hard-gated pool at the same per-image count.',
                metric='Maximum one-to-one matches at IoU >= .5, capped at original selected count on each image. Only Task 1 labels supplied to mining.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    image_root = resolve_image_root(MANIFEST, manifest)
    task1, task2 = (set(manifest['splits'][k]) for k in ('t1_train', 't2_train'))
    rows, headroom = {}, {}
    for seed, job in jobs.items():
        checkpoint = Path(job['parent'])
        model, criterion, ema = frozen_miner(checkpoint, manifest)
        rows[seed] = {}
        for label, annotation in [('reference128', Path(job['small'])), ('holdout1024', large)]:
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                dataset, loader = make_loader(image_root, annotation, 2, 2)
                sources = {}
                for image, meta in dataset.coco.imgs.items():
                    name = meta['owod_source_id']
                    membership = [s for s, ids in [('task1', task1), ('task2', task2)] if name in ids]
                    if not membership:
                        raise ValueError(f'Image outside training sources: {name}')
                    sources[image] = '_and_'.join(membership)
                print(f'Seed {seed}, {label}: frozen Task 1 miner, {len(dataset)} images', flush=True)
                counts = measure(model, criterion, ema, loader, manifest['stages'][1]['classes'], sources)
                write_json(path, dict(plan_sha256=digest, counts=counts))
            row = read_json(path)
            if row['plan_sha256'] != digest:
                raise ValueError(f'Unexpected diagnostic plan in {path}')
            rows[seed][label] = row['counts']
            observed = row['counts']['all']
            if label == 'reference128':
                ref = read_json(Path(job['reference']))['coverage']
                checks = dict(images=observed['sample']['images'], task2_gt=observed['sample']['gt'],
                              any_query=observed['raw']['covered_gt'], pseudo=observed['selected']['covered_gt'],
                              pseudo_predictions=observed['selected']['predictions'])
                if any(ref[k] != v for k, v in checks.items()):
                    raise ValueError(f'Historical candidate coverage failed to reproduce: {seed}, {checks}, {ref}')
            elif (observed['sample']['images'], observed['sample']['gt']) != (1024, last['rows'][seed]['gt']):
                raise ValueError(f'Unexpected held-out source counts: {seed}')
        headroom[seed] = matching_headroom(rows[seed]['holdout1024'])
        del model, criterion, ema
    supported = [name for name in ('ranking', *GUARDS) if all(row[name] >= 5 for row in headroom.values())]
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               headroom_pp=headroom, common_headroom_above_gate=supported))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
