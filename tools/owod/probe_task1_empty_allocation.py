"""Pair one proposal from empty images with one removable proposal in the same training source."""
from pathlib import Path
import subprocess

import torch

from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_soft_discovery as soft
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task1-empty-allocation'
SLICES = ('reference128', 'holdout1024')


def transfer_pairs(rows):
    # Only the observed quota, image ID and frozen proposal score determine the exchanges.
    receivers = sorted((r for r in rows if r['quota'] == 0 and r['query'] is not None),
                       key=lambda r: (-r['score'], r['image']))
    donors = sorted((r for r in rows if r['quota'] >= 2), key=lambda r: (r['score'], r['image']))
    pairs = []
    for receiver, donor in zip(receivers, donors):
        if receiver['score'] <= donor['score']:
            break
        pairs.append((receiver['image'], donor['image']))
    return pairs


def summarize(rows):
    pairs = transfer_pairs(rows)
    images = {r['image']: r for r in rows}
    gain = sum(images[i]['added_hit'] for i, _ in pairs)
    loss = sum(images[j]['removed_hit'] for _, j in pairs)
    baseline = sum(r['original_hits'] for r in rows)
    # Without score ordering, one useful receiver and one zero-cost donor give exactly one net hit.
    possible = sum(r['can_hit'] for r in rows if r['quota'] == 0)
    free = sum(r['can_spare'] for r in rows if r['quota'] >= 2)
    return dict(pairs=pairs, counts=dict(images=len(rows), gt=sum(r['gt'] for r in rows),
                proposals=sum(r['quota'] for r in rows), original_hits=baseline, transferred_hits=baseline + gain - loss,
                swaps=len(pairs), added_hits=gain, removed_hits=loss,
                empty_images=sum(r['quota'] == 0 for r in rows), coverable_empty_images=possible,
                no_loss_donors=free, oracle_net_gain=min(possible, free)))


@torch.inference_mode()
def collect(model, criterion, ema, loader, current, sources, previous):
    rows = {}
    new_ids = torch.tensor(current, device='cuda')
    for step, (targets, output, traces, mined) in enumerate(gates.candidate_batches(model, criterion, ema, loader), 1):
        for i, target in enumerate(targets):
            image = int(target['image_id'].item())
            score, eligible = soft.continuous_evidence(output, mined, traces[i], i)
            selected = mined['pseudo'][i].cpu()
            ids = selected.nonzero().flatten()
            quota = len(ids)
            q = None
            if quota == 0 and eligible.any():
                q = int(score.masked_fill(~eligible, -torch.inf).argmax())
            elif quota >= 2:
                q = int(ids[score.cpu()[ids].argmin()])
            truth = target['boxes'][torch.isin(target['labels'], new_ids)]
            boxes = output['pred_boxes'][i]
            edges = (box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0] >= .5).cpu()
            hits = gates.measure_boxes(edges, selected, quota)['one_to_one']
            ref = previous[str(image)]
            if (image in rows or ids.tolist() != ref['query_ids']['original'] or hits != ref['hits']['original']
                    or len(truth) != ref['gt']):
                raise ValueError(f'Changed original image selections or matches: {image}')
            row = dict(image=image, source=sources[image], quota=quota, original_hits=hits, gt=len(truth), query=q,
                       score=float(score[q]) if q is not None else None,
                       box=boxes[q].tolist() if q is not None else None)
            if quota == 0:
                row.update(added_hit=int(q is not None and edges[q].any()),
                           can_hit=int(edges[eligible.cpu()].any()))
            elif quota >= 2:
                loss = {}
                for idx in ids.tolist():
                    remaining = selected.clone()
                    remaining[idx] = False
                    loss[idx] = hits - gates.measure_boxes(edges, remaining, quota - 1)['one_to_one']
                row.update(removed_hit=loss[q], can_spare=int(min(loss.values()) == 0))
            rows[image] = row
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} images', flush=True)
    if set(rows) != set(sources) or len(rows) != len(previous):
        raise ValueError('Incomplete image pairing')
    return rows


def main():
    source = read_json(soft.OUTPUT / 'plan.json')
    ref = read_json(soft.OUTPUT / 'comparison.json')
    manifest = read_json(MANIFEST)
    large = holdout.OUTPUT / 'heldout.json'
    if (ref['status'] != 'complete' or ref['plan_sha256'] != file_sha256(soft.OUTPUT / 'plan.json')
            or source['manifest_sha256'] != file_sha256(MANIFEST) or source['holdout_sha256'] != file_sha256(large)
            or source['torch_version'] != str(torch.__version__) or source['gpu'] != torch.cuda.get_device_name()):
        raise ValueError('Original per-image audit and its recorded execution environment are required')
    refs = {}
    for seed, job in source['jobs'].items():
        if any(file_sha256(Path(job[k])) != job[k + '_sha256'] for k in ('parent', 'small')):
            raise ValueError(f'Registered input for seed {seed} changed')
        refs[seed] = {label: file_sha256(soft.OUTPUT / f'seed{seed}-{label}.json') for label in SLICES}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=ref['plan_sha256'], source_result_sha256=file_sha256(soft.OUTPUT / 'comparison.json'),
                jobs=source['jobs'], per_image_sha256=refs, manifest_sha256=source['manifest_sha256'],
                holdout_sha256=source['holdout_sha256'], batch_size=source['batch_size'], gpu=source['gpu'],
                torch_version=source['torch_version'], paper_comparable=False,
                rule='Within each author training-source stratum, transfer one proposal from an image with >=2 original selections to an empty image only if the best admissible candidate outscores the weakest original donor. One exchange per image.',
                criterion='Both seeds gain >=5 pp one-to-one coverage on holdout1024 and neither loses hits on reference128.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    memberships = {name: set(manifest['splits'][name + '_train']) for name in ('t1', 't2')}
    rows, passed = {}, {}
    for seed, job in source['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        rows[seed] = {}
        for label, anno in zip(SLICES, (Path(job['small']), large)):
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                dataset, loader = make_loader(root, anno, plan['batch_size'], 2)
                previous = read_json(soft.OUTPUT / f'seed{seed}-{label}.json')
                if previous['plan_sha256'] != ref['plan_sha256'] or previous['totals'] != ref['rows'][seed][label]:
                    raise ValueError('Per-image exposure record does not match its completed comparison')
                sources = {i: '_'.join(k for k, ids in memberships.items() if m['owod_source_id'] in ids)
                           for i, m in dataset.coco.imgs.items()}
                if '' in sources.values():
                    raise ValueError('Unexpected image outside author training sources')
                print(f'Seed {seed}, {label}: evaluating empty receivers and donors on {len(dataset)} images', flush=True)
                images = collect(model, criterion, ema, loader, manifest['stages'][1]['classes'], sources, previous['images'])
                groups = {k: summarize([r for r in images.values() if r['source'] == k]) for k in sorted(set(sources.values()))}
                totals = {k: sum(g['counts'][k] for g in groups.values()) for k in next(iter(groups.values()))['counts']}
                write_json(path, dict(plan_sha256=digest, groups=groups, images=images, totals=totals))
            part = read_json(path)
            totals = part['totals']
            base = ref['rows'][seed][label]['original']
            if (part['plan_sha256'] != digest or totals['proposals'] != base['proposals']
                    or totals['gt'] != base['gt'] or totals['original_hits'] != base['hits']
                    or totals['transferred_hits'] - totals['original_hits'] > totals['oracle_net_gain']):
                raise ValueError(f'Unpaired result or invalid ceiling: {path}')
            rows[seed][label] = totals
        small, big = (rows[seed][k] for k in SLICES)
        passed[seed] = (100 * (big['transferred_hits'] - big['original_hits']) >= 5 * big['gt']
                        and small['transferred_hits'] >= small['original_hits'])
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows, passed_by_seed=passed,
                                               passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
