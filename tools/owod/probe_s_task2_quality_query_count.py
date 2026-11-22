"""Measure the effect of extra encoder queries under the frozen fitted quality ranking."""
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from models.graph_local.instance_proposals import select_encoder_cells
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_cell_pair_quality import OUTPUT as SOURCE
from tools.owod.probe_s_task2_shortlist_depth import OUTPUT as NATIVE, STAGES, compare, record_geometry
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-instance-task2-quality-query-count')
CRITERION = ('With the fitted rank head fixed, truncate its selected encoder prefix to the actual native count '
             'on each image and restore displaced learned queries. Confirm both natural selections by cell ID. '
             'Attribute a substantial part of the proxy recall deficit to the count change only if truncation '
             'recovers at least half of the native-minus-fitted gap in mean two-fold U-Recall on each slice. '
             'Report full known and open metrics, and retain the original head unless both slices also pass '
             'the existing geometry, known and open gates against native. No refitting.')
RANK_HASH = {'original': '908ce4dadfa1b13a3a0cc2348206e2db26636392102b57e0b1c8400e517dcc0a',
             'independent': '27a881085a5cfcb50f50acbf2c016d31d670b054794048c6ea13b791a740a5e9'}


def prepare():
    src = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != '0474181c60d3097d3468819e6d89d2a7f12385225a3d4efa07d8d43f90b5fc30'
            or result['status'] != 'complete' or result['follow_up'] or file_sha256(SOURCE/'plan.json') != result['plan_sha256']
            or file_sha256(SOURCE/'fit.json') != result['fit_sha256'] or file_sha256(MANIFEST) != src['manifest_sha256']
            or file_sha256(Path(src['checkpoint']['path'])) != src['checkpoint']['sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed quality fit, reference, checkpoint, protocol or runtime')
    for name, entry in src['slices'].items():
        if (file_sha256(Path(entry['annotation'])) != entry['annotation_sha256']
                or file_sha256(NATIVE/f'{name}-native.json') != src['native_sha256'][name]
                or file_sha256(SOURCE/f'{name}-rank.json') != RANK_HASH[name]
                or file_sha256(Path(entry['native_targets'])) != entry['native_targets_sha256']
                or file_sha256(Path(entry['heldout']['path'])) != entry['heldout']['sha256']):
            raise ValueError(f'Changed held-out references on {name}')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version', 'tail_class_ids', 'native_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=file_sha256(SOURCE/'comparison.json'), fit_sha256=result['fit_sha256'],
                natural_rank_sha256=RANK_HASH, criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan


def match_counts(native_head, module, inputs, output):
    memory, _, *learned = inputs
    *mixed, geo = output
    logits = native_head(torch.cat((memory, geo['boxes']), -1)).squeeze(-1)
    quota = geo['chosen'].shape[1]
    native_ids = []
    for i in range(len(memory)):
        valid = torch.where(geo['valid'][i])[0]
        native_ids.append(select_encoder_cells(geo['boxes'][i], valid, logits[i], quota, module.shortlist_multiplier))
    counts = torch.tensor([len(ids) for ids in native_ids], device=memory.device)
    keep = torch.arange(quota, device=memory.device)[None, :] < counts[:, None]
    if (keep & ~geo['active']).any():
        raise ValueError('Fitted head has fewer available queries than the native head')
    tensors = [torch.cat((torch.where(keep[..., None], value[:, :quota], original[:, :quota]), value[:, quota:]), 1)
               for value, original in zip(mixed, learned)]
    natural = [geo['chosen'][i, active] for i, active in enumerate(geo['active'])]
    return (*tensors, dict(geo, native_ids=native_ids, natural_ids=natural, active=keep,
                           chosen=geo['chosen'].masked_fill(~keep, 0)))


def trace(hits, choices, dataset, native, fitted, tail, threshold, outputs, targets, batch_images, score_tables):
    geo = outputs['geometry']
    for i, image_id in batch_images.items():
        key = dataset.coco.imgs[image_id]['owod_source_id']
        if geo['native_ids'][i].tolist() != native[key] or geo['natural_ids'][i].tolist() != fitted[key]:
            raise ValueError(f'Native or fitted cell sequence replay changed on {key}')
    record_geometry(hits, choices, dataset.coco, tail, 12, threshold, outputs, targets, batch_images, score_tables)


def proxy_recall(views):
    return mean(views[key]['metrics']['u_recall'] for key in ('proxy_even', 'proxy_odd'))


def main():
    plan = prepare()
    plan_hash = file_sha256(OUTPUT/'plan.json')
    manifest = read_json(MANIFEST)
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    model.requires_grad_(False).eval()
    proposal = model.detr.transformer.instance_proposals
    if proposal.shortlist_multiplier != 12 or proposal.rank_within_level or proposal.match_encoder_counts:
        raise ValueError('Expected original proposal settings')
    original = deepcopy(proposal.quality)
    weights = read_json(SOURCE/'fit.json')['heads']['rank']
    proposal.quality.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
    summaries = {}
    for name, entry in plan['slices'].items():
        native = read_json(NATIVE/f'{name}-native.json')
        fitted = read_json(SOURCE/f'{name}-rank.json')
        path = OUTPUT/f'{name}.json'
        if not path.exists():
            ann = Path(entry['annotation'])
            ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), ann, plan['batch_size'], 2)
            order = [im['owod_source_id'] for im in read_json(ann)['images']]
            hits, choices = {stage: set() for stage in STAGES}, {}
            observer = partial(trace, hits, choices, ds, native['choices'], fitted['choices'],
                               set(plan['tail_class_ids']), manifest['evaluation']['iou_threshold'])
            print(f'Evaluating {name} with matched encoder counts on {len(ds)} images', flush=True)
            with proposal.register_forward_hook(partial(match_counts, original)):
                views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, fp_sources=True,
                                         rank_observer=observer)
            write_json(path, dict(plan_sha256=plan_hash, views=views, hits={k: sorted(v) for k, v in hits.items()},
                                  choices=choices))
        matched = read_json(path)
        expected = {k: fitted['choices'][k][:len(ids)] for k, ids in native['choices'].items()}
        if (matched['plan_sha256'] != plan_hash or matched['choices'] != expected
                or any(len(expected[k]) != len(ids) for k, ids in native['choices'].items())
                or matched['hits']['shortlist'] != fitted['hits']['shortlist']):
            raise ValueError(f'Unpaired fitted ranking or native query budget on {name}')
        against_native = compare(native, matched, entry)
        base, free, paired = (proxy_recall(e['views']) for e in (native, fitted, matched))
        if base <= free:
            raise ValueError('Expected a positive native-minus-fitted recall gap')
        recovery = (paired-free)/(base-free)
        summaries[name] = dict(native_mean_recall=base, fitted_mean_recall=free, matched_mean_recall=paired,
                               recovery_fraction=recovery, count_supported=recovery >= .5,
                               changed_count_images=sum(len(ids) != len(expected[k]) for k, ids in fitted['choices'].items()),
                               versus_native=against_native)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=plan_hash,
                                               slices=summaries, count_supported=all(v['count_supported'] for v in summaries.values()),
                                               follow_up=all(v['versus_native']['follow_up'] for v in summaries.values())))
    print('Matched encoder query count comparison saved', flush=True)


if __name__ == '__main__':
    main()
