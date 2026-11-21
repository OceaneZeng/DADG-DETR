"""Test old-class duplicate suppression at a fixed native unknown budget."""
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch
from torchvision.ops import batched_nms

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_score_competition import ARMS, MANIFEST, OUTPUT as SOURCE
from tools.owod.run_s_task2_voc_precision import OUTPUT as PRECISION
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('s-instance-task2-old-dedup')
LIMIT, OLD_POOL, DEDUP_IOU = 50, 300, .70


class OldClassRefill(DiscoveryPostProcess):
    def __init__(self, known, previous):
        super().__init__(known_ids=known, max_detections=LIMIT, scoring='focal')
        self.old_columns = [i for i, label in enumerate(known) if label in previous]

    def candidate_scores(self, outputs):
        probabilities, categories, unknown = super().candidate_scores(outputs)
        columns = len(categories)
        flat = probabilities.flatten(1)
        native = flat.topk(LIMIT, dim=1).indices
        locked_unknown = torch.zeros_like(flat, dtype=torch.bool)
        locked_unknown.scatter_(1, native, native % columns == columns - 1)
        known_quota = LIMIT - locked_unknown.sum(dim=1)

        old_columns = categories.new_tensor(self.old_columns)
        old_scores = probabilities[..., old_columns].flatten(1)
        scores, positions = old_scores.topk(OLD_POOL, dim=1)
        queries = positions // len(old_columns)
        labels = old_columns[positions % len(old_columns)]
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        filtered = probabilities.clone()
        filtered[..., -1] = -torch.inf
        for row, query in enumerate(queries):
            kept = batched_nms(boxes[row, query] + 1., scores[row], labels[row], DEDUP_IOU)
            suppressed = torch.ones_like(query, dtype=torch.bool)
            suppressed[kept] = False
            filtered[row, query[suppressed], labels[row, suppressed]] = -torch.inf

        remaining_known = filtered.flatten(1).topk(LIMIT, dim=1).indices
        within_quota = torch.arange(LIMIT, device=flat.device)[None, :] < known_quota[:, None]
        keep_known = torch.zeros_like(locked_unknown).scatter_(1, remaining_known, within_quota)
        budgeted = flat.masked_fill(~(keep_known | locked_unknown), -torch.inf).reshape_as(probabilities)
        return budgeted, categories, unknown


def endpoint(view):
    stats = view['previous_class_stats']
    return dict(metrics=view['metrics'], composition=view['composition'],
                duplicate_fp=sum(r['fp_duplicates'] for r in stats.values()),
                fixed_match_ap50_without_duplicate_fp=mean(r['ap50_without_duplicate_fp'] for r in stats.values()))


def main():
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for frozen inference')
    source = read_json(SOURCE / 'plan.json')
    scoring = read_json(SOURCE / 'comparison.json')
    precision = read_json(PRECISION / 'comparison.json')
    annotation, manifest = Path(source['annotation']), read_json(MANIFEST)
    eval_order = [r['owod_source_id'] for r in read_json(annotation)['images']]
    if (scoring['status'] != 'complete' or scoring['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or
            precision['status'] != 'complete' or not precision['rank_follow_up'] or
            precision['plan_sha256'] != file_sha256(PRECISION / 'plan.json') or
            read_json(PRECISION / 'plan.json')['source_scoring_sha256'] != file_sha256(SOURCE / 'comparison.json') or
            file_sha256(MANIFEST) != source['manifest_sha256'] or file_sha256(annotation) != source['annotation_sha256'] or
            sha256('\n'.join(eval_order).encode()).hexdigest() != source['eval_order_sha256']):
        raise ValueError('Changed S duration sources or missing precision evidence')
    checkpoints = source['checkpoints']
    for entry in checkpoints.values():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Changed checkpoint: {entry["path"]}')
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', paper_comparable=False, seed=source['seed'], images=source['images'], batch_size=2,
                gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__), device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                annotation=str(annotation), annotation_sha256=source['annotation_sha256'], eval_order_sha256=source['eval_order_sha256'],
                manifest_sha256=source['manifest_sha256'], scoring_sha256=file_sha256(SOURCE / 'comparison.json'),
                precision_sha256=file_sha256(PRECISION / 'comparison.json'), checkpoints=checkpoints,
                max_detections=LIMIT, old_pool=OLD_POOL, nms_iou=DEDUP_IOU, unknown_policy='native query identities, boxes and scores',
                criterion='On the 6-epoch arm improvement in Previous >=0.75 pp, Current >=-0.25 pp, Known >0 pp, '
                          'and fewer duplicate FPs. A pass merits independent training-source validation.')
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    evaluate_pair(OUTPUT, plan, manifest, annotation, eval_order, scoring['arms'])


def evaluate_pair(output, plan, manifest, annotation, eval_order, source_arms=None):
    plan_hash = file_sha256(output / 'plan.json')
    stage = manifest['stages'][1]
    known, previous, current = stage['active_classes'], manifest['stages'][0]['active_classes'], stage['classes']
    variants = {name: (known, 'focal', LIMIT) for name in ('native', 'dedup')}
    detector = OldClassRefill(known, previous)
    spec = dict(manifest['evaluation'], image_source_ids=eval_order, known_only_screening=True)
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    rows, deltas = {}, {}
    for arm in ARMS:
        entry = plan['checkpoints'][arm]
        path = output / f'{arm}.json'
        if not path.is_file():
            print(f'Evaluating native and duplicate-suppressed {arm}', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            views = evaluate_views(model, loader, dataset.coco, variants, spec, previous, current, 'cuda',
                                   include_previous_stats=True, postprocessors={'dedup': detector}, print_freq=64)
            write_json(path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], views=views))
            del model
        report = read_json(path)
        rows[arm] = {name: endpoint(report['views'][name]) for name in variants}
        native, dedup = rows[arm]['native'], rows[arm]['dedup']
        if (report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != entry['sha256'] or
                (source_arms is not None and native['metrics'] != source_arms[arm]['native']) or
                any(row['composition']['detections'] != LIMIT * plan['images'] for row in (native, dedup)) or
                native['composition']['unknown'] != dedup['composition']['unknown']):
            raise ValueError(f'Native baseline or fixed quota mismatch for {arm}')
        deltas[arm] = {key: 100*(dedup['metrics'][key]-value) for key, value in native['metrics'].items()}
        deltas[arm]['duplicate_fp'] = dedup['duplicate_fp'] - native['duplicate_fp']
    result = deltas['6epochs']
    follow_up = (result['previous_ap50'] >= .75 and result['current_ap50'] >= -.25 and
                 result['known_ap50'] > 0 and result['duplicate_fp'] < 0)
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
               arms=rows, dedup_minus_native=deltas, follow_up=follow_up))
    print('Comparison saved:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
