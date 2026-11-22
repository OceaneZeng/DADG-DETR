"""Restore selected increment weights at the frozen Task 2 finetune endpoint."""

import argparse
from pathlib import Path
import subprocess

from pycocotools.coco import COCO
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_fp_open_set import OUTPUT as FULL
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_score_margin import OUTPUT as MARGINS
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task2-m-s-current-head-restore')
FFN_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-restore')
LOGITS_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-current-logits')
BOX_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-boxes')
BOX_LOGITS_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-boxes-current-logits')
DISCOVERY_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-discovery-scores')
ROLE_OUTPUT = SOURCE.with_name('task2-m-s-decoder-ffn-native-role-counts')
HEAD_RESULT = 'e984ea7167a3759d1eefa4dc990a18787bce649993621f890aa6699f568ba349'
FFN_RESULT = '41f7f4d1cfc22fd08234925d6665e68f8d61f3f6432e481fb0b1485296060c20'
LOGITS_RESULT = '73cdfdb230dd038eefb895a0d687568eef2e8b7b34f32c74b2e55cdc0c9043d5'
BOX_RESULT = '4221a9a17eee574e42e5fad4df749b1a5360d8448652dfb5ac9fc738993fa585'
BOX_LOGITS_RESULT = '690ac010e50aade1a63fb1e4f069cd517505fa3ccdd6a7fa1c56ecdddbb6cfe6'
DISCOVERY_RESULT = '50b42ecdd07a4b4430155c051241be6bb91b4ebed95ef83d0e152c1009854ee4'
DISCOVERY_SCORES = ('pred_objectness', 'pred_novelty')
MARGIN_RESULT = 'f7db72ce7f691be1999440c4115b943b2313edb035a236b288fb2bbf1bc25eec'
FULL_RESULT = '5626099bf497f1467272789a293eec0e0164a7597bc7a779973b68f326f0ae38'
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')
AP_MARGIN = .005
MIN_RECOVERY = .25


class NativeRoleCounts(DiscoveryPostProcess):
    def __init__(self, previous, current, limit):
        super().__init__(known_ids=previous + current, max_detections=limit, scoring='focal')
        self.roles = (tuple(previous), tuple(current), (91,))

    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        native, _, _ = super().candidate_scores(outputs['native_outputs'])
        flat = scores.flatten(1)
        native_labels = categories[native.flatten(1).topk(self.max_detections, dim=1).indices % len(categories)]
        selected = torch.full_like(flat, -torch.inf)
        ranks = torch.arange(self.max_detections, device=scores.device)
        for role in self.roles:
            labels = categories.new_tensor(role)
            count = torch.isin(native_labels, labels).sum(dim=1, keepdim=True)
            eligible = torch.isin(categories, labels).repeat(scores.shape[1])
            values, indices = flat.masked_fill(~eligible, -torch.inf).topk(self.max_detections, dim=1)
            rows, columns = (ranks < count).nonzero(as_tuple=True)
            selected[rows, indices[rows, columns]] = values[rows, columns]
        return selected.reshape_as(scores), categories, unknown


class RestoredOutputs(torch.nn.Module):
    def __init__(self, native, restored, current, replace_boxes, discovery_scores=False, fixed_role_counts=False):
        super().__init__()
        self.native = native
        self.restored = restored
        self.replace_boxes = replace_boxes
        self.discovery_scores = discovery_scores
        self.fixed_role_counts = fixed_role_counts
        indices = None if current is None else torch.tensor(current, device=next(native.parameters()).device)
        self.register_buffer('current', indices)

    def forward(self, samples):
        output = self.native(samples)
        restored = self.restored(samples)
        if self.fixed_role_counts:
            return dict(restored, native_outputs=output)
        if self.replace_boxes:
            output['pred_boxes'] = restored['pred_boxes']
        if self.current is not None:
            changed = restored['pred_logits'].index_select(-1, self.current)
            output['pred_logits'] = output['pred_logits'].index_copy(-1, self.current, changed)
        if self.discovery_scores:
            for name in DISCOVERY_SCORES:
                output[name] = restored[name]
        return output


def restored_detector(info, restore_ffn=False, current_logits=False, replace_boxes=False, discovery_scores=False,
                      fixed_role_counts=False):
    models = {}
    for phase, endpoint in info['endpoints'].items():
        path = Path(endpoint['path'])
        if file_sha256(path) != endpoint['sha256']:
            raise ValueError(f'Changed frozen endpoint: {path}')
        models[phase], _ = load_detector(None, path, torch.device('cuda'))
    before, after = (models[phase].detr for phase in ('increment', 'finetune'))
    if restore_ffn:
        source = before.transformer.decoder.layers[-1]
        destination = after.transformer.decoder.layers[-1]
        for name in ('linear1', 'linear2', 'norm3'):
            getattr(destination, name).load_state_dict(getattr(source, name).state_dict())
        if current_logits or replace_boxes:
            del models['increment'], before, source
            native, _ = load_detector(None, Path(info['endpoints']['finetune']['path']), torch.device('cuda'))
            current = info['current'] if current_logits else None
            return RestoredOutputs(native, models['finetune'], current, replace_boxes, discovery_scores,
                                   fixed_role_counts).eval()
        return models['finetune'].eval()
    original, restored = before.class_embed[-1], after.class_embed[-1]
    if not isinstance(original, torch.nn.Linear) or not isinstance(restored, torch.nn.Linear):
        raise ValueError('Expected linear final classification heads')
    classes = torch.tensor(info['current'], device=restored.weight.device)
    with torch.no_grad():
        restored.weight.index_copy_(0, classes, original.weight.index_select(0, classes))
        restored.bias.index_copy_(0, classes, original.bias.index_select(0, classes))
    return models['finetune'].eval()


def main(restore_ffn=False, current_logits=False, replace_boxes=False, discovery_scores=False, fixed_role_counts=False):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    margins = verified(MARGINS / 'comparison.json', MARGIN_RESULT)
    margin_plan = verified(MARGINS / 'plan.json', margins['plan_sha256'])
    baseline = verified(FULL / 'comparison.json', FULL_RESULT)
    full_plan = verified(FULL / 'plan.json', baseline['plan_sha256'])
    source = verified(SOURCE / 'plan.json', margin_plan['parent_plan'])
    if (margins['status'] != 'complete' or baseline['status'] != 'complete'
            or full_plan['parent_plan'] != margin_plan['parent_plan']):
        raise ValueError('Changed matched cohorts or missing completed comparison')
    references = {label: file_sha256(FULL / f'{label}.json') for label in full_plan['reports']}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=margin_plan['parent_plan'], margin_result=MARGIN_RESULT, full_result=FULL_RESULT,
                references=references, ap_margin=AP_MARGIN, minimum_recovery=MIN_RECOVERY,
                paper_comparable=False)
    discovery_scores = discovery_scores or fixed_role_counts
    current_logits = current_logits or discovery_scores
    replace_boxes = replace_boxes or discovery_scores
    restore_ffn = restore_ffn or current_logits or replace_boxes
    output = FFN_OUTPUT if restore_ffn else OUTPUT
    if restore_ffn:
        head = verified(OUTPUT / 'comparison.json', HEAD_RESULT)
        prior = verified(OUTPUT / 'plan.json', head['plan_sha256'])
        if head['status'] != 'complete' or any(prior[k] != plan[k] for k in plan if k != 'commit'):
            raise ValueError('Changed classifier restoration reference')
        plan.update(head_result=HEAD_RESULT, decoder_layer=-1, restored_modules=['linear1', 'linear2', 'norm3'])
    if current_logits or replace_boxes:
        previous = verified(FFN_OUTPUT / 'comparison.json', FFN_RESULT)
        previous_plan = verified(FFN_OUTPUT / 'plan.json', previous['plan_sha256'])
        if previous['status'] != 'complete' or any(previous_plan[k] != plan[k] for k in plan if k != 'commit'):
            raise ValueError('Changed decoder FFN restoration reference')
        output = LOGITS_OUTPUT
        plan.update(ffn_result=FFN_RESULT, outputs_from_restoration='current_logits_only')
        if replace_boxes:
            logits = verified(LOGITS_OUTPUT / 'comparison.json', LOGITS_RESULT)
            logits_plan = verified(LOGITS_OUTPUT / 'plan.json', logits['plan_sha256'])
            if logits['status'] != 'complete' or any(logits_plan[k] != plan[k] for k in plan if k != 'commit'):
                raise ValueError('Changed Current-only logit restoration reference')
            output = BOX_OUTPUT
            plan.update(logits_result=LOGITS_RESULT, outputs_from_restoration='boxes_only')
            if current_logits:
                boxes = verified(BOX_OUTPUT / 'comparison.json', BOX_RESULT)
                boxes_plan = verified(BOX_OUTPUT / 'plan.json', boxes['plan_sha256'])
                if boxes['status'] != 'complete' or any(boxes_plan[k] != plan[k] for k in plan if k != 'commit'):
                    raise ValueError('Changed boxes-only restoration reference')
                output = BOX_LOGITS_OUTPUT
                plan.update(boxes_result=BOX_RESULT, outputs_from_restoration='boxes_and_current_logits')
    if discovery_scores:
        combined = verified(BOX_LOGITS_OUTPUT / 'comparison.json', BOX_LOGITS_RESULT)
        combined_plan = verified(BOX_LOGITS_OUTPUT / 'plan.json', combined['plan_sha256'])
        if combined['status'] != 'complete' or any(combined_plan[k] != plan[k] for k in plan if k != 'commit'):
            raise ValueError('Changed boxes and Current-logits restoration reference')
        output = DISCOVERY_OUTPUT
        plan.update(boxes_logits_result=BOX_LOGITS_RESULT,
                    outputs_from_restoration='boxes_current_logits_and_discovery_scores')
    if fixed_role_counts:
        discovery = verified(DISCOVERY_OUTPUT / 'comparison.json', DISCOVERY_RESULT)
        discovery_plan = verified(DISCOVERY_OUTPUT / 'plan.json', discovery['plan_sha256'])
        if discovery['status'] != 'complete' or any(discovery_plan[k] != plan[k] for k in plan if k != 'commit'):
            raise ValueError('Changed discovery-score restoration reference')
        output = ROLE_OUTPUT
        plan.update(discovery_result=DISCOVERY_RESULT, outputs_from_restoration='all_with_native_role_counts')
    plan_path = output / 'plan.json'
    pin(plan_path, plan)
    digest = file_sha256(plan_path)
    results = {}
    for protocol, info in source['protocols'].items():
        model = restored_detector(info, restore_ffn, current_logits, replace_boxes, discovery_scores, fixed_role_counts)
        known = info['previous'] + info['current']
        posts = (dict(native=NativeRoleCounts(info['previous'], info['current'], source['max_detections']))
                 if fixed_role_counts else None)
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        results[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            entry = cohort['by_protocol'][protocol]
            annotation = SOURCE / entry['file']
            verified(annotation, entry['sha256'])
            full = full_plan['reports'][label]
            full_path = FULL / full['file']
            if file_sha256(full_path) != full['sha256']:
                raise ValueError(f'Changed full annotations: {label}')
            path = output / f'{label}.json'
            if not path.exists():
                restored = 'last decoder base FFN and normalization' if restore_ffn else 'Current classifier rows'
                if fixed_role_counts:
                    restored = 'final decoder FFN with native per-image Previous, Current and unknown counts'
                elif discovery_scores:
                    restored = 'boxes, Current logits and discovery scores from the restored final decoder FFN'
                elif current_logits and replace_boxes:
                    restored = 'boxes and Current logits from the restored final decoder FFN'
                elif current_logits:
                    restored = 'Current logits from the restored final decoder FFN'
                elif replace_boxes:
                    restored = 'boxes from the restored final decoder FFN'
                print(f'Evaluating {protocol}/{fold} with restored {restored}', flush=True)
                dataset, loader = make_loader(root, annotation, source['batch_size'], 2)
                spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
                report = measure(model, dataset, loader, known, info['previous'], info['current'], spec,
                                 source['max_detections'], evaluation_coco=COCO(str(full_path)),
                                 include_perturbed=False, extra_posts=posts)
                write_json(path, dict(plan_sha256=digest, **report))
            report = read_json(path)
            reference = verified(FULL / f'{label}.json', references[label])
            if (report['plan_sha256'] != digest or reference['plan_sha256'] != baseline['plan_sha256']
                    or report['per_image'].keys() != reference['per_image'].keys()):
                raise ValueError(f'Changed matched evaluation: {label}')
            if fixed_role_counts and any(counts['native'] != reference['per_image'][image]['native']
                                         for image, counts in report['per_image'].items()):
                raise ValueError(f'Changed fixed per-image output counts: {label}')
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            wanted = margins['protocols'][protocol][fold]['rows']
            targets = {pairs[row['target']]['annotation_ids'][protocol] for row in wanted}
            native, restored = (row['views']['native'] for row in (reference, report))
            if len(targets) != len(wanted) or targets.intersection(native['matched']):
                raise ValueError(f'Changed dropped Current targets: {label}')
            deltas = {key: restored['metrics'][key] - native['metrics'][key] for key in METRICS}
            recovered = sorted(targets.intersection(restored['matched']))
            passed = (len(recovered) >= MIN_RECOVERY * len(targets)
                      and deltas['current_ap50'] >= AP_MARGIN and deltas['previous_ap50'] >= -AP_MARGIN
                      and deltas['known_ap50'] >= 0 and deltas['a_ose'] <= 0)
            results[protocol][fold] = dict(deltas=deltas, targets=len(targets), recovered=recovered, passed=passed)
        del model
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=results, paper_comparable=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--restore-ffn', action='store_true')
    mode.add_argument('--current-logits', action='store_true')
    mode.add_argument('--boxes-only', action='store_true')
    mode.add_argument('--boxes-current-logits', action='store_true')
    mode.add_argument('--discovery-scores', action='store_true')
    mode.add_argument('--fixed-role-counts', action='store_true')
    args = parser.parse_args()
    main(args.restore_ffn, args.current_logits or args.boxes_current_logits,
         args.boxes_only or args.boxes_current_logits, args.discovery_scores, args.fixed_role_counts)
