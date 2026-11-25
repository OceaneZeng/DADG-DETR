"""Compare two novelty readouts of a single frozen S Task 1 detector."""

import copy
import subprocess
from pathlib import Path

import torch
from torch import nn

from main import load_local_checkpoint
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_duplicate_novelty_audit import CANDIDATE_SHA
from tools.owod.probe_s_task1_duplicate_novelty_geometry import OUTPUT as GEOMETRY
from tools.owod.probe_s_task1_rank_independent import PROTOCOL, evaluate as evaluate_released
from tools.owod.probe_s_task1_same_query_competition import METRICS
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_duplicate_novelty_rank import OUTPUT as DONOR
from tools.owod.run_s_task1_known_score_order import verified_endpoint
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = GEOMETRY.with_name('task1-s-novelty-head-transfer')
GEOMETRY_SHA = '6a11e2b8058183bddccdfa64be90fdca69b564bfaefdf4a2fdaa4406b54ea44d'


def prepare():
    result = verified(GEOMETRY / 'comparison.json', GEOMETRY_SHA)
    source = verified(GEOMETRY / 'plan.json', result['plan_sha256'])
    donor = verified(DONOR / 'comparison.json', CANDIDATE_SHA)
    training = verified(DONOR / 'plan.json', donor['plan_sha256'])
    shared = {key: source[key] for key in ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit')}
    if (result['status'] != 'complete' or result['decision'] != 'all_query_geometry_loss'
            or donor['status'] != 'complete' or donor['passed']
            or donor['checkpoint_sha256'] != source['arms']['candidate']['sha256']
            or any(training[key] != value for key, value in shared.items())):
        raise ValueError('Expected the pinned S endpoints with a verified localization gap')
    for arm in source['arms'].values():
        if file_sha256(Path(arm['checkpoint'])) != arm['sha256']:
            raise ValueError('Changed a frozen S checkpoint')
    plan = dict(**shared, **{key: training[key] for key in ('min_known_ap50', 'folds', 'fold_native',
                                                           'max_wi_increase', 'max_aose_fraction')},
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                geometry_sha256=GEOMETRY_SHA, arms=source['arms'], paper_comparable=False,
                intervention='Only the donor novelty linear projection reads the frozen half-gradient embedding.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


class FrozenNoveltyViews(nn.Module):
    def __init__(self, detector, donor_checkpoint):
        super().__init__()
        self.detector = detector
        self.donor = copy.deepcopy(detector.affinity_graph.novelty)
        state = load_local_checkpoint(donor_checkpoint)['model']
        self.donor.load_state_dict({key: state[f'affinity_graph.novelty.{key}'] for key in self.donor.state_dict()})
        self.requires_grad_(False)

    def forward(self, samples):
        alternate = []

        def capture(_head, inputs, _scores):
            alternate.append(self.donor(inputs[0]).squeeze(-1))

        with self.detector.affinity_graph.novelty.register_forward_hook(capture):
            outputs = self.detector(samples)
        outputs['transferred_novelty'], = alternate
        return outputs


class TransferredNoveltyPostprocess(CompeteWithEmittedUnknown):
    def candidate_scores(self, outputs):
        return super().candidate_scores(dict(outputs, pred_novelty=outputs['transferred_novelty']))


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen novelty-head comparison')
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    control, donor = (plan['arms'][key] for key in ('control', 'candidate'))
    binding = dict(plan_sha256=digest, checkpoint_sha256=control['sha256'], donor_sha256=donor['sha256'])
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    device = torch.device('cuda')
    detector, _ = load_detector(None, Path(control['checkpoint']), device)
    model = FrozenNoveltyViews(detector, Path(donor['checkpoint']))
    head = detector.affinity_graph.novelty
    head_stats = dict(weight_cosine=torch.cosine_similarity(head.weight, model.donor.weight).item(),
                      native_bias=head.bias.item(), donor_bias=model.donor.bias.item())
    known = manifest['stages'][0]['classes']
    settings = dict(known_ids=known, scoring='focal', max_detections=plan['limit'])
    grafted = TransferredNoveltyPostprocess(**settings)
    released = Path(plan['holdout'])
    if file_sha256(released) != plan['holdout_sha256']:
        raise ValueError('Changed released-only holdout')
    dest = OUTPUT / 'released.json'
    if not dest.is_file():
        print('Evaluating frozen S novelty transfer on the released-only holdout', flush=True)
        report = evaluate_released(model, manifest, released, plan, extra_postprocessors={'transferred': grafted})
        write_json(dest, dict(**binding, report=report))
    report = verified_endpoint(dest, binding)
    before = report['emitted']['metrics']['known_ap50']
    after = report['transferred']['metrics']['known_ap50']
    small = report['released_small_learned_query_coverage']
    if (before != control['expected_ap'] or small != control['expected_small']
            or report['raw_current_box_recall_iou50'] != control['expected_raw']):
        raise ValueError('Native half-gradient released holdout did not reproduce')
    released_result = dict(native=before, transferred=after, delta=after - before, small=small,
                           passed=after > plan['min_known_ap50'])

    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    processors = dict(native=CompeteWithEmittedUnknown(**settings), transferred=grafted)
    views = {key: (known, 'focal', plan['limit']) for key in processors}
    folds = {}
    for name, entry in plan['folds'].items():
        path = Path(entry['path'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'Changed S {name} fold')
        dest = OUTPUT / f'{name}.json'
        if not dest.is_file():
            print(f'Evaluating frozen S novelty transfer on the {name} fold', flush=True)
            dataset, loader = make_loader(root, path, plan['batch_size'], 2)
            spec = dict(manifest['evaluation'],
                        image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
            report = evaluate_views(model, loader, dataset.coco, views, spec, [], known, device,
                                    include_current_stats=True, postprocessors=processors, print_freq=128)
            write_json(dest, dict(**binding, report=report))
        report = verified_endpoint(dest, binding)
        metrics = {view: {key: report[view]['metrics'][key] for key in METRICS} for view in views}
        native = metrics['native']
        if native != plan['fold_native'][name]:
            raise ValueError(f'Native half-gradient metrics did not reproduce on {name}')
        delta = {key: metrics['transferred'][key] - native[key] for key in METRICS}
        gates = dict(known_ap50=delta['known_ap50'] >= 0., u_recall=delta['u_recall'] >= 0.,
                     wi=delta['wi'] <= plan['max_wi_increase'],
                     a_ose=delta['a_ose'] <= max(1., plan['max_aose_fraction'] * native['a_ose']))
        folds[name] = dict(metrics=metrics, delta=delta, gates=gates, passed=all(gates.values()))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', **binding, released=released_result,
                                               folds=folds, head_stats=head_stats,
                                               passed=released_result['passed'] and all(f['passed'] for f in folds.values())))


if __name__ == '__main__':
    main()
