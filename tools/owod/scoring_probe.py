"""Counterfactual scores on released Task 1 training labels; never official AP."""
from types import SimpleNamespace

import torch

from models.deformable_detr import PostProcess
from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import evaluate_author_benchmark


class TrainingScoreAudit:
    """Reuse one forward pass and the author's one-to-one VOC evaluator.

    Task 2 increment annotations omit old classes, so they cannot measure Known
    AP. Keep this optional diagnostic restricted to Task 1 training inputs.
    """

    def __init__(self, coco, image_ids, saved, evaluation):
        if saved.owod_stage != 0 or saved.owod_phase != 'increment':
            raise ValueError('Score audit requires Task 1 increment training labels')
        if not image_ids or len(image_ids) != len(set(image_ids)):
            raise ValueError('Score audit requires a nonempty, unique training sample')
        self.known = list(saved.owod_known_class_ids)
        if self.known != list(saved.owod_current_class_ids):
            raise ValueError('Score audit requires all Known classes to be Current')
        self.coco = SimpleNamespace(imgs={i: coco.imgs[i] for i in image_ids},
            imgToAnns={i: coco.imgToAnns.get(i, []) for i in image_ids})
        if any(a['category_id'] not in self.known for group in self.coco.imgToAnns.values()
               for a in group):
            raise ValueError('Score audit received unreleased training labels')
        self.specification = dict(evaluation, known_only_screening=True,
            image_source_ids=[self.coco.imgs[i]['owod_source_id'] for i in image_ids])
        common = dict(known_ids=self.known, max_detections=saved.owod_max_detections)
        self.processors = {mode: DiscoveryPostProcess(**common, scoring=mode)
                           for mode in ('hierarchical', 'focal')}
        self.processors['known_sigmoid_only'] = PostProcess(**common, include_unknown=False)
        self.predictions = {mode: {} for mode in self.processors}

    @torch.no_grad()
    def add(self, outputs, targets):
        image_ids = [int(t['image_id'].item()) for t in targets]
        if (len(image_ids) != len(set(image_ids)) or any(i not in self.coco.imgs or
                i in self.predictions['hierarchical'] for i in image_ids)):
            raise ValueError('Unexpected or repeated score-audit training image')
        sizes = torch.stack([t['orig_size'] for t in targets])
        for mode, processor in self.processors.items():
            results = processor(outputs, sizes)
            for image_id, prediction in zip(image_ids, results):
                self.predictions[mode][image_id] = {key: prediction[key].detach().cpu()
                                                   for key in ('scores', 'labels', 'boxes')}

    def summarize(self):
        metrics = {}
        for mode, predictions in self.predictions.items():
            if set(predictions) != set(self.coco.imgs):
                raise ValueError('Score audit did not cover the selected training sample')
            result = evaluate_author_benchmark(predictions, self.coco, self.known, [],
                                              self.known, self.specification)
            metrics[mode] = dict(train_known_ap50=result['known_ap50'],
                known_predictions=sum(int((p['labels'] != 91).sum()) for p in predictions.values()),
                unknown_predictions=sum(int((p['labels'] == 91).sum()) for p in predictions.values()))
        return dict(diagnostic_only=True, train_subset_only=True, metrics=metrics,
            note='Counterfactual scores on the same forward outputs and released Task 1 training '
                 'labels; one-to-one VOC matching with duplicates and difficult flags. '
                 'Not official AP or a trained alternative. No unknown recall/precision/WI/A-OSE. '
                 'known_sigmoid_only is an internal detection reference without unknown outputs.')
