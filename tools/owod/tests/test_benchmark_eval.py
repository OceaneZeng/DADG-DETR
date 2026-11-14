import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.owod_benchmark_eval import evaluate_author_benchmark, voc_ap
from tools.owod.benchmark_spec import specification


class AuthorEvaluationTests(unittest.TestCase):
    def evaluate(self, objects, labels, boxes, scores=None, mode='integral', screening=False):
        anns = [{'category_id': label, 'voc_bbox': box, 'difficult': difficult}
                for label, box, difficult in objects]
        coco = SimpleNamespace(imgs={1: {'owod_source_id': 'one'}}, imgToAnns={1: anns})
        predictions = {1: {'labels': torch.tensor(labels),
                            'boxes': torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4),
                            'scores': torch.tensor(scores if scores is not None else [.9] * len(labels))}}
        spec = {**specification('m-owodb'), 'image_source_ids': ['one'], 'ap_mode': mode}
        spec['known_only_screening'] = screening
        return evaluate_author_benchmark(predictions, coco, [1], [], [1], spec)

    def test_screening_reports_known_ap_without_unknown_claims(self):
        stats = self.evaluate([(1, [1, 1, 10, 10], False)],
                              [1, 91], [[0, 0, 10, 10], [20, 0, 30, 10]], screening=True)
        self.assertEqual(stats, {'known_ap50': 1., 'current_ap50': 1.})

    def test_unknown_matching_is_one_to_one(self):
        known, unknown = [1, 1, 10, 10], [21, 1, 30, 10]
        stats = self.evaluate([(1, known, False), (5, unknown, False), (5, unknown, False)],
                              [1, 91], [[0, 0, 10, 10], [20, 0, 30, 10]])
        self.assertEqual(stats['u_recall'], .5)
        self.assertEqual(stats['unknown_precision'], 1.)
        self.assertEqual(stats['a_ose'], 0.)

    def test_aose_counts_misclassified_predictions_and_wi_uses_known_curve(self):
        stats = self.evaluate([(1, [1, 1, 10, 10], False), (5, [21, 1, 30, 10], False)],
                              [1, 1, 1], [[20, 0, 30, 10], [20, 0, 30, 10], [0, 0, 10, 10]],
                              [.99, .98, .9])
        self.assertEqual(stats['a_ose'], 2.)
        self.assertAlmostEqual(stats['wi'], 2 / 3)
        self.assertAlmostEqual(stats['known_ap50'], 1 / 3)

    def test_difficult_objects_are_not_recall_denominator_and_task4_omits_unknown(self):
        stats = self.evaluate([(1, [1, 1, 10, 10], False), (5, [21, 1, 30, 10], True)],
                              [1, 91], [[0, 0, 10, 10], [20, 0, 30, 10]])
        self.assertEqual(stats['unknown_gt'], 0.)
        self.assertNotIn('u_recall', stats)
        self.assertNotIn('wi', stats)

    def test_integral_and_voc07_ap_are_distinct(self):
        recall, precision = np.array([.5, 1.]), np.array([1., .5])
        self.assertAlmostEqual(voc_ap(recall, precision, 'integral'), .75)
        self.assertAlmostEqual(voc_ap(recall, precision, 'voc2007_11point'), 8.5 / 11.)

    def test_parity_with_actual_owdetr_author_voc_evaluator(self):
        self.skipTest('The external OW-DETR checkout is intentionally not vendored in this repository')


if __name__ == '__main__':
    unittest.main()
