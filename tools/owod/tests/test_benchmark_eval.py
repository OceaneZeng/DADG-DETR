import ast
import functools
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
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

    def test_unknown_threshold_is_defensive_for_external_predictions(self):
        anns = [
            {'category_id': 1, 'voc_bbox': [1, 1, 10, 10], 'difficult': False},
            {'category_id': 5, 'voc_bbox': [21, 1, 30, 10], 'difficult': False},
        ]
        coco = SimpleNamespace(imgs={1: {'owod_source_id': 'one'}}, imgToAnns={1: anns})
        predictions = {1: {
            'labels': torch.tensor([91, 1]),
            'boxes': torch.tensor([[20, 0, 30, 10], [0, 0, 10, 10]], dtype=torch.float64),
            'scores': torch.tensor([.99, .9]),
            'unknown_scores': torch.tensor([.2, .2]),
        }}
        spec = {**specification('m-owodb'), 'image_source_ids': ['one']}
        stats = evaluate_author_benchmark(predictions, coco, [1], [], [1], spec,
                                          unknown_threshold=.5)
        self.assertEqual(stats['unknown_predictions'], 0.)
        self.assertEqual(stats['u_recall'], 0.)

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
        path = Path(__file__).resolve().parents[3] / 'baselines/ow-detr/datasets/open_world_eval.py'
        if not path.is_file():
            self.skipTest('author submodule not initialized')
        source = path.read_text().replace('np.bool)', 'np.bool_)')
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'parse_rec', 'voc_ap', 'voc_eval'}]
        namespace = {'np': np, 'ET': ET, 'functools': functools, 'os': os}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / 'one.xml'
            xml.write_text('''<annotation><object><name>person</name><difficult>0</difficult>
<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
<object><name>airplane</name><difficult>0</difficult>
<bndbox><xmin>21</xmin><ymin>1</ymin><xmax>30</xmax><ymax>10</ymax></bndbox></object></annotation>''')
            lines = ['one 0.990 21.0 1.0 30.0 10.0', 'one 0.900 1.0 1.0 10.0 10.0']
            for mode in ('integral', 'voc2007_11point'):
                rec, prec, ap, aose, nunk, count, fpopen = namespace['voc_eval'](
                    lines, [str(xml)], ['one'], 'person', use_07_metric=mode == 'voc2007_11point',
                    known_classes=['person'])
                stats = self.evaluate([(1, [1, 1, 10, 10], False), (5, [21, 1, 30, 10], False)],
                                      [1, 1], [[20, 0, 30, 10], [0, 0, 10, 10]], [.99, .9], mode)
                self.assertAlmostEqual(stats['known_ap50'], ap)
                self.assertEqual(stats['a_ose'], aose)
                self.assertEqual(stats['unknown_gt'], nunk)
                index = int(np.argmin(abs(rec - .8)))
                self.assertAlmostEqual(stats['wi'], fpopen[index] / count[index])


if __name__ == '__main__':
    unittest.main()
