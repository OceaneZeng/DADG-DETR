import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.owod.protocol import build_official_manifest, file_sha256
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.tests.test_protocol import make_tree, fixture_split


def arguments(manifest, stage=0, phase='increment'):
    return SimpleNamespace(owod_manifest=str(manifest), owod_stage=stage, owod_phase=phase,
                           owod_recipe='ore-paper',
                           owod_known_class_ids=None, owod_previous_class_ids=None,
                           owod_current_class_ids=None, paper_baseline=None,
                           train_ann='', val_ann='', lightweight=False, no_augmentation=False,
                           replay_sampling_fraction=0, pretrained='', resume='', backbone_weights='')


class StrictSettingTests(unittest.TestCase):
    def test_diagnostic_checkpoint_is_never_a_formal_parent(self):
        with self.assertRaisesRegex(ValueError, 'Diagnostic checkpoints'):
            validate_checkpoint(SimpleNamespace(), {'diagnostic_only': True})
    def test_discovery_version_and_message_mode_cannot_change_in_chain(self):
        import copy
        from models.graph_local.discovery import DISCOVERY_VERSION
        args = SimpleNamespace(owod_manifest='verified.json', owod_manifest_sha256='same',
                               owod_recipe='ow-detr-paper', owod_discovery=True,
                               discovery_version=DISCOVERY_VERSION, discovery_no_message=False)
        for field, wrong in [('discovery_version', DISCOVERY_VERSION - 1), ('discovery_no_message', True)]:
            saved = copy.deepcopy(args)
            setattr(saved, field, wrong)
            with self.assertRaisesRegex(ValueError, field):
                validate_checkpoint(args, {'args': saved})
        saved = copy.deepcopy(args)
        saved.discovery_scoring = 'focal'
        with self.assertRaisesRegex(ValueError, 'discovery_scoring'):
            validate_checkpoint(args, {'args': saved})

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_phase_files_derive_labels_and_reject_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            args = arguments(path)
            configure_run(args)
            self.assertEqual(len(args.owod_known_class_ids), 20)
            self.assertTrue(args.train_ann.endswith('instances_increment_train2017.json'))
            self.assertEqual(args.owod_manifest_sha256, file_sha256(path))
            with self.assertRaisesRegex(ValueError, 'old/custom'):
                validate_checkpoint(args, {'args': SimpleNamespace(owod_stage=0)})
            args = arguments(path, stage=1, phase='finetune')
            args.pretrained = 'stage_1_increment.pth'
            configure_run(args)
            self.assertTrue(args.train_ann.endswith('instances_train2017.json'))
            source = SimpleNamespace(owod_manifest_sha256=args.owod_manifest_sha256,
                                     owod_stage=1, owod_phase='increment', epochs=50,
                                     owod_recipe='ore-paper')
            validate_checkpoint(args, {'args': source, 'epoch': 49})
            with self.assertRaisesRegex(ValueError, 'completed preceding phase'):
                validate_checkpoint(args, {'args': source, 'epoch': 48})
            source.owod_recipe = 'ow-detr-paper'
            with self.assertRaisesRegex(ValueError, 'recipe changed'):
                validate_checkpoint(args, {'args': source, 'epoch': 49})
            source.owod_recipe = 'ore-paper'
            source.owod_phase = 'finetune'
            with self.assertRaisesRegex(ValueError, 'Expected checkpoint'):
                validate_checkpoint(args, {'args': source, 'epoch': 49})

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_future_class_metadata_and_custom_replay_setting_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            args = arguments(path)
            args.owod_known_class_ids = list(range(1, 21))
            with self.assertRaisesRegex(ValueError, 'author-defined'):
                configure_run(args)
            args = arguments(path)
            args.replay_sampling_fraction = .1
            with self.assertRaisesRegex(ValueError, 'retired'):
                configure_run(args)

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_graph_checkpoint_method_and_layout_are_bound_to_protocol(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            args = arguments(path, stage=1, phase='finetune')
            args.pretrained = 'increment.pth'
            args.graph_routed_adapter = True
            args.neighbor_scoped_lora = args.reset_classifier = args.masks = args.two_stage = False
            args.frozen_weights = None
            args.off_projection_coef = 0
            args.dec_layers = 6
            args.graph_adapter_rank = 8
            args.graph_adapter_last_n = 2
            args.graph_risk_beta = 2.
            args.graph_route_loss_coef = 1.
            args.graph_sketch_max_images = 12
            configure_run(args)
            saved = copy.deepcopy(args)
            saved.owod_phase = 'increment'
            saved.epochs = 50
            saved.graph_adapter_specs = [dict(stage=1, class_ids=args.owod_current_class_ids,
                                              old_ids=args.owod_previous_class_ids,
                                              rank=8, last_n=2, risk_beta=2.)]
            payload = {'args': saved, 'epoch': 49}
            validate_checkpoint(args, payload)
            args.graph_localization = 'bbox-last'
            with self.assertRaisesRegex(ValueError, 'graph_localization'):
                validate_checkpoint(args, payload)
            args.graph_localization = 'frozen'
            saved.graph_adapter_specs[0]['class_ids'] = [999]
            with self.assertRaisesRegex(ValueError, 'layout'):
                validate_checkpoint(args, payload)
            saved.graph_routed_adapter = False
            with self.assertRaisesRegex(ValueError, 'switch'):
                validate_checkpoint(args, payload)

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_run_plan_has_separate_phases_and_previous_finetuning_chain(self):
        from tools.owod.run_benchmark import build_plan
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            for stem in ('100001', '100002', '900001'):
                (root / (stem + '.jpg')).write_bytes(b'fixture path only')
            weights = root / 'dino.pth'
            weights.write_bytes(b'fixture path only')
            args = SimpleNamespace(manifest=path, recipe='ow-detr-paper', finetune_epochs=None,
                                   backbone_weights=weights, output_dir=root / 'output', seed=42,
                                   nproc_per_node=2, batch_size=2, num_workers=0)
            plan = build_plan(args)
            args.task1_only = True
            task1 = build_plan(args)
            self.assertEqual(len(task1['runs']), 1)
            self.assertEqual(task1['runs'][0]['stage'], 0)
            self.assertEqual(task1['runs'][0]['epochs'], 50)
            for flag in ('--pretrained', '--resume', '--owod-discovery', '--graph-routed-adapter', '--owod-screening'):
                self.assertNotIn(flag, task1['runs'][0]['command'])
            args.owod_screening = root / 'screen.json'
            with self.assertRaisesRegex(ValueError, 'task1-only'):
                build_plan(args)
            args.owod_screening = None
            args.task1_only = False
            self.assertEqual([r['phase'] for r in plan['runs']],
                             ['increment', 'increment', 'finetune', 'increment', 'finetune', 'increment', 'finetune'])
            self.assertEqual([r['epochs'] for r in plan['runs']], [50, 50, 20, 50, 20, 50, 20])
            command = plan['runs'][3]['command']
            self.assertIn('stage_1/finetune/checkpoint.pth', command[command.index('--pretrained') + 1].replace('\\', '/'))
            args.method = 'graph-adapter'
            args.graph_adapter_rank = 8
            args.graph_adapter_last_n = 2
            args.graph_risk_beta = 2.
            args.graph_route_loss_coef = 1.
            args.graph_sketch_max_images = 12
            graph_plan = build_plan(args)
            calibrations = [r for r in graph_plan['runs'] if r.get('kind') == 'calibration']
            training = [r for r in graph_plan['runs'] if r.get('kind') != 'calibration']
            self.assertEqual(len(calibrations), 3)
            self.assertEqual([r['phase'] for r in training], [r['phase'] for r in plan['runs']])
            self.assertEqual([r['epochs'] for r in training], [r['epochs'] for r in plan['runs']])
            for calibration, index in zip(calibrations, (0, 2, 4)):
                command = calibration['command']
                self.assertEqual(command[command.index('--checkpoint') + 1],
                                 str(Path(training[index]['output_dir']) / 'checkpoint.pth'))
            for run in training:
                self.assertIn('--graph-routed-adapter', run['command'])
                if run['stage'] > 0:
                    self.assertIn('--old-class-distillation', run['command'])
            args.method = 'discovery-scratch'
            scratch_plan = build_plan(args)
            for run in scratch_plan['runs']:
                if run.get('kind') != 'calibration':
                    self.assertIn('--owod-discovery', run['command'])
                    self.assertIn('--discovery-no-transfer', run['command'])
                    if run['stage'] > 0:
                        self.assertIn('--old-class-distillation', run['command'])
            args.method = 'discovery-no-message'
            message_plan = build_plan(args)
            for run in message_plan['runs']:
                if run.get('kind') != 'calibration':
                    self.assertIn('--owod-discovery', run['command'])
                    self.assertIn('--discovery-no-message', run['command'])
                    self.assertNotIn('--discovery-no-transfer', run['command'])


if __name__ == '__main__':
    unittest.main()
