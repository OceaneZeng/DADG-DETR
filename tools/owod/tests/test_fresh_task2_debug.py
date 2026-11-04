"""Fresh diagnostic branches consume Task 1 knowledge, never failed Task 2 weights."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.owod.debug_discovery_revision import task2_arguments, prepare_task2


def test_task2_arguments_preserve_parent_and_use_author_stage_one_files():
    parent = SimpleNamespace(owod_stage=0, owod_phase='increment', owod_known_class_ids=[1, 2],
                             graph_adapter_rank=8, seed=42)
    before = copy.deepcopy(vars(parent))
    manifest = dict(image_root='/images', stages=[dict(active_classes=[1, 2]),
                    dict(active_classes=[1, 2, 3], classes=[3])])
    files = dict(increment_train=Path('/task2/train.json'), full_val=Path('/task2/val.json'))
    args = task2_arguments(parent, manifest, files, Path('/manifest.json'), Path('/task1/gnn.pt'))
    assert vars(parent) == before
    assert args.owod_stage == 1 and args.owod_previous_class_ids == [1, 2]
    assert args.owod_current_class_ids == [3]
    assert Path(args.train_ann) == Path('/task2/train.json')
    assert Path(args.graph_gnn_checkpoint) == Path('/task1/gnn.pt')
    manifest['stages'][0]['active_classes'].reverse()
    with pytest.raises(ValueError, match='class order'):
        task2_arguments(parent, manifest, files, Path('/manifest.json'), Path('/task1/gnn.pt'))


def test_fresh_transition_uses_official_graph_and_promotion_without_training(monkeypatch):
    from models.graph_local.discovery import DiscoveryDetector
    from models.graph_local.tests.test_discovery import args
    from models.graph_local.tests.test_routed_adapter import Detector, spec
    configuration = args()
    configuration.owod_known_class_ids = [1, 3]
    model = DiscoveryDetector(Detector(), configuration)
    criterion = SimpleNamespace(known_ids=[1], base=SimpleNamespace(active_class_ids=[1]))
    before = copy.deepcopy(model.state_dict())
    seen = []
    def build(model_arg, criterion_arg, args_arg, source):
        assert source == Path('/task1.pth')
        assert model_arg.known_ids == criterion_arg.known_ids == [1, 3]
        seen.append('official-risk')
        return spec(classes=(3,), old=(1,))
    def collect(*unused):
        seen.append('released-train-centroids')
        return {3: torch.ones(4)}, {3: [torch.ones(4)]}
    monkeypatch.setattr('tools.owod.graph_adaptation.build_stage_spec', build)
    monkeypatch.setattr('tools.owod.graph_adaptation.collect_promotion_features', collect)
    layout, means, routes = prepare_task2(model, criterion, configuration, Path('/task1.pth'))
    assert seen == ['official-risk', 'released-train-centroids']
    assert layout['class_ids'] == [3] and means[3] == [1.] * 4
    assert routes[3] == [[1.] * 4]
    # Existing Task 1 weights and prototypes survive layout construction intact.
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_server_launcher_has_no_task2_checkpoint_dependency():
    launcher = Path(__file__).resolve().parents[1] / 'run_discovery_debug.sh'
    source = launcher.read_text()
    assert '--start-task2' in source
    assert 'stage_0/increment/checkpoint0049.pth' in source
    assert 'stage_1/increment/checkpoint' not in source
