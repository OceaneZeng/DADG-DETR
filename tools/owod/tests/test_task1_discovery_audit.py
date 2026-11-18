from collections import Counter, defaultdict
from types import SimpleNamespace

import torch

from tools.analysis.audit_task1_discovery import frozen_slots, score_image


def test_mining_coverage_and_slot_composition_have_different_denominators():
    memory = SimpleNamespace(known_slots=1, counts=torch.tensor([4., 0., 1., 4.]),
                             centers=torch.tensor([[0., 0.], [1., 0.], [1., 0.], [0., 1.]]))
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., 1.]])
    slots, counts = frozen_slots(SimpleNamespace(memory=memory), features)
    assert slots.tolist() == [1, 2, -1, 2]
    a, b, c = [0., 0., .1, .1], [.4, .4, .6, .6], [.9, .9, 1., 1.]
    annotations = [(10, a), (11, b), (1, c)]
    boxes = torch.tensor([a, a, c, c, b])
    pseudo = torch.tensor([True, True, True, True, False])
    classes, groups = defaultdict(Counter), defaultdict(Counter)
    score_image(boxes, pseudo, slots, counts, 3, annotations, {10, 11}, classes, groups)
    assert dict(classes[10]) == dict(gt=1, any_query=1, pseudo=1, mature_slot=1)
    assert dict(classes[11]) == dict(gt=1, any_query=1, pseudo=0, mature_slot=0)
    assert groups == {'1': {'task2_class_10': 1}, '2': {'task2_class_10': 1, 'task1_known': 1},
                      'no_existing_slot': {'task1_known': 1}}
    # Images without Task 1/2 boxes still contribute pseudo proposals to the denominator.
    score_image(boxes, pseudo, slots, counts, 3, [], {10, 11}, classes, groups)
    assert groups['2']['unresolved'] == 2
    empty, _ = frozen_slots(SimpleNamespace(memory=memory), features[:0])
    score_image(boxes, ~torch.ones_like(pseudo), empty, counts, 3, annotations, {10, 11}, classes, groups)
    assert dict(classes[10]) == dict(gt=2, any_query=2, pseudo=1, mature_slot=1)
