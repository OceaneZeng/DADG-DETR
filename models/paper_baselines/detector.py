"""Ports OW-DETR's three components and implements EW-DETR from its paper.

OW reference: akshitac8/OW-DETR@3515ff8c36687a4f582ef6a2c83866966b56ce23.
Dataset, schedules, masking and geometry compatibility changes are documented.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.deformable_detr import build as build_base, SetCriterion
from models.segmentation import sigmoid_focal_loss
from util import box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized
from .cat_modules import AdaptivePseudoLabeler, cat_pseudo_queries
from .ew_modules import DualLoRA, QueryNormEUMix, inject_dual_lora, merge_beta
from .objectness import ProbObjectnessHead, SketchObjectnessHead


class PaperDetector(nn.Module):
    def __init__(self, detector, args):
        super().__init__()
        self.detr = detector
        self.method = args.paper_baseline
        self.known_ids = list(args.owod_known_class_ids)
        self.unknown_id = 91
        detector.return_baseline_features = True
        if self.method in ('ow-detr', 'cat', 'prob', 'owobj'):
            if self.method in ('ow-detr', 'cat'):
                self.objectness = nn.Linear(args.hidden_dim, 1)
                nn.init.constant_(self.objectness.bias, -math.log(99))
            if self.method == 'cat':
                detector.cascade_decoder = True
                self.cat_adaptive = AdaptivePseudoLabeler(
                    memory_size=args.cat_loss_memory,
                    recent_size=args.cat_recent_window,
                    start_iteration=args.cat_start_iteration,
                    update_interval=args.cat_update_interval,
                    positive_momentum=args.cat_positive_momentum,
                    negative_momentum=args.cat_negative_momentum)
            elif self.method == 'prob':
                self.prob_objectness = nn.ModuleList(
                    [ProbObjectnessHead(args.hidden_dim) for _ in range(args.dec_layers)])
            elif self.method == 'owobj':
                self.prob_objectness = nn.ModuleList(
                    [SketchObjectnessHead(args.hidden_dim, args.owobj_sketch_sigma)
                     for _ in range(args.dec_layers)])
        else:
            self.adapter_layers = inject_dual_lora(detector, args.ew_rank)
            self.calibration = QueryNormEUMix(args.hidden_dim, self.known_ids)

    def forward(self, samples):
        output = self.detr(samples)
        features = output.pop('decoder_features')
        layers = [*output.get('aux_outputs', []), output]
        layer_features = features if len(layers) > 1 else features[-1:]
        heads = self.detr.class_embed if len(layers) > 1 else self.detr.class_embed[-1:]
        for index, (layer, hidden, head) in enumerate(zip(layers, layer_features, heads)):
            if self.method in ('ow-detr', 'cat', 'prob', 'owobj'):
                logits = head(hidden) if self.method == 'cat' else layer['pred_logits']
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[self.known_ids + [self.unknown_id]] = False
                layer['pred_logits'] = logits.masked_fill(mask, -1e8)
                if self.method in ('ow-detr', 'cat'):
                    layer['pred_objectness'] = self.objectness(hidden)
                elif self.method == 'prob':
                    layer['pred_obj'] = self.prob_objectness[index](hidden)
                else:
                    clean, sketch = self.prob_objectness[index](hidden)
                    layer['pred_obj'] = clean
                    layer['pred_obj_sketch'] = sketch
            else:
                layer['pred_logits'] = self.calibration(hidden, head)
        if self.method == 'cat':
            output.pop('location_decoder_features')
        if self.method == 'ew-detr':
            output.pop('attention_feature')
            output.pop('padded_size')
        return output

    @torch.no_grad()
    def consolidate(self, current_samples, previous_samples):
        beta = merge_beta(current_samples, previous_samples)
        for module in self.modules():
            if isinstance(module, DualLoRA):
                module.consolidate(beta)
        return beta


@torch.no_grad()
def attention_pseudo_queries(feature, boxes, matched, padded_size, top_k=5):
    """Integral-image box means, with clipped valid boxes and GT queries excluded."""
    height, width = padded_size
    attention = F.interpolate(feature.mean(1, keepdim=True), size=(height, width),
                              mode='bilinear', align_corners=False)[:, 0]
    integral = F.pad(attention.cumsum(1).cumsum(2), (1, 0, 1, 0))
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    coords = (xyxy * boxes.new_tensor([width, height, width, height])).long()
    coords[..., 0::2].clamp_(0, width)
    coords[..., 1::2].clamp_(0, height)
    selected = []
    for batch, (src, _) in enumerate(matched):
        x1, y1, x2, y2 = coords[batch].unbind(-1)
        area = (x2 - x1) * (y2 - y1)
        sums = (integral[batch, y2, x2] - integral[batch, y1, x2]
                - integral[batch, y2, x1] + integral[batch, y1, x1])
        valid = area > 0
        valid[src] = False
        candidates = torch.where(valid)[0]
        means = sums[candidates] / area[candidates]
        chosen = candidates[means.topk(min(top_k, candidates.numel())).indices]
        selected.append(chosen)
    return selected


class PaperCriterion(SetCriterion):
    def __init__(self, original, args, detector=None):
        super().__init__(92, original.matcher, dict(original.weight_dict), original.losses,
                         original.focal_alpha)
        self.method = args.paper_baseline
        self.known_ids = list(args.owod_known_class_ids)
        self.epoch = 0
        self.warmup = args.ow_pseudo_warmup
        self.top_k = args.ow_top_unknown
        self.cat_adaptive = getattr(detector, 'cat_adaptive', None)
        if self.method in ('ow-detr', 'cat'):
            self.weight_dict['loss_NC'] = 0.1
            for layer in range(args.dec_layers - 1):
                self.weight_dict[f'loss_NC_{layer}'] = 0.1
        if self.method in ('prob', 'owobj'):
            self.weight_dict['loss_obj_ll'] = args.prob_objectness_coef
            for layer in range(args.dec_layers - 1):
                self.weight_dict[f'loss_obj_ll_{layer}'] = args.prob_objectness_coef
        if self.method == 'owobj':
            self.weight_dict['loss_energy'] = args.owobj_energy_coef
            for layer in range(args.dec_layers - 1):
                self.weight_dict[f'loss_energy_{layer}'] = args.owobj_energy_coef

    def known_targets(self, targets):
        filtered = []
        for target in targets:
            mask = torch.zeros_like(target['labels'], dtype=torch.bool)
            for category in self.known_ids:
                mask |= target['labels'] == category
            filtered.append({**target, 'labels': target['labels'][mask],
                             'boxes': target['boxes'][mask]})
        return filtered

    def forward(self, outputs, targets):
        # Full validation labels remain available to the evaluator, never matcher/loss.
        targets = self.known_targets(targets)
        if self.method in ('ew-detr', 'prob', 'owobj'):
            losses = super().forward(outputs, targets)
            if self.method in ('prob', 'owobj'):
                count = outputs['pred_logits'].new_tensor(
                    [sum(len(t['labels']) for t in targets)])
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(count)
                count = (count / get_world_size()).clamp(min=1).item()
                for index, layer in enumerate([outputs, *outputs.get('aux_outputs', [])]):
                    matched = self.matcher(layer, targets)
                    indices = self._get_src_permutation_idx(matched)
                    energy = layer['pred_obj'][indices]
                    suffix = '' if index == 0 else f'_{index - 1}'
                    losses[f'loss_obj_ll{suffix}'] = energy.clamp_min(
                        -256 * math.log(0.9)).sum() / count
                    if self.method == 'owobj':
                        sketch = layer['pred_obj_sketch'][indices]
                        losses[f'loss_energy{suffix}'] = (
                            sketch - energy).abs().mean()
                        known = layer['pred_logits'][..., self.known_ids]
                        unknown = layer['pred_logits'][..., 91]
                        e_in = -torch.logsumexp(known, dim=-1)
                        e_out = -unknown
                        losses[f'loss_energy{suffix}'] = losses[f'loss_energy{suffix}'] + F.relu(
                            e_out - e_in + 0.2).mean()
            return losses
        count = outputs['pred_logits'].new_tensor([sum(len(t['labels']) for t in targets)])
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(count)
        count = (count / get_world_size()).clamp(min=1).item()
        losses = {}
        for index, layer in enumerate([outputs, *outputs.get('aux_outputs', [])]):
            matched = self.matcher(layer, targets)
            classification_targets = copy.deepcopy(targets)
            classification_indices = matched
            if self.training and self.epoch >= self.warmup:
                if self.method == 'cat':
                    selected = cat_pseudo_queries(
                        outputs['attention_feature'], layer['pred_boxes'], matched,
                        [target['proposal_boxes'] for target in targets],
                        outputs['padded_size'], self.cat_adaptive.model_weight,
                        self.cat_adaptive.input_weight, self.top_k)
                else:
                    selected = attention_pseudo_queries(
                        outputs['attention_feature'], layer['pred_boxes'], matched,
                        outputs['padded_size'], self.top_k)
                classification_indices = []
                for target, (src, dst), queries in zip(classification_targets, matched, selected):
                    offset = len(target['labels'])
                    target['labels'] = torch.cat([target['labels'], target['labels'].new_full((len(queries),), 91)])
                    classification_indices.append((torch.cat([src.to(queries.device), queries]),
                        torch.cat([dst.to(queries.device), torch.arange(offset, offset + len(queries), device=queries.device)])))
            layer_losses = self.loss_labels(layer, classification_targets, classification_indices, count, log=index == 0)
            layer_losses.update(self.loss_boxes(layer, targets, matched, count))
            foreground = torch.zeros_like(layer['pred_objectness'])
            foreground[self._get_src_permutation_idx(classification_indices)] = 1
            layer_losses['loss_NC'] = sigmoid_focal_loss(layer['pred_objectness'], foreground, count,
                                                        alpha=self.focal_alpha, gamma=2) * foreground.shape[1]
            suffix = '' if index == 0 else f'_{index - 1}'
            losses.update({key + suffix: value for key, value in layer_losses.items()})
        if self.method == 'cat' and self.training:
            controller_loss = sum(
                losses[name] * self.weight_dict[name]
                for name in ('loss_ce', 'loss_bbox', 'loss_giou', 'loss_NC'))
            if is_dist_avail_and_initialized():
                controller_loss = controller_loss.detach().clone()
                torch.distributed.all_reduce(controller_loss)
                controller_loss /= get_world_size()
            self.cat_adaptive.observe(controller_loss)
        return losses


class PaperPostProcess(nn.Module):
    def __init__(self, known_ids, threshold, max_detections=100):
        super().__init__()
        self.known_ids = list(known_ids)
        self.threshold = threshold
        self.max_detections = max_detections

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        logits = outputs['pred_logits']
        # Sparse COCO slots stay intact; 91 is a dedicated unknown category.
        categories = torch.tensor(self.known_ids + ([] if len(self.known_ids) == 80 else [91]), device=logits.device)
        probabilities = logits[..., categories].sigmoid()
        scores, indices = probabilities.flatten(1).topk(
            min(self.max_detections, probabilities[0].numel()), dim=1)
        queries = indices // len(categories)
        labels = categories[indices % len(categories)]
        boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
        boxes = boxes.gather(1, queries[..., None].expand(-1, -1, 4))
        height, width = target_sizes.unbind(1)
        boxes *= torch.stack([width, height, width, height], -1)[:, None]
        unknown = logits[..., 91].sigmoid().gather(1, queries)
        return [dict(scores=s, labels=l, boxes=b, unknown_scores=u, unknown_mask=u >= self.threshold)
                for s, l, b, u in zip(scores, labels, boxes, unknown)]


class EnergyPostProcess(nn.Module):
    """Convert PROB/OWOBJ energies to the evaluator's common result schema."""

    def __init__(self, known_ids, threshold, temperature=1.3, max_detections=100,
                 multiply_objectness=True):
        super().__init__()
        self.known_ids = list(known_ids)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.max_detections = int(max_detections)
        self.multiply_objectness = bool(multiply_objectness)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        logits = outputs['pred_logits']
        energies = outputs['pred_obj']
        categories = torch.tensor(self.known_ids, device=logits.device)
        known_prob = logits[..., categories].sigmoid()
        if self.multiply_objectness:
            object_prob = torch.exp(-self.temperature * energies).unsqueeze(-1)
            scores_by_class = known_prob * object_prob
        else:
            scores_by_class = known_prob
        flat = scores_by_class.flatten(1)
        count = min(self.max_detections, flat.shape[1])
        scores, flat_indices = flat.topk(count, dim=1)
        queries = flat_indices // len(categories)
        labels = categories[flat_indices % len(categories)]
        boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
        boxes = boxes.gather(1, queries[..., None].expand(-1, -1, 4))
        height, width = target_sizes.unbind(1)
        boxes *= torch.stack([width, height, width, height], -1)[:, None]
        max_known = known_prob.amax(-1)
        # Energy is a background score; combine it with the classifier gap so
        # known high-confidence detections are not counted as unknowns.
        energy_unknown = 1.0 - torch.exp(-self.temperature * energies).clamp(0, 1)
        unknown = (energy_unknown * (1.0 - max_known)).gather(1, queries)
        return [dict(scores=s, labels=l, boxes=b, unknown_scores=u,
                     unknown_mask=u >= self.threshold)
                for s, l, b, u in zip(scores, labels, boxes, unknown)]


def build(args):
    if args.paper_baseline == 'ew-detr' and args.replay_sampling_fraction:
        raise ValueError('EW-DETR is exemplar-free; replay is not allowed')
    if ((args.lr_backbone == 0 and not getattr(args, 'no_pretrained_backbone', False))
            or args.neighbor_scoped_lora
            or args.teacher_completion or args.old_class_distillation or args.masks
            or args.two_stage or args.with_box_refine or args.num_feature_levels < 3
            or args.local_margin_coef or args.off_projection_coef):
        raise ValueError('Paper baselines require the isolated one-stage detector configuration')
    if args.num_classes != 92 or not args.owod_known_class_ids:
        raise ValueError('Expected sparse COCO labels with 92 slots (unknown=91)')
    if args.paper_baseline == 'cat' and not args.cat_proposals:
        raise ValueError('CAT requires --cat-proposals generated by prepare_cat_proposals.py')
    detector, criterion, _ = build_base(args)
    model = PaperDetector(detector, args)
    if args.paper_baseline in ('prob', 'owobj'):
        postprocessor = EnergyPostProcess(
            args.owod_known_class_ids, args.unknown_threshold,
            max_detections=getattr(args, 'owod_max_detections', 100),
            # The paper scales temperature by the decoder feature dimension.
            temperature=args.prob_objectness_temperature / args.hidden_dim,
            multiply_objectness=args.paper_baseline == 'prob')
    else:
        postprocessor = PaperPostProcess(
            args.owod_known_class_ids, args.unknown_threshold,
            max_detections=50 if args.paper_baseline == 'cat' else getattr(args, 'owod_max_detections', 100))
    return model, PaperCriterion(criterion, args, model), {'bbox': postprocessor}
