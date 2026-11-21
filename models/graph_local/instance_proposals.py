"""Category-independent geometry and mixed decoder references for instance discovery."""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

from util.box_ops import box_cxcywh_to_xyxy


def select_encoder_cells(boxes, valid_ids, scores, quota):
    pool = valid_ids[scores[valid_ids].topk(min(len(valid_ids), quota * 12)).indices]
    keep = nms(box_cxcywh_to_xyxy(boxes[pool]).float(), scores[pool].float(), 0.6)[:quota]
    return pool[keep]


class InstanceProposals(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rank_within_level = False
        self.match_encoder_counts = False
        self.finest_anchor_size = 0.05
        self.offset = nn.Linear(dim, 4)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.quality = nn.Sequential(nn.Linear(dim + 4, 32), nn.ReLU(), nn.Linear(32, 1))
        self.position = nn.Linear(4, dim)

    def within_level_scores(self, quality, masks, row):
        # Compare valid cells by within-level rank; ties use the original quality.
        rank = torch.full_like(quality, len(quality), dtype=torch.long)
        offset = 0
        for mask in masks:
            ids = torch.where(~mask[row].flatten())[0] + offset
            order = ids[torch.argsort(quality[ids], descending=True, stable=True)]
            rank[order] = torch.arange(len(order), device=quality.device)
            offset += mask[row].numel()
        by_quality = torch.argsort(quality, descending=True, stable=True)
        ordered = by_quality[torch.argsort(rank[by_quality], stable=True)]
        priority = torch.empty_like(quality, dtype=torch.float32)
        priority[ordered] = torch.arange(len(ordered), 0, -1, device=quality.device, dtype=priority.dtype)
        return priority

    def logits(self, features, boxes):
        return self.quality(torch.cat((features, boxes.detach()), -1)).squeeze(-1)

    def finest_anchor_sides(self, features, mask, xy):
        return torch.full_like(xy, self.finest_anchor_size)

    def forward(self, memory, masks, query, content, references):
        anchors = []
        for level, mask in enumerate(masks):
            height, width = mask.shape[-2:]
            y, x = torch.meshgrid(
                torch.arange(height, device=mask.device),
                torch.arange(width, device=mask.device),
                indexing="ij",
            )
            valid_wh = torch.stack(((~mask[:, 0]).sum(-1), (~mask[:, :, 0]).sum(-1)), -1)
            xy = (torch.stack((x, y), -1).flatten(0, 1) + 0.5) / valid_wh[:, None].clamp_min(1)
            if level == 0:
                sides = self.finest_anchor_sides(memory[:, :height * width], mask, xy)
            else:
                sides = torch.full_like(xy, min(0.8, 0.05 * 2**level))
            anchors.append(torch.cat((xy, sides), -1))
        anchors = torch.cat(anchors, 1)
        valid = ~torch.cat([mask.flatten(1) for mask in masks], 1)
        boxes = (torch.logit(anchors.clamp(0.001, 0.999)) + self.offset(memory)).sigmoid()
        logits = self.logits(memory, boxes)
        from .instance_transfer import SETTINGS

        quota = max(1, int(query.size(1) * SETTINGS["proposal_fraction"]))
        indices, live, natural_counts = [], [], []
        for row in range(len(boxes)):
            with torch.no_grad():
                ids = torch.where(valid[row])[0]
                quality = logits[row]
                if self.match_encoder_counts:
                    global_ids = select_encoder_cells(boxes[row], ids, quality, quota)
                    priority = self.within_level_scores(quality, masks, row)
                    level_ids = select_encoder_cells(boxes[row], ids, priority, quota)
                    counts = (len(global_ids), len(level_ids))
                    natural_counts.append(counts)
                    chosen = (level_ids if self.rank_within_level else global_ids)[:min(counts)]
                else:
                    priority = self.within_level_scores(quality, masks, row) if self.rank_within_level else quality
                    chosen = select_encoder_cells(boxes[row], ids, priority, quota)
                indices.append(F.pad(chosen, (0, quota - len(chosen))))
                live.append(torch.arange(quota, device=memory.device) < len(chosen))
        index = torch.stack(indices)
        active = torch.stack(live)[..., None]
        rows = torch.arange(len(index), device=memory.device)[:, None]
        selected = boxes[rows, index].detach()

        def mix(original, replacement):
            return torch.cat(
                (torch.where(active, replacement, original[:, :quota]), original[:, quota:]), 1
            )

        state = dict(
            boxes=boxes,
            logits=logits,
            anchors=anchors,
            valid=valid,
            memory=memory,
            masks=masks,
            chosen=index,
            active=active.squeeze(-1),
        )
        if self.match_encoder_counts:
            state["natural_counts"] = natural_counts
        return (
            mix(query, self.position(selected)),
            mix(content, memory[rows, index]),
            mix(references, selected[..., :2]),
            state,
        )

    def refined_quality(self, state, boxes):
        # Sample the encoder level whose nominal object scale best fits the refined box.
        masks = state["masks"]
        scale = boxes[..., 2:].detach().prod(-1).clamp_min(1e-8).sqrt()
        level_ids = (scale / 0.05).log2().round().long().clamp(0, len(masks) - 1)
        maps = state["memory"].split([m.shape[1] * m.shape[2] for m in masks], dim=1)
        features = []
        for mask, feature in zip(masks, maps):
            b, h, w = mask.shape
            valid = torch.stack(((~mask[:, 0]).sum(-1) / w, (~mask[:, :, 0]).sum(-1) / h), -1)
            grid = (2 * boxes[..., :2].detach() * valid[:, None] - 1).unsqueeze(2)
            sample = F.grid_sample(
                feature.transpose(1, 2).reshape(b, -1, h, w),
                grid,
                align_corners=False,
                padding_mode="border",
            )
            features.append(sample.squeeze(-1).transpose(1, 2))
        stacked = torch.stack(features, dim=2)
        row = torch.arange(len(boxes), device=boxes.device)[:, None]
        q = torch.arange(boxes.shape[1], device=boxes.device)
        return self.logits(stacked[row, q, level_ids], boxes)


def geometry_loss(state, targets):
    # Only released boxes supervise geometry. Sample close and offset anchors for quality.
    loss = state["boxes"].sum() * 0 + state["logits"].sum() * 0
    count = 0
    for row, target in enumerate(targets):
        gt = target["boxes"]
        if not len(gt):
            continue
        anchors = state["anchors"][row]
        scale = gt[:, 2:].clamp_min(0.01)
        distance = ((gt[:, None, :2] - anchors[None, :, :2]) / scale[:, None]).abs().sum(-1)
        distance += 0.5 * (scale.log()[:, None] - anchors[None, :, 2:].log()).abs().sum(-1)
        valid_ids = torch.where(state["valid"][row])[0]
        near = valid_ids[distance[:, valid_ids].topk(min(16, len(valid_ids)), largest=False).indices]
        pred = state["boxes"][row, near]
        a = box_cxcywh_to_xyxy(pred.detach())
        b = box_cxcywh_to_xyxy(gt[:, None])
        inter = (
            (torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2]))
            .clamp_min(0)
            .prod(-1)
        )
        iou = inter / (pred.detach()[..., 2:].prod(-1) + gt[:, None, 2:].prod(-1) - inter).clamp_min(1e-8)
        loss = loss + F.l1_loss(pred[:, :1], gt[:, None], reduction="sum")
        loss = (
            loss
            + F.binary_cross_entropy_with_logits(state["logits"][row, near], iou, reduction="none")
            .mean(-1)
            .sum()
        )
        count += len(gt)
    return loss / max(count, 1)
