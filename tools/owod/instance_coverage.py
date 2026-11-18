"""Read-only Task 1 coverage on the registered, training-disjoint screening images."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.graph_local.discovery import make_ema
from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from tools.owod.gnn_calibration import build_calibration_dataset, load_detector, move_targets
from tools.owod.screening_data import screening_files
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import collate_fn
from main import load_local_checkpoint


@torch.no_grad()
def task1_coverage(checkpoint, manifest, screen):
    payload = load_local_checkpoint(checkpoint)
    args = payload["args"]
    device = torch.device("cuda")
    model, criterion = load_detector(args, checkpoint, device)
    model.eval()
    ema = make_ema(model)
    ema.load_state_dict(payload["discovery_ema"])
    spec, files = screening_files(manifest, screen, 1)
    dataset = build_calibration_dataset(Path(args.owod_image_root), files["full_val"])
    new_ids = torch.tensor(spec["stages"][1]["classes"], device=device)
    known_ids = torch.tensor(model.known_ids, device=device)
    stats = dict(images=0, task2_gt=0, any_query=0, pseudo=0, pseudo_predictions=0)
    for images, targets in DataLoader(dataset, batch_size=2, collate_fn=collate_fn):
        targets = move_targets(targets, device)
        images = images.to(device)
        out = model(images)
        known = []
        for target in targets:
            mask = torch.isin(target["labels"], known_ids)
            known.append({k: target[k][mask] for k in ("labels", "boxes")})
        mined = mine_candidates(
            out,
            known,
            criterion.matcher(out, known),
            model.known_ids,
            ema.forward_teacher(flip_samples(images)),
            criterion.top_k,
            criterion.saliency_threshold,
            criterion.consistency_threshold,
        )
        for row, target in enumerate(targets):
            truth = target["boxes"][torch.isin(target["labels"], new_ids)]
            overlap = (
                box_iou(box_cxcywh_to_xyxy(out["pred_boxes"][row]), box_cxcywh_to_xyxy(truth))[0] >= 0.5
            )
            selected = mined["pseudo"][row]
            stats["images"] += 1
            stats["task2_gt"] += len(truth)
            stats["any_query"] += int(overlap.any(0).sum())
            stats["pseudo"] += int(overlap[selected].any(0).sum())
            stats["pseudo_predictions"] += int(selected.sum())
            if "geometry" in out:
                geometry = out["geometry"]
                boxes = geometry["boxes"][row, geometry["chosen"][row][geometry["active"][row]]]
                geometry_overlap = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0]
                stats["encoder_top_proposals"] = stats.get("encoder_top_proposals", 0) + int(
                    (geometry_overlap >= 0.5).any(0).sum()
                )
    return stats
