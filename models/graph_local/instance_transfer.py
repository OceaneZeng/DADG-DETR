"""Frozen stage-boundary instance retrieval for discovery v10."""

from collections import Counter, defaultdict
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import util.misc as utils
from util.box_ops import box_iou, box_cxcywh_to_xyxy
from .discovery_candidates import flip_samples, mine_candidates


# Serialized into each checkpoint and experiment plan. These are fixed pilot settings.
SETTINGS = dict(
    proposal_fraction=0.25,
    projection_dim=64,
    slots=16,
    per_slot=4,
    bank_images=512,
    query_images=1536,
    slot_cosine=0.85,
    min_similarity=0.45,
    margin=0.02,
    localization_iou=0.5,
    independent_images=2,
    alignment_weight=0.25,
)


def bank_settings(expanded=False):
    settings = dict(SETTINGS)
    if expanded:
        settings.update(bank_images=4096, slots=32, per_slot=8)
    return settings


def paired_iou(boxes, truth):
    return box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth))[0].diagonal()


class Representatives:
    def __init__(self, settings=None):
        self.settings = SETTINGS if settings is None else settings
        self.slots = []

    def add(self, feature, image, confidence, quality):
        entry = (image, feature, confidence, quality)
        if self.slots:
            centers = F.normalize(
                torch.stack([torch.stack([e[1] for e in slot]).mean(0) for slot in self.slots])
            )
            similarities = centers @ feature
            index = int(similarities.argmax())
            if similarities[index] >= self.settings["slot_cosine"] or len(self.slots) == self.settings["slots"]:
                slot = self.slots[index]
                if similarities[index] < self.settings["slot_cosine"]:
                    return
                same = next((j for j, e in enumerate(slot) if e[0] == image), None)
                if same is not None:
                    if confidence > slot[same][2]:
                        slot[same] = entry
                elif len(slot) < self.settings["per_slot"]:
                    slot.append(entry)
                else:
                    weakest = min(range(len(slot)), key=lambda j: slot[j][2])
                    if confidence > slot[weakest][2]:
                        slot[weakest] = entry
                return
        self.slots.append([entry])

    def query(self, feature, image):
        candidates = []
        for index, slot in enumerate(self.slots):
            entries = [e for e in slot if e[0] != image]
            if len(entries) < self.settings["independent_images"]:
                continue
            vectors = torch.stack([e[1] for e in entries])
            center = F.normalize(vectors.mean(0), dim=0)
            scores = vectors @ feature
            selected = scores >= self.settings["min_similarity"]
            if selected.sum() < self.settings["independent_images"]:
                continue
            weights = (scores[selected] / 0.1).softmax(0)
            target = F.normalize(weights @ vectors[selected], dim=0)
            sources = [entry[0] for entry, keep in zip(entries, selected.tolist()) if keep]
            candidates.append((float(center @ feature), index, target, sources, weights.tolist()))
        candidates.sort(key=lambda item: -item[0])
        if not candidates or candidates[0][0] < self.settings["min_similarity"]:
            return None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < self.settings["margin"]:
            return None
        return candidates[0]


def sample_loader(args, stage, phase, budget):
    from util.training_runtime import loader_worker_options
    from tools.owod.screening_data import experiment_files
    from tools.owod.gnn_calibration import build_calibration_dataset

    _, files = experiment_files(args, stage)
    dataset = build_calibration_dataset(Path(args.owod_image_root), files[phase], source_objects=True)
    indices = random.Random(args.seed + stage).sample(range(len(dataset)), min(len(dataset), budget))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        shuffle=False,
        drop_last=False,
        **loader_worker_options(args.num_workers),
    )


@torch.no_grad()
def calibration(parent, criterion, args, payload):
    from scipy.optimize import linear_sum_assignment
    from .discovery import make_ema
    from tools.owod.gnn_calibration import move_targets

    device = next(parent.parameters()).device
    previous = args.owod_stage - 1
    ema = make_ema(parent)
    ema.load_state_dict(payload["discovery_ema"], strict=True)
    settings = args.instance_settings
    bank = Representatives(settings)
    scanned_images = 0
    count = 0
    for images, targets in sample_loader(
        args, previous, "increment_train" if previous == 0 else "train", settings["bank_images"]
    ):
        images = images.to(device)
        targets = move_targets(targets, device)
        out = parent(images)
        scanned_images += len(targets)
        match = criterion.matcher(out, targets)
        other = ema.forward_teacher(flip_samples(images))
        mined = mine_candidates(
            out,
            targets,
            match,
            parent.known_ids,
            other,
            criterion.top_k,
            criterion.saliency_threshold,
            criterion.consistency_threshold,
        )
        for row, target in enumerate(targets):
            image = int(target["image_id"].item())
            selected = torch.where(mined["pseudo"][row])[0]
            for j in selected.tolist():
                bank.add(
                    out["instance_features"][row, j].cpu(),
                    image,
                    float(mined["weights"][row, j]),
                    float(out["geometry_quality"][row, j]),
                )
            count += len(selected)
    del ema
    groups = defaultdict(list)
    localized = 0
    query_gt, localized_gt = Counter(), Counter()
    for images, targets in sample_loader(
        args, args.owod_stage, "increment_train", settings["query_images"]
    ):
        targets = move_targets(targets, device)
        out = parent(images.to(device))
        for row, target in enumerate(targets):
            # Future classifier rows in the frozen parent have no released semantics.
            query_gt.update(target["labels"].tolist())
            overlap = box_iou(
                box_cxcywh_to_xyxy(out["pred_boxes"][row]), box_cxcywh_to_xyxy(target["boxes"])
            )[0]
            q, gt = linear_sum_assignment(-overlap.cpu().numpy())
            q = torch.as_tensor(q, device=device)
            gt = torch.as_tensor(gt, device=device)
            ious = overlap[q, gt]
            image = int(target["image_id"].item())
            for idx in torch.where(ious >= settings["localization_iou"])[0].tolist():
                localized += 1
                g = gt[idx]
                localized_gt[int(target["labels"][g])] += 1
                result = bank.query(out["instance_features"][row, q[idx]].cpu(), image)
                if result:
                    similarity, slot, feature, sources, mix = result
                    groups[int(target["labels"][g]), slot].append(
                        dict(
                            image=image,
                            obj=int(target["source_objects"][g]),
                            weight=similarity,
                            feature=feature.tolist(),
                            source_images=sources,
                            source_weights=mix,
                            slot=slot,
                            label=int(target["labels"][g]),
                        )
                    )
    records = []
    claims = defaultdict(list)
    for (label, slot), entries in groups.items():
        # Give each independent image one vote, even if it contains many instances.
        support = defaultdict(list)
        for entry in entries:
            support[entry["image"]].append(entry["weight"])
        if len(support) >= settings["independent_images"]:
            strength = sum(max(scores) for scores in support.values()) / len(support)
            claims[slot].append((strength, label, entries))
    for competing in claims.values():
        competing.sort(key=lambda item: -item[0])
        if len(competing) == 1 or competing[0][0] - competing[1][0] >= settings["margin"]:
            records.extend(competing[0][2])
    representatives = [
        [dict(image=i, feature=f.tolist(), weight=w, quality=q) for i, f, w, q in slot]
        for slot in bank.slots
    ]
    return dict(
        bank_images_scanned=scanned_images,
        representatives=representatives,
        candidates=count,
        localized=localized,
        accepted=len(records),
        query_gt_by_class=dict(query_gt), localized_by_class=dict(localized_gt),
        accepted_by_class=dict(Counter(r["label"] for r in records)),
        accepted_classes=sorted({r["label"] for r in records}),
        records=records,
    )


def stage_evidence(parent, criterion, args, payload, spec):
    from tools.owod.protocol import file_sha256, read_json
    from tools.owod.screening_data import write_json

    identity = dict(
        parent=spec["source_checkpoint_sha256"],
        manifest=args.owod_manifest_sha256,
        screen=getattr(args, "owod_screening_sha256", None),
        settings=args.instance_settings,
        seed=args.seed,
        stage=args.owod_stage,
        train=spec["training_annotation_sha256"],
    )
    result = [None]
    if utils.is_main_process():
        folder = Path(args.instance_cache or args.output_dir)
        path = folder / f"instances-stage-{args.owod_stage}-{identity['parent']}.json"
        if path.is_file():
            cached = read_json(path)
            if cached["identity"] != identity:
                raise ValueError(f"Instance calibration cache identity mismatch: {path}")
        else:
            with torch.random.fork_rng(devices=[]):
                data = calibration(parent.eval(), criterion, args, payload)
            write_json(path, dict(identity=identity, data=data))
        result[0] = {**read_json(path)["data"], "cache_sha256": file_sha256(path)}
    if utils.is_dist_avail_and_initialized():
        torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


def configure_targets(criterion, spec, arm):
    records = spec.get("records", [])
    targets = list(records)
    if arm == "shuffled" and records:
        # Sort by historical slot, then permute between slots wherever possible.
        records = sorted(records, key=lambda r: (r["slot"], r["image"], r["obj"]))
        groups = defaultdict(int)
        for row in records:
            groups[row["slot"]] += 1
        shift = max(groups.values()) % len(records) or 1
        targets = records[shift:] + records[:shift]
    criterion.instance_targets = (
        {(r["image"], r["obj"], r["label"]): (t["feature"], r["weight"]) for r, t in zip(records, targets)}
        if arm != "off"
        else {}
    )
    return sum(r["slot"] != t["slot"] for r, t in zip(records, targets))


def alignment_loss(features, targets, matches, lookup):
    selected, references, weights = [], [], []
    for row, (queries, gt) in enumerate(matches):
        target = targets[row]
        image = int(target["image_id"].item())
        for q, obj, label in zip(
            queries.tolist(), target["source_objects"][gt].tolist(), target["labels"][gt].tolist()
        ):
            record = lookup.get((image, obj, label))
            if record is not None:
                vector, weight = record
                selected.append(features[row, q])
                references.append(vector)
                weights.append(weight)
    weight = features.new_tensor(weights)
    total = weight.sum()
    numerator = features.sum() * 0
    if selected:
        cost = 1 - F.cosine_similarity(torch.stack(selected), features.new_tensor(references))
        numerator = (weight * cost).sum()
    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(total)
        total /= utils.get_world_size()
    return numerator / total.clamp_min(1e-8), features.new_tensor(len(selected))
