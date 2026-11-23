# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import os
import sqlite3
import zlib

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False,
                 local_rank=0, local_size=1, proposal_file=None, source_objects=False, source_box_area=False):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, source_objects, source_box_area)
        self.proposal_enabled = bool(proposal_file)
        self.proposal_file = str(proposal_file) if proposal_file else ''
        self._proposal_connection = None
        self._proposal_pid = None
        self.proposals = {}
        if proposal_file:
            proposal_path = Path(proposal_file)
            if proposal_path.suffix == '.sqlite3':
                with sqlite3.connect(proposal_path) as connection:
                    metadata = dict(connection.execute('SELECT key, value FROM metadata'))
                if metadata.get('coordinate_format') != 'xywh_absolute':
                    raise ValueError('CAT proposal database must use xywh_absolute coordinates')
            else:
                payload = json.loads(proposal_path.read_text(encoding='utf-8'))
                if payload.get('coordinate_format') != 'xywh_absolute':
                    raise ValueError('CAT proposal file must use xywh_absolute coordinates')
                self.proposals = {int(key): value for key, value in payload.get('proposals', {}).items()}

    def _proposal_boxes(self, image_id):
        if Path(self.proposal_file).suffix != '.sqlite3':
            return self.proposals.get(int(image_id), [])
        pid = os.getpid()
        if self._proposal_connection is None or self._proposal_pid != pid:
            if self._proposal_connection is not None:
                self._proposal_connection.close()
            uri = Path(self.proposal_file).resolve().as_uri() + '?mode=ro'
            self._proposal_connection = sqlite3.connect(uri, uri=True)
            self._proposal_pid = pid
        row = self._proposal_connection.execute(
            'SELECT boxes FROM image_proposals WHERE image_id = ?',
            (int(image_id),)).fetchone()
        return [] if row is None else json.loads(zlib.decompress(row[0]))

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self.proposal_enabled:
            proposals = torch.as_tensor(self._proposal_boxes(image_id), dtype=torch.float32).reshape(-1, 4)
            if proposals.numel():
                proposals[:, 2:] += proposals[:, :2]
            target['proposal_boxes'] = proposals
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, source_objects=False, source_box_area=False):
        self.return_masks = return_masks
        self.source_objects = source_objects
        self.source_box_area = source_box_area

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.source_objects:
            # VOC source IDs end in the object index, COCO uses integer annotation IDs.
            ids = [int(str(obj["owod_source_annotation_id"]).rsplit(":", 1)[-1]) for obj in anno]
            target["source_objects"] = torch.tensor(ids, dtype=torch.long)[keep]
        if self.source_box_area:
            box_area = torch.as_tensor([obj["bbox"][2] * obj["bbox"][3] for obj in anno], dtype=torch.float32)
            target["source_box_area"] = box_area[keep]
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, lightweight=False, no_augmentation=False,
                         no_random_crop=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if lightweight:
        scales = [256, 288, 320, 352, 384]
        max_size = 640
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333

    if image_set == 'train':
        if no_augmentation:
            return T.Compose([
                T.RandomResize([384 if lightweight else 800], max_size=max_size),
                normalize,
            ])
        if lightweight:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ])
        if no_random_crop:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ])
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([384 if lightweight else 800], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    if getattr(args, 'owod_image_root', ''):
        img_folder = Path(args.owod_image_root)
    annotation_override = getattr(args, f'{image_set}_ann', '')
    if annotation_override:
        ann_file = Path(annotation_override)
        if not ann_file.is_absolute():
            ann_file = root / ann_file
    proposal_file = getattr(args, 'cat_proposals', '') if image_set == 'train' else ''
    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_coco_transforms(
                                image_set, getattr(args, 'lightweight', False),
                                getattr(args, 'no_augmentation', False),
                                getattr(args, 'no_random_crop', False)),
                            return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                            proposal_file=proposal_file,
                            source_objects=getattr(args, "instance_mode", "legacy") != "legacy",
                            source_box_area=(getattr(args, "screen_small_box_extent", False)
                                             or getattr(args, "screen_small_box_quality_rank", False)))
    return dataset
