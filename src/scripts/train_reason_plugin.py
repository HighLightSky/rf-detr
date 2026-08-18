#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train the FFT consistency-reasoning plugin on the frozen-detector cache.

Loads the candidate cache produced by ``build_reason_candidates.py`` (candidate
boxes/scores/classes/labels + image paths from the frozen baseline detector),
crops the corresponding image patches at train time, and trains
:class:`~rfdetr.reasoning.plugin.ConsistencyReasonPlugin` with a class-aware
binary-cross-entropy loss that predicts whether each blurry candidate ``B``
matches a ground-truth box.

The detector itself is **frozen**: only its ``class_embed.weight`` is read (as
the CRDe cross-attention key/value), never updated.

Usage::

    python scripts/train_reason_plugin.py \\
        --candidates output/reason_candidates_train.npz \\
        --checkpoint output/rfdetr_nano_redo_baseline/checkpoint_best_total.pth \\
        --output output/reason_plugin.pth --epochs 30 --batch_size 8
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rfdetr.reasoning import CandidateBuilder, ConsistencyReasonPlugin, PluginLoader, ReasonConfig


class CandidateDataset(Dataset[dict[str, torch.Tensor]]):
    """Per-image candidate pairs for plugin training.

    Each item is one training image: its patch image (loaded lazily from the
    cached path), the clear-A and blurry-B candidate boxes/scores/classes in
    pixel coordinates, and the binary label for each B.  Images with no A or no
    B are dropped.
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray],
        config: ReasonConfig,
        num_classes: int,
        fsc_class_id: int | None = None,
    ) -> None:
        """Initialise from the npz-loaded cache.

        Args:
            cache: Dict of per-image arrays from ``np.load``.
            config: Plugin hyper-parameters.
            num_classes: Number of object classes.
            fsc_class_id: If set, only candidates of this class are kept — a
                class-specialised training set using the same detection-candidate
                A/B framework (clear A, blurry B + synthetic masked B), but
                restricted to one class.
        """
        self.config = config
        self.num_classes = num_classes
        self.builder = CandidateBuilder(config)
        self.fsc_class_id = fsc_class_id
        # Each item: (boxes, scores, classes, labels, b_inds, path, has_synthetic_B)
        # has_synthetic_B = True means the image has no natural B candidates but
        # does have clear A's — during __getitem__ we randomly mask 10% of the A's
        # to synthesise occluded B positives (CTRP-style random masking).
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool]] = []
        # npz files are lazy archives: indexing cache[key] re-decompresses the
        # whole array each time, which is O(n_images) zipfile passes and hangs on
        # the full 4000+ image cache.  Copy everything into memory once first.
        boxes_all = cache["boxes"]
        scores_all = cache["scores"]
        classes_all = cache["class_ids"]
        labels_all = cache["labels"]
        paths_all = cache["paths"]
        lengths_all = cache["lengths"]
        for row in range(boxes_all.shape[0]):
            n = int(lengths_all[row])
            boxes = boxes_all[row, :n]
            scores = scores_all[row, :n]
            classes = classes_all[row, :n]
            labels = labels_all[row, :n]
            path = str(paths_all[row])
            if fsc_class_id is not None:
                keep = classes == fsc_class_id
                if not keep.any():
                    continue
                boxes = boxes[keep]
                scores = scores[keep]
                classes = classes[keep]
                labels = labels[keep]
            a_inds, b_inds = self.builder.split(boxes, scores)
            if a_inds.size == 0:
                continue  # no clear A anchor -> cannot form any relation pair
            has_synthetic = b_inds.size == 0
            self.items.append((boxes, scores, classes, labels, b_inds, path, has_synthetic))

    def label_stats(self) -> tuple[int, int]:
        """Count positive/negative B labels over all usable images.

        Natural B candidates come from the cache labels (0/1); synthetic
        occluded-B images contribute ``mask_prob * num_A`` positives each
        (matching what ``__getitem__`` synthesises).  Used to size the BCE
        ``pos_weight`` so the effective gradient contribution is balanced.

        Returns:
            ``(num_positive, num_negative)`` over the dataset's B candidates.
        """
        num_pos = 0
        num_neg = 0
        for boxes, scores, _classes, labels, _b_inds, _path, has_synthetic in self.items:
            b_labels = labels[labels >= 0]  # only natural B entries carry 0/1
            num_pos += int((b_labels > 0.5).sum())
            num_neg += int((b_labels <= 0.5).sum())
            if has_synthetic and self.config.mask_prob > 0:
                a_inds, _ = self.builder.split(boxes, scores)
                num_synth = max(1, int(self.config.mask_prob * a_inds.size))
                num_pos += num_synth
        return num_pos, num_neg

    def __len__(self) -> int:
        """Return the number of usable images."""
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one image's training tensors.

        Returns:
            Dict with ``a_patches``, ``b_patches``, ``boxes_a``, ``boxes_b``,
            ``class_a``, ``target`` and ``image_shape``.
        """
        boxes, scores, classes, labels, b_inds, path, has_synthetic = self.items[idx]
        a_inds, _ = self.builder.split(boxes, scores)
        image = _load_image(path)
        image_shape = (image.shape[0], image.shape[1])

        if has_synthetic:
            # No natural B candidates: synthesise occluded-B positives by
            # randomly masking ~mask_prob of the clear A's (pixels zeroed),
            # keeping the remaining A's as anchors (CTRP-style random masking).
            return self._synthetic_item(
                boxes=boxes,
                scores=scores,
                classes=classes,
                a_inds=a_inds,
                image=image,
                image_shape=image_shape,
            )

        # Natural path: pair each blurry B with its nearest clear A.
        pairs = self.builder.pair_topk(boxes, a_inds, b_inds)
        if pairs.size == 0:
            # Should not happen (guarded in __init__), but keep the collator safe.
            return {
                "a_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "b_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "joint_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "boxes_a": torch.zeros(0, 4),
                "boxes_b": torch.zeros(0, 4),
                "class_a": torch.zeros(0, dtype=torch.long),
                "target": torch.zeros(0),
                "image_shape": torch.zeros(0, 2),
            }
        # Choose one pair per B deterministically from the top-k (first nearest).
        # Keeping the nearest A is the most informative for the co-occurrence
        # gate and keeps the pair count equal to the B count per image.
        selected: list[tuple[int, int]] = []
        seen_b: set[int] = set()
        for a_i, b_i in pairs:
            if b_i not in seen_b:
                selected.append((int(a_i), int(b_i)))
                seen_b.add(b_i)
        a_idx = np.array([p[0] for p in selected], dtype=np.int64)
        b_idx = np.array([p[1] for p in selected], dtype=np.int64)

        a_patches = _crop_patches(image, boxes[a_idx], self.config.patch_size)
        b_patches = _crop_patches(image, boxes[b_idx], self.config.patch_size)
        joint_patches = _crop_joint_patches(image, boxes[a_idx], boxes[b_idx], self.config.patch_size)

        n_pairs = len(selected)
        # labels is indexed by ORIGINAL candidate index; b_idx already holds
        # original indices (pairs[:, 1] == b_inds[bi]).
        target = torch.from_numpy(labels[b_idx].astype(np.float32))
        return {
            "a_patches": torch.from_numpy(a_patches),
            "b_patches": torch.from_numpy(b_patches),
            "joint_patches": torch.from_numpy(joint_patches),
            "boxes_a": torch.from_numpy(boxes[a_idx].astype(np.float32)),
            "boxes_b": torch.from_numpy(boxes[b_idx].astype(np.float32)),
            "class_a": torch.from_numpy(classes[a_idx].astype(np.int64)),
            "target": target,
            # One (h, w) row per pair so a collated batch stays aligned with
            # the flattened a_patches/b_patches tensors.
            "image_shape": torch.tensor(image_shape, dtype=torch.float32).repeat(n_pairs, 1),
        }

    def _synthetic_item(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        a_inds: np.ndarray,
        image: np.ndarray,
        image_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Build a training item for an image with no natural B candidates.

        Randomly selects ``~mask_prob`` of the clear ``A`` targets and masks
        (zeroes) ``mask_fraction`` of each selected patch's pixels, treating them
        as occluded ``B`` positives; the remaining A's act as anchors.  All
        synthetic B labels are 1 (true positives to be recovered).

        Args:
            boxes: Candidate boxes ``(N, 4)``.
            scores: Candidate scores ``(N,)``.
            classes: Candidate class ids ``(N,)``.
            a_inds: Indices of clear A candidates.
            image: Original ``(H, W, 3)`` uint8 image.
            image_shape: Image ``(height, width)``.

        Returns:
            The standard training-item dict.
        """
        from rfdetr.reasoning.plugin import mask_pixels

        # Randomly pick ~mask_prob of the A's to mask into synthetic B's.
        num_synth = max(1, int(self.config.mask_prob * a_inds.size))
        synth_inds = np.random.choice(a_inds, size=num_synth, replace=False)
        anchor_inds = a_inds[~np.isin(a_inds, synth_inds)]
        if anchor_inds.size == 0:
            # All A's would be masked — keep at least one anchor.
            anchor_inds = np.array([synth_inds[0]], dtype=np.int64)
            synth_inds = synth_inds[1:]

        # Pair each synthetic B with its nearest anchor A.
        pairs = self.builder.pair_topk(boxes, anchor_inds, synth_inds)
        if pairs.size == 0:
            return {
                "a_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "b_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "joint_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "boxes_a": torch.zeros(0, 4),
                "boxes_b": torch.zeros(0, 4),
                "class_a": torch.zeros(0, dtype=torch.long),
                "target": torch.zeros(0),
                "image_shape": torch.zeros(0, 2),
            }
        a_idx = pairs[:, 0]
        b_idx = pairs[:, 1]

        a_patches = _crop_patches(image, boxes[a_idx], self.config.patch_size)
        b_patches = _crop_patches(image, boxes[b_idx], self.config.patch_size)
        # Mask the synthetic B patches (training-only occlusion augmentation).
        b_masked = np.stack(
            [mask_pixels(b_patches[i], self.config.mask_fraction) for i in range(b_patches.shape[0])],
            axis=0,
        )
        joint_patches = _crop_joint_patches(image, boxes[a_idx], boxes[b_idx], self.config.patch_size)

        n_pairs = pairs.shape[0]
        target = torch.ones(n_pairs, dtype=torch.float32)  # synthetic B are positives
        return {
            "a_patches": torch.from_numpy(a_patches),
            "b_patches": torch.from_numpy(b_masked),
            "joint_patches": torch.from_numpy(joint_patches),
            "boxes_a": torch.from_numpy(boxes[a_idx].astype(np.float32)),
            "boxes_b": torch.from_numpy(boxes[b_idx].astype(np.float32)),
            "class_a": torch.from_numpy(classes[a_idx].astype(np.int64)),
            "target": target,
            "image_shape": torch.tensor(image_shape, dtype=torch.float32).repeat(n_pairs, 1),
        }


class FscGtDataset(Dataset[dict[str, torch.Tensor]]):
    """FSC-specific training set.

    Only images whose ground-truth contains ``fsc_class_name`` (e.g. FSC) are
    used.  For each image:

    - **B (missing entity)**: ``fsc_gt_frac`` (5%-10%) of the FSC GT boxes are
      randomly chosen, cropped, masked (``mask_fraction`` pixels zeroed), forced
      to score 0.1 and labelled 1 (a true target to be recovered).  GT is used
      because only GT boxes correspond to physically real targets — masking them
      is what makes "reconstruction" meaningful.
    - **A (clear entity)**: detection candidates with score >= 0.5 (all classes)
      act as anchors — the same distribution the model sees at inference.
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray],
        config: ReasonConfig,
        num_classes: int,
        dataset_dir: str,
        fsc_gt_frac: float,
        fsc_class_name: str = "FSC",
    ) -> None:
        """Initialise from the npz cache and the YOLO GT labels.

        Args:
            cache: Dict of per-image arrays from ``np.load``.
            config: Plugin hyper-parameters.
            num_classes: Number of object classes.
            dataset_dir: SHWX dataset root (to locate labels and class names).
            fsc_gt_frac: Fraction of a FSC image's GT boxes chosen as masked B.
            fsc_class_name: Class name to specialise on (default FSC).
        """
        import sys
        from pathlib import Path

        from eval_metrics import build_image_size_map, load_class_names, load_yolo_labels

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        self.config = config
        self.num_classes = num_classes
        self.builder = CandidateBuilder(config)
        self.fsc_gt_frac = fsc_gt_frac
        self.fsc_class_id = load_class_names(dataset_dir).index(fsc_class_name)

        dataset_root = Path(dataset_dir)
        image_dir = dataset_root / "images" / "train"
        label_dir = dataset_root / "labels" / "train"
        image_size_map = build_image_size_map(image_dir)
        gt_records = load_yolo_labels(label_dir, image_size_map)
        gt_by_image: dict[str, list[tuple[np.ndarray, int]]] = {}
        for rec in gt_records:
            gt_by_image.setdefault(rec.image_id, []).append((np.asarray(rec.xyxy, dtype=np.float32), rec.class_id))

        # npz is a lazy archive; copy arrays into memory once.
        boxes_all = cache["boxes"]
        scores_all = cache["scores"]
        classes_all = cache["class_ids"]
        paths_all = cache["paths"]
        lengths_all = cache["lengths"]
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
        for row in range(boxes_all.shape[0]):
            n = int(lengths_all[row])
            img_id = str(cache["image_ids"][row])
            gt = gt_by_image.get(img_id, [])
            fsc_gt = [g for g in gt if g[1] == self.fsc_class_id]
            if not fsc_gt:
                continue  # only images containing the FSC class
            boxes = boxes_all[row, :n]
            scores = scores_all[row, :n]
            classes = classes_all[row, :n]
            path = str(paths_all[row])
            a_inds, _ = self.builder.split(boxes, scores)
            if a_inds.size == 0:
                continue  # no clear A anchor
            fsc_gt_boxes = np.stack([g[0] for g in fsc_gt], axis=0)
            # All GT boxes (any class) of this image, used to identify background
            # false-alarm candidates (IoU < 0.3 with every GT) as negative B.
            all_gt = np.stack([g[0] for g in gt], axis=0) if gt else np.zeros((0, 4), dtype=np.float32)
            self.items.append((boxes, scores, classes, fsc_gt_boxes, all_gt, path))

    def __len__(self) -> int:
        """Return the number of usable images."""
        return len(self.items)

    def label_stats(self) -> tuple[int, int]:
        """Count synthetic positives vs natural negatives for pos_weight.

        Every synthetic B is a positive (label 1); the negatives come from the
        images' natural low-confidence candidates (0/1 labels from the cache are
        approximated as the count of non-positive candidates).  Since the FSC-GT
        set is all-positive, we return ``(num_B, 0)`` and rely on a small
        default ``pos_weight``.

        Returns:
            ``(num_positive, 0)``.
        """
        total_pos = 0
        total_neg = 0
        for _boxes, scores, _classes, fsc_gt_boxes, all_gt, _path in self.items:
            num_pos = max(1, int(self.fsc_gt_frac * fsc_gt_boxes.shape[0]))
            total_pos += num_pos
            # Negatives: background detection candidates (IoU<0.3 w/ every GT),
            # 1-2 per image.
            num_neg = min(2, int((scores >= self.config.conf_low).sum()))
            total_neg += num_neg
        return total_pos, total_neg

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Build one image's training tensors (masked GT B vs detection A).

        Positive B: a random ``fsc_gt_frac`` of the FSC GT boxes, masked
        (pixels zeroed) and labelled 1.  Negative B: 1-2 background detection
        candidates with IoU < 0.3 against every GT (label 0), so the model
        learns discrimination, not just "everything FSC is a target".
        Both are paired with their nearest A (detection candidate, score >= 0.5).
        """
        from rfdetr.reasoning.plugin import mask_pixels

        boxes, scores, classes, fsc_gt_boxes, all_gt, path = self.items[idx]
        a_inds, _ = self.builder.split(boxes, scores)
        # A anchors: detection candidates with score >= 0.5 (all classes).
        a_inds = a_inds[scores[a_inds] >= 0.5]
        if a_inds.size == 0:
            return {
                "a_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "b_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "joint_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "boxes_a": torch.zeros(0, 4),
                "boxes_b": torch.zeros(0, 4),
                "class_a": torch.zeros(0, dtype=torch.long),
                "target": torch.zeros(0),
                "image_shape": torch.zeros(0, 2),
            }

        # ── Positive B: random fsc_gt_frac of the FSC GT boxes, masked. ──────
        num_pos = max(1, int(self.fsc_gt_frac * fsc_gt_boxes.shape[0]))
        num_pos = min(num_pos, fsc_gt_boxes.shape[0])
        pos_inds = np.random.choice(fsc_gt_boxes.shape[0], size=num_pos, replace=False)
        pos_boxes = fsc_gt_boxes[pos_inds]

        # ── Negative B: background detection candidates (IoU < 0.3 w/ every GT). ──
        # Low-score candidates that overlap no GT are background false alarms.
        neg_candidates = [
            i for i in range(scores.shape[0]) if scores[i] >= self.config.conf_low and _max_iou(boxes[i], all_gt) < 0.3
        ]
        num_neg = min(2, len(neg_candidates))
        neg_inds = (
            np.random.choice(neg_candidates, size=num_neg, replace=False)
            if num_neg > 0
            else np.array([], dtype=np.int64)
        )
        neg_boxes = boxes[neg_inds] if neg_inds.size else np.zeros((0, 4), dtype=np.float32)

        # ── Concatenate positive and negative B, pair each with nearest A. ──
        all_b = np.concatenate([pos_boxes, neg_boxes], axis=0) if neg_boxes.size else pos_boxes
        targets = (
            np.concatenate([np.ones(pos_boxes.shape[0]), np.zeros(neg_boxes.shape[0])])
            if neg_boxes.size
            else np.ones(pos_boxes.shape[0])
        )
        if all_b.shape[0] == 0:
            return {
                "a_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "b_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "joint_patches": torch.zeros(0, 3, self.config.patch_size, self.config.patch_size),
                "boxes_a": torch.zeros(0, 4),
                "boxes_b": torch.zeros(0, 4),
                "class_a": torch.zeros(0, dtype=torch.long),
                "target": torch.zeros(0),
                "image_shape": torch.zeros(0, 2),
            }
        a_c = boxes[a_inds][:, :2]
        b_c = all_b[:, :2]
        dist = np.sqrt(((b_c[:, None, :] - a_c[None, :, :]) ** 2).sum(-1))
        nearest_a = np.argmin(dist, axis=1)  # (num_b,) index into a_inds
        a_idx = a_inds[nearest_a]

        image = _load_image(path)
        image_shape = (image.shape[0], image.shape[1])
        a_patches = _crop_patches(image, boxes[a_idx], self.config.patch_size)
        b_patches = _crop_patches(image, all_b, self.config.patch_size)
        # Mask only the positive (FSC GT) B patches; negatives stay as-is.
        b_masked = b_patches.copy()
        for i in range(pos_boxes.shape[0]):
            b_masked[i] = mask_pixels(b_masked[i], self.config.mask_fraction)
        joint_patches = _crop_joint_patches(image, boxes[a_idx], all_b, self.config.patch_size)

        n_pairs = all_b.shape[0]
        target = torch.from_numpy(targets.astype(np.float32))
        return {
            "a_patches": torch.from_numpy(a_patches),
            "b_patches": torch.from_numpy(b_masked),
            "joint_patches": torch.from_numpy(joint_patches),
            "boxes_a": torch.from_numpy(boxes[a_idx].astype(np.float32)),
            "boxes_b": torch.from_numpy(all_b.astype(np.float32)),
            "class_a": torch.from_numpy(classes[a_idx].astype(np.int64)),
            "target": target,
            "image_shape": torch.tensor(image_shape, dtype=torch.float32).repeat(n_pairs, 1),
        }


def _max_iou(box: np.ndarray, gt_boxes: np.ndarray) -> float:
    """Maximum IoU between one box and a set of GT boxes.

    Args:
        box: Single ``(4,)`` pixel box ``[x1, y1, x2, y2]``.
        gt_boxes: ``(M, 4)`` pixel GT boxes (may be empty).

    Returns:
        Max IoU, or 0.0 when ``gt_boxes`` is empty.
    """
    if gt_boxes.shape[0] == 0:
        return 0.0
    x1 = np.maximum(gt_boxes[:, 0], box[0])
    y1 = np.maximum(gt_boxes[:, 1], box[1])
    x2 = np.minimum(gt_boxes[:, 2], box[2])
    y2 = np.minimum(gt_boxes[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ga = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    ba = (box[2] - box[0]) * (box[3] - box[1])
    union = ga + ba - inter
    ious = inter / np.maximum(union, 1e-6)
    return float(ious.max())


def _load_image(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 array.

    Args:
        path: Image file path.

    Returns:
        ``(H, W, 3)`` uint8 array.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _crop_patches(source_image: np.ndarray, boxes: np.ndarray, patch_size: int) -> np.ndarray:
    """Crop and bilinear-resize square patches from an image.

    Args:
        source_image: ``(H, W, 3)`` uint8 image.
        boxes: ``(N, 4)`` pixel boxes.
        patch_size: Square output size.

    Returns:
        Float32 ``(N, 3, patch_size, patch_size)`` in ``[0, 1]``.
    """
    from rfdetr.reasoning.plugin import _crop_patches as _core

    return _core(source_image, boxes, patch_size)


def _crop_joint_patches(
    source_image: np.ndarray,
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Crop the A∪B union rectangles (RoI visual base of the relation feature).

    Args:
        source_image: ``(H, W, 3)`` uint8 image.
        boxes_a: ``(N, 4)`` pixel boxes of the clear targets A.
        boxes_b: ``(N, 4)`` pixel boxes of the blurry targets B.
        patch_size: Square output size.

    Returns:
        Float32 ``(N, 3, patch_size, patch_size)`` in ``[0, 1]``.
    """
    from rfdetr.reasoning.plugin import _crop_joint_patches as _core

    return _core(source_image, boxes_a, boxes_b, patch_size)


def main() -> None:
    """Train the plugin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="npz cache from build_reason_candidates.py")
    parser.add_argument("--checkpoint", required=True, help="Frozen baseline detector (for class_embed.weight)")
    parser.add_argument(
        "--dataset_dir",
        default="/home/liu/datasets/SHWX-dataset-dict-redo",
        help="Dataset root (needed for FSC-GT mode to read GT labels).",
    )
    parser.add_argument("--output", default="output/reason_plugin.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--boost-scale",
        type=float,
        default=None,
        help="Inference score adjustment scale stored in the checkpoint "
        "(persisted so eval defaults match the tuned value).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=None,
        help="Inference sigmoid decision line stored in the checkpoint.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW LR.  Kept low: 1e-3 drives the small plugin's logits extreme "
        "(observed negative BCE loss in the smoke run).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument(
        "--patches_per_batch",
        type=int,
        default=64,
        help="cap on patches per batch (B pairs)",
    )
    parser.add_argument(
        "--neg_margin",
        type=float,
        default=0.05,
        help="Auxiliary loss: push false-alarm (negative) B probabilities below "
        "this sigmoid margin.  Strengthens discrimination so boosting at "
        "inference does not pull false alarms over the threshold.",
    )
    parser.add_argument(
        "--neg_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the negative-margin auxiliary loss relative to BCE.",
    )
    parser.add_argument(
        "--pos_margin",
        type=float,
        default=0.8,
        help="True-positive probabilities are pulled above this margin by the "
        "contrastive loss (widening separation from false alarms).",
    )
    parser.add_argument(
        "--pos_loss_weight",
        type=float,
        default=0.5,
        help="Weight of the true-positive contrastive pull relative to BCE.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="BCE positive-class weight.  Defaults to num_neg/num_pos measured "
        "from the cache (balances the effective gradient contribution of each "
        "class); set explicitly to override.",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=None,
        help="Fraction of clear A's masked into synthetic occluded-B positives "
        "for images with no natural B candidates.  Defaults to the config value.",
    )
    parser.add_argument(
        "--mask_fraction",
        type=float,
        default=None,
        help="Fraction of a synthetic-B patch's pixels zeroed.  Lower keeps more "
        "texture (0.75 fully black, 0.4 partial occlusion).",
    )
    parser.add_argument(
        "--fsc_only",
        action="store_true",
        help="FSC-specific training: B is a randomly masked FSC ground-truth box "
        "(score forced to 0.1, label 1), A is a detection candidate with "
        "score >= 0.5.  Only images that contain FSC GT are used.",
    )
    parser.add_argument(
        "--fsc_train",
        action="store_true",
        help="Class-specialised training using the SAME detection-candidate A/B "
        "framework (clear A, blurry B + synthetic masked B) but restricted to "
        "the FSC class only.",
    )
    parser.add_argument(
        "--fsc_gt_frac",
        type=float,
        default=0.075,
        help="Fraction of a FSC image's GT boxes randomly chosen as masked B (user spec: 5%-10%, default 7.5%).",
    )
    parser.add_argument(
        "--fsc_class_name",
        type=str,
        default="FSC",
        help="Class name to specialise on (default FSC).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = ReasonConfig()
    if args.boost_scale is not None:
        config.boost_scale = args.boost_scale
    if args.p_threshold is not None:
        config.p_threshold = args.p_threshold
    if args.mask_prob is not None:
        config.mask_prob = args.mask_prob
    if args.mask_fraction is not None:
        config.mask_fraction = args.mask_fraction
    cache = np.load(args.candidates, allow_pickle=True)
    # Both dataset classes expose __len__/label_stats/__getitem__; declare the
    # common base so mypy accepts the union assignment below.
    dataset: Dataset[dict[str, torch.Tensor]]
    if args.fsc_only:
        dataset = FscGtDataset(
            cache,
            config,
            args.num_classes,
            args.dataset_dir,
            fsc_gt_frac=args.fsc_gt_frac,
            fsc_class_name=args.fsc_class_name,
        )
    elif args.fsc_train:
        from eval_metrics import load_class_names

        fsc_id = load_class_names(args.dataset_dir).index(args.fsc_class_name)
        dataset = CandidateDataset(cache, config, args.num_classes, fsc_class_id=fsc_id)
    else:
        dataset = CandidateDataset(cache, config, args.num_classes)
    print(f"[dataset] {len(dataset)} usable images")
    if len(dataset) == 0:
        raise SystemExit("no usable images in cache (every image lacks A or B)")

    # Default pos_weight = num_neg/num_pos balances the two classes' effective
    # gradient contribution; too high a value collapses training to all-positive
    # (observed with pos_weight=8 on a ~1:4.5 cache).  The FSC-GT set is
    # all-positive, so keep a mild default instead of a degenerate pos_weight.
    if args.pos_weight is None:
        num_pos, num_neg = dataset.label_stats()
        if num_pos == 0:
            raise SystemExit("no positive B labels in cache; cannot train a recall booster")
        args.pos_weight = (num_neg / num_pos) if num_neg > 0 else 1.0
        print(f"[imbalance] pos={num_pos} neg={num_neg} -> pos_weight={args.pos_weight:.2f}")

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Concatenate per-image pair tensors into flat batched patches.

        Every tensor is already per-pair and flat within an item, so a plain ``torch.cat`` along dim 0 aligns
        patches/boxes/classes/labels and the per-pair ``image_shape`` rows across the batch.
        """
        out: dict[str, list[torch.Tensor]] = {k: [] for k in batch[0]}
        for item in batch:
            for k, v in item.items():
                out[k].append(v)
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=_collate,
    )

    # Sanity-check shapes before training on the first item.
    _check_batch(loader, args)

    # Frozen detector class-embedding matrix for the CRDe key/value.
    from rfdetr import RFDETR

    detector = RFDETR.from_checkpoint(args.checkpoint)
    class_embed_weight = detector.model.model.class_embed.weight.detach().to(args.device)
    del detector

    plugin = ConsistencyReasonPlugin(args.num_classes, config).to(args.device)
    optimizer = torch.optim.AdamW(plugin.parameters(), lr=args.lr, weight_decay=1e-4)
    assert args.pos_weight is not None  # resolved above
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=args.device))

    steps = 0
    for epoch in range(args.epochs):
        plugin.train()
        total_loss = 0.0
        total_samples = 0
        total_pos = 0
        for batch in loader:
            a_patches = batch["a_patches"].to(args.device)
            b_patches = batch["b_patches"].to(args.device)
            joint_patches = batch["joint_patches"].to(args.device)
            boxes_a = batch["boxes_a"].to(args.device)
            boxes_b = batch["boxes_b"].to(args.device)
            class_a = batch["class_a"].to(args.device)
            target = batch["target"].to(args.device)
            image_shape = batch["image_shape"].to(args.device)  # (total_pairs, 2)
            if target.numel() == 0:
                continue
            logits, gates, _ = plugin(
                a_patches,
                b_patches,
                joint_patches,
                boxes_a,
                boxes_b,
                class_a,
                class_embed_weight,
                img_shape=image_shape,
            )
            loss = criterion(logits, target)
            # Contrastive auxiliary: pull the two classes' probabilities apart
            # with a two-sided squared hinge.  False alarms are dragged toward 0
            # (below neg_margin) and true positives toward 1 (above pos_margin),
            # widening the separation so inference boosting recovers true
            # positives without dragging false alarms over the threshold.
            prob = torch.sigmoid(logits)
            neg_mask = target < 0.5
            if neg_mask.any():
                neg_over = torch.relu(prob[neg_mask] - args.neg_margin)
                neg_penalty = (neg_over**2).mean()
                loss = loss + args.neg_loss_weight * neg_penalty
            pos_mask = target >= 0.5
            if pos_mask.any():
                pos_under = torch.relu(args.pos_margin - prob[pos_mask])
                pos_penalty = (pos_under**2).mean()
                loss = loss + args.pos_loss_weight * pos_penalty
            optimizer.zero_grad()
            loss.backward()
            # Guard against any pathological sample blowing the loss/gradients
            # (the small plugin is prone to logit explosion without clipping).
            torch.nn.utils.clip_grad_norm_(plugin.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_samples += target.numel()
            total_pos += int((target > 0.5).sum())
            steps += 1
            if steps % 20 == 0:
                print(f"[{steps:5d}] loss={loss.item():.4f} pos={total_pos}/{total_samples}", flush=True)

        print(
            f"epoch {epoch + 1}/{args.epochs}: loss={total_loss / max(total_samples, 1):.4f} "
            f"pos={total_pos}/{total_samples}",
            flush=True,
        )
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            PluginLoader.save(plugin, args.output)
            print(f"saved {args.output}", flush=True)


def _check_batch(loader: DataLoader[torch.Tensor], args: argparse.Namespace) -> None:
    """Run one batch through the plugin to validate shapes before training.

    Args:
        loader: The candidate data loader.
        args: Parsed arguments (used for device placement).
    """
    batch = next(iter(loader))
    a_patches = batch["a_patches"]
    b_patches = batch["b_patches"]
    boxes_a = batch["boxes_a"]
    class_a = batch["class_a"]
    target = batch["target"]
    image_shape = batch["image_shape"]
    print(
        f"[sanity] batch patches: A={tuple(a_patches.shape)} B={tuple(b_patches.shape)} "
        f"target={tuple(target.shape)} boxes={tuple(boxes_a.shape)} shape={tuple(image_shape.shape)}"
    )
    if a_patches.numel() == 0:
        raise SystemExit("empty first batch (no pairs); check the candidate cache")
    # Basic invariant: one patch per pair, and one (h, w) row per pair.
    assert a_patches.shape[0] == boxes_a.shape[0] == class_a.shape[0] == target.shape[0]
    assert image_shape.shape == (a_patches.shape[0], 2)


if __name__ == "__main__":
    main()
