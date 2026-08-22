#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build the FFT consistency-plugin training cache from a frozen detector.

Runs the (frozen) baseline detector over the SHWX training split at a low
confidence threshold, buckets every candidate into *clear* (``A``, score >=
``tau_high``) and *blurry* (``B``, ``conf_low <= score < conf_high``), and
labels each ``B`` by class-aware IoU matching against ground truth
(``IoU >= 0.5``).  The result is saved as a lightweight ``.npz`` (boxes,
scores, class ids, labels and image paths — not the pixels), which
``train_reason_plugin.py`` loads to crop patches and train the plugin.

Usage::

    python scripts/build_reason_candidates.py \\
        --checkpoint output/rfdetr_nano_redo_baseline/checkpoint_best_total.pth \\
        --output output/reason_candidates_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_metrics import (
    IMAGE_EXTENSIONS,
    build_image_size_map,
    load_class_names,
    load_yolo_labels,
)

from rfdetr import RFDETR
from rfdetr.reasoning import CandidateBuilder, ReasonConfig


def build_candidates(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    split: str,
    config: ReasonConfig,
    device: str,
    limit_images: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect per-image candidates and GT-match labels from the frozen detector.

    Args:
        checkpoint: Baseline detector checkpoint path.
        dataset_dir: SHWX dataset root.
        split: Dataset split to run (``"train"``, ``"val"`` or ``"test"``).
        config: Plugin hyper-parameters (used for the A/B confidence band).
        device: Torch device string.
        limit_images: If set, only process this many images (debug/smoke).

    Returns:
        ``{image_id: {"boxes", "scores", "class_ids", "labels", "image_path"}}``
        where ``labels`` is 1 for a GT-matched B, 0 otherwise (A candidates
        carry a placeholder label of ``-1`` and are skipped by training).
    """
    dataset_root = Path(dataset_dir)
    names = load_class_names(dataset_dir)
    name2id = {name: class_id for class_id, name in enumerate(names)}
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_size_map = build_image_size_map(image_dir)
    gt_records = load_yolo_labels(label_dir, image_size_map)

    gt_by_image: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec in gt_records:
        box = np.array(rec.xyxy, dtype=np.float32)
        gt_by_image.setdefault(rec.image_id, ([], []))
        gt_by_image[rec.image_id][0].append(box)
        gt_by_image[rec.image_id][1].append(rec.class_id)

    model = RFDETR.from_checkpoint(checkpoint)
    model.model.device = device

    id_to_path = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    builder = CandidateBuilder(config)
    per_image: dict[str, dict[str, np.ndarray]] = {}

    image_paths = sorted(id_to_path.items())
    if limit_images is not None:
        image_paths = image_paths[:limit_images]
    with torch.no_grad():
        for img_id, path in image_paths:
            res = model.predict(
                str(path),
                threshold=config.conf_low,
                include_source_image=False,
            )
            class_names_pred = res.data.get("class_name", [])
            boxes_list: list[np.ndarray] = []
            scores_list: list[float] = []
            classes_list: list[int] = []
            for i in range(len(res.xyxy)):
                class_name = class_names_pred[i]
                if class_name not in name2id:
                    continue
                boxes_list.append(np.asarray(res.xyxy[i], dtype=np.float32))
                scores_list.append(float(res.confidence[i]))
                classes_list.append(name2id[class_name])
            if not boxes_list:
                continue

            boxes = np.stack(boxes_list, axis=0)
            scores = np.asarray(scores_list, dtype=np.float32)
            class_ids = np.asarray(classes_list, dtype=np.int64)

            a_inds, b_inds = builder.split(boxes, scores)
            # labels is indexed by ORIGINAL candidate index: A entries stay -1
            # (never used as training targets), every B entry is 0 (false alarm)
            # unless it matches GT, in which case it becomes 1.  Initialising all
            # B's to 0 (not leaving them -1) guarantees training targets are only
            # 0/1 even for images that have B candidates but no GT overlap.
            labels = np.full(boxes.shape[0], -1.0, dtype=np.float32)
            if b_inds.size:
                labels[b_inds] = 0.0
                gt_boxes, gt_classes = gt_by_image.get(
                    img_id, (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64))
                )
                if len(gt_boxes) > 0:
                    b_labels = builder.label_pairs(
                        b_inds,
                        boxes[b_inds],
                        class_ids[b_inds],
                        np.asarray(gt_boxes, dtype=np.float32),
                        np.asarray(gt_classes, dtype=np.int64),
                    )
                    labels[b_inds] = np.maximum(labels[b_inds], b_labels)

            per_image[img_id] = {
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "labels": labels,
                "image_path": str(path),
            }
            print(
                f"[{len(per_image):5d}] {img_id}: {boxes.shape[0]} cands, "
                f"{a_inds.size} A, {b_inds.size} B, {(labels == 1).sum():.0f} positive",
                flush=True,
            )

    del model
    return per_image


def main() -> None:
    """Parse args and write the candidate cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen baseline checkpoint (*.pth)")
    parser.add_argument("--dataset_dir", default="/home/liu/datasets/SHWX-dataset-dict")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", default="output/reason_candidates_train.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf_low", type=float, default=0.05)
    parser.add_argument("--conf_high", type=float, default=0.25)
    parser.add_argument("--tau_high", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit_images", type=int, default=None, help="debug: only process this many images")
    args = parser.parse_args()

    config = ReasonConfig(
        conf_low=args.conf_low,
        conf_high=args.conf_high,
        tau_high=args.tau_high,
        top_k=args.top_k,
    )
    per_image = build_candidates(
        args.checkpoint,
        args.dataset_dir,
        args.split,
        config,
        args.device,
        limit_images=args.limit_images,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pad ragged arrays to a common length for npz storage.
    max_len = max((v["boxes"].shape[0] for v in per_image.values()), default=0)
    image_ids = sorted(per_image)
    boxes = np.zeros((len(image_ids), max_len, 4), dtype=np.float32)
    scores = np.zeros((len(image_ids), max_len), dtype=np.float32)
    class_ids = np.zeros((len(image_ids), max_len), dtype=np.int64)
    labels = np.full((len(image_ids), max_len), -1.0, dtype=np.float32)
    paths = np.array([per_image[i]["image_path"] for i in image_ids], dtype=object)
    lengths = np.zeros(len(image_ids), dtype=np.int64)
    for row, img_id in enumerate(image_ids):
        n = per_image[img_id]["boxes"].shape[0]
        lengths[row] = n
        boxes[row, :n] = per_image[img_id]["boxes"]
        scores[row, :n] = per_image[img_id]["scores"]
        class_ids[row, :n] = per_image[img_id]["class_ids"]
        labels[row, :n] = per_image[img_id]["labels"]

    np.savez(
        out_path,
        image_ids=np.array(image_ids, dtype=object),
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        labels=labels,
        paths=paths,
        lengths=lengths,
        conf_low=config.conf_low,
        conf_high=config.conf_high,
        tau_high=config.tau_high,
        top_k=config.top_k,
    )
    total_b = int(np.sum(lengths))
    total_pos = int(np.sum(labels == 1))
    print(
        f"\nSaved {out_path}: {len(image_ids)} images, {total_b} candidates, "
        f"{total_pos} positive B ({total_pos / max(total_b, 1):.2%})"
    )


if __name__ == "__main__":
    main()
