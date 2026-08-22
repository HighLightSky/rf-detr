#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Visualise GT vs baseline vs reason-plugin prediction boxes on the test split.

Selects a few representative test images and renders, side by side:
  - left:  ground-truth boxes (green),
  - middle: baseline detector boxes at ``--conf`` (blue),
  - right:  reason-plugin-adjusted boxes at ``--conf`` (blue = kept, red =
    newly added above threshold, orange = suppressed).

Usage::

    python scripts/viz_reason_boxes.py \
        --checkpoint output/rfdetr_nano_redo_baseline/checkpoint_best_total.pth \
        --dataset_dir /home/liu/datasets/SHWX-dataset-dict-redo \
        --reason-plugin output/reason_plugin_disc.pth \
        --output output/viz_reason_boxes.png --conf 0.25 --n 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_metrics import (
    IMAGE_EXTENSIONS,
    build_image_size_map,
    load_class_names,
    load_yolo_labels,
)

from rfdetr import RFDETR
from rfdetr.reasoning import PluginLoader


def main() -> None:
    """Render the GT/baseline/plugin comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default="/home/liu/datasets/SHWX-dataset-dict-redo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--reason-plugin", required=True)
    parser.add_argument("--output", default="output/viz_reason_boxes.png")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--n", type=int, default=4, help="number of images to render")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    names = load_class_names(args.dataset_dir)
    name2id = {name: cid for cid, name in enumerate(names)}
    id2name = {cid: name for name, cid in name2id.items()}
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    image_size_map = build_image_size_map(image_dir)
    gt_records = load_yolo_labels(label_dir, image_size_map)

    plugin = PluginLoader.load(args.reason_plugin)
    plugin.to(args.device)
    model = RFDETR.from_checkpoint(args.checkpoint)
    model.model.device = args.device
    class_embed_weight = model.model.model.class_embed.weight.detach().to(args.device)

    # Group GT per image.
    gt_by_image: dict[str, list[tuple[np.ndarray, int]]] = {}
    for rec in gt_records:
        gt_by_image.setdefault(rec.image_id, []).append((np.asarray(rec.xyxy, dtype=np.float32), rec.class_id))

    id_to_path = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    # Deterministic sample of images that have GT boxes.
    rng = np.random.default_rng(args.seed)
    candidates = [i for i in id_to_path if i in gt_by_image and gt_by_image[i]]
    chosen = list(rng.choice(candidates, size=min(args.n, len(candidates)), replace=False))

    fig, axes = plt.subplots(args.n, 3, figsize=(3 * 6, args.n * 5))
    if args.n == 1:
        axes = axes[None, :]
    for row, img_id in enumerate(chosen):
        path = id_to_path[img_id]
        source = np.asarray(
            __import__("PIL").Image.open(path).convert("RGB"),
            dtype=np.uint8,
        )

        # Baseline predictions at target conf.
        base = model.predict(str(path), threshold=args.conf, include_source_image=False)
        base_boxes = np.asarray(base.xyxy, dtype=np.float32)
        base_scores = np.asarray(base.confidence, dtype=np.float32)
        base_cls = [c for c in base.data.get("class_name", [])]

        # Plugin predictions: low-threshold candidates -> re-score -> conf.
        low = model.predict(str(path), threshold=plugin.config.conf_low, include_source_image=True)
        d = low
        cand_boxes = np.asarray(d.xyxy, dtype=np.float32)
        cand_scores = np.asarray(d.confidence, dtype=np.float32)
        cand_classes = np.asarray([name2id.get(c, -1) for c in d.data.get("class_name", [])], dtype=np.int64)
        valid = cand_classes >= 0
        cand_boxes, cand_scores, cand_classes = (
            cand_boxes[valid],
            cand_scores[valid],
            cand_classes[valid],
        )
        if cand_boxes.size:
            p_boxes, p_scores, p_classes = plugin.predict_detections(
                source_image=d.metadata["source_image"],
                candidate_boxes=cand_boxes,
                candidate_scores=cand_scores,
                candidate_classes=cand_classes,
                class_names=names,
                class_embed_weight=class_embed_weight,
                device=args.device,
                target_conf=args.conf,
            )
        else:
            p_boxes, p_scores, p_classes = (
                np.zeros((0, 4)),
                np.zeros(0),
                np.zeros(0, dtype=np.int64),
            )

        # Left: GT.
        ax = axes[row, 0]
        ax.imshow(source)
        for gbox, gcls in gt_by_image[img_id]:
            x1, y1, x2, y2 = gbox
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", lw=1.2))
            ax.text(x1, y1 - 2, id2name.get(gcls, "?"), color="lime", fontsize=6, va="bottom")
        ax.set_title(f"GT ({img_id[-24:]})", fontsize=8)

        # Middle: baseline.
        ax = axes[row, 1]
        ax.imshow(source)
        for bbox, bscore, bcls in zip(base_boxes, base_scores, base_cls):
            x1, y1, x2, y2 = bbox
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", lw=1.2))
            ax.text(x1, y1 - 2, f"{bcls} {bscore:.2f}", color="cyan", fontsize=6, va="bottom")
        ax.set_title(f"Baseline ({len(base_boxes)} boxes)", fontsize=8)

        # Right: plugin. Newly-added vs kept vs suppressed.
        ax = axes[row, 2]
        ax.imshow(source)
        base_set = {tuple(np.round(b, 0)) for b in base_boxes}
        for pbox, pscore, pcls in zip(p_boxes, p_scores, p_classes):
            is_new = tuple(np.round(pbox, 0)) not in base_set
            color = "red" if is_new else "blue"
            x1, y1, x2, y2 = pbox
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=1.4))
            ax.text(x1, y1 - 2, f"{id2name.get(int(pcls), '?')} {pscore:.2f}", color=color, fontsize=6, va="bottom")
        ax.set_title(f"Plugin ({len(p_boxes)} boxes, red=new)", fontsize=8)

        for c in range(3):
            axes[row, c].axis("off")

    fig.suptitle(f"GT vs Baseline vs Reason-plugin (conf={args.conf})", fontsize=12)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
