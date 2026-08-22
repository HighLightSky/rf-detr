#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Visualise where the reason plugin ADDS boxes (false positives in red).

For each test image processed, marks:
  - green: ground-truth boxes
  - blue:  plugin output that the baseline already produced
  - lime:  plugin-only box that matches a GT box (recovered true positive)
  - red:   plugin-only box that matches no GT (new false positive)

Only processes a limited number of images to stay fast even while a long
training job is sharing the GPU.

Usage::

    python scripts/viz_false_positives.py \
        --checkpoint output/rfdetr_nano_redo_baseline/checkpoint_best_total.pth \
        --dataset_dir /home/liu/datasets/SHWX-dataset-dict-redo \
        --reason-plugin output/reason_plugin_disc.pth \
        --output output/viz_false_positives.png --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_metrics import (
    IMAGE_EXTENSIONS,
    build_image_size_map,
    load_class_names,
    load_yolo_labels,
)

from rfdetr import RFDETR
from rfdetr.reasoning import PluginLoader


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def main() -> None:
    """Render plugin-only box locations (FP in red, recovered TP in lime)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default="/home/liu/datasets/SHWX-dataset-dict-redo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--reason-plugin", required=True)
    parser.add_argument("--output", default="output/viz_false_positives.png")
    parser.add_argument("--n", type=int, default=5, help="number of images to render")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    names = load_class_names(args.dataset_dir)
    name2id = {name: cid for cid, name in enumerate(names)}
    id2name = {cid: name for name, cid in name2id.items()}
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    ims = build_image_size_map(image_dir)
    gt_records = load_yolo_labels(label_dir, ims)

    plugin = PluginLoader.load(args.reason_plugin)
    plugin.to(args.device)
    model = RFDETR.from_checkpoint(args.checkpoint)
    model.model.device = args.device
    ce = model.model.model.class_embed.weight.detach().to(args.device)

    gt_by_image: dict[str, list[tuple[np.ndarray, int]]] = {}
    for rec in gt_records:
        gt_by_image.setdefault(rec.image_id, []).append((np.asarray(rec.xyxy, float), rec.class_id))

    id_to_path = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    rng = np.random.default_rng(args.seed)
    chosen = list(rng.choice(sorted(id_to_path), size=min(args.n, len(id_to_path)), replace=False))

    ncols = len(chosen)
    grid = Image.new("RGB", (ncols * 512, 512), "white")
    drawn = 0
    with torch.no_grad():
        for col, img_id in enumerate(chosen):
            path = id_to_path[img_id]
            img = Image.open(path).convert("RGB")
            img.thumbnail((512, 512), Image.LANCZOS)
            canvas = Image.new("RGB", (512, 512), (20, 20, 20))
            canvas.paste(img, ((512 - img.width) // 2, (512 - img.height) // 2))
            draw = ImageDraw.Draw(canvas)

            base = model.predict(str(path), threshold=0.25, include_source_image=False)
            base_set = {tuple(np.round(b, 1)) for b in np.asarray(base.xyxy, float)}
            low = model.predict(str(path), threshold=plugin.config.conf_low, include_source_image=True)
            d = low
            cb = np.asarray(d.xyxy, float)
            cs = np.asarray(d.confidence, float)
            cc = np.asarray([name2id.get(c, -1) for c in d.data.get("class_name", [])], int)
            v = cc >= 0
            cb, cs, cc = cb[v], cs[v], cc[v]
            pb = ps = pc = None
            if cb.size:
                pb, ps, pc = plugin.predict_detections(
                    d.metadata["source_image"],
                    cb,
                    cs,
                    cc,
                    names,
                    ce,
                    args.device,
                    0.25,
                )
            g = gt_by_image.get(img_id, [])

            # Draw GT in green.
            for gb, _ in g:
                draw.rectangle(list(gb), outline="green", width=2)
            if pb is not None:
                for b, s, c in zip(pb, ps, pc):
                    if tuple(np.round(b, 1)) in base_set:
                        draw.rectangle(list(b), outline="blue", width=2)
                    elif any(_iou(b, gb) >= 0.5 for gb, _ in g):
                        draw.rectangle(list(b), outline="lime", width=3)
                        draw.text((b[0], b[1] - 8), f"{id2name.get(int(c), '?')} {s:.2f}", fill="lime")
                        drawn += 1
                    else:
                        draw.rectangle(list(b), outline="red", width=3)
                        draw.text((b[0], b[1] - 8), f"{id2name.get(int(c), '?')} {s:.2f} FP", fill="red")
                        drawn += 1
            grid.paste(canvas, (col * 512, 0))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"saved {out} ({len(chosen)} images, {drawn} plugin-only boxes drawn)")


if __name__ == "__main__":
    main()
