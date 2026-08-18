#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Diagnose the FFT reason plugin's boost behaviour on the test split.

For every test image, buckets the frozen detector's low-threshold candidates
into clear ``A`` and blurry ``B``, runs the plugin, and aggregates how the
plugin's adjustment moves each ``B`` relative to the final ``conf`` threshold,
split by whether the ``B`` matches a ground-truth box.

Three bands matter:
  - ``B < conf``  — missed-object candidates; boosting them across ``conf``
    recovers recall.
  - ``conf <= B < conf_high`` — detections that already clear ``conf``;
    suppressing them *loses* valid detections (and recall).

Usage::

    python scripts/diag_reason_plugin.py \\
        --checkpoint output/rfdetr_nano_redo_baseline/checkpoint_best_total.pth \\
        --dataset_dir /home/liu/datasets/SHWX-dataset-dict-redo \\
        --reason-plugin output/reason_plugin_c5.pth --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_metrics import (
    IMAGE_EXTENSIONS,
    build_image_size_map,
    load_class_names,
    load_yolo_labels,
)

from rfdetr import RFDETR
from rfdetr.reasoning import CandidateBuilder, PluginLoader


def main() -> None:
    """Run the diagnostic and print per-band boost statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default="/home/liu/datasets/SHWX-dataset-dict-redo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--reason-plugin", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--boost-scale",
        type=float,
        default=None,
        help="Override the plugin's boost_scale (useful for sweeping without retraining).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=None,
        help="Override the plugin's p_threshold decision line.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    names = load_class_names(args.dataset_dir)
    name2id = {name: cid for cid, name in enumerate(names)}
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    image_size_map = build_image_size_map(image_dir)
    gt_records = load_yolo_labels(label_dir, image_size_map)

    gt_by_image: dict[str, list[tuple[np.ndarray, int]]] = {}
    for rec in gt_records:
        gt_by_image.setdefault(rec.image_id, []).append((np.asarray(rec.xyxy, dtype=np.float32), rec.class_id))

    plugin = PluginLoader.load(args.reason_plugin)
    if args.boost_scale is not None:
        plugin.config.boost_scale = args.boost_scale
    if args.p_threshold is not None:
        plugin.config.p_threshold = args.p_threshold
    plugin.to(args.device)
    model = RFDETR.from_checkpoint(args.checkpoint)
    model.model.device = args.device
    class_embed_weight = model.model.model.class_embed.weight.detach().to(args.device)

    builder = CandidateBuilder(plugin.config)
    id_to_path = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    # Aggregate: for each B, (orig_score, is_gt_matched, plugin_prob, new_score)
    rows: list[tuple[float, int, float, float]] = []
    with torch.no_grad():
        for img_id, path in sorted(id_to_path.items()):
            res = model.predict(str(path), threshold=plugin.config.conf_low, include_source_image=True)
            d = res
            class_names_pred = d.data.get("class_name", [])
            source_image = d.metadata.get("source_image")
            if source_image is None:
                continue
            cand_boxes = np.asarray(d.xyxy, dtype=np.float32)
            cand_scores = np.asarray(d.confidence, dtype=np.float32)
            cand_classes = np.asarray([name2id.get(c, -1) for c in class_names_pred], dtype=np.int64)
            valid = cand_classes >= 0
            cand_boxes, cand_scores, cand_classes = cand_boxes[valid], cand_scores[valid], cand_classes[valid]
            if not cand_boxes.size:
                continue

            a_inds, b_inds = builder.split(cand_boxes, cand_scores)
            if not (b_inds.size and a_inds.size):
                continue
            pairs = builder.pair_topk(cand_boxes, a_inds, b_inds)
            if not pairs.size:
                continue
            a_idx = pairs[:, 0]
            b_idx = pairs[:, 1]

            from rfdetr.reasoning.plugin import _crop_joint_patches, _crop_patches

            a_patches = _crop_patches(source_image, cand_boxes[a_idx])
            b_patches = _crop_patches(source_image, cand_boxes[b_idx])
            joint_patches = _crop_joint_patches(source_image, cand_boxes[a_idx], cand_boxes[b_idx])
            logits, _, _ = plugin(
                torch.from_numpy(a_patches).to(args.device),
                torch.from_numpy(b_patches).to(args.device),
                torch.from_numpy(joint_patches).to(args.device),
                torch.from_numpy(cand_boxes[a_idx].astype(np.float32)).to(args.device),
                torch.from_numpy(cand_boxes[b_idx].astype(np.float32)).to(args.device),
                torch.from_numpy(cand_classes[a_idx]).to(args.device),
                class_embed_weight,
                img_shape=(source_image.shape[0], source_image.shape[1]),
            )
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            adjust = plugin.config.boost_scale * (prob - plugin.config.p_threshold)

            # Per-B: mean adjust over its pairs, then apply to the original score.
            per_b = np.zeros(b_inds.size, dtype=np.float32)
            counts = np.zeros(b_inds.size, dtype=np.float32)
            b_pos = np.searchsorted(b_inds, b_idx)
            np.add.at(per_b, b_pos, adjust)
            np.add.at(counts, b_pos, 1.0)
            adj_mean = np.where(counts > 0, per_b / np.maximum(counts, 1), 0.0)
            new_scores = cand_scores[b_inds] + adj_mean
            b_probs = np.zeros(b_inds.size, dtype=np.float32)
            np.add.at(b_probs, b_pos, prob)
            b_prob_mean = np.where(counts > 0, b_probs / np.maximum(counts, 1), 0.5)

            # Match each B to GT (class-aware IoU >= 0.5).
            gts = gt_by_image.get(img_id, [])
            for i, bi in enumerate(b_inds):
                matched = 0
                for gbox, gcls in gts:
                    if gcls != cand_classes[bi]:
                        continue
                    if _iou(gbox, cand_boxes[bi]) >= 0.5:
                        matched = 1
                        break
                rows.append(
                    (
                        float(cand_scores[bi]),
                        matched,
                        float(b_prob_mean[i]),
                        float(new_scores[i]),
                    )
                )

    rows_arr = np.asarray(rows, dtype=np.float64)
    print(f"\n=== 诊断: {len(rows)} 个 B 候选, conf={args.conf}, boost_scale={plugin.config.boost_scale} ===")
    print(f"B 区间: [{plugin.config.conf_low}, {plugin.config.conf_high}), A 阈值: {plugin.config.tau_high}")

    for label, mask in [
        ("全部 B", np.ones(len(rows), dtype=bool)),
        ("匹配 GT (漏检候选)", rows_arr[:, 1] == 1),
        ("虚警 (无 GT 匹配)", rows_arr[:, 1] == 0),
    ]:
        sub = rows_arr[mask]
        if not len(sub):
            print(f"\n  [{label}] 无样本")
            continue
        orig = sub[:, 0]
        new = sub[:, 3]
        prob = sub[:, 2]
        below = orig < args.conf
        print(f"\n  [{label}] n={len(sub)}")
        below = 100 * np.mean(orig < 0.25)
        band = 100 * np.mean((orig >= 0.25) & (orig < plugin.config.conf_high))
        print(
            f"    原始分: 均值={orig.mean():.3f} 中位={np.median(orig):.3f} "
            f"分布[0.05,0.25)={below:.0f}% [0.25,{plugin.config.conf_high})={band:.0f}%"
        )
        print(f"    插件概率 p: 均值={prob.mean():.3f} 中位={np.median(prob):.3f}")
        print(f"    提分(均值): {np.mean(new - orig):+.4f} (正=提, 负=压)")
        if below.any():
            rec_ratio = np.mean(new[below] >= args.conf)
            print(f"    原始<{args.conf} 的 {int(below.sum())} 个: 提分后过阈值 {100 * rec_ratio:.1f}%")
        already = orig >= args.conf
        if already.any():
            lost_ratio = np.mean(new[already] < args.conf)
            n_already = int(already.sum())
            print(
                f"    原始>={args.conf} 的 {n_already} 个: "
                f"被压到阈值下 {100 * lost_ratio:.1f}% (丢失有效检测!)"
            )

    # Net effect on recall-ish proxy: matched B below conf that crossed vs matched B above conf that dropped.
    matched = rows_arr[rows_arr[:, 1] == 1]
    if len(matched):
        below_m = matched[:, 0] < args.conf
        gained = np.sum((matched[below_m, 3] >= args.conf)) if below_m.any() else 0
        already_m = matched[:, 0] >= args.conf
        lost = np.sum((matched[already_m, 3] < args.conf)) if already_m.any() else 0
        print(f"\n  [净效果·匹配GT] 提回漏检: {int(gained)} 个 | 压丢有效: {int(lost)} 个 | 净: {int(gained - lost)}")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


if __name__ == "__main__":
    main()
