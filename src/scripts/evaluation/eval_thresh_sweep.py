# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""逐类置信度阈值扫描：给定 checkpoint 与测试预测，找"FDR 降幅大、召回损失小"的操作点。

用法（两阶段）：
    1. 先生成测试预测（yolo_preds）：
       python src/scripts/test.py -c <测试配置(save_yolo_preds: true)>
    2. 扫描：
       python src/scripts/evaluation/eval_thresh_sweep.py <pred_dir> [--out <输出 json>]

输出：
    - 基线（全 0.25）与候选逐类阈值下的比赛口径指标
      （三大类宏平均 recall/FDR + total + R-F）；
    - 自动挑选"每类 FDR 降 >2pt 且 recall 损失 <6pt"的阈值组合，
      写为 ``class_conf_thresholds`` 字典（可直接贴入测试配置）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from val.competition_metrics import compute_iou, load_yolo_labels, load_yolo_predictions  # noqa: E402

GROUP_IOU = {"ship": 0.5, "aircraft": 0.5, "vehicle": 0.35}
GROUP_CLASSES = {"ship": [0, 1, 2, 3], "aircraft": list(range(4, 24)), "vehicle": [24]}
CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_FDR_GAIN = 0.02  # 候选要求：该类 FDR 降幅 > 2pt
MAX_RECALL_LOSS = 0.04  # 候选要求：该类召回损失 < 4pt


def big_group(class_id: int) -> str:
    """类别 → 大类。"""
    if class_id < 4:
        return "ship"
    if class_id < 24:
        return "aircraft"
    return "vehicle"


def _match_class(plist, glist, iou_th):
    """比赛口径逐类匹配（conf 降序、一对一、IoU 阈值）。返回 (tp, fp)。"""
    tp = fp = 0
    used = [False] * len(glist)
    for p in plist:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(glist):
            if used[j]:
                continue
            iou = compute_iou(p.xyxy, g.xyxy)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_th:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp


def evaluate(gts_by_img, preds_by_img, th_dict: dict[int, float]) -> dict:
    """按比赛口径全量评估，返回三大类宏平均与 total。"""
    n_gt: dict[int, int] = defaultdict(int)
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    for im_id, plist in preds_by_img.items():
        g_all = gts_by_img[im_id]
        for g in g_all:
            n_gt[g.class_id] += 1
        all_ids = {p.class_id for p in plist} | {g.class_id for g in g_all}
        for cid in all_ids:
            plist_c = sorted((p for p in plist if p.class_id == cid), key=lambda b: -b.score)
            plist_c = [p for p in plist_c if p.score >= th_dict.get(cid, 0.25)]
            glist_c = [g for g in g_all if g.class_id == cid]
            t, f = _match_class(plist_c, glist_c, GROUP_IOU[big_group(cid)])
            tp[cid] += t
            fp[cid] += f
    g_R: dict[str, float] = {}
    g_F: dict[str, float] = {}
    for gname, cls in GROUP_CLASSES.items():
        rs = [tp[c] / n_gt[c] for c in cls if n_gt[c] > 0]
        fs = [fp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0 for c in cls]
        g_R[gname] = float(np.mean(rs))
        g_F[gname] = float(np.mean(fs))
    tot_r = float(np.mean(list(g_R.values())))
    tot_f = float(np.mean(list(g_F.values())))
    return {"total_recall": tot_r, "total_fdr": tot_f, "r_minus_f": tot_r - tot_f,
            "group_recall": g_R, "group_fdr": g_F}


def main() -> None:
    """扫描候选阈值并输出推荐组合。"""
    parser = argparse.ArgumentParser(description="逐类置信度阈值扫描")
    parser.add_argument("pred_dir", type=str, help="yolo_preds 目录")
    parser.add_argument("--labels-dir", type=str,
                        default="/home/liu/wzt/datasets/SHWX-dataset-dict-redo/labels/test",
                        help="测试 GT 标签目录")
    parser.add_argument("--images-dir", type=str,
                        default="/home/liu/wzt/datasets/SHWX-dataset-dict-redo/images/test",
                        help="测试图像目录（用于取尺寸）")
    parser.add_argument("--norm-size", type=int, default=1024, help="预测归一化基准尺寸")
    parser.add_argument("--out", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    img_root = Path(args.images_dir)
    image_size_map = {
        p.stem: (args.norm_size, args.norm_size)
        for p in img_root.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    }
    gts = load_yolo_labels(args.labels_dir, image_size_map)
    preds = load_yolo_predictions(args.pred_dir, image_size_map)
    gts_by_img: dict[str, list] = defaultdict(list)
    for g in gts:
        gts_by_img[g.image_id].append(g)
    preds_by_img: dict[str, list] = defaultdict(list)
    for p in preds:
        preds_by_img[p.image_id].append(p)

    base = evaluate(gts_by_img, preds_by_img, {})
    print(f"基线(全0.25): total R={base['total_recall']:.4f} F={base['total_fdr']:.4f} R-F={base['r_minus_f']:.4f}")

    # 逐类扫描：FDR 降 >2pt 且 recall 伤 <6pt 的候选
    chosen: dict[int, float] = {}
    for cid in sorted({p.class_id for p in preds}):
        ng = sum(1 for g in gts if g.class_id == cid)
        if ng == 0:
            continue
        best: tuple[float, float, float] | None = None  # (th, fdr_gain, recall_loss)
        for th in CANDIDATE_THRESHOLDS:
            th_dict = {cid: th}
            res = evaluate(gts_by_img, preds_by_img, th_dict)
            # 该类指标：从 total 不可得，需单独算——直接复用 evaluate 的 group 变化不精确，
            # 改为按类计算
            t, f = 0, 0
            for im_id, plist in preds_by_img.items():
                plist_c = sorted((p for p in plist if p.class_id == cid), key=lambda b: -b.score)
                plist_c = [p for p in plist_c if p.score >= th]
                glist_c = [g for g in gts_by_img[im_id] if g.class_id == cid]
                tt, ff = _match_class(plist_c, glist_c, GROUP_IOU[big_group(cid)])
                t += tt
                f += ff
            R = t / ng
            F = f / (t + f) if t + f else 0.0
            # 基线该类
            t0, f0 = 0, 0
            for im_id, plist in preds_by_img.items():
                plist_c = sorted((p for p in plist if p.class_id == cid), key=lambda b: -b.score)
                plist_c = [p for p in plist_c if p.score >= 0.25]
                glist_c = [g for g in gts_by_img[im_id] if g.class_id == cid]
                tt, ff = _match_class(plist_c, glist_c, GROUP_IOU[big_group(cid)])
                t0 += tt
                f0 += ff
            R0 = t0 / ng
            F0 = f0 / (t0 + f0) if t0 + f0 else 0.0
            fdr_gain = F0 - F
            recall_loss = R0 - R
            if fdr_gain >= MIN_FDR_GAIN and recall_loss <= MAX_RECALL_LOSS:
                if best is None or fdr_gain > best[1] or (fdr_gain == best[1] and recall_loss < best[2]):
                    best = (th, fdr_gain, recall_loss)
        if best is not None:
            chosen[cid] = best[0]
            print(f"  cls{cid}: th={best[0]:.2f} (ΔFDR{best[1]:+.3f}, ΔR{best[2]:+.3f})")

    final = evaluate(gts_by_img, preds_by_img, chosen)
    print(f"\n推荐阈值组合: {json.dumps(chosen)}")
    print(f"应用后: total R={final['total_recall']:.4f} F={final['total_fdr']:.4f} "
          f"R-F={final['r_minus_f']:.4f} (基线 R-F={base['r_minus_f']:.4f}, 提升 {final['r_minus_f'] - base['r_minus_f']:+.4f})")
    if args.out:
        payload = {
            "baseline": base,
            "recommended_thresholds": chosen,
            "final": final,
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()
