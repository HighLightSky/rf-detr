# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""虚警成因分解分析脚本。

对指定 checkpoint 在 SHWX 测试集上的舰船大类虚警做 5 类成因分解：

1. **同小类重复检测**：与同小类真实框 IoU≥0.5 但该真实框已被更高置信度预测匹配。
2. **船类小类混淆**：与**其他**船类小类真实框 IoU≥0.5（真船分错小类，比赛口径
   下计为 FP + 另一个类的 FN）。
3. **跨大类混淆**：与飞机/车辆真实框 IoU≥0.5。
4. **定位临界**：与任意舰船真实框 IoU 在 [0.3, 0.5)（模型找到了船，但水平框
   匹配 IoU 不足 0.5，对细长斜置船尤其常见）。
5. **纯背景虚警**：与任何真实框 IoU<0.3（海岸、港口设施、建筑物等被误检为船）。

同时输出虚警与正确检出的置信度分布，判断固定阈值 0.25 附近是否存在大量
"贴线"虚警（可用阈值/校准手段压制）还是高置信度"硬"虚警（需模型改进）。

用法：
    python src/scripts/analyze_fp_decomposition.py <checkpoint.pth> [输出目录]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

# 复用 test.py 的推理管线与数据集配置（DATASET="shwx" 为默认值）
import test  # noqa: E402

from rfdetr import RFDETRMedium  # noqa: E402
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    compute_iou,
    evaluate_competition_metrics,
    load_yolo_labels,
)


def match_fps(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    config: EvalConfig,
) -> list[tuple[BoxRecord, str, float]]:
    """复刻比赛匹配规则（同小类、置信度降序、一对一），返回每个 ship 虚警的成因。

    Args:
        gt_records: 真实框记录列表。
        pred_records: 预测框记录列表。
        config: 比赛评测配置。

    Returns:
        ``(虚警框, 成因标签, 关联IoU)`` 列表。
    """
    ship_class_ids = {cid for cid, g in config.class_to_group.items() if g == "ship"}

    # 按图像分组
    gt_by_img: dict[str, list[BoxRecord]] = {}
    pred_by_img: dict[str, list[BoxRecord]] = {}
    for r in gt_records:
        gt_by_img.setdefault(r.image_id, []).append(r)
    for r in pred_records:
        pred_by_img.setdefault(r.image_id, []).append(r)

    findings: list[tuple[BoxRecord, str, float]] = []
    for image_id, preds in pred_by_img.items():
        # 只关心 ship 大类的预测
        ship_preds = [p for p in preds if p.class_id in ship_class_ids]
        if not ship_preds:
            continue
        gts = gt_by_img.get(image_id, [])
        ship_gts = [g for g in gts if g.class_id in ship_class_ids]
        other_group_gts = [g for g in gts if g.class_id not in ship_class_ids]

        # 与比赛一致：置信度降序、同小类、一对一匹配
        matched: set[int] = set()
        for pred in sorted(ship_preds, key=lambda r: r.score or 0.0, reverse=True):
            best_gt, best_iou = None, 0.0
            for gi, gt in enumerate(ship_gts):
                if gi in matched or gt.class_id != pred.class_id:
                    continue
                iou = compute_iou(pred.xyxy, gt.xyxy)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_gt is not None and best_iou >= config.group_iou_thresholds["ship"]:
                matched.add(best_gt)
            else:
                findings.append(_classify_fp(pred, ship_gts, other_group_gts))
    return findings


def _classify_fp(
    pred: BoxRecord,
    ship_gts: list[BoxRecord],
    other_group_gts: list[BoxRecord],
) -> tuple[BoxRecord, str, float]:
    """对单个虚警框判定成因类别。"""
    # 1) 同小类真实框：≥0.5 → 重复检测；[0.3,0.5) → 定位临界
    same_iou = max((compute_iou(pred.xyxy, g.xyxy) for g in ship_gts if g.class_id == pred.class_id), default=0.0)
    if same_iou >= 0.5:
        return pred, "同小类重复检测", same_iou
    # 2) 其他船类小类真实框
    cross_iou = max((compute_iou(pred.xyxy, g.xyxy) for g in ship_gts if g.class_id != pred.class_id), default=0.0)
    if cross_iou >= 0.5:
        return pred, "船类小类混淆", cross_iou
    # 3) 跨大类（飞机/车辆）真实框
    group_iou = max((compute_iou(pred.xyxy, g.xyxy) for g in other_group_gts), default=0.0)
    if group_iou >= 0.5:
        return pred, "跨大类混淆", group_iou
    # 4) 任意舰船真实框 [0.3, 0.5)
    ship_marginal = max((compute_iou(pred.xyxy, g.xyxy) for g in ship_gts), default=0.0)
    if ship_marginal >= 0.3:
        return pred, "定位临界", ship_marginal
    return pred, "纯背景虚警", 0.0


def main() -> None:
    """主流程：推理 → 比赛口径匹配 → 虚警分解与置信度统计。"""
    if len(sys.argv) < 2:
        print("用法: python src/scripts/analyze_fp_decomposition.py <checkpoint.pth>")
        sys.exit(1)
    checkpoint_path = Path(sys.argv[1]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    cfg = test.DATASET_CONFIGS["shwx"]
    data_dir = Path(cfg["data_dir"])
    test_image_paths = test.read_test_image_paths(data_dir / cfg["image_dir"])
    image_size_map = test.build_image_size_map(test_image_paths)
    gt_records = load_yolo_labels(data_dir / cfg["label_dir"], image_size_map)

    class_to_group = {int(k): v for k, v in cfg["class_to_group"].items()}
    config = EvalConfig(
        class_to_group=class_to_group,
        group_iou_thresholds={k: float(v) for k, v in cfg["group_iou_thresholds"].items()},
        default_iou_threshold=0.50,
        class_aware=True,
    )

    device = test.resolve_device("cuda:0")
    print(f"[i] 加载 {checkpoint_path} ...")
    model = RFDETRMedium.from_checkpoint(str(checkpoint_path))
    model.model.model = model.model.model.to(device)
    model.model.model.eval()
    pred_records, _, _, _ = test.predict_batched_to_records(
        model,
        test_image_paths,
        device,
        conf_threshold=0.25,
        batch_size=32,
        num_workers=12,
    )

    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)
    ship_result = eval_results["groups"]["ship"]
    print(f"[i] 比赛口径 ship: TP={ship_result.tp} FP={ship_result.fp} FN={ship_result.fn}")

    findings = match_fps(gt_records, pred_records, config)
    counter = Counter(label for _, label, _ in findings)
    print("\n===== ship 虚警成因分解 =====")
    for label, count in counter.most_common():
        print(f"  {label:12s} {count:5d}  ({count / max(len(findings), 1) * 100:5.1f}%)")

    # 各类成因的置信度区间
    bins = [(0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print("\n===== 虚警置信度分布（按成因）=====")
    print(f"  {'成因':<12s} " + " ".join(f"[{a:.2f},{b:.2f})" for a, b in bins))
    for label in counter:
        scores = [p.score or 0.0 for p, l, _ in findings if l == label]
        row = [sum(1 for s in scores if a <= s < b) for a, b in bins]
        print(f"  {label:<12s} " + " ".join(f"{n:6d}" for n in row))

    # 正确检出的置信度分布
    tp_scores: list[float] = []
    pred_by_img: dict[str, list[BoxRecord]] = {}
    for r in pred_records:
        pred_by_img.setdefault(r.image_id, []).append(r)
    matched_gt_ids: set[int] = set()
    for image_id, preds in pred_by_img.items():
        gts = [g for g in gt_records if g.image_id == image_id]
        for p in sorted(preds, key=lambda r: r.score or 0.0, reverse=True):
            for gi, g in enumerate(gts):
                key = (image_id, gi)
                if key in matched_gt_ids or g.class_id != p.class_id:
                    continue
                if compute_iou(p.xyxy, g.xyxy) >= config.group_iou_thresholds.get(
                    config.class_to_group[g.class_id], 0.5
                ):
                    tp_scores.append(p.score or 0.0)
                    matched_gt_ids.add(key)
                    break
    print("\n===== 正确检出置信度分布 =====")
    row = [sum(1 for s in tp_scores if a <= s < b) for a, b in bins]
    print("  TP " + " ".join(f"{n:6d}" for n in row) + f"   (n={len(tp_scores)})")

    # ── 置信度阈值扫描：只对 ship 大类生效，观察 TP/FP/FN 与 FDR/Recall 变化 ──
    print("\n===== ship 大类置信度阈值扫描（其他类保持 0.25）=====")
    ship_class_ids = {cid for cid, g in config.class_to_group.items() if g == "ship"}
    for thr in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        filtered = [p for p in pred_records if p.class_id not in ship_class_ids or (p.score or 0.0) >= thr]
        res = evaluate_competition_metrics(gt_records, filtered, config)
        s = res["groups"]["ship"]
        a = res["all"]
        print(
            f"  ship 阈值 {thr:.2f}: ship TP={s.tp:4d} FP={s.fp:4d} FN={s.fn:4d} "
            f"Recall={s.recall:.4f} FDR={s.fdr:.4f} Prec={s.precision:.4f} | "
            f"all Recall={a.recall:.4f} FDR={a.fdr:.4f}"
        )


if __name__ == "__main__":
    main()
