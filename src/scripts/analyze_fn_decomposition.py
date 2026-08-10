# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""漏检（FN）成因分解分析脚本。

对指定 checkpoint 在 SHWX 测试集上的全部漏检框做成因分解，回答
"当前 FN 的最大来源是什么"：

1. **检测到但分类错**：与跨小类预测框 IoU≥0.5（模型找到了目标但报错类别，
   比赛口径下该 GT 计为 FN、预测计为 FP）。
2. **同类竞争被挤掉**：存在同类预测框 IoU≥0.5，但比赛 1-to-1 匹配把它
   分配给了同图的另一个 GT（密集排列/重复检测竞争）。
3. **定位临界**：存在任意预测框 IoU∈[0.3,0.5)（模型覆盖了目标，但水平框
   匹配 IoU 不足 0.5，对细长斜置船尤其常见）。
4. **完全漏检**：与任何预测框 IoU<0.3（模型完全没有覆盖该目标）。

同时输出：
- 成因 × 类别矩阵（哪些类漏检最多、漏在哪个成因上）；
- 成因 × 面积分桶（小 <20×20px、中 20-50、大 >50，判断是否小目标漏检）；
- 分类错的去向（漏检目标被报成了哪个类——混淆信息）。

用法：
    python src/scripts/analyze_fn_decomposition.py [checkpoint.pth] [yolo_preds_dir]

- checkpoint.pth：必需（未提供 yolo_preds_dir 时用它推理；也可只做匹配）。
- yolo_preds_dir：可选。提供时直接读取 test.py 输出的 YOLO 格式预测
  （SAVE_YOLO_PREDS=True），跳过推理。
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

import test  # noqa: E402

from rfdetr import RFDETRMedium  # noqa: E402
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    compute_iou,
    evaluate_competition_metrics,
    load_yolo_labels,
    load_yolo_predictions,
)

# 面积分桶（像素）：小 <20×20、中 20×20-50×50、大 >50×50（640 分辨率下）
_AREA_BINS = [(0, 400, "小(<20px)"), (400, 2500, "中(20-50px)"), (2500, float("inf"), "大(>50px)")]

def _area_bucket(gt: BoxRecord) -> str:
    """按 GT 框像素面积返回面积分桶标签。"""
    x0, y0, x1, y1 = gt.xyxy
    area = (x1 - x0) * (y1 - y0)
    for low, high, label in _AREA_BINS:
        if low <= area < high:
            return label
    return "大(>50px)"


def match_and_find_fn(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    config: EvalConfig,
) -> list[tuple[BoxRecord, str, str | None, float]]:
    """复刻比赛匹配规则（同小类、置信度降序、一对一），返回每个 FN 的成因。

    Args:
        gt_records: 真实框记录列表。
        pred_records: 预测框记录列表。
        config: 比赛评测配置（class_to_group / group_iou_thresholds）。

    Returns:
        ``(FN 真实框, 成因标签, 关联预测类别或 None, 关联 IoU)`` 列表。
    """
    # 按图像分组
    gt_by_img: dict[str, list[BoxRecord]] = {}
    pred_by_img: dict[str, list[BoxRecord]] = {}
    for r in gt_records:
        gt_by_img.setdefault(r.image_id, []).append(r)
    for r in pred_records:
        pred_by_img.setdefault(r.image_id, []).append(r)

    fn_findings: list[tuple[BoxRecord, str, str | None, float]] = []
    for image_id, gts in gt_by_img.items():
        preds = pred_by_img.get(image_id, [])
        # 比赛口径：预测按置信度降序，同类、一对一匹配 IoU≥组阈值
        matched_gt: set[int] = set()
        for pred in sorted(preds, key=lambda r: r.score or 0.0, reverse=True):
            group_thr = config.group_iou_thresholds.get(config.class_to_group[pred.class_id], 0.5)
            best_gt, best_iou = None, 0.0
            for gi, gt in enumerate(gts):
                if gi in matched_gt or gt.class_id != pred.class_id:
                    continue
                iou = compute_iou(pred.xyxy, gt.xyxy)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_gt is not None and best_iou >= group_thr:
                matched_gt.add(best_gt)

        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            # 归因：跨类 IoU≥0.5 → 分类错；同类 IoU≥0.5 → 同类竞争；
            # 任意 IoU∈[0.3,0.5) → 定位临界；否则完全漏检
            cross_iou, cross_cls = 0.0, None
            same_iou = 0.0
            any_iou = 0.0
            for p in preds:
                iou = compute_iou(p.xyxy, gt.xyxy)
                any_iou = max(any_iou, iou)
                if p.class_id == gt.class_id:
                    same_iou = max(same_iou, iou)
                elif iou > cross_iou:
                    cross_iou, cross_cls = iou, test.CLASS_NAMES[p.class_id]
            if cross_iou >= 0.5:
                fn_findings.append((gt, "检测到但分类错", cross_cls, cross_iou))
            elif same_iou >= 0.5:
                fn_findings.append((gt, "同类竞争被挤掉", None, same_iou))
            elif any_iou >= 0.3:
                fn_findings.append((gt, "定位临界", None, any_iou))
            else:
                fn_findings.append((gt, "完全漏检", None, any_iou))
    return fn_findings


def main() -> None:
    """主流程：加载 GT/预测 → 比赛口径匹配 → FN 成因分解。"""
    checkpoint_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    yolo_preds_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if checkpoint_path is None:
        print("用法: python src/scripts/analyze_fn_decomposition.py [checkpoint.pth] [yolo_preds_dir]")
        sys.exit(1)

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

    # 预测来源：优先读 YOLO 预测目录（test.py SAVE_YOLO_PREDS 输出），否则推理
    if yolo_preds_dir is not None and yolo_preds_dir.exists():
        print(f"[i] 读取 YOLO 预测: {yolo_preds_dir}")
        pred_records = load_yolo_predictions(yolo_preds_dir, image_size_map)
    else:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
        device = test.resolve_device("cuda:0")
        print(f"[i] 加载 {checkpoint_path} 并推理 ...")
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
        # 顺带输出 YOLO 格式预测（checkpoint 同目录 yolo_preds/），供后续统计复用
        test.save_yolo_predictions(pred_records, checkpoint_path.parent / "yolo_preds", image_size_map)

    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)
    all_result = eval_results["all"]
    print(f"[i] 比赛口径: TP={all_result.tp} FP={all_result.fp} FN={all_result.fn}")

    findings = match_and_find_fn(gt_records, pred_records, config)
    n_fn = len(findings)
    counter = Counter(label for _, label, _, _ in findings)
    print(f"\n===== 漏检（FN={n_fn}）成因分解 =====")
    for label, count in counter.most_common():
        print(f"  {label:10s} {count:5d}  ({count / max(n_fn, 1) * 100:5.1f}%)")

    # 成因 × 大类（船/飞机/车辆）
    group_of = {cid: g for cid, g in class_to_group.items()}
    print("\n===== 成因 × 大类 =====")
    groups = ("ship", "aircraft", "vehicle")
    print(f"  {'成因':<12s} " + " ".join(f"{g:>9s}" for g in groups))
    for label in counter:
        row = [sum(1 for gt, l, _, _ in findings if l == label and group_of[gt.class_id] == g) for g in groups]
        print(f"  {label:<12s} " + " ".join(f"{n:9d}" for n in row))

    # 成因 × 面积分桶
    print("\n===== 成因 × 面积分桶（像素）=====")
    bin_labels = [b[2] for b in _AREA_BINS]
    print(f"  {'成因':<12s} " + " ".join(f"{b:>12s}" for b in bin_labels))
    for label in counter:
        row = [sum(1 for gt, l, _, _ in findings if l == label and _area_bucket(gt) == b) for b in bin_labels]
        print(f"  {label:<12s} " + " ".join(f"{n:12d}" for n in row))

    # 漏检最多的类别（top 8）
    print("\n===== 漏检最多的类别（Top 8）=====")
    by_class = Counter(test.CLASS_NAMES[gt.class_id] for gt, _, _, _ in findings)
    for cls_name, count in by_class.most_common(8):
        n_total = sum(1 for g in gt_records if test.CLASS_NAMES[g.class_id] == cls_name)
        print(f"  {cls_name:<10s} FN={count:4d}  (该类 GT 总数 {n_total})")

    # 分类错的去向（混淆信息）
    print("\n===== '检测到但分类错'的去向（被报成哪个类）=====")
    wrong_to = Counter(cls for _, label, cls, _ in findings if label == "检测到但分类错" and cls)
    for cls_name, count in wrong_to.most_common(6):
        print(f"  → {cls_name:<10s} {count:4d}")

    # 结论：最大来源
    top_label, top_count = counter.most_common(1)[0]
    print("\n" + "=" * 60)
    print(f"结论：FN 最大来源 = 【{top_label}】（{top_count}/{n_fn} = {top_count / max(n_fn, 1) * 100:.1f}%）")
    print("=" * 60)


if __name__ == "__main__":
    main()
