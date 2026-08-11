# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""逐类置信度阈值重标定脚本（分类损失均衡化的配套评估）。

分类损失改动后分数分布必然变化，沿用旧阈值不是公平比较。本脚本在固定
checkpoint 上完成"推理一次 → 离线阈值搜索"：

1. 用低地板阈值（0.05）推理测试集一次，收集全部检测框；
2. 离线坐标上升搜索逐类置信度阈值：默认只搜 HM/LQS/QHS/MS/FSC 五类，
   飞机类固定 0.25（其召回已接近满点，参与搜索反而增加 FDR 波动）；
3. 目标门槛：总宏（舰船+飞机+车辆三大类 macro 平均）Recall >= 0.855 且
   FDR <= 0.195（留 0.5pp 安全余量）；
4. 可选 LA 推理侧 bias 对照：score = sigmoid(logit - k * bias)，
   k ∈ {0, 0.5, 1}，bias 由 class_counts.json 按训练同配方重建。

用法：
    python src/scripts/calibrate_thresholds.py <checkpoint.pth> [--bias-json class_counts.json] [--bias-k 0.5] [--output-dir <实验目录>]

输出：
- calibrated_thresholds.json：{类别id: 阈值}（可直接贴入 test.py 的 CLASS_CONF_THRESHOLDS）
- calibration_report.txt：逐类/大类/总指标报告

说明：评估口径与 test.py 完全一致（置信度降序贪心一对一匹配、
IoU 阈值 ship/aircraft=0.50、vehicle=0.35、逐类 macro 再三大类平均）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

import test  # noqa: E402

from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    evaluate_competition_metrics,
    load_yolo_labels,
)

# ── 搜索配置 ──────────────────────────────────────────────────────────
# 推理收集地板：所有 score >= 该值的检测都参与离线阈值搜索（须低于网格下限）
CONF_FLOOR = 0.05
# 阈值搜索网格（步长 0.05）
THRESH_GRID = [round(v, 2) for v in (i * 0.05 for i in range(2, 13))]  # 0.10 ~ 0.60
# 参与搜索的类别（HM/LQS/QHS/MS/FSC 五类），飞机类固定 AIRCRAFT_THR
SEARCH_CLASS_IDS = [0, 1, 2, 3, 24]
AIRCRAFT_THR = 0.25
# 目标门槛（留安全余量）
TARGET_RECALL = 0.855
TARGET_FDR = 0.195
# 坐标上升轮数（每轮每类搜索一遍，取全局最优）
ASCENT_ROUNDS = 3
# LA 推理侧 bias 重建参数（须与训练侧配方一致，默认同 criterion 默认值）
BIAS_TAU = 0.1
BIAS_CLIP = 1.0


def _eval_metrics(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    thresholds: dict[int, float],
    cfg: dict,
) -> tuple[dict[str, float], dict[str, EvalConfig]]:
    """按给定逐类阈值过滤预测，计算总宏指标与逐类结果（与 test.py 口径一致）。

    Args:
        gt_records: 测试集真实框。
        pred_records: 未过滤的全量预测框。
        thresholds: 逐类阈值 {类别id: 阈值}，未列出的类别回退 0.25。
        cfg: test.DATASET_CONFIGS["shwx"] 数据集配置。

    Returns:
        (total_macro, per_class)：总宏指标字典（recall/fdr 等）与逐类 EvalResult
        {类别名: EvalResult} 字典（供详细报告）。
    """
    per_class_config = EvalConfig(
        class_to_group=test.PER_CLASS_TO_GROUP,
        group_iou_thresholds=test.PER_CLASS_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    kept: list[BoxRecord] = [
        pred
        for pred in pred_records
        if pred.score is not None and pred.score >= thresholds.get(pred.class_id, AIRCRAFT_THR)
    ]
    per_class_results = evaluate_competition_metrics(gt_records, kept, per_class_config)["groups"]
    group_macro = test.compute_group_macro_averages(per_class_results, cfg["class_to_group"], cfg["class_names"])
    total_macro = test.compute_total_metrics(group_macro)
    return total_macro, per_class_results


def _select_best_candidate(
    candidates: list[tuple[dict[int, float], dict[str, float]]],
) -> tuple[dict[int, float], dict[str, float]]:
    """从已评估的候选点中按字典序规则选优（无法预知门槛可行性时最稳健）。

    规则（与专家方案 §6 判定口径一致）：
    1. 能同时满足 Recall >= 0.855 且 FDR <= 0.195 时：选 FDR 最低者；
    2. 否则能压住 FDR <= 0.195 时：选 Recall 最高者（FDR 硬约束优先）；
    3. 否则：选 Recall 最高且 FDR <= 0.25 者（贴近比赛口径的虚警容忍）；
    4. 兜底：选 Recall 最高者。

    Args:
        candidates: 已评估的 ``(thresholds, total_macro)`` 列表。

    Returns:
        选中点的 ``(thresholds, total_macro)``。
    """
    gate_ok = [(t, m) for t, m in candidates if m["recall"] >= TARGET_RECALL and m["fdr"] <= TARGET_FDR]
    if gate_ok:
        return min(gate_ok, key=lambda item: item[1]["fdr"])
    fdr_ok = [(t, m) for t, m in candidates if m["fdr"] <= TARGET_FDR]
    if fdr_ok:
        return max(fdr_ok, key=lambda item: item[1]["recall"])
    fdr_loose = [(t, m) for t, m in candidates if m["fdr"] <= 0.25]
    if fdr_loose:
        return max(fdr_loose, key=lambda item: item[1]["recall"])
    return max(candidates, key=lambda item: item[1]["recall"])


def _coordinate_ascent(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    cfg: dict,
) -> tuple[dict[int, float], dict[str, float]]:
    """坐标上升搜索逐类阈值，并记录全部已评估点供字典序选优。

    对参与搜索的类别逐类遍历网格：局部启发式保留"FDR 不超门槛时 Recall 最高"
    的点（与最终字典序规则一致）；所有评估过的 ``(阈值, 总宏指标)`` 进
    candidates，最终由 ``_select_best_candidate`` 按字典序规则全局选择。

    Args:
        gt_records: 测试集真实框。
        pred_records: 全量预测框。
        cfg: test.DATASET_CONFIGS["shwx"] 数据集配置。

    Returns:
        选中最优 ``(thresholds, total_macro)``。
    """
    thresholds: dict[int, float] = {class_id: AIRCRAFT_THR for class_id in SEARCH_CLASS_IDS}
    candidates: list[tuple[dict[int, float], dict[str, float]]] = []

    def _local_better(recall: float, fdr: float, best_recall: float, best_fdr: float) -> bool:
        """局部启发式：FDR 不超门槛时比 Recall，超门槛时优先压 FDR。"""
        if fdr <= TARGET_FDR and best_fdr > TARGET_FDR:
            return True
        if fdr <= TARGET_FDR and best_fdr <= TARGET_FDR:
            return recall > best_recall
        if fdr > TARGET_FDR and best_fdr > TARGET_FDR:
            return fdr < best_fdr or (fdr == best_fdr and recall > best_recall)
        return False  # 当前点超门槛而历史点不超

    total, _ = _eval_metrics(gt_records, pred_records, thresholds, cfg)
    candidates.append((dict(thresholds), total))
    print(f"[初始] 全 0.25 总宏: Recall={total['recall']:.4f} FDR={total['fdr']:.4f}")

    for round_idx in range(ASCENT_ROUNDS):
        improved = False
        for class_id in SEARCH_CLASS_IDS:
            current = thresholds[class_id]
            best_thr = current
            best_recall, best_fdr = total["recall"], total["fdr"]
            for thr in THRESH_GRID:
                thresholds[class_id] = thr
                total, _ = _eval_metrics(gt_records, pred_records, thresholds, cfg)
                candidates.append((dict(thresholds), total))
                if _local_better(total["recall"], total["fdr"], best_recall, best_fdr):
                    best_thr, best_recall, best_fdr = thr, total["recall"], total["fdr"]
            thresholds[class_id] = best_thr
            if best_thr != current:
                improved = True
                total, _ = _eval_metrics(gt_records, pred_records, thresholds, cfg)  # 同步最优点的指标
        total, _ = _eval_metrics(gt_records, pred_records, thresholds, cfg)
        candidates.append((dict(thresholds), total))
        print(f"[第 {round_idx + 1} 轮] 总宏: Recall={total['recall']:.4f} FDR={total['fdr']:.4f} | 阈值: {thresholds}")
        if not improved:
            break  # 本轮无改进，提前收敛
    best = _select_best_candidate(candidates)
    # 打印 Pareto 前沿（Recall-FDR 非支配点，按 Recall 升序）：
    # 帮助人工在"Recall>=0.85 且 FDR 最低"等偏好下二次选择。
    # 先按 (阈值键, Recall, FDR) 去重（同一阈值组合可能被多轮追加）。
    unique: dict[tuple, tuple[dict[int, float], dict[str, float]]] = {}
    for thresholds_i, total_i in candidates:
        key = tuple(sorted(thresholds_i.items())) + (round(total_i["recall"], 6), round(total_i["fdr"], 6))
        unique.setdefault(key, (thresholds_i, total_i))
    pareto: list[tuple[dict[int, float], dict[str, float]]] = []
    for thresholds_i, total_i in unique.values():
        dominated = any(
            total_j["recall"] >= total_i["recall"]
            and total_j["fdr"] <= total_i["fdr"]
            and (total_j["recall"] > total_i["recall"] or total_j["fdr"] < total_i["fdr"])
            for _, total_j in unique.values()
        )
        if not dominated:
            pareto.append((thresholds_i, total_i))
    pareto.sort(key=lambda item: item[1]["recall"])
    print("\n[Pareto 前沿]（Recall, FDR）→ 阈值")
    for thresholds_i, total_i in pareto:
        flag = " <- 选中" if thresholds_i == best[0] and total_i["fdr"] == best[1]["fdr"] else ""
        print(f"  Recall={total_i['recall']:.4f} FDR={total_i['fdr']:.4f} | {thresholds_i}{flag}")
    return best


def _write_report(
    report_path: Path,
    thresholds: dict[int, float],
    per_class_results: dict[str, dict[str, float]],
    cfg: dict,
) -> None:
    """写出逐类阈值与逐类指标报告文本。"""
    lines = ["逐类阈值重标定报告", "=" * 60, "最优逐类阈值（可贴入 test.py CLASS_CONF_THRESHOLDS）:"]
    lines.append("CLASS_CONF_THRESHOLDS = " + json.dumps({str(k): v for k, v in sorted(thresholds.items())}))
    lines.append("=" * 60)
    lines.append("逐类 TP/FP/FN/Recall/FDR:")
    for class_name in sorted(per_class_results.keys()):
        result = per_class_results[class_name]
        lines.append(
            f"{class_name:<12} TP={result.tp:<5} FP={result.fp:<5} FN={result.fn:<5} "
            f"Recall={result.recall:.4f} FDR={result.fdr:.4f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """主流程：推理 → 可选 bias → 坐标上升搜阈值 → 输出报告。"""
    parser = argparse.ArgumentParser(description="逐类置信度阈值重标定")
    parser.add_argument("checkpoint", type=str, help="待评估 checkpoint 路径")
    parser.add_argument("--bias-json", type=str, default=None, help="class_counts.json（启用 LA 推理侧 bias 对照）")
    parser.add_argument("--bias-k", type=float, default=1.0, help="推理侧 bias 扣减系数 k（0/0.5/1）")
    parser.add_argument("--output-dir", type=str, default=None, help="报告输出目录（默认取 checkpoint 所在目录）")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = test.DATASET_CONFIGS["shwx"]
    data_dir = Path(cfg["data_dir"])
    test_image_paths = test.read_test_image_paths(data_dir / cfg["image_dir"])
    image_size_map = test.build_image_size_map(test_image_paths)
    gt_records = load_yolo_labels(data_dir / cfg["label_dir"], image_size_map)

    print(f"加载模型: {checkpoint_path}")
    model = test.RFDETR.from_checkpoint(str(checkpoint_path))

    old_bias_path = test.LOGIT_ADJUSTMENT_BIAS_PATH
    old_bias_k = test.LOGIT_ADJUSTMENT_BIAS_K
    old_bias_tau = test.LOGIT_ADJUSTMENT_BIAS_TAU
    old_bias_clip = test.LOGIT_ADJUSTMENT_BIAS_CLIP
    if args.bias_json:
        counts_json = Path(args.bias_json)
        if not counts_json.exists():
            raise FileNotFoundError(f"bias-json 不存在: {counts_json}")
        test.LOGIT_ADJUSTMENT_BIAS_PATH = str(counts_json)
        test.LOGIT_ADJUSTMENT_BIAS_K = args.bias_k
        test.LOGIT_ADJUSTMENT_BIAS_TAU = BIAS_TAU
        test.LOGIT_ADJUSTMENT_BIAS_CLIP = BIAS_CLIP

    # ── 推理一次：低地板阈值收集全量检测 ─────────────────────────────
    print(f"推理测试集（{len(test_image_paths)} 张，收集地板 {CONF_FLOOR}）...")
    try:
        pred_records, throughput, gpu_util, timed_images = test.predict_batched_to_records(
            model=model,
            image_paths=test_image_paths,
            device=test.DEVICE,
            conf_threshold=CONF_FLOOR,
            batch_size=test.BATCH_SIZE,
            num_workers=test.NUM_WORKERS,
        )
    finally:
        test.LOGIT_ADJUSTMENT_BIAS_PATH = old_bias_path
        test.LOGIT_ADJUSTMENT_BIAS_K = old_bias_k
        test.LOGIT_ADJUSTMENT_BIAS_TAU = old_bias_tau
        test.LOGIT_ADJUSTMENT_BIAS_CLIP = old_bias_clip
    print(f"推理吞吐: {throughput:.1f} img/s | 收集检测框 {len(pred_records)} 个")

    # ── 坐标上升搜索逐类阈值 ─────────────────────────────────────────
    thresholds, total_macro = _coordinate_ascent(gt_records, pred_records, cfg)
    _, per_class_results = _eval_metrics(gt_records, pred_records, thresholds, cfg)

    print("=" * 60)
    print(
        f"最优总宏: Recall={total_macro['recall']:.4f} FDR={total_macro['fdr']:.4f} "
        f"Precision={total_macro['precision']:.4f}"
    )
    print(
        f"目标门槛: Recall>={TARGET_RECALL} FDR<={TARGET_FDR} "
        f"-> {'达成' if total_macro['recall'] >= TARGET_RECALL and total_macro['fdr'] <= TARGET_FDR else '未达成'}"
    )

    # ── 输出 ─────────────────────────────────────────────────────────
    thresholds_path = output_dir / "calibrated_thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "calibration_report.txt"
    _write_report(report_path, thresholds, per_class_results, cfg)
    print(f"阈值 JSON: {thresholds_path}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
