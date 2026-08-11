# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本实验三向对照评估（基线 / k=3 / k=5）。

复用 ``eval_ablation.py`` 的评估管线（test.py 推理 + 比赛评分）：
- baseline：复用 0807-SHWX-SSCL-Proj-原型+实例正样本 checkpoint（不重跑）；
- hardneg_k3 / hardneg_k5：本次难例实验的两个臂。

输出整体/舰船/飞机/车辆 P/R/F1、fp_ship、HM/LQS/QHS/MS 逐类 P/R、FSC
recall，Markdown + CSV 到 output/0810-SHWX-SSCL-HardNeg-compare/。

用法：
    python src/scripts/ret-sscl/eval_hardneg.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ── 路径设置（与 eval_ablation.py 一致）──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
for _p in (str(SRC_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rfdetr import RFDETRMedium  # noqa: E402
from scripts import eval_lib as _lib  # noqa: E402  (复用统一评估库：推理管线与 SHWX 配置)
from val.competition_metrics import EvalConfig, evaluate_competition_metrics, load_yolo_labels  # noqa: E402

# ── 三臂：基线（复用 0807 checkpoint）+ 难例 k=3 / k=5 ──
RUNS = {
    "baseline": {
        "desc": "基线（0807 原型+实例正样本，复用）",
        "output_dir": PROJECT_ROOT / "output/0807-SHWX-SSCL-Proj-原型+实例正样本",
        "checkpoint_override": (PROJECT_ROOT / "output/0807-SHWX-SSCL-Proj-原型+实例正样本/checkpoint_best_total.pth"),
    },
    "hardneg_k3": {
        "desc": "难例 k=3",
        "output_dir": PROJECT_ROOT / "output/0810-SHWX-SSCL-Proj-HardNeg-k3",
        "checkpoint_override": None,
    },
    "hardneg_k5": {
        "desc": "难例 k=5",
        "output_dir": PROJECT_ROOT / "output/0810-SHWX-SSCL-Proj-HardNeg-k5",
        "checkpoint_override": None,
    },
}

OUTPUT_DIR = PROJECT_ROOT / "output/0810-SHWX-SSCL-HardNeg-compare"

# SHWX 数据集配置与推理参数（eval_lib，与 test 模板一致）
_DS = _lib.build_dataset_cfg("shwx")
_INF = _lib.InferenceCfg()
CONF_THRESHOLD = _INF.conf_threshold
BATCH_SIZE = _INF.batch_size
NUM_WORKERS = _INF.num_workers
DEVICE = _INF.device
NOVEL_CLASSES = [0, 1, 2, 3]
FSC_CLASS = 24

TEST_IMAGE_DIR = _DS.test_image_dir
LABEL_DIR = _DS.label_dir
CLASS_NAMES: dict[int, str] = _DS.class_names
CLASS_TO_GROUP = _DS.class_to_group
GROUP_IOU_THRESHOLDS = _DS.group_iou_thresholds
PER_CLASS_TO_GROUP = _DS.per_class_to_group
PER_CLASS_IOU_THRESHOLDS = _DS.per_class_iou_thresholds


def _precision_recall_f1(result) -> tuple[float, float, float]:
    """由 EvalResult 计算 precision/recall/F1。"""
    tp, fp, fn = float(result.tp), float(result.fp), float(result.fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_run(name: str) -> dict[str, float]:
    """对单个臂做一轮完整评估，返回指标字典。

    Args:
        name: 臂名（baseline / hardneg_k3 / hardneg_k5）。
    """
    spec = RUNS[name]
    checkpoint = spec["checkpoint_override"]
    if checkpoint is None:
        checkpoint = spec["output_dir"] / "checkpoint_best_total.pth"
        if not checkpoint.exists():
            checkpoint = spec["output_dir"] / "checkpoint_best_regular.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"难例实验缺少 checkpoint: {spec['output_dir']}")

    image_paths = _lib.read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = _lib.build_image_size_map(image_paths)
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)

    model = RFDETRMedium.from_checkpoint(str(checkpoint))
    pred_records, _, _, _ = _lib.predict_batched_to_records(
        model, image_paths, DEVICE, conf_threshold=CONF_THRESHOLD, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    del model
    _lib.release_cuda_cache(DEVICE)

    config = EvalConfig(
        class_to_group=CLASS_TO_GROUP,
        group_iou_thresholds=GROUP_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    group_results = evaluate_competition_metrics(gt_records, pred_records, config)
    per_class_config = EvalConfig(
        class_to_group=PER_CLASS_TO_GROUP,
        group_iou_thresholds=PER_CLASS_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    per_class_results = evaluate_competition_metrics(gt_records, pred_records, per_class_config)

    metrics: dict[str, float] = {}
    all_r = group_results["all"]
    metrics["overall_p"], metrics["overall_r"], metrics["overall_f1"] = _precision_recall_f1(all_r)
    for group in ("ship", "aircraft", "vehicle"):
        r = group_results["groups"][group]
        metrics[f"{group}_p"], metrics[f"{group}_r"], metrics[f"{group}_f1"] = _precision_recall_f1(r)
    metrics["fp_ship"] = float(group_results["groups"]["ship"].fp)
    for c in NOVEL_CLASSES:
        r = per_class_results["groups"][CLASS_NAMES[c]]
        metrics[f"p_{CLASS_NAMES[c]}"], metrics[f"r_{CLASS_NAMES[c]}"], _ = _precision_recall_f1(r)
    fsc_r = per_class_results["groups"][CLASS_NAMES[FSC_CLASS]]
    _, metrics["fsc_recall"], _ = _precision_recall_f1(fsc_r)
    return metrics


def main() -> None:
    """评估三臂并输出 Markdown + CSV 对照表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"评估目录: {OUTPUT_DIR}")
    print(f"测试集: {TEST_IMAGE_DIR}（{len(_lib.read_test_image_paths(TEST_IMAGE_DIR))} 张）")

    results: dict[str, dict[str, float]] = {}
    for name in RUNS:
        print(f"\n=== 评估 {name}（{RUNS[name]['desc']}）===")
        try:
            results[name] = evaluate_run(name)
        except Exception as exc:  # 单个臂失败不阻塞整体
            print(f"[警告] {name} 评估失败: {exc}")
            results[name] = {}

    rows = [
        ("overall_p", "总体 Precision"),
        ("overall_r", "总体 Recall"),
        ("overall_f1", "总体 F1"),
        ("ship_p", "舰船 Precision"),
        ("ship_r", "舰船 Recall"),
        ("ship_f1", "舰船 F1"),
        ("aircraft_f1", "飞机 F1"),
        ("vehicle_f1", "车辆 F1"),
        ("fp_ship", "舰船 FP 数"),
        ("p_HM", "HM Precision"),
        ("p_LQS", "LQS Precision"),
        ("p_QHS", "QHS Precision"),
        ("p_MS", "MS Precision"),
        ("r_HM", "HM Recall"),
        ("r_LQS", "LQS Recall"),
        ("fsc_recall", "FSC Recall"),
    ]
    metric_keys = [key for key, _ in rows]
    descs = {k: RUNS[k]["desc"] for k in RUNS}

    lines: list[str] = []
    lines.append("# 难例负样本实验对照（基线 vs k=3 vs k=5）\n")
    lines.append("| 指标 | " + " | ".join(descs[k] for k in RUNS) + " |")
    lines.append("|---|" + "---|" * len(RUNS))
    for key, label in rows:
        cells = []
        for name in RUNS:
            val = results[name].get(key)
            cells.append(f"{val:.4f}" if val is not None else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # ── 相对基线判定 ──
    baseline = results.get("baseline", {})
    lines.append("\n## 相对基线判定（难例实验的预期收益）\n")
    if baseline:
        checks = [
            ("fp_ship", "舰船 FP ≤ 基线", lambda v: v <= baseline.get("fp_ship", 0) + 5),
            (
                "overall_r",
                "总体 Recall ≥ 基线（不伤召回）",
                lambda v: v >= baseline.get("overall_r", 0) - 0.01,
            ),
            ("ship_f1", "舰船 F1 > 基线", lambda v: v > baseline.get("ship_f1", 0)),
        ]
        lines.append("| 实验 | " + " | ".join(label for _, label, _ in checks) + " |")
        lines.append("|---|" + "---|" * len(checks))
        for name in RUNS:
            r = results.get(name, {})
            flags = []
            for key, _, check in checks:
                val = r.get(key)
                flags.append("✅" if val is not None and check(val) else "❌" if val is not None else "—")
            lines.append(f"| {descs[name]} | " + " | ".join(flags) + " |")
    else:
        lines.append("\n（基线评估失败，无法判定）")

    report_text = "\n".join(lines)
    (OUTPUT_DIR / "comparison.md").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)

    with open(OUTPUT_DIR / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + metric_keys)
        for name in RUNS:
            r = results.get(name, {})
            writer.writerow([name] + [r.get(k, "") for k in metric_keys])
    print(f"\n对比表已保存: {OUTPUT_DIR / 'comparison.md'}")
    print(f"CSV 已保存: {OUTPUT_DIR / 'comparison.csv'}")
    print("\nFP 成因分解请另跑（每个 checkpoint）：")
    for name in RUNS:
        if name == "baseline":
            print(f"  uv run python src/scripts/analyze_fp_decomposition.py {RUNS[name]['checkpoint_override']}")
        else:
            print(
                "  uv run python src/scripts/analyze_fp_decomposition.py "
                f"{RUNS[name]['output_dir'] / 'checkpoint_best_total.pth'}"
            )


if __name__ == "__main__":
    main()
