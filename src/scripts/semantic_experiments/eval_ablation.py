# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""批量评估 8 个语义头消融实验并生成对比表（含硬约束 PASS/FAIL 判定）。

复用 ``src/scripts/test.py`` 的推理与比赛评分管线（SHWX：ship/aircraft IoU=0.50、
vehicle=0.35、conf=0.25），对每个实验的输出 checkpoint 做一轮完整评估，收集
overall/ship/aircraft/vehicle 的 P/R/F1、HM/LQS/QHS/MS 逐类 P/R、FP_ship、FSC
recall，输出 Markdown 对比表 + CSV。

**硬约束**（相对 P1 锚点基线，见方案文档 §5.4）：
1. aircraft F1 ≥ P1（飞机类不可回退）
2. FSC recall ≥ P1（发射车召回不可回退）
3. FP_ship ≤ P1 + 10（防舰船 FP 重演）
4. HM/LQS precision > P1（细粒度精度提升——语义头的核心卖点）

用法：
    python src/scripts/semantic_experiments/eval_ablation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ── 路径设置（与 test.py 一致：src 与项目根加入 sys.path）──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
for _p in (str(SRC_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from rfdetr import RFDETRMedium  # noqa: E402
from rfdetr.sscl.semantic_head import attach_from_checkpoint  # noqa: E402
from scripts import eval_lib as _lib  # noqa: E402  (复用统一评估库：推理管线与 SHWX 配置)
from val.competition_metrics import EvalConfig, evaluate_competition_metrics, load_yolo_labels  # noqa: E402

# ── 8 个实验的输出目录后缀（与 ablation_configs.ABLATIONS 一致）──
RUN_SUFFIXES = {
    "p1": "p1-anchor",
    "e1a": "e1a-full",
    "e2b": "e2b-maskoff",
    "e1c": "e1c-alpha0",
    "e3b": "e3b-alphafix",
    "e3c": "e3c-novelalpha",
    "e4b": "e4b-omega1",
    "e4c": "e4c-nosscl",
}

# 训练输出前缀（与 stage2_train.py 的 OUTPUT_PREFIX 保持一致）
OUTPUT_PREFIX = "output/0808-SHWX-SemHead/0808-SHWX-SemHead-"

# 锚点 p1 复用已存在的 0807-SHWX-SSCL-Proj-原型+实例正样本 checkpoint：
# 该实验配置与 p1 完全一致（6 epochs、lr=1e-5、conservative、原型+投影头+实例
# 正样本、λ=0.02、start=0、无蒸馏），无需重跑。key = 实验名 -> checkpoint 路径。
CHECKPOINT_OVERRIDES: dict[str, Path] = {
    "p1": PROJECT_ROOT / "output/0807-SHWX-SSCL-Proj-原型+实例正样本/checkpoint_best_total.pth",
}
EXPERIMENT_DESCS = {
    "p1": "锚点(无语义头)",
    "e1a": "完整(M+αS)",
    "e2b": "仅αS(M=1)",
    "e1c": "仅M(α=0)",
    "e3b": "α冻结",
    "e3c": "novelα大",
    "e4b": "ω=1",
    "e4c": "无SSCL",
}

OUTPUT_DIR = PROJECT_ROOT / "output/0808-SHWX-SemHead-compare"

# SHWX 数据集配置与推理参数（eval_lib，与 test 模板一致）
_DS = _lib.build_dataset_cfg("shwx")
_INF = _lib.InferenceCfg()
CONF_THRESHOLD = _INF.conf_threshold
BATCH_SIZE = _INF.batch_size
NUM_WORKERS = _INF.num_workers
DEVICE = _INF.device
# novel 类（HM/LQS/QHS/MS）与 FSC
NOVEL_CLASSES = [0, 1, 2, 3]
FSC_CLASS = 24
FP_SHIP_DELTA = 10  # 舰船 FP 允许超出的最大数量

# SHWX 数据集配置展开
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


def evaluate_run(suffix: str) -> dict[str, float]:
    """对单个实验的 checkpoint 做一轮完整评估，返回指标字典。

    Args:
        suffix: 实验输出目录后缀（含 p1 等，可能命中 CHECKPOINT_OVERRIDES 复用历史 checkpoint）。
    """
    # 锚点等复用历史 checkpoint 时直接取覆盖路径
    checkpoint = CHECKPOINT_OVERRIDES.get(suffix)
    if checkpoint is None:
        run_dir = PROJECT_ROOT / f"{OUTPUT_PREFIX}{suffix}"
        checkpoint = run_dir / "checkpoint_best_total.pth"
        if not checkpoint.exists():
            checkpoint = run_dir / "checkpoint_best_regular.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"实验输出目录缺少 checkpoint: {run_dir}")

    image_paths = _lib.read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = _lib.build_image_size_map(image_paths)
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)

    model = RFDETRMedium.from_checkpoint(str(checkpoint))
    # [SemHead] 重建语义残差模块（checkpoint 含 semantic_residual.* 键时生效）
    _sd = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    attach_from_checkpoint(model.model.model, _sd.get("model", _sd))
    del _sd

    pred_records, _, _, _ = _lib.predict_batched_to_records(
        model, image_paths, DEVICE, conf_threshold=CONF_THRESHOLD, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    del model
    _lib.release_cuda_cache(DEVICE)

    # 大类评估
    config = EvalConfig(
        class_to_group=CLASS_TO_GROUP,
        group_iou_thresholds=GROUP_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    group_results = evaluate_competition_metrics(gt_records, pred_records, config)
    # 逐类评估
    per_class_config = EvalConfig(
        class_to_group=PER_CLASS_TO_GROUP,
        group_iou_thresholds=PER_CLASS_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    per_class_results = evaluate_competition_metrics(gt_records, pred_records, per_class_config)

    metrics: dict[str, float] = {}
    # 大类指标
    all_r = group_results["all"]
    metrics["overall_p"], metrics["overall_r"], metrics["overall_f1"] = _precision_recall_f1(all_r)
    for group in ("ship", "aircraft", "vehicle"):
        r = group_results["groups"][group]
        metrics[f"{group}_p"], metrics[f"{group}_r"], metrics[f"{group}_f1"] = _precision_recall_f1(r)
    metrics["fp_ship"] = float(group_results["groups"]["ship"].fp)
    # novel 类逐类指标
    for c in NOVEL_CLASSES:
        r = per_class_results["groups"][CLASS_NAMES[c]]
        metrics[f"p_{CLASS_NAMES[c]}"], metrics[f"r_{CLASS_NAMES[c]}"], _ = _precision_recall_f1(r)
    # FSC recall（车辆类）
    fsc_r = per_class_results["groups"][CLASS_NAMES[FSC_CLASS]]
    _, metrics["fsc_recall"], _ = _precision_recall_f1(fsc_r)
    return metrics


def main() -> None:
    """评估全部实验并输出对比表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"评估目录: {OUTPUT_DIR}")
    print(f"测试集: {TEST_IMAGE_DIR}（{len(_lib.read_test_image_paths(TEST_IMAGE_DIR))} 张）")

    results: dict[str, dict[str, float]] = {}
    for name, suffix in RUN_SUFFIXES.items():
        print(f"\n=== 评估 {name} ({EXPERIMENT_DESCS[name]}) ===")
        try:
            results[name] = evaluate_run(suffix)
        except Exception as exc:  # 单个实验失败不阻塞整体
            print(f"[警告] {name} 评估失败: {exc}")
            results[name] = {}

    anchor = results.get("p1", {})

    # ── 生成 Markdown 对比表 ──
    rows = [
        ("overall_P", "总体 Precision"),
        ("overall_R", "总体 Recall"),
        ("overall_F1", "总体 F1"),
        ("ship_P", "舰船 Precision"),
        ("ship_R", "舰船 Recall"),
        ("ship_F1", "舰船 F1"),
        ("aircraft_F1", "飞机 F1"),
        ("vehicle_F1", "车辆 F1"),
        ("fp_ship", "舰船 FP 数"),
        ("HM_P", "HM Precision"),
        ("LQS_P", "LQS Precision"),
        ("QHS_P", "QHS Precision"),
        ("MS_P", "MS Precision"),
        ("HM_R", "HM Recall"),
        ("LQS_R", "LQS Recall"),
        ("FSC_R", "FSC Recall"),
    ]
    metric_keys = [key for key, _ in rows]

    lines: list[str] = []
    lines.append("# 语义分类头消融实验对比\n")
    lines.append("| 指标 | " + " | ".join(EXPERIMENT_DESCS[k] for k in RUN_SUFFIXES) + " |")
    lines.append("|---|" + "---|" * len(RUN_SUFFIXES))
    for key, label in rows:
        cells = []
        for name in RUN_SUFFIXES:
            val = results[name].get(key)
            cells.append(f"{val:.4f}" if val is not None else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # ── 硬约束判定 ──
    lines.append("\n## 硬约束判定（相对 P1 锚点）\n")
    if anchor:
        constraints = [
            ("aircraft_F1", "飞机 F1 ≥ P1", lambda v: v >= anchor.get("aircraft_f1", 1.0)),
            ("FSC_R", "FSC recall ≥ P1", lambda v: v >= anchor.get("fsc_recall", 1.0)),
            ("fp_ship", "FP_ship ≤ P1 + 10", lambda v: v <= anchor.get("fp_ship", 0) + FP_SHIP_DELTA),
            ("HM_P", "HM precision > P1", lambda v: v > anchor.get("p_HM", 1.0)),
            ("LQS_P", "LQS precision > P1", lambda v: v > anchor.get("p_LQS", 1.0)),
        ]
        lines.append("| 实验 | " + " | ".join(f"{label}" for _, label, _ in constraints) + " |")
        lines.append("|---|" + "---|" * len(constraints))
        for name in RUN_SUFFIXES:
            r = results.get(name, {})
            flags = []
            for key, _, check in constraints:
                val = r.get(key)
                flags.append("✅" if val is not None and check(val) else "❌" if val is not None else "—")
            lines.append(f"| {EXPERIMENT_DESCS[name]} | " + " | ".join(flags) + " |")
    else:
        lines.append("\n（P1 锚点评估失败，无法判定硬约束）")

    report_text = "\n".join(lines)
    (OUTPUT_DIR / "comparison.md").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)

    # ── CSV 输出 ──
    with open(OUTPUT_DIR / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + metric_keys)
        for name in RUN_SUFFIXES:
            r = results.get(name, {})
            writer.writerow([name] + [r.get(k, "") for k in metric_keys])
    print(f"\n对比表已保存: {OUTPUT_DIR / 'comparison.md'}")
    print(f"CSV 已保存: {OUTPUT_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
