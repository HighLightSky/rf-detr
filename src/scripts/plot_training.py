# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练指标可视化脚本。

读取训练输出的 ``metrics.csv``，生成训练过程中的主要指标变化曲线图。
"""

import sys
from pathlib import Path

# ── 路径配置 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
METRICS_CSV = PROJECT_ROOT / "output" / "0724-shwx-rfdetr_medium" / "metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "0724-shwx-rfdetr_medium" / "plots"

# 25 个细粒度类别名称（按训练时的类别顺序）
CLASS_NAMES: dict[int, str] = {
    0: "HM",
    1: "LQS",
    2: "QHS",
    3: "MS",
    4: "A1_SU-35",
    5: "A2_C-130",
    6: "A3_C-17",
    7: "A4_C-5",
    8: "A5_F-16",
    9: "A6_TU-160",
    10: "A7_E-3",
    11: "A8_B-52",
    12: "A9_P-3C",
    13: "A10_B-1B",
    14: "A11_E-8",
    15: "A12_TU-22",
    16: "A13_F-15",
    17: "A14_KC-135",
    18: "A15_F-22",
    19: "A16_FA-18",
    20: "A17_TU-95",
    21: "A18_KC-10",
    22: "A19_SU-34",
    23: "A20_SU-24",
    24: "FSC",
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.training import (  # noqa: E402
    plot_loss_metrics,
    plot_lr_schedule,
    plot_map_metrics,
    plot_metrics,
    plot_per_class_ap,
)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_str = str(METRICS_CSV)

    print("[i] 正在生成训练指标可视化图表...")
    print(f"    数据源: {METRICS_CSV}")
    print(f"    输出目录: {OUTPUT_DIR}")

    # 综合总图（Loss + AP + AR + F1/P/R）
    print("[i] 生成综合指标总图...")
    plot_metrics(metrics_str, str(OUTPUT_DIR / "training_summary.png"))

    # 单独 Loss 面板
    print("[i] 生成损失曲线图...")
    plot_loss_metrics(metrics_str, str(OUTPUT_DIR / "loss_curves.png"))

    # 单独 mAP 面板
    print("[i] 生成 mAP 曲线图...")
    plot_map_metrics(metrics_str, str(OUTPUT_DIR / "map_curves.png"))

    # 逐类 AP 热力图
    print("[i] 生成逐类 AP 热力图...")
    plot_per_class_ap(metrics_str, str(OUTPUT_DIR / "per_class_ap.png"), CLASS_NAMES)

    # 学习率调度曲线
    print("[i] 生成学习率曲线图...")
    plot_lr_schedule(metrics_str, str(OUTPUT_DIR / "lr_schedule.png"))

    print(f"\n[完成] 训练指标可视化图表已保存至: {OUTPUT_DIR}")
