# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练收敛分析脚本的 CSV 聚合与图片输出测试。"""

from pathlib import Path

import pytest

from scripts.analysis.plot_training_convergence import generate_plots, read_metric


def test_read_metric_aggregates_non_empty_rows_by_epoch(tmp_path: Path) -> None:
    """同一 Epoch 的重复日志行应聚合为均值，空值行应跳过。"""
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,step,val/F1\n0,1,0.8\n0,2,\n0,3,0.9\n1,4,0.7\n",
        encoding="utf-8",
    )

    assert read_metric(csv_path, "val/F1") == [(0, pytest.approx(0.85)), (1, pytest.approx(0.7))]


def test_generate_plots_writes_stage_curves_and_losses(tmp_path: Path) -> None:
    """生成流程应输出阶段 II、III 指标图和阶段 III 损失图。"""
    stage_ii_csv = tmp_path / "stage_ii.csv"
    stage_iii_csv = tmp_path / "stage_iii.csv"
    loss_columns = [
        "train/loss_ce",
        "train/loss_bbox",
        "train/loss_giou",
        "train/loss_sscl",
        "train/loss_sscl_hard_neg",
    ]
    csv_content = (
        "epoch,step,val/F1,val/recall,val/mAP_50,val/mAP_50_95,"
        + ",".join(loss_columns)
        + "\n"
        + "0,1,0.8,0.7,0.75,0.6,"
        + ",".join(["1"] * len(loss_columns))
        + "\n"
        + "8,2,0.84,0.74,0.79,0.65,"
        + ",".join(["0.5"] * len(loss_columns))
        + "\n"
    )
    stage_ii_csv.write_text(csv_content, encoding="utf-8")
    stage_iii_csv.write_text(csv_content, encoding="utf-8")

    generated = generate_plots(stage_ii_csv, stage_iii_csv, tmp_path / "plots")

    assert len(generated) == 13
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
