# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练指标可视化函数。

重新导出 ``rfdetr.visualize.training`` 中已有的公共 API， 并补充库中未覆盖的逐类 AP 热力图和学习率曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# 配置 matplotlib 中文字体支持（优先使用 Noto Sans CJK SC）
try:
    import matplotlib
    from matplotlib.font_manager import fontManager

    _CJK_FONT_CANDIDATES = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "AR PL UMing CN",
    ]
    _available_fonts = {f.name for f in fontManager.ttflist}
    for _font_name in _CJK_FONT_CANDIDATES:
        if _font_name in _available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
except ImportError:
    pass

# 直接复用 RF-DETR 库中已写好的完善可视化函数
from rfdetr.visualize.training import plot_loss_metrics as _plot_loss_metrics
from rfdetr.visualize.training import plot_map_metrics as _plot_map_metrics
from rfdetr.visualize.training import plot_metrics as _plot_metrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_loss_metrics(metrics_csv: str, output_path: str | None = None, loss_log_scale: bool = False) -> "Figure":
    """绘制训练损失曲线（包装 ``rfdetr.visualize.training.plot_loss_metrics``）。

    Args:
        metrics_csv: PTL CSVLogger 输出的 metrics.csv 文件路径。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。
        loss_log_scale: 是否对 loss 轴使用对数刻度。

    Returns:
        matplotlib Figure 对象。
    """
    return _plot_loss_metrics(metrics_csv, output_path=output_path, loss_log_scale=loss_log_scale)


def plot_map_metrics(metrics_csv: str, output_path: str | None = None) -> "Figure":
    """绘制 mAP 指标曲线（包装 ``rfdetr.visualize.training.plot_map_metrics``）。

    Args:
        metrics_csv: PTL CSVLogger 输出的 metrics.csv 文件路径。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。

    Returns:
        matplotlib Figure 对象。
    """
    return _plot_map_metrics(metrics_csv, output_path=output_path)


def plot_metrics(metrics_csv: str, output_path: str | None = None, loss_log_scale: bool = False) -> "Figure":
    """绘制训练综合指标总图（包装 ``rfdetr.visualize.training.plot_metrics``）。

    包含 Loss、AP@0.50、AP@0.50:0.95、AP@0.75、AR、F1/Precision/Recall 全部子图。

    Args:
        metrics_csv: PTL CSVLogger 输出的 metrics.csv 文件路径。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。
        loss_log_scale: 是否对 loss 轴使用对数刻度。

    Returns:
        matplotlib Figure 对象。
    """
    return _plot_metrics(metrics_csv, output_path=output_path, loss_log_scale=loss_log_scale)


def plot_per_class_ap(
    metrics_csv: str,
    output_path: str | None = None,
    class_names: dict[int, str] | None = None,
) -> "Figure":
    """绘制逐类 AP 随时间变化的热力图。

    从 metrics.csv 中提取 ``val/AP/{类名}`` 列，绘制 (类别 × epoch) 热力图，
    直观展示各类别的学习进度差异。

    Args:
        metrics_csv: PTL CSVLogger 输出的 metrics.csv 文件路径。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。
        class_names: 类别 ID 到名称的映射字典，若为 None 则自动从列名推断。

    Returns:
        matplotlib Figure 对象。

    Raises:
        ImportError: 当 pandas 或 matplotlib 未安装时。
        FileNotFoundError: 当 metrics_csv 不存在时。
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas 是绘制热力图必需的依赖。请通过 `pip install pandas` 安装。") from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib 是绘制热力图必需的依赖。请通过 `pip install matplotlib` 安装。") from exc

    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv 未找到: {csv_path}")

    df = pd.read_csv(csv_path)
    epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()

    # 提取 val/AP/ 开头的列
    ap_cols = [c for c in epoch_df.columns if c.startswith("val/AP/")]
    if not ap_cols:
        raise ValueError("metrics.csv 中未找到 val/AP/* 列，无法绘制逐类 AP 热力图。")

    # 提取类别名称和 AP 值矩阵
    ap_names = [c.removeprefix("val/AP/") for c in ap_cols]
    ap_matrix = epoch_df[ap_cols].values.T  # shape: (num_classes, num_epochs)

    # 如果提供了 class_names 映射，按 class_id 排序
    if class_names is not None:
        # class_names 是 {0: "HM", 1: "LQS", ...}，按顺序取名称
        ordered_names = [class_names[i] for i in range(len(class_names)) if class_names[i] in ap_names]
        # 按名称索引重排矩阵
        name_to_idx = {name: i for i, name in enumerate(ap_names)}
        row_order = [name_to_idx[name] for name in ordered_names if name in name_to_idx]
        ap_matrix = ap_matrix[row_order]
        ap_names = ordered_names
        if not ap_names:
            ap_names = [c.removeprefix("val/AP/") for c in ap_cols]

    epochs = epoch_df["epoch"].values

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, max(6, len(ap_names) * 0.3)))
    im = ax.imshow(ap_matrix, aspect="auto", cmap="YlOrRd", origin="upper")

    ax.set_xticks(np.linspace(0, len(epochs) - 1, min(10, len(epochs)), dtype=int))
    ax.set_xticklabels([str(int(epochs[i])) for i in np.linspace(0, len(epochs) - 1, min(10, len(epochs)), dtype=int)])
    ax.set_yticks(range(len(ap_names)))
    ax.set_yticklabels(ap_names, fontsize=8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("类别", fontsize=11)
    ax.set_title("逐类 AP 热力图 (val/AP)", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("AP", fontsize=10)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lr_schedule(
    metrics_csv: str,
    output_path: str | None = None,
) -> "Figure":
    """绘制学习率随 epoch 的变化曲线。

    从 metrics.csv 中提取 ``train/lr``、``train/lr_min``、``train/lr_max`` 列，
    展示学习率调度策略的执行过程。

    Args:
        metrics_csv: PTL CSVLogger 输出的 metrics.csv 文件路径。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。

    Returns:
        matplotlib Figure 对象。

    Raises:
        ImportError: 当 pandas 或 matplotlib 未安装时。
        FileNotFoundError: 当 metrics_csv 不存在时。
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas 是绘制学习率曲线必需的依赖。请通过 `pip install pandas` 安装。") from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib 是绘制学习率曲线必需的依赖。请通过 `pip install matplotlib` 安装。") from exc

    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv 未找到: {csv_path}")

    df = pd.read_csv(csv_path)

    lr_cols = ["train/lr", "train/lr_min", "train/lr_max"]
    available_cols = [c for c in lr_cols if c in df.columns]
    if not available_cols:
        raise ValueError("metrics.csv 中未找到 train/lr* 列，无法绘制学习率曲线。")

    # 按 epoch 聚合（非 val 行才包含 lr 数据）
    lr_df = df[["epoch"] + available_cols].dropna(subset=available_cols)
    if lr_df.empty:
        raise ValueError("metrics.csv 中 train/lr* 列全为 NaN，无法绘制学习率曲线。")

    epoch_lr = lr_df.groupby("epoch").mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"train/lr": "C0", "train/lr_min": "C1", "train/lr_max": "C2"}
    labels = {
        "train/lr": "基础学习率 (lr)",
        "train/lr_min": "最小学习率 (lr_min)",
        "train/lr_max": "最大学习率 (lr_max)",
    }

    for col in available_cols:
        ax.plot(
            epoch_lr["epoch"],
            epoch_lr[col],
            color=colors.get(col, "C0"),
            linewidth=1.7,
            label=labels.get(col, col),
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("学习率", fontsize=11)
    ax.set_title("学习率调度曲线", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
