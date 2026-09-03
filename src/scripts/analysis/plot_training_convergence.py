# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""生成第 4.2 节训练过程与收敛分析所需的独立图片。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE_II_CSV = PROJECT_ROOT / "output/0825baseline/metrics.csv"
STAGE_III_CSV = PROJECT_ROOT / "output/0825发射车虚警抑制/0826原型前向加难负样本抑制/metrics.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs/assets/training_convergence"

_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "AR PL UMing CN",
)

SERIES = (
    ("train/loss_ce", r"$\mathcal{L}_{\mathrm{cls}}$", "loss_classification.png", "#e64b35"),
    ("train/loss_bbox", r"$\mathcal{L}_{\mathrm{bbox}}$", "loss_bbox.png", "#00a087"),
    ("train/loss_giou", r"$\mathcal{L}_{\mathrm{GIoU}}$", "loss_giou.png", "#3c5488"),
    ("train/loss_sscl", r"$\mathcal{L}_{\mathrm{SSCL}}$", "loss_sscl.png", "#7e6148"),
    ("train/loss_sscl_hard_neg", r"$\mathcal{L}_{\mathrm{HN}}$", "loss_hard_negative.png", "#8491b4"),
)


def configure_chinese_font() -> None:
    """选择可用的中文字体，并关闭负号乱码。"""
    from matplotlib.font_manager import fontManager

    available = {font.name for font in fontManager.ttflist}
    selected = next((name for name in _FONT_CANDIDATES if name in available), None)
    matplotlib.rcParams["font.sans-serif"] = [selected or "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成训练收敛与损失分解图片")
    parser.add_argument("--stage-ii-csv", type=Path, default=STAGE_II_CSV)
    parser.add_argument("--stage-iii-csv", type=Path, default=STAGE_III_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _mean_by_epoch(rows: Iterable[dict[str, str]], column: str) -> list[tuple[int, float]]:
    """提取指定列并按 Epoch 求均值。"""
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        raw_epoch = row.get("epoch", "")
        raw_value = row.get(column, "")
        if not raw_epoch or not raw_value:
            continue
        try:
            epoch = int(float(raw_epoch))
            value = float(raw_value)
        except ValueError:
            continue
        if value == value:
            values[epoch].append(value)
    if not values:
        raise ValueError(f"CSV 中没有可用于 {column} 的有效数据。")
    return [(epoch, sum(items) / len(items)) for epoch, items in sorted(values.items())]


def read_metric(csv_path: Path, column: str) -> list[tuple[int, float]]:
    """读取 CSV 中的训练指标，并按 Epoch 聚合。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"指标文件不存在：{csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{csv_path} 中未找到字段 {column}。")
        return _mean_by_epoch(reader, column)


def _add_events(ax: plt.Axes, epochs: list[int], values: list[float]) -> None:
    """在曲线上添加原型预热和难负样本抑制事件。"""
    left, right = min(epochs), max(epochs)
    if left <= 1:
        ax.axvspan(-0.5, 1.5, color="#dceefb", alpha=0.7, zorder=0)
        y_top = max(values)
        y_bottom = min(values)
        y_position = y_top - (y_top - y_bottom) * 0.08 if y_top != y_bottom else y_top
        ax.text(0.5, y_position, "原型预热期", ha="center", va="top", fontsize=10, color="#24527a")
    if left <= 2 <= right:
        ax.axvline(2, color="#2f855a", linestyle="--", linewidth=1.2)
        ax.text(
            2.15,
            0.08,
            "难负样本抑制启用",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            fontsize=9,
            color="#276749",
        )


def plot_series(
    series: list[tuple[int, float]],
    ylabel: str,
    color: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
    x_tick_step: int | None = None,
    show_events: bool = True,
) -> None:
    """绘制并保存一条独立的中文指标曲线。"""
    epochs = [epoch for epoch, _ in series]
    values = [value for _, value in series]
    figure, ax = plt.subplots(figsize=(8.5, 4.8))
    figure.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.plot(epochs, values, color=color, marker="o", markersize=4.8, linewidth=2.5, zorder=3)
    ax.fill_between(epochs, values, min(values), color=color, alpha=0.12, zorder=1)
    ax.scatter(epochs[-1], values[-1], color=color, edgecolor="#ffffff", linewidth=1.2, s=65, zorder=4)
    if show_events:
        _add_events(ax, epochs, values)
    ax.set_xlabel("训练轮次（Epoch）", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    if x_tick_step is None:
        tick_epochs = epochs
    else:
        tick_start = (epochs[0] // x_tick_step) * x_tick_step
        tick_epochs = list(range(tick_start, epochs[-1] + 1, x_tick_step))
        if epochs[-1] not in tick_epochs:
            tick_epochs.append(epochs[-1])
    ax.set_xticks(tick_epochs)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7a7a7a")
    ax.tick_params(axis="both", labelsize=10, color="#555555")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _metric_y_limits(series: list[tuple[int, float]]) -> tuple[float, float]:
    """根据指标实际范围设置纵轴，避免从零开始导致变化不明显。"""
    values = [value for _, value in series]
    low, high = min(values), max(values)
    span = high - low
    padding = max(span * 0.25, 0.002)
    return max(0.0, low - padding), min(1.0, high + padding)


def generate_plots(stage_ii_csv: Path, stage_iii_csv: Path, output_dir: Path) -> list[Path]:
    """使用阶段 II 和阶段 III 数据生成独立收敛与损失曲线。"""
    configure_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    metric_specs = (
        ("val/F1", "平均 F1（Avg F1）", "avg_f1", "#e64b35"),
        ("val/recall", "召回率（Recall）", "recall", "#00a087"),
        ("val/mAP_50", "mAP@0.50", "map50", "#3c5488"),
        ("val/mAP_50_95", "mAP@0.50:0.95", "map50_95", "#f39b7f"),
    )
    for stage_name, csv_path, x_tick_step in (("stage_ii", stage_ii_csv, 20), ("stage_iii", stage_iii_csv, 2)):
        for column, ylabel, suffix, color in metric_specs:
            series = read_metric(csv_path, column)
            path = output_dir / f"{stage_name}_{suffix}.png"
            plot_series(
                series,
                ylabel,
                color,
                path,
                _metric_y_limits(series),
                x_tick_step,
                show_events=stage_name == "stage_iii",
            )
            generated.append(path)

    for column, ylabel, filename, color in SERIES:
        series = read_metric(stage_iii_csv, column)
        path = output_dir / filename
        plot_series(series, ylabel, color, path)
        generated.append(path)
    return generated


def main() -> None:
    """执行图片生成流程。"""
    args = parse_args()
    generated = generate_plots(args.stage_ii_csv, args.stage_iii_csv, args.output_dir)
    print(f"已生成 {len(generated)} 张独立图片：{args.output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
