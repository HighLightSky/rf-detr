#!/usr/bin/env python3
"""生成 SSCL 语义加权排斥力的二维示意图。

该脚本使用固定随机种子生成轻微扰动的模拟特征点，并输出左右对比图：
标准监督对比学习（SCL）与语义加权监督对比学习（SSCL）。图中箭头由
``quiver`` 绘制，右图额外包含语义权重热图和局部放大圈，便于导入 PPT
后继续调整文字与标注。

用法：
    python src/scripts/analysis/visualize_sscl_mechanism.py
    python src/scripts/analysis/visualize_sscl_mechanism.py --out figures/sscl
    python src/scripts/analysis/visualize_sscl_mechanism.py --ppt --out figures/sscl_ppt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Ellipse


ANCHOR = np.array([0.0, 0.0])
GREEN = "#009E73"
RED = "#FF0000"
GRAY = "#7F8C8D"
LIGHT_GRAY = "#C8C8C8"
GRID = "#B8BEC3"
TEXT = "#263238"


def configure_style(font_name: str | None = None) -> None:
    """配置适合论文和 PPT 的 Matplotlib 全局样式。"""
    candidates = [
        font_name,
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    sans_fonts = [font for font in candidates if font]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans_fonts,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#5D666D",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def jitter(points: Sequence[Sequence[float]], rng: np.random.Generator, scale: float = 0.07) -> np.ndarray:
    """对手工设定的点加入很小的确定性扰动，使示意图更自然。"""
    values = np.asarray(points, dtype=float)
    return values + rng.normal(0.0, scale, size=values.shape)


def draw_quiver(
    ax: Axes,
    points: np.ndarray,
    color: str,
    width: float,
    *,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    """用 quiver 从锚点指向一组样本点。"""
    origins = np.repeat(ANCHOR[None, :], len(points), axis=0)
    vectors = points - origins
    ax.quiver(
        origins[:, 0],
        origins[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=width,
        headwidth=4.2,
        headlength=5.5,
        headaxislength=4.8,
        alpha=alpha,
        zorder=zorder,
        minlength=0.1,
    )


def draw_points(ax: Axes, positives: np.ndarray, hard_negatives: np.ndarray, easy_negatives: np.ndarray) -> None:
    """绘制正样本、难负样本、易负样本和锚点。"""
    ax.scatter(
        positives[:, 0],
        positives[:, 1],
        s=75,
        c=GREEN,
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        hard_negatives[:, 0],
        hard_negatives[:, 1],
        s=82,
        c=RED,
        marker="s",
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        easy_negatives[:, 0],
        easy_negatives[:, 1],
        s=78,
        c=LIGHT_GRAY,
        marker="s",
        edgecolors="#7A858C",
        linewidths=0.8,
        zorder=5,
    )
    ax.scatter(
        [ANCHOR[0]],
        [ANCHOR[1]],
        s=430,
        c="#F6C945",
        marker="*",
        edgecolors="#8A6500",
        linewidths=1.2,
        zorder=7,
    )


def draw_panel(
    ax: Axes,
    *,
    title: str,
    hard_negatives: np.ndarray,
    easy_negatives: np.ndarray,
    sscl: bool,
    show_force_annotations: bool = True,
    show_axis_labels: bool = True,
) -> None:
    """绘制一个 SCL 或 SSCL 特征空间面板。"""
    positives = jitter([[1.05, 1.05], [1.40, 1.25], [1.55, 0.90]], np.random.default_rng(11))
    ax.add_patch(Circle((0.0, 0.0), 3.0, fill=False, linestyle=(0, (4, 3)), linewidth=1.0, color=GRID, zorder=0))
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.25)
    ax.set_axisbelow(True)

    if show_force_annotations:
        draw_quiver(ax, positives, GREEN, 0.0075, zorder=2)
        if sscl:
            draw_quiver(ax, hard_negatives, RED, 0.021, zorder=2)
            draw_quiver(ax, easy_negatives, LIGHT_GRAY, 0.003, alpha=0.95, zorder=2)
            for point in easy_negatives:
                ax.plot(
                    [ANCHOR[0], point[0]],
                    [ANCHOR[1], point[1]],
                    linestyle=(0, (3, 3)),
                    color=LIGHT_GRAY,
                    linewidth=0.8,
                    alpha=0.85,
                    zorder=1,
                )
        else:
            all_negatives = np.concatenate([hard_negatives, easy_negatives], axis=0)
            draw_quiver(ax, all_negatives, GRAY, 0.0085, alpha=0.9, zorder=2)

    draw_points(ax, positives, hard_negatives, easy_negatives)
    ax.annotate("锚点样本", xy=ANCHOR, xytext=(-0.22, -0.72), ha="center", va="top", fontsize=9)
    ax.text(1.65, 1.40, "正样本", color=GREEN, fontsize=9, weight="bold", ha="center")
    ax.text(
        1.20 if not sscl else 2.22,
        -1.42 if not sscl else -2.12,
        "困难负样本\n(驱逐舰、两栖舰)",
        color=RED,
        fontsize=8.5,
        ha="center",
        va="top",
    )
    ax.text(-2.20, -0.78, "容易负样本\n(飞机、汽车)", color="#67737A", fontsize=8.5, ha="center", va="top")
    if sscl:
        if show_force_annotations:
            ax.text(0.92, -0.18, "强排斥力\n(w≈1.8)", color=RED, fontsize=8.5, weight="bold", ha="center")
            ax.text(-1.20, 0.40, "弱排斥力\n(w≈1.0)", color="#839096", fontsize=8.5, ha="center")
        ax.add_patch(
            Ellipse(
                (1.95, -1.18),
                width=2.00,
                height=1.65,
                fill=False,
                linestyle=(0, (3, 2)),
                linewidth=1.35,
                color=RED,
                alpha=0.85,
                zorder=1,
            )
        )
        if show_force_annotations:
            ax.text(2.77, -0.35, "局部放大区域", color=RED, fontsize=8, rotation=18, ha="center")
    elif not sscl and show_force_annotations:
        ax.text(0.72, -0.05, "均等排斥力\n(w=1.0)", color=GRAY, fontsize=8.5, ha="center")

    ax.set_title(title, pad=12)
    ax.set_xlim(-3.35, 3.35)
    ax.set_ylim(-3.05, 2.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Feature_1" if show_axis_labels else "")
    ax.set_ylabel("Feature_2" if show_axis_labels else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_weight_inset(ax: Axes, figure: plt.Figure) -> None:
    """在 SSCL 面板右上角绘制语义权重热图及映射箭头。"""
    inset = ax.inset_axes([0.72, 0.73, 0.25, 0.20])
    values = np.array([[1.00], [1.80], [1.00]])
    cmap = LinearSegmentedColormap.from_list("sscl_weight", ["#E2E5E7", "#FFB3B3", RED])
    inset.imshow(values, cmap=cmap, norm=Normalize(vmin=1.0, vmax=1.8), aspect="auto")
    inset.set_xticks([0], ["锚点类\n驱护舰"], fontsize=7)
    inset.set_yticks([0, 1, 2], ["驱逐舰", "两栖舰", "飞机"], fontsize=7)
    inset.tick_params(length=0, pad=2)
    for row, value in enumerate(values[:, 0]):
        inset.text(0, row, f"{value:.1f}", ha="center", va="center", fontsize=8, weight="bold")
    for spine in inset.spines.values():
        spine.set_color("#78838A")
        spine.set_linewidth(0.7)
    inset.set_title("语义权重矩阵", fontsize=8.5, pad=3, weight="bold")
    connection = ConnectionPatch(
        xyA=(0.12, 0.15),
        coordsA=inset.transAxes,
        xyB=(1.38, -0.60),
        coordsB=ax.transData,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        color=RED,
        alpha=0.8,
        zorder=8,
    )
    figure.add_artist(connection)
    ax.text(1.78, 0.12, "权重映射", fontsize=8, color=RED, rotation=-12, ha="center")


def add_figure_legend(figure: plt.Figure, show_force_annotations: bool = True) -> None:
    """添加跨面板的颜色、线型和箭头宽度图例。"""
    handles = [
        Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="#F6C945", markeredgecolor="#8A6500",
            markersize=13, label="锚点",
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=8, label="正样本"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, markersize=8, label="语义相似负样本"),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=LIGHT_GRAY, markeredgecolor="#7A858C",
            markersize=8, label="语义无关负样本",
        ),
    ]
    if show_force_annotations:
        handles.extend(
            [
                Line2D([0], [0], color=RED, linewidth=3.6, label="强排斥力"),
                Line2D([0], [0], color=LIGHT_GRAY, linewidth=1.0, linestyle=(0, (3, 3)), label="弱排斥力"),
            ]
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3 if show_force_annotations else 2,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8.5,
        handlelength=2.0,
        columnspacing=1.6,
    )


def create_figure(seed: int = 7, font_name: str | None = None, ppt: bool = False) -> plt.Figure:
    """创建完整的 SCL 与 SSCL 对比图。"""
    configure_style(font_name)
    rng = np.random.default_rng(seed)
    hard_scl = jitter([[0.95, -0.55], [1.28, -0.40], [0.78, -0.90]], rng)
    hard_sscl = jitter([[2.10, -1.35], [2.42, -0.98], [1.95, -1.62]], rng)
    easy = jitter([[-2.25, 0.08], [-1.88, -0.50]], rng)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), constrained_layout=False)
    title_scl = "(a) 标准监督对比学习（SCL）"
    title_sscl = "(b) 语义加权监督对比学习（SSCL）"
    draw_panel(
        axes[0],
        title=title_scl,
        hard_negatives=hard_scl,
        easy_negatives=easy,
        sscl=False,
        show_force_annotations=not ppt,
        show_axis_labels=not ppt,
    )
    draw_panel(
        axes[1],
        title=title_sscl,
        hard_negatives=hard_sscl,
        easy_negatives=easy,
        sscl=True,
        show_force_annotations=not ppt,
        show_axis_labels=not ppt,
    )
    if not ppt:
        draw_weight_inset(axes[1], figure)
    add_figure_legend(figure, show_force_annotations=not ppt)
    figure.subplots_adjust(left=0.055, right=0.975, top=0.90, bottom=0.17, wspace=0.12)
    return figure


def main() -> None:
    """解析命令行参数并保存论文与 PPT 所需的三种格式。"""
    parser = argparse.ArgumentParser(description="生成 SSCL 语义加权排斥力示意图")
    parser.add_argument("--out", type=Path, default=Path("figures/sscl_mechanism"), help="输出目录")
    parser.add_argument("--seed", type=int, default=7, help="模拟点的随机种子")
    parser.add_argument("--font", type=str, default=None, help="可选的中文字体名称")
    parser.add_argument(
        "--ppt",
        action="store_true",
        help="隐藏箭头、权重矩阵和排斥力文字，保留样本点与边界",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    figure = create_figure(seed=args.seed, font_name=args.font, ppt=args.ppt)
    stem = "sscl_points_for_ppt" if args.ppt else "sscl_mechanism"
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 320})):
        output_path = args.out / f"{stem}.{suffix}"
        figure.savefig(output_path, **options)
        print(f"已保存: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
