"""绘制检测模型精度、推理时间与参数量的二维气泡散点图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main() -> None:
    """生成并保存对比实验二维气泡散点图。"""
    models = [
        ("YOLOv8-m", 0.7825, 33.6, 25.9, "单阶段实时检测器"),
        ("YOLO11-m", 0.7852, 34.0, 20.0, "单阶段实时检测器"),
        ("YOLO26-m", 0.7262, 49.6, 20.4, "单阶段实时检测器"),
        ("Cascade R-CNN", 0.7582, 23.7, 68.9, "多阶段级联检测器"),
        ("RT-DETR-m", 0.8026, 59.3, 36.0, "DETR 同源架构"),
        ("RT-DETRv4-m", 0.8077, 64.4, 19.0, "DETR 同源架构"),
        ("D-FINE", 0.7810, 59.2, 19.8, "DETR 同源架构"),
        ("CTRP-DETR-m", 0.8359, 42.4, 44.7, "RF-DETR 进阶变体"),
        ("RF-DETR-m", 0.8622, 32.7, 33.7, "DETR 同源架构"),
        ("本方案", 0.8826, 40.0, 35.4, "本文方法"),
    ]

    # 将原表的推理速度换算为单张图像的推理时间。
    names = [item[0] for item in models]
    f1 = np.array([item[1] for item in models])
    latency = 1000.0 / np.array([item[2] for item in models])
    params = np.array([item[3] for item in models])
    groups = [item[4] for item in models]

    colors = {
        "单阶段实时检测器": "#F03B20",
        "多阶段级联检测器": "#7A1FA2",
        "DETR 同源架构": "#0072B2",
        "RF-DETR 进阶变体": "#00A676",
        "本文方法": "#F59E0B",
    }
    plt.rcParams.update(
        {
            # 使用系统中 Matplotlib 可直接识别的 CJK 字体，保证中文标签完整显示。
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F9FC")

    for group, color in colors.items():
        indexes = [index for index, value in enumerate(groups) if value == group]
        ax.scatter(
            f1[indexes],
            latency[indexes],
            s=params[indexes] * 38,
            c=color,
            marker="o",
            alpha=0.88 if group != "本文方法" else 1.0,
            edgecolors="white" if group != "本文方法" else "#7C4A03",
            linewidths=1.5,
            zorder=3,
        )

    # 在本文方法的圆形气泡中心叠加五角星，保持所有数据点的几何形状一致。
    proposed_index = names.index("本方案")
    ax.scatter(
        f1[proposed_index],
        latency[proposed_index],
        s=params[proposed_index] * 38 * 0.52,
        marker="*",
        c="#FFD166",
        edgecolors="#7C4A03",
        linewidths=1.2,
        zorder=4,
    )

    # 所有标签统一放在气泡右侧；仅用垂直偏移错开相邻气泡的标签。
    vertical_offsets = {
        "YOLOv8-m": -12,
        "YOLO11-m": 12,
        "YOLO26-m": 0,
        "Cascade R-CNN": 0,
        "RT-DETR-m": 0,
        "RT-DETRv4-m": 0,
        "D-FINE": -30,
        "CTRP-DETR-m": 0,
        "RF-DETR-m": 0,
        "本方案": 0,
    }
    for name, x_value, y_value in zip(names, f1, latency):
        label = name
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(38, vertical_offsets[name]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold" if name == "本方案" else "normal",
            color="#7C4A03" if name == "本方案" else "#263238",
            arrowprops={
                "arrowstyle": "-",
                "color": "#B0BAC5",
                "lw": 0.8,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=5,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white" if group != "本文方法" else "#7C4A03",
            markeredgewidth=1.2,
            markersize=9,
            label=group,
        )
        for group, color in colors.items()
    ]
    group_legend = ax.legend(handles=legend_handles, title="模型类别", loc="lower right", frameon=True)
    group_legend.get_frame().set_facecolor("white")
    group_legend.get_frame().set_edgecolor("#D5DCE5")
    ax.add_artist(group_legend)
    ax.set_xlabel("综合精度 Avg F1", fontsize=13, labelpad=10)
    ax.set_ylabel("推理时间（毫秒/张）", fontsize=13, labelpad=10)
    ax.text(
        0.01,
        0.01,
        "注：输入分辨率为 1024×1024。",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#5F6B78",
    )
    ax.set_xlim(0.715, 0.940)
    # 反向显示纵轴，使较小的推理时间位于图的上方。
    ax.set_ylim(45.0, 14.0)
    ax.grid(True, linestyle="--", linewidth=0.7, color="#D8DEE7", alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB4C0")
    ax.spines["bottom"].set_color("#AAB4C0")
    fig.tight_layout()

    output_path = Path("outputs/comparison_bubble.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(output_path)


if __name__ == "__main__":
    main()
