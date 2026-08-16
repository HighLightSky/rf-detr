# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义矩阵（CLIP 类别相似度）关系可视化。

对 ``data/semantic_matrix_shwx.pt`` 输出三张图：
1. **原始热图**：25×25 类别余弦相似度（含归一化后视图）；
2. **层次聚类重排热图**：把相似类别聚在一起，暴露矩阵结构问题
   （跨域污染、类对齐错误、区分度塌缩）；
3. **关键类相似度剖面**：指定类（默认 MS/FSC）与全部类的相似度条形图，
   直观看出"最像的混淆类是谁"。

用法：
    python src/scripts/analysis/visualize_semantic_matrix.py \
      --matrix data/semantic_matrix_shwx.pt \
      --out output/_score_analysis/semantic_matrix_viz \
      --focus 3 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.sscl.prompts import SHWX_CLASS_NAMES  # noqa: E402


def load_matrix(path: str | Path) -> np.ndarray:
    """加载语义矩阵（支持 {semantic_matrix: [C,C]} 或裸张量两种格式）。"""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for key in ("semantic_matrix", "matrix", "s_matrix"):
            if key in data:
                data = data[key]
                break
    mat = np.asarray(data, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"语义矩阵必须是 [C, C]，收到 {mat.shape}")
    return mat


def minmax_normalize(mat: np.ndarray) -> np.ndarray:
    """与训练侧 sscl_matrix_normalize=minmax 一致：全局映射到 [0, 1]。"""
    lo, hi = mat.min(), mat.max()
    return (mat - lo) / (hi - lo) if hi > lo else mat - lo


def plot_heatmap(mat: np.ndarray, names: list[str], title: str, out_path: Path) -> None:
    """带类名的注释热图。"""
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(mat, cmap="viridis", vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(len(names)), [n[:8] for n in names], rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), [n[:8] for n in names], fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                    color="white" if mat[i, j] > (mat.min() + mat.max()) / 2 else "black")
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def plot_dendrogram_heatmap(mat: np.ndarray, names: list[str], title: str, out_path: Path) -> None:
    """层次聚类重排热图（1 - 相似度 作为距离）。"""
    dist = 1.0 - mat
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    fig, (ax_d, ax_h) = plt.subplots(1, 2, figsize=(16, 11),
                                     gridspec_kw={"width_ratios": [1.2, 3]})
    dendrogram(Z, labels=names, ax=ax_d, leaf_rotation=90)
    ax_d.set_title("Hierarchical clustering (1 - similarity)", fontsize=11)
    order = dendrogram(Z, no_plot=True)["leaves"]
    reordered = mat[np.ix_(order, order)]
    im = ax_h.imshow(reordered, cmap="viridis", vmin=mat.min(), vmax=mat.max())
    short = [n[:10] for n in names]
    ax_h.set_xticks(range(len(names)), [short[i] for i in order], rotation=90, fontsize=7)
    ax_h.set_yticks(range(len(names)), [short[i] for i in order], fontsize=7)
    ax_h.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax_h, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def plot_profiles(mat: np.ndarray, names: list[str], focus: list[int], out_path: Path) -> None:
    """关键类与全部类的相似度剖面（按相似度降序）。"""
    n = len(focus)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, focus, strict=False):
        sims = mat[cls].copy()
        sims[cls] = -np.inf
        order = np.argsort(-sims)
        bars = ax.barh([names[i][:10] for i in order][::-1], sims[order][::-1], color="#4C72B0")
        # 标注 top3 与船类
        for idx in order[:3]:
            pos = list(order[::-1]).index(idx)
            bars[pos].set_color("#C44E52")
        for j, s in enumerate(sims):
            if j < 4 and j != cls:
                pos = list(order[::-1]).index(j)
                bars[pos].set_color("#55A868")
        ax.axvline(mat[cls].mean(), color="gray", ls="--", lw=0.8)
        ax.set_title(f"{names[cls]} similarity profile (red=top3, green=ship classes)", fontsize=11)
        ax.set_xlim(mat.min() - 0.01, 1.01)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    """主流程：加载矩阵 → 三张可视化 → 打印结构诊断。"""
    parser = argparse.ArgumentParser(description="语义矩阵关系可视化")
    parser.add_argument("--matrix", type=str, default="data/semantic_matrix_shwx.pt")
    parser.add_argument("--out", type=str, default="output/_score_analysis/semantic_matrix_viz")
    parser.add_argument("--focus", type=int, nargs="+", default=[3, 24],
                        help="剖面图中重点展示的类别索引（默认 MS=3, FSC=24）")
    args = parser.parse_args()

    mat = load_matrix(args.matrix)
    names = [SHWX_CLASS_NAMES.get(i, f"c{i}") for i in range(mat.shape[0])]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(mat, names, f"Semantic matrix raw similarity [{args.matrix}]", out_dir / "heatmap_raw.png")
    norm = minmax_normalize(mat)
    plot_heatmap(norm, names, "Semantic matrix minmax normalized [0,1]", out_dir / "heatmap_normalized.png")
    plot_dendrogram_heatmap(mat, names, "Hierarchical reorder", out_dir / "dendrogram.png")
    plot_profiles(mat, names, args.focus, out_dir / "profiles.png")

    # 结构诊断输出
    print("\n===== Structure diagnostics =====")
    print(f"矩阵范围: [{mat.min():.4f}, {mat.max():.4f}]（跨度 {mat.max()-mat.min():.4f}）")
    print(f"Off-diagonal mean: {mat[~np.eye(mat.shape[0], dtype=bool)].mean():.4f}")
    # 每个类的最近邻（非自身）
    print("\nNearest neighbor per class:")
    for i in range(mat.shape[0]):
        sims = mat[i].copy()
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        print(f"  {names[i]:<10} -> {names[j]:<10} ({sims[j]:.4f})")
    # 归一化后 ship 类区分度
    print("\nNormalized ship-class (0-3) similarity:")
    for i in range(4):
        row = "  ".join(f"{norm[i, j]:.3f}" for j in range(4))
        print(f"  {names[i]:<5}: {row}")
    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
