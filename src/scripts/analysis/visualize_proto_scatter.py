# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""绘制多模态原型的二维散点图。

视觉原型和文本原型处于不同特征空间，因此默认分别降维并绘图，
不会把 256 维视觉向量与 768 维文本向量直接拼接。

用法：
    uv run --no-sync python src/scripts/analysis/visualize_proto_scatter.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "data/proto_guidance_shwx_1024_from120ep.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/_score_analysis/proto_scatter"


def _embed(features: np.ndarray, method: str, perplexity: float, seed: int) -> np.ndarray:
    """将特征先 PCA 再降到二维。"""
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    pca_dim = min(50, normalized.shape[0] - 1, normalized.shape[1])
    reduced = PCA(n_components=max(2, pca_dim), random_state=seed).fit_transform(normalized)
    if method == "pca":
        return reduced[:, :2]
    effective_perplexity = min(perplexity, max(2.0, (len(features) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)


def _class_separated_layout(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    perplexity: float,
    seed: int,
) -> np.ndarray:
    """生成类别分离的展示布局，保持类内形状但压缩类内半径。

    该布局仅用于可视化：类别中心使用 t-SNE，槽位相对中心的位置使用
    类内 PCA 残差，并将每类半径限制在最近类别中心距离的一小部分。
    因此不能把二维坐标间距解释为原始特征空间中的精确距离。
    """
    class_centers = np.stack(
        [features[labels == class_id].mean(axis=0) for class_id in range(num_classes)]
    )
    centers_2d = _embed(class_centers, "tsne", min(perplexity, 8.0), seed)
    distances = np.sqrt(((centers_2d[:, None] - centers_2d[None, :]) ** 2).sum(axis=-1))
    distances[distances == 0] = np.inf
    target_radius = 0.5 * np.min(distances, axis=1)
    output = np.empty((len(features), 2), dtype=np.float32)
    for class_id in range(num_classes):
        mask = labels == class_id
        local = features[mask] - class_centers[class_id]
        if len(local) > 1 and np.any(np.linalg.norm(local, axis=1) > 1e-8):
            local_2d = PCA(n_components=2, random_state=seed).fit_transform(local)
            local_2d -= local_2d.mean(axis=0, keepdims=True)
            radius = np.max(np.linalg.norm(local_2d, axis=1))
            if radius > 1e-8:
                local_2d = local_2d * (target_radius[class_id] / radius)
        else:
            local_2d = np.zeros((int(mask.sum()), 2), dtype=np.float32)
        output[mask] = centers_2d[class_id] + local_2d
    return output


def _scatter(
    axis: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    palette: np.ndarray,
    *,
    title: str,
    annotate: bool,
    alpha: float = 0.9,
) -> None:
    """绘制按类别着色的散点和可选类别中心标签。"""
    for class_id, name in enumerate(names):
        mask = labels == class_id
        if not np.any(mask):
            continue
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=19 if not annotate else 42,
            alpha=alpha,
            color=palette[class_id],
            edgecolors="white" if annotate else "none",
            linewidths=0.35,
            label=name,
        )
        if annotate:
            center = points[mask].mean(axis=0)
            axis.scatter(
                center[0], center[1], marker="*", s=105, color=palette[class_id],
                edgecolors="#101828", linewidths=0.45, zorder=4,
            )
            axis.annotate(
                name, center, xytext=(3, 3), textcoords="offset points",
                fontsize=7, color="#101828",
            )
        # 用覆盖该类别全部点的虚线圆突出类别范围。
        center = points[mask].mean(axis=0)
        radius = float(np.max(np.linalg.norm(points[mask] - center, axis=1)))
        axis.add_patch(
            Circle(
                center,
                max(radius * 1.18, 1e-3),
                fill=False,
                linestyle=(0, (3, 3)),
                linewidth=0.9,
                edgecolor=palette[class_id],
                alpha=0.9,
                zorder=2,
            )
        )
    axis.set_title(title, fontsize=11, fontweight="bold", pad=8)
    axis.set_xlabel("Layout dimension 1", fontsize=9, labelpad=4)
    axis.set_ylabel("Layout dimension 2", fontsize=9, labelpad=4)
    axis.tick_params(axis="both", labelsize=8, length=3, color="#667085")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("#D0D5DD")


def main() -> None:
    """加载原型、生成散点图并保存统计摘要。"""
    parser = argparse.ArgumentParser(description="多模态原型二维散点图")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method",
        choices=("tsne", "pca", "class-separated"),
        default="class-separated",
        help="降维方式；class-separated 仅用于改善类别可读性，不代表真实距离",
    )
    parser.add_argument("--perplexity", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data: dict[str, Any] = torch.load(args.artifacts, map_location="cpu", weights_only=True)
    visual = data["visual_prototypes"].float()
    valid = data["valid_slots"].bool()
    text = data["text_prototypes"].float()
    names = [str(name) for name in data["class_names"]]
    if visual.ndim != 3 or valid.shape != visual.shape[:2] or text.ndim != 2:
        raise ValueError("原型形状必须为 visual[C,M,D]、valid[C,M]、text[C,T]")

    visual_points = visual[valid].numpy()
    visual_labels = np.repeat(np.arange(visual.shape[0]), visual.shape[1])[valid.flatten().numpy()]

    visual_embedding = (
        _class_separated_layout(
            visual_points, visual_labels, len(names), args.perplexity, args.seed
        )
        if args.method == "class-separated"
        else _embed(visual_points, args.method, args.perplexity, args.seed)
    )
    # 使用固定色相、高饱和度调色板，避免连续色图中的浅色类别难以辨认。
    hues = np.linspace(0.0, 1.0, len(names), endpoint=False)
    palette = hsv_to_rgb(np.column_stack((hues, np.full(len(names), 0.86), np.full(len(names), 0.78))))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    figure, axis = plt.subplots(figsize=(12, 10))
    _scatter(
        axis, visual_embedding, visual_labels, names, palette,
        title="Visual slots", annotate=False,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.91, 0.5), fontsize=8, frameon=False)
    figure.subplots_adjust(right=0.78, bottom=0.105, top=0.93)
    figure.text(
        0.5, 0.055,
        "Each point is a visual prototype slot; color indicates class. "
        "Class-separated layout improves readability; 2D distances are not original feature distances.",
        ha="center", va="bottom", fontsize=9, color="#475467",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "proto_scatter.png", dpi=240, bbox_inches="tight")
    figure.savefig(args.output_dir / "proto_scatter.pdf", bbox_inches="tight")
    plt.close(figure)

    stats = {
        "artifacts": str(args.artifacts),
        "method": args.method,
        "seed": args.seed,
        "visual_slots": int(visual_labels.size),
        "num_classes": len(names),
        "layout_note": (
            "class-separated 仅用于增强可读性，二维类间距离不代表原始特征距离。"
            if args.method == "class-separated"
            else "二维坐标由 PCA/t-SNE 产生，坐标轴和绝对距离不具备物理意义。"
        ),
        "visual_slot_silhouette": (
            float(silhouette_score(visual_embedding, visual_labels))
            if args.method != "class-separated" and len(np.unique(visual_labels)) > 1
            else None
        ),
    }
    (args.output_dir / "proto_scatter.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已保存: {args.output_dir / 'proto_scatter.png'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
