# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""可视化多模态原型的类别关系矩阵。

从离线原型产物和训练后 checkpoint 恢复视觉、CLIP 文本及融合原型，输出：

1. 三种原型的 C x C 类间余弦相似度热图；
2. 文本减视觉、融合减视觉、融合减文本三张关系差异热图；
3. 可复用的 JSON 数值摘要。

图中对角线会被掩盖，因为自相似恒为 1，不能用于判断类别结构。不同原始
维度的视觉与文本原型只通过各自的类别关系矩阵比较；不会直接拼接向量。

用法：
    uv run --no-sync python src/scripts/analysis/visualize_multimodal_prototype_relations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from torch import Tensor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.sscl.proto_guidance.artifacts import load_proto_artifacts  # noqa: E402
from rfdetr.sscl.proto_guidance.guidance import ProtoGuidance  # noqa: E402

DEFAULT_ARTIFACTS = Path("data/proto_guidance_shwx_1024_from120ep.pt")
# PL 原始 checkpoint 才完整保存 ProtoGuidance 的投影层与融合权重；
# checkpoint_best_total.pth 适合检测推理，但当前导出流程可能保留旧的原型状态。
DEFAULT_CHECKPOINT = Path("output/0831-final-多模态对齐第二步/last.ckpt")
DEFAULT_OUTPUT_DIR = Path("output/_score_analysis/multimodal_prototype_relations")
CHECKPOINT_PREFIX = "model.transformer.proto_guidance."


def _load_checkpoint_state(path: Path) -> dict[str, Tensor]:
    """加载 checkpoint 中的状态字典。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 缺少 state_dict: {path}")
    return state


def _load_prototypes(
    artifacts_path: Path,
    checkpoint_path: Path,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str], tuple[float, float]]:
    """恢复原始视觉、文本和训练后融合原型。"""
    artifacts = load_proto_artifacts(artifacts_path)
    visual = artifacts["visual_prototypes"]
    text = artifacts["text_prototypes"]
    valid_slots = artifacts["valid_slots"]
    class_names = [str(name) for name in artifacts["class_names"]]
    if visual.ndim != 3 or text.ndim != 2 or valid_slots.shape != visual.shape[:2]:
        raise ValueError("原型产物的视觉、文本或有效槽位形状不合法。")
    if len(class_names) != visual.shape[0] or text.shape[0] != visual.shape[0]:
        raise ValueError("原型产物中的类别名称、视觉原型和文本原型类别数不一致。")

    module = ProtoGuidance.build(
        num_classes=int(visual.shape[0]),
        hidden_dim=int(visual.shape[-1]),
        text_dim=int(text.shape[-1]),
        num_slots=int(visual.shape[1]),
        artifacts_path=artifacts_path,
        tau_p=0.2,
    )
    if module is None:
        raise ValueError(f"无法加载原型产物: {artifacts_path}")
    state = _load_checkpoint_state(checkpoint_path)
    proto_state = {
        key[len(CHECKPOINT_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(CHECKPOINT_PREFIX)
    }
    if not proto_state:
        raise ValueError(f"checkpoint 中没有 ProtoGuidance 参数: {checkpoint_path}")
    missing, unexpected = module.load_state_dict(proto_state, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint 含无法识别的 ProtoGuidance 参数: {unexpected}")
    required_prefixes = ("fusion.", "visual_bank.", "P_t_clip")
    if any(not any(key.startswith(prefix) for key in proto_state) for prefix in required_prefixes):
        raise ValueError("checkpoint 缺少融合分析所需的 ProtoGuidance 参数。")
    if missing:
        print(f"提示：未从 checkpoint 恢复的非必要参数: {missing}")

    module.eval()
    with torch.no_grad():
        fused, checkpoint_valid = module.fused_prototypes()
    weights = (
        float(F.softplus(module.fusion.w_v).item()),
        float(F.softplus(module.fusion.w_t).item()),
    )
    return (
        module.visual_bank.prototypes.detach().cpu(),
        module.P_t_clip.detach().cpu(),
        fused.detach().cpu(),
        checkpoint_valid.detach().cpu(),
        class_names,
        weights,
    )


def _class_relation_matrix(prototypes: Tensor, valid_slots: Tensor | None = None) -> np.ndarray:
    """聚合多槽位原型并计算类间余弦相似度矩阵。"""
    if prototypes.ndim == 3:
        if valid_slots is None or valid_slots.shape != prototypes.shape[:2]:
            raise ValueError("多槽位原型需要同形状的有效槽位掩码。")
        weights = valid_slots.to(dtype=prototypes.dtype).unsqueeze(-1)
        class_prototypes = (prototypes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    elif prototypes.ndim == 2:
        class_prototypes = prototypes
    else:
        raise ValueError(f"原型必须是 [C, D] 或 [C, M, D]，收到 {tuple(prototypes.shape)}")
    normalized = F.normalize(class_prototypes, dim=-1)
    return (normalized @ normalized.T).numpy()


def _upper_triangle_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个关系矩阵非对角上三角元素的 Pearson 相关。"""
    upper = np.triu_indices_from(first, k=1)
    first_values = first[upper]
    second_values = second[upper]
    if np.std(first_values) < 1e-12 or np.std(second_values) < 1e-12:
        return 1.0 if np.allclose(first_values, second_values) else 0.0
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    """返回关系矩阵去除自相似对角线后的元素。"""
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _similarity_limits(matrix: np.ndarray) -> tuple[float, float]:
    """用稳健分位数确定单张相似度热图的色标范围。"""
    lower, upper = np.quantile(_off_diagonal_values(matrix), [0.01, 0.99])
    padding = max((upper - lower) * 0.05, 1e-4)
    return float(lower - padding), float(upper + padding)


def _difference_limit(matrix: np.ndarray) -> float:
    """计算单张差异热图以零为中心的稳健对称范围。"""
    return max(float(np.quantile(np.abs(_off_diagonal_values(matrix)), 0.99)), 1e-4)


def _masked(matrix: np.ndarray) -> np.ma.MaskedArray:
    """掩盖自相似对角线，突出真正的类间关系。"""
    return np.ma.masked_array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))


def _short_labels(class_names: list[str]) -> list[str]:
    """为密集坐标轴生成保留类别索引的紧凑标签。"""
    return [f"{index:02d} {name}" for index, name in enumerate(class_names)]


def _style_axis(ax: plt.Axes, labels: list[str], show_x: bool, show_y: bool) -> None:
    """设置热图坐标轴与海陆空类别组分隔线。"""
    count = len(labels)
    ax.set_xticks(np.arange(count))
    ax.set_yticks(np.arange(count))
    ax.set_xticklabels(labels if show_x else [], rotation=90, fontsize=6.4)
    ax.set_yticklabels(labels if show_y else [], fontsize=6.7)
    ax.tick_params(axis="both", length=0, pad=2)
    for boundary in (3.5, count - 1.5):
        ax.axhline(boundary, color="#FFFFFF", linewidth=1.4, alpha=0.9)
        ax.axvline(boundary, color="#FFFFFF", linewidth=1.4, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#D0D5DD")
        spine.set_linewidth(0.65)


def _add_group_labels(ax: plt.Axes, count: int) -> None:
    """在首图顶部标出 SHWX 的舰船、飞机和发射车类别组。"""
    groups = ((0, 3, "Ships"), (4, count - 2, "Aircraft"), (count - 1, count - 1, "FSC"))
    for start, end, name in groups:
        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            -2.2,
            name,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#344054",
            clip_on=False,
        )


def _plot(
    visual: np.ndarray,
    text: np.ndarray,
    fused: np.ndarray,
    class_names: list[str],
    weights: tuple[float, float],
    output_dir: Path,
) -> dict[str, float]:
    """绘制六联热图并返回关系一致性摘要。"""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )
    labels = _short_labels(class_names)
    differences = (text - visual, fused - visual, fused - text)
    relation_stats = {
        "visual_text_pearson": _upper_triangle_correlation(visual, text),
        "visual_fused_pearson": _upper_triangle_correlation(visual, fused),
        "text_fused_pearson": _upper_triangle_correlation(text, fused),
        "fusion_visual_weight": weights[0],
        "fusion_text_weight": weights[1],
    }

    figure = plt.figure(figsize=(21.2, 13.3), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=(1, 0.04, 1, 0.04, 1, 0.04),
        wspace=0.08,
        hspace=0.11,
    )
    axes = np.array(
        [[figure.add_subplot(grid[row, column]) for column in (0, 2, 4)] for row in range(2)]
    )
    color_axes = np.array(
        [[figure.add_subplot(grid[row, column]) for column in (1, 3, 5)] for row in range(2)]
    )
    similarity_cmap = plt.get_cmap("cividis").copy()
    difference_cmap = plt.get_cmap("RdBu_r").copy()
    similarity_cmap.set_bad("#F2F4F7")
    difference_cmap.set_bad("#F2F4F7")

    top_titles = (
        "(a) Visual prototypes",
        "(b) CLIP text prototypes",
        "(c) Trained fused prototypes",
    )
    top_matrices = (visual, text, fused)
    for index, (axis, color_axis, matrix, title) in enumerate(
        zip(axes[0], color_axes[0], top_matrices, top_titles, strict=True)
    ):
        lower, upper = _similarity_limits(matrix)
        image = axis.imshow(
            _masked(matrix),
            cmap=similarity_cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        _style_axis(axis, labels, show_x=False, show_y=index == 0)
        axis.set_title(f"{title}\n[{lower:.3f}, {upper:.3f}]", fontsize=11.5, pad=10, color="#101828")
        color_bar = figure.colorbar(image, cax=color_axis)
        color_bar.set_label("Cosine similarity", fontsize=8.5)
        color_bar.ax.tick_params(labelsize=7.5)
    _add_group_labels(axes[0, 0], len(labels))

    bottom_titles = (
        f"(d) Text - visual  (r = {relation_stats['visual_text_pearson']:.2f})",
        f"(e) Fused - visual  (r = {relation_stats['visual_fused_pearson']:.2f})",
        f"(f) Fused - text  (r = {relation_stats['text_fused_pearson']:.2f})",
    )
    for index, (axis, color_axis, matrix, title) in enumerate(
        zip(axes[1], color_axes[1], differences, bottom_titles, strict=True)
    ):
        difference_limit = _difference_limit(matrix)
        difference_norm = TwoSlopeNorm(vmin=-difference_limit, vcenter=0.0, vmax=difference_limit)
        image = axis.imshow(
            _masked(matrix),
            cmap=difference_cmap,
            norm=difference_norm,
            interpolation="nearest",
            aspect="equal",
        )
        _style_axis(axis, labels, show_x=True, show_y=index == 0)
        axis.set_title(
            f"{title}\n$\\pm${difference_limit:.3f}",
            fontsize=11.5,
            pad=10,
            color="#101828",
        )
        color_bar = figure.colorbar(image, cax=color_axis)
        color_bar.set_label("Cosine change", fontsize=8.5)
        color_bar.ax.tick_params(labelsize=7.5)

    figure.suptitle(
        "Class-relation geometry of multimodal prototypes",
        fontsize=18,
        fontweight="bold",
        color="#101828",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        "Diagonal entries are masked. Each panel uses a robust local color scale; r denotes off-diagonal Pearson correlation.",
        ha="center",
        fontsize=9,
        color="#475467",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "multimodal_prototype_relations.png", dpi=240, bbox_inches="tight")
    figure.savefig(output_dir / "multimodal_prototype_relations.pdf", bbox_inches="tight")
    plt.close(figure)
    return relation_stats


def main() -> None:
    """解析参数、恢复训练后原型并保存热图和数值摘要。"""
    parser = argparse.ArgumentParser(description="多模态原型类别关系热图")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS, help="离线原型产物")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练后 checkpoint")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    visual_slots, text_prototypes, fused_slots, valid_slots, class_names, weights = _load_prototypes(
        args.artifacts,
        args.checkpoint,
    )
    visual_relation = _class_relation_matrix(visual_slots, valid_slots)
    text_relation = _class_relation_matrix(text_prototypes)
    fused_relation = _class_relation_matrix(fused_slots, valid_slots)
    stats = _plot(
        visual_relation,
        text_relation,
        fused_relation,
        class_names,
        weights,
        args.output_dir,
    )
    report: dict[str, Any] = {
        "artifacts": str(args.artifacts),
        "checkpoint": str(args.checkpoint),
        "class_names": class_names,
        "num_classes": len(class_names),
        "valid_slots": int(valid_slots.sum().item()),
        "total_slots": int(valid_slots.numel()),
        "stats": stats,
    }
    report_path = args.output_dir / "multimodal_prototype_relations.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 PNG: {args.output_dir / 'multimodal_prototype_relations.png'}")
    print(f"已保存 PDF: {args.output_dir / 'multimodal_prototype_relations.pdf'}")
    print(f"已保存摘要: {report_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
