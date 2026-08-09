# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义方向投影 f_sem（离线训练，训练后冻结）。

f_sem 把类别 CLIP 文本向量 t_c ∈ ℝ^768 投影到 decoder 特征空间 ℝ^d（d=256），
输出类别语义方向 s_c = f_sem(t_c)。参照 ReSet 论文的文本锚定思想（Talk2DINO
的 warping，见方案文档 Eq.1），结构为两层仿射 + tanh。

训练目标：对称 InfoNCE，使 f_sem(t_c) 与类别 c 在 Stage-1 模型上提取的
matched query 特征方向对齐。训练数据只含 base 类（绝不含少样本类），
避免对齐被噪声带歪。训练完成后 S 矩阵（每行 s_c）以 buffer 形式存入
SemanticResidual（见 semantic_head.py），本模块只参与离线准备阶段。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# 默认 CLIP 文本向量维度（CLIP ViT-L/14 text projection 输出）
DEFAULT_TEXT_DIM = 768


class FSemProjection(nn.Module):
    """两层仿射 + tanh 的语义方向投影：``Linear(text_dim→hid) → Tanh → Linear(hid→out_dim)``。

    输入类别 CLIP 文本向量，输出 decoder 特征空间中的类别语义方向。
    输出不做归一化，由调用方（训练损失 / 语义头装配）在需要时归一化。

    Args:
        text_dim: 输入文本向量维度（CLIP 文本向量 768）。
        hidden_dim: 中间隐藏维度。
        out_dim: 输出维度（必须等于 decoder hidden dim）。
    """

    def __init__(self, text_dim: int = DEFAULT_TEXT_DIM, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.linear1 = nn.Linear(text_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, text_embeddings: Tensor) -> Tensor:
        """把文本向量投影到语义方向空间。

        Args:
            text_embeddings: 文本向量 ``[N, text_dim]``。

        Returns:
            语义方向 ``[N, out_dim]``（未归一化）。
        """
        return self.linear2(self.tanh(self.linear1(text_embeddings)))


def save_fsem_artifacts(
    path: str | Path,
    s_matrix: Tensor,
    fsem_state_dict: dict[str, Tensor],
    meta: dict[str, Any],
) -> None:
    """保存 f_sem 训练产物（语义方向矩阵 + f_sem 权重 + 元信息）。

    Args:
        path: 输出文件路径（``.pt`` 后缀）。
        s_matrix: 语义方向矩阵 ``[C, d]``，行 = 各类别语义方向 s_c（L2 归一化）。
        fsem_state_dict: ``FSemProjection`` 的 ``state_dict()``（供离线复现/继续训练）。
        meta: 元信息字典（class_names、训练 epoch、对齐校验报告等）。
    """
    torch.save({"S": s_matrix.cpu(), "fsem_state_dict": fsem_state_dict, "meta": meta}, path)
    logger.info(f"f_sem 产物已保存到: {path}（S 形状: {tuple(s_matrix.shape)}）")


def load_fsem_artifacts(path: str | Path) -> dict[str, Any]:
    """加载 f_sem 训练产物。

    Args:
        path: ``save_fsem_artifacts`` 保存的文件路径。

    Returns:
        ``{"S": Tensor, "fsem_state_dict": dict, "meta": dict}``。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        KeyError: 当文件中缺少 ``S`` 键时抛出。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"f_sem 产物文件不存在: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "S" not in data:
        raise KeyError(f"文件 {path} 中缺少 'S' 键。")
    return data


def evaluate_alignment(features_by_class: dict[int, Tensor], s_matrix: Tensor) -> dict[str, Any]:
    """离线对齐校验：验证语义方向 s_c 与各类 matched 特征均值的方向一致性。

    对每个类别 c：``cos(mean(h_c), s_c)`` 为同类对齐度，``mean_{c'≠c} cos(mean(h_c), s_{c'})``
    为跨类对齐度，二者之差（gap）衡量语义方向对该类是否可判别。用于阶段 0 门控。

    Args:
        features_by_class: ``{class_id: matched 特征 [N_c, d]}`` 映射。
        s_matrix: 语义方向矩阵 ``[C, d]``，行 = 各类别语义方向。

    Returns:
        报告字典：``{"per_class": {class_id: {"align": 同类cos, "cross_mean": 跨类均值cos,
        "gap": 同类cos - 跨类均值cos}}, "mean_align": ..., "mean_gap": ...}``。
    """
    if not features_by_class:
        raise ValueError("features_by_class 不能为空。")
    if len(features_by_class) == 1:
        # 仅一个类别时无法计算跨类均值，直接返回同类对齐度
        logger.warning("仅有一个类别，跳过跨类对齐计算。")
    s_norm = F.normalize(s_matrix, dim=-1)

    per_class: dict[int, dict[str, float]] = {}
    for c, feats in sorted(features_by_class.items()):
        if feats.numel() == 0 or c >= s_matrix.shape[0]:
            continue
        h_mean = F.normalize(feats.mean(dim=0), dim=-1)
        align = float(F.cosine_similarity(h_mean.unsqueeze(0), s_norm[c].unsqueeze(0)).item())
        # 跨类对齐度：与所有其他类别语义方向的平均余弦
        other = torch.cat([s_norm[j].unsqueeze(0) for j in sorted(features_by_class) if j != c])
        cross = float(F.cosine_similarity(h_mean.unsqueeze(0), other, dim=-1).mean().item()) if other.numel() else 0.0
        per_class[c] = {"align": align, "cross_mean": cross, "gap": align - cross}

    aligns = [v["align"] for v in per_class.values()]
    gaps = [v["gap"] for v in per_class.values()]
    report: dict[str, Any] = {
        "per_class": per_class,
        "mean_align": sum(aligns) / len(aligns) if aligns else 0.0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
    }
    return report
