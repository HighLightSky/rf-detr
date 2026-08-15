# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多模态原型融合模块。

视觉原型（细节）与文本原型（泛化）经投影后融合为多模态原型。v1 使用
``SimpleProtoFusion``（加权归一化融合，文本作为少样本早期稳定锚）；
论文式门控融合 ``GatedProtoFusion`` 仅搭结构骨架，v1 不启用
（``fusion_mode="gated"`` 时构建直接抛错，防止误用）。
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.sscl.fsem import FSemProjection

# 默认 CLIP 文本向量维度
DEFAULT_TEXT_DIM = 768


class _PrototypeProjectors(nn.Module):
    """原型投影子模块：视觉/文本/打分 token 三个可学习投影。

    - ``proj_v``：视觉原型均值 → 融合空间（可学习，Linear + LayerNorm）；
    - ``proj_t``：CLIP 文本原型 → 融合空间（复用 ``FSemProjection``）；
    - ``proj_token``：待打分 token 特征 → 融合空间（Linear + LayerNorm）。
    """

    def __init__(self, hidden_dim: int, text_dim: int = DEFAULT_TEXT_DIM) -> None:
        super().__init__()
        self.proj_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_t = FSemProjection(text_dim=text_dim, hidden_dim=512, out_dim=hidden_dim)
        self.proj_token = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )


class SimpleProtoFusion(nn.Module):
    """简化加权融合：``P_mm = normalize(w_v * proj_v(P_v) + w_t * proj_t(P_t))``。

    视觉原型按槽位独立投影（保留多形态），文本原型逐槽位广播。融合权重
    ``w_v / w_t`` 为可学习标量（softplus 保证正性），初始值偏向文本
    （少样本早期视觉原型噪声大，文本作为稳定锚）。

    Args:
        hidden_dim: 融合空间维度（= decoder hidden dim）。
        text_dim: 文本向量维度。
        w_v_init: 视觉融合权重初始值。
        w_t_init: 文本融合权重初始值。
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int = DEFAULT_TEXT_DIM,
        w_v_init: float = 0.3,
        w_t_init: float = 0.7,
    ) -> None:
        super().__init__()
        self.projectors = _PrototypeProjectors(hidden_dim, text_dim)
        self.w_v = nn.Parameter(torch.tensor(w_v_init))
        self.w_t = nn.Parameter(torch.tensor(w_t_init))

    def forward(
        self,
        visual_prototypes: Tensor,
        text_prototypes: Tensor,
        valid_slots: Tensor,
    ) -> Tensor:
        """融合视觉与文本原型。

        Args:
            visual_prototypes: 视觉原型 ``[C, M, d]``。
            text_prototypes: 文本原型 ``[C, text_dim]``。
            valid_slots: 槽位有效掩码 ``[C, M]``（无效槽位输出置零向量）。

        Returns:
            多模态原型 ``[C, M, d]``（逐槽位 L2 归一化，无效槽位为零向量）。
        """
        w_v = F.softplus(self.w_v)
        w_t = F.softplus(self.w_t)

        h_v = self.projectors.proj_v(visual_prototypes)  # [C, M, d]
        h_t = self.projectors.proj_t(text_prototypes)  # [C, d]
        # 文本原型逐槽位广播参与融合
        fused = w_v * h_v + w_t * h_t.unsqueeze(1)
        fused = F.normalize(fused, dim=-1)
        # 无效槽位置零（不参与打分/注意力）
        fused = fused.where(valid_slots.unsqueeze(-1), torch.zeros_like(fused))
        return fused


class GatedProtoFusion(nn.Module):
    """论文式门控融合骨架（v1 不实现，仅占位保证配置枚举完整）。

    论文（MP-DETR）流程：特征对齐 → 跨模态注意力（文本作 Q、视觉槽位作 K/V
    提取细节）→ 门控 MLP 融合。v1 使用 ``SimpleProtoFusion`` 验证有效性后再启用。

    Args:
        hidden_dim: 融合空间维度。
        text_dim: 文本向量维度。
        w_v_init: 视觉融合权重初始值。
        w_t_init: 文本融合权重初始值。
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int = DEFAULT_TEXT_DIM,
        w_v_init: float = 0.3,
        w_t_init: float = 0.7,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "GatedProtoFusion（门控融合）为 v1.5 扩展，v1 请使用 fusion_mode='simple'。"
        )


def build_fusion(
    fusion_mode: Literal["simple", "gated"],
    hidden_dim: int,
    text_dim: int = DEFAULT_TEXT_DIM,
    w_v_init: float = 0.3,
    w_t_init: float = 0.7,
) -> nn.Module:
    """按配置构建融合模块。

    Args:
        fusion_mode: ``"simple"`` 或 ``"gated"``（gated 未实现时抛错）。
        hidden_dim: 融合空间维度。
        text_dim: 文本向量维度。
        w_v_init: 视觉融合权重初始值。
        w_t_init: 文本融合权重初始值。

    Returns:
        融合模块实例。

    Raises:
        NotImplementedError: ``fusion_mode="gated"`` 时抛出（v1 未实现）。
    """
    if fusion_mode == "gated":
        return GatedProtoFusion(hidden_dim, text_dim, w_v_init, w_t_init)
    if fusion_mode != "simple":
        raise ValueError(f"未知融合模式: {fusion_mode!r}")
    return SimpleProtoFusion(hidden_dim, text_dim, w_v_init, w_t_init)


def _cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """计算归一化余弦相似度（输入已归一化则退化为点积）。"""
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-2, -1)


def _content_context(tgt_q: Tensor, proto_slots: Tensor, valid: Tensor) -> Tensor:
    """内容增强用的槽位交叉注意力：``softmax((tgt·P^T)/√d) @ P``。

    Args:
        tgt_q: 内容查询 ``[bs, Q, d]``。
        proto_slots: 关联类别的槽位原型 ``[bs, Q, M, d]``。
        valid: 槽位有效掩码 ``[bs, Q, M]``。

    Returns:
        注意力上下文 ``[bs, Q, d]``。
    """
    scores = (tgt_q.unsqueeze(-2) @ proto_slots.transpose(-2, -1)) / math.sqrt(tgt_q.shape[-1])
    scores = scores.squeeze(-2)  # [bs, Q, M]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    # 防御：某 query 的所有槽位都无效时退回均匀注意力（正常不应发生）
    valid_any = valid.any(dim=-1, keepdim=True)
    scores = scores.where(valid_any, torch.zeros_like(scores))
    weights = F.softmax(scores, dim=-1)  # [bs, Q, M]
    return torch.einsum("bqm,bqmd->bqd", weights, proto_slots)
