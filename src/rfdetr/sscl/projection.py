# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 对比空间投影头。

把 decoder 输出的 matched foreground query features 投影到低维对比空间， 再在该空间内计算 SSCL 对比损失。参照 CVPR 2026《Balanced Hierarchical
Contrastive Learning with Decoupled Queries》的投影头做法：对比损失是强 约束，直接作用在共享特征（同时喂给 class_embed 与 bbox_embed）上会迫使
特征按对比几何剧烈变形，干扰任务分支。投影头提供一个可学习的缓冲层， 让对比压力先被消化在低维空间，共享特征只被软约束。原型库同样建立在 投影空间，保证正/负样本与原型在同一几何中计算。
"""

from __future__ import annotations

from torch import Tensor, nn


class ProjectionHead(nn.Module):
    """两层 MLP 投影头：``Linear(in_dim, proj_dim) → ReLU → Linear(proj_dim, proj_dim)``。

    输入为 decoder hidden dim 的特征，输出为低维对比空间向量，随后在
    SSCLLoss 中做 L2 归一化。不引入 LayerNorm/BatchNorm（query 特征逐样本
    独立、batch 内样本少，BN 统计不稳定），保持最小设计。

    Args:
        in_dim: 输入特征维度（decoder hidden dim）。
        out_dim: 投影空间维度（对比损失在该维度内计算，通常低于 ``in_dim``）。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, features: Tensor) -> Tensor:
        """把特征投影到对比空间。

        Args:
            features: 输入特征 ``[*, in_dim]``。

        Returns:
            投影后的特征 ``[*, out_dim]``。
        """
        return self.linear2(self.relu(self.linear1(features)))
