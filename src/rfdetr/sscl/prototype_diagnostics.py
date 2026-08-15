# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""原型空间的几何诊断工具。

ProtoGuidance 与 SSCL 的特征维度和网络层不同，不能直接比较向量本身。
本模块只比较按类别聚合后的余弦关系矩阵，因此可用于判断两个空间是否
保持了相同的类别结构，也可在训练中提前发现原型塌缩。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名
from torch import Tensor


def _class_prototypes(prototypes: Tensor, valid_slots: Tensor | None = None) -> Tensor:
    """把单 slot 或多 slot 原型聚合为每类一个单位向量。"""
    if prototypes.ndim == 2:
        class_prototypes = prototypes
    elif prototypes.ndim == 3:
        if valid_slots is None:
            valid_slots = torch.ones(
                prototypes.shape[:2], dtype=torch.bool, device=prototypes.device
            )
        if valid_slots.shape != prototypes.shape[:2]:
            raise ValueError("valid_slots 形状必须与原型的前两维一致。")
        weights = valid_slots.to(dtype=prototypes.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        class_prototypes = (prototypes * weights).sum(dim=1) / counts
    else:
        raise ValueError("prototypes 必须是 [C, D] 或 [C, M, D]。")
    if class_prototypes.shape[0] < 2:
        raise ValueError("原型至少需要两个类别才能计算类间几何。")
    return F.normalize(class_prototypes, dim=-1)


def prototype_geometry(
    prototypes: Tensor,
    valid_slots: Tensor | None = None,
) -> dict[str, Tensor]:
    """计算原型类间相似度、有效秩和塌缩指标。

    Args:
        prototypes: 原型张量，形状为 ``[C, D]`` 或 ``[C, M, D]``。
        valid_slots: 多 slot 原型的有效槽位掩码 ``[C, M]``。

    Returns:
        包含 ``offdiag_cos_mean``、``offdiag_cos_std``、
        ``offdiag_cos_max``、``effective_rank`` 的标量张量字典。
    """
    class_prototypes = _class_prototypes(prototypes, valid_slots)
    gram = class_prototypes @ class_prototypes.T
    offdiag = gram[~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)]
    singular_values = torch.linalg.svdvals(class_prototypes)
    effective_rank = singular_values.sum().square() / singular_values.square().sum().clamp_min(1e-12)
    return {
        "offdiag_cos_mean": offdiag.mean(),
        "offdiag_cos_std": offdiag.std(unbiased=False),
        "offdiag_cos_max": offdiag.max(),
        "effective_rank": effective_rank,
    }


def prototype_relation_alignment(first: Tensor, second: Tensor) -> Tensor:
    """比较两个不同维度原型空间的类别关系矩阵。

    Args:
        first: 第一个空间的原型 ``[C, D1]`` 或 ``[C, M, D1]``。
        second: 第二个空间的原型 ``[C, D2]`` 或 ``[C, K, D2]``。

    Returns:
        两个类间余弦矩阵上三角元素的 Pearson 相关系数，范围约为
        ``[-1, 1]``。当两个关系矩阵都没有变化时返回 1。

    Raises:
        ValueError: 两个输入的类别数不一致。
    """
    first_class = _class_prototypes(first)
    second_class = _class_prototypes(second)
    if first_class.shape[0] != second_class.shape[0]:
        raise ValueError("两个原型空间的类别数必须一致。")
    first_gram = first_class @ first_class.T
    second_gram = second_class @ second_class.T
    upper = torch.triu(torch.ones_like(first_gram, dtype=torch.bool), diagonal=1)
    first_values = first_gram[upper]
    second_values = second_gram[upper]
    first_centered = first_values - first_values.mean()
    second_centered = second_values - second_values.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator) <= 1e-12:
        return first_values.new_tensor(1.0 if torch.allclose(first_values, second_values) else 0.0)
    return (first_centered @ second_centered) / denominator


__all__ = ["prototype_geometry", "prototype_relation_alignment"]
