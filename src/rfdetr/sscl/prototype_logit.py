# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""视觉原型分类 logit 校准模块。"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名
from torch import Tensor, nn


class PrototypeLogitCalibrator(nn.Module):
    """用类别视觉原型为指定小样本类增加相对判别证据。

    原型只作为一个小幅 residual 加到检测分类 logit 上，不改变框回归分支。
    当原型尚未初始化或目标类别为空时，输出恒为零，模型退化为原始 RF-DETR。

    Args:
        num_classes: 前景类别数量，不包含 background。
        hidden_dim: decoder hidden feature 维度。
        max_slots: 每个类别的最大原型 slot 数。
        target_classes: 参与 logit 校准的类别列表。
        alpha: 原型证据的 logit 增益系数。
        margin: 目标类原型相对其他类原型的余弦间隔。
        temperature: 原型相对间隔的平滑温度。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        max_slots: int,
        target_classes: list[int] | None,
        alpha: float,
        margin: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}。")
        if max_slots < 1:
            raise ValueError(f"max_slots 必须 >= 1，收到 {max_slots}。")
        if temperature <= 0.0:
            raise ValueError(f"temperature 必须 > 0，收到 {temperature}。")

        self.num_classes = num_classes
        self.max_slots = max_slots
        self.target_classes = sorted(
            {int(class_id) for class_id in (target_classes or []) if 0 <= int(class_id) < num_classes}
        )
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.register_buffer("prototypes", torch.zeros(num_classes, max_slots, hidden_dim))
        self.register_buffer("valid_slots", torch.zeros(num_classes, max_slots, dtype=torch.bool))

    @torch.no_grad()
    def sync_from_bank(self, bank: nn.Module) -> None:
        """从训练期 EMA 原型库同步原型状态。

        Args:
            bank: ``SlotPrototypeBank`` 实例。

        Raises:
            ValueError: 原型类别数、slot 数或维度不一致时抛出。
        """
        bank_prototypes = getattr(bank, "prototypes", None)
        bank_valid = getattr(bank, "slot_valid_mask", None)
        bank_updates = getattr(bank, "slot_num_updates", None)
        if not isinstance(bank_prototypes, Tensor) or not isinstance(bank_valid, Tensor):
            raise ValueError("原型库缺少 prototypes 或 slot_valid_mask。")
        if bank_prototypes.shape != self.prototypes.shape:
            raise ValueError(
                f"原型维度不一致: bank={tuple(bank_prototypes.shape)}, calibrator={tuple(self.prototypes.shape)}"
            )

        valid_slots = bank_valid
        if isinstance(bank_updates, Tensor):
            valid_slots = valid_slots & (bank_updates > 0)
        self.prototypes.copy_(bank_prototypes.to(device=self.prototypes.device, dtype=self.prototypes.dtype))
        self.valid_slots.copy_(valid_slots.to(device=self.valid_slots.device))

    def forward(self, features: Tensor) -> Tensor:
        """计算追加到前景分类 logit 的原型证据。

        Args:
            features: decoder hidden states，形状为 ``[..., hidden_dim]``。

        Returns:
            形状为 ``[..., num_classes]`` 的 logit residual。
        """
        zero = features.new_zeros(*features.shape[:-1], self.num_classes)
        if not self.target_classes or not bool(self.valid_slots.any()):
            return zero

        valid = self.valid_slots & torch.isfinite(self.prototypes).all(dim=-1)
        if not bool(valid.any()):
            return zero

        feature_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("...d,ckd->...ck", feature_norm, proto_norm)
        similarity = similarity.masked_fill(~valid, torch.finfo(similarity.dtype).min)
        class_similarity = similarity.max(dim=-1).values
        class_valid = valid.any(dim=-1)
        class_similarity = class_similarity.masked_fill(
            ~class_valid,
            torch.finfo(similarity.dtype).min,
        )

        residual = zero
        for class_id in self.target_classes:
            competitor_valid = class_valid.clone()
            competitor_valid[class_id] = False
            if not bool(class_valid[class_id]) or not bool(competitor_valid.any()):
                continue

            target_similarity = class_similarity[..., class_id]
            other_similarity = class_similarity.masked_fill(
                ~competitor_valid,
                torch.finfo(class_similarity.dtype).min,
            )
            competitor_similarity = other_similarity.max(dim=-1).values
            relative_margin = target_similarity - competitor_similarity
            evidence = F.softplus((relative_margin - self.margin) / self.temperature) * self.temperature
            class_mask = F.one_hot(
                torch.tensor(class_id, device=features.device),
                num_classes=self.num_classes,
            ).to(dtype=features.dtype)
            residual = residual + self.alpha * evidence.unsqueeze(-1) * class_mask
        return residual
