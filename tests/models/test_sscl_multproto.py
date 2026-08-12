# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多 slot 原型 SSCL 的单元测试。"""

from __future__ import annotations

import torch

from rfdetr.sscl import SSCLLoss

_SEMANTIC_MATRIX = torch.tensor(
    [
        [1.0, 0.7, 0.5, 0.3, 0.1],
        [0.7, 1.0, 0.5, 0.3, 0.1],
        [0.5, 0.5, 1.0, 0.6, 0.1],
        [0.3, 0.3, 0.6, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ]
)


def _make_multproto_loss(group_weight: float = 1.0) -> SSCLLoss:
    """构造 HM/LQS/QHS/MS 多 slot 原型损失。"""
    return SSCLLoss(
        semantic_matrix=_SEMANTIC_MATRIX,
        prototype_mode=True,
        hidden_dim=8,
        anchor_classes=[0, 1, 2, 3],
        confusing_classes=[0, 1, 2, 3],
        prototype_max_slots=2,
        prototype_multi_slot_classes=[0, 1, 2, 3],
        prototype_group_pairs=[[0, 1], [2, 3]],
        prototype_group_weight=group_weight,
    )


def _warm_bank(loss_fn: SSCLLoss) -> None:
    """用确定性数据让四个舰船类各自建立多 slot 原型。"""
    generator = torch.Generator().manual_seed(12)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    features = torch.randn(labels.shape[0], 8, generator=generator)
    loss_fn.update_prototypes(features, labels)


class TestSSCLMultiPrototypeLoss:
    """多 slot 原型损失的行为。"""

    def test_positive_uses_class_slots_and_is_finite(self) -> None:
        """同类多个 slot 可作为正样本集合，损失保持有限非负。"""
        loss_fn = _make_multproto_loss()
        _warm_bank(loss_fn)
        generator = torch.Generator().manual_seed(21)
        features = torch.randn(4, 8, generator=generator, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_group_weight_increases_sibling_pressure(self) -> None:
        """组内 sibling 加压只会增大或保持原型分母，不应降低损失。"""
        plain = _make_multproto_loss(group_weight=1.0)
        grouped = _make_multproto_loss(group_weight=1.5)
        _warm_bank(plain)
        _warm_bank(grouped)
        generator = torch.Generator().manual_seed(33)
        features = torch.randn(8, 8, generator=generator)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        plain_loss = plain(features, labels)
        grouped_loss = grouped(features, labels)

        assert torch.isfinite(grouped_loss)
        assert grouped_loss >= plain_loss

    def test_single_slot_defaults_keep_class_level_interface(self) -> None:
        """默认单 slot 仍保留旧的类别聚合接口形状。"""
        loss_fn = SSCLLoss(
            semantic_matrix=_SEMANTIC_MATRIX,
            prototype_mode=True,
            hidden_dim=8,
        )
        _warm_bank(loss_fn)

        slot_norm, slot_valid = loss_fn.prototype_bank.get_normalized_slot_prototypes()
        class_norm, class_valid = loss_fn.prototype_bank.get_normalized_prototypes()

        assert slot_norm.shape == (5, 1, 8)
        assert slot_valid.shape == (5, 1)
        assert class_norm.shape == (5, 8)
        assert class_valid.shape == (5,)
