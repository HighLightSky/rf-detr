# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""类别原型库与原型锚定 SSCL 损失的单元测试。

不依赖 GPU 与网络，只测试：
- PrototypeBank 的 EMA 更新、惰性初始化、维度校验、噪声防护与持久化。
- 原型锚定模式下 SSCL 损失的计算、梯度流、语义权重单调性与边界情况。
- 与实例模式的关键差异：每类仅 1 个样本时原型模式损失非零（规避 batch 影响）。
"""

from __future__ import annotations

import io

import pytest
import torch

from rfdetr.sscl import PrototypeBank, SSCLLoss

# 5 类测试用语义相似度矩阵（模拟舰船内部高相似、跨大类低相似）
_SEMANTIC_MATRIX = torch.tensor(
    [
        [1.0, 0.7, 0.5, 0.3, 0.1],
        [0.7, 1.0, 0.5, 0.3, 0.1],
        [0.5, 0.5, 1.0, 0.3, 0.1],
        [0.3, 0.3, 0.3, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ]
)


def _make_prototype_loss(**overrides: object) -> SSCLLoss:
    """构造原型锚定模式的 SSCLLoss（默认 5 类、hidden_dim=16）。"""
    kwargs: dict[str, object] = {
        "semantic_matrix": _SEMANTIC_MATRIX,
        "prototype_mode": True,
        "hidden_dim": 16,
    }
    kwargs.update(overrides)
    return SSCLLoss(**kwargs)  # type: ignore[arg-type]


def _warm_bank(loss_fn: SSCLLoss, n_per_class: int = 2) -> None:
    """用确定性数据为每类建立原型（多次调用结果一致，便于对照）。"""
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(5).repeat_interleave(n_per_class)
    features = torch.randn(5 * n_per_class, 16, generator=generator)
    loss_fn.update_prototypes(features, labels)


class TestPrototypeBank:
    """类别原型库的 EMA 更新与状态管理。"""

    def test_ema_update_correctness(self) -> None:
        """首次更新直接赋类内均值，之后按 momentum 做 EMA。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16, momentum=0.5)
        m1 = torch.randn(16)
        m2 = torch.randn(16)
        bank.update(torch.stack([m1, m1]), torch.tensor([0, 0]))
        assert torch.allclose(bank.prototypes[0], m1)
        bank.update(torch.stack([m2, m2]), torch.tensor([0, 0]))
        assert torch.allclose(bank.prototypes[0], 0.5 * m1 + 0.5 * m2)
        assert int(bank.num_updates[0]) == 2

    def test_lazy_dim_initialization(self) -> None:
        """不传 hidden_dim 时首次 update 惰性确定维度。"""
        bank = PrototypeBank(num_classes=5)
        assert bank.prototypes.shape == (5, 0)
        bank.update(torch.randn(4, 16), torch.tensor([0, 1, 2, 3]))
        assert bank.prototypes.shape == (5, 16)

    def test_hidden_dim_mismatch_raises(self) -> None:
        """已初始化后再遇到不同维度应抛出 ValueError。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16)
        bank.update(torch.randn(2, 16), torch.tensor([0, 1]))
        with pytest.raises(ValueError, match="hidden_dim"):
            bank.update(torch.randn(2, 32), torch.tensor([0, 1]))

    def test_empty_features_noop(self) -> None:
        """空特征更新为 no-op，不改变任何状态。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16)
        bank.update(torch.zeros(0, 16), torch.tensor([], dtype=torch.long))
        assert int(bank.num_updates.sum()) == 0

    def test_min_samples_skips_update(self) -> None:
        """某类样本数低于 min_samples 时跳过该类更新（防噪声）。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16, min_samples=2)
        bank.update(torch.randn(3, 16), torch.tensor([0, 1, 1]))
        assert int(bank.num_updates[0]) == 0  # 类 0 仅 1 样本，跳过
        assert int(bank.num_updates[1]) == 1  # 类 1 有 2 样本，更新

    def test_nan_features_skip_update(self) -> None:
        """输入特征含 NaN 时跳过更新，避免毒化原型。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16)
        feats = torch.randn(2, 16)
        feats[0, 0] = float("nan")
        bank.update(feats, torch.tensor([0, 0]))
        assert int(bank.num_updates[0]) == 0

    def test_update_detaches_and_no_grad(self) -> None:
        """更新不建立梯度图，原型 buffer 不可训练。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16)
        feats = torch.randn(2, 16, requires_grad=True)
        bank.update(feats, torch.tensor([0, 0]))
        assert int(bank.num_updates[0]) == 1
        assert not bank.prototypes.requires_grad

    def test_save_load_roundtrip(self) -> None:
        """state_dict 经 BytesIO 往返后原型与计数完全一致。"""
        bank = PrototypeBank(num_classes=5, hidden_dim=16, momentum=0.9)
        bank.update(torch.randn(6, 16), torch.tensor([0, 1, 2, 3, 4, 0]))
        buffer = io.BytesIO()
        torch.save(bank.state_dict(), buffer)
        buffer.seek(0)
        new_bank = PrototypeBank(num_classes=5, hidden_dim=16, momentum=0.9)
        new_bank.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        assert torch.allclose(new_bank.prototypes, bank.prototypes)
        assert torch.equal(new_bank.num_updates, bank.num_updates)


class TestSSCLPrototypeLoss:
    """原型锚定 SSCL 损失的行为。"""

    def test_single_sample_per_class_prototype_loss_nonzero(self) -> None:
        """核心：预热原型后，每类仅 1 个样本时原型模式损失非零且有限，
        而实例模式同输入返回零损失——证明原型锚定规避了 batch 构成影响。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)
        assert loss.item() > 1e-6

        # 实例模式同输入：每类仅 1 个样本，无同类正样本 → 零损失
        inst_loss = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)(features.detach(), labels)
        assert inst_loss.item() == 0.0

    def test_gradient_flows_to_features(self) -> None:
        """原型损失应产生有限梯度，且原型本身不可训练。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert not loss_fn.prototype_bank.prototypes.requires_grad

    def test_semantic_weight_monotonicity(self) -> None:
        """启用语义放大（rho>0）的损失应不小于不放大（rho=0）的损失。"""
        labels = torch.arange(5)
        features = torch.randn(5, 16)
        loss_weighted = _make_prototype_loss(rho=0.3)
        _warm_bank(loss_weighted)
        loss_plain = _make_prototype_loss(rho=0.0)
        _warm_bank(loss_plain)
        assert loss_weighted(features, labels) >= loss_plain(features, labels)

    def test_no_valid_prototype_returns_zero(self) -> None:
        """原型库为空时损失应为 0，而非 NaN。"""
        loss_fn = _make_prototype_loss()
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        loss = loss_fn(features, labels)
        assert loss.item() == 0.0
        assert torch.isfinite(loss)

    def test_anchor_filtering(self) -> None:
        """anchor_classes 过滤 + 原型有效掩码共同决定参与损失的 anchor。"""
        loss_fn = _make_prototype_loss()
        loss_fn.update_prototypes(torch.randn(4, 16), torch.tensor([0, 0, 1, 1]))
        features = torch.randn(5, 16)
        labels = torch.arange(5)
        # anchor 类已建原型 → 损失非零
        loss_fn.anchor_classes = [0, 1]
        assert loss_fn(features, labels).item() > 1e-6
        # anchor 类未建原型 → 无有效 anchor，损失为 0
        loss_fn.anchor_classes = [2]
        assert loss_fn(features, labels).item() == 0.0
        # anchor_classes=None 时，仅已建原型的类参与，仍有有效 anchor
        loss_fn.anchor_classes = None
        assert loss_fn(features, labels).item() > 1e-6

    def test_confusing_classes_focus_prototype(self) -> None:
        """原型模式下仅对易混类别列放大语义权重的损失应不高于全列放大。"""
        labels = torch.arange(5)
        features = torch.randn(5, 16)
        focused = _make_prototype_loss(rho=0.3, confusing_classes=[0, 1, 2])
        _warm_bank(focused)
        unfocused = _make_prototype_loss(rho=0.3)
        _warm_bank(unfocused)
        focused_val = focused(features, labels)
        unfocused_val = unfocused(features, labels)
        assert torch.isfinite(focused_val)
        assert focused_val >= 0.0
        # 缩小放大范围会降低放大强度，损失不高于全类别放大
        assert focused_val <= unfocused_val + 1e-6

    def test_update_prototypes_noop_in_instance_mode(self) -> None:
        """实例模式不创建原型库，update_prototypes 为 no-op 不报错。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)
        assert not hasattr(loss_fn, "prototype_bank")
        loss_fn.update_prototypes(torch.randn(3, 16), torch.tensor([0, 1, 2]))

    def test_mismatched_lengths_raise_prototype(self) -> None:
        """原型模式下 features 与 labels 长度不一致同样抛出 ValueError。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        with pytest.raises(ValueError, match="数量不一致"):
            loss_fn(torch.randn(3, 16), torch.tensor([0, 1]))
