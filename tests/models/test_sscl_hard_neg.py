# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""原型模式 SSCL 难例负样本（hard negatives）的单元测试。

不依赖 GPU 与网络，只测试：
- 零难例与基线完全一致（None 与空张量等价）。
- 难例列只进分母：loss(含难例) >= loss(不含难例) 恒成立。
- NaN/Inf 难例列被守卫（等价于删除该列）。
- 难例特征 detach：难例方向不产生梯度，anchor 仍有梯度。
- 与实例正样本组合、实例模式忽略难例。
- hardness_stats 诊断统计的返回值域与边界。
"""

from __future__ import annotations

import torch

from rfdetr.sscl import SSCLLoss

# 5 类测试用语义相似度矩阵（与 test_sscl_prototype.py 一致）
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
    """用确定性数据为每类建立原型（与 test_sscl_prototype.py 一致）。"""
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(5).repeat_interleave(n_per_class)
    features = torch.randn(5 * n_per_class, 16, generator=generator)
    loss_fn.update_prototypes(features, labels)


class TestSSCLHardNegativeLoss:
    """难例负样本列对原型模式损失的影响。"""

    def test_zero_hard_negatives_identical_to_baseline(self) -> None:
        """None 与空张量难例都不改变损失（与基线完全一致）。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        base = loss_fn(features, labels)
        none_hn = loss_fn(features, labels, hard_neg_features=None)
        empty_hn = loss_fn(features, labels, hard_neg_features=torch.zeros(0, 16))
        assert torch.equal(none_hn, base)
        assert torch.equal(empty_hn, base)

    def test_hard_negative_increases_denominator(self) -> None:
        """难例列只进分母：loss(含难例) >= loss(不含) 恒成立。

        难例取 anchor 自身的克隆（cos=1，最难的情况）时损失严格增大。
        """
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        base = loss_fn(features, labels)
        hn = features[:1].detach().clone()  # 与 anchor 0 完全同向 → sim = 1/τ
        with_hn = loss_fn(features, labels, hard_neg_features=hn)
        assert torch.isfinite(with_hn)
        assert with_hn.item() > base.item() + 1e-4

    def test_hard_negative_nan_row_guarded(self) -> None:
        """含 NaN/Inf 行的难例等价于删除该行；全 NaN 等价于基线。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16)
        labels = torch.arange(5)
        extra_row = torch.randn(1, 16)
        valid_hn = torch.randn(2, 16)
        bad_hn = torch.cat([extra_row, torch.full((1, 16), float("nan")), valid_hn], dim=0)
        # NaN 行等价于"删除该行"：与 [extra_row, valid_hn] 完全一致
        loss_expected = loss_fn(features, labels, hard_neg_features=torch.cat([extra_row, valid_hn], dim=0))
        loss_with_bad = loss_fn(features, labels, hard_neg_features=bad_hn)
        assert torch.isfinite(loss_with_bad)
        assert torch.allclose(loss_with_bad, loss_expected, atol=1e-6)

        # Inf 行同样被守卫（Inf 行被屏蔽 ≡ 不存在该行）
        inf_hn = torch.cat([valid_hn, torch.full((1, 16), float("inf"))], dim=0)
        loss_valid = loss_fn(features, labels, hard_neg_features=valid_hn)
        assert torch.allclose(loss_fn(features, labels, hard_neg_features=inf_hn), loss_valid, atol=1e-6)

        # 全 NaN → 与基线一致（所有列置 -inf，不参与分母）
        all_nan = torch.full((2, 16), float("nan"))
        base = loss_fn(features, labels)
        assert torch.allclose(loss_fn(features, labels, hard_neg_features=all_nan), base, atol=1e-6)

    def test_hard_negative_detached(self) -> None:
        """难例特征 detach：难例方向无梯度（hn.grad 为 None），anchor 仍有有限梯度。"""
        loss_fn = _make_prototype_loss(projection_dim=8)
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        hn = torch.randn(3, 16, requires_grad=True)
        loss = loss_fn(features, labels, hard_neg_features=hn)
        loss.backward()
        assert hn.grad is None  # 难例方向不产生梯度
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        # 投影头仍应收到梯度（它同时映射 anchor 与难例）
        assert loss_fn.projection_head.linear1.weight.grad is not None

    def test_hard_negative_with_instance_pos(self) -> None:
        """与实例正样本组合：损失有限，且同配置下仍满足 loss(含难例) >= loss(不含)。"""
        loss_fn = _make_prototype_loss(prototype_instance_pos=True)
        _warm_bank(loss_fn)
        features = torch.randn(5, 16, requires_grad=True)
        labels = torch.arange(5)
        base = loss_fn(features, labels)
        hn = features[:1].detach().clone()
        with_hn = loss_fn(features, labels, hard_neg_features=hn)
        assert torch.isfinite(with_hn)
        assert with_hn.item() > base.item() + 1e-4
        assert with_hn.item() >= 0.0

    def test_instance_mode_ignores_hard_negatives(self) -> None:
        """实例模式（非原型模式）忽略难例参数：损失与不传完全一致。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)
        features = torch.randn(6, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 3])
        base = loss_fn(features, labels)
        with_hn = loss_fn(features, labels, hard_neg_features=torch.randn(2, 16))
        assert torch.equal(with_hn, base)

    def test_hardness_stats_values_and_boundaries(self) -> None:
        """hardness_stats 返回正确键与值域；空库/空难例/实例模式返回空字典。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(5, 16)
        hn = torch.randn(3, 16)
        random_f = torch.randn(3, 16)

        # 完整三组对照
        stats = loss_fn.hardness_stats(features, hn, random_features=random_f)
        assert set(stats) == {
            "hn_proto_cos",
            "matched_proto_cos",
            "random_proto_cos",
            "hn_vs_random_gap",
            "hn_vs_matched_gap",
        }
        for key in ("hn_proto_cos", "matched_proto_cos", "random_proto_cos"):
            assert -1.0 <= stats[key] <= 1.0

        # 不传 random_features 时只有两组余弦
        stats2 = loss_fn.hardness_stats(features, hn)
        assert set(stats2) == {"hn_proto_cos", "matched_proto_cos"}

        # 空难例 → 空字典
        assert loss_fn.hardness_stats(features, torch.zeros(0, 16)) == {}

        # 原型库为空 → 空字典
        empty = _make_prototype_loss()
        assert empty.hardness_stats(features, hn) == {}

        # 实例模式 → 空字典
        inst = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)
        assert inst.hardness_stats(features, hn) == {}

    def test_hard_negative_all_matched_anchors_still_finite(self) -> None:
        """难例与 anchor 数量无关的边界：batch 内 anchor 很少时难例路径不崩。"""
        loss_fn = _make_prototype_loss()
        _warm_bank(loss_fn)
        features = torch.randn(1, 16)  # 单 anchor
        labels = torch.tensor([0])
        hn = torch.randn(3, 16)
        loss = loss_fn(features, labels, hard_neg_features=hn)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
