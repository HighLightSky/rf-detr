# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 投影头（ProjectionHead）的单元测试。

不依赖 GPU 与网络，验证：
- ProjectionHead 前向输出形状（含空 batch）。
- SSCLLoss 启用投影头后：实例模式损失有限非负、原型库维度 == projection_dim、
  update_prototypes 先投影再更新。
- 原型模式实例正样本：损失不高于纯原型模式（同类正样本提供更强引力）。
- projection_dim=None 时保持原行为（回归安全）。
- 零损失路径仍保持投影头在计算图中（利于 DDP）。
- 梯度流到投影头参数与输入特征。
"""

from __future__ import annotations

import io

import pytest
import torch

from rfdetr.sscl import SSCLLoss
from rfdetr.sscl.projection import ProjectionHead

# 5 类测试用语义相似度矩阵（与 test_sscl.py / test_sscl_prototype.py 一致）
_SEMANTIC_MATRIX = torch.tensor(
    [
        [1.0, 0.7, 0.5, 0.3, 0.1],
        [0.7, 1.0, 0.5, 0.3, 0.1],
        [0.5, 0.5, 1.0, 0.3, 0.1],
        [0.3, 0.3, 0.3, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ]
)


def _make_projection_loss(**overrides: object) -> SSCLLoss:
    """构造启用投影头的 SSCLLoss，默认 hidden_dim=16、projection_dim=8。

    Args:
        **overrides: 覆盖 SSCLLoss 构造参数（如 prototype_mode / prototype_instance_pos）。

    Returns:
        SSCLLoss 实例。
    """
    kwargs: dict[str, object] = {
        "semantic_matrix": _SEMANTIC_MATRIX,
        "hidden_dim": 16,
        "projection_dim": 8,
    }
    kwargs.update(overrides)
    return SSCLLoss(**kwargs)  # type: ignore[arg-type]


def _warm_bank(loss_fn: SSCLLoss, n_per_class: int = 2) -> None:
    """用确定性数据预热原型库（5 类每类 n 样本）。"""
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(5).repeat_interleave(n_per_class)
    features = torch.randn(5 * n_per_class, 16, generator=generator)
    loss_fn.update_prototypes(features, labels)


class TestProjectionHead:
    """ProjectionHead 前向行为。"""

    def test_output_shape(self) -> None:
        """输入 [4, 16] 输出 [4, 8]。"""
        head = ProjectionHead(in_dim=16, out_dim=8)
        out = head(torch.randn(4, 16))
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_zero_batch(self) -> None:
        """空 batch 输入 [0, 16] 输出 [0, 8]（兼容 num_fg==0 分支）。"""
        head = ProjectionHead(in_dim=16, out_dim=8)
        out = head(torch.randn(0, 16))
        assert out.shape == (0, 8)


class TestSSCLProjectionLoss:
    """SSCLLoss 启用投影头后的行为。"""

    def test_instance_projected_loss_finite_nonneg(self) -> None:
        """实例模式 + 投影：有同类对时损失有限且 >= 0。"""
        loss_fn = _make_projection_loss(prototype_mode=False)
        features = torch.randn(6, 16)
        labels = torch.arange(3).repeat_interleave(2)  # 每类 2 个实例
        loss = loss_fn(features, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_instance_gradient_flows_to_head_and_features(self) -> None:
        """实例模式 + 投影：梯度流到投影头参数与输入特征。"""
        loss_fn = _make_projection_loss(prototype_mode=False)
        features = torch.randn(6, 16, requires_grad=True)
        labels = torch.arange(3).repeat_interleave(2)
        loss = loss_fn(features, labels)
        loss.backward()
        assert loss_fn.projection_head.linear1.weight.grad is not None
        assert torch.isfinite(loss_fn.projection_head.linear1.weight.grad).all()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_prototype_bank_dim_equals_projection_dim(self) -> None:
        """原型模式 + 投影：原型库维度 == projection_dim（原型住投影空间）。"""
        loss_fn = _make_projection_loss(prototype_mode=True)
        assert loss_fn.prototype_bank.prototypes.shape[-1] == 8
        _warm_bank(loss_fn)
        assert loss_fn.prototype_bank.prototypes.shape[-1] == 8

    def test_update_prototypes_projects_before_update(self) -> None:
        """update_prototypes 在投影空间更新原型：预热后维度为 projection_dim 且就位。"""
        loss_fn = _make_projection_loss(prototype_mode=True)
        _warm_bank(loss_fn)
        num_updates = loss_fn.prototype_bank.num_updates
        assert int(num_updates[0].item()) >= 1
        assert loss_fn.prototype_bank.prototypes.shape[-1] == 8
        # 原型应归一化到投影空间（行模长有效）
        proto_norm, valid = loss_fn.prototype_bank.get_normalized_prototypes()
        assert valid.any()
        assert proto_norm.shape[1] == 8

    def test_instance_pos_loss_not_above_pure_prototype(self) -> None:
        """实例正样本不增加损失：同一投影头/原型下，开启实例正样本的损失 <= 纯原型。

        数学上：正样本同时加入分子分母，且 loss = log(denom) - log(num) >= 0，
        由 log 凹性可知 loss 随正样本增多而单调不增。
        """
        loss_fn = _make_projection_loss(prototype_mode=True, prototype_instance_pos=False)
        _warm_bank(loss_fn)
        generator = torch.Generator().manual_seed(1)
        features = torch.randn(6, 16, generator=generator)
        labels = torch.arange(3).repeat_interleave(2)  # 每类 2 个实例，有同类正样本

        loss_plain = loss_fn(features, labels)
        loss_fn.prototype_instance_pos = True
        loss_inst = loss_fn(features, labels)

        assert torch.isfinite(loss_inst)
        assert loss_inst.item() >= 0.0
        assert loss_inst.item() <= loss_plain.item() + 1e-6

    def test_prototype_instance_pos_gradient_flows(self) -> None:
        """原型模式 + 投影 + 实例正样本：梯度流到投影头参数与输入特征。"""
        loss_fn = _make_projection_loss(prototype_mode=True, prototype_instance_pos=True)
        _warm_bank(loss_fn)
        features = torch.randn(6, 16, requires_grad=True)
        labels = torch.arange(3).repeat_interleave(2)
        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert loss_fn.projection_head.linear1.weight.grad is not None
        assert torch.isfinite(loss_fn.projection_head.linear1.weight.grad).all()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_projection_dim_none_no_head_unchanged(self) -> None:
        """projection_dim=None 时不创建投影头、原型维度 = hidden_dim（回归安全）。"""
        loss_fn = _make_projection_loss(prototype_mode=True, projection_dim=None)
        assert not hasattr(loss_fn, "projection_head")
        assert loss_fn.prototype_bank.prototypes.shape[-1] == 16

    def test_projection_with_hidden_dim_none_raises(self) -> None:
        """projection_dim 非 None 但 hidden_dim=None 时必须报错（投影头需输入维度）。"""
        with pytest.raises(ValueError, match="hidden_dim"):
            SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, projection_dim=8)

    def test_zero_loss_path_keeps_head_in_graph(self) -> None:
        """实例模式 num_fg=1 的零损失路径仍保持投影头在计算图中（DDP 安全）。"""
        loss_fn = _make_projection_loss(prototype_mode=False)
        features = torch.randn(1, 16, requires_grad=True)
        labels = torch.tensor([0])
        loss = loss_fn(features, labels)
        loss.backward()
        # 图连接存在，backward 后投影头参数收到零梯度而非 None
        assert loss_fn.projection_head.linear1.weight.grad is not None

    def test_prototype_zero_loss_path_keeps_head_in_graph(self) -> None:
        """原型模式原型未建立时的零损失路径同样保持投影头在计算图中。"""
        loss_fn = _make_projection_loss(prototype_mode=True)
        # 不预热原型，valid_proto 全 False → 零损失
        features = torch.randn(4, 16, requires_grad=True)
        labels = torch.arange(4)
        loss = loss_fn(features, labels)
        loss.backward()
        assert loss_fn.projection_head.linear1.weight.grad is not None

    def test_state_dict_roundtrip_projection(self) -> None:
        """投影头参数与原型 buffer 随 state_dict 保存/加载往返。"""
        loss_fn = _make_projection_loss(prototype_mode=True, prototype_instance_pos=True)
        _warm_bank(loss_fn)
        buffer = io.BytesIO()
        torch.save(loss_fn.state_dict(), buffer)
        buffer.seek(0)
        restored = _make_projection_loss(prototype_mode=True, prototype_instance_pos=True)
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        # 投影头权重一致
        for p1, p2 in zip(
            loss_fn.projection_head.parameters(),
            restored.projection_head.parameters(),
        ):
            assert torch.equal(p1, p2)
        # 原型 buffer 一致
        assert torch.equal(loss_fn.prototype_bank.prototypes, restored.prototype_bank.prototypes)
