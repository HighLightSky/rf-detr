# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL（语义相似度引导的监督对比学习）模块单元测试。

不依赖 GPU 与网络，只测试：
- SSCLLoss 的损失计算、梯度流与边界情况。
- BaseClassDistillLoss 的 MSE / KL 两种蒸馏方式。
- 语义矩阵的归一化、保存与加载。
- prompts 的完整性（25 个类别均有提示词）。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.sscl import (
    BaseClassDistillLoss,
    SSCLLoss,
    load_semantic_matrix,
    normalize_semantic_matrix,
    save_semantic_matrix,
)
from rfdetr.sscl.prompts import SHWX_CLASS_NAMES, SHWX_CLASS_PROMPTS

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


class TestSSCLLoss:
    """SSCL 损失的计算正确性与边界情况。"""

    def test_loss_backpropagates(self) -> None:
        """SSCL 损失应产生有限梯度并可反向传播。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, anchor_classes=[0, 1])
        features = torch.randn(6, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 3])
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_aligned_classes_give_lower_loss(self) -> None:
        """同类特征对齐且类间可分的场景损失应显著低于随机特征。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, anchor_classes=[0, 1])
        labels = torch.tensor([0, 0, 1, 1, 2, 3])

        # 构造类间正交的归一化特征：类 0 与类 1 使用互不相同且正交的单位向量，
        # 同类完全一致（正样本距离为 0），类间可分离
        class0 = torch.zeros(1, 16)
        class0[0, 0] = 1.0
        class1 = torch.zeros(1, 16)
        class1[0, 1] = 1.0
        feats_aligned = torch.cat([class0, class0, class1, class1, torch.randn(2, 16)])
        feats_random = torch.randn(6, 16)

        loss_aligned = loss_fn(feats_aligned, labels).detach()
        loss_random = loss_fn(feats_random, labels).detach()
        assert loss_aligned < loss_random

    def test_semantic_weight_increases_loss(self) -> None:
        """启用语义放大（rho>0）的损失应不小于不放大（rho=0）的损失。"""
        labels = torch.tensor([0, 0, 1, 1, 2, 3])
        features = torch.randn(6, 16)
        loss_weighted = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, tau=0.1, rho=0.3, anchor_classes=[0, 1])(
            features, labels
        ).detach()
        loss_plain = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, tau=0.1, rho=0.0, anchor_classes=[0, 1])(
            features, labels
        ).detach()
        assert loss_weighted >= loss_plain

    def test_no_valid_anchor_returns_zero(self) -> None:
        """Batch 内无同类正样本（无有效 anchor）时损失应为 0，而非 NaN。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, anchor_classes=[0])
        features = torch.randn(2, 16, requires_grad=True)
        labels = torch.tensor([0, 2])  # 两个类别各仅一个样本，无同类正样本
        loss = loss_fn(features, labels)
        assert loss.item() == 0.0
        assert torch.isfinite(loss)

    def test_single_sample_returns_zero(self) -> None:
        """只有 1 个前景样本时无法构成对比对，损失应为 0。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)
        features = torch.randn(1, 16, requires_grad=True)
        labels = torch.tensor([0])
        assert loss_fn(features, labels).item() == 0.0

    def test_confusing_classes_focus(self) -> None:
        """仅对易混负样本施加语义放大，其余负样本权重保持 1.0。"""
        labels = torch.tensor([0, 0, 1, 1, 2, 3])
        features = torch.randn(6, 16)
        focused = SSCLLoss(
            semantic_matrix=_SEMANTIC_MATRIX,
            rho=0.3,
            anchor_classes=[0, 1],
            confusing_classes=[0, 1, 2],
        )(features, labels).detach()
        unfocused = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX, rho=0.3, anchor_classes=[0, 1])(
            features, labels
        ).detach()
        assert focused >= 0.0
        assert torch.isfinite(focused)
        # 缩小负样本范围会降低放大强度，损失应不高于全类别放大
        assert focused <= unfocused + 1e-6

    def test_mismatched_lengths_raise(self) -> None:
        """Features 与 labels 长度不一致时抛出 ValueError。"""
        loss_fn = SSCLLoss(semantic_matrix=_SEMANTIC_MATRIX)
        features = torch.randn(3, 16)
        labels = torch.tensor([0, 1])
        with pytest.raises(ValueError, match="数量不一致"):
            loss_fn(features, labels)


class TestBaseClassDistillLoss:
    """基类蒸馏损失（仅作用于受保护类别通道）。"""

    def test_mse_mode(self) -> None:
        """MSE 蒸馏应等于受保护类别通道上的均方误差。"""
        distill = BaseClassDistillLoss(protected_classes=[2, 3], mode="mse")
        student = torch.randn(2, 10, 6)
        teacher = torch.randn(2, 10, 6)
        loss = distill(student, teacher)
        expected = torch.nn.functional.mse_loss(student[..., [2, 3]], teacher[..., [2, 3]])
        assert torch.allclose(loss, expected)

    def test_kl_mode_runs(self) -> None:
        """KL 蒸馏应返回有限标量。"""
        distill = BaseClassDistillLoss(protected_classes=[1, 2], mode="kl", temperature=2.0)
        student = torch.randn(2, 8, 5)
        teacher = torch.randn(2, 8, 5)
        loss = distill(student, teacher)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_identical_logits_give_zero_kl(self) -> None:
        """学生与教师 logits 完全一致时 KL 蒸馏应为 0。"""
        distill = BaseClassDistillLoss(protected_classes=[0, 1], mode="kl", temperature=2.0)
        logits = torch.randn(2, 8, 5)
        loss = distill(logits, logits)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)

    def test_invalid_mode_raises(self) -> None:
        """不支持的蒸馏方式抛出 ValueError。"""
        distill = BaseClassDistillLoss(protected_classes=[0], mode="unknown")
        with pytest.raises(ValueError, match="不支持的蒸馏方式"):
            distill(torch.randn(1, 2, 3), torch.randn(1, 2, 3))


class TestSemanticMatrix:
    """语义矩阵的归一化、保存与加载。"""

    def test_minmax_normalize(self) -> None:
        """Minmax 归一化后非对角线元素应在 [0, 1]，对角线保持 1。"""
        normalized = normalize_semantic_matrix(_SEMANTIC_MATRIX, mode="minmax")
        off_diag = normalized[~torch.eye(5, dtype=torch.bool)]
        assert off_diag.min() >= 0.0
        assert off_diag.max() <= 1.0
        assert torch.allclose(normalized.diag(), torch.ones(5))

    def test_softmax_normalize(self) -> None:
        """Softmax 归一化应保持对称性且每行非负。"""
        normalized = normalize_semantic_matrix(_SEMANTIC_MATRIX, mode="softmax", temperature=0.1)
        assert (normalized >= 0).all()
        assert torch.allclose(normalized, normalized.T, atol=1e-5)

    def test_save_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """矩阵保存后加载应完全一致。"""
        path = str(tmp_path / "matrix.pt")
        save_semantic_matrix(_SEMANTIC_MATRIX, path)
        loaded = load_semantic_matrix(path)
        assert torch.allclose(loaded, _SEMANTIC_MATRIX)

    def test_load_missing_file_raises(self) -> None:
        """加载不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="不存在"):
            load_semantic_matrix("/nonexistent/path/matrix.pt")


class TestPrompts:
    """SHWX 25 类提示词的完整性。"""

    def test_all_classes_have_prompts(self) -> None:
        """25 个类别均应有非空提示词，且类别名称一一对应。"""
        assert len(SHWX_CLASS_PROMPTS) == 25
        assert len(SHWX_CLASS_NAMES) == 25
        assert set(SHWX_CLASS_PROMPTS.keys()) == set(range(25))
        assert set(SHWX_CLASS_NAMES.keys()) == set(range(25))
        for class_id, prompts in SHWX_CLASS_PROMPTS.items():
            assert len(prompts) >= 2, f"类别 {class_id} 至少需要 2 个提示词"
            assert all(isinstance(p, str) and p.strip() for p in prompts)

    def test_ship_prompts_mention_deck(self) -> None:
        """舰船类提示词应包含甲板/船体等形态关键词。"""
        ship_prompts = SHWX_CLASS_PROMPTS[0] + SHWX_CLASS_PROMPTS[1]
        assert any("deck" in p for p in ship_prompts)
