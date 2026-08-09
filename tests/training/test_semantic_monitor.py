# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义头训练监控 SemanticMonitor 的单元测试。

不依赖 GPU，验证：
- update 累加多步后 on_train_epoch_end 输出 ``train/sem/*`` 指标（novel/base 聚合正确）。
- 空累加器时 on_train_epoch_end 不产生任何指标（安全返回）。
- 梯度范数指标随 update_grad_norms 正确输出。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.sscl import SemanticMonitor

_CLASS_NAMES = ["HM", "LQS", "QHS", "MS", "A1", "A2", "FSC"]  # 7 类，novel=[0..3]
_NOVEL = [0, 1, 2, 3]
_BASE = [4, 5, 6]
_C = 7
_D = 8


class _FakeModule:
    """捕获 log_dict 调用的假 LightningModule。"""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float], **kwargs: object) -> None:
        self.metrics.update(metrics)


def _step_stats(novel_alpha: float = 0.5) -> dict[str, object]:
    """构造一个采样步的统计（novel 类 α 显著高于 base，便于校验分组均值）。"""
    alpha = torch.tensor([novel_alpha] * 4 + [0.1, 0.1, 0.1])
    theta = torch.full((_C,), 250.0)
    m = torch.zeros(_C, _D)
    m[:, :4] = 0.9  # 前 4 通道激活
    m[:, 4:] = 0.1
    ratio_sem = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
    ratio_mask = torch.zeros(_C)
    align = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4])
    return {
        "alpha": alpha,
        "theta": theta,
        "M": m,
        "ratio_sem": ratio_sem,
        "ratio_mask": ratio_mask,
        "align": align,
    }


def test_monitor_outputs_expected_metrics() -> None:
    """两次采样后 epoch 末输出含 novel/base 聚合与逐类指标。"""
    monitor = SemanticMonitor(_CLASS_NAMES, _NOVEL, _BASE, align_classes=_NOVEL, num_classes=_C)
    monitor.update(_step_stats())
    monitor.update(_step_stats())

    fake = _FakeModule()
    monitor.on_train_epoch_end(fake)
    metrics = fake.metrics

    # α 聚合：novel 均值 ≈ 0.5，base 均值 ≈ 0.1
    assert metrics["train/sem/alpha_mean_novel"] == pytest.approx(0.5)
    assert metrics["train/sem/alpha_mean_base"] == pytest.approx(0.1)
    # 逐类 α（novel 4 类）
    assert metrics["train/sem/alpha/HM"] == pytest.approx(0.5)
    assert metrics["train/sem/alpha/MS"] == pytest.approx(0.5)
    # 掩码激活率：novel 前 4 通道激活 → 0.5 激活率；整体 M 均值 = 0.5
    assert metrics["train/sem/mask_active_ratio_novel"] == pytest.approx(0.5)
    assert metrics["train/sem/mask_mean"] == pytest.approx(0.5)
    # 贡献占比
    assert metrics["train/sem/ratio_sem_novel"] == pytest.approx(0.5)
    assert metrics["train/sem/ratio_sem_base"] == pytest.approx(0.05)
    # 对齐余弦（novel 均值 0.6）
    assert metrics["train/sem/align_cos_novel_mean"] == pytest.approx(0.6)
    assert "train/sem/align_cos/HM" in metrics
    # 阈值逐类（novel 类 θ 恒为初始值 → 冻结生效的观察项）
    assert metrics["train/sem/theta/HM"] == pytest.approx(250.0)


def test_monitor_empty_no_output() -> None:
    """无采样时 on_train_epoch_end 不产生指标（安全返回）。"""
    monitor = SemanticMonitor(_CLASS_NAMES, _NOVEL, _BASE, num_classes=_C)
    fake = _FakeModule()
    monitor.on_train_epoch_end(fake)
    assert fake.metrics == {}


def test_monitor_grad_norms() -> None:
    """梯度范数随 update_grad_norms 正确输出。"""
    monitor = SemanticMonitor(_CLASS_NAMES, _NOVEL, _BASE, num_classes=_C)
    monitor.update(_step_stats())
    alpha_grad = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0])
    theta_grad = torch.zeros(_C)
    monitor.update_grad_norms(alpha_grad, theta_grad)

    fake = _FakeModule()
    monitor.on_train_epoch_end(fake)
    metrics = fake.metrics
    assert "train/sem/alpha_grad_norm" in metrics
    assert metrics["train/sem/alpha_grad_norm_novel"] == pytest.approx(0.2)
    assert metrics["train/sem/alpha_grad_norm_base"] == pytest.approx(0.0)
    assert "train/sem/theta_grad_norm" in metrics
