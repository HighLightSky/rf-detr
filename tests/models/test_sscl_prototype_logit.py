# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""视觉原型 logit 校准模块的单元测试。"""

from __future__ import annotations

import io

import torch

from rfdetr.sscl import PrototypeLogitCalibrator, SlotPrototypeBank


def _make_calibrator() -> PrototypeLogitCalibrator:
    """构造三类别、二维特征的轻量校准器。"""
    return PrototypeLogitCalibrator(
        num_classes=3,
        hidden_dim=2,
        max_slots=1,
        target_classes=[0],
        alpha=0.1,
        margin=0.05,
        temperature=0.1,
    )


def _make_bank() -> SlotPrototypeBank:
    """构造与校准器维度一致的原型库。"""
    return SlotPrototypeBank(
        num_classes=3,
        hidden_dim=2,
        momentum=0.9,
        max_slots=1,
        multi_slot_classes=[],
    )


class TestPrototypeLogitCalibrator:
    """验证原型校准的边界、判别性、梯度与持久化。"""

    def test_empty_bank_returns_zero(self) -> None:
        """原型尚未建立时不得改变检测分类 logit。"""
        calibrator = _make_calibrator()
        output = calibrator(torch.randn(2, 4, 2))
        assert torch.equal(output, torch.zeros_like(output))

    def test_target_without_competitor_returns_zero(self) -> None:
        """只有目标类原型时缺少相对证据，校准应保持关闭。"""
        calibrator = _make_calibrator()
        bank = _make_bank()
        bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        calibrator.sync_from_bank(bank)

        output = calibrator(torch.tensor([[[1.0, 0.0]]]))
        assert torch.equal(output, torch.zeros_like(output))

    def test_relative_evidence_only_boosts_target_class(self) -> None:
        """特征越接近目标原型，目标类增益越大，其他类别保持不变。"""
        calibrator = _make_calibrator()
        bank = _make_bank()
        bank.update(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            torch.tensor([0, 1]),
        )
        calibrator.sync_from_bank(bank)

        features = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ],
            requires_grad=True,
        )
        output = calibrator(features)

        assert output[0, 0, 0] > output[1, 0, 0]
        assert torch.equal(output[..., 1:], torch.zeros_like(output[..., 1:]))
        output.sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_sync_and_state_dict_roundtrip(self) -> None:
        """同步后的原型及有效掩码应随模型 state_dict 保存。"""
        calibrator = _make_calibrator()
        bank = _make_bank()
        bank.update(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
            torch.tensor([0, 1, 2]),
        )
        calibrator.sync_from_bank(bank)

        buffer = io.BytesIO()
        torch.save(calibrator.state_dict(), buffer)
        buffer.seek(0)
        restored = _make_calibrator()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))

        assert torch.equal(restored.prototypes, calibrator.prototypes)
        assert torch.equal(restored.valid_slots, calibrator.valid_slots)
