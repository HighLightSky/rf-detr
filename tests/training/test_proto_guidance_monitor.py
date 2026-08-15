# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ProtoGuidanceMonitor 训练监控累加器单元测试。

验证：
- update 累加、on_train_epoch_end 输出 ``train/proto/*`` 键齐全并清空累加器。
- 空累加器安全返回（不输出任何指标）。
- 梯度范数累加与逐类选中占比输出。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.sscl.proto_guidance.monitor import ProtoGuidanceMonitor

_C, _BS = 5, 2


def _make_stats(bs: int = _BS) -> dict[str, torch.Tensor]:
    """构造一采样步的统计字典。"""
    return {
        "topk_overlap": torch.tensor([0.95, 0.97]),
        "proto_selected_ratio": torch.tensor([0.05, 0.03]),
        "lambda_effective": torch.tensor([0.3, 0.31]),
        "proto_logits_pmax_mean": torch.tensor([0.4, 0.42]),
        "proto_logits_entropy_mean": torch.tensor([2.0, 1.9]),
        "selected_class_hist": torch.arange(1.0, _C + 1.0),
        "gate_mean": torch.tensor([0.08, 0.09]),
    }


def _make_pl_module() -> MagicMock:
    """构造 mock LightningModule。"""
    module = MagicMock()
    module.log_dict = MagicMock()
    return module


class TestProtoGuidanceMonitor:
    def test_epoch_output_keys(self) -> None:
        """update 后 epoch 输出 train/proto/* 键齐全并清空。"""
        monitor = ProtoGuidanceMonitor(
            class_names=[f"c{c}" for c in range(_C)],
            watch_classes=[0, 1],
        )
        monitor.update(_make_stats())
        pl = _make_pl_module()
        monitor.on_train_epoch_end(pl)
        metrics = pl.log_dict.call_args.args[0]
        expected_prefix = {
            "train/proto/topk_overlap",
            "train/proto/proto_selected_ratio",
            "train/proto/lambda_effective",
            "train/proto/proto_logits_pmax_mean",
            "train/proto/proto_logits_entropy_mean",
            "train/proto/gate_mean",
            "train/proto/selected_ratio/c0",
            "train/proto/selected_ratio/c1",
        }
        assert expected_prefix.issubset(metrics.keys())
        # 累加器已清空
        assert monitor._step_count == 0
        assert not monitor._acc

    def test_empty_accumulator_is_safe(self) -> None:
        """空累加器 epoch 输出不抛错且不产生指标。"""
        monitor = ProtoGuidanceMonitor(class_names=[f"c{c}" for c in range(_C)])
        pl = _make_pl_module()
        monitor.on_train_epoch_end(pl)
        pl.log_dict.assert_not_called()

    def test_grad_norms_and_selected_ratio(self) -> None:
        """梯度范数输出与选中占比按 watch 类输出。"""
        monitor = ProtoGuidanceMonitor(
            class_names=[f"c{c}" for c in range(_C)],
            watch_classes=[2],
        )
        monitor.update(_make_stats())
        monitor.update_grad_norms({"fusion_projectors_proj_token_0_weight": 0.5})
        pl = _make_pl_module()
        monitor.on_train_epoch_end(pl)
        metrics = pl.log_dict.call_args.args[0]
        assert "train/proto/grad_fusion_projectors_proj_token_0_weight" in metrics
        assert "train/proto/selected_ratio/c2" in metrics
        # c2 的选中占比 = hist[2] / hist.sum() = 3 / 15
        assert float(metrics["train/proto/selected_ratio/c2"]) == pytest.approx(3.0 / 15.0)

    def test_multiple_updates_averaged(self) -> None:
        """多次 update 后指标为样本均值。"""
        monitor = ProtoGuidanceMonitor(class_names=[f"c{c}" for c in range(_C)])
        monitor.update(_make_stats())
        monitor.update(_make_stats())
        pl = _make_pl_module()
        monitor.on_train_epoch_end(pl)
        metrics = pl.log_dict.call_args.args[0]
        assert float(metrics["train/proto/topk_overlap"]) == pytest.approx(0.96)
