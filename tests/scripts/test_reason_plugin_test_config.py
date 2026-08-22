# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试评测流程中的 FFT 一致性插件配置与转发测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_lib import ReasonPluginCfg, _rescore_reason_candidates
from scripts.large_cut.large_cut_pipeline import infer_detector_on_crops


class _FakeReasonPlugin:
    """记录批量评测侧插件调用，并对指定类别加固定分数。"""

    def __init__(self, class_ids: tuple[int, ...] | None) -> None:
        self.config = SimpleNamespace(reason_class_ids=class_ids)
        self.calls: list[tuple[float, tuple[int, ...], bool]] = []

    def predict_detections(
        self,
        *,
        candidate_boxes: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_classes: np.ndarray,
        target_conf: float,
        reason_class_ids: tuple[int, ...],
        filter_final: bool,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """模拟只修改本次分组的类别分数。"""
        self.calls.append((target_conf, reason_class_ids, filter_final))
        scores = candidate_scores.copy()
        scores[np.isin(candidate_classes, reason_class_ids)] += 0.1
        return candidate_boxes, scores, candidate_classes


def test_reason_plugin_test_config_is_disabled_by_default() -> None:
    """缺省或 enabled=false 的测试配置不创建插件运行时配置。"""
    assert ReasonPluginCfg.from_config(None) is None
    assert ReasonPluginCfg.from_config({"enabled": False}) is None


def test_reason_plugin_test_config_parses_enabled_values() -> None:
    """启用后的测试配置转换为不可变运行时配置。"""
    cfg = ReasonPluginCfg.from_config(
        {
            "enabled": True,
            "checkpoint": "/tmp/reason_plugin.pth",
            "class_ids": [24, "3"],
            "conf_low": "0.05",
        }
    )
    assert cfg == ReasonPluginCfg(
        checkpoint="/tmp/reason_plugin.pth",
        class_ids=(24, 3),
        conf_low=0.05,
    )


def test_reason_plugin_test_config_requires_checkpoint() -> None:
    """开启插件但遗漏 checkpoint 时拒绝启动评测。"""
    with pytest.raises(ValueError, match="checkpoint"):
        ReasonPluginCfg.from_config({"enabled": True})


def test_rescore_reason_candidates_respects_per_class_thresholds() -> None:
    """不同逐类阈值的目标类别分组重打分，非目标类别保持不变。"""
    plugin = _FakeReasonPlugin((24, 3))
    boxes = np.array([[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15]], dtype=np.float32)
    scores = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    class_ids = np.array([24, 3, 8], dtype=np.int64)

    out_boxes, out_scores, out_class_ids = _rescore_reason_candidates(
        plugin,
        torch.empty((25, 256)),
        [str(index) for index in range(25)],
        np.zeros((16, 16, 3), dtype=np.uint8),
        boxes,
        scores,
        class_ids,
        conf_threshold=0.25,
        class_conf_thresholds={3: 0.4},
        device="cpu",
    )

    assert np.array_equal(out_boxes, boxes)
    assert np.array_equal(out_class_ids, class_ids)
    assert out_scores == pytest.approx([0.3, 0.3, 0.2])
    assert set(plugin.calls) == {(0.25, (24,), False), (0.4, (3,), False)}


def test_large_crop_inference_forwards_reason_plugin() -> None:
    """大图裁窗检测启用时向模型推理接口传递插件参数。"""
    captured: dict[str, object] = {}

    class _Model:
        def predict(self, _images: list[np.ndarray], **kwargs: object) -> list[object]:
            """记录调用参数。"""
            captured.update(kwargs)
            return []

    plugin = object()
    result = infer_detector_on_crops(
        _Model(),
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        0.25,
        reason_plugin=plugin,
        reason_class_ids=(24,),
        reason_conf_low=0.05,
    )

    assert result == []
    assert captured == {
        "threshold": 0.25,
        "include_source_image": False,
        "reason_plugin": plugin,
        "reason_class_ids": (24,),
        "reason_conf_low": 0.05,
    }
