# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""推理阶段 FFT 重打分插件的集成测试。"""


import numpy as np
import PIL.Image
import pytest
import torch

from rfdetr.reasoning import ReasonConfig

from .helpers import _DummyModel, _DummyRFDETR


class _LowScoreModel(_DummyModel):
    """返回一个低置信检测，并提供插件所需的类别嵌入。"""

    def __init__(self) -> None:
        super().__init__(class_names=[f"class_{i}" for i in range(24)] + ["FSC"], labels=[24])
        self.model.class_embed = torch.nn.Linear(2, 25)

    def postprocess(self, predictions: object, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """为每张图返回一个 FSC 低置信候选。"""
        return [
            {
                "scores": torch.tensor([0.1]),
                "labels": torch.tensor([24]),
                "boxes": torch.tensor([[2.0, 2.0, 18.0, 18.0]]),
            }
            for _ in range(target_sizes.shape[0])
        ]


class _FakeReasonPlugin:
    """记录插件输入并把候选框提升到最终阈值之上。"""

    def __init__(self) -> None:
        self.config = ReasonConfig()
        self.calls: list[dict[str, object]] = []

    def to(self, device: torch.device) -> "_FakeReasonPlugin":
        """模拟真实插件的设备迁移接口。"""
        return self

    def predict_detections(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """记录单图输入并返回一条提升后的候选。"""
        self.calls.append(kwargs)
        boxes = np.asarray(kwargs["candidate_boxes"], dtype=np.float32)
        classes = np.asarray(kwargs["candidate_classes"], dtype=np.int64)
        return boxes, np.asarray([0.8], dtype=np.float32), classes


def test_predict_applies_reason_plugin_before_final_threshold() -> None:
    """插件应在最终阈值前看到源图，并能把低置信 FSC 候选加入输出。"""
    model = _DummyRFDETR()
    model.model = _LowScoreModel()
    plugin = _FakeReasonPlugin()

    detections = model.predict(
        PIL.Image.new("RGB", (32, 32), color=(128, 128, 128)),
        threshold=0.5,
        include_source_image=False,
        reason_plugin=plugin,
    )

    assert len(detections) == 1
    assert detections.class_id.tolist() == [24]
    np.testing.assert_allclose(detections.confidence, [0.8])
    assert "source_image" not in detections.metadata
    assert len(plugin.calls) == 1
    assert isinstance(plugin.calls[0]["source_image"], np.ndarray)
    assert plugin.config.reason_class_ids == (24,)


def test_predict_reason_plugin_rejects_optimized_model() -> None:
    """插件需要原始 class_embed，不能运行在已清理基础模型的优化路径上。"""
    model = _DummyRFDETR()
    model.model = _LowScoreModel()
    model._is_optimized_for_inference = True

    with pytest.raises(RuntimeError, match="unoptimized"):
        model.predict(PIL.Image.new("RGB", (32, 32)), reason_plugin=_FakeReasonPlugin())
