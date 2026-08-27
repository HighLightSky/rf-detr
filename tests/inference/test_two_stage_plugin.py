# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""通用二阶段复核插件测试。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfdetr.refinement import TwoStageConfig, TwoStagePlugin, TwoStagePluginLoader
from scripts.eval_lib import InferenceCfg, _two_stage_collection_thresholds
from val.competition_metrics import BoxRecord


def test_two_stage_config_disabled_and_enabled() -> None:
    """关闭配置返回空值，启用配置解析所有公共字段。"""
    assert TwoStageConfig.from_config({"enabled": False}) is None
    config = TwoStageConfig.from_config(
        {
            "enabled": True,
            "backend": "resnet18",
            "checkpoint": "output/verifier.pth",
            "class_ids": [24, 25],
            "candidate_floor": 0.05,
            "context_scale": 2.0,
            "image_size": 224,
            "batch_size": 8,
        }
    )
    assert config is not None
    assert config.class_ids == (24, 25)
    assert config.batch_size == 8


def test_two_stage_config_uses_default_probability_gate() -> None:
    """未配置时正类概率阈值应保持 argmax 的等价边界。"""
    config = TwoStageConfig.from_config({"enabled": True, "checkpoint": "output/verifier.pth"})
    assert config is not None
    assert config.positive_threshold == 0.5
    assert config.bypass_score is None


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "checkpoint": "x", "backend": "unknown"},
        {"enabled": True, "checkpoint": "x", "candidate_floor": 1.1},
        {"enabled": True, "checkpoint": "x", "positive_threshold": 1.1},
        {"enabled": True, "checkpoint": "x", "bypass_score": -0.1},
        {"enabled": True, "checkpoint": "x", "detector_score_weight": 1.1},
        {"enabled": True, "checkpoint": "x", "class_ids": []},
    ],
)
def test_two_stage_config_rejects_invalid_values(value: dict[str, object]) -> None:
    """非法 backend、阈值和类别配置应立即报错。"""
    with pytest.raises(ValueError):
        TwoStageConfig.from_config(value)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_nms_iou",
        "candidate_containment_nms_enabled",
        "candidate_nms_containment",
        "candidate_nms_center_ratio",
    ],
)
def test_two_stage_config_rejects_migrated_nms_fields(field: str) -> None:
    """一级候选去重字段迁移后不得继续出现在二阶段配置中。"""
    with pytest.raises(ValueError, match="fsc_containment_nms"):
        TwoStageConfig.from_config({"enabled": False, field: 0.5})


def test_two_stage_loader_requires_checkpoint(tmp_path: Path) -> None:
    """启用插件时 checkpoint 不存在应直接失败。"""
    config = TwoStageConfig(enabled=True, backend="resnet18", checkpoint=tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError):
        TwoStagePluginLoader.load(config)


def test_two_stage_collection_floor_precedes_final_class_threshold() -> None:
    """目标类候选阈值应降低到 floor，最终阈值仍单独保留。"""
    config = TwoStageConfig(enabled=True, backend="resnet18", checkpoint="unused")
    thresholds = _two_stage_collection_thresholds(
        InferenceCfg(conf_threshold=0.25, class_conf_thresholds={24: 0.35}), config
    )
    assert thresholds == {24: 0.05}


def test_two_stage_collection_floor_cannot_exceed_final_threshold() -> None:
    """候选 floor 高于最终阈值时应拒绝可能造成的隐式漏检。"""
    config = TwoStageConfig(enabled=True, backend="resnet18", checkpoint="unused", candidate_floor=0.4)
    with pytest.raises(ValueError, match="不能高于"):
        _two_stage_collection_thresholds(InferenceCfg(conf_threshold=0.25), config)


class _FakeTwoStagePlugin(TwoStagePlugin):
    """只用固定结果模拟二阶段分类器。"""

    def __init__(self, decisions: list[bool], **config_values: object) -> None:
        """初始化固定分类结果。"""
        values: dict[str, object] = {
            "enabled": True,
            "backend": "resnet18",
            "checkpoint": "unused",
            "candidate_floor": 0.05,
            "batch_size": 8,
        }
        values.update(config_values)
        self.config = TwoStageConfig(**values)
        self._decisions = decisions

    def _predict(self, images: list[Image.Image], boxes: list[BoxRecord]) -> np.ndarray:
        """返回预设的保留/拒绝结果。"""
        return np.asarray(self._decisions[: len(boxes)], dtype=bool)


class _ScoredFakeTwoStagePlugin(TwoStagePlugin):
    """返回固定二阶段 FSC 概率的测试插件。"""

    def __init__(self, probabilities: list[float]) -> None:
        """初始化固定二阶段概率。"""
        self.config = TwoStageConfig(
            enabled=True,
            backend="resnet18",
            checkpoint="unused",
            candidate_floor=0.05,
            batch_size=8,
        )
        self._probabilities = probabilities

    def _predict(self, images: list[Image.Image], boxes: list[BoxRecord]) -> np.ndarray:
        """返回预设的二阶段 FSC 概率。"""
        return np.asarray(self._probabilities[: len(boxes)], dtype=np.float32)


def test_two_stage_refine_keeps_other_classes_without_candidate_nms(tmp_path: Path) -> None:
    """二阶段只负责分类，不在内部去重一级候选。"""
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(image_path)
    records = [
        BoxRecord("image", 24, (10.0, 10.0, 40.0, 40.0), 0.30),
        BoxRecord("image", 24, (11.0, 11.0, 39.0, 39.0), 0.20),
        BoxRecord("image", 24, (60.0, 60.0, 80.0, 80.0), 0.04),
        BoxRecord("image", 3, (1.0, 1.0, 8.0, 8.0), 0.10),
    ]
    plugin = _FakeTwoStagePlugin([True, True])
    output, stats = plugin.refine_records(records, {"image": image_path})
    assert len(output) == 3
    assert output[0].image_id == records[0].image_id
    assert output[0].class_id == records[0].class_id
    assert output[0].xyxy == records[0].xyxy
    assert output[0].score == 1.0
    assert output[1].xyxy == records[1].xyxy
    assert output[2] == records[3]
    assert stats.routed == 2
    assert stats.kept == 2
    assert stats.rejected == 0


def test_two_stage_refine_rejects_target_candidate(tmp_path: Path) -> None:
    """二阶段拒绝目标候选时不影响其他类别框。"""
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_path)
    target = BoxRecord("image", 24, (2.0, 2.0, 20.0, 20.0), 0.10)
    other = BoxRecord("image", 2, (1.0, 1.0, 10.0, 10.0), 0.10)
    output, stats = _FakeTwoStagePlugin([False]).refine_records([target, other], {"image": image_path})
    assert output == [other]
    assert stats.routed == 1
    assert stats.kept == 0
    assert stats.rejected == 1


def test_two_stage_refine_replaces_accepted_target_score_with_second_stage_probability(tmp_path: Path) -> None:
    """二阶段接受低分候选时，最终分数应来自二阶段而非一级。"""
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_path)
    target = BoxRecord("image", 24, (2.0, 2.0, 20.0, 20.0), 0.08)

    output, stats = _ScoredFakeTwoStagePlugin([0.76]).refine_records([target], {"image": image_path})

    assert len(output) == 1
    assert output[0].image_id == target.image_id
    assert output[0].class_id == target.class_id
    assert output[0].xyxy == target.xyxy
    assert output[0].score == pytest.approx(0.76)
    assert stats.kept == 1


def test_two_stage_refine_empty_candidates() -> None:
    """没有目标类别候选时不执行分类并返回空统计。"""
    output, stats = _FakeTwoStagePlugin([]).refine_records([], {})
    assert output == []
    assert stats.routed == 0
    assert stats.images == 0
