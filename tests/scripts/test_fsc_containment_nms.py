# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""发射车一级包含感知 NMS 的配置与几何测试。"""

from __future__ import annotations

import pytest

from scripts.fsc_containment_nms import FscContainmentNmsConfig, apply_fsc_containment_nms
from val.competition_metrics import BoxRecord


def _record(
    class_id: int,
    box: tuple[float, float, float, float],
    score: float,
) -> BoxRecord:
    """构造单张图像的预测记录。"""
    return BoxRecord("image", class_id, box, score)


def test_config_defaults_to_disabled() -> None:
    """未配置时一级 FSC NMS 应保持关闭。"""
    config = FscContainmentNmsConfig.from_config(None)

    assert config == FscContainmentNmsConfig()


def test_config_parses_all_thresholds() -> None:
    """顶层配置应解析完整的一级候选去重规则。"""
    config = FscContainmentNmsConfig.from_config(
        {
            "enabled": True,
            "containment_enabled": True,
            "iou_threshold": 0.4,
            "containment_threshold": 0.95,
            "center_ratio_threshold": 0.35,
        }
    )

    assert config.enabled is True
    assert config.containment_enabled is True
    assert config.iou_threshold == 0.4
    assert config.containment_threshold == 0.95
    assert config.center_ratio_threshold == 0.35


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "iou_threshold": 0.0},
        {"enabled": True, "containment_threshold": 1.1},
        {"enabled": True, "center_ratio_threshold": -0.1},
        {"enabled": True, "unknown": 1},
    ],
)
def test_config_rejects_invalid_or_unknown_values(value: dict[str, object]) -> None:
    """非法阈值和拼写错误应在解析 YAML 时失败。"""
    with pytest.raises(ValueError):
        FscContainmentNmsConfig.from_config(value)


def test_disabled_config_keeps_records_unchanged() -> None:
    """关闭时不得修改一级检测结果。"""
    records = [_record(24, (0.0, 0.0, 10.0, 10.0), 0.9)]

    output, stats = apply_fsc_containment_nms(records, FscContainmentNmsConfig())

    assert output == records
    assert stats.to_dict() == {
        "input_count": 1,
        "output_count": 1,
        "iou_suppressed": 0,
        "containment_suppressed": 0,
    }


def test_containment_suppresses_only_fsc_low_iou_duplicate() -> None:
    """高分大框包含低分 FSC 小框时应抑制小框。"""
    records = [
        _record(24, (10.0, 10.0, 50.0, 50.0), 0.90),
        _record(24, (20.0, 20.0, 40.0, 40.0), 0.80),
        _record(3, (20.0, 20.0, 40.0, 40.0), 0.70),
    ]
    config = FscContainmentNmsConfig(enabled=True, containment_enabled=True, iou_threshold=0.4)

    output, stats = apply_fsc_containment_nms(records, config)

    assert output == [records[0], records[2]]
    assert stats.containment_suppressed == 1
    assert stats.iou_suppressed == 0


def test_far_center_contained_fsc_box_is_preserved() -> None:
    """中心偏移明显的包含框可能是相邻目标，必须保留。"""
    records = [
        _record(24, (0.0, 0.0, 100.0, 100.0), 0.90),
        _record(24, (0.0, 0.0, 10.0, 10.0), 0.80),
    ]

    output, stats = apply_fsc_containment_nms(records, FscContainmentNmsConfig(enabled=True))

    assert output == records
    assert stats.containment_suppressed == 0


def test_iou_duplicate_is_suppressed_before_containment_check() -> None:
    """高 IoU 的 FSC 重复框应按一级分数只保留最高分框。"""
    records = [
        _record(24, (0.0, 0.0, 20.0, 20.0), 0.90),
        _record(24, (1.0, 1.0, 21.0, 21.0), 0.80),
    ]

    output, stats = apply_fsc_containment_nms(records, FscContainmentNmsConfig(enabled=True))

    assert output == records[:1]
    assert stats.iou_suppressed == 1
    assert stats.containment_suppressed == 0
