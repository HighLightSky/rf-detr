# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MS 保守 NMS 的几何与配置测试。"""

from __future__ import annotations

import pytest

from scripts.ms_nms import MsNmsConfig, apply_shwx_ms_nms
from val.competition_metrics import BoxRecord


def _record(class_id: int, box: tuple[float, float, float, float], score: float) -> BoxRecord:
    """构造单张图像的预测记录。"""
    return BoxRecord("image", class_id, box, score)


def test_default_config_keeps_records_unchanged() -> None:
    """默认配置关闭时不应修改通用推理输出。"""
    records = [_record(3, (0.0, 0.0, 10.0, 10.0), 0.9)]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig())
    assert output == records
    assert stats.to_dict() == {
        "input_count": 1,
        "output_count": 1,
        "same_class_suppressed": 0,
        "cross_class_suppressed": 0,
        "ambiguous_cross_class_kept": 0,
    }


def test_same_class_ms_duplicate_keeps_high_score() -> None:
    """高度重叠的 MS 重复框只保留高分框。"""
    records = [
        _record(3, (0.0, 0.0, 100.0, 100.0), 0.90),
        _record(3, (2.0, 2.0, 98.0, 98.0), 0.80),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records[:1]
    assert stats.same_class_suppressed == 1


def test_nearby_ms_targets_are_preserved() -> None:
    """中心距离过大的相邻 MS 目标不能被合并。"""
    records = [
        _record(3, (0.0, 0.0, 40.0, 100.0), 0.90),
        _record(3, (35.0, 0.0, 75.0, 100.0), 0.80),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records
    assert stats.same_class_suppressed == 0


def test_contained_ms_box_with_shifted_center_is_preserved() -> None:
    """包含率很高但中心偏移过大的框不能被删除。"""
    records = [
        _record(3, (0.0, 0.0, 100.0, 100.0), 0.90),
        _record(3, (20.0, 0.0, 70.0, 100.0), 0.80),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records
    assert stats.same_class_suppressed == 0


def test_cross_class_strong_overlap_keeps_higher_score() -> None:
    """MS 与其他船类强重叠且分数有差距时保留高分类别。"""
    records = [
        _record(3, (0.0, 0.0, 100.0, 100.0), 0.95),
        _record(1, (1.0, 1.0, 99.0, 99.0), 0.70),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records[:1]
    assert stats.cross_class_suppressed == 1


def test_cross_class_score_tie_is_kept_and_logged() -> None:
    """跨类别分数接近时保留两个框，保护召回。"""
    records = [
        _record(3, (0.0, 0.0, 100.0, 100.0), 0.90),
        _record(1, (1.0, 1.0, 99.0, 99.0), 0.88),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records
    assert stats.ambiguous_cross_class_kept == 1


def test_non_ship_cross_class_overlap_is_untouched() -> None:
    """MS 与飞机或发射车重叠时不执行跨类别抑制。"""
    records = [
        _record(3, (0.0, 0.0, 100.0, 100.0), 0.95),
        _record(4, (1.0, 1.0, 99.0, 99.0), 0.70),
        _record(24, (2.0, 2.0, 98.0, 98.0), 0.60),
    ]
    output, stats = apply_shwx_ms_nms(records, MsNmsConfig(enabled=True))
    assert output == records
    assert stats.cross_class_suppressed == 0


def test_empty_input_and_original_order_are_deterministic() -> None:
    """空输入安全返回，非重叠输入保持原始顺序。"""
    config = MsNmsConfig(enabled=True)
    output, stats = apply_shwx_ms_nms([], config)
    assert output == []
    assert stats.input_count == 0

    records = [
        _record(1, (200.0, 0.0, 210.0, 10.0), 0.6),
        _record(3, (0.0, 0.0, 10.0, 10.0), 0.9),
    ]
    output, _ = apply_shwx_ms_nms(records, config)
    assert output == records


def test_config_rejects_unknown_fields() -> None:
    """拼写错误的配置项必须显式报错。"""
    with pytest.raises(ValueError, match="未知配置项"):
        MsNmsConfig.from_config({"enabled": True, "same_class_iou_typo": 0.8})
