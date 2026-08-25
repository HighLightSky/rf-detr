# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np

from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds


def _class_data(*, tp: int, fp: int, total_gt: int) -> dict[str, object]:
    """构造单类别置信度扫描数据。"""
    return {
        "scores": np.ones(tp + fp, dtype=np.float32),
        "matches": np.asarray([1] * tp + [0] * fp, dtype=np.int64),
        "ignore": np.zeros(tp + fp, dtype=np.bool_),
        "total_gt": total_gt,
    }


def test_grouped_macro_f1_averages_classes_then_groups() -> None:
    """分组 F1 应先求组内类别平均，再求组间平均。"""
    results = sweep_confidence_thresholds(
        [
            _class_data(tp=1, fp=0, total_gt=1),  # F1=1.0，组 A
            _class_data(tp=1, fp=0, total_gt=2),  # F1=2/3，组 A
            _class_data(tp=1, fp=0, total_gt=2),  # F1=2/3，组 B
        ],
        [0.5],
        [0, 1, 2],
        class_ids=[0, 1, 2],
        class_groups={"group_a": [0, 1], "group_b": [2]},
    )

    result = results[0]
    expected = (((1.0 + 2.0 / 3.0) / 2.0) + (2.0 / 3.0)) / 2.0
    assert result["macro_f1"] == expected
    assert result["group_f1"] == {"group_a": (1.0 + 2.0 / 3.0) / 2.0, "group_b": 2.0 / 3.0}


def test_f1_sweep_without_groups_keeps_class_macro_behavior() -> None:
    """未配置分组时仍按有标注类别直接求 macro-F1。"""
    results = sweep_confidence_thresholds(
        [_class_data(tp=1, fp=0, total_gt=1), _class_data(tp=0, fp=1, total_gt=0)],
        [0.5],
        [0],
    )

    assert results[0]["macro_f1"] == 1.0
