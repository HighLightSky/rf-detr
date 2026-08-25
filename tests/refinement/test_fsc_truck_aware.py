# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试 truck 竞争类别的 FSC 级联逻辑。"""

import numpy as np

from scripts.refinement.predict_fsc_truck_aware import _suppress_truck_conflicts


def test_truck_conflict_suppresses_lower_scored_fsc() -> None:
    """同框 truck 分数更高时应移除 FSC。"""
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.asarray([0.4, 0.6], dtype=np.float32)
    classes = np.asarray([24, 25], dtype=np.int64)
    np.testing.assert_array_equal(_suppress_truck_conflicts(boxes, scores, classes), [False, True])


def test_truck_conflict_keeps_higher_scored_fsc() -> None:
    """同框 FSC 分数更高时应保留 FSC。"""
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.asarray([0.7, 0.6], dtype=np.float32)
    classes = np.asarray([24, 25], dtype=np.int64)
    np.testing.assert_array_equal(_suppress_truck_conflicts(boxes, scores, classes), [True, True])
