# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""置信度阈值扫描与精确率、召回率、F1 计算。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def sweep_confidence_thresholds(
    per_class_data: list[dict[str, Any]],
    conf_thresholds: Any,
    classes_with_gt: list[int],
    *,
    class_ids: Sequence[int] | None = None,
    class_groups: Mapping[str, Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """扫描置信度阈值并计算精确率、召回率和 F1。

    ``class_groups`` 为空时保持原有行为：对有真实框的类别直接求 macro-F1。
    配置分组后，先对每组内有真实框的类别取平均，再对所有组取平均，得到用于选权的
    ``macro_f1``。组内没有真实框的类别不参与该组平均；没有有效类别的组记为 0。

    Args:
        per_class_data: 按类别排列的匹配数据，
            每项含 ``scores``、``matches``、``ignore`` 和 ``total_gt``。
        conf_thresholds: 待评估的置信度阈值序列。
        classes_with_gt: 至少包含一个真实框的类别在 ``per_class_data`` 中的索引。
        class_ids: ``per_class_data`` 每项对应的真实类别 ID，配置分组时必需。
        class_groups: 分组名到类别 ID 序列的映射。

    Returns:
        每个阈值对应的指标字典；启用分组时额外返回 ``group_f1``。
    """
    if class_groups is not None and class_ids is None:
        raise ValueError("class_ids is required when class_groups is provided")
    if class_ids is not None and len(class_ids) != len(per_class_data):
        raise ValueError("class_ids must have the same length as per_class_data")

    grouped_indices: dict[str, list[int]] = {}
    if class_groups is not None:
        class_id_to_index = {
            class_id: index for index, class_id in enumerate(class_ids if class_ids is not None else ())
        }
        valid_indices = set(classes_with_gt)
        grouped_indices = {
            group_name: [
                class_id_to_index[class_id]
                for class_id in group_class_ids
                if class_id in class_id_to_index and class_id_to_index[class_id] in valid_indices
            ]
            for group_name, group_class_ids in class_groups.items()
        }

    results: list[dict[str, Any]] = []
    for conf_thresh in conf_thresholds:
        per_class_precisions: list[float] = []
        per_class_recalls: list[float] = []
        per_class_f1s: list[float] = []

        for data in per_class_data:
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            valid = (scores >= conf_thresh) & ~ignore
            valid_matches = matches[valid]
            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(float(precision))
            per_class_recalls.append(float(recall))
            per_class_f1s.append(float(f1))

        group_f1: dict[str, float] = {}
        if class_groups is not None:
            group_precision: list[float] = []
            group_recall: list[float] = []
            for group_name, indices in grouped_indices.items():
                if indices:
                    group_f1[group_name] = float(np.mean([per_class_f1s[index] for index in indices]))
                    group_precision.append(float(np.mean([per_class_precisions[index] for index in indices])))
                    group_recall.append(float(np.mean([per_class_recalls[index] for index in indices])))
                else:
                    group_f1[group_name] = 0.0
                    group_precision.append(0.0)
                    group_recall.append(0.0)
            macro_f1 = float(np.mean(list(group_f1.values()))) if group_f1 else 0.0
            macro_precision = float(np.mean(group_precision)) if group_precision else 0.0
            macro_recall = float(np.mean(group_recall)) if group_recall else 0.0
        elif classes_with_gt:
            macro_precision = float(np.mean([per_class_precisions[index] for index in classes_with_gt]))
            macro_recall = float(np.mean([per_class_recalls[index] for index in classes_with_gt]))
            macro_f1 = float(np.mean([per_class_f1s[index] for index in classes_with_gt]))
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
                "group_f1": group_f1,
            }
        )

    return results
