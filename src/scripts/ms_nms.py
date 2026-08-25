# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SHWX 民船候选框的保守去重。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from val.competition_metrics import BoxRecord


@dataclass(frozen=True)
class MsNmsConfig:
    """SHWX MS 专用保守 NMS 配置。"""

    enabled: bool = False
    ms_class_id: int = 3
    ship_class_ids: tuple[int, ...] = (0, 1, 2, 3)
    same_class_iou: float = 0.80
    same_class_containment: float = 0.90
    same_class_center_ratio: float = 0.35
    cross_class_iou: float = 0.90
    cross_class_containment: float = 0.95
    cross_class_center_ratio: float = 0.25
    cross_class_score_margin: float = 0.05
    keep_ambiguous_cross_class: bool = True

    def __post_init__(self) -> None:
        """校验类别和几何阈值，防止配置错误扩大抑制范围。"""
        if self.ms_class_id not in self.ship_class_ids:
            raise ValueError("ms_class_id 必须包含在 ship_class_ids 中")
        if not self.ship_class_ids:
            raise ValueError("ship_class_ids 不能为空")
        for name in (
            "same_class_iou",
            "same_class_containment",
            "cross_class_iou",
            "cross_class_containment",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        for name in ("same_class_center_ratio", "cross_class_center_ratio", "cross_class_score_margin"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} 不能为负数")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> MsNmsConfig:
        """从 YAML 字典解析配置；未配置时保持关闭。"""
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("ms_nms 必须是字典配置")
        values: dict[str, Any] = dict(config)
        if "ship_class_ids" in values:
            raw_ids = values["ship_class_ids"]
            if not isinstance(raw_ids, (list, tuple)):
                raise ValueError("ms_nms.ship_class_ids 必须是整数列表")
            values["ship_class_ids"] = tuple(int(class_id) for class_id in raw_ids)
        for name in (
            "ms_class_id",
            "same_class_iou",
            "same_class_containment",
            "same_class_center_ratio",
            "cross_class_iou",
            "cross_class_containment",
            "cross_class_center_ratio",
            "cross_class_score_margin",
        ):
            if name in values:
                values[name] = int(values[name]) if name == "ms_class_id" else float(values[name])
        if "enabled" in values:
            values["enabled"] = bool(values["enabled"])
        if "keep_ambiguous_cross_class" in values:
            values["keep_ambiguous_cross_class"] = bool(values["keep_ambiguous_cross_class"])
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"ms_nms 存在未知配置项: {', '.join(sorted(unknown))}")
        return cls(**values)


@dataclass(frozen=True)
class SuppressionStats:
    """NMS 输入、输出和冲突统计。"""

    input_count: int = 0
    output_count: int = 0
    same_class_suppressed: int = 0
    cross_class_suppressed: int = 0
    ambiguous_cross_class_kept: int = 0

    def to_dict(self) -> dict[str, int]:
        """转换为 JSON 可序列化字典。"""
        return asdict(self)


def _box_area(box: tuple[float, float, float, float]) -> float:
    """计算合法矩形面积。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_metrics(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """计算 IoU、较小框包含率和中心距离归一化值。"""
    inter_x1 = max(first[0], second[0])
    inter_y1 = max(first[1], second[1])
    inter_x2 = min(first[2], second[2])
    inter_y2 = min(first[3], second[3])
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    first_area = _box_area(first)
    second_area = _box_area(second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    smaller_area = min(first_area, second_area)
    containment = intersection / smaller_area if smaller_area > 0.0 else 0.0
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    smaller_diagonal = min(
        max(0.0, first[2] - first[0]) ** 2 + max(0.0, first[3] - first[1]) ** 2,
        max(0.0, second[2] - second[0]) ** 2 + max(0.0, second[3] - second[1]) ** 2,
    ) ** 0.5
    center_ratio = distance / smaller_diagonal if smaller_diagonal > 0.0 else float("inf")
    return iou, containment, center_ratio


def _is_duplicate(
    first: BoxRecord,
    second: BoxRecord,
    iou_threshold: float,
    containment_threshold: float,
    center_ratio_threshold: float,
) -> bool:
    """判断两个框是否满足保守重复条件。"""
    iou, containment, center_ratio = _overlap_metrics(first.xyxy, second.xyxy)
    return (iou >= iou_threshold or containment >= containment_threshold) and center_ratio <= center_ratio_threshold


def apply_shwx_ms_nms(
    records: Sequence[BoxRecord],
    config: MsNmsConfig,
) -> tuple[list[BoxRecord], SuppressionStats]:
    """对每张图像执行 MS 同类去重和船舶跨类冲突处理。

    Args:
        records: 已完成置信度筛选、坐标还原的检测记录。
        config: MS NMS 配置。

    Returns:
        保持输入顺序的保留记录和抑制统计。
    """
    if not config.enabled or not records:
        stats = SuppressionStats(input_count=len(records), output_count=len(records))
        return list(records), stats

    keep = [True] * len(records)
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_image[record.image_id].append(index)
    same_removed = 0
    cross_removed = 0
    ambiguous_kept = 0

    for image_indices in by_image.values():
        ms_indices = [index for index in image_indices if records[index].class_id == config.ms_class_id]
        ms_indices.sort(key=lambda index: (-float(records[index].score or 0.0), index))
        retained_ms: list[int] = []
        for candidate_index in ms_indices:
            if any(
                _is_duplicate(
                    records[candidate_index],
                    records[retained_index],
                    config.same_class_iou,
                    config.same_class_containment,
                    config.same_class_center_ratio,
                )
                for retained_index in retained_ms
            ):
                keep[candidate_index] = False
                same_removed += 1
            else:
                retained_ms.append(candidate_index)

        active_indices = [index for index in image_indices if keep[index]]
        for position, first_index in enumerate(active_indices):
            if not keep[first_index]:
                continue
            first = records[first_index]
            if first.class_id not in config.ship_class_ids:
                continue
            for second_index in active_indices[position + 1 :]:
                if not keep[second_index]:
                    continue
                second = records[second_index]
                if second.class_id not in config.ship_class_ids or first.class_id == second.class_id:
                    continue
                if config.ms_class_id not in (first.class_id, second.class_id):
                    continue
                if not _is_duplicate(
                    first,
                    second,
                    config.cross_class_iou,
                    config.cross_class_containment,
                    config.cross_class_center_ratio,
                ):
                    continue
                first_score = float(first.score or 0.0)
                second_score = float(second.score or 0.0)
                if abs(first_score - second_score) < config.cross_class_score_margin:
                    ambiguous_kept += 1
                    if not config.keep_ambiguous_cross_class:
                        lower_index = first_index if first_score <= second_score else second_index
                        keep[lower_index] = False
                        cross_removed += 1
                    continue
                lower_index = first_index if first_score < second_score else second_index
                keep[lower_index] = False
                cross_removed += 1
                if lower_index == first_index:
                    break

    kept_records = [record for index, record in enumerate(records) if keep[index]]
    stats = SuppressionStats(
        input_count=len(records),
        output_count=len(kept_records),
        same_class_suppressed=same_removed,
        cross_class_suppressed=cross_removed,
        ambiguous_cross_class_kept=ambiguous_kept,
    )
    return kept_records, stats
