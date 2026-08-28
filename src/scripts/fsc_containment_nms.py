# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""发射车一级候选的包含感知 NMS。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from val.competition_metrics import BoxRecord

_FSC_CLASS_ID = 24


@dataclass(frozen=True)
class FscContainmentNmsConfig:
    """发射车一级候选框的 IoU 与包含感知去重配置。"""

    enabled: bool = False
    containment_enabled: bool = False
    iou_threshold: float = 0.5
    containment_threshold: float = 0.95
    center_ratio_threshold: float = 0.35

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "FscContainmentNmsConfig":
        """从 YAML 映射解析配置。"""
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("fsc_containment_nms 必须是字典配置")
        values = dict(config)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"fsc_containment_nms 存在未知配置项: {', '.join(sorted(unknown))}")
        try:
            parsed = cls(
                enabled=bool(values.get("enabled", False)),
                containment_enabled=bool(values.get("containment_enabled", False)),
                iou_threshold=float(values.get("iou_threshold", 0.5)),
                containment_threshold=float(values.get("containment_threshold", 0.95)),
                center_ratio_threshold=float(values.get("center_ratio_threshold", 0.35)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("fsc_containment_nms 阈值必须是数值") from exc
        parsed.validate()
        return parsed

    def validate(self) -> None:
        """校验去重阈值范围。"""
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("fsc_containment_nms.iou_threshold 必须位于 (0, 1]")
        if not 0.0 <= self.containment_threshold <= 1.0:
            raise ValueError("fsc_containment_nms.containment_threshold 必须位于 [0, 1]")
        if self.center_ratio_threshold < 0.0:
            raise ValueError("fsc_containment_nms.center_ratio_threshold 不能为负数")


@dataclass(frozen=True)
class FscContainmentNmsStats:
    """一级 FSC NMS 的输入输出和抑制统计。"""

    input_count: int = 0
    output_count: int = 0
    iou_suppressed: int = 0
    containment_suppressed: int = 0

    def to_dict(self) -> dict[str, int]:
        """转换为可序列化字典。"""
        return asdict(self)


def _overlap_metrics(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """计算 IoU、较小框包含率和中心距离归一化值。"""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_width = max(0.0, first[2] - first[0])
    first_height = max(0.0, first[3] - first[1])
    second_width = max(0.0, second[2] - second[0])
    second_height = max(0.0, second[3] - second[1])
    first_area = first_width * first_height
    second_area = second_width * second_height
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    smaller_area = min(first_area, second_area)
    containment = intersection / smaller_area if smaller_area > 0.0 else 0.0
    first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
    second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
    center_distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    smaller_diagonal = min((first_width**2 + first_height**2) ** 0.5, (second_width**2 + second_height**2) ** 0.5)
    center_ratio = center_distance / smaller_diagonal if smaller_diagonal > 0.0 else float("inf")
    return iou, containment, center_ratio


def apply_fsc_containment_nms(
    records: Sequence[BoxRecord],
    config: FscContainmentNmsConfig,
) -> tuple[list[BoxRecord], FscContainmentNmsStats]:
    """只对同图的发射车一级候选执行保守去重。"""
    if not config.enabled or not records:
        return list(records), FscContainmentNmsStats(input_count=len(records), output_count=len(records))

    keep = [True] * len(records)
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.class_id == _FSC_CLASS_ID:
            by_image[record.image_id].append(index)

    iou_suppressed = 0
    containment_suppressed = 0
    for image_indices in by_image.values():
        selected: list[int] = []
        for candidate_index in sorted(image_indices, key=lambda index: (-float(records[index].score or 0.0), index)):
            reason: str | None = None
            for selected_index in selected:
                iou, containment, center_ratio = _overlap_metrics(
                    records[candidate_index].xyxy,
                    records[selected_index].xyxy,
                )
                if center_ratio > config.center_ratio_threshold:
                    continue
                if iou > config.iou_threshold:
                    reason = "iou"
                    break
                if config.containment_enabled and containment >= config.containment_threshold:
                    reason = "containment"
                    break
            if reason is None:
                selected.append(candidate_index)
            else:
                keep[candidate_index] = False
                if reason == "iou":
                    iou_suppressed += 1
                else:
                    containment_suppressed += 1

    output = [record for index, record in enumerate(records) if keep[index]]
    return output, FscContainmentNmsStats(
        input_count=len(records),
        output_count=len(output),
        iou_suppressed=iou_suppressed,
        containment_suppressed=containment_suppressed,
    )
