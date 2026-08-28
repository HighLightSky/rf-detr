"""SHWX 民船与发射车候选去重。"""

from __future__ import annotations

from collections import defaultdict

from competition.config import FscNmsConfig, MsNmsConfig
from competition.contracts import RawDetection

FSC_CLASS_ID = 24


def overlap_metrics(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """计算 IoU、较小框包含率和中心距离归一化值。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_width, first_height = max(0.0, first[2] - first[0]), max(0.0, first[3] - first[1])
    second_width, second_height = max(0.0, second[2] - second[0]), max(0.0, second[3] - second[1])
    first_area, second_area = first_width * first_height, second_width * second_height
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    smaller_area = min(first_area, second_area)
    containment = intersection / smaller_area if smaller_area > 0.0 else 0.0
    first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
    second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
    distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    smaller_diagonal = min(
        (first_width**2 + first_height**2) ** 0.5,
        (second_width**2 + second_height**2) ** 0.5,
    )
    center_ratio = distance / smaller_diagonal if smaller_diagonal > 0.0 else float("inf")
    return iou, containment, center_ratio


def _duplicate(
    first: RawDetection,
    second: RawDetection,
    iou_threshold: float,
    containment_threshold: float,
    center_ratio_threshold: float,
) -> bool:
    """判断两框是否满足保守重复条件。"""
    iou, containment, center_ratio = overlap_metrics(first.xyxy, second.xyxy)
    return (iou >= iou_threshold or containment >= containment_threshold) and center_ratio <= center_ratio_threshold


def apply_fsc_containment_nms(records: list[RawDetection], config: FscNmsConfig) -> list[RawDetection]:
    """仅对同图发射车一级候选执行 IoU 与可选包含感知去重。"""
    if not config.enabled:
        return records
    kept = [True] * len(records)
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.class_id == FSC_CLASS_ID:
            by_image[record.image_id].append(index)
    for indices in by_image.values():
        selected: list[int] = []
        for candidate_index in sorted(indices, key=lambda index: (-records[index].score, index)):
            candidate = records[candidate_index]
            duplicate = False
            for selected_index in selected:
                iou, containment, center_ratio = overlap_metrics(candidate.xyxy, records[selected_index].xyxy)
                if center_ratio > config.center_ratio_threshold:
                    continue
                if iou > config.iou_threshold or (
                    config.containment_enabled and containment >= config.containment_threshold
                ):
                    duplicate = True
                    break
            if duplicate:
                kept[candidate_index] = False
            else:
                selected.append(candidate_index)
    return [record for index, record in enumerate(records) if kept[index]]


def apply_ms_nms(records: list[RawDetection], config: MsNmsConfig) -> list[RawDetection]:
    """复现训练评估中的民船同类和跨类保守去重策略。"""
    if not config.enabled:
        return records
    kept = [True] * len(records)
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_image[record.image_id].append(index)
    for indices in by_image.values():
        ms_indices = [index for index in indices if records[index].class_id == config.ms_class_id]
        selected_ms: list[int] = []
        for candidate_index in sorted(ms_indices, key=lambda index: (-records[index].score, index)):
            candidate = records[candidate_index]
            if any(
                _duplicate(
                    candidate,
                    records[selected_index],
                    config.same_class_iou,
                    config.same_class_containment,
                    config.same_class_center_ratio,
                )
                for selected_index in selected_ms
            ):
                kept[candidate_index] = False
            else:
                selected_ms.append(candidate_index)

        active_indices = [index for index in indices if kept[index]]
        for position, first_index in enumerate(active_indices):
            if not kept[first_index] or records[first_index].class_id not in config.ship_class_ids:
                continue
            first = records[first_index]
            for second_index in active_indices[position + 1 :]:
                if not kept[second_index]:
                    continue
                second = records[second_index]
                if second.class_id not in config.ship_class_ids or first.class_id == second.class_id:
                    continue
                if config.ms_class_id not in (first.class_id, second.class_id):
                    continue
                if not _duplicate(
                    first,
                    second,
                    config.cross_class_iou,
                    config.cross_class_containment,
                    config.cross_class_center_ratio,
                ):
                    continue
                if abs(first.score - second.score) < config.cross_class_score_margin:
                    if not config.keep_ambiguous_cross_class:
                        lower = first_index if first.score <= second.score else second_index
                        kept[lower] = False
                    continue
                lower = first_index if first.score < second.score else second_index
                kept[lower] = False
                if lower == first_index:
                    break
    return [record for index, record in enumerate(records) if kept[index]]
