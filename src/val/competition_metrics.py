# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""比赛指标评测工具函数。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class BoxRecord:
    """单个检测框或真实框记录。"""

    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    score: float | None = None


@dataclass(frozen=True)
class EvalConfig:
    """比赛指标评测配置。"""

    class_to_group: Mapping[int, str]
    group_iou_thresholds: Mapping[str, float]
    default_iou_threshold: float = 0.50
    class_aware: bool = True


@dataclass(frozen=True)
class EvalResult:
    """TP/FP/FN 及其派生指标。"""

    tp: int
    fp: int
    fn: int
    recall: float = field(init=False)
    fdr: float = field(init=False)
    precision: float = field(init=False)

    def __post_init__(self):
        """根据 TP/FP/FN 自动计算召回率、虚警率和精确率。"""
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0.0
        fdr = self.fp / (self.fp + self.tp) if self.fp + self.tp > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0.0

        object.__setattr__(self, "recall", recall)
        object.__setattr__(self, "fdr", fdr)
        object.__setattr__(self, "precision", precision)


def xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int | float,
    image_height: int | float,
) -> tuple[float, float, float, float]:
    """将 YOLO 归一化 xywh 坐标转换为像素级 xyxy 坐标。"""
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2
    return x1, y1, x2, y2


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """计算两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 计算交集区域
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 计算并集区域
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_yolo_labels(
    label_dir: str | Path,
    image_size_map: Mapping[str, tuple[int, int]],
) -> list[BoxRecord]:
    """读取 YOLO 真实标注 txt 文件。"""
    return _load_yolo_txt_records(
        txt_dir=label_dir,
        image_size_map=image_size_map,
        with_confidence=False,
    )


def load_yolo_predictions(
    pred_dir: str | Path,
    image_size_map: Mapping[str, tuple[int, int]],
) -> list[BoxRecord]:
    """读取 YOLO 预测结果 txt 文件。"""
    return _load_yolo_txt_records(
        txt_dir=pred_dir,
        image_size_map=image_size_map,
        with_confidence=True,
    )


def evaluate_competition_metrics(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    config: EvalConfig,
) -> dict[str, EvalResult | dict[str, EvalResult]]:
    """按比赛方案计算整体和分组 TP/FP/FN、召回率、虚警率。

    返回:     {"all": EvalResult, "groups": {目标组名: EvalResult}}
    """
    # 将真实标注和预测记录分别按图像 ID 和组名进行分组
    gt_by_image_group = _group_records_by_image_and_group(gt_records, config)
    pred_by_image_group = _group_records_by_image_and_group(pred_records, config)

    # |:合并两者中所有的图像 ID，并自动去重
    all_image_ids = set(gt_by_image_group) | set(pred_by_image_group)

    # 先累计原始计数，最后统一转换为 EvalResult
    all_counts = {"tp": 0, "fp": 0, "fn": 0}
    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for image_id in sorted(all_image_ids):
        groups = set(gt_by_image_group[image_id]) | set(pred_by_image_group[image_id])
        for group_name in sorted(groups):
            gt_group_records = gt_by_image_group[image_id].get(group_name, [])
            pred_group_records = pred_by_image_group[image_id].get(group_name, [])
            iou_threshold = config.group_iou_thresholds.get(
                group_name,
                config.default_iou_threshold,
            )

            tp, fp, fn = _match_single_image_group(
                gt_records=gt_group_records,
                pred_records=pred_group_records,
                iou_threshold=iou_threshold,
                class_aware=config.class_aware,
            )

            # 累计分组结果
            group_counts[group_name]["tp"] += tp
            group_counts[group_name]["fp"] += fp
            group_counts[group_name]["fn"] += fn

            # 累计总体结果
            all_counts["tp"] += tp
            all_counts["fp"] += fp
            all_counts["fn"] += fn

    group_results = {group_name: EvalResult(**counts) for group_name, counts in sorted(group_counts.items())}
    return {
        "all": EvalResult(**all_counts),
        "groups": group_results,
    }


def _load_yolo_txt_records(
    txt_dir: str | Path,
    image_size_map: Mapping[str, tuple[int, int]],
    with_confidence: bool,
) -> list[BoxRecord]:
    """读取 YOLO txt 并转换为 BoxRecord 列表。"""
    txt_dir = Path(txt_dir)
    records: list[BoxRecord] = []

    for txt_path in sorted(txt_dir.glob("*.txt")):
        image_id = txt_path.stem
        if image_id not in image_size_map:
            continue

        image_width, image_height = image_size_map[image_id]
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                min_len = 6 if with_confidence else 5
                if len(parts) < min_len:
                    continue

                # YOLO 标注格式：class xc yc w h [confidence]
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:5])
                score = float(parts[5]) if with_confidence else None
                xyxy = xywhn_to_xyxy(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    image_width=image_width,
                    image_height=image_height,
                )

                records.append(
                    BoxRecord(
                        image_id=image_id,
                        class_id=class_id,
                        xyxy=xyxy,
                        score=score,
                    )
                )

    return records


def _group_records_by_image_and_group(
    records: list[BoxRecord],
    config: EvalConfig,
) -> dict[str, dict[str, list[BoxRecord]]]:
    """按图像 ID 和目标大类组织检测框。"""
    grouped: dict[str, dict[str, list[BoxRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        group_name = config.class_to_group[record.class_id]
        grouped[record.image_id][group_name].append(record)
    return grouped


def _match_single_image_group(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    iou_threshold: float,
    class_aware: bool,
) -> tuple[int, int, int]:
    """对单张图、单个目标组执行置信度排序一对一匹配。"""
    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0

    # 预测框按置信度从高到低依次匹配
    sorted_preds = sorted(
        pred_records,
        key=lambda record: record.score if record.score is not None else 0.0,
        reverse=True,
    )

    for pred in sorted_preds:
        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt in enumerate(gt_records):
            # 每个真实框最多被一个预测框匹配
            if gt_index in matched_gt_indices:
                continue
            if class_aware and pred.class_id != gt.class_id:
                continue

            iou = compute_iou(pred.xyxy, gt.xyxy)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is None:
            # 未匹配真实目标的预测框
            fp += 1
        else:
            tp += 1
            matched_gt_indices.add(best_gt_index)

    fn = len(gt_records) - len(matched_gt_indices)
    return tp, fp, fn
