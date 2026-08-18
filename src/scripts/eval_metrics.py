# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""用比赛评分方案评估 RF-DETR checkpoint 的检测结果。

真实框直接从 SHWX 数据集原生 YOLO 格式读取（``data.yaml`` +
``labels/{split}/*.txt`` + ``images/{split}/*.jpg``）。

比赛判定口径（补充说明确认）：
    - 预测框与真值 IoU >= τ 且小类别一致            -> TP
    - 预测框匹配不到任何同类真值（IoU 不达标或类别错）-> FP
    - 真值未被任何预测框匹配                        -> FN
IoU 阈值 τ 按真值所属目标大类取：车辆目标 0.35，其他目标 0.50（比赛表2）。

匹配为贪心一对一：预测按置信度降序，每个预测框在同类真值中找 IoU 最高且 >= τ 的
未匹配真值；同一真值被多框匹配时，置信度最高者计 TP、其余计 FP。

指标：
    Recall    = TP / (TP + FN)
    FDR       = FP / (TP + FP)
    Precision = TP / (TP + FP)

输出：总指标 + 舰船/飞机/车辆三大类分组指标 + 逐小类指标 + 各大类下小类指标的
macro 平均。

Usage:
    python scripts/eval_metrics.py \
        --checkpoint output/rfdetr_nano_shwx/checkpoint_best_total.pth \
        --split test \
        --dataset_dir /home/liu/datasets/SHWX-dataset-dict \
        --conf 0.25
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from PIL import Image

from rfdetr import RFDETR

# ══════════════════════════════════════════════════════════════════════
#  比赛口径配置
# ══════════════════════════════════════════════════════════════════════

# 比赛三大类：舰船 / 飞机 / 车辆。舰船四型与车辆（FSC 发射车）显式列出，
# data.yaml 中的其余类别归为飞机。
SHIP_CLASS_NAMES: set[str] = {"HM", "LQS", "QHS", "MS"}
VEHICLE_CLASS_NAMES: set[str] = {"FSC"}

# IoU 阈值（比赛表2）：车辆目标 0.35，其他目标 0.50。
GROUP_IOU_THRESHOLDS: dict[str, float] = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ══════════════════════════════════════════════════════════════════════
#  比赛指标评测（来源：val/competition_metrics.py，原样内联）
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
#  数据读取与分组映射
# ══════════════════════════════════════════════════════════════════════


def load_class_names(dataset_dir: str) -> list[str]:
    """Load ordered class names from the YOLO ``data.yaml`` mapping.

    Args:
        dataset_dir: Root of the SHWX dataset (contains ``data.yaml``).

    Returns:
        Class names ordered by YOLO class id 0..N-1.

    Raises:
        ValueError: If ``names`` is neither a contiguous 0..N-1 dict nor a list.
    """
    data = yaml.safe_load((Path(dataset_dir) / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        for i in range(len(names)):
            if i not in names:
                raise ValueError(f"data.yaml 'names' keys must be contiguous 0..{len(names) - 1}")
        return [str(names[i]) for i in range(len(names))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"data.yaml 'names' must be a dict or list, got {type(names).__name__}")


def build_class_to_group(names: list[str]) -> dict[int, str]:
    """按类别名把 class id 映射到比赛大类（舰船/飞机/车辆）。

    舰船四型与车辆（FSC）显式列出，其余类别归为飞机。

    Args:
        names: 按类别 id 有序的类别名列表。

    Returns:
        ``{类别 id: 大类名}`` 映射。
    """
    class_to_group: dict[int, str] = {}
    for class_id, name in enumerate(names):
        if name in SHIP_CLASS_NAMES:
            class_to_group[class_id] = "ship"
        elif name in VEHICLE_CLASS_NAMES:
            class_to_group[class_id] = "vehicle"
        else:
            class_to_group[class_id] = "aircraft"
    return class_to_group


def build_image_size_map(image_dir: Path) -> dict[str, tuple[int, int]]:
    """读取图像目录中每张图的尺寸，返回 {stem: (width, height)}。"""
    size_map: dict[str, tuple[int, int]] = {}
    for img_path in sorted(image_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(img_path) as img:
            size_map[img_path.stem] = img.size  # (width, height)
    return size_map


# ══════════════════════════════════════════════════════════════════════
#  报表辅助
# ══════════════════════════════════════════════════════════════════════


def format_eval_line(name: str, result: EvalResult) -> str:
    """把单组比赛评测结果格式化为一行。

    Args:
        name: 组名（如 ``all``/``ship`` 或类别名）。
        result: 单组比赛评测结果。

    Returns:
        格式化后的评测结果行。
    """
    return (
        f"{name:<11s}TP={result.tp:<7d}FP={result.fp:<7d}FN={result.fn:<7d}"
        f"Recall={result.recall:.4f} FDR={result.fdr:.4f} Precision={result.precision:.4f}"
    )


def format_macro_line(name: str, macro: Mapping[str, float]) -> str:
    """把大类下小类指标的 macro 平均结果格式化为一行。

    Args:
        name: 大类名（如 ``ship``/``aircraft``/``vehicle``）或 ``total``。
        macro: 含 ``avg_tp``/``avg_fp``/``avg_fn``/``recall``/``fdr``/``precision``
            六项的 macro 平均指标。

    Returns:
        格式化后的 macro 平均结果行。
    """
    return (
        f"{name:<11s}avgTP={macro['avg_tp']:.2f} avgFP={macro['avg_fp']:6.2f} "
        f"avgFN={macro['avg_fn']:6.2f} avgRecall={macro['recall']:.4f} "
        f"avgFDR={macro['fdr']:.4f} avgPrecision={macro['precision']:.4f}"
    )


def compute_group_macro_averages(
    per_class_results: Mapping[str, EvalResult],
    class_to_group: Mapping[int, str],
    class_names: Mapping[int, str],
) -> dict[str, dict[str, float]]:
    """计算每个大类下小类指标的平均值（macro 平均）。

    对大类中的每个小类，先按比赛口径计算各项指标（TP/FP/FN、召回率、虚警率、
    精确率），再直接对同大类下所有小类的指标取算术平均，而不是先累计样本数再
    计算指标。例如舰船的召回率 = 四型船（HM、LQS、QHS、MS）召回率的平均值。

    Args:
        per_class_results: ``{类别名: EvalResult}`` 的逐类评估结果。
        class_to_group: ``{类别 id: 大类名}`` 映射。
        class_names: ``{类别 id: 类别名}`` 映射。

    Returns:
        ``{大类名: {"avg_tp": ..., "avg_fp": ..., "avg_fn": ..., "recall": ...,
        "fdr": ..., "precision": ...}}``，大类顺序按其内最小类别 id 升序排列
        （即舰船→飞机→车辆）。
    """
    # 把 {类别id: 大类} 反转为 {大类: [类别id]}，类别 id 升序
    group_to_class_ids: dict[str, list[int]] = defaultdict(list)
    for class_id, group_name in sorted(class_to_group.items()):
        group_to_class_ids[group_name].append(class_id)

    # 按大类内最小类别 id 排序大类，保证舰船→飞机→车辆的展示顺序
    group_macro: dict[str, dict[str, float]] = {}
    for group_name, class_ids in sorted(
        group_to_class_ids.items(),
        key=lambda item: min(item[1]),
    ):
        # 测试集中不存在（无真实框也无预测）的小类取全零结果，保持按全部小类平均
        class_results = [
            per_class_results.get(class_names[class_id], EvalResult(tp=0, fp=0, fn=0)) for class_id in class_ids
        ]
        num_classes = len(class_results)
        group_macro[group_name] = {
            "avg_tp": sum(result.tp for result in class_results) / num_classes,
            "avg_fp": sum(result.fp for result in class_results) / num_classes,
            "avg_fn": sum(result.fn for result in class_results) / num_classes,
            "recall": sum(result.recall for result in class_results) / num_classes,
            "fdr": sum(result.fdr for result in class_results) / num_classes,
            "precision": sum(result.precision for result in class_results) / num_classes,
        }
    return group_macro


def compute_total_metrics(group_macro: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """计算总指标：各大类平均指标再取算术平均（即（舰船+飞机+车辆）/3）。

    Args:
        group_macro: ``compute_group_macro_averages`` 的返回结果。

    Returns:
        与单大类 macro 相同键结构的六项总指标。
    """
    metric_keys = ("avg_tp", "avg_fp", "avg_fn", "recall", "fdr", "precision")
    num_groups = len(group_macro)
    if num_groups == 0:
        return {key: 0.0 for key in metric_keys}
    return {key: sum(group[key] for group in group_macro.values()) / num_groups for key in metric_keys}


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset_dir", default="/home/liu/datasets/SHWX-dataset-dict")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="兜底 IoU 阈值（未在大类阈值表列出的分组；车辆 0.35 / 其他 0.50 已按比赛口径配置）",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reason-plugin",
        default="",
        help="Trained FFT consistency plugin (*.pth). When set, the plugin re-scores "
        "blurry candidates from a low-threshold pass before the final --conf cut-off.",
    )
    parser.add_argument(
        "--reason-conf-low",
        type=float,
        default=None,
        help="Candidate collection threshold when --reason-plugin is used.  "
        "Defaults to the plugin's own conf_low so train/eval stay consistent.",
    )
    parser.add_argument(
        "--boost-scale",
        type=float,
        default=None,
        help="Override the plugin's boost_scale (score adjustment per unit of probability above p_threshold).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=None,
        help="Override the plugin's p_threshold (sigmoid probability decision line).",
    )
    parser.add_argument(
        "--reason-class-ids",
        type=str,
        default=None,
        help="Comma-separated class ids whose blurry B's the plugin re-scores at "
        "inference; other classes keep baseline scores (e.g. '24' for FSC only).",
    )
    args = parser.parse_args()

    split = args.split
    dataset_root = Path(args.dataset_dir)
    names = load_class_names(args.dataset_dir)
    name2id = {name: class_id for class_id, name in enumerate(names)}
    class_names = {class_id: name for class_id, name in enumerate(names)}
    class_to_group = build_class_to_group(names)

    # 读取测试集真实框（YOLO 归一化坐标 → 像素 xyxy）
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_size_map = build_image_size_map(image_dir)
    gt_records = load_yolo_labels(label_dir, image_size_map)

    model = RFDETR.from_checkpoint(args.checkpoint)
    # predict()'s device-move + inference-mode decorators read self.model.device
    # and place the weights there lazily. Point it at the requested device.
    model.model.device = args.device

    # Optional FFT consistency plugin: load it and expose the frozen detector's
    # class-embedding matrix as the CRDe key/value.
    reason_plugin = None
    class_embed_weight = None
    if args.reason_plugin:
        from rfdetr.reasoning import PluginLoader

        reason_plugin = PluginLoader.load(args.reason_plugin)
        if args.boost_scale is not None:
            reason_plugin.config.boost_scale = args.boost_scale
        if args.p_threshold is not None:
            reason_plugin.config.p_threshold = args.p_threshold
        if args.reason_class_ids is not None:
            reason_plugin.config.reason_class_ids = tuple(int(x) for x in args.reason_class_ids.split(",") if x.strip())
        reason_plugin.to(args.device)
        # model.model is the ModelContext; model.model.model is the underlying
        # LWDETR module owning the class-embedding head.
        class_embed_weight = model.model.model.class_embed.weight.detach().to(args.device)
        # Default the candidate-collection threshold to the plugin's own conf_low
        # so training and inference use the same A/B band.
        if args.reason_conf_low is None:
            args.reason_conf_low = reason_plugin.config.conf_low
        print(f"[reason-plugin] {args.reason_plugin} conf_low={args.reason_conf_low} target_conf={args.conf}")

    # Build per-image stem -> original image path for inference.
    id_to_path = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    # 逐张推理并把预测框转成 BoxRecord（class_name → class_id）
    pred_records: list[BoxRecord] = []
    with torch.no_grad():
        for img_id, path in id_to_path.items():
            if not path.exists():
                print(f"  [skip] missing {path}")
                continue
            res = model.predict(
                str(path),
                threshold=args.reason_conf_low if reason_plugin is not None else args.conf,
                include_source_image=reason_plugin is not None,
            )
            d = res
            class_names_pred = d.data.get("class_name", [])
            if reason_plugin is not None:
                # Re-score: low-threshold candidates -> plugin -> final --conf cut.
                source_image = d.metadata.get("source_image")
                if source_image is None:
                    raise RuntimeError("--reason-plugin requires include_source_image")
                cand_boxes = np.asarray(d.xyxy, dtype=np.float32)
                cand_scores = np.asarray(d.confidence, dtype=np.float32)
                cand_classes = np.asarray([name2id.get(c, -1) for c in class_names_pred], dtype=np.int64)
                valid = cand_classes >= 0
                cand_boxes = cand_boxes[valid]
                cand_scores = cand_scores[valid]
                cand_classes = cand_classes[valid]
                if cand_boxes.size:
                    out_boxes, out_scores, out_classes = reason_plugin.predict_detections(
                        source_image=source_image,
                        candidate_boxes=cand_boxes,
                        candidate_scores=cand_scores,
                        candidate_classes=cand_classes,
                        class_names=names,
                        class_embed_weight=class_embed_weight,
                        device=args.device,
                        target_conf=args.conf,
                    )
                    for i in range(len(out_boxes)):
                        pred_records.append(
                            BoxRecord(
                                image_id=img_id,
                                class_id=int(out_classes[i]),
                                xyxy=tuple(float(v) for v in out_boxes[i]),
                                score=float(out_scores[i]),
                            )
                        )
                continue
            for i in range(len(d.xyxy)):
                class_name = class_names_pred[i]
                if class_name not in name2id:
                    print(f"  [skip] 未知类别 '{class_name}' @ {img_id}")
                    continue
                pred_records.append(
                    BoxRecord(
                        image_id=img_id,
                        class_id=name2id[class_name],
                        xyxy=tuple(float(v) for v in d.xyxy[i]),
                        score=float(d.confidence[i]),
                    )
                )
    del model

    # ── 比赛指标：总 + 三大类（舰船/飞机/车辆）───────────────────────
    config = EvalConfig(
        class_to_group=class_to_group,
        group_iou_thresholds=GROUP_IOU_THRESHOLDS,
        default_iou_threshold=args.iou,
        class_aware=True,
    )
    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)

    # ── 细粒度逐类指标（每类独立成组，FSC=0.35 其余 0.50）─────────────
    per_class_config = EvalConfig(
        class_to_group=class_names,
        group_iou_thresholds={name: (0.35 if name in VEHICLE_CLASS_NAMES else 0.50) for name in names},
        default_iou_threshold=args.iou,
        class_aware=True,
    )
    per_class_results = evaluate_competition_metrics(gt_records, pred_records, per_class_config)

    # ── 每个大类下小类指标的 macro 平均 与 总指标 ─────────────────────
    group_macro = compute_group_macro_averages(
        per_class_results["groups"],
        class_to_group,
        class_names,
    )
    total_macro = compute_total_metrics(group_macro)

    # ══════════════ 输出 ══════════════
    print(f"\n=== Results (split={split}, conf={args.conf}) ===")
    print("IoU 阈值: 车辆=0.35，其他目标=0.50")
    print(f"GT boxes={len(gt_records)}  Pred boxes={len(pred_records)}")

    # 总指标 + 三大类分组（顺序：all + 舰船/飞机/车辆）
    group_names = list(group_macro.keys())  # 按大类内最小类别 id 排序：ship, aircraft, vehicle
    print()
    for name, result in [("all", eval_results["all"]), *[(g, eval_results["groups"][g]) for g in group_names]]:
        print(format_eval_line(name, result))

    # 逐类表格
    print("\n--- Per-class (TP/FP/FN, Recall, FDR, Precision) ---")
    print(f"{'class':<12}{'TP':>5}{'FP':>5}{'FN':>5}  {'Recall':>8} {'FDR':>8} {'Precision':>10}")
    for class_name in sorted(per_class_results["groups"].keys()):
        result = per_class_results["groups"][class_name]
        print(
            f"{class_name:<12}{result.tp:>5}{result.fp:>5}{result.fn:>5}  "
            f"{result.recall:>7.2%} {result.fdr:>7.2%} {result.precision:>9.2%}"
        )

    # 每个大类下小类指标的 macro 平均
    print("\n--- 每个大类下小类指标的平均值（macro 平均）---")
    for group_name, macro in group_macro.items():
        print(format_macro_line(group_name, macro))

    # 总指标（各大类平均指标再取算术平均）
    print("\n--- 总指标（各大类平均指标再取算术平均，即（舰船+飞机+车辆）/3）---")
    print(format_macro_line("total", total_macro))


if __name__ == "__main__":
    main()
