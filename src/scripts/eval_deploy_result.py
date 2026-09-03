"""评测部署镜像输出的 result.json，并可按开关绘制船舶类与发射车类可视化。

脚本位于 ``deploy/`` 目录之外，不会被部署 Dockerfile 打包。用法示例：
``python src/scripts/eval_deploy_result.py --result deploy/test-output/result.json --labels /path/to/labels``。
附加可视化只需增加 ``--visualize --images /path/to/images``。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

# 允许直接执行 ``python src/scripts/eval_deploy_result.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.sscl.prompts import SHWX_CLASS_NAMES  # noqa: E402
from scripts.eval_lib import compute_group_macro_averages, compute_total_metrics  # noqa: E402
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    EvalResult,
    evaluate_competition_metrics,
    xywhn_to_xyxy,
)

CLASS_NAMES = dict(SHWX_CLASS_NAMES)
CLASS_TO_GROUP = {
    **{class_id: "ship" for class_id in range(4)},
    **{class_id: "aircraft" for class_id in range(4, 24)},
    24: "vehicle",
}
IGNORED_GT_CLASS_IDS = frozenset({25})
GROUP_IOU_THRESHOLDS = {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35}
RUNTIME_FIELDS = ("runtime_ms", "run_duration_ms", "duration_ms", "elapsed_ms")

# 可视化相关的类别集合：船舶（0-3）与发射车（24）是比赛重点观察对象。
SHIP_CLASS_IDS = frozenset(range(4))
VEHICLE_CLASS_ID = 24
DEFAULT_VIS_CLASS_IDS = frozenset(SHIP_CLASS_IDS | {VEHICLE_CLASS_ID})
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# 画框颜色：GT 恒为绿色；预测框按分组区分船舶与发射车。
GT_COLOR = (0, 255, 0)
VEHICLE_COLOR = (0, 140, 255)
SHIP_COLOR = (255, 0, 0)


@dataclass(frozen=True)
class EvaluationReport:
    """部署结果评测汇总。"""

    all_result: EvalResult
    group_results: Mapping[str, EvalResult]
    group_macro: Mapping[str, Mapping[str, float]]
    total_macro: Mapping[str, float]
    longest_runtime_ms: float | None
    timed_image_id: str | None


def _read_gt_records(labels_dir: Path, images: list[Mapping[str, Any]]) -> list[BoxRecord]:
    """读取结果中图像对应的 YOLO 真实框。"""
    records: list[BoxRecord] = []
    for image in images:
        image_id = str(image["image_id"])
        width = int(image["width"])
        height = int(image["height"])
        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"缺少真实标签: {label_path}")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"标签格式错误: {label_path}:{line_number}")
            class_id = int(float(parts[0]))
            if class_id in IGNORED_GT_CLASS_IDS:
                continue
            if class_id not in CLASS_TO_GROUP:
                raise ValueError(f"未知类别 id={class_id}: {label_path}:{line_number}")
            xyxy = xywhn_to_xyxy(*map(float, parts[1:5]), width, height)
            records.append(BoxRecord(image_id, class_id, xyxy))
    return records


def _read_pred_records(images: list[Mapping[str, Any]]) -> list[BoxRecord]:
    """读取 result.json 中的预测框。"""
    records: list[BoxRecord] = []
    for image in images:
        image_id = str(image["image_id"])
        for obj in image.get("objects", []):
            class_id = int(obj["category_id"])
            bbox = tuple(float(value) for value in obj["bbox"])
            if len(bbox) != 4 or class_id not in CLASS_TO_GROUP:
                raise ValueError(f"预测框格式错误: image_id={image_id}")
            records.append(BoxRecord(image_id, class_id, bbox, float(obj.get("score", 0.0))))
    return records


def _runtime_values(images: list[Mapping[str, Any]]) -> tuple[float | None, str | None]:
    """提取显式或由相邻结束时间戳估算的单图耗时。"""
    values: list[tuple[str, float]] = []
    for image in images:
        image_id = str(image["image_id"])
        for field in RUNTIME_FIELDS:
            if field in image:
                values.append((image_id, float(image[field])))
                break
    if not values:
        previous: float | None = None
        for image in images:
            timestamp = image.get("run_end_timestamp")
            if timestamp is not None and previous is not None:
                delta = float(timestamp) - previous
                if delta >= 0:
                    values.append((str(image["image_id"]), delta))
            if timestamp is not None:
                previous = float(timestamp)
    if not values:
        return None, None
    image_id, runtime = max(values, key=lambda item: item[1])
    return runtime, image_id


def _find_image(image_dir: Path, image_id: str) -> Path | None:
    """按扩展名在图像目录中查找指定 image_id 的图像文件。

    Args:
        image_dir: 原始图像目录。
        image_id: 图像名（不含扩展名）。

    Returns:
        匹配的图像路径，未找到时返回 None。
    """
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _group_color(class_id: int) -> tuple[int, int, int]:
    """返回预测框分组颜色：发射车橙、船舶蓝。

    Args:
        class_id: 类别索引。

    Returns:
        BGR 三元组颜色。
    """
    return VEHICLE_COLOR if class_id == VEHICLE_CLASS_ID else SHIP_COLOR


def _draw_box_label(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """在 BGR 图像上原地绘制单个框与带底色文字标签。

    Args:
        image: 原始 BGR 图像，原地修改。
        x1: 框左上角 x。
        y1: 框左上角 y。
        x2: 框右下角 x。
        y2: 框右下角 y。
        text: 标签文本。
        color: BGR 颜色。
        thickness: 线宽，负数表示填充。
    """
    height, width = image.shape[:2]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
    label_y1 = max(y1 - text_height - baseline, 0)
    cv2.rectangle(
        image,
        (x1, label_y1),
        (min(x1 + text_width, width - 1), label_y1 + text_height + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1, label_y1 + text_height),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_boxes(
    image: np.ndarray,
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
) -> None:
    """在 BGR 图像上叠加绘制 GT（绿）与预测（按分组配色）框。

    Args:
        image: 原始 BGR 图像，原地修改。
        gt_records: 已按目标类别过滤的真实框列表。
        pred_records: 已按目标类别过滤的预测框列表。
    """
    for record in gt_records:
        x1, y1, x2, y2 = (int(round(value)) for value in record.xyxy)
        name = CLASS_NAMES.get(record.class_id, str(record.class_id))
        _draw_box_label(image, x1, y1, x2, y2, f"GT {name}", GT_COLOR, 1)
    for record in pred_records:
        x1, y1, x2, y2 = (int(round(value)) for value in record.xyxy)
        name = CLASS_NAMES.get(record.class_id, str(record.class_id))
        score = float(record.score) if record.score is not None else 0.0
        _draw_box_label(image, x1, y1, x2, y2, f"{name} {score:.2f}", _group_color(record.class_id), 2)


def visualize_result_file(
    result_path: Path,
    labels_dir: Path,
    images_dir: Path,
    vis_dir: Path,
    class_ids_to_draw: frozenset[int] = DEFAULT_VIS_CLASS_IDS,
) -> list[Path]:
    """把 result.json 中的预测框与真实标签按类别集合绘制到原图上。

    同一张图叠加绘制 GT 框（绿色）与预测框（船舶蓝/发射车橙），并横向打印
    数量小结；仅当图像中存在目标类别时落盘，避免产生空图。

    Args:
        result_path: result.json 路径。
        labels_dir: YOLO 真实标签目录。
        images_dir: 原始图像目录（按 image_id + 扩展名匹配）。
        vis_dir: 可视化输出目录，自动创建。
        class_ids_to_draw: 需要绘制的类别 id 集合，默认船舶与发射车。

    Returns:
        实际写出的图像路径列表。

    Raises:
        ValueError: result.json 缺少 images 数组。
    """
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("result.json 必须包含 images 数组")
    images = payload["images"]
    gt_by_image: dict[str, list[BoxRecord]] = {}
    for record in _read_gt_records(labels_dir, images):
        gt_by_image.setdefault(record.image_id, []).append(record)
    pred_by_image: dict[str, list[BoxRecord]] = {}
    for record in _read_pred_records(images):
        pred_by_image.setdefault(record.image_id, []).append(record)

    vis_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for image in images:
        image_id = str(image["image_id"])
        image_path = _find_image(images_dir, image_id)
        if image_path is None:
            print(f"[vis] 跳过 {image_id}: 图像文件不存在")
            continue
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[vis] 跳过 {image_id}: 图像读取失败")
            continue
        width = image_bgr.shape[1]
        gt_records = [record for record in gt_by_image.get(image_id, []) if record.class_id in class_ids_to_draw]
        pred_records = [
            record for record in pred_by_image.get(image_id, []) if record.class_id in class_ids_to_draw
        ]
        if not gt_records and not pred_records:
            continue
        ship_gt = sum(1 for record in gt_records if record.class_id in SHIP_CLASS_IDS)
        ship_pred = sum(1 for record in pred_records if record.class_id in SHIP_CLASS_IDS)
        vehicle_gt = sum(1 for record in gt_records if record.class_id == VEHICLE_CLASS_ID)
        vehicle_pred = sum(1 for record in pred_records if record.class_id == VEHICLE_CLASS_ID)
        summary = (
            f"{image_id} ({width}x{image_bgr.shape[0]})  "
            f"船舶 GT/P={ship_gt}/{ship_pred}  发射车 GT/P={vehicle_gt}/{vehicle_pred}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        (bar_width, _), _ = cv2.getTextSize(summary, font, 0.5, 1)
        cv2.rectangle(image_bgr, (0, 0), (min(bar_width + 8, width - 1), 22), (0, 0, 0), -1)
        cv2.putText(image_bgr, summary, (4, 16), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        _draw_boxes(image_bgr, gt_records, pred_records)
        output_path = vis_dir / f"{image_id}.png"
        cv2.imwrite(str(output_path), image_bgr)
        written.append(output_path)
    return written


def evaluate_result_file(result_path: Path, labels_dir: Path) -> EvaluationReport:
    """读取部署结果并计算比赛指标与最长单图耗时。"""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("result.json 必须包含 images 数组")
    images = payload["images"]
    gt_records = _read_gt_records(labels_dir, images)
    pred_records = _read_pred_records(images)
    config = EvalConfig(CLASS_TO_GROUP, GROUP_IOU_THRESHOLDS)
    results = evaluate_competition_metrics(gt_records, pred_records, config)
    per_class = evaluate_competition_metrics(
        gt_records,
        pred_records,
        EvalConfig(CLASS_NAMES, {name: 0.35 if class_id == 24 else 0.50 for class_id, name in CLASS_NAMES.items()}),
    )
    group_macro = compute_group_macro_averages(per_class["groups"], CLASS_TO_GROUP, CLASS_NAMES)
    longest_runtime_ms, timed_image_id = _runtime_values(images)
    return EvaluationReport(
        all_result=results["all"],
        group_results=results["groups"],
        group_macro=group_macro,
        total_macro=compute_total_metrics(group_macro),
        longest_runtime_ms=longest_runtime_ms,
        timed_image_id=timed_image_id,
    )


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评测部署镜像输出的 result.json")
    parser.add_argument("--result", type=Path, required=True, help="result.json 路径")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO 真实标签目录")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="开启可视化，绘制船舶类与发射车类的 GT/预测框",
    )
    parser.add_argument("--images", type=Path, default=None, help="原始图像目录（可视化为开启时必填）")
    parser.add_argument("--vis-dir", type=Path, default=None, help="可视化输出目录（默认 result 同级 viz/）")
    parser.add_argument(
        "--vis-classes",
        type=str,
        default="0,1,2,3,24",
        help="需要绘制的类别 id 逗号分隔列表，默认船舶(0-3)与发射车(24)",
    )
    args = parser.parse_args()
    if args.visualize and args.images is None:
        parser.error("--visualize 需要同时提供 --images")
    return args


def _parse_vis_classes(value: str) -> frozenset[int]:
    """把逗号分隔的类别 id 字符串解析为集合。

    Args:
        value: 如 ``"0,1,2,3,24"``。

    Returns:
        类别 id 冻结集合。
    """
    try:
        return frozenset(int(class_id) for class_id in value.split(",") if class_id.strip())
    except ValueError as error:
        raise ValueError(f"--vis-classes 必须是逗号分隔的整数: {value!r}") from error


def main() -> None:
    """执行评测并打印汇总结果，可选落盘可视化图。"""
    args = _parse_args()
    report = evaluate_result_file(args.result, args.labels)
    print(f"TP={report.all_result.tp} FP={report.all_result.fp} FN={report.all_result.fn}")
    for group_name, result in report.group_results.items():
        print(
            f"{group_name}: TP={result.tp} FP={result.fp} FN={result.fn} "
            f"虚警率={result.fdr:.4f} 召回率={result.recall:.4f}"
        )
    for group_name, macro in report.group_macro.items():
        print(f"{group_name}平均: 虚警率={macro['fdr']:.4f} 召回率={macro['recall']:.4f}")
    print(f"总虚警率={report.total_macro['fdr']:.4f} 总召回率={report.total_macro['recall']:.4f}")
    if report.longest_runtime_ms is None:
        print("最长单张运行时长: 无法从 result.json 还原")
    else:
        print(f"最长单张运行时长: {report.longest_runtime_ms:.3f} ms ({report.timed_image_id})")
    if args.visualize:
        vis_dir = args.vis_dir or (args.result.parent / "viz")
        written = visualize_result_file(
            args.result,
            args.labels,
            args.images,
            vis_dir,
            class_ids_to_draw=_parse_vis_classes(args.vis_classes),
        )
        print(f"可视化完成: 共写入 {len(written)} 张图 -> {vis_dir}")


if __name__ == "__main__":
    main()
