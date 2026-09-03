"""筛选 baseline 检测错误但 ours 完全正确的图片对比目录。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from compare_predictions import _draw, _gt_boxes, _predict
from rfdetr import RFDETR


def _iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    """计算两个 xyxy 框的交并比。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_completely_correct(
    predictions: list[tuple[float, float, float, float, int, float]],
    ground_truth: list[tuple[float, float, float, float, int, float]],
    iou_threshold: float,
) -> bool:
    """判断预测是否与全部 GT 一对一正确匹配且没有额外预测。"""
    if len(predictions) != len(ground_truth):
        return False
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, target in enumerate(ground_truth):
            if int(prediction[4]) != int(target[4]):
                continue
            overlap = _iou(prediction, target)
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_index, gt_index))
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    for _, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_targets:
            continue
        matched_predictions.add(pred_index)
        matched_targets.add(gt_index)
    return len(matched_targets) == len(ground_truth)


def main() -> None:
    """运行两套模型并复制满足筛选条件的三图对比目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--ignore-class", type=int, action="append", default=[25])
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.preferred_blas_library("cublas")
    baseline = RFDETR.from_checkpoint(str(args.baseline))
    ours = RFDETR.from_checkpoint(str(args.ours))
    if not torch.cuda.is_available():
        baseline.model.device = torch.device("cpu")
        ours.model.device = torch.device("cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows: list[tuple[str, str, int, int, int]] = []
    for split in ("train", "val", "test"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)
        selected = 0
        for index, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            targets = _gt_boxes(label_dir / f"{image_path.stem}.txt", width, height)
            baseline_boxes = _predict(baseline, image, args.threshold)
            ours_boxes = _predict(ours, image, args.threshold)
            ignored = set(args.ignore_class)
            targets = [box for box in targets if box[4] not in ignored]
            baseline_boxes = [box for box in baseline_boxes if box[4] not in ignored]
            ours_boxes = [box for box in ours_boxes if box[4] not in ignored]
            baseline_correct = _is_completely_correct(baseline_boxes, targets, args.iou_threshold)
            ours_correct = _is_completely_correct(ours_boxes, targets, args.iou_threshold)
            if not baseline_correct and ours_correct:
                source = args.comparison / split / image_path.stem
                destination = args.output / split / image_path.stem
                destination.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destination / "01_baseline.jpg"), _draw(image.copy(), baseline_boxes))
                cv2.imwrite(str(destination / "02_ours.jpg"), _draw(image.copy(), ours_boxes))
                cv2.imwrite(str(destination / "03_gt.jpg"), _draw(image.copy(), targets))
                rows.append((split, image_path.stem, len(targets), len(baseline_boxes), len(ours_boxes)))
                selected += 1
            if index % 500 == 0 or index == len(images):
                print(f"{split}: {index}/{len(images)}，已选 {selected}", flush=True)

    manifest = args.output / "筛选清单.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["数据集", "图片名", "GT数量", "baseline预测数量", "ours预测数量"])
        writer.writerows(rows)
    print(f"筛选完成：共 {len(rows)} 张，输出目录 {args.output}", flush=True)


if __name__ == "__main__":
    main()
