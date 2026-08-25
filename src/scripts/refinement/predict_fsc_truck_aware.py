# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""运行带 truck 竞争类别的 FSC 两级级联推理。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import FSCVerifier, iou_xyxy  # noqa: E402


FSC_CLASS_ID = 24
TRUCK_CLASS_ID = 25
DETECTOR_THRESHOLD = 0.25
HIGH_CONFIDENCE_FLOOR = 0.70
TRUCK_OVERLAP = 0.50


def _parse_args() -> argparse.Namespace:
    """解析 truck 感知级联参数。"""
    parser = argparse.ArgumentParser(description="运行带 truck 类竞争的 FSC 两级检测")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _images(path: Path) -> list[Path]:
    """获取单张图像或目录中的图像。"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if path.is_file() and path.suffix.lower() in suffixes:
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
    raise FileNotFoundError(f"未找到图像: {path}")


def _draw(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, path: Path, color: tuple[int, int, int]) -> None:
    """保存检测框可视化。"""
    canvas = image.copy()
    for box, score in zip(boxes, scores, strict=True):
        x0, y0, x1, y1 = (int(value) for value in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"FSC {float(score):.3f}", (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(path), canvas)


def _suppress_truck_conflicts(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    """用显式 truck 类抑制同框且 truck 分数更高的 FSC 候选。"""
    fsc_indices = np.flatnonzero(class_ids == FSC_CLASS_ID)
    truck_indices = np.flatnonzero(class_ids == TRUCK_CLASS_ID)
    keep = np.ones(len(class_ids), dtype=bool)
    for index in fsc_indices.tolist():
        for truck_index in truck_indices.tolist():
            if iou_xyxy(tuple(boxes[index]), tuple(boxes[truck_index])) >= TRUCK_OVERLAP and scores[truck_index] >= scores[index]:
                keep[index] = False
                break
    return keep


def main() -> None:
    """对图像运行一级检测、truck 冲突处理和 FSC 二级复核。"""
    args = _parse_args()
    detector = RFDETR.from_checkpoint(args.detector)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    verifier = FSCVerifier.from_checkpoint(args.verifier, device=device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for image_path in _images(Path(args.image).resolve()):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_array = np.asarray(image).copy()
        detections = detector.predict(image_array, threshold=DETECTOR_THRESHOLD, include_source_image=False)
        boxes = np.asarray(detections.xyxy, dtype=np.float32)
        scores = np.asarray(detections.confidence, dtype=np.float32)
        class_ids = np.asarray(detections.class_id, dtype=np.int64)
        conflict_keep = _suppress_truck_conflicts(boxes, scores, class_ids)
        fsc_indices = np.flatnonzero((class_ids == FSC_CLASS_ID) & conflict_keep)
        low_indices = [index for index in fsc_indices.tolist() if scores[index] < HIGH_CONFIDENCE_FLOOR]
        accepted = set(index for index in fsc_indices.tolist() if scores[index] >= HIGH_CONFIDENCE_FLOOR)
        if low_indices:
            probabilities = verifier.predict_probabilities(image, [boxes[index] for index in low_indices])
            accepted.update(index for row, index in enumerate(low_indices) if int(probabilities[row].argmax().item()) == 1)
        final_keep = conflict_keep.copy()
        final_keep[class_ids == FSC_CLASS_ID] = False
        final_keep[list(accepted)] = True
        final_boxes = boxes[final_keep & (class_ids == FSC_CLASS_ID)]
        final_scores = scores[final_keep & (class_ids == FSC_CLASS_ID)]
        _draw(image_array[:, :, ::-1], boxes[class_ids == FSC_CLASS_ID], scores[class_ids == FSC_CLASS_ID], output_dir / f"{image_path.stem}_stage1.jpg", (0, 0, 255))
        _draw(image_array[:, :, ::-1], final_boxes, final_scores, output_dir / f"{image_path.stem}_stage2.jpg", (0, 255, 0))
        reports.append(
            {
                "image": str(image_path),
                "detector_threshold": DETECTOR_THRESHOLD,
                "high_confidence_floor": HIGH_CONFIDENCE_FLOOR,
                "stage1_fsc": int((class_ids == FSC_CLASS_ID).sum()),
                "truck_conflicts_suppressed": int(((class_ids == FSC_CLASS_ID) & ~conflict_keep).sum()),
                "low_confidence_routed": len(low_indices),
                "stage2_fsc": int(len(final_boxes)),
                "stage2_boxes": final_boxes.tolist(),
                "stage2_scores": [float(score) for score in final_scores],
            }
        )
    (output_dir / "prediction_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
