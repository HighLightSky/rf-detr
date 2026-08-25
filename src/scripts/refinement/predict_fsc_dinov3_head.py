# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""运行冻结 DINOv3 特征 FSC 二级复核。"""

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
from rfdetr.refinement import crop_fsc_context, iou_xyxy  # noqa: E402
from scripts.refinement.train_fsc_dinov3_head import FORMAT, FSCDinoV3Head  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析 DINOv3 推理参数。"""
    parser = argparse.ArgumentParser(description="运行 DINOv3 FSC 两阶段检测")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """执行候选 FSC 框的置信度优先 NMS。"""
    order = np.argsort(-scores)
    kept: list[int] = []
    for index in order.tolist():
        if all(iou_xyxy(tuple(boxes[index]), tuple(boxes[chosen])) <= threshold for chosen in kept):
            kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def _draw(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, output: Path, color: tuple[int, int, int]) -> None:
    """保存阶段检测框可视化。"""
    canvas = image.copy()
    for box, score in zip(boxes, scores, strict=True):
        x0, y0, x1, y1 = (int(value) for value in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"FSC {float(score):.3f}", (x0, max(y0 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    """用一级 FSC 候选和 DINOv3 固定 argmax 生成最终结果。"""
    args = _parse_args()
    payload = torch.load(args.verifier, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"不是 {FORMAT} checkpoint")
    import timm
    from timm.data import create_transform, resolve_model_data_config

    metadata = dict(payload["metadata"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = timm.create_model(metadata["model_name"], pretrained=True, num_classes=0).to(device).eval()
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    head = FSCDinoV3Head(int(payload["feature_dim"])).to(device).eval()
    head.load_state_dict(payload["state_dict"])
    detector = RFDETR.from_checkpoint(args.detector)
    image_path = Path(args.image).resolve()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image_array = np.asarray(image).copy()
    detections = detector.predict(image_array, threshold=0.05, include_source_image=False)
    boxes = np.asarray(detections.xyxy, dtype=np.float32)
    scores = np.asarray(detections.confidence, dtype=np.float32)
    classes = np.asarray(detections.class_id, dtype=np.int64)
    fsc = np.flatnonzero(classes == 24)
    fsc = fsc[_nms(boxes[fsc], scores[fsc])]
    crops = [
        transform(
            crop_fsc_context(
                image,
                boxes[index],
                context_scale=float(metadata["context_scale"]),
                output_size=224,
            )
        )
        for index in fsc
    ]
    with torch.inference_mode():
        keep = head(backbone(torch.stack(crops).to(device))).argmax(1).cpu().numpy().astype(bool)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _draw(image_array[:, :, ::-1], boxes[fsc], scores[fsc], output / f"{image_path.stem}_stage1.jpg", (0, 0, 255))
    _draw(
        image_array[:, :, ::-1],
        boxes[fsc][keep],
        scores[fsc][keep],
        output / f"{image_path.stem}_stage2.jpg",
        (0, 255, 0),
    )
    report = {
        "image": str(image_path),
        "stage1_fsc": int(len(fsc)),
        "stage2_fsc": int(keep.sum()),
        "stage2_boxes": boxes[fsc][keep].tolist(),
        "stage2_scores": [float(score) for score in scores[fsc][keep]],
        "selection": metadata["selection"],
    }
    (output / "prediction_report.json").write_text(json.dumps([report], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([report], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
