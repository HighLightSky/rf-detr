# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图目标检测推理器：nano 边界检测切分 + 目标检测器逐裁窗推理。

复用 ``large_cut_pipeline`` 的切分方案（边界检测 → 裁切+padding → 目标检测 →
坐标映射回原图），但面向测试集批量评估优化：

- nano 边界检测器在构造时只加载一次，所有大图共用，不逐图重复加载；
- 边界检测对全部大图做一次批量前向（letterbox 预处理），再逐图裁切检测；
- 逐图记录目标检测耗时（裁切 → 推理 → 映射回原图），供评估报告输出
  平均/最大检测时长。

输出预测与 ``eval_lib.predict_batched_to_records`` 的 ``BoxRecord`` 一致
（坐标为原图像素空间），可直接参与比赛指标与 FP/FN 可视化。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from scripts.large_cut_pipeline import (
    _nms_boxes,
    crop_with_padding,
    infer_detector_on_crops,
    map_boxes_to_original,
    predict_batched_letterbox,
)
from val.competition_metrics import BoxRecord


class LargeImageTiler:
    """大图切分目标检测器（nano 边界检测器只加载一次）。

    Args:
        detector_model: 已加载的目标检测器（RFDETR 实例，如 SHWX 25 类）。
        boundary_checkpoint: nano 边界检测器 checkpoint 路径。
        boundary_resolution: 边界检测器输入分辨率（须与训练一致，默认 704）。
        boundary_conf: 边界框置信度阈值。
        detector_conf: 目标检测置信度阈值。
        padding: 裁窗外扩像素。
        nms_iou: 边界框 NMS IoU 阈值（0 关闭）。
        square_stretch: 边界检测用方形拉伸替代 letterbox。
        device: 推理设备。
        batch_size: 边界检测器 GPU 批量大小。
        num_workers: 边界检测器预取 worker 数。
    """

    def __init__(
        self,
        detector_model: Any,
        boundary_checkpoint: str | Path,
        *,
        boundary_resolution: int = 704,
        boundary_conf: float = 0.25,
        detector_conf: float = 0.25,
        padding: int = 32,
        nms_iou: float = 0.0,
        square_stretch: bool = False,
        device: str = "cuda:0",
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        from rfdetr import RFDETR

        self.detector_model = detector_model
        self.boundary_checkpoint = str(boundary_checkpoint)
        self.boundary_resolution = boundary_resolution
        self.boundary_conf = boundary_conf
        self.detector_conf = detector_conf
        self.padding = padding
        self.nms_iou = nms_iou
        self.square_stretch = square_stretch
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # nano 边界检测器只加载一次（所有大图共用）
        print(f"[i] 加载大图边界检测器: {self.boundary_checkpoint}")
        self.boundary_model = RFDETR.from_checkpoint(
            self.boundary_checkpoint,
            resolution=self.boundary_resolution,
        )

    def predict(
        self,
        image_paths: list[Path],
    ) -> tuple[list[BoxRecord], dict[str, float], dict[str, list[tuple[float, float, float, float]]]]:
        """对一批大图执行切分目标检测。

        Args:
            image_paths: 大图路径列表。

        Returns:
            ``(pred_records, per_image_seconds, crop_boxes_by_image)``：
            - pred_records: 全部大图的目标检测记录（原图像素坐标）；
            - per_image_seconds: ``{image_id: 秒}``，逐大图的目标检测耗时
              （裁切 → 推理 → 映射，不含 nano 边界检测与模型加载）；
            - crop_boxes_by_image: ``{image_id: [(x0, y0, x1, y1), ...]}``，
              每张大图的裁窗在原图中的坐标（含 padding），供可视化叠加。
        """
        if not image_paths:
            return [], {}, {}

        # 1. 所有大图一次性批量边界检测（letterbox 预处理）
        boundary_results = predict_batched_letterbox(
            self.boundary_model,
            image_paths,
            device=self.device,
            resolution=self.boundary_resolution,
            conf_threshold=self.boundary_conf,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            square_stretch=self.square_stretch,
        )
        boundary_by_id = {r["image_id"]: r for r in boundary_results}

        pred_records: list[BoxRecord] = []
        per_image_seconds: dict[str, float] = {}
        crop_boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
        for image_path in image_paths:
            stem = image_path.stem
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[w] 无法读取大图，跳过: {image_path}", flush=True)
                continue
            image_h, image_w = image_bgr.shape[:2]
            batch_result = boundary_by_id.get(stem)
            if batch_result is None:
                print(f"[w] 缺少边界检测结果，跳过: {stem}", flush=True)
                continue

            # 2. 边界框映射回原图坐标（+ 可选 NMS）
            boxes_orig = map_boxes_to_original(
                batch_result["boxes"],
                batch_result["scale"],
                batch_result["pad_x"],
                batch_result["pad_y"],
                image_w,
                image_h,
            )
            if self.nms_iou > 0 and boxes_orig.shape[0] > 0:
                boxes_orig = _nms_boxes(boxes_orig, self.nms_iou)

            # 3. 裁切 + padding + 目标检测 + 映射回原图（逐图计时）
            crops_rgb: list = []
            crop_offsets: list[tuple[int, int, int, int]] = []
            for box in boxes_orig:
                crop_bgr, crop_xyxy = crop_with_padding(
                    image_bgr,
                    tuple(int(v) for v in box),
                    self.padding,
                )
                crops_rgb.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                crop_offsets.append(crop_xyxy)
            crop_boxes_by_image[stem] = [
                (float(x0), float(y0), float(x1), float(y1)) for x0, y0, x1, y1 in crop_offsets
            ]

            t0 = time.perf_counter()
            detections = infer_detector_on_crops(
                self.detector_model,
                crops_rgb,
                self.detector_conf,
            )
            for det, (crop_x0, crop_y0, _, _) in zip(detections, crop_offsets, strict=False):
                for xyxy, score, class_id in zip(det.xyxy, det.confidence, det.class_id, strict=False):
                    x0 = float(xyxy[0]) + crop_x0
                    y0 = float(xyxy[1]) + crop_y0
                    x1 = float(xyxy[2]) + crop_x0
                    y1 = float(xyxy[3]) + crop_y0
                    pred_records.append(
                        BoxRecord(
                            image_id=stem,
                            class_id=int(class_id),
                            xyxy=(x0, y0, x1, y1),
                            score=float(score),
                        )
                    )
            per_image_seconds[stem] = time.perf_counter() - t0
        return pred_records, per_image_seconds, crop_boxes_by_image
