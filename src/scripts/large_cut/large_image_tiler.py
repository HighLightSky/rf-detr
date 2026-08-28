# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图边界检测和 crop 准备器。

边界检测器只加载一次并批量处理所有大图，完成边界映射后释放；目标检测由统一
detector runtime 处理小图和 crop，避免两个模型同时占用 GPU 显存。
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from scripts.large_cut.image_source import create_image_source
from scripts.large_cut.large_cut_pipeline import (
    _nms_boxes,
    map_boxes_to_original,
    predict_proxy_boundaries,
)


def _crop_bounds(
    image_shape: tuple[int, ...],
    box_xyxy: tuple[int, int, int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    """计算含 padding 且限制在图像范围内的 crop 坐标。"""
    image_h, image_w = image_shape[:2]
    x0, y0, x1, y1 = box_xyxy
    return (
        max(int(x0) - padding, 0),
        max(int(y0) - padding, 0),
        min(int(x1) + padding, image_w),
        min(int(y1) + padding, image_h),
    )


class LargeImageTiler:
    """批量运行边界模型并准备延迟生成 crop 所需的来源信息。"""

    def __init__(
        self,
        boundary_checkpoint: str | Path,
        *,
        boundary_resolution: int = 704,
        boundary_conf: float = 0.25,
        padding: int = 32,
        nms_iou: float = 0.0,
        square_stretch: bool = False,
        device: str = "cuda:0",
        batch_size: int = 8,
        num_workers: int = 4,
        roi_backend: str = "auto",
        proxy_max_side: int | None = None,
        roi_cache_dir: str | Path | None = None,
        strict_roi_backend: bool = False,
        progress_interval_s: float = 1.0,
    ) -> None:
        self.boundary_checkpoint = str(boundary_checkpoint)
        suffix = Path(boundary_checkpoint).suffix.lower()
        backend_by_suffix = {".pth": "rfdetr", ".onnx": "onnx", ".pt": "yolo"}
        if suffix not in backend_by_suffix:
            raise ValueError(f"边界模型仅支持 .pth、.onnx 或 .pt，实际为: {boundary_checkpoint}")
        self.boundary_backend = backend_by_suffix[suffix]
        self.boundary_resolution = boundary_resolution
        self.boundary_conf = boundary_conf
        self.padding = padding
        self.nms_iou = nms_iou
        self.square_stretch = square_stretch
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.roi_backend = roi_backend
        self.proxy_max_side = proxy_max_side or boundary_resolution
        self.roi_cache_dir = roi_cache_dir
        self.strict_roi_backend = strict_roi_backend
        self.progress_interval_s = progress_interval_s

        # 边界模型只加载一次，所有大图共享同一个实例。
        print(f"[i] 加载大图边界检测器: {self.boundary_checkpoint}")
        if self.boundary_backend == "yolo":
            from ultralytics import YOLO

            self.boundary_model = YOLO(self.boundary_checkpoint)
        elif self.boundary_backend == "onnx":
            from rfdetr.export._onnx.inference import ONNXDetector

            providers = None if str(device).startswith("cuda") else ["CPUExecutionProvider"]
            self.boundary_model = ONNXDetector(self.boundary_checkpoint, providers=providers)
        else:
            from rfdetr import RFDETR

            self.boundary_model = RFDETR.from_checkpoint(
                self.boundary_checkpoint,
                resolution=self.boundary_resolution,
            )
        self.last_stats: dict[str, float | str] = {}

    def release_boundary_model(self) -> None:
        """释放边界模型引用和 CUDA 缓存。"""
        boundary_model = self.boundary_model
        self.boundary_model = None
        del boundary_model
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_crops(
        self,
        image_paths: list[Path],
    ) -> tuple[
        list[tuple[str, Path, tuple[int, int, int, int]]],
        dict[str, list[tuple[float, float, float, float]]],
        dict[str, float],
    ]:
        """批量检测边界并返回带原图偏移的 crop 来源。"""
        if not image_paths:
            return [], {}, {"boundary_seconds": 0.0, "crop_prepare_seconds": 0.0}

        boundary_t0 = time.perf_counter()
        proxy_t0 = time.perf_counter()
        proxy_records: list[tuple[str, np.ndarray, tuple[int, int]]] = []
        fallback_count = 0
        for image_path in image_paths:
            source = create_image_source(
                image_path,
                backend=self.roi_backend,
                cache_dir=self.roi_cache_dir,
                strict=self.strict_roi_backend,
            )
            fallback_count += int(source.used_fallback)
            proxy, original_size = source.read_proxy(self.proxy_max_side)
            proxy_records.append((image_path.stem, proxy, original_size))
        proxy_elapsed = time.perf_counter() - proxy_t0
        print(f"[i] proxy 边界检测开始: {len(image_paths)} 张 | side={self.proxy_max_side} | batch={self.batch_size}", flush=True)
        try:
            boundary_results = predict_proxy_boundaries(
                self.boundary_model,
                proxy_records,
                device=self.device,
                resolution=self.boundary_resolution,
                conf_threshold=self.boundary_conf,
                batch_size=self.batch_size,
                backend=self.boundary_backend,
                square_stretch=self.square_stretch,
                progress_interval_s=self.progress_interval_s,
            )
        finally:
            self.release_boundary_model()
        boundary_elapsed = time.perf_counter() - boundary_t0
        boundary_by_id = {result["image_id"]: result for result in boundary_results}

        crop_prepare_t0 = time.perf_counter()
        crop_sources: list[tuple[str, Path, tuple[int, int, int, int]]] = []
        crop_boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
        last_progress = crop_prepare_t0
        for image_path in image_paths:
            stem = image_path.stem
            batch_result = boundary_by_id.get(stem)
            if batch_result is None:
                continue
            proxy_w, proxy_h = batch_result["proxy_size"]
            image_w, image_h = batch_result["original_size"]
            if self.boundary_backend == "yolo":
                boxes_proxy = batch_result["boxes"]
            else:
                boxes_proxy = map_boxes_to_original(
                    batch_result["boxes"], batch_result["scale"], batch_result["pad_x"], batch_result["pad_y"], proxy_w, proxy_h
                )
            boxes_orig = boxes_proxy.astype(np.float64, copy=True)
            boxes_orig[:, [0, 2]] *= float(image_w) / max(proxy_w, 1)
            boxes_orig[:, [1, 3]] *= float(image_h) / max(proxy_h, 1)
            if self.nms_iou > 0 and boxes_orig.shape[0] > 0:
                boxes_orig = _nms_boxes(boxes_orig, self.nms_iou)
            offsets: list[tuple[int, int, int, int]] = []
            for box in boxes_orig:
                crop_xyxy = _crop_bounds((image_h, image_w), tuple(int(value) for value in box), self.padding)
                crop_sources.append((stem, image_path, crop_xyxy))
                offsets.append(crop_xyxy)
            crop_boxes_by_image[stem] = [(float(x0), float(y0), float(x1), float(y1)) for x0, y0, x1, y1 in offsets]
            now = time.perf_counter()
            if self.progress_interval_s > 0 and (now - last_progress >= self.progress_interval_s or image_path == image_paths[-1]):
                print(f"\r[i] ROI 元数据准备 {len(crop_boxes_by_image)}/{len(image_paths)}", end="", flush=True)
                if image_path == image_paths[-1]:
                    print(flush=True)
                last_progress = now
        crop_prepare_elapsed = time.perf_counter() - crop_prepare_t0
        self.last_stats = {
            "proxy_seconds": proxy_elapsed,
            "boundary_seconds": boundary_elapsed,
            "crop_prepare_seconds": crop_prepare_elapsed,
            "crop_count": float(len(crop_sources)),
            "fallback_count": float(fallback_count),
        }
        return crop_sources, crop_boxes_by_image, {
            "proxy_seconds": proxy_elapsed,
            "boundary_seconds": boundary_elapsed,
            "fallback_count": float(fallback_count),
            "crop_prepare_seconds": crop_prepare_elapsed,
        }

    def prepare_one(
        self,
        image_path: Path,
    ) -> tuple[
        list[tuple[str, Path, tuple[int, int, int, int]]],
        list[tuple[float, float, float, float]],
        dict[str, float],
    ]:
        """为单张大图执行边界检测并生成 crop 来源，保留边界模型供后续图像复用。"""
        boundary_t0 = time.perf_counter()
        proxy_t0 = boundary_t0
        source = create_image_source(
            image_path,
            backend=self.roi_backend,
            cache_dir=self.roi_cache_dir,
            strict=self.strict_roi_backend,
        )
        proxy, original_size = source.read_proxy(self.proxy_max_side)
        proxy_elapsed = time.perf_counter() - proxy_t0
        boundary_results = predict_proxy_boundaries(
            self.boundary_model,
            [(image_path.stem, proxy, original_size)],
            device=self.device,
            resolution=self.boundary_resolution,
            conf_threshold=self.boundary_conf,
            batch_size=1,
            backend=self.boundary_backend,
            square_stretch=self.square_stretch,
            progress_interval_s=0.0,
        )
        if self.device.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
        boundary_elapsed = time.perf_counter() - boundary_t0
        result = boundary_results[0]
        proxy_w, proxy_h = result["proxy_size"]
        image_w, image_h = result["original_size"]
        if self.boundary_backend == "yolo":
            boxes_proxy = result["boxes"]
        else:
            boxes_proxy = map_boxes_to_original(
                result["boxes"], result["scale"], result["pad_x"], result["pad_y"], proxy_w, proxy_h
            )
        boxes_orig = boxes_proxy.astype(np.float64, copy=True)
        boxes_orig[:, [0, 2]] *= float(image_w) / max(proxy_w, 1)
        boxes_orig[:, [1, 3]] *= float(image_h) / max(proxy_h, 1)
        if self.nms_iou > 0 and boxes_orig.shape[0] > 0:
            boxes_orig = _nms_boxes(boxes_orig, self.nms_iou)
        crop_sources: list[tuple[str, Path, tuple[int, int, int, int]]] = []
        crop_boxes: list[tuple[float, float, float, float]] = []
        crop_t0 = time.perf_counter()
        for box in boxes_orig:
            crop_xyxy = _crop_bounds((image_h, image_w), tuple(int(value) for value in box), self.padding)
            crop_sources.append((image_path.stem, image_path, crop_xyxy))
            crop_boxes.append(tuple(float(value) for value in crop_xyxy))
        return crop_sources, crop_boxes, {
            "proxy_seconds": proxy_elapsed,
            "boundary_seconds": boundary_elapsed,
            "crop_prepare_seconds": time.perf_counter() - crop_t0,
            "fallback_count": float(source.used_fallback),
        }
