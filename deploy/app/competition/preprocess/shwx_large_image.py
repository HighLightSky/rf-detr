"""SHWX 大图边界检测与裁切预处理。"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from competition.config import PreprocessConfig
from competition.contracts import CoordinateTransform, ImagePlan, InferenceTask, RawDetection
from competition.preprocess.base import Preprocessor


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """计算两个 xyxy 框的交并比。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms(detections: list[RawDetection], threshold: float) -> list[RawDetection]:
    """按类别执行稳定的贪心 NMS。"""
    selected: list[RawDetection] = []
    for candidate in sorted(detections, key=lambda item: (-item.score, item.class_id, item.xyxy)):
        if all(_iou(candidate.xyxy, kept.xyxy) < threshold for kept in selected):
            selected.append(candidate)
    return selected


class ShwxLargeImagePreprocessor(Preprocessor):
    """以边界模型定位大图候选区域，再对原图裁切调用主模型。"""

    def __init__(self, config: PreprocessConfig) -> None:
        """保存大图拆分参数。"""
        self._config = config

    def prepare(
        self,
        image_id: str,
        pixels: NDArray[np.uint8],
        main_resolution: int,
        boundary_resolution: int | None,
    ) -> ImagePlan:
        """小图直接进入主模型，大图进入边界模型。"""
        if boundary_resolution is None:
            raise ValueError("大图预处理必须提供 boundary 模型分辨率")
        height, width = pixels.shape[:2]
        if max(width, height) < self._config.large_image_min_side:
            return self._direct_plan(image_id, pixels, main_resolution)

        proxy_scale = min(1.0, self._config.proxy_max_side / max(width, height))
        proxy_width = max(1, round(width * proxy_scale))
        proxy_height = max(1, round(height * proxy_scale))
        proxy = cv2.resize(pixels, (proxy_width, proxy_height), interpolation=cv2.INTER_LINEAR)
        letterbox_scale = boundary_resolution / max(proxy_width, proxy_height)
        resized_width = max(1, round(proxy_width * letterbox_scale))
        resized_height = max(1, round(proxy_height * letterbox_scale))
        resized = cv2.resize(proxy, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((boundary_resolution, boundary_resolution, 3), dtype=np.uint8)
        pad_x = (boundary_resolution - resized_width) // 2
        pad_y = (boundary_resolution - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        task = InferenceTask(
            image_id=image_id,
            role="boundary",
            pixels=canvas,
            transform=CoordinateTransform(
                scale_x=letterbox_scale * proxy_width / width,
                scale_y=letterbox_scale * proxy_height / height,
                pad_x=pad_x,
                pad_y=pad_y,
            ),
            task_index=0,
        )
        return ImagePlan(
            image_id=image_id,
            pixels=pixels,
            original_size=(width, height),
            boundary_task=task,
        )

    def expand(
        self,
        plan: ImagePlan,
        boundary_detections: list[RawDetection],
        main_resolution: int,
    ) -> list[InferenceTask]:
        """将边界候选框去重、补边并裁切为主检测任务。"""
        if plan.boundary_task is None:
            return plan.main_tasks
        candidates = [item for item in boundary_detections if item.score >= self._config.boundary_confidence]
        candidates = _nms(candidates, self._config.boundary_nms_iou)
        image_height, image_width = plan.pixels.shape[:2]
        tasks: list[InferenceTask] = []
        for task_index, candidate in enumerate(candidates):
            left, top, right, bottom = candidate.xyxy
            crop_left = max(0, int(np.floor(left)) - self._config.padding)
            crop_top = max(0, int(np.floor(top)) - self._config.padding)
            crop_right = min(image_width, int(np.ceil(right)) + self._config.padding)
            crop_bottom = min(image_height, int(np.ceil(bottom)) + self._config.padding)
            if crop_right <= crop_left or crop_bottom <= crop_top:
                continue
            crop = plan.pixels[crop_top:crop_bottom, crop_left:crop_right]
            crop_height, crop_width = crop.shape[:2]
            resized = cv2.resize(crop, (main_resolution, main_resolution), interpolation=cv2.INTER_LINEAR)
            tasks.append(
                InferenceTask(
                    image_id=plan.image_id,
                    role="main",
                    pixels=resized,
                    transform=CoordinateTransform(
                        scale_x=main_resolution / crop_width,
                        scale_y=main_resolution / crop_height,
                        offset_x=crop_left,
                        offset_y=crop_top,
                    ),
                    task_index=task_index,
                )
            )
        return tasks

    @staticmethod
    def _direct_plan(image_id: str, pixels: NDArray[np.uint8], resolution: int) -> ImagePlan:
        """为未达到大图阈值的图像创建整图任务。"""
        height, width = pixels.shape[:2]
        resized = cv2.resize(pixels, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        task = InferenceTask(
            image_id=image_id,
            role="main",
            pixels=resized,
            transform=CoordinateTransform(scale_x=resolution / width, scale_y=resolution / height),
            task_index=0,
        )
        return ImagePlan(
            image_id=image_id,
            pixels=pixels,
            original_size=(width, height),
            main_tasks=[task],
        )
