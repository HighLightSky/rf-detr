"""普通图直接推理预处理。"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from competition.contracts import CoordinateTransform, ImagePlan, InferenceTask, RawDetection
from competition.preprocess.base import Preprocessor


class DirectPreprocessor(Preprocessor):
    """将整张图等比之外的几何变化限制为方形缩放。"""

    def prepare(
        self,
        image_id: str,
        pixels: NDArray[np.uint8],
        main_resolution: int,
        boundary_resolution: int | None,
    ) -> ImagePlan:
        """生成一个覆盖整张原图的主检测任务。"""
        del boundary_resolution
        height, width = pixels.shape[:2]
        resized = cv2.resize(pixels, (main_resolution, main_resolution), interpolation=cv2.INTER_LINEAR)
        task = InferenceTask(
            image_id=image_id,
            role="main",
            pixels=resized,
            transform=CoordinateTransform(
                scale_x=main_resolution / width,
                scale_y=main_resolution / height,
            ),
            task_index=0,
        )
        return ImagePlan(
            image_id=image_id,
            pixels=pixels,
            original_size=(width, height),
            main_tasks=[task],
        )

    def expand(
        self,
        plan: ImagePlan,
        boundary_detections: list[RawDetection],
        main_resolution: int,
    ) -> list[InferenceTask]:
        """直接模式无需边界检测扩展，返回既有主任务。"""
        del boundary_detections, main_resolution
        return plan.main_tasks
