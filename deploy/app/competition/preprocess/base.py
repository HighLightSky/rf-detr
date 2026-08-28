"""预处理模块的稳定接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from competition.contracts import ImagePlan, InferenceTask, RawDetection


class Preprocessor(ABC):
    """把原图转换为主检测和边界检测任务。"""

    @abstractmethod
    def prepare(
        self,
        image_id: str,
        pixels: NDArray[np.uint8],
        main_resolution: int,
        boundary_resolution: int | None,
    ) -> ImagePlan:
        """为一张图创建第一阶段推理计划。"""

    @abstractmethod
    def expand(
        self,
        plan: ImagePlan,
        boundary_detections: list[RawDetection],
        main_resolution: int,
    ) -> list[InferenceTask]:
        """根据边界检测结果创建主检测任务。"""
