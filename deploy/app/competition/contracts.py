"""组件之间传递的稳定数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CoordinateTransform:
    """将模型输入坐标恢复到原图坐标的仿射变换。"""

    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def restore(self, xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """把输入坐标系中的 xyxy 框映射回原图。"""
        if self.scale_x <= 0.0 or self.scale_y <= 0.0:
            raise ValueError("坐标变换缩放系数必须为正数")
        return (
            (xyxy[0] - self.pad_x) / self.scale_x + self.offset_x,
            (xyxy[1] - self.pad_y) / self.scale_y + self.offset_y,
            (xyxy[2] - self.pad_x) / self.scale_x + self.offset_x,
            (xyxy[3] - self.pad_y) / self.scale_y + self.offset_y,
        )


@dataclass(frozen=True)
class InferenceTask:
    """一个已完成几何预处理、等待某个模型角色推理的任务。"""

    image_id: str
    role: str
    pixels: NDArray[np.uint8]
    transform: CoordinateTransform
    task_index: int = 0

    @property
    def input_size(self) -> tuple[int, int]:
        """返回模型输入的宽高。"""
        height, width = self.pixels.shape[:2]
        return width, height


@dataclass(frozen=True)
class RawDetection:
    """模型输出或坐标恢复后的单个检测框。"""

    image_id: str
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]
    task_index: int = 0


@dataclass
class ImagePlan:
    """一张原图在预处理阶段生成的推理计划。"""

    image_id: str
    pixels: NDArray[np.uint8]
    original_size: tuple[int, int]
    main_tasks: list[InferenceTask] = field(default_factory=list)
    boundary_task: InferenceTask | None = None
