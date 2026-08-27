"""检测器后端的稳定接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from competition.contracts import InferenceTask, RawDetection


class Detector(ABC):
    """执行已完成几何预处理任务的检测器。"""

    @abstractmethod
    def check_gpu(self) -> None:
        """校验该后端将实际使用 GPU 推理。"""

    @abstractmethod
    def predict(self, tasks: list[InferenceTask]) -> list[RawDetection]:
        """对同一角色的一批任务执行推理。"""
