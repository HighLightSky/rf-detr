"""后处理模块的稳定接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from competition.contracts import RawDetection


class Postprocessor(ABC):
    """将原图坐标检测结果转换为赛事对象。"""

    @abstractmethod
    def process(
        self,
        detections: list[RawDetection],
        image_size: tuple[int, int],
    ) -> list[dict[str, object]]:
        """完成过滤、去重、类别映射和输出格式化。"""
