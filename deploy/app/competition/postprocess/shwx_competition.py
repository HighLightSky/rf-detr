"""SHWX 比赛 COCO 风格输出后处理。"""

from __future__ import annotations

from pathlib import Path

import yaml

from competition.config import PostprocessConfig
from competition.contracts import RawDetection
from competition.postprocess.base import Postprocessor
from competition.postprocess.nms import apply_fsc_containment_nms, apply_ms_nms


class ShwxCompetitionPostprocessor(Postprocessor):
    """按比赛 JSON 契约输出经过实验配置去重的检测框。"""

    def __init__(self, config: PostprocessConfig) -> None:
        """加载受控资源中的类别编号和类别名称。"""
        self._config = config
        resource_path = Path(__file__).resolve().parents[1] / config.class_names_path
        if not resource_path.is_file():
            raise FileNotFoundError(f"类别名称资源不存在: {resource_path}")
        loaded = yaml.safe_load(resource_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("class_names"), list):
            raise ValueError("类别名称资源必须包含 class_names 列表")
        self._class_names = [str(name) for name in loaded["class_names"]]

    def process(
        self,
        detections: list[RawDetection],
        image_size: tuple[int, int],
    ) -> list[dict[str, object]]:
        """过滤并去重后构造比赛要求的 objects 数组。"""
        width, height = image_size
        candidates: list[RawDetection] = []
        for item in detections:
            if item.score < self._config.confidence_threshold:
                continue
            if item.class_id < 0 or item.class_id >= len(self._class_names):
                continue
            left, top, right, bottom = item.xyxy
            bounded = (max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom))
            if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                continue
            if (
                item.class_id == self._config.ms_nms.ms_class_id
                and self._config.ms_min_box_area > 0.0
                and (bounded[2] - bounded[0]) * (bounded[3] - bounded[1]) < self._config.ms_min_box_area
            ):
                continue
            candidates.append(
                RawDetection(
                    image_id=item.image_id,
                    class_id=item.class_id,
                    score=item.score,
                    xyxy=bounded,
                    task_index=item.task_index,
                )
            )
        candidates = apply_fsc_containment_nms(candidates, self._config.fsc_containment_nms)
        candidates = apply_ms_nms(candidates, self._config.ms_nms)
        return [
            {
                "category_id": item.class_id,
                "category_name": self._class_names[item.class_id],
                "score": item.score,
                "bbox": [item.xyxy[0], item.xyxy[1], item.xyxy[2], item.xyxy[3]],
            }
            for item in candidates
        ]
