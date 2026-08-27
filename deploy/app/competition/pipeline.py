"""模块化比赛推理流水线。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from competition.config import SubmissionConfig
from competition.contracts import InferenceTask, RawDetection
from competition.detector.multi_backend import MultiBackendDetector
from competition.postprocess.base import Postprocessor
from competition.preprocess.base import Preprocessor
from competition.registry import build_postprocessor, build_preprocessor


class CompetitionPipeline:
    """串联预处理、按角色检测和比赛后处理。"""

    def __init__(
        self,
        config: SubmissionConfig,
        preprocessor: Preprocessor,
        detector: MultiBackendDetector,
        postprocessor: Postprocessor,
    ) -> None:
        """保存已经由注册表构建好的流水线模块。"""
        self._config = config
        self._preprocessor = preprocessor
        self._detector = detector
        self._postprocessor = postprocessor

    @classmethod
    def from_config(cls, config: SubmissionConfig) -> "CompetitionPipeline":
        """从严格校验后的 YAML 配置构建所有模块。"""
        return cls(
            config=config,
            preprocessor=build_preprocessor(config.preprocess),
            detector=MultiBackendDetector(config.detector),
            postprocessor=build_postprocessor(config.postprocess),
        )

    def check_gpu(self) -> None:
        """检查所有启用检测角色均可在 GPU 上运行。"""
        self._detector.check_gpu()

    def predict(self, image_id: str, image: Image.Image) -> list[dict[str, object]]:
        """对一张内存图像完成完整的比赛推理。"""
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        main_resolution = self._config.detector.roles["main"].resolution
        boundary_role = self._config.detector.roles.get("boundary")
        plan = self._preprocessor.prepare(
            image_id=image_id,
            pixels=pixels,
            main_resolution=main_resolution,
            boundary_resolution=boundary_role.resolution if boundary_role is not None else None,
        )
        main_tasks = plan.main_tasks
        if plan.boundary_task is not None:
            boundary_detections = self._restore(
                plan.boundary_task,
                self._detector.predict("boundary", [plan.boundary_task]),
            )
            main_tasks = self._preprocessor.expand(plan, boundary_detections, main_resolution)
        detections = self._restore_many(main_tasks, self._detector.predict("main", main_tasks))
        return self._postprocessor.process(detections, plan.original_size)

    @staticmethod
    def _restore(task: InferenceTask, detections: list[RawDetection]) -> list[RawDetection]:
        """将单个模型任务的检测框恢复至原图坐标。"""
        return [
            RawDetection(
                image_id=item.image_id,
                class_id=item.class_id,
                score=item.score,
                xyxy=task.transform.restore(item.xyxy),
                task_index=item.task_index,
            )
            for item in detections
        ]

    @staticmethod
    def _restore_many(tasks: list[InferenceTask], detections: list[RawDetection]) -> list[RawDetection]:
        """按解码输出的任务顺序恢复一批检测框。"""
        if not tasks or not detections:
            return []
        task_by_index: dict[int, InferenceTask] = {}
        for task in tasks:
            if task.task_index in task_by_index:
                raise RuntimeError("主检测任务编号重复")
            task_by_index[task.task_index] = task
        restored: list[RawDetection] = []
        for item in detections:
            task = task_by_index.get(item.task_index)
            if task is None:
                raise RuntimeError("检测结果引用了不存在的主检测任务")
            restored.extend(CompetitionPipeline._restore(task, [item]))
        return restored
