"""按角色组合 PyTorch 与 ONNX 检测器。"""

from __future__ import annotations

from competition.config import DetectorConfig
from competition.contracts import InferenceTask, RawDetection
from competition.detector.base import Detector
from competition.detector.onnx_rfdetr import OnnxRfdetrDetector
from competition.detector.pytorch_rfdetr import PytorchRfdetrDetector


class MultiBackendDetector:
    """为 main 和 boundary 角色分别选择已注册检测器后端。"""

    def __init__(self, config: DetectorConfig) -> None:
        """按 YAML 的受限 backend 枚举创建所有模型角色。"""
        self._detectors: dict[str, Detector] = {}
        for role_name, role in config.roles.items():
            try:
                if role.backend == "onnx":
                    detector = OnnxRfdetrDetector(role, config)
                elif role.backend == "pytorch":
                    detector = PytorchRfdetrDetector(role, config)
                else:
                    raise ValueError(f"未注册的检测器后端: {role.backend}")
            except Exception as exc:
                raise RuntimeError(
                    f"初始化检测角色 {role_name} 失败（后端: {role.backend}，模型: {role.model_path.name}）"
                ) from exc
            self._detectors[role_name] = detector

    def check_gpu(self) -> None:
        """确认每个已启用角色均会使用 GPU。"""
        for detector in self._detectors.values():
            detector.check_gpu()

    def predict(self, role: str, tasks: list[InferenceTask]) -> list[RawDetection]:
        """将任务路由给对应的模型角色。"""
        if role not in self._detectors:
            raise ValueError(f"未配置的检测器角色: {role}")
        if any(task.role != role for task in tasks):
            raise ValueError(f"检测任务角色与请求角色不一致: {role}")
        return self._detectors[role].predict(tasks)
