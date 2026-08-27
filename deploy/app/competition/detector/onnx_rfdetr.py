"""ONNX Runtime GPU 检测器。"""

from __future__ import annotations

import numpy as np

from competition.config import DetectorConfig, RoleConfig
from competition.contracts import InferenceTask, RawDetection
from competition.detector.base import Detector
from competition.detector.decoding import decode_rfdetr_outputs


class OnnxRfdetrDetector(Detector):
    """使用 CUDAExecutionProvider 运行 RF-DETR ONNX 模型。"""

    def __init__(self, role: RoleConfig, detector_config: DetectorConfig) -> None:
        """创建严格要求 CUDA provider 的 ONNX Runtime 会话。"""
        if role.backend != "onnx":
            raise ValueError("ONNX 检测器只能接收 onnx 角色配置")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX 后端缺少 onnxruntime-gpu 依赖") from exc
        preload_dlls = getattr(ort, "preload_dlls", None)
        if not callable(preload_dlls):
            raise RuntimeError("当前 onnxruntime-gpu 不支持 preload_dlls，请使用 1.21 或更高版本")
        try:
            preload_dlls(directory="")
        except Exception as exc:
            raise RuntimeError("无法从 Python NVIDIA 运行库预加载 CUDA/cuDNN 动态库") from exc
        self._role = role
        self._batch_size = detector_config.batch_size
        self._session = ort.InferenceSession(
            str(role.model_path),
            providers=[("CUDAExecutionProvider", {"device_id": detector_config.device.removeprefix("cuda:")})],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [item.name for item in self._session.get_outputs()]
        self.check_gpu()

    def check_gpu(self) -> None:
        """拒绝 CUDA provider 不可用导致的 CPU 回退。"""
        providers = self._session.get_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"ONNX Runtime 未启用 CUDAExecutionProvider，实际 providers: {providers}")

    def predict(self, tasks: list[InferenceTask]) -> list[RawDetection]:
        """对固定分辨率任务分批执行 ONNX GPU 推理。"""
        if not tasks:
            return []
        outputs: list[RawDetection] = []
        for start in range(0, len(tasks), self._batch_size):
            batch_tasks = tasks[start : start + self._batch_size]
            batch = np.stack([item.pixels for item in batch_tasks]).astype(np.float32) / 255.0
            batch = (batch - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
            batch = np.transpose(batch, (0, 3, 1, 2))
            result = self._session.run(self._output_names, {self._input_name: batch})
            logits, boxes = self._select_outputs(result)
            outputs.extend(decode_rfdetr_outputs(logits, boxes, batch_tasks))
        return outputs

    @staticmethod
    def _select_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """依据最后一个维度从 ONNX 输出中识别分类与回归张量。"""
        boxes = next((item for item in outputs if item.ndim == 3 and item.shape[-1] == 4), None)
        logits = next((item for item in outputs if item.ndim == 3 and item.shape[-1] != 4), None)
        if boxes is None or logits is None:
            shapes = [tuple(item.shape) for item in outputs]
            raise RuntimeError(f"ONNX 模型输出不符合 RF-DETR 格式: {shapes}")
        return logits, boxes
