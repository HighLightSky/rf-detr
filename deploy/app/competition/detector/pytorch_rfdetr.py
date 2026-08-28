"""PyTorch RF-DETR GPU 检测器。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from competition.config import DetectorConfig, RoleConfig
from competition.contracts import InferenceTask, RawDetection
from competition.detector.base import Detector
from competition.detector.decoding import decode_rfdetr_outputs


class PytorchRfdetrDetector(Detector):
    """加载交付目录内 RF-DETR 运行时并执行 PyTorch GPU 推理。"""

    def __init__(self, role: RoleConfig, detector_config: DetectorConfig) -> None:
        """加载指定 checkpoint，并按配置挂载可选的 ProtoGuidance 工件。"""
        if role.backend != "pytorch":
            raise ValueError("PyTorch 检测器只能接收 pytorch 角色配置")
        vendor_dir = Path(__file__).resolve().parents[2] / "vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        try:
            import torch

            from rfdetr import RFDETR
        except ImportError as exc:
            raise RuntimeError("PyTorch 后端缺少 torch 或交付内 vendor/rfdetr") from exc
        self._torch = torch
        self._device = detector_config.device
        self._batch_size = detector_config.batch_size
        self.check_gpu()
        checkpoint_kwargs: dict[str, str] = {}
        if role.proto_guidance_artifact is not None:
            checkpoint_kwargs["proto_guidance_artifacts_path"] = str(role.proto_guidance_artifact)
        self._model = RFDETR.from_checkpoint(str(role.model_path), **checkpoint_kwargs)
        proto_guidance = getattr(self._model.model.model.transformer, "proto_guidance", None)
        if self._model.model_config.proto_guidance_enabled and proto_guidance is None:
            raise RuntimeError(
                f"PyTorch 模型 {role.model_path.name} 启用了 ProtoGuidance，但未能挂载匹配的原型工件"
            )
        self._model.model.model.to(self._device).eval()

    def check_gpu(self) -> None:
        """验证 PyTorch CUDA 可用且所选设备是 CUDA。"""
        if not self._device.startswith("cuda") or not self._torch.cuda.is_available():
            raise RuntimeError("PyTorch 后端要求可用的 CUDA 设备")

    def predict(self, tasks: list[InferenceTask]) -> list[RawDetection]:
        """按训练时归一化方式执行原始 RF-DETR 前向推理。"""
        if not tasks:
            return []
        outputs: list[RawDetection] = []
        means = self._torch.tensor(self._model.means, device=self._device).view(1, 3, 1, 1)
        stds = self._torch.tensor(self._model.stds, device=self._device).view(1, 3, 1, 1)
        for start in range(0, len(tasks), self._batch_size):
            batch_tasks = tasks[start : start + self._batch_size]
            batch = np.stack([item.pixels for item in batch_tasks]).astype(np.float32) / 255.0
            tensor = self._torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).to(self._device)
            tensor = (tensor - means) / stds
            with self._torch.inference_mode():
                raw = self._model.model.model(tensor)
            outputs.extend(
                decode_rfdetr_outputs(
                    raw["pred_logits"].detach().float().cpu().numpy(),
                    raw["pred_boxes"].detach().float().cpu().numpy(),
                    batch_tasks,
                )
            )
        return outputs
