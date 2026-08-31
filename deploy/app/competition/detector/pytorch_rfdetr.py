"""PyTorch RF-DETR GPU 检测器。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from competition.config import DetectorConfig, RoleConfig
from competition.contracts import InferenceTask, RawDetection
from competition.detector.base import Detector
from competition.detector.batch import clamp_batch, safe_batch_forward
from competition.detector.decoding import decode_rfdetr_outputs

logger = logging.getLogger("rf-detr.runtime")

# 启动标定的两个探测批量，差值除以图数得到单图增量显存。
_PROBE_LOW = 1
_PROBE_HIGH = 5


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
        self._resolution = role.resolution
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
        self._batch_size = self._calibrate_batch(detector_config.batch_size)

    def check_gpu(self) -> None:
        """验证 PyTorch CUDA 可用且所选设备是 CUDA。"""
        if not self._device.startswith("cuda") or not self._torch.cuda.is_available():
            raise RuntimeError("PyTorch 后端要求可用的 CUDA 设备")

    def _calibrate_batch(self, requested: int) -> int:
        """按空闲显存把请求的批量下探到安全值。

        通过两个探测批量（1 与 5）在模型分辨率上前向，用 torch 缓存分配器峰值之差除以
        图数，估算单图增量显存，再按空闲显存与安全比例 clamp。标定失败（如非常小显存
        甚至在探测时 OOM）时保守返回 ``requested``，交由运行时 OOM 退避兜底。
        """
        torch = self._torch
        device = self._device
        res = self._resolution
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            free_mb = free_bytes / 1024 / 1024
        except Exception as exc:  # noqa: BLE001 - 标定只是优化，失败不影响正确性
            logger.warning("读取空闲显存失败(%s)，保持请求 batch=%d", exc, requested)
            return requested

        def peak_for(batch: int) -> float:
            dummy = torch.zeros(batch, 3, res, res, device=device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                _ = self._model.model.model(dummy)
            torch.cuda.synchronize(device)
            return torch.cuda.max_memory_allocated(device) / 1024 / 1024

        try:
            low = peak_for(_PROBE_LOW)
            high = peak_for(_PROBE_HIGH)
        except Exception as exc:  # noqa: BLE001 - 标定只是优化，失败不影响正确性
            logger.warning("显存标定失败(%s)，保持请求 batch=%d", exc, requested)
            return requested

        per_image_mb = (high - low) / (_PROBE_HIGH - _PROBE_LOW)
        eff = clamp_batch(requested, free_mb, per_image_mb)
        logger.info(
            "启动显存标定: free=%.0fMB, 每图增量≈%.0fMB, 请求 batch=%d → 安全 batch=%d",
            free_mb,
            per_image_mb,
            requested,
            eff,
        )
        return eff

    def predict(self, tasks: list[InferenceTask]) -> list[RawDetection]:
        """按训练时归一化方式执行原始 RF-DETR 前向推理，支持批量与 OOM 退避。"""
        if not tasks:
            return []
        means = self._torch.tensor(self._model.means, device=self._device).view(1, 3, 1, 1)
        stds = self._torch.tensor(self._model.stds, device=self._device).view(1, 3, 1, 1)

        def run_slice(start: int, count: int) -> list[RawDetection]:
            """对 tasks[start:start+count] 组成一个 batch 前向并解码。"""
            batch_tasks = tasks[start : start + count]
            batch = np.stack([item.pixels for item in batch_tasks]).astype(np.float32) / 255.0
            tensor = self._torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).to(self._device)
            tensor = (tensor - means) / stds
            with self._torch.inference_mode():
                raw = self._model.model.model(tensor)
            return decode_rfdetr_outputs(
                raw["pred_logits"].detach().float().cpu().numpy(),
                raw["pred_boxes"].detach().float().cpu().numpy(),
                batch_tasks,
            )

        outputs, eff = safe_batch_forward(
            len(tasks),
            self._batch_size,
            run_slice,
            is_oom=lambda e: isinstance(e, self._torch.cuda.OutOfMemoryError),
            on_oom=self._torch.cuda.empty_cache,
        )
        # 持久化退避后的批量，供后续图像复用，避免重复 OOM。
        if eff != self._batch_size:
            logger.info("本次推理将批量持久化为 %d 以适配当前显存", eff)
            self._batch_size = eff
        return outputs
