# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR ONNX Runtime 推理辅助函数和现有评估 runtime 适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image as PILImage
from supervision import Detections

from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _static_shape_dim(shape: list[Any], idx: int) -> int | None:
    """从 ONNX 输出形状中读取静态维度；符号/``None``/非正维度返回 ``None``。"""
    try:
        dim = int(shape[idx])
    except (TypeError, ValueError, IndexError):
        return None
    return dim if dim > 0 else None


def _static_dims(shape: list[Any], idx: int) -> tuple[int, int] | None:
    """读取形状中 ``idx`` 与 ``idx+1`` 两维；任一维符号化则返回 ``None``。"""
    first = _static_shape_dim(shape, idx)
    second = _static_shape_dim(shape, idx + 1)
    if first is None or second is None:
        return None
    return first, second


class ONNXDetector:
    """把 RF-DETR ONNX 图包装成现有评估 runtime 使用的模型接口。"""

    def __init__(self, model_path: str | Path, *, providers: list[str] | None = None, num_select: int = 300) -> None:
        """创建 ONNX Runtime session，并读取固定输入规格。"""
        self.session = _create_onnx_session(model_path, providers=providers)
        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape
        if len(shape) != 4:
            raise ValueError(f"ONNX 输入必须是 NCHW 四维张量，实际为 {shape!r}")
        self.input_name = input_meta.name
        if any(dim is None or isinstance(dim, str) for dim in shape[1:]):
            raise ValueError(f"当前 ONNX detector 需要固定通道和空间输入，实际为 {shape!r}")
        self.channels = int(shape[1])
        self.resolution = int(shape[2])
        if shape[2] != shape[3]:
            raise ValueError(f"当前 ONNX detector 需要固定方形输入，实际为 {shape!r}")
        self.means = [0.485, 0.456, 0.406][: self.channels]
        self.stds = [0.229, 0.224, 0.225][: self.channels]
        self.postprocess = PostProcess(num_select=num_select)
        self.model = self
        self._resolve_output_layout()

    def _resolve_output_layout(self) -> None:
        """解析 boxes/logits 输出并缓存名称、下标与静态形状（构造时一次性完成）。"""
        outputs = self.session.get_outputs()
        names = [item.name.lower() for item in outputs]
        boxes_idx = next((i for i, name in enumerate(names) if "det" in name or "box" in name), None)
        logits_idx = next((i for i, name in enumerate(names) if "label" in name or "logit" in name), None)
        if boxes_idx is None or logits_idx is None:
            candidates = [i for i, item in enumerate(outputs) if len(item.shape) == 3 and item.shape[-1] == 4]
            other = [i for i, item in enumerate(outputs) if len(item.shape) == 3 and item.shape[-1] != 4]
            if len(candidates) != 1 or len(other) != 1:
                raise ValueError(f"无法识别 ONNX 输出，输出形状为 {[item.shape for item in outputs]}")
            boxes_idx, logits_idx = candidates[0], other[0]
        self._boxes_idx = boxes_idx
        self._logits_idx = logits_idx
        self._boxes_name = outputs[boxes_idx].name
        self._logits_name = outputs[logits_idx].name
        boxes_dims = _static_dims(outputs[boxes_idx].shape, 1)
        logits_dims = _static_dims(outputs[logits_idx].shape, 1)
        if boxes_dims is None or logits_dims is None:
            # query 维可能是符号化（如 RF-DETR 导出的 Concatdets_dim_1 / Addlabels_dim_1），
            # 用一次 dummy 前向发现实际输出形状（构造时仅一次）。
            dummy = np.zeros((1, self.channels, self.resolution, self.resolution), dtype=np.float32)
            discovered = self.session.run(None, {self.input_name: dummy})
            boxes_dims = (int(discovered[boxes_idx].shape[1]), int(discovered[boxes_idx].shape[2]))
            logits_dims = (int(discovered[logits_idx].shape[1]), int(discovered[logits_idx].shape[2]))
        self._boxes_dims = boxes_dims
        self._logits_dims = logits_dims
        active_providers = self.session.get_providers()
        self._uses_cuda = bool(active_providers) and active_providers[0].startswith("CUDA")

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行归一化后的 NCHW batch，并返回与 RF-DETR 相同的原始输出。"""
        if batch.ndim != 4 or batch.shape[1] != self.channels:
            raise ValueError(f"ONNX 输入形状不匹配，期望通道数 {self.channels}，实际为 {tuple(batch.shape)}")
        if batch.is_cuda and self._uses_cuda:
            return self._forward_cuda_iobinding(batch)
        return self._forward_cpu_numpy(batch)

    def _forward_cuda_iobinding(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """通过 IOBinding 让输入/输出张量全程留在 CUDA 显存，消除 CPU↔GPU 搬运。"""
        batch = batch.detach().contiguous()
        b = batch.shape[0]
        device_id = batch.get_device()
        io_binding = self.session.io_binding()
        io_binding.bind_input(
            self.input_name, "cuda", device_id, np.float32, list(batch.shape), batch.data_ptr()
        )
        boxes = torch.empty((b, *self._boxes_dims), dtype=torch.float32, device=batch.device)
        logits = torch.empty((b, *self._logits_dims), dtype=torch.float32, device=batch.device)
        io_binding.bind_output(
            self._boxes_name, "cuda", device_id, np.float32, [b, *self._boxes_dims], boxes.data_ptr()
        )
        io_binding.bind_output(
            self._logits_name, "cuda", device_id, np.float32, [b, *self._logits_dims], logits.data_ptr()
        )
        self.session.run_with_iobinding(io_binding)
        return {"pred_boxes": boxes, "pred_logits": logits}

    def _forward_cpu_numpy(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """CPU 回退路径：保持原有 numpy 语义，仅改用缓存的输出下标。"""
        outputs = self.session.run(None, {self.input_name: batch.detach().cpu().numpy().astype("float32", copy=False)})
        return {
            "pred_boxes": torch.from_numpy(outputs[self._boxes_idx]).to(batch.device),
            "pred_logits": torch.from_numpy(outputs[self._logits_idx]).to(batch.device),
        }

    @property
    def device(self) -> torch.device:
        """返回 ONNX Runtime 输入输出使用的设备标识。"""
        return torch.device("cpu")

    def to(self, device: str | torch.device) -> ONNXDetector:
        """兼容 RF-DETR 模型包装器的设备迁移接口。"""
        del device
        return self

    def eval(self) -> ONNXDetector:
        """兼容 ``nn.Module.eval`` 接口。"""
        return self

    __call__ = forward


def _create_onnx_session(model_path: str | Path, providers: list[str] | None = None) -> Any:
    """Load an ONNX model and create an ONNX Runtime inference session.

    Imports ``onnxruntime`` at call time so that the rest of the package remains usable without it installed.  Input and
    output names / shapes are logged at DEBUG level for troubleshooting.

    When ``providers`` is ``None``, the session auto-selects the best available backend: CUDA if ``onnxruntime-gpu`` is
    installed, otherwise CPU (with a warning).  Pass an explicit list to pin the backend — useful for benchmarking
    CPU vs CUDA side-by-side.

    Args:
        model_path: Path to the ``.onnx`` model file.
        providers: Ordered list of ORT execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.  When ``None`` (default), the best available
            provider is selected automatically.

    Returns:
        An ``onnxruntime.InferenceSession`` ready for inference.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.

    Examples:
        .. code-block:: python

            sess = _create_onnx_session("model.onnx")
            print(sess.get_inputs()[0].name)
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX Runtime inference requires 'onnxruntime'. Install it: `pip install onnxruntime`"
        ) from exc

    if providers is None:
        _preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _available = ort.get_available_providers()
        providers = [p for p in _preferred if p in _available] or ["CPUExecutionProvider"]
        if providers[0] == "CPUExecutionProvider":
            logger.warning(
                "CUDAExecutionProvider not available — running ONNX inference on CPU. "
                "Install onnxruntime-gpu for GPU acceleration: `pip install onnxruntime-gpu`"
            )
    session = ort.InferenceSession(str(model_path), providers=providers)
    logger.debug("ONNX Runtime providers in use: %s", session.get_providers())
    for inp in session.get_inputs():
        logger.debug("Input  : name=%s  shape=%s  type=%s", inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        logger.debug("Output : name=%s  shape=%s  type=%s", out.name, out.shape, out.type)
    return session


def _preprocess_pil_to_nchw(
    image: PILImage.Image,
    height: int,
    width: int,
    channels: int = 3,
) -> NDArray[np.float32]:
    """Resize and normalise a PIL image to an ``(1, C, H, W)`` float32 NCHW tensor.

    Resizes using ``BILINEAR`` to match ``torchvision.transforms.functional.resize()`` (PIL's default is ``BICUBIC``
    which produces slightly different values and can lower confidence scores).  Normalises with ImageNet statistics:
    ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.

    Args:
        image: Input PIL image; any mode — converted to ``"RGB"`` (3-channel) or ``"L"`` (1-channel) internally.
        height: Target spatial height expected by the model.
        width: Target spatial width expected by the model.
        channels: Number of channels the model expects (``1`` for grayscale, ``3`` for RGB).

    Returns:
        Float32 ndarray of shape ``(1, channels, height, width)``.

    Examples:
        .. code-block:: python

            inp = _preprocess_pil_to_nchw(image, height=640, width=640)
    """
    _imagenet_mean = [0.485, 0.456, 0.406]
    _imagenet_std = [0.229, 0.224, 0.225]
    mean = np.array([_imagenet_mean[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([_imagenet_std[i % 3] for i in range(channels)], dtype=np.float32)
    pil_mode = "L" if channels == 1 else "RGB"
    # BILINEAR matches torchvision default; PIL default (BICUBIC) causes confidence drop
    arr = (
        np.array(
            image.convert(pil_mode).resize((width, height), PILImage.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    if arr.ndim == 2:  # "L" → (H, W); needs (H, W, 1)
        arr = arr[:, :, np.newaxis]
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, C, H, W)


def _run_inference(
    session: Any,
    image_path: str | Path,
    threshold: float = 0.3,
) -> tuple[Detections, PILImage.Image]:
    """Preprocess one image, run ONNX Runtime inference, and decode detections.

    Reads input shape from the session (NCHW ``float32``), resizes and normalises the image with ImageNet statistics,
    invokes the model, then decodes the ``dets`` / ``labels`` output tensors into a :class:`supervision.Detections`
    object with pixel-space ``xyxy`` boxes.

    **Input contract** (must match ``RFDETR.predict()`` preprocessing exactly):

    - Image is opened as-is and converted to ``"RGB"`` (3-channel) or ``"L"``
      (1-channel greyscale) depending on the model's channel count.
    - Resize uses ``PIL.Image.Resampling.BILINEAR`` — matching
      ``torchvision.transforms.functional.resize()`` which defaults to ``InterpolationMode.BILINEAR``.  Using PIL's
      default (``BICUBIC``) would produce slightly different pixel values and can degrade confidence.
    - Pixel values are scaled to ``[0, 1]`` then normalised with ImageNet
      statistics: ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.
    - The tensor is kept as ``[1, C, H, W]`` (NCHW) — unlike the TFLite helper
      which uses NHWC because ``onnx2tf`` transposes at export time.  ONNX RT consumes the native ONNX NCHW layout
      directly.

    Args:
        session: ONNX Runtime ``InferenceSession`` returned by
            ``_create_onnx_session``.
        image_path: Path to the input image (any format supported by Pillow).
            RGB images are used as-is; RGBA / palette images are converted.
        threshold: Confidence threshold; detections below this are discarded.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains pixel-space ``xyxy`` boxes and ``pil_img`` is
        the original PIL image at its original resolution.

    Examples:
        .. code-block:: python

            sess = _create_onnx_session("model.onnx")
            dets, img = _run_inference(sess, "photo.jpg", threshold=0.3)
            print(dets.confidence)
    """
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_name = inputs[0].name
    # ONNX NCHW: [batch, channels, height, width]
    _, channels, height, width = inputs[0].shape

    with PILImage.open(image_path) as pil_img:
        inp_tensor = _preprocess_pil_to_nchw(pil_img, height, width, channels)

    raw_outputs = session.run(None, {input_name: inp_tensor})

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to output reordering.
    output_names = [out.name for out in outputs]
    boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), None)
    logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), None)
    if boxes_idx is None or logits_idx is None:
        # Fall back to shape-based matching: boxes (*, 4) and logits (*, num_classes+1).
        logger.warning(
            "Name-based ONNX output matching failed (available names: %s). Falling back to shape-based matching.",
            output_names,
        )
        shape_boxes_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] == 4
        ]
        shape_logits_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] != 4
        ]
        if len(shape_boxes_candidates) == 1 and len(shape_logits_candidates) == 1:
            boxes_idx = shape_boxes_candidates[0]
            logits_idx = shape_logits_candidates[0]
        elif len(raw_outputs) == 2:
            # Ambiguous shapes (e.g. num_classes==3 → logits dim==4 == boxes dim).
            # ONNX preserves output order: index 0 = dets (boxes), index 1 = labels (logits).
            logger.warning(
                "Shape-based ONNX output matching is ambiguous (both outputs have last dim==4, "
                "which happens when num_classes==3).  Falling back to positional order: "
                "output 0 = boxes ('dets'), output 1 = logits ('labels').  "
                "If detections look wrong, inspect output names with _create_onnx_session() "
                "and set LOG_LEVEL=DEBUG."
            )
            boxes_idx = 0
            logits_idx = 1
        else:
            available_shapes = [list(arr_out.shape) for arr_out in raw_outputs]
            raise ValueError(
                f"Shape-based ONNX output matching failed. Expected exactly one rank-3 tensor with "
                f"last dim == 4 (boxes) and one rank-3 tensor with last dim != 4 (logits). "
                f"Available output shapes: {available_shapes}"
            )

    boxes_cwh = raw_outputs[boxes_idx][0]  # (Q, 4) normalised cxcywh
    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = raw_outputs[logits_idx][0, :, :-1]  # (Q, num_classes)

    # RF-DETR uses per-class sigmoid (not softmax) — mirrors PostProcess.forward in postprocess.py.
    logger.debug(
        "Logits stats: shape=%s min=%.3f max=%.3f mean=%.3f",
        logits.shape,
        float(logits.min()),
        float(logits.max()),
        float(logits.mean()),
    )
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    scores = scores_all.max(axis=-1)
    cls = scores_all.argmax(axis=-1)
    logger.debug(
        "Scores stats: min=%.3f max=%.3f — detections above threshold %.2f: %d",
        float(scores.min()),
        float(scores.max()),
        threshold,
        int((scores > threshold).sum()),
    )
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cwh[keep].T
    ow, oh = pil_img.size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    return Detections(xyxy=xyxy, confidence=scores[keep], class_id=cls[keep].astype(int)), pil_img


# Benchmarking helper — not part of production inference API; subject to removal.
def _onnx_runtime(
    onnx_path: Path | str,
    image: PILImage.Image,
    providers: list[str],
    warmup: int = 20,
    runs: int = 100,
) -> tuple[float, float, str]:
    """Benchmark ONNX Runtime inference for one image and provider list.

    Creates a fresh ``InferenceSession`` with the requested providers, preprocesses ``image`` once using ImageNet
    normalisation, then runs timed inference with ``time.perf_counter``.  GPU timings may underestimate real latency
    if the CUDA execution provider is configured for asynchronous execution; for accurate GPU timing use CUDA events.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        image: Input image (any size); resized to the model's expected spatial resolution.
        providers: Ordered list of ORT execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        warmup: Number of un-timed warm-up runs before measurement begins.
        runs: Number of timed runs used to compute statistics.

    Returns:
        A ``(mean_ms, std_ms, provider_label)`` tuple where ``provider_label`` is the first active provider with
        ``"ExecutionProvider"`` stripped, e.g. ``"CUDA"`` or ``"CPU"``.

    Examples:
        .. code-block:: python

            mean_ms, std_ms, label = _onnx_runtime("model.onnx", image, ["CPUExecutionProvider"])
            print(f"{label}: {mean_ms:.1f} ms ± {std_ms:.1f}")
    """
    import time

    sess = _create_onnx_session(onnx_path, providers=providers)
    active = sess.get_providers()[0]
    if active != providers[0]:
        raise RuntimeError(
            f"Requested provider {providers[0]!r} not active — ORT fell back to {active!r}. "
            "Install onnxruntime-gpu: `pip install onnxruntime-gpu`"
        )
    input_meta = sess.get_inputs()[0]
    _, channels, height, width = input_meta.shape
    inp = _preprocess_pil_to_nchw(image, height, width, channels)
    feed = {input_meta.name: inp}

    for _ in range(warmup):
        sess.run(None, feed)
    timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr_t = np.array(timings)
    provider_label = sess.get_providers()[0].replace("ExecutionProvider", "")
    return float(arr_t.mean()), float(arr_t.std()), provider_label
