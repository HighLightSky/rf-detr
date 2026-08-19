# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import functools
import importlib
import json
import os
import warnings
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from pydantic_core import PydanticUndefined
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

EncoderName: TypeAlias = Literal["dinov2_windowed_small", "dinov2_windowed_base", "dinov2_registers_windowed_small"]
PathLikeStr: TypeAlias = str | Path

__all__ = [
    "AugmentationBackend",
    "ModelConfig",
    "RFDETRBaseConfig",
    "RFDETRLargeDeprecatedConfig",
    "RFDETRNanoConfig",
    "RFDETRSmallConfig",
    "RFDETRMediumConfig",
    "RFDETRLargeConfig",
    "RFDETRSegPreviewConfig",
    "RFDETRSegNanoConfig",
    "RFDETRSegSmallConfig",
    "RFDETRSegMediumConfig",
    "RFDETRSegLargeConfig",
    "RFDETRSegXLargeConfig",
    "RFDETRSeg2XLargeConfig",
    "RFDETRKeypointPreviewConfig",
    "TrainConfig",
    "SegmentationTrainConfig",
    "KeypointTrainConfig",
]

#: Legacy augmentation-backend string aliases, mapped to their current form.
_LEGACY_AUGMENTATION_BACKEND_ALIASES: Dict[str, str] = {
    "gpu": "kornia",
    "tv": "torchvision",
    "albu": "albumentations",
}


def _package_importable(module_name: str) -> bool:
    """Return ``True`` when *module_name* can be imported.

    Args:
        module_name: Dotted module path to probe (e.g. ``"kornia.augmentation"``).

    Returns:
        ``True`` if the import succeeds, ``False`` on ``ImportError``.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


class AugmentationBackend(str, Enum):
    """Concrete augmentation backend selector for ``TrainConfig.augmentation_backend``.

    Only holds directly-usable, concrete backends — ``TV`` (torchvision), ``ALBU`` (Albumentations), and ``KORNIA``.
    ``GPU`` is a Python enum alias for ``KORNIA`` (same value ``"kornia"``): Kornia augmentation always runs on-device
    (GPU), so the two names refer to the same backend; ``GPU`` exists only so legacy ``augmentation_backend="gpu"``
    strings keep resolving correctly.

    ``"cpu"`` and ``"auto"`` are accepted as *input* strings (on ``TrainConfig.augmentation_backend`` and by
    :meth:`from_str`) but are never stored or returned as a member of this enum — they are auto-pick sentinels resolved
    to a concrete member at :meth:`from_str` call time. Resolution stays late (re-checked at dataset-build time against
    whatever is installed in the current environment) rather than baked into ``TrainConfig`` at construction time, so a
    saved config using ``"cpu"``/``"auto"`` remains portable across environments with different optional packages
    installed. Pass a concrete value (``"torchvision"``, ``"albumentations"``, or ``"kornia"``) explicitly to pin the
    backend regardless of environment.
    """

    TV = "torchvision"
    ALBU = "albumentations"
    KORNIA = "kornia"
    GPU = "kornia"  # alias for KORNIA — backward compat name; kornia is always the GPU-side path

    @classmethod
    def from_str(cls, value: str, *, has_cuda: bool = False) -> "AugmentationBackend":
        """Resolve a string to a concrete backend, auto-picking the best installed one.

        Legacy string aliases (``"gpu"``, ``"tv"``, ``"albu"``) are mapped to their current form
        first. ``"cpu"`` auto-picks the best *installed* CPU backend: Albumentations > Kornia
        (CPU) > torchvision. ``"auto"`` additionally prefers Kornia first when ``has_cuda=True``
        and Kornia is installed, then falls back to the same CPU priority. The concrete backend
        ``"cpu"``/``"auto"`` resolve to can therefore vary across environments — pass
        ``"torchvision"`` explicitly to force torchvision regardless of what's installed.

        Args:
            value: Backend name string.
            has_cuda: Whether a CUDA device is available. Only consulted for ``"auto"`` — callers
                that care about CUDA-gated GPU selection (e.g. dataset builders) compute this via
                their own fork-safe CUDA check and pass it in; this function does not probe CUDA
                itself to avoid importing device-detection code from other modules.

        Returns:
            Concrete ``AugmentationBackend`` member.

        Raises:
            ValueError: When *value* is not a recognised backend name.

        Examples:
            >>> AugmentationBackend.from_str("torchvision")
            <AugmentationBackend.TV: 'torchvision'>
            >>> AugmentationBackend.from_str("gpu")
            <AugmentationBackend.KORNIA: 'kornia'>
        """
        value = _LEGACY_AUGMENTATION_BACKEND_ALIASES.get(value, value)
        if value in ("cpu", "auto"):
            if value == "auto" and has_cuda and cls._is_kornia_available():
                return cls.KORNIA
            if cls._is_albu_available():
                return cls.ALBU
            if cls._is_kornia_available():
                return cls.KORNIA
            return cls.TV
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown augmentation_backend {value!r}; expected one of 'cpu', 'auto', 'torchvision', "
                "'albumentations', 'kornia'."
            ) from None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _is_albu_available(cls) -> bool:
        """Return ``True`` when Albumentations is importable.

        Cached for the process lifetime — package installation state does not change at runtime.
        Tests that need to simulate "not installed" should patch this method directly (e.g.
        ``patch.object(AugmentationBackend, "_is_albu_available", return_value=False)``) rather
        than blocking the underlying import, since the cache is keyed on this method, not on the
        import machinery.

        Returns:
            ``True`` if ``albumentations`` can be imported.
        """
        return _package_importable("albumentations")

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _is_kornia_available(cls) -> bool:
        """Return ``True`` when Kornia's augmentation module is importable.

        Cached for the process lifetime — see :meth:`_is_albu_available` for the caching and test
        rationale.

        Returns:
            ``True`` if ``kornia.augmentation`` can be imported.
        """
        return _package_importable("kornia.augmentation")

    @classmethod
    def _is_tv_available(cls) -> bool:
        """Return ``True`` — torchvision is a hard (non-optional) RF-DETR dependency.

        Not cached: the result is a compile-time constant, not worth the caching machinery.

        Returns:
            Always ``True``.
        """
        return True


class PretrainWeightsCompatibilityWarning(UserWarning):
    """Warning emitted when ``ModelConfig`` overrides are likely to prevent the variant's published pretrained weights
    from loading into the model — leaving large portions of the model randomly initialized and typically producing much
    lower accuracy."""


def _detect_device() -> str:
    """Detect the best available device **without** initialising the CUDA runtime.

    ``torch.cuda.is_available()`` creates a CUDA driver context that makes ``_is_in_bad_fork()`` return ``True`` in
    child processes.  This breaks fork-based DDP strategies (e.g. ``ddp_notebook``) in notebook environments.

    We defer to :func:`torch.accelerator.current_accelerator` (PyTorch ≥ 2.4) when available — it queries the driver
    through NVML without creating a primary context.  On older builds we fall back to ``torch.cuda.is_available()``.

    ``check_available=True`` is required: without it ``current_accelerator()`` only reports the *compile-time*
    accelerator, so the default CUDA wheel on a machine without an NVIDIA driver yields ``"cuda"`` and every model build
    crashes with "Found no NVIDIA driver".  The runtime check is NVML-backed and still avoids creating a CUDA context.
    Builds whose ``current_accelerator`` predates the ``check_available`` kwarg get the same runtime verification via
    ``torch.accelerator.is_available``.
    """
    accelerator = getattr(torch, "accelerator", None)
    current_accelerator = getattr(accelerator, "current_accelerator", None)
    if current_accelerator is not None:
        try:
            try:
                accel = current_accelerator(check_available=True)
            except TypeError:
                accel = current_accelerator()
                if accel is not None and not accelerator.is_available():
                    accel = None
            if accel is not None:
                return str(accel)
            return "cpu"
        except RuntimeError:
            return "cpu"
    # Fallback for PyTorch < 2.4 — this DOES create a CUDA driver context.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _detect_device()
_OPTIMIZER_MANAGED_KWARGS = {"params", "lr", "weight_decay", "fused"}


def _resolve_native_optimizer(name: str) -> type[Optimizer]:
    """Resolve a bare optimizer short name to a ``torch.optim`` optimizer class.

    Only native ``torch.optim`` optimizers may be selected by short name; the match
    is case-insensitive (``"adamw"`` → ``torch.optim.AdamW``, ``"sgd"`` → ``torch.optim.SGD``).
    Any other optimizer must be given as a full dotted import path or a callable.

    Args:
        name: A bare optimizer name (no dotted import path).

    Returns:
        The matching ``torch.optim`` optimizer class.

    Raises:
        ValueError: If ``name`` is not a native ``torch.optim`` optimizer.

    Examples:
        >>> _resolve_native_optimizer("adamw") is torch.optim.AdamW
        True
    """
    target = name.strip().lower()
    for attribute in dir(torch.optim):
        candidate = getattr(torch.optim, attribute)
        if isinstance(candidate, type) and issubclass(candidate, Optimizer) and attribute.lower() == target:
            return candidate
    raise ValueError(
        f"Unknown native optimizer {name!r}. Short names must name a torch.optim optimizer "
        "(e.g. 'adamw', 'sgd', 'adam'); use a full dotted import path or a callable for anything else."
    )


def _is_managed_optimizer_name(optimizer: object) -> bool:
    """Return whether an optimizer config selects RF-DETR's managed construction.

    Managed mode covers bare ``torch.optim`` short names (e.g. ``"adamw"``, ``"sgd"``);
    RF-DETR injects ``lr`` and a signature-aware ``weight_decay`` there. A dotted import
    path or a callable selects explicit mode, where the optimizer is built only from
    ``optimizer_kwargs`` (or the callable's own bound arguments).

    Args:
        optimizer: The ``TrainConfig.optimizer`` value.

    Returns:
        ``True`` for managed short-name strings, ``False`` for dotted paths and callables.

    Examples:
        >>> _is_managed_optimizer_name("sgd")
        True
        >>> _is_managed_optimizer_name("torch.optim.AdamW")
        False
    """
    return isinstance(optimizer, str) and "." not in optimizer


def _desugar_optimizer_callable(
    optimizer: Callable[..., Optimizer],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Decompose a callable optimizer into a serializable ``(dotted_path, kwargs)`` form.

    Reconstructable callables — an importable top-level class or function, optionally
    wrapped in ``functools.partial`` with JSON-serializable keyword arguments and no
    positional arguments — desugar to a dotted import path plus keyword arguments that
    round-trip through ``training_config.json``.

    Args:
        optimizer: A callable or ``functools.partial`` given as ``TrainConfig.optimizer``.

    Returns:
        ``(dotted_path, kwargs, None)`` when reconstructable, otherwise
        ``(None, None, reason)`` where ``reason`` explains how to make it compatible.
    """
    func: Any = optimizer
    extracted_kwargs: dict[str, Any] = {}
    if isinstance(optimizer, functools.partial):
        if optimizer.args:
            return None, None, "pass every functools.partial argument as a keyword, not positionally"
        func = optimizer.func
        extracted_kwargs = dict(optimizer.keywords or {})

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None or "<" in qualname:
        return (
            None,
            None,
            "define the optimizer as an importable top-level class or function (no lambda or nested definition)",
        )

    try:
        json.dumps(extracted_kwargs)
    except (TypeError, ValueError):
        return (
            None,
            None,
            "use only JSON-serializable functools.partial keyword arguments (no tensors, modules, or callables)",
        )

    return f"{module}.{qualname}", extracted_kwargs, None


_MANAGED_SCHEDULER_PRESETS = {"step", "cosine"}
_DEPRECATED_LR_FIELD_KWARGS = {"lr_drop": "lr_drop", "lr_min_factor": "min_factor"}
# Keys the managed "step" / "cosine" presets actually consume from lr_scheduler_kwargs.
_MANAGED_SCHEDULER_KWARGS = {"min_factor", "lr_drop"}

# ReduceLROnPlateau does not subclass LRScheduler but is a supported explicit scheduler.
SchedulerType: TypeAlias = LRScheduler | ReduceLROnPlateau


def _is_managed_scheduler_name(lr_scheduler: object) -> bool:
    """Return whether an lr_scheduler config selects an RF-DETR managed preset.

    Managed presets are the built-in ``"step"`` and ``"cosine"`` schedules, which own warmup
    and total-step sizing. A dotted import path or a callable instead selects an explicit
    scheduler built from ``lr_scheduler_kwargs`` (or the callable's own bound arguments).

    Args:
        lr_scheduler: The ``TrainConfig.lr_scheduler`` value.

    Returns:
        ``True`` for managed preset short names, ``False`` for dotted paths and callables.

    Examples:
        >>> _is_managed_scheduler_name("cosine")
        True
        >>> _is_managed_scheduler_name("torch.optim.lr_scheduler.StepLR")
        False
    """
    return isinstance(lr_scheduler, str) and lr_scheduler.strip().lower() in _MANAGED_SCHEDULER_PRESETS


def _desugar_scheduler_callable(
    lr_scheduler: Callable[..., SchedulerType],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Decompose a callable lr_scheduler into a serializable ``(dotted_path, kwargs)`` form.

    Reconstructable callables — an importable top-level class or function, optionally wrapped in
    ``functools.partial`` with JSON-serializable keyword arguments and no positional arguments —
    desugar to a dotted import path plus keyword arguments that round-trip through
    ``training_config.json``. The optimizer is supplied at build time, never baked into the callable.

    Args:
        lr_scheduler: A callable or ``functools.partial`` given as ``TrainConfig.lr_scheduler``.

    Returns:
        ``(dotted_path, kwargs, None)`` when reconstructable, otherwise
        ``(None, None, reason)`` where ``reason`` explains how to make it compatible.
    """
    func: Any = lr_scheduler
    extracted_kwargs: dict[str, Any] = {}
    if isinstance(lr_scheduler, functools.partial):
        if lr_scheduler.args:
            return None, None, "pass every functools.partial argument as a keyword, not positionally"
        func = lr_scheduler.func
        extracted_kwargs = dict(lr_scheduler.keywords or {})

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None or "<" in qualname:
        return (
            None,
            None,
            "define the lr_scheduler as an importable top-level class or function (no lambda or nested definition)",
        )

    try:
        json.dumps(extracted_kwargs)
    except (TypeError, ValueError):
        return (
            None,
            None,
            "use only JSON-serializable functools.partial keyword arguments (no tensors, modules, or callables)",
        )

    return f"{module}.{qualname}", extracted_kwargs, None


class BaseConfig(BaseModel):
    """Base configuration class that validates input parameters against the defined model schema.

    If any unknown fields are provided, a ValueError is raised listing the unknown and available parameters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def catch_typo_kwargs(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        if cls.model_config.get("extra") != "forbid":
            return values
        allowed_params = set(cls.model_fields.keys())
        provided_params = set(values)
        unknown_params = provided_params - allowed_params
        if unknown_params:
            unknown_params_list = ", ".join(f"'{param}'" for param in sorted(unknown_params))
            allowed_params_list = ", ".join(sorted(allowed_params))
            raise ValueError(
                f"Unknown parameter(s): {unknown_params_list}. Available parameter(s): {allowed_params_list}."
            )
        return values

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        raise ValueError(f"Unknown attribute: '{name}'.")


class ModelConfig(BaseConfig):
    """Core architecture configuration for RF-DETR models.

    Concrete subclasses (e.g. ``RFDETRBaseConfig``, ``RFDETRLargeConfig``) must supply every field
    that has no default; direct instantiation of ``ModelConfig`` is unsupported.

    Attributes:
        encoder: Vision-transformer backbone identifier. Must be provided by concrete subclass.
        out_feature_indexes: Encoder layer indices whose feature maps are forwarded to the decoder.
            Must be provided by concrete subclass.
        dec_layers: Number of transformer decoder layers. Must be provided by concrete subclass.
        projector_scale: Feature-pyramid levels fed to the decoder cross-attention (subset of
            ``["P3", "P4", "P5"]``). Must be provided by concrete subclass.
        hidden_dim: Width of the decoder hidden state. Must be provided by concrete subclass.
        patch_size: ViT patch size used by the backbone. Must be provided by concrete subclass.
        num_windows: Number of windowed-attention windows in the backbone. Must be provided by
            concrete subclass.
        sa_nheads: Number of heads in decoder self-attention. Must be provided by concrete
            subclass.
        ca_nheads: Number of heads in decoder cross-attention. Must be provided by concrete
            subclass.
        dec_n_points: Deformable attention points per head per level in the decoder. Must be
            provided by concrete subclass.
        resolution: Square input resolution (pixels). Must be provided by concrete subclass.
        positional_encoding_size: Side length (in patches) of the sinusoidal positional grid.
            Must be provided by concrete subclass.
        num_queries: Number of object queries used during inference (and per group during
            training). Defaults to ``300``.
        num_classes: Number of output classes (background-free). Defaults to ``90`` (COCO).
        group_detr: Number of duplicate query groups used during training for GroupPose-style
            convergence acceleration. ``num_queries * group_detr`` predictions are produced in
            training mode; ``num_queries`` in eval mode. ``num_queries`` must be divisible by
            ``group_detr``. Defaults to ``13``.
        amp: Enable automatic mixed precision (bfloat16/float16). Defaults to ``True``.
        compile: Compile the model with ``torch.compile`` for faster throughput. Defaults to
            ``False``.
        pretrain_weights: Path or URL to pretrained checkpoint. ``None`` trains from scratch.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``). Auto-detected if not set.
        gradient_checkpointing: Trade compute for memory by checkpointing activations. Defaults
            to ``False``.
    """

    encoder: EncoderName
    out_feature_indexes: list[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: list[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    num_queries: int = 300
    # ModelConfig is the sole owner of `num_select` for PTL/inference; it is read via `_namespace_from_configs`.
    num_select: int = 300
    postprocess_trace_alpha: float = Field(default=0.2, ge=0.0)
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_channels: int = Field(default=3, ge=1)
    num_classes: int = 90
    pretrain_weights: PathLikeStr | None = None
    # torch.device values are accepted at validation time and normalized to string.
    device: str = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    prototype_logit_enabled: bool = False
    """是否在推理分类 logit 中启用视觉原型相对证据校准。"""

    prototype_logit_target_classes: list[int] = Field(default_factory=list)
    """参与视觉原型 logit 校准的小样本类别索引。"""

    prototype_logit_max_slots: int = Field(default=2, ge=1)
    """视觉原型 logit 校准使用的每类最大 slot 数。"""

    prototype_logit_alpha: float = Field(default=0.1, ge=0.0)
    """视觉原型相对证据转换为 logit residual 的增益。"""

    prototype_logit_margin: float = 0.05
    """目标类原型相对其他类原型的余弦间隔阈值。"""

    prototype_logit_temperature: float = Field(default=0.1, gt=0.0)
    """视觉原型相对证据的平滑温度。"""

    # ------------------------------------------------------------------
    # [ProtoGuidance] 多模态原型引导 query selection / content enhancement（默认全关）
    # 见 docs/改进方案-dinov2-proto/RF-DETR-DINOv2多模态原型引导方案.md
    # 放 ModelConfig 而非 TrainConfig：影响模型前向（top-k 选择），推理/评估必须
    # 与训练一致，且 from_checkpoint 从 checkpoint 的 args 重建模型时字段无损往返。
    # ------------------------------------------------------------------
    proto_guidance_enabled: bool = False
    """多模态原型引导总开关（E1 起启用；子开关见下方 *_position/*_content 等）。"""

    proto_guidance_artifacts_path: PathLikeStr | None = None
    """离线原型产物路径（stage0_build_proto_guidance.py 产出 .pt 文件）。"""

    proto_guidance_fusion_mode: Literal["simple", "gated"] = "simple"
    """原型融合模式：v1 仅支持 simple（加权归一化融合）；gated 未实现。"""

    proto_guidance_num_slots: int = Field(default=10, ge=1)
    """每类视觉子原型槽位数 M（可 5/1 消融；离线脚本产物须与此一致）。"""

    proto_guidance_target_classes: list[int] = Field(default_factory=list)
    """位置打分 max 集合（空 = 全部类别；selected_class 始终全类 argmax）。"""

    proto_guidance_tau_p: float = Field(default=0.1, gt=0.0)
    """原型相似度温度（缩放 cosine logits）。"""

    proto_guidance_lambda_pos_init: float = Field(default=0.05, ge=0.0)
    """位置 residual 权重初始值（近恒等起步，warmup 至上限）。"""

    proto_guidance_lambda_pos_max: float = Field(default=1.0, ge=0.0)
    """位置 residual 权重上限（warmup 目标；topk_overlap 长期≈1 时上调）。"""

    proto_guidance_gamma_content_init: float = Field(default=0.05, ge=0.0)
    """内容注入强度初始值（近恒等起步）。"""

    proto_guidance_gamma_content_max: float = Field(default=1.0, ge=0.0)
    """内容注入强度上限（warmup 目标）。"""

    proto_guidance_gate_bias_init: float = -2.944
    """内容 gate bias 初始值（logit 空间，默认 logit(0.05) 保留梯度不饱和）。"""

    proto_guidance_w_v_init: float = 0.3
    """融合时视觉原型权重初始值。"""

    proto_guidance_w_t_init: float = 0.7
    """融合时文本原型权重初始值（少样本早期文本作稳定锚）。"""

    proto_guidance_warmup_epochs: float = Field(default=2.0, ge=0.0)
    """lambda/gamma 线性 warmup 的 epoch 数（0 = 直接到上限）。"""

    proto_guidance_position_enabled: bool = True
    """位置引导子开关（E1；关闭时 top-k 恒等于原版）。"""

    proto_guidance_content_enabled: bool = False
    """内容引导子开关（E3 打开；关闭时 tgt 原样通过）。"""

    proto_guidance_aux_loss_enabled: bool = False
    """原型分类辅助损失子开关（E2 打开；配合 criterion loss_proto_labels）。"""

    proto_guidance_aux_loss_weight: float = Field(default=1.0, ge=0.0)
    """原型分类辅助损失权重。"""

    proto_guidance_visual_ema_update: bool = False
    """训练期是否用 GT 框特征 EMA 更新视觉原型（扩展项，实验默认关）。"""

    proto_guidance_visual_ema_momentum: float = 0.99
    """训练期视觉原型 EMA 更新动量。"""

    proto_guidance_dense_loss_enabled: bool = False
    """是否对全部 encoder proposal token 启用 dense 原型对齐损失。"""

    proto_guidance_dense_loss_weight: float = Field(default=1.0, ge=0.0)
    """dense 原型对齐损失权重。"""

    proto_guidance_dense_iou_pos: float = Field(default=0.3, ge=0.0, le=1.0)
    """dense proposal 分配为前景正样本的 IoU 阈值。"""

    proto_guidance_dense_iou_ignore: float = Field(default=0.1, ge=0.0, le=1.0)
    """dense proposal 低于该 IoU 时作为背景忽略。"""

    proto_guidance_dense_center_fallback_topk: int = Field(default=4, ge=1)
    """小目标无高 IoU proposal 时按中心 fallback 选取的 token 数。"""

    proto_guidance_dense_foreground_loss_enabled: bool = False
    """是否对 dense token 增加独立的前景/背景二分类监督。"""

    proto_guidance_dense_foreground_loss_weight: float = Field(default=1.0, ge=0.0)
    """dense 前景/背景二分类损失权重。"""

    proto_guidance_dense_background_ratio: float = Field(default=1.0, gt=0.0)
    """每个前景 token 最多采样的背景 token 数，用于平衡 hard negative。"""

    proto_guidance_position_score_mode: Literal["margin", "foreground", "foreground_semantic"] = "margin"
    """位置 selection 分数：语义 margin、foregroundness 或二者组合。"""

    proto_guidance_position_semantic_weight: float = Field(default=0.5, ge=0.0)
    """foreground_semantic 模式中语义 target margin 的相对权重。"""

    proto_guidance_position_score_std_ratio: float = Field(default=0.1, gt=0.0)
    """位置分数校准后相对线性 objectness 标准差的比例。"""

    proto_guidance_slot_reduction: Literal["max", "lse"] = "max"
    """多槽位聚合方式：max 或温度化 log-sum-exp。"""

    proto_guidance_slot_reduction_tau: float = Field(default=0.1, gt=0.0)
    """log-sum-exp 槽位聚合温度。"""

    proto_guidance_monitor_log_interval: int = Field(default=100, ge=1)
    """原型引导监控的采样节流步数（每 N 步采样一次）。"""

    compile: bool = False
    fused_optimizer: bool = True
    positional_encoding_size: int
    ia_bce_loss: bool = True
    segmentation_head: bool = False
    use_grouppose_keypoints: bool = False
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    grouppose_keypoint_dim_downscale: int = 1
    dual_projector: bool = False
    dual_projector_kp_only: bool = False
    num_keypoints_per_class: list[int] = Field(default_factory=list)
    num_decoder_registers: int = 0
    mask_downsample_ratio: int = 4
    backbone_lora: bool = False
    freeze_encoder: bool = False
    license: str = "Apache-2.0"
    model_name: str | None = Field(
        default=None,
        description=(
            'Name of the model class stored in training checkpoints (e.g. ``"RFDETRLarge"``). '
            "Set automatically by ``RFDETR.train()`` before saving. "
            "Used by ``RFDETR.from_checkpoint()`` to resolve the correct subclass directly "
            "without inspecting ``pretrain_weights``."
        ),
    )

    @model_validator(mode="after")
    def _sync_pe_with_resolution(self) -> "ModelConfig":
        """Auto-update positional_encoding_size when resolution is explicitly provided.

        When a user provides a custom ``resolution`` at construction time (e.g., ``RFDETRLarge(resolution=640)``),
        ``positional_encoding_size`` is updated proportionally, provided the class-default PE is formula-derived
        (``default_pe == default_resolution // patch_size``).

        Configs with a pretrained-specific PE (e.g., ``RFDETRBaseConfig`` with ``positional_encoding_size=37`` for
        DINOv2's native 518 px grid, while ``resolution=560``) are left unchanged.
        """
        if "resolution" not in self.model_fields_set or "positional_encoding_size" in self.model_fields_set:
            return self

        cls = type(self)
        default_resolution = cls.model_fields["resolution"].default
        default_pe = cls.model_fields["positional_encoding_size"].default
        default_patch_size = cls.model_fields["patch_size"].default

        # Skip when any relevant default is not a concrete integer (abstract base
        # class fields have no defaults; required fields use PydanticUndefined,
        # not int).
        if (
            not isinstance(default_resolution, int)
            or not isinstance(default_pe, int)
            or not isinstance(default_patch_size, int)
        ):
            return self

        # Only update PE when the class default is formula-derived from the class
        # default resolution and patch size.
        if default_pe == default_resolution // default_patch_size:
            self.positional_encoding_size = self.resolution // self.patch_size

        return self

    @model_validator(mode="after")
    def validate_proto_guidance(self) -> "ModelConfig":
        """[ProtoGuidance] 多模态原型引导字段校验。

        启用时必须提供离线产物路径；关闭时要求产物路径为空，避免误配。
        """
        if self.proto_guidance_enabled and not self.proto_guidance_artifacts_path:
            raise ValueError(
                "proto_guidance_enabled=True 时必须提供 proto_guidance_artifacts_path"
                "（stage0_build_proto_guidance.py 产出的 .pt 文件）。"
            )
        if not self.proto_guidance_enabled and self.proto_guidance_artifacts_path:
            raise ValueError("proto_guidance_artifacts_path 仅在 proto_guidance_enabled=True 时使用，当前模块未启用。")
        if self.proto_guidance_enabled:
            invalid = [c for c in self.proto_guidance_target_classes if not 0 <= int(c) < self.num_classes]
            if invalid:
                raise ValueError(f"proto_guidance_target_classes 含非法类别索引: {invalid}")
            if self.proto_guidance_lambda_pos_max < self.proto_guidance_lambda_pos_init:
                raise ValueError(
                    "proto_guidance_lambda_pos_max 必须 >= lambda_pos_init，"
                    f"收到 {self.proto_guidance_lambda_pos_max} < {self.proto_guidance_lambda_pos_init}。"
                )
            if self.proto_guidance_gamma_content_max < self.proto_guidance_gamma_content_init:
                raise ValueError(
                    "proto_guidance_gamma_content_max 必须 >= gamma_content_init，"
                    f"收到 {self.proto_guidance_gamma_content_max} < {self.proto_guidance_gamma_content_init}。"
                )
        return self

    @model_validator(mode="after")
    def _warn_pretrain_compatibility(self) -> "ModelConfig":
        """Warn when overrides are likely to prevent published pretrained weights from loading.

        Three cases:

        1. ``pretrain_weights`` was explicitly set to ``None`` and the variant
           has a non-``None`` default → warn that the model is being initialised from scratch.
        2. ``pretrain_weights`` was explicitly set to a non-``None`` custom path
           → suppress the architecture-override check (we cannot know the architecture stored in a user-supplied
           checkpoint at config time). The load-time partial-load detector in
           :func:`rfdetr.models.weights.load_pretrain_weights` covers this case by inspecting the checkpoint contents
           directly.
        3. ``pretrain_weights`` is the variant's published default → check
           architecture-affecting fields against the variant defaults and emit a single consolidated warning listing
           every load-breaking override.

        The warning class is :class:`PretrainWeightsCompatibilityWarning` (a :class:`UserWarning` subclass), silenceable
        via the standard ``warnings.filterwarnings`` machinery.
        """
        cls = type(self)
        fields_set = self.model_fields_set
        pretrain_user_set = "pretrain_weights" in fields_set

        if pretrain_user_set and self.pretrain_weights is None:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            if default_pretrain is not PydanticUndefined and default_pretrain is not None:
                warnings.warn(
                    f"{cls.__name__} was instantiated with pretrain_weights=None. "
                    f"The model will be initialised from scratch, which typically "
                    f"produces lower accuracy than fine-tuning from the published "
                    f"checkpoint ({default_pretrain!r}).",
                    PretrainWeightsCompatibilityWarning,
                    stacklevel=2,
                )
            return self

        if pretrain_user_set and self.pretrain_weights is not None:
            # Custom checkpoint: architecture overrides may match what the
            # checkpoint was trained with.  Defer to the load-time partial-load
            # detector which can read the file.
            # Exception: when the user explicitly passes the variant's own
            # published-default path string (e.g. ``"rf-detr-nano.pth"``), it
            # IS the published checkpoint — treat it as case 3 so architecture-
            # override checks still apply.  Compare after expand_path so bare
            # filenames resolve to the same cache-dir path as self.pretrain_weights.
            _default_pretrain = cls.model_fields["pretrain_weights"].default
            if _default_pretrain is not None and _default_pretrain is not PydanticUndefined:
                _expanded_default = cls.expand_path(_default_pretrain)
                if self.pretrain_weights != _expanded_default:
                    return self
                # Falls through to case-3 when the user passed the exact variant default.
            else:
                return self

        # `pretrain_weights` is the variant's published default — check
        # architecture overrides against the class defaults.
        # Skip entirely when this variant has no published checkpoint (default
        # is None/PydanticUndefined); warning would reference "(None)" which is
        # misleading and confusing for users of the abstract base config.
        _class_default_pretrain = cls.model_fields["pretrain_weights"].default
        if _class_default_pretrain is None or _class_default_pretrain is PydanticUndefined:
            return self

        overrides: list[tuple[str, Any, Any]] = []

        # Fields that, when explicitly overridden to any value other than the
        # variant default, prevent the published checkpoint from loading cleanly.
        # Includes major architecture knobs, "less obvious" knobs (bbox_reparam,
        # lite_refpoint_refine, layer_norm, two_stage), defense-in-depth for
        # fields that currently raise hard errors (patch_size, segmentation_head),
        # and num_channels (loads via heuristic but result isn't real pretrained
        # weights for the new input domain).
        breaking_fields: tuple[str, ...] = (
            "encoder",
            "hidden_dim",
            "dec_layers",
            "num_windows",
            "sa_nheads",
            "ca_nheads",
            "dec_n_points",
            "out_feature_indexes",
            "projector_scale",
            "bbox_reparam",
            "lite_refpoint_refine",
            "layer_norm",
            "two_stage",
            "patch_size",
            "segmentation_head",
            "num_channels",
        )
        # Fields where only an *increase* above the variant default is load-breaking:
        # num_queries / group_detr add slots whose shape differs — decrease is fine.
        breaking_on_increase: tuple[str, ...] = (
            "num_queries",
            "group_detr",
        )

        for name in breaking_fields:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined:
                continue
            current = getattr(self, name)
            if current != default:
                overrides.append((name, current, default))

        for name in breaking_on_increase:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined or not isinstance(default, int):
                continue
            current = getattr(self, name)
            if isinstance(current, int) and current > default:
                overrides.append((name, current, default))

        # ``mask_downsample_ratio`` only affects segmentation models — skip on
        # detector-only variants to avoid a misleading "weights won't load" warning.
        if "mask_downsample_ratio" in fields_set and self.segmentation_head:
            _mdr_info = cls.model_fields.get("mask_downsample_ratio")
            if _mdr_info is not None and not _mdr_info.is_required():
                _mdr_default = _mdr_info.default
                if _mdr_default is not PydanticUndefined:
                    _mdr_current = self.mask_downsample_ratio
                    if _mdr_current != _mdr_default:
                        overrides.append(("mask_downsample_ratio", _mdr_current, _mdr_default))

        if overrides:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            lines = "\n".join(
                f"  {name}: {current!r} (variant default: {default!r})" for name, current, default in overrides
            )
            warnings.warn(
                f"{cls.__name__} was instantiated with overrides that differ from the variant "
                f"defaults in ways that prevent the published pretrained weights "
                f"({default_pretrain!r}) from loading correctly:\n"
                f"{lines}\n"
                "Loading the checkpoint with this configuration will leave significant portions "
                "of the model randomly initialised, which typically produces lower accuracy. "
                "To suppress this warning: revert the override(s), pick a variant whose defaults "
                "match, or pass pretrain_weights=None to acknowledge that you intend to train "
                "from scratch.",
                PretrainWeightsCompatibilityWarning,
                stacklevel=2,
            )

        return self

    @field_validator("pretrain_weights", mode="before")
    @classmethod
    def expand_path(cls, v: PathLikeStr | None) -> str | None:
        """Expand and resolve the pretrain_weights path.

        Bare filenames (no directory component, e.g. ``rf-detr-base.pth``) are resolved to the model cache directory so
        weights land in a stable, user-configurable location (``~/.roboflow/models`` by default, or the path set via the
        ``RF_HOME`` environment variable) instead of CWD.

        Paths that already contain a directory separator (e.g. ``~/models/x.pth``, ``/abs/path/x.pth``,
        ``models/x.pth``) are normalised with ``os.path.realpath`` as before.
        """
        if v is None:
            return v
        expanded = os.path.expanduser(os.fspath(v))
        if not os.path.dirname(expanded):
            # Bare filename → use model cache dir so weights don't land in CWD.
            from rfdetr.assets.model_weights import get_model_cache_dir

            return os.path.join(get_model_cache_dir(), expanded)
        return os.path.realpath(expanded)

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, v: Any) -> str:
        """Normalize supported device inputs to a canonical torch-style string.

        Args:
            v: Device specifier provided by callers. Supported values are
                ``str`` (for example ``"cpu"``, ``"cuda"``, ``"cuda:1"``) and ``torch.device``.

        Returns:
            Canonical string form of the parsed device (for example ``"cuda:1"``).

        Raises:
            ValueError: If a string value cannot be parsed as a valid torch device.
            ValueError: If ``v`` is not a string or ``torch.device``.
        """
        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, str):
            try:
                return str(torch.device(v))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"Invalid device specifier: {v!r}.") from exc
        raise ValueError("device must be a string or torch.device.")


class RFDETRBaseConfig(ModelConfig):
    """The configuration for an RF-DETR Base model."""

    encoder: EncoderName = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: list[int] = [2, 5, 8, 11]
    pretrain_weights: PathLikeStr | None = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37


class RFDETRLargeDeprecatedConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Large model."""

    encoder: EncoderName = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: PathLikeStr | None = "rf-detr-large.pth"


class RFDETRNanoConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Nano model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: PathLikeStr | None = "rf-detr-nano.pth"


class RFDETRSmallConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Small model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: PathLikeStr | None = "rf-detr-small.pth"


class RFDETRMediumConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Medium model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: PathLikeStr | None = "rf-detr-medium.pth"


# res 704, ps 16, 2 windows, 4 dec layers, 300 queries, ViT-S basis
class RFDETRLargeConfig(ModelConfig):
    """Configuration for the RF-DETR Large model variant."""

    encoder: Literal["dinov2_windowed_small"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_windows: int = 2
    patch_size: int = 16
    projector_scale: list[Literal["P4",]] = ["P4"]
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_classes: int = 90
    positional_encoding_size: int = 704 // 16
    pretrain_weights: PathLikeStr | None = "rf-detr-large-2026.pth"
    resolution: int = 704
    # Explicit so populate_args and _build_args_from_configs agree.
    # ModelConfig does not define these fields; without them the legacy path
    # picks up populate_args defaults (num_select=100) while the PTL path falls
    # back to TrainConfig.num_select (300), causing a postprocess mismatch.
    num_queries: int = 300
    num_select: int = 300


class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Preview model."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-preview.pt"
    num_classes: int = 90


class RFDETRSegNanoConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Nano model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 1
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 312
    positional_encoding_size: int = 312 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-nano.pt"
    num_classes: int = 90


class RFDETRSegSmallConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Small model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 384
    positional_encoding_size: int = 384 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-small.pt"
    num_classes: int = 90


class RFDETRSegMediumConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Medium model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 432 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-medium.pt"
    num_classes: int = 90


class RFDETRSegLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Large model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 504
    positional_encoding_size: int = 504 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-large.pt"
    num_classes: int = 90


class RFDETRSegXLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation XLarge model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 624
    positional_encoding_size: int = 624 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-xlarge.pt"
    num_classes: int = 90


class RFDETRSeg2XLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation 2XLarge model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 768
    positional_encoding_size: int = 768 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-xxlarge.pt"
    num_classes: int = 90


class RFDETRKeypointPreviewConfig(RFDETRBaseConfig):
    """Configuration for the preview keypoint model."""

    use_grouppose_keypoints: bool = True
    dual_projector: bool = True
    dual_projector_kp_only: bool = True
    num_keypoints_per_class: list[int] = [17]
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    grouppose_keypoint_dim_downscale: int = 1
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 576
    positional_encoding_size: int = 576 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-keypoint-preview-xlarge.pth"
    num_classes: int = 90


class TrainConfig(BaseConfig):
    """Training hyperparameters and auto-batching configuration.

    Notes:
        * ``auto_batch_target_effective`` is interpreted as the **per-device**
          effective batch size target, i.e. the number of images seen by a single process in one optimizer step after
          accounting for ``grad_accum_steps``. In multi-GPU / multi-node runs the global effective batch size is
          therefore:

            ``global_effective_batch = auto_batch_target_effective * devices * num_nodes``

          This avoids silently changing behavior when scaling from single-GPU to multi-GPU training.
    """

    # extra="forbid" arms BaseConfig.catch_typo_kwargs so typo'd train() kwargs (e.g. ``epoch`` instead of
    # ``epochs``) raise with a helpful message instead of being silently ignored.  Legacy kwargs handled by
    # RFDETR.train() (resolution/device/callbacks/start_epoch/do_benchmark) are popped before construction.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int | Literal["auto"] = 4
    grad_accum_steps: int = 4
    auto_batch_target_effective: int = 16  # per-device effective batch size target (before devices * num_nodes)
    # Auto-batch probe: worst-case assumptions when batch_size="auto".
    auto_batch_max_targets_per_image: int = 100
    auto_batch_ema_headroom: float = 0.7  # scale safe batch by this when use_ema=True (EMA uses extra memory)
    epochs: int = 100
    resume: PathLikeStr | None = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = Field(default=10, ge=1)
    skip_best_epochs: int = Field(default=0, ge=0)
    smooth_alpha: float = 0.0
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    cls_loss_coef: float = 1.0
    keypoint_flip_pairs: list[int] = Field(default_factory=list)
    keypoint_l1_loss_coef: float = 0
    keypoint_findable_loss_coef: float = 0
    keypoint_visible_loss_coef: float = 0
    keypoint_nll_loss_coef: float = 0
    keypoint_oks_sigmas: list[float] | None = None
    dataset_file: Literal["coco", "o365", "roboflow", "yolo"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: PathLikeStr | None
    dataset_cache_mode: Literal["off", "raw"] = "off"
    dataset_cache_dir: PathLikeStr | None = None
    dataset_cache_rebuild: bool = False
    output_dir: PathLikeStr = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    ema_update_interval: int = 1
    # Validation-only: forward through the EMA model instead of the base model, and skip the
    # duplicate base-model forward pass COCOEvalCallback would otherwise also run — halves
    # per-batch validation compute when EMA is enabled. Requires use_ema=True.
    # Metric-key remap: when active, val/mAP_50_95 (and val/segm_mAP_*) report EMA-model
    # quality, not base-model quality — the base model is never evaluated this epoch. Best-
    # checkpoint tracking follows: the "regular" checkpoint track sees no data (safely no-ops)
    # while the EMA checkpoint track receives the real per-epoch EMA score. val/F1 is not
    # remapped (no parallel EMA-tracked accumulator) and still reflects EMA-quality predictions
    # under the regular key.
    eval_ema_only: bool = False
    num_workers: int = 2
    weight_decay: float = 1e-4
    amp_dtype: Literal["auto", "bf16", "fp16"] = Field(
        default="auto",
        description=(
            "Mixed-precision autocast dtype. "
            "'auto' selects bf16-mixed on Ampere+ CUDA, fp16 otherwise. "
            "'bf16' forces bfloat16 (falls back to fp16 with a warning if unsupported). "
            "'fp16' forces fp16. "
            "Has no effect when model_config.amp=False or when training on CPU."
        ),
    )
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    progress_bar: Literal["tqdm", "rich"] | None = None  # Progress bar style: "rich", "tqdm", or None to disable.
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    clearml: bool = False  # Not yet implemented — reserved for future use.
    project: str | None = None
    run: str | None = None
    class_names: list[str] | None = None
    run_test: bool = False
    eval_max_dets: int = 500
    eval_interval: int = 1
    log_per_class_metrics: bool = True
    # Segmentation only. Skip upsampling predicted masks to full image resolution during
    # validation/test, returning them at the mask head's native (lower) resolution instead —
    # cheaper, but ground-truth masks must then be compared at that same lower resolution
    # (handled in COCOEvalCallback). No effect on non-segmentation models or on inference output
    # (RFDETR.predict always upsamples regardless of this flag).
    # Metric comparability: GT masks are nearest-downsized to the mask head's native resolution
    # (e.g. 512x512 -> ~16x16) before comparison, so val/segm_mAP under this flag is NOT
    # comparable to a full-resolution run — small objects can collapse to empty masks at the
    # lower resolution, and IoU is computed on a coarser pixel grid either way.
    eval_masks_head_resolution: bool = False
    aug_config: Optional[Dict[str, Any]] = None
    mosaic_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Mosaic 增强触发概率，0 表示关闭。")
    patch_paste_enabled: bool = Field(
        default=False,
        description="正/负样本补丁粘贴增强开关（仅训练侧，默认关闭）。",
    )
    patch_paste_dir: PathLikeStr | None = Field(
        default=None,
        description="补丁池根目录（含 manifest.json），由 build_fsc_patch_pool.py 生成；"
        "相对路径自动解析到项目根。",
    )
    patch_paste_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="补丁粘贴触发概率（还需宿主图含 target_classes 类才真正粘贴）。",
    )
    patch_paste_max_patches: int = Field(
        default=2,
        ge=1,
        description="每图最多粘贴补丁数，实际张数在 [1, max] 均匀采样。",
    )
    patch_paste_target_classes: list[int] = Field(
        default_factory=lambda: [24],
        description="宿主硬约束：仅含这些类中任一类的训练图才允许粘贴补丁。",
    )
    patch_paste_neg_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="负样本占单图补丁的比例（0=全正样本，1=全负样本）。",
    )
    patch_paste_scale_range: tuple[float, float] = Field(
        default=(0.8, 1.5),
        description="补丁相对宿主原图的缩放范围（粘贴在原始分辨率上，下游 resize 适配）。",
    )
    class_balanced_sampling: bool = Field(
        default=False,
        description=(
            "平方根频率过采样开关（MMDetection ClassBalancedDataset 风格）："
            "repeat_factor(c)=max(1, int(sqrt(t/freq(c))))，每图取白名单类最大倍率，"
            "数据集长度 = Σ r(I)，DataLoader shuffle 全局混合。默认关闭，不影响基线行为。"
        ),
    )
    class_balanced_threshold: float | None = Field(
        default=None,
        gt=0,
        description="平方根频率过采样阈值 t。None 时自动推导 t = 4 × max(freq(目标类集))。",
    )
    class_balanced_class_ids: list[int] = Field(
        default_factory=list,
        description="平方根频率过采样目标类集白名单（SHWX: 0=HM, 1=LQS）。空列表 = 全部类别。",
    )
    scale_jitter: bool = True
    augmentation_backend: AugmentationBackend | Literal["cpu", "auto"] = "cpu"
    save_dataset_grids: bool = False

    @field_validator("augmentation_backend", mode="before")
    @classmethod
    def _coerce_augmentation_backend(cls, v: Any) -> Any:
        """Map legacy backend name strings (``"gpu"``, ``"tv"``, ``"albu"``) to their current form.

        ``"cpu"``/``"auto"`` pass through unchanged — they are auto-pick sentinels resolved lazily by
        :meth:`AugmentationBackend.from_str` at dataset-build time, not at config construction time, so a saved config
        stays portable across environments. See :class:`AugmentationBackend` for the full rationale.
        """
        if isinstance(v, str):
            return _LEGACY_AUGMENTATION_BACKEND_ALIASES.get(v, v)
        return v

    @field_serializer("augmentation_backend")
    def _serialize_augmentation_backend(self, value: AugmentationBackend | str) -> str:
        """Serialize the backend selector to its plain string form.

        Returns the enum's ``.value`` for :class:`AugmentationBackend` members and passes the
        ``"cpu"``/``"auto"`` sentinel strings through unchanged. This lets checkpoint writers call
        plain :meth:`model_dump` and still get a JSON-safe ``str`` for this one field, instead of a
        blanket ``model_dump(mode="json")`` that would also coerce every *other* field's serialized
        shape (e.g. the ``int`` keypoint loss coefficients to ``float``). Applies in both python and
        json ``model_dump`` modes, so the two checkpoint writers stay consistent.

        Args:
            value: Stored backend selector — an :class:`AugmentationBackend` member or a
                ``"cpu"``/``"auto"`` sentinel string.

        Returns:
            The plain string form of the backend selector.

        Examples:
            >>> cfg = TrainConfig(dataset_dir="ds", augmentation_backend="torchvision")
            >>> cfg.model_dump()["augmentation_backend"]
            'torchvision'
        """
        if isinstance(value, AugmentationBackend):
            return value.value
        return value

    notes: Optional[Any] = Field(
        default=None,
        description=(
            "User-defined provenance metadata embedded in best-model .pth checkpoints "
            "under checkpoint['args']['notes'] and in exported ONNX files under the "
            "'rfdetr_notes' metadata property. Accepts any JSON-serialisable value "
            "(string, dict, list, int, float, bool). String values are stored verbatim; "
            "all other types are JSON-encoded."
        ),
    )

    # ------------------------------------------------------------------
    # [SSCL] 语义相似度引导的监督对比学习配置（默认全部关闭）
    # ------------------------------------------------------------------
    sscl_enabled: bool = False
    """是否启用 SSCL（语义相似度引导的监督对比学习）损失。"""
    sscl_semantic_matrix_path: str | None = None
    """CLIP 类别语义相似度矩阵文件路径（.pt），由 build_semantic_matrix.py 生成。"""
    sscl_matrix_normalize: str = "minmax"
    """加载语义矩阵时的后处理方式：

    - "minmax": 线性映射到 [0, 1]（默认，推荐，增强易混对的判别度）。
    - "softmax": 温度缩放的 softmax 行归一化。
    - "none": 使用原始 CLIP 余弦相似度。
    """
    sscl_lambda: float = 0.01
    """SSCL 损失权重 λ_sscl（推荐 0.01 ~ 0.05）。"""
    sscl_tau: float = 0.1
    """SSCL 对比学习温度 τ。"""
    sscl_rho: float = 0.3
    """语义先验对负样本强度的放大系数 ρ（推荐 0.2 ~ 0.5）。"""
    sscl_omega_max: float = 2.0
    """负样本语义权重上限 ω_max，防止训练不稳定。"""
    sscl_anchor_classes: list[int] | None = None
    """重点 anchor 类别索引列表，默认 None（使用全部类别），推荐 C_focus = [0, 1]（HM/LQS）。"""
    sscl_confusing_classes: list[int] | None = None
    """易混负样本类别索引列表，默认 None（对所有异类负样本加权），推荐 C_ship = [0, 1, 2, 3]。"""
    sscl_distill_enabled: bool = False
    """是否启用基类 logit 蒸馏（保护飞机类/FSC 指标）。"""
    sscl_distill_lambda: float = 0.5
    """蒸馏损失权重 λ_distill（推荐 0.5 ~ 1.0）。"""
    sscl_distill_temperature: float = 2.0
    """蒸馏温度 T。"""
    sscl_distill_mode: str = "mse"
    """蒸馏方式："mse"（logit 蒸馏，第一版推荐）或 "kl"（伯努利软标签蒸馏）。"""
    sscl_teacher_checkpoint: str | None = None
    """教师模型（原始 RF-DETR checkpoint）路径，启用蒸馏时必需。"""
    sscl_protected_classes: list[int] | None = None
    """受保护的基类类别索引列表，默认 None（自动使用飞机类 4-23 + FSC 24，舰船类 0-3 不蒸馏）。"""
    sscl_start_epoch: int = 30
    """SSCL 损失开始生效的 epoch（从 0 起计）。在此 epoch 之前 ``loss_sscl`` 权重被置 0，仅由常规检测损失训练；达到该 epoch 后按 ``sscl_lambda`` 加权。
    适用于从预训练直接训练、希望先让基类收敛再施加语义对比约束的实验。"""
    sscl_freeze_strategy: Literal["conservative", "none"] = "conservative"
    """SSCL 冻结策略：
    - ``"conservative"``: 冻结 backbone/encoder/bbox 头/decoder 前几层，仅解冻
      decoder 最后一层 + norm + class_embed（在已收敛 checkpoint 上微调时使用）。
    - ``"none"``: 不冻结任何参数，保持全量微调（从预训练直接开始训练时使用）。"""
    sscl_unfreeze_decoder_layers: int = 1
    """保守冻结策略下解冻 decoder 末尾层数。1=旧行为，仅解冻最后一层；2/3 可用于扩大 SSCL 微调容量。"""
    proto_guidance_freeze_all_except_proto: bool = False
    """[ProtoGuidance] 阶段 A 冷启动专用：除原型引导模块外冻结全部参数 （跳过 decoder/norm/class_embed 的解冻分支）。实验阶段由 stageA 配方启用。"""
    proto_guidance_trainable_scope: Literal["all", "token", "token_fg"] = "all"
    """原型模块内部可训练范围：all、仅 token 投影层，或 token+foreground head。"""
    sscl_prototype_enabled: bool = False
    """是否启用类别原型库锚定的 SSCL（原型模式）。开启时正样本为本类原型、 负样本为全部类别原型，每个样本恒有正负锚点，摆脱 batch 内同类样本不足 导致的零损失问题。"""
    sscl_prototype_momentum: float = 0.99
    """类别原型 EMA 更新系数 ``p <- m*p + (1-m)*batch_mean``（推荐 0.9 ~ 0.999， 越小跟随越快）。"""
    sscl_prototype_min_samples: int = 1
    """原型模式下单次 batch 中某类样本数低于该阈值时跳过该类原型更新（防噪声）。 默认 1 使少样本场景首个样本即建立原型。"""
    sscl_prototype_sync_ddp: bool = False
    """是否在 DDP 多卡时先 ``all_gather`` 各 rank 特征再更新原型，保证各 rank 原型一致（``register_buffer`` 不会被 DDP 自动同步）。单卡无需，默认关闭。"""
    sscl_prototype_max_slots: int = 1
    """每类最多原型 slot 数。1=旧版单视觉原型；2 可用于 HM/LQS/QHS/MS 等多模态易混类。"""
    sscl_prototype_multi_slot_classes: list[int] | None = None
    """启用多 slot 的类别索引列表。列表外类别仅使用 slot 0，保持单原型。"""
    sscl_prototype_group_pairs: list[list[int]] | None = None
    """固定易混类别组。同组不同类 slot 在 SSCL 分母中作为 sibling 难负样本额外加权。"""
    sscl_prototype_group_weight: float = 1.0
    """同组 sibling 负样本权重放大系数。1.0=关闭组内额外加压；推荐从 1.5 起步。"""
    sscl_projection_enabled: bool = False
    """是否启用 SSCL 投影头：先把 matched features 投影到低维对比空间再施加 对比损失，缓解对比压力对共享特征（同时喂给 class_embed 与 bbox_embed）的
    直接冲击。开启后原型库也住在投影空间。"""
    sscl_projection_dim: int = 128
    """SSCL 投影头输出维度（对比空间维度），通常低于 decoder hidden dim。 仅在 ``sscl_projection_enabled=True`` 时生效。"""
    sscl_prototype_instance_pos: bool = False
    """原型模式是否加入同类别实例正样本（对齐论文 Eq.9）。开启时正样本 = 本类原型 ∪ 同类别实例，用真实同类实例锚定随机初始化投影头的冷启动； 负样本仍为全部有效原型（语义加权）。推荐投影实验开启。"""
    sscl_hard_neg_enabled: bool = False
    """是否启用难负样本抑制：对高分未匹配 query 增加前景 logit 抑制和可选原型排斥损失。"""
    sscl_hard_neg_topk: int = 3
    """每图最多选取的难负样本数量 k。"""
    sscl_hard_neg_score_thresh: float = -2.0
    """难例目标前景 logit 下限（原始 logit，非概率）。"""
    sscl_hard_neg_log_interval: int = 100
    """难例诊断监控采样步间隔（每 N 步采样一次，epoch 末聚合输出到 train/sscl/*）。"""
    sscl_hard_neg_target_classes: list[int] | None = None
    """难例挖掘与 logit 抑制的目标前景类别；None 表示全部前景类。"""
    sscl_hard_neg_iou_low: float = 0.0
    """难负样本与任一 GT 最大 IoU 的下界。"""
    sscl_hard_neg_iou_high: float = 0.3
    """难负样本与任一 GT 最大 IoU 的上界，用于排除真实目标重复检测。"""
    sscl_hard_neg_loss_lambda: float = 0.3
    """难负样本 logit 抑制损失权重。"""
    sscl_hard_neg_logit_margin: float = -1.5
    """目标前景 logit 的软上界；高于该值的难负样本会被额外惩罚。"""
    sscl_hard_neg_target_logit_margins: dict[int, float] = Field(
        default_factory=dict,
        description="按类覆盖 logit 抑制 margin：{类别: margin}。未列出的类别用 "
        "``sscl_hard_neg_logit_margin`` 默认值。用于给高真阳量类（如 FSC）单独"
        "放宽抑制，避免把定位偏差的真阳一起压掉（E4-hard-neg -22pt / +FSC -9pt 教训）。",
    )
    sscl_hard_neg_target_loss_lambdas: dict[int, float] = Field(
        default_factory=dict,
        description="按类覆盖 logit 抑制权重：{类别: lambda}。未列出的类别用 "
        "``sscl_hard_neg_loss_lambda`` 默认值（最终损失仍乘 weight_dict 中的全局权重）。",
    )
    sscl_hard_neg_logit_temperature: float = 1.0
    """难负样本 logit 抑制的 softplus 温度。"""
    sscl_hard_neg_proto_lambda: float = 0.05
    """难负样本前景原型排斥项相对权重；0 表示关闭。"""
    sscl_hard_neg_proto_margin: float = 0.1
    """难负样本与前景原型余弦相似度的软上界。"""
    sscl_hard_neg_proto_temperature: float = 0.1
    """难负样本原型排斥的 softplus 温度。"""
    sscl_hard_neg_start_epoch: int = 0
    """难负样本抑制开始生效的 epoch（从 0 起计），独立于 ``sscl_start_epoch``。

    与 SSCL 对比损失解耦，可在 SSCL 已运行若干 epoch、匹配质量稳定后再引入
    难例抑制。早期未匹配 query 的高分多为训练噪声，过早抑制会把真阳误判为
    难负样本（E4-hard-neg 实验中抑制从 epoch 0 全强度施加导致 MS recall
    从 0.7568 暴跌至 0.5340 的教训）。
    """
    # ------------------------------------------------------------------
    # [分类损失均衡化] 正样本类均衡 IA-BCE + 居中截断 Logit Adjustment（默认全部关闭）
    # 见 docs/改进方案-SSCL/RF-DETR分类损失均衡化改进方案.md
    # P0：仅对正样本 slot 乘类别权重，不降低负样本惩罚（优先保护 FDR）。
    # P1：对分类 logit 加"居中 + 截断 + warmup"的先验去偏置 margin（LA 的安全化形式）。
    # ------------------------------------------------------------------
    class_balance_enabled: bool = False
    """P0 总开关：对 IA-BCE 正样本 slot 乘类别频率权重（CB Loss 风格）。"""
    class_balance_counts_path: str | None = None
    """类别实例数统计 JSON 路径（由 scripts/stat_class_counts.py 生成），格式 {"counts": [n0, n1, ...]}。"""
    class_balance_beta: float = 0.25
    """幂律权重指数 β：w_c = (N_ref / max(n_c, n_min)) ** beta（推荐 {0.25, 0.5}）。"""
    class_balance_max_weight: float = 3.0
    """权重上限 w_max，防止稀有类过拟合（首发 3.0，上限测试 5.0）。"""
    class_balance_min_count: int = 10
    """分母下限 n_min：防极端小样本类（如 HM=6）产生极端权重。"""
    class_balance_ref_count: float | None = None
    """参考样本数 N_ref。默认 None 时自动取 sqrt(N_max * N_min)（几何平均）， 不要直接用 N_max 以免权重过大。"""
    class_balance_target_classes: list[int] | None = None
    """生效类别索引列表，其余类别权重固定为 1.0。默认 None（全部类别）。首发 [0, 1]（HM/LQS）。"""
    logit_adjustment_enabled: bool = False
    """P1 总开关：对分类 logit 加居中截断的先验 bias（训练侧）。"""
    logit_adjustment_tau: float = 0.1
    """LA 强度 τ（推荐 {0.1, 0.25, 0.5}，不要首发 1.0/2.0）。"""
    logit_adjustment_bias_clip: float = 1.0
    """居中 bias 的截断上限（推荐 {1.0, 2.0}）。"""
    logit_adjustment_warmup_epochs: float = 1.0
    """Bias warmup 轮数：前 N 个 epoch 从 0 线性升到目标值（防早期分配噪声放大）。"""
    # ------------------------------------------------------------------
    # 语义分类头（SemanticResidual，默认全关）
    # 见 docs/改进方案-SSCL/RF-DETR语义分类头改进方案.md
    # ------------------------------------------------------------------
    semantic_head_enabled: bool = False
    """语义分类头总开关。开启后在 decoder 分类 logits 上叠加语义残差增量。"""
    semantic_fsem_path: str | None = None
    """f_sem 产物路径（``fsem_shwx.pt``，含 S 矩阵），由 stage0_train_fsem.py 产出。"""
    semantic_channel_stats_path: str | None = None
    """通道 TF-IDF 统计路径（``channel_stats_shwx.pt``），由 stage0_train_fsem.py 产出。"""
    semantic_mask_enabled: bool = True
    """是否启用通道掩码增量（``mask_delta``）。关闭（M=1）时等价于纯语义方向注入 （消融实验 E2b）。"""
    semantic_alpha_enabled: bool = True
    """是否启用语义方向增量（``sem_delta``）。关闭（α=0）时等价于仅通道掩码 （消融实验 E1c）。"""
    semantic_alpha_learnable: bool = True
    """Α 是否可学习。关闭时 α 冻结在初始值（消融实验 E3b，验证 α 自适应的价值）。"""
    semantic_parallel_logit: bool = False
    """平行 logit 结构开关（保留用于结构对比实验）。平行模式 = 仅叠加 ``α·(hs@Sᵀ)``，
    与"残差 + 掩码关闭"数学等价（E1b 与 E2b 对照）。"""
    semantic_alpha_init: float = 0.1
    """Base 类 α 初始值。α 为语义注入强度，clamp 到 [0, semantic_alpha_max]。"""
    semantic_novel_alpha_init: float = 0.1
    """Novel 类（少样本舰船类）α 初始值，通常略高于 base（消融实验 E3c 置 0.5）。"""
    semantic_alpha_max: float = 2.0
    """Α 上限，防止语义项把舰船 logit 整体抬高引发 FP。"""
    semantic_mask_tau: float = 1.0
    """掩码软度 τ_mask：``M = sigmoid((θ − r)/τ_mask)``。rank 范围 [1,d] 很大，
    实际有效 τ = max(该值, d/16)（d=256 时≈16），保证 soft 带覆盖约 1/4 通道、
    θ 梯度存活（过小会让 M≈1 饱和、掩码学不动）。"""
    semantic_theta_init: float = 0.0
    """θ 初始化偏移：``θ = d + semantic_theta_init·τ_mask``。默认 0（θ=d），M 对
    最差通道≈0.5、对多数通道≈1，初始接近"全保留"但掩码梯度存活；可加大让掩码
    初始更激进（更早收窄）。"""
    semantic_novel_classes: list[int] | None = None
    """Novel 类（少样本类）索引列表。该类 α 用 ``semantic_novel_alpha_init``。 默认 None（训练脚本显式传 [0,1,2,3] 舰船类）。"""
    semantic_frozen_threshold_classes: list[int] | None = None
    """Θ 冻结于初始值（base 均值语义）的类别索引。少样本类样本太少学不准， 默认 None（训练脚本显式传 [0,1,2,3]）。"""
    semantic_lr: float = 1e-4
    """Α/θ 独立参数组学习率。标量参数需足够 LR，否则学不出来（投影头同款教训）。"""
    semantic_align_classes: list[int] | None = None
    """对齐监控逐类输出的类别列表，默认 None（训练脚本显式传 novel 类）。"""
    semantic_monitor_log_interval: int = 100
    """语义头监控采样步间隔（每 N 步采样一次，epoch 末聚合输出到 train/sem/*）。"""
    # ------------------------------------------------------------------
    # [QNorm-Obj + EUMix] query 范数物体性门控与熵感知校准（默认全部关闭）
    # 见 docs/改进方案-QNorm-Obj/RF-DETR引入QNorm-Obj与EUMix方案.md
    # ------------------------------------------------------------------
    qnorm_obj_enabled: bool = False
    """QNorm-Obj + EUMix 总开关。开启后在 decoder 分类 logits 上施加 query 范数物体性门控与熵感知校准，不加任何辅助监督损失。"""
    qnorm_obj_tau: float = 2.0
    """物体性头温度 τ：``z_obj = f_obj(‖h‖₂) / τ``。缩放 MLP 输出到 sigmoid
    敏感区间，防止 logits 过陡导致 σ 饱和、梯度消失。"""
    qnorm_obj_feature_mix: bool = True
    """是否启用 QNorm 特征混合（论文 Eq.7-9）：``h_cls = (1-α_mix)·h + α_mix·h_norm``
    后重算分类 logits，解耦特征方向（语义）与幅度（物体性）。关闭时用原始特征。"""
    qnorm_obj_gate: bool = True
    """是否启用物体性门控：``z_known *= σ(z_obj)``（只乘前景类列，背景列不动）。 低物体性的背景框被压向背景。关闭时门恒为 1。"""
    qnorm_obj_eumix: bool = True
    """是否启用熵感知校准（论文 Eq.16-23，闭集适配：背景列扮演 unknown）： 用物体性 × 熵缺口 校准背景 logit 并对前景做软抑制。关闭时仅保留门控。"""
    qnorm_obj_obj_hidden_dim: int = 64
    """物体性头隐藏维度：``MLP(d → obj_hidden_dim → 1)``。"""
    qnorm_obj_gamma_init: float = 1.0
    """熵缺口指数 γ 的初始**有效值**（论文 Eq.17）：``γ = softplus(θ_γ)``，
    参数 θ_γ 初始化为 softplus⁻¹(该值)。γ>1 只对"非常不确定"放行缺口，γ<1 过渡平滑。"""
    qnorm_obj_alpha_init: float = 0.1
    """EUMix 背景混合权重 α 的初始**有效值**（θ_α 初始化为 logit⁻¹(α)）。
    闭集适配为 logit 空间混合：z_bg_new = (1-α)·(z_bg+b_obj) + α·logit(p_obj_bg)。
    默认 0.1 → 起点 z_bg_new ≈ z_bg（恒等起步，不扰动预训练背景 logit）；
    训练中 α 可学习上升，让物体性证据按损失信号参与背景校准。"""
    qnorm_obj_lambda_init: float = 0.5
    """前景软抑制强度 λ 初始值：``z_known -= λ·p_obj_bg``（clamp ≥ 0）。 控制"物体性高但已知类不确定"的 query 被压向前景的程度。"""

    @field_validator("progress_bar", mode="before")
    @classmethod
    def _coerce_legacy_progress_bar(cls, value: Any) -> Any:
        """Normalize legacy boolean progress_bar values to the new string/None representation.

        This preserves compatibility with older configs where ``progress_bar`` was a bool.
        """
        if isinstance(value, bool):
            return "tqdm" if value else None
        return value

    @field_validator("amp_dtype", mode="before")
    @classmethod
    def _coerce_amp_dtype(cls, value: Any) -> Any:
        """Fall back to ``'auto'`` (with a warning) for an unrecognised or wrong-typed ``amp_dtype``.

        Mixed precision is a best-effort speed/memory optimisation, so an invalid request degrades to the auto-selected
        dtype rather than failing the whole training run.
        """
        if value not in ("auto", "bf16", "fp16"):
            # stacklevel=2 points into Pydantic internals; unavoidable with @field_validator in Pydantic v2.
            warnings.warn(
                f"Unknown amp_dtype={value!r}; expected one of 'auto', 'bf16', 'fp16'. Falling back to 'auto'.",
                UserWarning,
                stacklevel=2,
            )
            return "auto"
        return value

    # Promoted from populate_args() — PTL migration (T4-2).
    # device is intentionally absent: PTL auto-detects accelerator via Trainer(accelerator="auto").
    accelerator: str = "auto"
    clip_max_norm: float = 0.1
    seed: int | None = None
    sync_bn: bool = False
    # strategy maps to PTL Trainer(strategy=...). Common values: "auto", "ddp",
    # "ddp_spawn", "fsdp", "deepspeed". Invalid values surface as PTL errors.
    strategy: str = "auto"
    devices: int | str = 1
    # num_nodes maps to PTL Trainer(num_nodes=...) for multi-machine training.
    # Single-machine DDP users should leave this at 1 (the default).
    num_nodes: int = 1
    fp16_eval: bool = False
    lr_scheduler: str | Callable[..., SchedulerType] = "step"
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    lr_scheduler_interval: Literal["step", "epoch"] = "step"
    lr_scheduler_monitor: str = "val/loss"
    # Deprecated aux LR knobs — kept for one cycle; folded into lr_scheduler_kwargs (see _map_deprecated_lr_fields).
    lr_min_factor: float = 0.0
    optimizer: str | Callable[..., Optimizer] = "adamw"
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    dont_save_weights: bool = False
    # PTL runtime/perf tuning knobs.
    train_log_sync_dist: bool = False
    train_log_on_step: bool = False
    compute_train_metrics: bool = False
    compute_val_loss: bool = True
    compute_test_loss: bool = True
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None

    @field_validator("batch_size", mode="after")
    @classmethod
    def validate_batch_size(cls, v: int | Literal["auto"]) -> int | Literal["auto"]:
        """Validate batch_size is a positive integer or the literal 'auto'."""
        if v == "auto":
            return v
        if v < 1:
            raise ValueError("batch_size must be >= 1, or 'auto'.")
        return v

    @field_validator(
        "grad_accum_steps", "auto_batch_target_effective", "auto_batch_max_targets_per_image", mode="after"
    )
    @classmethod
    def validate_positive_train_steps(cls, v: int) -> int:
        """Validate accumulation, target-effective batch, and max targets are >= 1."""
        if v < 1:
            raise ValueError(
                "grad_accum_steps, auto_batch_target_effective, and auto_batch_max_targets_per_image must be >= 1."
            )
        return v

    @field_validator("auto_batch_ema_headroom", mode="after")
    @classmethod
    def validate_ema_headroom(cls, v: float) -> float:
        """Validate auto_batch_ema_headroom is in (0, 1]."""
        if not (0 < v <= 1.0):
            raise ValueError("auto_batch_ema_headroom must be in (0, 1].")
        return v

    @field_validator("smooth_alpha", mode="after")
    @classmethod
    def validate_smooth_alpha(cls, v: float) -> float:
        """Validate smooth_alpha is in [0.0, 1.0)."""
        if not (0.0 <= v < 1.0):
            raise ValueError("smooth_alpha must be in [0.0, 1.0).")
        return v

    @field_validator("ema_update_interval", "eval_interval", mode="after")
    @classmethod
    def validate_positive_intervals(cls, v: int) -> int:
        """Validate interval fields are >= 1."""
        if v < 1:
            raise ValueError("Interval fields must be >= 1.")
        return v

    @model_validator(mode="before")
    @classmethod
    def _desugar_callable_optimizer(cls, data: Any) -> Any:
        """Desugar a reconstructable callable optimizer into its serializable string form.

        A callable ``optimizer`` (a class or ``functools.partial``) that can be imported is rewritten to a dotted import
        path plus ``optimizer_kwargs`` so the config round-trips through ``training_config.json``. User-supplied
        ``optimizer_kwargs`` are ignored for callable optimizers (bake arguments into the callable instead). Non-
        reconstructable callables are kept as-is and only warned about.
        """
        if not isinstance(data, dict):
            return data
        optimizer = data.get("optimizer")
        if optimizer is None or isinstance(optimizer, str) or not callable(optimizer):
            return data

        if data.get("optimizer_kwargs"):
            warnings.warn(
                "optimizer_kwargs is ignored when optimizer is a callable; bake arguments into the "
                "callable (for example with functools.partial) instead.",
                stacklevel=2,
            )

        path, kwargs, reason = _desugar_optimizer_callable(optimizer)
        if reason is None:
            data["optimizer"] = path
            data["optimizer_kwargs"] = kwargs
        else:
            data["optimizer_kwargs"] = {}
            label = getattr(optimizer, "__qualname__", None) or repr(optimizer)
            warnings.warn(
                f"optimizer callable {label!r} cannot be saved to training_config.json and restored: "
                f"{reason}. Training proceeds with the in-memory callable; only saved-config "
                "reproducibility is affected.",
                stacklevel=2,
            )
        return data

    @field_validator("optimizer", mode="after")
    @classmethod
    def validate_optimizer_name(cls, v: str | Callable[..., Optimizer]) -> str | Callable[..., Optimizer]:
        """Validate a string optimizer: a bare name must be a native torch.optim optimizer."""
        if not isinstance(v, str):
            return v
        optimizer = v.strip()
        if not optimizer:
            raise ValueError("optimizer must be a non-empty string.")
        # Bare short names must resolve to a torch.optim optimizer (checked eagerly).
        # Dotted import paths are validated lazily at train start (the module may be optional).
        if "." not in optimizer:
            _resolve_native_optimizer(optimizer)
        return optimizer

    @model_validator(mode="after")
    def validate_eval_ema_only(self) -> "TrainConfig":
        """``eval_ema_only`` has no EMA model to evaluate without ``use_ema=True``."""
        if self.eval_ema_only and not self.use_ema:
            raise ValueError("eval_ema_only=True requires use_ema=True.")
        return self

    @model_validator(mode="after")
    def validate_optimizer_kwargs(self) -> "TrainConfig":
        """Reserved optimizer kwargs are only rejected for managed (short-name) optimizers."""
        if _is_managed_optimizer_name(self.optimizer):
            reserved_present = _OPTIMIZER_MANAGED_KWARGS.intersection(self.optimizer_kwargs)
            if reserved_present:
                reserved = ", ".join(sorted(reserved_present))
                raise ValueError(f"optimizer_kwargs cannot include RF-DETR-managed key(s): {reserved}.")
        return self

    @model_validator(mode="after")
    def validate_lr_scheduler_kwargs(self) -> "TrainConfig":
        """Reject unknown ``lr_scheduler_kwargs`` keys for the managed ``"step"`` / ``"cosine"`` presets.

        Managed presets consume only ``min_factor`` and ``lr_drop``; any other key would be silently ignored, so surface
        it as an error (mirroring ``validate_optimizer_kwargs``). Explicit schedulers forward their kwargs verbatim to
        the constructor and are left unchecked here.
        """
        if _is_managed_scheduler_name(self.lr_scheduler):
            unknown = set(self.lr_scheduler_kwargs) - _MANAGED_SCHEDULER_KWARGS
            if unknown:
                allowed = ", ".join(sorted(_MANAGED_SCHEDULER_KWARGS))
                unknown_keys = ", ".join(sorted(unknown))
                raise ValueError(
                    f"lr_scheduler_kwargs for a managed preset ({self.lr_scheduler!r}) accepts only "
                    f"{{{allowed}}}; unknown key(s): {unknown_keys}."
                )
        return self

    @model_validator(mode="after")
    def validate_sscl_hard_neg(self) -> "TrainConfig":
        """难负样本字段的基础校验。"""
        if self.sscl_hard_neg_topk < 1:
            raise ValueError(f"sscl_hard_neg_topk 必须 >= 1，收到 {self.sscl_hard_neg_topk}。")
        if not 0.0 <= self.sscl_hard_neg_iou_low <= 1.0:
            raise ValueError(f"sscl_hard_neg_iou_low 必须在 [0, 1] 内，收到 {self.sscl_hard_neg_iou_low}。")
        if not 0.0 <= self.sscl_hard_neg_iou_high <= 1.0:
            raise ValueError(f"sscl_hard_neg_iou_high 必须在 [0, 1] 内，收到 {self.sscl_hard_neg_iou_high}。")
        if self.sscl_hard_neg_iou_low > self.sscl_hard_neg_iou_high:
            raise ValueError(
                "sscl_hard_neg_iou_low 必须 <= sscl_hard_neg_iou_high，"
                f"收到 {self.sscl_hard_neg_iou_low} > {self.sscl_hard_neg_iou_high}。"
            )
        if self.sscl_hard_neg_loss_lambda < 0.0:
            raise ValueError(f"sscl_hard_neg_loss_lambda 必须 >= 0，收到 {self.sscl_hard_neg_loss_lambda}。")
        if self.sscl_hard_neg_logit_temperature <= 0.0:
            raise ValueError(f"sscl_hard_neg_logit_temperature 必须 > 0，收到 {self.sscl_hard_neg_logit_temperature}。")
        if self.sscl_hard_neg_proto_lambda < 0.0:
            raise ValueError(f"sscl_hard_neg_proto_lambda 必须 >= 0，收到 {self.sscl_hard_neg_proto_lambda}。")
        if self.sscl_hard_neg_proto_temperature <= 0.0:
            raise ValueError(f"sscl_hard_neg_proto_temperature 必须 > 0，收到 {self.sscl_hard_neg_proto_temperature}。")
        if self.sscl_hard_neg_target_classes is not None:
            invalid = [c for c in self.sscl_hard_neg_target_classes if c < 0]
            if invalid:
                raise ValueError(f"sscl_hard_neg_target_classes 含非法类别索引: {invalid}")
        if self.sscl_hard_neg_start_epoch < 0:
            raise ValueError(
                f"sscl_hard_neg_start_epoch 必须 >= 0，收到 {self.sscl_hard_neg_start_epoch}。"
            )
        for c, v in (self.sscl_hard_neg_target_logit_margins or {}).items():
            if int(c) < 0:
                raise ValueError(f"sscl_hard_neg_target_logit_margins 含非法类别索引: {c}")
            if v > 0:
                raise ValueError(
                    f"sscl_hard_neg_target_logit_margins 的类别 {c} 取值为 {v}，"
                    "margin 是 logit 软上界，必须 <= 0"
                )
        for c, v in (self.sscl_hard_neg_target_loss_lambdas or {}).items():
            if int(c) < 0:
                raise ValueError(f"sscl_hard_neg_target_loss_lambdas 含非法类别索引: {c}")
            if v < 0:
                raise ValueError(f"sscl_hard_neg_target_loss_lambdas 的类别 {c} 取值为 {v}，必须 >= 0")
        return self

    @model_validator(mode="after")
    def validate_sscl_multislot_prototype(self) -> "TrainConfig":
        """多 slot 原型字段的基础校验。"""
        if self.sscl_unfreeze_decoder_layers < 1:
            raise ValueError(f"sscl_unfreeze_decoder_layers 必须 >= 1，收到 {self.sscl_unfreeze_decoder_layers}。")
        if self.sscl_prototype_max_slots < 1:
            raise ValueError(f"sscl_prototype_max_slots 必须 >= 1，收到 {self.sscl_prototype_max_slots}。")
        if self.sscl_prototype_group_weight < 1.0:
            raise ValueError(f"sscl_prototype_group_weight 必须 >= 1.0，收到 {self.sscl_prototype_group_weight}。")
        if self.sscl_prototype_multi_slot_classes is not None:
            invalid = [c for c in self.sscl_prototype_multi_slot_classes if c < 0]
            if invalid:
                raise ValueError(f"sscl_prototype_multi_slot_classes 含非法类别索引: {invalid}")
        if self.sscl_prototype_group_pairs is not None:
            seen: set[int] = set()
            for group in self.sscl_prototype_group_pairs:
                if len(group) < 2:
                    raise ValueError("sscl_prototype_group_pairs 中每个组至少需要 2 个类别。")
                for class_id in group:
                    if class_id < 0:
                        raise ValueError(f"sscl_prototype_group_pairs 含非法类别索引: {class_id}")
                    if class_id in seen:
                        raise ValueError(f"类别 {class_id} 被重复放入多个 sscl_prototype_group_pairs。")
                    seen.add(class_id)
        return self

    @model_validator(mode="before")
    @classmethod
    def _map_deprecated_lr_fields(cls, data: Any) -> Any:
        """Fold the deprecated ``lr_drop`` / ``lr_min_factor`` fields into ``lr_scheduler_kwargs``.

        These loose knobs are deprecated in favor of ``lr_scheduler_kwargs``. When either is supplied with a non-default
        value for a managed preset (``"step"`` / ``"cosine"``), it is copied into ``lr_scheduler_kwargs`` (without
        overriding an explicit kwarg) and a ``FutureWarning`` is emitted. Default values are ignored silently so round-
        tripping a dumped config (which always carries these fields) never warns. For explicit (dotted-path / callable)
        schedulers the deprecated fields are preset-specific and left untouched.
        """
        if not isinstance(data, dict):
            return data
        # Only managed presets consume these knobs; default lr_scheduler ("step") is managed.
        if not _is_managed_scheduler_name(data.get("lr_scheduler", "step")):
            # Explicit / callable scheduler: these preset knobs are inert. Warn (never fold) when a non-default
            # value is set so a stale lr_drop / lr_min_factor carried over from a managed config is not silently
            # dropped — a reproducibility footgun when migrating a saved config to an explicit scheduler.
            for field_name in _DEPRECATED_LR_FIELD_KWARGS:
                if field_name in data and data[field_name] != cls.model_fields[field_name].default:
                    warnings.warn(
                        f"{field_name} is ignored for the explicit (non-managed) lr_scheduler "
                        f"{data.get('lr_scheduler')!r}; it only applies to the managed 'step'/'cosine' presets.",
                        FutureWarning,
                        stacklevel=2,
                    )
            return data
        kwargs = dict(data.get("lr_scheduler_kwargs") or {})
        for field_name, kwarg_name in _DEPRECATED_LR_FIELD_KWARGS.items():
            if field_name not in data:
                continue
            # A default value (common when reloading a dumped config) is a no-op: the managed builder falls back to
            # the same default. Skip silently so serialization round-trips don't emit spurious deprecation warnings.
            if data[field_name] == cls.model_fields[field_name].default:
                continue
            # Already migrated: a dumped config carries both the top-level field and the folded kwarg. If the kwarg
            # already holds this value, the field adds nothing — skip silently so migrated-config reloads never warn.
            if kwargs.get(kwarg_name) == data[field_name]:
                continue
            # Both set to different values: the kwarg wins (setdefault below is a no-op). Say so explicitly rather
            # than implying the deprecated field was migrated, which would mislead — the field value is discarded.
            if kwarg_name in kwargs:
                warnings.warn(
                    f"{field_name}={data[field_name]!r} is ignored because lr_scheduler_kwargs already sets "
                    f"{kwarg_name!r}={kwargs[kwarg_name]!r} (the kwarg wins); remove the deprecated {field_name}.",
                    FutureWarning,
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"{field_name} is deprecated; pass it via lr_scheduler_kwargs={{'{kwarg_name}': ...}} instead.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs[kwarg_name] = data[field_name]
        if kwargs:
            data["lr_scheduler_kwargs"] = kwargs
        return data

    @model_validator(mode="before")
    @classmethod
    def _desugar_callable_lr_scheduler(cls, data: Any) -> Any:
        """Desugar a reconstructable callable lr_scheduler into its serializable string form.

        A callable ``lr_scheduler`` (a class or ``functools.partial``) that can be imported is rewritten to a dotted
        import path plus ``lr_scheduler_kwargs`` so the config round-trips through ``training_config.json``. User-
        supplied ``lr_scheduler_kwargs`` are ignored for callable schedulers (bake arguments into the callable instead).
        Non-reconstructable callables are kept as-is and only warned about.
        """
        if not isinstance(data, dict):
            return data
        lr_scheduler = data.get("lr_scheduler")
        if lr_scheduler is None or isinstance(lr_scheduler, str) or not callable(lr_scheduler):
            return data

        if data.get("lr_scheduler_kwargs"):
            warnings.warn(
                "lr_scheduler_kwargs is ignored when lr_scheduler is a callable; bake arguments into the "
                "callable (for example with functools.partial) instead.",
                stacklevel=2,
            )

        path, kwargs, reason = _desugar_scheduler_callable(lr_scheduler)
        if reason is None:
            data["lr_scheduler"] = path
            data["lr_scheduler_kwargs"] = kwargs
        else:
            data["lr_scheduler_kwargs"] = {}
            label = getattr(lr_scheduler, "__qualname__", None) or repr(lr_scheduler)
            warnings.warn(
                f"lr_scheduler callable {label!r} cannot be saved to training_config.json and restored: "
                f"{reason}. Training proceeds with the in-memory callable; only saved-config "
                "reproducibility is affected.",
                stacklevel=2,
            )
        return data

    @field_validator("lr_scheduler", mode="after")
    @classmethod
    def validate_lr_scheduler_name(cls, v: str | Callable[..., SchedulerType]) -> str | Callable[..., SchedulerType]:
        """Validate a string lr_scheduler: a bare name must be a managed preset, else use a dotted path."""
        if not isinstance(v, str):
            return v
        lr_scheduler = v.strip()
        if not lr_scheduler:
            raise ValueError("lr_scheduler must be a non-empty string.")
        # Bare names must be a managed preset; dotted import paths are validated lazily at train start.
        if "." not in lr_scheduler and not _is_managed_scheduler_name(lr_scheduler):
            presets = ", ".join(sorted(_MANAGED_SCHEDULER_PRESETS))
            raise ValueError(
                f"Unknown lr_scheduler {v!r}. Bare names must be a managed preset ({presets}); "
                "use a full dotted import path (e.g. 'torch.optim.lr_scheduler.StepLR') or a callable "
                "for anything else."
            )
        return lr_scheduler

    @field_validator("prefetch_factor", mode="after")
    @classmethod
    def validate_prefetch_factor(cls, v: int | None) -> int | None:
        """Validate prefetch_factor is None or >= 1."""
        if v is not None and v < 1:
            raise ValueError("prefetch_factor must be >= 1 when provided.")
        return v

    @field_validator("dataset_dir", "dataset_cache_dir", "output_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: PathLikeStr | None) -> str | None:
        """Expand and normalize dataset/output directory paths via ``os.fspath`` → ``expanduser`` → ``realpath``."""
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(os.fspath(v)))

    @field_validator("resume", mode="before")
    @classmethod
    def _coerce_resume_path(cls, v: PathLikeStr | None) -> str | None:
        """Normalise the resume checkpoint value to ``str`` without resolving it.

        Unlike ``dataset_dir``/``output_dir``, ``resume`` is forwarded verbatim to PyTorch Lightning's
        ``trainer.fit(ckpt_path=...)``, which also accepts sentinel values such as ``"last"``. Running
        ``os.path.realpath`` would rewrite those sentinels into spurious absolute paths, so this validator only coerces
        the type (``Path`` -> ``str``) and leaves the value untouched.
        """
        if v is None:
            return v
        return os.fspath(v)


class SegmentationTrainConfig(TrainConfig):
    """Training configuration for instance segmentation models.

    Extends :class:`TrainConfig` with segmentation-specific loss coefficients.

    Attributes:
        mask_point_sample_ratio: Number of points sampled per mask for point-based
            mask loss computation.
        mask_ce_loss_coef: Cross-entropy loss weight for mask prediction.
        mask_dice_loss_coef: Dice loss weight for mask prediction.
        cls_loss_coef: Classification loss weight. Defaults to ``1.0`` to match the
            effective pre-v1.7 value (the v1.7 TrainConfig ownership migration
            silently activated a dormant ``5.0``; this field restores the correct
            weight). To reproduce pre-fix segmentation behaviour pass
            ``cls_loss_coef=5.0`` explicitly.
    """

    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 1.0


class KeypointTrainConfig(TrainConfig):
    """Training configuration for keypoint detection models.

    Extends :class:`TrainConfig` with keypoint-specific loss coefficients and
    metric-smoothing defaults tuned for the NLL-Cholesky keypoint head, which
    produces noisy per-epoch OKS metrics during early fine-tuning.

    Attributes:
        cls_loss_coef: Classification loss weight.
        keypoint_l1_loss_coef: L1 regression loss weight for keypoint coordinates.
        keypoint_findable_loss_coef: Loss weight for the keypoint visibility head.
        keypoint_visible_loss_coef: Loss weight for the keypoint visibility score.
        keypoint_nll_loss_coef: NLL-Cholesky loss weight. Restored to ``1.0`` to
            align with the other keypoint loss terms (``keypoint_l1_loss_coef``,
            ``keypoint_findable_loss_coef``, ``keypoint_visible_loss_coef``).
            Previously set to ``0.5`` to dampen OKS@75 oscillation; reverted as
            the under-weighting was not beneficial in practice.
        smooth_alpha: EMA smoothing factor for :class:`BestModelCallback` metric
            comparison. Overrides the :class:`TrainConfig` default of ``0.0``
            (disabled) to ``0.5``, which balances responsiveness and noise
            suppression for noisy keypoint mAP curves.
        skip_best_epochs: Number of epochs to skip before checkpoint selection begins.
            Overrides the :class:`TrainConfig` default of ``0`` to ``10`` because
            ``val/keypoint_map_50_95`` under the NLL-Cholesky loss is noisy in early
            fine-tuning and can lock checkpoint selection to a transient peak.
    """

    cls_loss_coef: float = 2.0  # TODO: verify empirically before final release; ported as-is from internal recipe.
    keypoint_l1_loss_coef: float = 1
    keypoint_findable_loss_coef: float = 1
    keypoint_visible_loss_coef: float = 1
    keypoint_nll_loss_coef: float = 1.0
    smooth_alpha: float = 0.5
    skip_best_epochs: int = Field(default=10, ge=0)
