# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import importlib
import inspect
import math
import random
import warnings
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias (see AGENTS.md)
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    ModelConfig,
    TrainConfig,
    _is_managed_optimizer_name,
    _is_managed_scheduler_name,
    _resolve_native_optimizer,
)
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import apply_lora, interpolate_position_embeddings, load_pretrain_weights
from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module
from rfdetr.training.param_groups import (
    get_param_dict,
    get_projection_head_param_dict,
    get_semantic_head_param_dict,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_OptimizerFactory = Callable[..., torch.optim.Optimizer]

_TRAIN_PROGRESS_LOSS_ALIASES: dict[str, str] = {
    "loss_ce": "loss_cls",
    "loss_bbox": "loss_box",
    "loss_giou": "loss_giou",
    "loss_mask_ce": "mask_ce",
    "loss_mask_dice": "mask_dice",
    "loss_keypoints_l1": "kp_l1",
    "loss_keypoints_findable": "kp_find",
    "loss_keypoints_visible": "kp_vis",
    "loss_keypoints_nll": "kp_nll",
    "loss_sscl_hard_neg": "hn",
}


def _is_builtin_fused_adamw(optimizer: object) -> bool:
    """Return whether the config selects RF-DETR's built-in (managed, fused) AdamW path.

    Args:
        optimizer: The ``TrainConfig.optimizer`` value (string or callable).

    Returns:
        ``True`` only for the built-in ``"adamw"`` short name.

    Examples:
        >>> _is_builtin_fused_adamw("adamw")
        True
        >>> _is_builtin_fused_adamw("torch.optim.AdamW")
        False
    """
    return isinstance(optimizer, str) and "." not in optimizer and optimizer.strip().lower() == "adamw"


_FUSED_IGNORED_MSG = (
    "fused_optimizer=True is ignored for optimizer=%r; the fused AdamW kernel only applies to the "
    "built-in optimizer='adamw' path."
)


def _import_optimizer_class(dotted_path: str) -> _OptimizerFactory:
    """Import an optimizer class or factory from a dotted path.

    Args:
        dotted_path: Fully-qualified path such as ``"torch.optim.AdamW"`` or
            ``"pytorch_optimizer.Lion"``.

    Returns:
        The imported optimizer class or factory.

    Raises:
        ValueError: If the module or attribute cannot be imported.
    """
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"optimizer {dotted_path!r} is not a valid dotted import path.")
    try:
        module = importlib.import_module(module_path)
        return cast(_OptimizerFactory, getattr(module, attribute))
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not import optimizer {dotted_path!r}: {exc}. "
            "Use a fully-qualified path to an importable optimizer class, e.g. 'torch.optim.AdamW'."
        ) from exc


def _instantiate_explicit_optimizer(
    optimizer_class: _OptimizerFactory,
    optimizer_name: str,
    param_dicts: list[dict[str, Any]],
    optimizer_kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    """Instantiate an explicitly-selected optimizer from param groups and kwargs only.

    Explicit optimizers (dotted import paths and callables) receive the RF-DETR
    parameter groups — which already carry per-group learning rates — plus the
    user's ``optimizer_kwargs`` verbatim. RF-DETR injects no ``lr`` or
    ``weight_decay`` of its own.

    Args:
        optimizer_class: Optimizer class or factory to instantiate.
        optimizer_name: Name used in error messages.
        param_dicts: RF-DETR parameter groups with layer-wise learning rates.
        optimizer_kwargs: Keyword arguments forwarded verbatim to the constructor.

    Returns:
        Instantiated optimizer.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint on failure.
    """
    try:
        return optimizer_class(param_dicts, **optimizer_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize optimizer {optimizer_name!r}: {exc}. "
            "Explicit optimizers (dotted paths and callables) are built from the RF-DETR parameter "
            "groups plus your `optimizer_kwargs` only, with no lr/weight_decay injected. Check that the "
            "class accepts these arguments, or pass a callable/functools.partial needing only `params`."
        ) from exc


def _optimizer_accepts_kwarg(optimizer_class: _OptimizerFactory, name: str) -> bool:
    """Return whether an optimizer constructor accepts a given keyword argument.

    Args:
        optimizer_class: Optimizer class or factory to inspect.
        name: Keyword-argument name to look for.

    Returns:
        ``True`` when the constructor declares ``name`` or accepts arbitrary
        keyword arguments, or when its signature cannot be introspected (e.g. a
        C-implemented constructor) — in which case the constructor validates the
        call itself.

    Examples:
        >>> _optimizer_accepts_kwarg(torch.optim.SGD, "weight_decay")
        True
    """
    try:
        signature = inspect.signature(optimizer_class)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return name in signature.parameters


def _instantiate_optimizer(
    optimizer_class: _OptimizerFactory,
    optimizer_name: str,
    param_dicts: list[dict[str, Any]],
    train_config: TrainConfig,
) -> torch.optim.Optimizer:
    """Instantiate an optimizer class with RF-DETR optimizer arguments.

    ``weight_decay`` is injected only when the optimizer constructor accepts it,
    so optimizers with a different regularization API are not forced to fail.

    Args:
        optimizer_class: Optimizer class or factory to instantiate.
        optimizer_name: Name used in error messages.
        param_dicts: RF-DETR parameter groups with layer-wise learning rates.
        train_config: Training config with base optimizer hyperparameters.

    Returns:
        Instantiated optimizer.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint when the
            optimizer constructor rejects the supplied arguments.
    """
    init_kwargs: dict[str, Any] = {"lr": train_config.lr}
    if _optimizer_accepts_kwarg(optimizer_class, "weight_decay"):
        init_kwargs["weight_decay"] = train_config.weight_decay
    init_kwargs.update(train_config.optimizer_kwargs)
    try:
        return optimizer_class(param_dicts, **init_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize optimizer {optimizer_name!r}: {exc}. "
            "For managed torch.optim short names, RF-DETR passes `params`, `lr`, and (when supported) "
            "`weight_decay`, then your `optimizer_kwargs`; this usually means an unsupported entry in "
            "`optimizer_kwargs`."
        ) from exc


_SchedulerFactory = Callable[..., LRScheduler | ReduceLROnPlateau]


def _import_scheduler_class(dotted_path: str) -> _SchedulerFactory:
    """Import an LR-scheduler class or factory from a dotted path.

    Args:
        dotted_path: Fully-qualified path such as ``"torch.optim.lr_scheduler.StepLR"`` or
            ``"pytorch_optimizer.CosineAnnealingWarmupRestarts"``.

    Returns:
        The imported scheduler class or factory.

    Raises:
        ValueError: If the module or attribute cannot be imported.
    """
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"lr_scheduler {dotted_path!r} is not a valid dotted import path.")
    try:
        module = importlib.import_module(module_path)
        return cast(_SchedulerFactory, getattr(module, attribute))
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not import lr_scheduler {dotted_path!r}: {exc}. "
            "Use a fully-qualified path to an importable scheduler class, e.g. 'torch.optim.lr_scheduler.StepLR'."
        ) from exc


def _instantiate_explicit_scheduler(
    scheduler_factory: _SchedulerFactory,
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    scheduler_kwargs: dict[str, Any],
) -> LRScheduler | ReduceLROnPlateau:
    """Instantiate an explicitly-selected LR scheduler from the optimizer and kwargs only.

    Explicit schedulers (dotted import paths and callables) receive the built optimizer plus the
    user's ``lr_scheduler_kwargs`` verbatim. RF-DETR injects no ``total_steps`` / ``T_max`` of its
    own — the managed ``"step"`` / ``"cosine"`` presets remain the runtime-aware option.

    Args:
        scheduler_factory: Scheduler class or factory to instantiate.
        scheduler_name: Name used in error messages.
        optimizer: The optimizer the scheduler drives.
        scheduler_kwargs: Keyword arguments forwarded verbatim to the constructor.

    Returns:
        Instantiated scheduler.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint on failure.
    """
    try:
        return scheduler_factory(optimizer, **scheduler_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize lr_scheduler {scheduler_name!r}: {exc}. "
            "Explicit schedulers (dotted paths and callables) are built from the optimizer plus your "
            "`lr_scheduler_kwargs` only, with no total_steps/T_max injected. Check that the class accepts these "
            "arguments, or pass a callable/functools.partial needing only the optimizer."
        ) from exc


def _build_managed_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    total_steps: int,
    steps_per_epoch: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the managed ``"step"`` / ``"cosine"`` scheduler as a warmup-aware ``LambdaLR``.

    Preserves RF-DETR's built-in schedule: a linear warmup ramp over ``warmup_steps`` followed by
    either cosine annealing down to ``min_factor`` or a 10x step decay after ``lr_drop`` epochs. The
    ``min_factor`` and ``lr_drop`` values are read from ``lr_scheduler_kwargs`` first (the current API),
    falling back to the deprecated ``lr_min_factor`` / ``lr_drop`` fields.

    Args:
        optimizer: The optimizer the scheduler drives.
        train_config: Training config carrying the preset name and schedule knobs.
        total_steps: Total optimizer steps over the whole run.
        steps_per_epoch: Optimizer steps per epoch.
        warmup_steps: Number of optimizer steps in the linear warmup ramp.

    Returns:
        A ``LambdaLR`` implementing the managed schedule.
    """
    kwargs = train_config.lr_scheduler_kwargs
    # Managed presets are always strings (guaranteed by the _is_managed_scheduler_name branch at the call site).
    preset = cast(str, train_config.lr_scheduler).strip().lower()
    min_factor = float(kwargs.get("min_factor", train_config.lr_min_factor))
    lr_drop = int(kwargs.get("lr_drop", train_config.lr_drop))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if preset == "cosine":
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        # Step decay: drop by 10x after lr_drop epochs.
        if current_step < lr_drop * steps_per_epoch:
            return 1.0
        return 0.1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _wrap_with_warmup(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Prepend a linear warmup ramp to an explicit scheduler via ``SequentialLR``.

    The warmup ramps the LR from ``1 / warmup_steps`` of its base value up to full over
    ``warmup_steps`` optimizer steps, then hands control to ``scheduler``. ``ReduceLROnPlateau`` is
    metric-driven and cannot be composed this way — callers must skip wrapping it.

    Args:
        scheduler: The explicit scheduler to run after warmup.
        optimizer: The optimizer both schedulers drive.
        warmup_steps: Number of optimizer steps in the linear warmup ramp.

    Returns:
        A ``SequentialLR`` chaining the warmup ramp and ``scheduler``.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, warmup_steps),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warmup_steps],
    )


class RFDETRModelModule(LightningModule):
    """LightningModule wrapping the RF-DETR model and training loop.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        # Manual optimization is enabled only for keypoint models so that the box-count
        # normalizer can be accumulated across grad-accum microbatches. Detection and
        # segmentation use Lightning's automatic optimization (PTL handles accumulation,
        # AMP, and gradient clipping), which keeps their step semantics unchanged from
        # the pre-fix/scaling behaviour.
        self._use_manual_optimization: bool = bool(getattr(model_config, "use_grouppose_keypoints", False))
        self.automatic_optimization = not self._use_manual_optimization
        # LR-scheduler stepping cadence resolved in configure_optimizers(); read by the manual-optimization
        # step loop and the epoch-end hook. Defaults keep pre-configure behaviour (per-step stepping).
        self._lr_scheduler_interval: str = "step"
        self._lr_scheduler_monitor: str | None = None
        self._accumulated_box_normalizer: Tensor | None = None
        # Allow partial state-dict loading when resuming from a .pth checkpoint
        # (which contains only model weights, not criterion/postprocess state).
        self.strict_loading = False

        # Model, criterion, and postprocessor.
        self.model = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            # Canonical loader handles PE interpolation, PTL .ckpt normalisation,
            # per-group query slicing, class-name extraction, partial-load warnings,
            # and writes any auto-aligned ``num_classes`` back onto ``model_config``.
            load_pretrain_weights(self.model, self.model_config)
            proto_guidance = getattr(self.model.transformer, "proto_guidance", None)
            artifacts_path = self.model_config.proto_guidance_artifacts_path
            if proto_guidance is not None and artifacts_path:
                # 显式实验原型必须覆盖起始 checkpoint 携带的旧原型 buffer。
                proto_guidance.reload_artifacts(artifacts_path)
            if (
                proto_guidance is not None
                and "proto_guidance_gate_bias_init" in self.model_config.model_fields_set
            ):
                # checkpoint 会覆盖 gate 参数；实验显式指定的初始开启度必须随后恢复，
                # 否则 content 强度扫描会被旧 checkpoint 的 gate 偏置静默抵消。
                proto_guidance.reset_content_gate_bias(self.model_config.proto_guidance_gate_bias_init)
                logger.info(
                    "[ProtoGuidance] 已按实验配置重设内容 gate 偏置：%.4f",
                    self.model_config.proto_guidance_gate_bias_init,
                )
            if model_config.use_grouppose_keypoints:
                # Older model shims may omit the keypoint reset hook; call it only when implemented.
                reset_keypoint_gaussian_parameters = getattr(self.model, "reset_keypoint_gaussian_parameters", None)
                if callable(reset_keypoint_gaussian_parameters):
                    reset_keypoint_gaussian_parameters()
                    logger.info(
                        "Reset keypoint Gaussian precision outputs to unit values after pretrained weight load."
                    )
        if model_config.backbone_lora:
            apply_lora(self.model)

        # Build criterion/postprocessors after potential num_classes alignment so
        # they are constructed with a config that matches the current model head.
        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)

        # [SSCL + SemHead + QNorm-Obj] 语义头/对比学习初始化顺序（必须在
        # configure_optimizers() 之前调用，确保优化器只包含解冻后的可训练参数）：
        # 1) 先装配语义头与 QNorm-Obj（创建各自参数）——必须在冻结之前，
        #    否则 freeze 会把 requires_grad 全部置 False；
        # 2) 保守冻结策略同时覆盖 SSCL 与"仅语义头"场景（E4c 关闭 SSCL 时仍生效）；
        # 3) 再初始化 SSCL 损失与基类蒸馏。
        if self.train_config.semantic_head_enabled:
            self._setup_semantic_head()
        if self.train_config.qnorm_obj_enabled:
            self._setup_qnorm_obj()
        if (
            self.train_config.sscl_enabled
            or self.train_config.semantic_head_enabled
            or self.model_config.proto_guidance_enabled
        ) and (self.train_config.sscl_freeze_strategy == "conservative"):
            self._apply_sscl_freeze()
        if self.train_config.sscl_enabled:
            self._setup_sscl()
        # [ProtoGuidance] 监控器装配（模块本体已在 LWDETR 构建；此处只建监控与回调）
        if self.model_config.proto_guidance_enabled:
            self._setup_proto_guidance_monitor()

        # torch.compile is opt-in: set model_config.compile=True to enable.
        # Only enabled on CUDA; MPS and CPU do not benefit from compilation.
        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from rfdetr.config import DEVICE

        accelerator = str(train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        # 门控：compile=True + CUDA 即启用编译；dynamic 按"实际生效的输入尺寸数"
        # 选择——单一 scale（默认配置 skip_random_resize 后固定 800×800）输入形状
        # 静态，dynamic=False 的 guard 检查最省 CPU；多 scale（动态形状）用
        # dynamic=True 的符号形状图覆盖所有 (H, W) 变体，避免逐 shape 重编译风暴。
        effective_scales = self._effective_input_scales()
        dynamic_shapes = len(effective_scales) > 1
        compile_enabled = model_config.compile and DEVICE == "cuda" and uses_cuda_accelerator
        if compile_enabled:
            logger.info(
                "torch.compile 已启用（有效输入 scale: %s, dynamic=%s）",
                effective_scales,
                dynamic_shapes,
            )
            # dynamic 仅当多 scale（动态输入 shape）时需要：一个编译图
            # 覆盖所有 (H, W) 变体，避免逐 shape 重编译。单一 scale 时输入
            # shape 固定，dynamic=False 的 guard 检查大幅简化——实测 dynamic=True
            # 的 dynamo 运行时开销持续吃 5+ 核 CPU（小 batch 下 GPU 闲置、step
            # 时间被固定开销主导），dynamic=False 可显著缓解。
            dynamic = dynamic_shapes
            # suppress_errors=True: if inductor can't
            # compile a subgraph (e.g. bicubic backward with symbolic shapes), it falls
            # back to eager mode for that subgraph rather than crashing.
            # capture_scalar_outputs=True: include Tensor.item() calls
            # (gen_encoder_output_proposals / ms_deform_attn use spatial-shape .item()
            # as Python slice indices). Safe because item() results are backed symbols
            # derived from input shapes — not unbacked symbols that would cause
            # PendingUnbackedSymbolNotFound (which only occurs without dynamic).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            # OptimizedModule forwards attribute access to the wrapped LWDETR via
            # __getattr__ at runtime, so self.model keeps working everywhere it's used below.
            self.model = torch.compile(self.model, dynamic=dynamic)  # type: ignore[assignment]

    def _effective_input_scales(self) -> list[int]:
        """返回训练时实际生效的输入尺寸列表（与数据集变换的 scale 推导保持一致）。

        数据增强管线（``make_coco_transforms`` / ``make_coco_transforms_square_div_64``）
        的 scale 逻辑：
        - ``multi_scale=False`` → 固定 ``[resolution]``；
        - ``multi_scale=True`` → ``compute_multi_scale_scales()`` 展开候选集
          （受 ``expanded_scales``、``patch_size``、``num_windows`` 影响）；
        - ``do_random_resize_via_padding=False``（默认）→ ``skip_random_resize``，
          只取最大单一 scale（本仓库默认配置下即固定 800×800）。

        该函数供 torch.compile 门控使用：只有单一 scale 时输入形状静态，
        ``dynamic=False`` 的编译图安全；多 scale 时输入形状动态，需禁用编译
        或改用 dynamic 模式。

        Returns:
            训练时实际出现的输入边长列表（像素）。
        """
        if not self.train_config.multi_scale:
            return [self.model_config.resolution]
        from rfdetr.datasets.coco import compute_multi_scale_scales

        scales = compute_multi_scale_scales(
            self.model_config.resolution,
            self.train_config.expanded_scales,
            self.model_config.patch_size,
            self.model_config.num_windows,
        )
        if not self.train_config.do_random_resize_via_padding:
            scales = [scales[-1]]
        return scales

    # ------------------------------------------------------------------
    # [SSCL] 语义相似度引导的对比学习
    # ------------------------------------------------------------------

    def _setup_sscl(self) -> None:
        """初始化 SSCL 训练逻辑。

        步骤：
        1. 加载 CLIP 类别语义相似度矩阵。
        2. 构建 SSCL 对比损失模块。
        3. 配置保守冻结策略（仅解冻 decoder 最后一层 + 分类头）。
        4. 可选地加载教师模型并构建基类蒸馏损失。
        5. 将 SSCL 损失权重加入 criterion 的 weight_dict。
        6. 注册 SSCL 损失回调到 criterion。

        Raises:
            ValueError: 当 SSCL 相关配置缺失或无效时抛出。
        """
        from rfdetr.sscl import (
            SSCLLoss,
            load_semantic_matrix,
            normalize_semantic_matrix,
        )

        cfg = self.train_config
        if cfg.sscl_semantic_matrix_path is None:
            raise ValueError("启用 SSCL 时必须指定 sscl_semantic_matrix_path。")
        semantic_matrix = load_semantic_matrix(cfg.sscl_semantic_matrix_path)
        # 语义矩阵后处理：CLIP 原始余弦相似度在军事/遥感文本空间中较密集，
        # 默认 minmax 归一化到 [0, 1] 以增强易混类别对的判别度
        if cfg.sscl_matrix_normalize != "none":
            semantic_matrix = normalize_semantic_matrix(
                semantic_matrix,
                mode=cfg.sscl_matrix_normalize,
            )

        # 构建 SSCL 对比损失（作用在 matched foreground query features 上）
        # hidden_dim 恒从 model_config 传入，避免原型库惰性 resize_ 与编译/续训交互；
        # projection_dim 非 None 时启用投影头（把特征映射到低维对比空间再算损失），
        # 原型库维度随之改为 projection_dim
        self.sscl_loss = SSCLLoss(
            semantic_matrix=semantic_matrix,
            tau=cfg.sscl_tau,
            rho=cfg.sscl_rho,
            omega_max=cfg.sscl_omega_max,
            anchor_classes=cfg.sscl_anchor_classes,
            confusing_classes=cfg.sscl_confusing_classes,
            prototype_mode=cfg.sscl_prototype_enabled,
            prototype_momentum=cfg.sscl_prototype_momentum,
            prototype_min_samples=cfg.sscl_prototype_min_samples,
            prototype_sync_ddp=cfg.sscl_prototype_sync_ddp,
            prototype_max_slots=cfg.sscl_prototype_max_slots,
            prototype_multi_slot_classes=cfg.sscl_prototype_multi_slot_classes,
            prototype_group_pairs=cfg.sscl_prototype_group_pairs,
            prototype_group_weight=cfg.sscl_prototype_group_weight,
            hidden_dim=self.model_config.hidden_dim,
            projection_dim=cfg.sscl_projection_dim if cfg.sscl_projection_enabled else None,
            prototype_instance_pos=cfg.sscl_prototype_instance_pos,
        )
        model_for_prototype = getattr(self.model, "_orig_mod", self.model)
        calibrator = getattr(model_for_prototype, "prototype_logit_calibrator", None)
        if calibrator is not None:
            if not cfg.sscl_prototype_enabled:
                raise ValueError("启用原型 logit 校准时必须启用 SSCL 原型库")
            if cfg.sscl_projection_enabled:
                raise ValueError("原型 logit 校准要求关闭 SSCL 投影层")
            if calibrator.max_slots != cfg.sscl_prototype_max_slots:
                raise ValueError("原型 logit 校准与 SSCL 原型库的槽位数必须一致")

        # 难例负样本监控累加器（仅启用时创建；epoch 末输出 train/sscl/* 指标）
        if cfg.sscl_hard_neg_enabled:
            from rfdetr.sscl.hard_neg_monitor import HardNegMonitor

            self._hard_neg_monitor = HardNegMonitor()

        # 冻结策略校验（实际冻结在 __init__ 中统一应用，同时覆盖 SSCL 与语义头场景）。
        if cfg.sscl_freeze_strategy == "none":
            logger.info("[SSCL] 冻结策略为 none：保持全量微调，不冻结任何参数")
        elif cfg.sscl_freeze_strategy != "conservative":
            raise ValueError(f"不支持的 sscl_freeze_strategy: {cfg.sscl_freeze_strategy}，可选: 'conservative', 'none'")

        # 可选：基类蒸馏（保护飞机类/FSC 指标）
        if cfg.sscl_distill_enabled:
            self._setup_sscl_distill()

        # 将 SSCL 损失权重注入 criterion 的 weight_dict，使现有训练循环
        # 的 loss 聚合逻辑自动包含 SSCL 项
        self.criterion.weight_dict["loss_sscl"] = cfg.sscl_lambda
        if cfg.sscl_hard_neg_enabled:
            self.criterion.weight_dict["loss_sscl_hard_neg"] = cfg.sscl_hard_neg_loss_lambda

        # 注册 SSCL 回调：在 criterion forward() 内复用 Hungarian matching
        # 的 indices，避免重复匹配。若同时启用语义头，把监控回调与 SSCL 回调
        # 组合成单一回调注册（set_sscl_loss_fn 是单槽）。
        if getattr(self, "_semantic_monitor", None) is not None:
            sscl_cb = self._sscl_loss_callback
            monitor_cb = self._semantic_monitor_callback

            def combined(
                outputs: dict[str, Any],
                targets: list[dict[str, Tensor]],
                indices: list[tuple[Tensor, Tensor]],
            ) -> dict[str, Tensor]:
                """组合回调：先算 SSCL 损失，再更新语义监控（返回空 dict 不污染 loss）。"""
                merged = dict(sscl_cb(outputs, targets, indices))
                monitor_cb(outputs, targets, indices)
                return merged

            self.criterion.set_sscl_loss_fn(combined)
        else:
            self.criterion.set_sscl_loss_fn(self._sscl_loss_callback)
        logger.info(
            f"[SSCL] 已启用：λ={cfg.sscl_lambda}, τ={cfg.sscl_tau}, ρ={cfg.sscl_rho}, "
            f"anchor={cfg.sscl_anchor_classes}, confusing={cfg.sscl_confusing_classes}, "
            f"start_epoch={cfg.sscl_start_epoch}, freeze={cfg.sscl_freeze_strategy}, "
            f"distill={cfg.sscl_distill_enabled}, "
            f"prototype={cfg.sscl_prototype_enabled}, "
            f"prototype_slots={cfg.sscl_prototype_max_slots}, "
            f"multi_slot_classes={cfg.sscl_prototype_multi_slot_classes}, "
            f"prototype_groups={cfg.sscl_prototype_group_pairs}, "
            f"group_weight={cfg.sscl_prototype_group_weight}, "
            f"projection={cfg.sscl_projection_enabled} (dim={cfg.sscl_projection_dim}), "
            f"instance_pos={cfg.sscl_prototype_instance_pos}, "
            f"hard_neg={cfg.sscl_hard_neg_enabled} "
            f"(topk={cfg.sscl_hard_neg_topk}, thresh={cfg.sscl_hard_neg_score_thresh}, "
            f"target_classes={cfg.sscl_hard_neg_target_classes}, "
            f"lambda={cfg.sscl_hard_neg_loss_lambda})"
        )

    def _apply_sscl_freeze(self) -> None:
        """应用 SSCL 保守冻结策略。

        冻结：backbone、encoder 主体、bbox 头、early decoder layers、 refpoint_embed、query_feat、enc_out_class_embed。 解冻：decoder
        末尾若干层、decoder 最终 LayerNorm、分类头 class_embed、 语义头（SemHead）、QNorm-Obj 与原型引导（ProtoGuidance）附加模块参数（若装配）。

        该策略保证 SSCL 只通过 decoder 最后一层重塑 query 特征空间， 不扰动主干与目标定位能力。

        [ProtoGuidance] ``proto_guidance_freeze_all_except_proto=True``（阶段 A 冷启动） 时跳过 decoder/norm/class_embed
        解冻，仅新模块参数可训练。
        """
        model = self.model
        # 先冻结全部参数
        for param in model.parameters():
            param.requires_grad = False
        decoder_layers = model.transformer.decoder.layers
        num_decoder_layers = len(decoder_layers)
        unfreeze_layers = min(int(self.train_config.sscl_unfreeze_decoder_layers), num_decoder_layers)
        # [ProtoGuidance] 阶段 A 冷启动：除原型模块外全部冻结（跳过下述解冻分支）
        freeze_all_except_proto = bool(self.train_config.proto_guidance_freeze_all_except_proto)
        if not freeze_all_except_proto:
            for layer in decoder_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # 解冻 decoder 最终 LayerNorm（作用于最后一层输出，含可训练仿射参数）
            if getattr(model.transformer.decoder, "norm", None) is not None:
                for param in model.transformer.decoder.norm.parameters():
                    param.requires_grad = True
            # 解冻分类头
            for param in model.class_embed.parameters():
                param.requires_grad = True
        # [SemHead] 语义参数纳入可训练（冻结全部后这里按配置恢复，与"先全冻结
        # 再选择性解冻"的既有策略一致）：α 由 alpha_enabled+learnable 决定，
        # θ 在 frozen_threshold_classes 上保持冻结（novel 类样本太少学不准，
        # 通过梯度置零 hook 实现）。注意：θ 是整张量参数，只能整张量设
        # requires_grad，冻结按类别走 _frozen_theta_mask。
        if getattr(model, "semantic_residual", None) is not None:
            cfg = self.train_config
            sr = model.semantic_residual
            frozen_theta = list(cfg.semantic_frozen_threshold_classes or [])
            sr.alpha.requires_grad = bool(cfg.semantic_alpha_enabled and cfg.semantic_alpha_learnable)
            sr.theta.requires_grad = bool(cfg.semantic_mask_enabled)
            sr.set_frozen_theta_classes(frozen_theta)
            logger.info(f"[SemHead] 语义参数可训练性恢复完成：{sr.describe_freeze()}")
        # [QNorm-Obj] QNorm 参数纳入可训练（近恒等初始化的门控/校准参数与语义头
        # 同理，在"先全冻结再选择性解冻"后恢复；freeze=none 全量微调下天然可训练）
        if getattr(model, "qnorm_obj", None) is not None:
            for param in model.qnorm_obj.parameters():
                param.requires_grad = True
            logger.info(f"[QNormObj] QNorm 参数可训练性恢复完成：{model.qnorm_obj.describe_freeze()}")
        # [ProtoGuidance] 原型引导模块参数纳入可训练（新模块默认学习，warmup
        # 控制注入强度；阶段 A 时此处是唯一可训练参数来源）
        proto_guidance = getattr(model.transformer, "proto_guidance", None)
        if proto_guidance is not None:
            for param in proto_guidance.parameters():
                param.requires_grad = True
            trainable_scope = getattr(self.train_config, "proto_guidance_trainable_scope", "all")
            if trainable_scope == "token":
                for param in proto_guidance.parameters():
                    param.requires_grad = False
                for param in proto_guidance.fusion.projectors.proj_token.parameters():
                    param.requires_grad = True
            elif trainable_scope == "token_fg":
                for param in proto_guidance.parameters():
                    param.requires_grad = False
                for param in proto_guidance.fusion.projectors.proj_token.parameters():
                    param.requires_grad = True
                for param in proto_guidance.foreground_head.parameters():
                    param.requires_grad = True
            logger.info(f"[ProtoGuidance] 原型引导参数可训练性恢复完成：{proto_guidance.describe_freeze()}")
        logger.info(
            "[SSCL] 冻结策略已应用：%s（decoder 末尾 %d/%d 层 + decoder norm + class_embed + 附加模块参数可训练）",
            "阶段 A 冷启动（仅原型模块）" if freeze_all_except_proto else "标准保守策略",
            unfreeze_layers,
            num_decoder_layers,
        )

    def _sscl_loss_callback(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        """SSCL 损失回调，在 criterion forward() 内被调用。

        从 decoder 最后一层的 hidden states 中提取 Hungarian matching 后
        的 matched foreground query features，计算语义加权的对比损失。

        Args:
            outputs: 模型输出字典，须包含 ``"hs"``。
            targets: 目标字典列表。
            indices: criterion 中已计算的 Hungarian matching 结果。

        Returns:
            包含 ``{"loss_sscl": 损失标量}`` 的字典；当无 ``"hs"`` 输出时返回空字典。
        """
        if "hs" not in outputs:
            return {}
        features, labels = self._extract_matched_query_features(outputs["hs"], indices, targets)
        # [SSCL-HN] 难负样本：仅训练阶段选择（验证阶段不选、不喂监控）。
        # 新机制直接监督未匹配高分 query 降低前景分数，并可选远离前景原型。
        # 以 _hard_neg_monitor 是否存在作为门控（_setup_sscl 中按配置创建），
        # 与 _semantic_monitor 的判空模式一致。
        hn_result: dict[str, Tensor] = {}
        if self.training and getattr(self, "_hard_neg_monitor", None) is not None:
            hn_batch_idx, hn_query_idx, hn_stats = self._select_hard_negatives(outputs, targets, indices)
            hn_result, hn_loss_stats = self._hard_negative_suppression_loss(outputs, hn_batch_idx, hn_query_idx)
            # 诊断监控：节流采样，只喂 CPU 标量。
            monitor = self._hard_neg_monitor
            interval = int(self.train_config.sscl_hard_neg_log_interval)
            if self.global_step % interval == 0:
                monitor.update({**hn_stats, **hn_loss_stats})
        loss = self.sscl_loss(features, labels)
        # [SSCL] 原型模式：训练阶段用 detach 特征更新类别原型库（无梯度）。
        # 不做 start_epoch 门控——sscl_start_epoch 前 loss 权重为 0，原型更新
        # 作为预热让锚点在 SSCL 生效前就绪；验证阶段 self.training 为 False 天然门控。
        if self.training and getattr(self.sscl_loss, "prototype_mode", False):
            self.sscl_loss.update_prototypes(features.detach(), labels)
            model_for_prototype = getattr(self, "model", None)
            if model_for_prototype is not None:
                model_for_prototype = getattr(model_for_prototype, "_orig_mod", model_for_prototype)
                calibrator = getattr(model_for_prototype, "prototype_logit_calibrator", None)
                if calibrator is not None:
                    calibrator.sync_from_bank(self.sscl_loss.prototype_bank)
        return {"loss_sscl": loss, **hn_result}

    def _extract_matched_query_features(
        self,
        hs: Tensor,
        indices: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        """从 decoder hidden states 中提取 matched foreground query features。

        Args:
            hs: decoder 最后一层 hidden states ``[B, Q, D]``。
            indices: Hungarian matching 结果（每个 query 匹配到的 GT 索引）。
            targets: 目标字典列表。

        Returns:
            ``(features, labels)`` 元组：
            - features: ``[N_fg, D]`` matched foreground query features。
            - labels: ``[N_fg]`` 对应的 GT 类别标签。
        """
        idx = self.criterion._get_src_permutation_idx(indices)  # (batch_idx, query_idx)
        features = hs[idx]
        labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        return features, labels

    def _select_hard_negatives(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """按 batch 选择难负样本（每图 top-k），返回 batch/query 索引与统计。

        逐图调用 ``select_hard_negatives_for_image``：排除 Hungarian matching
        匹配到的 query，IoU 带过滤，按目标类最大前景 logit 取 top-k。
        返回索引用于后续 gather logits/hs，保留梯度。

        Args:
            outputs: 模型输出字典（须含 ``"pred_logits"``、``"pred_boxes"``、
                ``"hs"``，均为最后一层，query 轴与 ``indices`` 一致）。
            targets: 目标字典列表（每图含 ``"boxes"``，cxcywh 归一化）。
            indices: Hungarian matching 结果（每图 ``(src_idx, tgt_idx)``）。

        Returns:
            ``(batch_idx, query_idx, stats)`` 元组：
            - batch_idx/query_idx: ``[K_total]`` 难例位置索引。
            - stats: 喂给 HardNegMonitor 的 CPU 标量（``hn_count`` 每图平均
              难例数、``hn_fill_rate`` IoU 带填充率、``n_unmatched_avg``
              每图平均未匹配数、``hn_score_mean`` 与 ``hn_iou_mean``）。
        """
        from rfdetr.sscl.hard_neg_selection import select_hard_negatives_for_image

        cfg = self.train_config
        batch_parts: list[Tensor] = []
        query_parts: list[Tensor] = []
        n_selected = n_band = n_unmatched = 0
        score_sum = iou_sum = 0.0
        for b in range(len(indices)):
            hn_idx, stats = select_hard_negatives_for_image(
                pred_logits=outputs["pred_logits"][b],
                pred_boxes=outputs["pred_boxes"][b],
                gt_boxes=targets[b]["boxes"],
                matched_src=indices[b][0],
                top_k=cfg.sscl_hard_neg_topk,
                score_thresh=cfg.sscl_hard_neg_score_thresh,
                target_classes=cfg.sscl_hard_neg_target_classes,
                iou_low=cfg.sscl_hard_neg_iou_low,
                iou_high=cfg.sscl_hard_neg_iou_high,
            )
            if hn_idx.shape[0] > 0:
                query_parts.append(hn_idx)
                batch_parts.append(torch.full_like(hn_idx, b))
                score_sum += float(stats["score_mean"]) * hn_idx.shape[0]
                iou_sum += float(stats["iou_mean"]) * hn_idx.shape[0]
            n_selected += int(stats["n_selected"])
            n_band += int(stats["n_band"])
            n_unmatched += int(stats["n_unmatched"])
        if not query_parts:
            empty = torch.empty(0, dtype=torch.long, device=outputs["pred_logits"].device)
            return (
                empty,
                empty,
                {
                    "hn_count": 0.0,
                    "hn_fill_rate": 0.0,
                    "n_unmatched_avg": float(n_unmatched) / max(1, len(indices)),
                    "hn_score_mean": 0.0,
                    "hn_iou_mean": 0.0,
                },
            )
        batch_stats = {
            "hn_count": float(n_selected) / len(indices),
            "hn_fill_rate": float(n_band) / max(1, n_unmatched),
            "n_unmatched_avg": float(n_unmatched) / max(1, len(indices)),
            "hn_score_mean": score_sum / max(1, n_selected),
            "hn_iou_mean": iou_sum / max(1, n_selected),
        }
        return torch.cat(batch_parts, dim=0), torch.cat(query_parts, dim=0), batch_stats

    def _hard_negative_suppression_loss(
        self,
        outputs: dict[str, Any],
        batch_idx: Tensor,
        query_idx: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        """计算难负样本直接抑制损失。

        Args:
            outputs: 模型输出字典，包含 ``pred_logits`` 与 ``hs``。
            batch_idx: 难负样本 batch 索引 ``[K]``。
            query_idx: 难负样本 query 索引 ``[K]``。

        Returns:
            ``(loss_dict, stats)``：loss_dict 含 ``loss_sscl_hard_neg``；
            stats 为训练监控 CPU 标量。
        """
        logits = outputs["pred_logits"]
        zero = logits.sum() * 0.0
        if batch_idx.numel() == 0:
            return {"loss_sscl_hard_neg": zero}, {"hn_logit_loss": 0.0, "hn_proto_loss": 0.0}

        cfg = self.train_config
        num_foreground = logits.shape[-1] - 1
        if cfg.sscl_hard_neg_target_classes is None:
            class_idx = torch.arange(num_foreground, device=logits.device)
        else:
            class_idx = torch.as_tensor(cfg.sscl_hard_neg_target_classes, dtype=torch.long, device=logits.device)
            class_idx = class_idx[(class_idx >= 0) & (class_idx < num_foreground)]
        if class_idx.numel() == 0:
            return {"loss_sscl_hard_neg": zero}, {"hn_logit_loss": 0.0, "hn_proto_loss": 0.0}

        hn_logits = logits[batch_idx, query_idx][:, class_idx]
        hardest_logit = hn_logits.max(dim=-1).values
        # [FSC 精准抑制] 按类覆盖 margin/lambda：每个难例取目标类内 argmax 的类别，
        # 用该类专属的 margin（默认全局值）。给高真阳量类（如 FSC）放宽抑制，
        # 避免把定位偏差的真阳一起压掉（E4-hard-neg -22pt / +FSC -9pt 教训）。
        hardest_idx = hn_logits.argmax(dim=-1)  # [K] 索引到 class_idx
        hardest_class = class_idx[hardest_idx]  # [K] 实际目标类别
        margins = torch.full(
            (num_foreground,),
            float(cfg.sscl_hard_neg_logit_margin),
            device=logits.device,
        )
        for c, m in (cfg.sscl_hard_neg_target_logit_margins or {}).items():
            if 0 <= int(c) < num_foreground:
                margins[int(c)] = float(m)
        margin_q = margins[hardest_class]
        lambdas = torch.ones(num_foreground, device=logits.device)
        for c, l in (cfg.sscl_hard_neg_target_loss_lambdas or {}).items():
            if 0 <= int(c) < num_foreground:
                lambdas[int(c)] = float(l)
        lambda_q = lambdas[hardest_class]
        logit_temp = float(cfg.sscl_hard_neg_logit_temperature)
        logit_loss = (
            F.softplus((hardest_logit - margin_q) / logit_temp) * logit_temp * lambda_q
        ).mean()

        proto_loss = logit_loss.new_zeros(())
        if (
            float(cfg.sscl_hard_neg_proto_lambda) > 0.0
            and "hs" in outputs
            and getattr(self.sscl_loss, "prototype_mode", False)
        ):
            proto_slots, valid_slots = self.sscl_loss.prototype_bank.get_normalized_slot_prototypes()
            valid_class = torch.zeros(valid_slots.shape[0], dtype=torch.bool, device=valid_slots.device)
            valid_class[class_idx.to(valid_slots.device)] = True
            valid_flat = (valid_slots & valid_class.unsqueeze(1)).reshape(-1)
            if valid_flat.any():
                proto_flat = proto_slots.reshape(-1, proto_slots.shape[-1])[valid_flat]
                hn_features = self.sscl_loss._project(outputs["hs"][batch_idx, query_idx])
                hn_norm = F.normalize(hn_features, dim=-1)
                sim = hn_norm @ proto_flat.T
                proto_temp = float(cfg.sscl_hard_neg_proto_temperature)
                proto_loss = (
                    F.softplus((sim.max(dim=-1).values - float(cfg.sscl_hard_neg_proto_margin)) / proto_temp).mean()
                    * proto_temp
                    * float(cfg.sscl_hard_neg_proto_lambda)
                )

        loss = logit_loss + proto_loss
        stats = {
            "hn_logit_loss": float(logit_loss.detach().item()),
            "hn_proto_loss": float(proto_loss.detach().item()),
        }
        return {"loss_sscl_hard_neg": loss}, stats

    def _setup_semantic_head(self) -> None:
        """装配语义分类头（SemanticResidual）与训练监控。

        从 f_sem/通道统计产物构建语义残差模块挂到 ``model.semantic_residual``， 并实例化 ``SemanticMonitor`` 累加器。冻结矩阵（novel 类 θ 冻结、α 可学习性） 在
        ``_apply_sscl_freeze`` 中统一恢复。若 SSCL 未启用（仅语义头场景）， 直接注册监控回调到 criterion。
        """
        from rfdetr.sscl import SemanticMonitor, SemanticResidual

        cfg = self.train_config
        model = self.model
        # DETR 约定：class_embed.out_features = 前景类数 + 1（background 占最后一位），
        # 语义头只作用于前景类，background 列在 lwdetr 前向中用 0 增量补齐。
        num_classes = int(model.class_embed.out_features) - 1
        hidden_dim = int(self.model_config.hidden_dim)
        model.semantic_residual = SemanticResidual.build(cfg, num_classes, hidden_dim)

        # novel/base 类划分（默认 novel=舰船 0-3、base=飞机 4-23 + FSC 24）
        novel_classes = cfg.semantic_novel_classes or [0, 1, 2, 3]
        base_classes = [c for c in range(num_classes) if c not in set(novel_classes)]
        align_classes = cfg.semantic_align_classes or novel_classes
        # 监控日志的类别名：优先用配置传入的 class_names，否则用 SHWX 提示词表
        # （语义头实验当前仅在 SHWX 上进行），再退化为 c{id}
        if cfg.class_names:
            class_names = list(cfg.class_names)
        else:
            try:
                from rfdetr.sscl.prompts import SHWX_CLASS_NAMES

                class_names = [SHWX_CLASS_NAMES.get(c, f"c{c}") for c in range(num_classes)]
            except Exception:
                class_names = [f"c{c}" for c in range(num_classes)]
        self._semantic_monitor = SemanticMonitor(
            class_names=class_names,
            novel_classes=novel_classes,
            base_classes=base_classes,
            align_classes=align_classes,
            num_classes=num_classes,
        )

        # 仅语义头（无 SSCL）时直接注册监控回调（复用 criterion 的 Hungarian indices）
        if not cfg.sscl_enabled:
            self.criterion.set_sscl_loss_fn(self._semantic_monitor_callback)
        logger.info(f"[SemHead] 语义头已装配：{model.semantic_residual.describe_freeze()}")

    def _setup_qnorm_obj(self) -> None:
        """装配 QNorm-Obj + EUMix 模块（query 范数物体性门控 + 熵感知校准）。

        在冻结策略之前调用（__init__ 顺序保证），确保门控参数创建后 按冻结策略统一处理可训练性。模块挂到 ``model.qnorm_obj``， 在 lwdetr 前向中于 class_embed/语义头之后校准全层
        logits。 不加任何辅助监督损失，参数由标准检测损失隐式训练。
        """
        from rfdetr.sscl.qnorm_obj import QNormObjectness

        cfg = self.train_config
        model = self.model
        # DETR 约定：class_embed.out_features = 前景类数 + 1（background 占末位），
        # QNorm 只对前景类列施加门控，背景列由 EUMix 单独校准。
        num_classes = int(model.class_embed.out_features) - 1
        hidden_dim = int(self.model_config.hidden_dim)
        model.qnorm_obj = QNormObjectness.build(cfg, num_classes, hidden_dim)
        logger.info(f"[QNormObj] 已装配：{model.qnorm_obj.describe_freeze()}")

    def _setup_proto_guidance_monitor(self) -> None:
        """装配原型引导监控（模块本体已在 LWDETR 构建）。

        创建 ``ProtoGuidanceMonitor`` 并把采样回调注册进 criterion 的单槽回调： 若已有回调（SSCL/语义监控），组合包装；否则直接注册。采样节流 由
        ``proto_guidance_monitor_log_interval`` 控制。
        """
        from rfdetr.sscl import ProtoGuidanceMonitor

        model = self.model
        proto_guidance = getattr(model.transformer, "proto_guidance", None)
        if proto_guidance is None:
            logger.warning("[ProtoGuidance] 模块未装配（离线产物缺失？），跳过监控装配。")
            return
        cfg = self.train_config
        num_classes = int(model.class_embed.out_features) - 1
        # 监控日志的类别名（与语义头解析一致）：优先配置，否则 SHWX 提示词表，再退化 c{id}
        if cfg.class_names:
            class_names = list(cfg.class_names)
        else:
            try:
                from rfdetr.sscl.prompts import SHWX_CLASS_NAMES

                class_names = [SHWX_CLASS_NAMES.get(c, f"c{c}") for c in range(num_classes)]
            except Exception:
                class_names = [f"c{c}" for c in range(num_classes)]
        watch_classes = proto_guidance.target_classes or list(range(num_classes))
        self._proto_guidance_monitor = ProtoGuidanceMonitor(
            class_names=class_names,
            watch_classes=watch_classes,
        )

        # 组合注册（set_sscl_loss_fn 是单槽）：已有回调时包装，先执行既有逻辑
        existing = getattr(self.criterion, "_sscl_loss_fn", None)
        if existing is not None:
            proto_cb = self._proto_guidance_monitor_callback

            def combined_with_proto(
                outputs: dict[str, Any],
                targets: list[dict[str, Tensor]],
                indices: list[tuple[Tensor, Tensor]],
            ) -> dict[str, Tensor]:
                """组合回调：先执行既有逻辑（SSCL 损失/语义监控），再更新原型监控。"""
                merged = dict(existing(outputs, targets, indices))
                proto_cb(outputs, targets, indices)
                return merged

            self.criterion.set_sscl_loss_fn(combined_with_proto)
        else:
            self.criterion.set_sscl_loss_fn(self._proto_guidance_monitor_callback)
        logger.info(
            f"[ProtoGuidance] 监控已装配（watch 类 {watch_classes}，"
            f"节流 {self.model_config.proto_guidance_monitor_log_interval} 步）"
        )

    def _proto_guidance_monitor_callback(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        """原型引导监控回调，在 criterion forward() 内被调用（复用 Hungarian indices）。

        Args:
            outputs: 模型输出字典，须包含 ``"proto_stats"``。
            targets: 目标字典列表。
            indices: criterion 中已计算的 Hungarian matching 结果。

        Returns:
            空字典（监控回调不产生损失）。
        """
        stats = outputs.get("proto_stats")
        if stats is None:
            return {}
        interval = int(self.model_config.proto_guidance_monitor_log_interval)
        if self.global_step % interval == 0:
            self._proto_guidance_monitor.update(stats)
        return {}

    def _semantic_monitor_callback(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        """语义监控回调，在 criterion forward() 内被调用（每 ``semantic_monitor_log_interval`` 步采样一次）。

        复用 Hungarian matching 提取 matched 特征，从 ``outputs["semantic_stats"]``
        读取掩码/语义增量，计算贡献占比与对齐余弦，喂给 ``SemanticMonitor``。

        Args:
            outputs: 模型输出字典（含 ``"semantic_stats"`` 与 ``"hs"``）。
            targets: 目标字典列表。
            indices: Hungarian matching 结果。

        Returns:
            空字典（监控不产生损失，避免污染 loss_dict 与 train/* 日志键）。
        """
        interval = int(getattr(self.train_config, "semantic_monitor_log_interval", 100))
        if self.global_step % interval != 0:
            return {}
        if "semantic_stats" not in outputs or getattr(self, "_semantic_monitor", None) is None:
            return {}
        stats = outputs["semantic_stats"]
        monitor: SemanticMonitor = self._semantic_monitor

        # matched 特征与标签（用于对齐余弦与贡献占比的类别分组）
        features, labels = self._extract_matched_query_features(outputs["hs"], indices, targets)
        s_matrix = self.model.semantic_residual.S
        alpha = stats["alpha"]  # [C]
        theta = stats["theta"]  # [C]
        m = stats["M"]  # [C, d]

        # 贡献占比：|增量| / (|原版 logits|+ε)，按类别聚合（matched 部分）。
        # 增量是 [L, B, Q, C] 全层栈，pred_logits 是最后一层，取 [-1] 对齐。
        pred_logits = outputs["pred_logits"]
        idx = self.criterion._get_src_permutation_idx(indices)
        matched_logits = pred_logits[idx]  # [N_fg, C+1]（末位为 background）
        matched_sem = stats["sem_delta"][-1][idx]  # [N_fg, C] 前景类增量
        matched_mask = stats["mask_delta"][-1][idx]
        num_classes = int(self.model.class_embed.out_features) - 1  # 前景类数
        # 贡献占比分母只统计前景类 logits（切掉 background 列）
        matched_fg = matched_logits[:, :num_classes]
        ratio_sem = torch.zeros(num_classes, device=matched_logits.device)
        ratio_mask = torch.zeros(num_classes, device=matched_logits.device)
        for c in range(num_classes):
            sel = labels == c
            if sel.any():
                denom = matched_fg[sel].abs().sum() + 1e-6
                ratio_sem[c] = matched_sem[sel].abs().sum() / denom
                ratio_mask[c] = matched_mask[sel].abs().sum() / denom

        # 对齐余弦：每类 matched 特征均值与 s_c 的余弦（验证特征是否向语义方向靠拢）
        with torch.no_grad():
            s_norm = torch.nn.functional.normalize(s_matrix, dim=-1)
            align = torch.zeros(num_classes, device=features.device)
            for c in range(num_classes):
                sel = labels == c
                if sel.any():
                    h_mean = torch.nn.functional.normalize(features[sel].mean(dim=0), dim=-1)
                    align[c] = torch.nn.functional.cosine_similarity(h_mean.unsqueeze(0), s_norm[c].unsqueeze(0))

        monitor.update(
            {"alpha": alpha, "theta": theta, "M": m, "ratio_sem": ratio_sem, "ratio_mask": ratio_mask, "align": align}
        )
        return {}

    def _setup_sscl_distill(self) -> None:
        """构建教师模型与基类蒸馏损失。

        教师使用与学生学习完全一致的架构，从原始 RF-DETR checkpoint 加载
        权重并完全冻结。蒸馏仅作用于受保护类别（默认飞机类 + FSC）的
        logits 通道，舰船类通道不参与蒸馏。

        Raises:
            ValueError: 当未指定 ``sscl_teacher_checkpoint`` 时抛出。
        """
        from copy import deepcopy

        from rfdetr.sscl import BaseClassDistillLoss

        cfg = self.train_config
        if cfg.sscl_teacher_checkpoint is None:
            raise ValueError("启用基类蒸馏时必须指定 sscl_teacher_checkpoint。")

        # 构建教师模型：架构与学生一致，从原始 checkpoint 加载权重
        teacher_config = deepcopy(self.model_config)
        teacher_config.pretrain_weights = cfg.sscl_teacher_checkpoint  # type: ignore[assignment]
        teacher_model = build_model_from_config(teacher_config, self.train_config)
        load_pretrain_weights(teacher_model, teacher_config)
        # 完全冻结教师模型，仅在 no_grad 下前向
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        self.sscl_teacher = teacher_model

        # 受保护类别：默认飞机类 (4-23) + FSC (24)，舰船类 (0-3) 不蒸馏。
        # 原因：蒸馏舰船类会把学生锚定到 teacher 已混乱的舰船边界，与 SSCL 目标冲突。
        protected_classes = cfg.sscl_protected_classes or list(range(4, self.model_config.num_classes))
        self.sscl_distill_loss = BaseClassDistillLoss(
            protected_classes=protected_classes,
            temperature=cfg.sscl_distill_temperature,
            mode=cfg.sscl_distill_mode,
        )
        self.criterion.weight_dict["loss_sscl_distill"] = cfg.sscl_distill_lambda
        logger.info(f"[SSCL] 基类蒸馏已启用：受保护类别 {protected_classes}，λ={cfg.sscl_distill_lambda}")

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Seed RNGs at fit start when ``TrainConfig.seed`` is set.

        This avoids hidden global side-effects in ``build_trainer`` while still preserving deterministic training
        behaviour for actual fit runs.
        """
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed + self.global_rank, workers=True)

    def on_train_start(self) -> None:
        """Normalize restored fused-optimizer state before the first training step.

        Lightning restores optimizer state after ``on_fit_start``.  Fused AdamW is strict about the dtype, device, and
        layout of its moment tensors, so resuming from a checkpoint can fail if Lightning rehydrates those tensors in a
        layout that no longer matches the live parameters.  Recasting same-shaped floating-point tensors here keeps the
        resumed optimizer compatible without discarding the saved momentum state.
        """
        if not self._use_fused_optimizer:
            return

        try:
            optimizers = self.optimizers(use_pl_optimizer=False)
        except RuntimeError:
            return
        if optimizers is None:
            return
        if isinstance(optimizers, list):
            optimizer_list = optimizers
        else:
            optimizer_list = [optimizers]

        normalized_tensors = 0
        for optimizer in optimizer_list:
            if not isinstance(optimizer, torch.optim.Optimizer):
                optimizer = getattr(optimizer, "optimizer", optimizer)
            if isinstance(optimizer, torch.optim.Optimizer):
                normalized_tensors += self._normalize_optimizer_state(optimizer)
        if normalized_tensors and getattr(self.trainer, "is_global_zero", True):
            logger.info(
                "Normalized %d restored fused AdamW state tensors after checkpoint resume.",
                normalized_tensors,
            )

    def on_train_batch_start(self, batch: tuple[Any, Any], batch_idx: int) -> None:
        """Apply optional multi-scale resize to the incoming batch.

        Modifications to ``batch`` (in-place on ``NestedTensor``) are visible in ``training_step`` because they share
        the same object.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Index of the current batch within the epoch.
        """
        tc = self.train_config
        mc = self.model_config

        if tc.multi_scale and not tc.do_random_resize_via_padding:
            samples, _ = batch
            scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)
            step = self.trainer.global_step
            # Use a step-local generator so the scale choice is deterministic and DDP-consistent
            # without reseeding the process-global RNG on every batch.
            scale = random.Random(step).choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

    def on_train_epoch_start(self) -> None:
        """Reset the accumulated box normalizer at the start of every training epoch.

        Lightning may reuse the module across epochs without calling ``_step_optimizer`` at the boundary (for example
        when an epoch ends mid-accumulation window with a non-divisible batch count). Clearing the accumulator here
        guarantees the manual-optimization path always starts each epoch from a known state, so the first microbatch's
        gradients are scaled by its own box count and not by a stale previous-epoch denominator.

        This is a no-op for non-keypoint models because they use Lightning's automatic optimization path and never
        populate ``self._accumulated_box_normalizer``.

        Note: on finite datasets the final-batch fallback in ``_should_step_optimizer`` always flushes a partial
        trailing window, so this reset is the only change needed.  On IterableDatasets (infinite
        ``num_training_batches``) a partial window may survive epoch end with un-stepped gradients; those are
        discarded here and the optimizer is zeroed so the first microbatch of the new epoch starts from a clean state.
        """
        if self._accumulated_box_normalizer is not None:
            # Discard any partial accumulation window that survived the epoch boundary
            # (only possible for IterableDatasets where num_training_batches is infinite).
            try:
                opts = self.optimizers()
                for opt in opts if isinstance(opts, list) else [opts]:
                    opt.zero_grad()
            except RuntimeError:
                pass  # Not attached to Trainer (unit-test context); nothing to zero.
        self._accumulated_box_normalizer = None
        # 每个 epoch 开始时清理 GPU 缓存，释放碎片化的显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # [ProtoGuidance] 注入当前 epoch 驱动 lambda/gamma 的 warmup 线性调度
        # （getattr 链保证无 model 属性的 mock 场景安全）
        proto_guidance = getattr(getattr(getattr(self, "model", None), "transformer", None), "proto_guidance", None)
        if proto_guidance is not None:
            proto_guidance.current_epoch = float(self.current_epoch)

    def training_step(self, batch: tuple[Any, Any], batch_idx: int) -> Tensor | dict[str, Any]:
        """Compute loss for one training step and log metrics.

        PTL handles AMP (``precision``) without a manual ``GradScaler``. Keypoint models perform manual optimization so
        box-count loss normalization is based on the full accumulated effective batch rather than each microbatch
        independently; detection and segmentation models keep Lightning's automatic optimization path.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor by default. When ``compute_train_metrics=True``,
            returns a Lightning-compatible dict containing ``loss`` plus
            detached postprocessed predictions for train mAP logging.
        """
        samples, targets = batch
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        if self._use_manual_optimization:
            loss_dict, raw_loss, normalizer = self._compute_train_losses(outputs, targets)
            loss_for_backward = self._scale_loss_for_accumulation(raw_loss, normalizer)
        else:
            loss_dict = self.criterion(outputs, targets)
            loss_for_backward = None
        # [SSCL] 基类蒸馏（自动优化路径）：教师模型在 no_grad 下前向，
        # 对学生与教师受保护类别的 logits 计算蒸馏损失并加入 loss_dict。
        if getattr(self, "sscl_teacher", None) is not None and not self._use_manual_optimization:
            with torch.no_grad():
                teacher_outputs = self.sscl_teacher(samples, targets)
            student_logits = outputs["pred_logits"]
            teacher_logits = teacher_outputs["pred_logits"]
            # 训练时学生使用 group_detr 组（如 13×300=3900 query），而教师以
            # eval 模式仅输出单组（300 query）。取学生的第 0 组与教师对齐，
            # 与推理时仅使用第 0 组的约定一致；若形状已一致则无需切片。
            if student_logits.shape[1] != teacher_logits.shape[1]:
                student_logits = student_logits[:, : teacher_logits.shape[1], :]
            loss_dict["loss_sscl_distill"] = self.sscl_distill_loss(
                student_logits,
                teacher_logits,
            )
        weight_dict = self.criterion.weight_dict
        # [SSCL] 起始 epoch 门控：在 sscl_start_epoch 之前将 SSCL 相关权重置 0，
        # 让常规检测损失先训练若干 epoch 收敛基类，再按策略从指定 epoch 开始
        # 施加语义对比约束。使用副本而非修改 self.criterion.weight_dict，
        # 避免污染后续 step 的权重。难负样本抑制独立门控
        # （sscl_hard_neg_start_epoch），可与 SSCL 解耦晚启动。
        if "loss_sscl" in weight_dict and self.current_epoch < self.train_config.sscl_start_epoch:
            weight_dict = {
                **weight_dict,
                "loss_sscl": 0.0,
            }
        if (
            "loss_sscl_hard_neg" in weight_dict
            and self.current_epoch < self.train_config.sscl_hard_neg_start_epoch
        ):
            weight_dict = {
                **weight_dict,
                "loss_sscl_hard_neg": 0.0,
            }
        loss: Tensor = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
        # Automatic optimization path: divide by accumulate_grad_batches so the accumulated
        # gradient matches a single large batch, matching the legacy engine.  PTL accumulates
        # full-scale gradients by default; dividing here keeps the effective LR identical.
        accumulate_grad_batches = max(1, int(self.trainer.accumulate_grad_batches))
        loss_for_return = loss if self._use_manual_optimization else loss / accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self._log_train_progress_metrics(loss, loss_dict, batch_size=batch_size)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        # Optimizer may have multiple param groups with different LRs (e.g., backbone/decoder).
        # Preserve the first group's LR for backward compatibility, but also log the
        # min/max across all groups so the progress bar reflects the full schedule.
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=False, on_step=True, on_epoch=False)
        if self._use_manual_optimization:
            # loss_for_backward is only None in the automatic-optimization branch above,
            # which is mutually exclusive with _use_manual_optimization.
            assert loss_for_backward is not None
            self.manual_backward(loss_for_backward)
            if self._should_step_optimizer(batch_idx):
                self._step_optimizer(optimizer)
        if self.train_config.compute_train_metrics:
            with torch.no_grad():
                orig_sizes = torch.stack([t["orig_size"] for t in targets])
                # Slice to group-0 queries only — mirrors the eval-mode path in
                # lwdetr.py that trims refpoint_embed to [:num_queries]. Without
                # this, training mode emits group_detr×num_queries queries (e.g.
                # 13×300=3900) and postprocess top-k selection draws from all
                # groups, producing OKS/mAP values ~50× below true accuracy.
                nq = self.model_config.num_queries
                # Only include tensor-valued keys — pred_masks is a dict in
                # train mode (sparse_forward) and postprocess cannot handle it.
                inference_outputs = {
                    k: v[:, :nq] if v.ndim >= 2 else v
                    for k, v in outputs.items()
                    if k in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints") and isinstance(v, Tensor)
                }
                results = self.postprocess(inference_outputs, orig_sizes)
            return {
                "loss": loss_for_return.detach() if self._use_manual_optimization else loss_for_return,
                "results": self._detach_results(results),
                "targets": targets,
            }
        return loss_for_return.detach() if self._use_manual_optimization else loss_for_return

    def _compute_train_losses(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Compute normalized losses for logging and raw weighted loss for backward.

        Args:
            outputs: Model output dictionary.
            targets: Target dictionaries for the current batch.

        Returns:
            A tuple of normalized loss dictionary, unnormalized weighted loss numerator, and box normalizer.
        """
        weight_dict = self.criterion.weight_dict
        if "loss_sscl" in weight_dict and self.current_epoch < self.train_config.sscl_start_epoch:
            weight_dict = {
                **weight_dict,
                "loss_sscl": 0.0,
            }
        if (
            "loss_sscl_hard_neg" in weight_dict
            and self.current_epoch < self.train_config.sscl_hard_neg_start_epoch
        ):
            weight_dict = {
                **weight_dict,
                "loss_sscl_hard_neg": 0.0,
            }
        if not getattr(self.criterion, "supports_loss_normalizer_override", False):
            raise ValueError(
                f"{type(self.criterion).__name__}.supports_loss_normalizer_override is False; "
                "manual optimization (keypoint models) requires a criterion that accepts a "
                "num_boxes keyword argument. Set supports_loss_normalizer_override = True on "
                "your criterion subclass and implement the num_boxes parameter in forward()."
            )
        normalizer = self.criterion.num_boxes_for_targets(outputs, targets)
        # [P1] Logit Adjustment warmup：前 logit_adjustment_warmup_epochs 个 epoch
        # bias 从 0 线性升到目标值（global_step 为优化器步数，与 LR warmup 同口径；
        # warmup_epochs <= 0 表示立即全量生效）。
        if getattr(self.criterion, "logit_adjustment_enabled", False):
            warmup_epochs = self.train_config.logit_adjustment_warmup_epochs
            if warmup_epochs > 0:
                total_steps = max(1, self.trainer.estimated_stepping_batches)
                steps_per_epoch = max(1.0, float(total_steps) / max(1, self.train_config.epochs))
                warmup_steps = max(1.0, warmup_epochs * steps_per_epoch)
                self.criterion.set_la_warmup_factor(min(1.0, self.global_step / warmup_steps))
            else:
                self.criterion.set_la_warmup_factor(1.0)
        numerator_loss_dict = self.criterion(outputs, targets, num_boxes=torch.ones_like(normalizer))
        # Keys in weight_dict are loss terms whose criterion implementation divides by num_boxes
        # (so passing num_boxes=1.0 yields raw numerators that we divide by normalizer here).
        # Keys outside weight_dict (e.g. "class_error", "cardinality_error") are diagnostics
        # that do NOT divide by num_boxes internally — they are passed through unchanged.
        # If a future loss term divides by num_boxes AND is omitted from weight_dict, its
        # logged value will be on a different scale than the keypoint path; verify when adding
        # new criterion terms.
        loss_dict = {
            key: value / normalizer if key in weight_dict else value for key, value in numerator_loss_dict.items()
        }
        raw_loss = sum(numerator_loss_dict[k] * weight_dict[k] for k in numerator_loss_dict if k in weight_dict)
        return loss_dict, raw_loss, normalizer

    def _scale_loss_for_accumulation(
        self,
        raw_loss: Tensor,
        normalizer: Tensor,
    ) -> Tensor:
        """Scale the current numerator loss by the accumulated box denominator.

        Args:
            raw_loss: Current microbatch weighted loss numerator.
            normalizer: Current microbatch box denominator.

        Returns:
            Loss scalar to pass to ``manual_backward``.
        """
        normalizer = normalizer.detach()
        previous_normalizer = self._accumulated_box_normalizer
        accumulated_normalizer = normalizer if previous_normalizer is None else previous_normalizer + normalizer
        if previous_normalizer is not None:
            self._rescale_accumulated_gradients(previous_normalizer / accumulated_normalizer)
        self._accumulated_box_normalizer = accumulated_normalizer.detach()
        return raw_loss / accumulated_normalizer

    def _rescale_accumulated_gradients(self, scale: Tensor) -> None:
        """Rescale gradients already accumulated in the current optimizer window.

        Args:
            scale: Multiplicative factor that converts previous gradients from the old denominator to the new one.
        """
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        """Return whether the current batch closes an optimizer accumulation window.

        The optimizer steps when either:

        - The current batch closes a complete ``grad_accum_steps`` window
          (``(batch_idx + 1) % grad_accum_steps == 0``), or
        - This is the final batch of the epoch and a partial accumulation window
          is still open, so the trailing microbatches are not silently dropped.

        Lightning's ``Trainer.num_training_batches`` may be reported as ``float('inf')``
        for iterable / streaming datasets where the epoch length is unknown. In that case
        only the modulo path can ever close the window — the final-batch fallback is
        skipped because ``batch_idx + 1`` can never reach infinity.

        Args:
            batch_idx: Batch index within the epoch.

        Returns:
            ``True`` when the optimizer should step after this batch.
        """
        accum_steps = max(1, int(self.train_config.grad_accum_steps))
        if (batch_idx + 1) % accum_steps == 0:
            return True
        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        return (
            isinstance(num_training_batches, (int, float))
            and math.isfinite(num_training_batches)
            and batch_idx + 1 >= num_training_batches
        )

    def _step_optimizer(self, optimizer: torch.optim.Optimizer | LightningOptimizer) -> None:
        """Clip gradients, step optimizer and scheduler, then reset accumulation state.

        Args:
            optimizer: Optimizer returned by Lightning.
        """
        trainer_gradient_clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if trainer_gradient_clip_val is None:
            gradient_clip_val = self.train_config.clip_max_norm
        elif isinstance(trainer_gradient_clip_val, (int, float)):
            gradient_clip_val = trainer_gradient_clip_val
        else:
            gradient_clip_val = None
        gradient_clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        if not isinstance(gradient_clip_algorithm, str):
            gradient_clip_algorithm = None
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        optimizer.step()
        optimizer.zero_grad()
        self._step_lr_scheduler()
        self._accumulated_box_normalizer = None

    def _current_lr_scheduler(self) -> LRScheduler | ReduceLROnPlateau | None:
        """Return the single configured LR scheduler, or ``None`` when none is available.

        Returns:
            The scheduler object (unwrapping Lightning's single-element list), or ``None`` when no
            scheduler is configured yet or Lightning is between fit stages.
        """
        try:
            scheduler = self.lr_schedulers()
        except (AttributeError, RuntimeError):
            return None
        if isinstance(scheduler, list):
            return scheduler[0] if scheduler else None
        return scheduler

    def _step_lr_scheduler(self) -> None:
        """Step step-interval schedulers once per optimizer step (manual-optimization path).

        Epoch-interval schedulers and metric-driven ``ReduceLROnPlateau`` are stepped at epoch boundaries by
        ``on_train_epoch_end`` / ``on_validation_epoch_end`` instead, so they are skipped here.
        """
        if self._lr_scheduler_interval != "step":
            return
        scheduler = self._current_lr_scheduler()
        if scheduler is None or isinstance(scheduler, ReduceLROnPlateau):
            return
        scheduler.step()

    def on_after_backward(self) -> None:
        """采集语义头 α/θ 的梯度范数供监控（自动优化路径 backward 完成后触发）。

        在 no_grad 下**只读取** α/θ 的 ``.grad`` 范数喂给 SemanticMonitor，绝不置空： PTL 自动优化路径在 optimizer.step() 前依赖这些梯度更新 α/θ，清空会导致
        语义参数永远不更新。梯度累积（accumulate_grad_batches>1）时读取的是当前 累计梯度，用于监控"是否在学"仍具代表性。仅在语义头启用时执行。
        """
        # [SemHead] 语义头梯度范数（独立于 proto：语义头未装配时不影响其他监控）
        monitor = getattr(self, "_semantic_monitor", None)
        semantic_residual = getattr(self.model, "semantic_residual", None)
        if monitor is not None and semantic_residual is not None:
            with torch.no_grad():
                alpha_grad = semantic_residual.alpha.grad
                theta_grad = semantic_residual.theta.grad
                if alpha_grad is not None:
                    monitor.update_grad_norms(
                        alpha_grad.detach(), theta_grad.detach() if theta_grad is not None else None
                    )
                else:
                    # 无梯度（冻结参数）时也上报，便于监控确认冻结生效
                    monitor.update_grad_norms(None, None)
        # [ProtoGuidance] 原型引导模块逐参数梯度范数采集（冻结参数应恒 0，
        # 与语义头同模式：no_grad 只读不置空）
        proto_monitor = getattr(self, "_proto_guidance_monitor", None)
        proto_guidance = getattr(getattr(getattr(self, "model", None), "transformer", None), "proto_guidance", None)
        if proto_monitor is not None and proto_guidance is not None:
            grad_norms: dict[str, float] = {}
            for name, param in proto_guidance.named_parameters():
                if param.grad is not None:
                    grad_norms[name.replace(".", "_")] = float(param.grad.detach().norm().item())
                elif param.requires_grad:
                    grad_norms[name.replace(".", "_")] = 0.0
            if grad_norms:
                proto_monitor.update_grad_norms(grad_norms)

    def on_train_epoch_end(self) -> None:
        """Step epoch-interval (non-plateau) schedulers on the manual-optimization path.

        The automatic-optimization path leaves scheduler stepping entirely to Lightning; only the manual keypoint loop
        steps schedulers itself. 语义头监控指标在此统一输出到 ``train/sem/*``。
        """
        # [SemHead] 语义监控 epoch 级聚合输出（自动/手动优化路径均适用）
        monitor = getattr(self, "_semantic_monitor", None)
        if monitor is not None:
            monitor.on_train_epoch_end(self)
        # [SSCL-HN] 难例监控 epoch 级聚合输出到 train/sscl/*
        hn_monitor = getattr(self, "_hard_neg_monitor", None)
        if hn_monitor is not None:
            hn_monitor.on_train_epoch_end(self)
        # [ProtoGuidance] 原型引导监控 epoch 级聚合输出到 train/proto/*
        proto_monitor = getattr(self, "_proto_guidance_monitor", None)
        if proto_monitor is not None:
            proto_monitor.on_train_epoch_end(self)
        if self.automatic_optimization or self._lr_scheduler_interval != "epoch":
            return
        scheduler = self._current_lr_scheduler()
        if scheduler is None or isinstance(scheduler, ReduceLROnPlateau):
            return
        scheduler.step()

    def on_validation_epoch_end(self) -> None:
        """Step ``ReduceLROnPlateau`` from the monitored metric on the manual-optimization path.

        The automatic-optimization path lets Lightning feed the monitored metric; the manual keypoint loop must read it
        from ``trainer.callback_metrics`` and step the scheduler itself. The pre-training sanity-check validation is
        skipped so plateau patience/cooldown bookkeeping is not seeded from the untrained model.
        """
        if self.automatic_optimization or self.trainer.sanity_checking:
            return
        scheduler = self._current_lr_scheduler()
        if not isinstance(scheduler, ReduceLROnPlateau):
            return
        monitor = self._lr_scheduler_monitor or "val/loss"
        metric = self.trainer.callback_metrics.get(monitor)
        if metric is None:
            # Warn-and-continue would let the LR never reduce while training silently proceeds. Fail loud instead,
            # mirroring Lightning's strict-monitor behavior on the automatic-optimization path.
            raise RuntimeError(
                f"ReduceLROnPlateau monitor {monitor!r} was not found in callback_metrics, so the learning rate "
                "would never be reduced. Ensure the monitored metric is logged every validation epoch (e.g. set "
                "compute_val_loss=True for the default 'val/loss' monitor), or set lr_scheduler_monitor to a metric "
                "that is produced."
            )
        scheduler.step(metric)

    @staticmethod
    def _detach_results(results: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        """Detach postprocessed result tensors before handing them to callbacks.

        Args:
            results: Per-image postprocessed prediction dictionaries.

        Returns:
            Per-image dictionaries with tensor values detached from the graph.
        """
        return [
            {key: value.detach() if torch.is_tensor(value) else value for key, value in result.items()}
            for result in results
        ]

    def _log_train_progress_metrics(
        self,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log compact per-step convergence metrics for the progress bar only.

        Args:
            loss: Unscaled aggregate training loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        # When ``train_log_on_step`` is True, the ``train/loss`` call in ``training_step``
        # logs with ``on_step=True, on_epoch=True``; Lightning forks that into
        # ``train/loss_step`` + ``train/loss_epoch``, and ``train/loss_step`` already
        # provides the live per-step progress-bar view. Emitting this separate ``loss``
        # scalar in that case just duplicates it, so only log it on the default
        # ``train_log_on_step=False`` path.
        if not bool(self.train_config.train_log_on_step):
            self.log(
                "loss",
                loss,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        for loss_name, progress_name in _TRAIN_PROGRESS_LOSS_ALIASES.items():
            value = loss_dict.get(loss_name)
            if value is None:
                continue
            self.log(
                progress_name,
                value,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

    def _log_val_loss_metrics(
        self,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log aggregate and component validation losses.

        Args:
            loss: Aggregate weighted validation loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _resolve_eval_model(self) -> Any:
        """Return the model to forward through for validation.

        When ``TrainConfig.eval_ema_only`` is set, validation forwards through the EMA-averaged
        weights directly instead of the base model — this replaces the second, duplicate
        base+EMA forward pass ``COCOEvalCallback`` would otherwise also run every validation
        batch (see issue 416). Falls back to the base model when EMA is enabled but not yet
        warmed up (e.g. the very first validation epoch, before ``RFDETREMACallback.setup``
        has built its averaged model), or when this module isn't attached to a ``Trainer`` at
        all (``LightningModule.trainer`` raises ``RuntimeError`` rather than returning ``None``
        when unattached — e.g. ``validation_step`` called directly outside ``Trainer.fit``/
        ``Trainer.validate``).

        Uses the same ``_get_ema_inner_module`` helper as ``COCOEvalCallback`` (see
        ``coco_eval.py``) so both consumers resolve the EMA-averaged detection net through one
        shared code path instead of independently duck-typing ``RFDETREMACallback``.

        Returns:
            The base model, or the EMA-averaged inner module's underlying detection net when
            ``eval_ema_only`` is active and available.
        """
        if not self.train_config.eval_ema_only:
            return self.model
        try:
            callbacks = getattr(self.trainer, "callbacks", [])
        except RuntimeError:
            return self.model
        for callback in callbacks:
            ema_inner = _get_ema_inner_module(callback)
            if ema_inner is not None:
                return ema_inner.model
        return self.model

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one validation step.

        Returns raw results and targets so ``COCOEvalCallback`` can accumulate them across the epoch via
        ``on_validation_batch_end``.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the validation epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self._resolve_eval_model()(samples)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    @property
    def _fused_adamw_env_eligible(self) -> bool:
        """Return whether the runtime would enable fused AdamW, ignoring optimizer choice.

        Captures only the hardware/precision preconditions (BF16 on CUDA), so the
        custom-optimizer path can tell whether a dropped ``fused_optimizer=True``
        would actually have mattered.

        Returns:
            ``True`` when fused AdamW is requested and the runtime supports it.
        """
        return (
            self.model_config.fused_optimizer
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and str(self.trainer.precision) in {"bf16-mixed", "bf16", "bf16-true"}
        )

    @property
    def _use_fused_optimizer(self) -> bool:
        """Return whether fused AdamW should be used for the current training configuration.

        Fused AdamW is only safe when the trainer's actual precision is a BF16 variant.  Checking GPU capability alone
        (``is_bf16_supported()``) is
        insufficient: on Ampere+ hardware that flag is always ``True`` even when
        the trainer is configured for ``32-true``, which causes a ``params, grads, exp_avgs, and exp_avg_sqs must have
        same dtype, device, and layout`` crash in DDP because gradient bucket views have non-matching strides in FP32.
        It additionally requires the built-in ``optimizer="adamw"`` selection: fused state normalization and gradient
        clipping are AdamW-specific and must not fire for custom optimizers.

        Returns:
            ``True`` when fused AdamW is requested, safe, and the built-in AdamW optimizer is selected.

        Examples:
            >>> from unittest.mock import patch
            >>> module = RFDETRModelModule.__new__(RFDETRModelModule)
            >>> module.model_config = type("Cfg", (), {"fused_optimizer": True})()
            >>> with patch("torch.cuda.is_available", return_value=False):
            ...     module._use_fused_optimizer
            False
        """
        if not self._fused_adamw_env_eligible:
            return False
        return _is_builtin_fused_adamw(self.train_config.optimizer)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Build the configured optimizer with layer-wise LR decay and scheduler.

        Uses ``trainer.estimated_stepping_batches`` for total step count so cosine annealing covers the full training
        run regardless of dataset size or accumulation settings.
        ``optimizer="adamw"`` keeps RF-DETR's fused torch AdamW path;
        other names can be loaded from ``pytorch-optimizer``.

        Returns:
            PTL optimizer config dict with optimizer and step-interval scheduler.
        """
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        # Unwrap torch.compile's OptimizedModule so get_param_dict sees the
        # original module's named_parameters() — compiled wrapper can cause
        # name-prefix mismatches that put the same tensor in multiple groups.
        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        # [SemHead] 语义头参数会被 get_param_dict 落入 other_params（lr=args.lr 主组），
        # 这里先按参数 id 从主组过滤，再在下方追加独立语义组（lr=semantic_lr），
        # 避免同一参数被两个参数组重复更新。
        semantic_residual = getattr(model_for_params, "semantic_residual", None)
        if semantic_residual is not None:
            sem_ids = {id(p) for p in semantic_residual.parameters()}
            param_dicts = [pg for pg in param_dicts if id(pg["params"]) not in sem_ids]
        param_dicts = [param_group for param_group in param_dicts if param_group["params"].requires_grad]
        # [SSCL] 投影头参数不在 LWDETR 内部，get_param_dict 收集不到，手动追加为
        # 独立参数组。须在上一行按 requires_grad 过滤之后追加：该过滤假定每组
        # params 是单个张量，而本组 params 是列表，提前追加会抛 AttributeError。
        # getattr 兼容未启用 SSCL（无 sscl_loss 属性）的场景。
        param_dicts += get_projection_head_param_dict(getattr(self, "sscl_loss", None), tc.lr)
        # [SemHead] 语义头 α/θ 独立参数组（标量参数需足够 LR 才能学出来）
        param_dicts += get_semantic_head_param_dict(semantic_residual, tc.semantic_lr)

        optimizer_cfg = tc.optimizer
        optimizer: torch.optim.Optimizer
        if _is_builtin_fused_adamw(optimizer_cfg):
            # Built-in managed fused AdamW path (unchanged behavior).
            try:
                optimizer = torch.optim.AdamW(
                    param_dicts,
                    lr=tc.lr,
                    weight_decay=tc.weight_decay,
                    fused=self._use_fused_optimizer,
                    **tc.optimizer_kwargs,
                )
            except TypeError as exc:
                raise TypeError(
                    f"Failed to initialize optimizer 'adamw': {exc}. "
                    "Check optimizer_kwargs for arguments supported by torch.optim.AdamW."
                ) from exc
        else:
            if self._fused_adamw_env_eligible:
                logger.warning(_FUSED_IGNORED_MSG, optimizer_cfg)
            if not isinstance(optimizer_cfg, str):
                # Explicit callable / functools.partial: called with param groups only.
                callable_name = getattr(optimizer_cfg, "__qualname__", None) or repr(optimizer_cfg)
                optimizer = _instantiate_explicit_optimizer(optimizer_cfg, callable_name, param_dicts, {})
            elif _is_managed_optimizer_name(optimizer_cfg):
                # Managed native torch.optim short name (lr + signature-aware weight_decay injected).
                native_class: _OptimizerFactory = _resolve_native_optimizer(optimizer_cfg)
                optimizer = _instantiate_optimizer(native_class, optimizer_cfg, param_dicts, tc)
            else:
                # Explicit dotted import path: constructed from optimizer_kwargs only.
                optimizer_class = _import_optimizer_class(optimizer_cfg)
                optimizer = _instantiate_explicit_optimizer(
                    optimizer_class, optimizer_cfg, param_dicts, tc.optimizer_kwargs
                )

        # ``trainer.estimated_stepping_batches`` is reported in *microbatch* units when
        # the keypoint path runs with ``Trainer(accumulate_grad_batches=1)`` and manages
        # accumulation manually. ``LambdaLR.step()`` is called once per optimizer-step
        # (i.e. every ``grad_accum_steps`` microbatches), so the schedule must be sized
        # in optimizer-step units rather than microbatches; otherwise warmup and cosine
        # decay finish ``grad_accum_steps``× too early. Detection / segmentation models
        # still rely on Lightning's automatic optimization, where PTL already accounts
        # for ``accumulate_grad_batches`` inside ``estimated_stepping_batches`` and the
        # division below is a no-op (``grad_accum_steps`` would be 1 in that path).
        grad_accum_steps = max(1, int(tc.grad_accum_steps))
        microbatches = int(self.trainer.estimated_stepping_batches)
        # _should_step_optimizer steps the final partial window at epoch end, so the true
        # number of optimizer steps is ceil(microbatches / grad_accum_steps).  Using floor
        # would undercount when the epoch is not evenly divisible, causing warmup / cosine
        # schedules to finish one step earlier than the last actual step fires.
        total_steps = (
            max(1, math.ceil(microbatches / grad_accum_steps)) if self._use_manual_optimization else microbatches
        )
        steps_per_epoch = max(1, total_steps // tc.epochs)
        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        scheduler_cfg = tc.lr_scheduler
        scheduler: LRScheduler | ReduceLROnPlateau
        interval = "step"
        monitor: str | None = None
        if _is_managed_scheduler_name(scheduler_cfg):
            # Managed "step" / "cosine" preset — warmup + total-step sizing baked into a LambdaLR (unchanged behavior).
            scheduler = _build_managed_scheduler(optimizer, tc, total_steps, steps_per_epoch, warmup_steps)
        else:
            if not isinstance(scheduler_cfg, str):
                # Explicit callable / functools.partial: built from the optimizer only (kwargs baked in).
                scheduler_name = getattr(scheduler_cfg, "__qualname__", None) or repr(scheduler_cfg)
                scheduler = _instantiate_explicit_scheduler(scheduler_cfg, scheduler_name, optimizer, {})
            else:
                # Explicit dotted import path: constructed from lr_scheduler_kwargs only.
                scheduler_class = _import_scheduler_class(scheduler_cfg)
                scheduler = _instantiate_explicit_scheduler(
                    scheduler_class, scheduler_cfg, optimizer, tc.lr_scheduler_kwargs
                )
            interval = tc.lr_scheduler_interval
            if isinstance(scheduler, ReduceLROnPlateau):
                monitor = tc.lr_scheduler_monitor
                # The monitored metric (e.g. val/loss) is only available per epoch, so plateau always steps
                # on the epoch boundary regardless of the configured interval.
                interval = "epoch"
                if warmup_steps > 0:
                    logger.warning(
                        "warmup_epochs=%s is ignored for ReduceLROnPlateau; a metric-driven scheduler cannot "
                        "be composed with a linear warmup ramp.",
                        tc.warmup_epochs,
                    )
            else:
                # Auto-wrap explicit schedulers with a linear warmup ramp. The wrap is stepped at the same cadence as
                # the scheduler, so size it in the scheduler's own units: optimizer steps for "step", epochs for "epoch"
                # (otherwise a step-sized ramp stepped once per epoch would stretch across the whole run).
                if interval == "step":
                    warmup_units = warmup_steps
                else:
                    # Epoch cadence: the ramp is stepped once per epoch. ceil keeps a fractional warmup_epochs from
                    # truncating to zero (a silently dropped warmup). A single-epoch ramp has start_factor == 1.0,
                    # i.e. a flat no-op that looks like warmup but isn't, so a gradual epoch-granular warmup needs
                    # >= 2 epochs; warn and skip rather than emit a degenerate ramp.
                    warmup_units = math.ceil(tc.warmup_epochs)
                    if tc.warmup_epochs > 0 and warmup_units < 2:
                        logger.warning(
                            "warmup_epochs=%s with lr_scheduler_interval='epoch' cannot form a gradual warmup ramp "
                            "(epoch-granular warmup needs >= 2 epochs); skipping warmup. Use "
                            "lr_scheduler_interval='step' for sub-epoch warmup, or set warmup_epochs >= 2.",
                            tc.warmup_epochs,
                        )
                        warmup_units = 0
                if warmup_units > 0:
                    scheduler = _wrap_with_warmup(scheduler, optimizer, warmup_units)

        self._lr_scheduler_interval = interval
        self._lr_scheduler_monitor = monitor

        lr_scheduler_config: LRSchedulerConfigType = {"scheduler": scheduler, "interval": interval}
        if monitor is not None:
            lr_scheduler_config["monitor"] = monitor
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Override PTL gradient clipping to support fused AdamW.

        PTL's AMP precision plugin refuses to clip gradients when the optimizer declares it handles unscaling internally
        (fused=True).  When fused is active we are on BF16 (no GradScaler) so ``clip_grad_norm_`` is correct.  For the
        non-fused path (FP16 + GradScaler or FP32) we delegate to ``super()`` to preserve scaler-aware unscaling.

        Args:
            optimizer: The current optimizer.
            gradient_clip_val: Maximum gradient norm.
            gradient_clip_algorithm: Clipping algorithm; forwarded to super()
                for the non-fused path.
        """
        if self._use_fused_optimizer:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        else:
            # PTL's own type stub only declares Optimizer here, but LightningOptimizer dynamically
            # multiply-inherits from the wrapped optimizer's class, so it satisfies this at runtime too.
            super().clip_gradients(
                optimizer,  # type: ignore[arg-type]
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    @staticmethod
    def _normalize_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
        """Cast restored floating-point optimizer state tensors to match live parameters.

        Args:
            optimizer: AdamW optimizer whose state may have been rehydrated with a mismatched dtype or layout.

        Returns:
            Number of state tensors that were reallocated to match the current parameter layout.
        """
        normalized = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                for key, value in list(state.items()):
                    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                        continue
                    if value.shape != param.shape:
                        continue
                    if value.device == param.device and value.dtype == param.dtype and value.stride() == param.stride():
                        continue
                    restored = torch.empty_like(param)
                    restored.copy_(value.to(device=param.device, dtype=param.dtype))
                    state[key] = restored
                    normalized += 1
        return normalized

    def test_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one test step.

        Mirrors :meth:`validation_step` so ``COCOEvalCallback`` can accumulate results via ``on_test_batch_end`` when
        ``trainer.test()`` is called (e.g. from :class:`~rfdetr.training.callbacks.BestModelCallback` at end of
        training).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the test epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: tuple[Any, Any], batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference on a preprocessed batch and return postprocessed results.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index.
            dataloader_idx: Index of the predict dataloader.

        Returns:
            Postprocessed detection results from ``PostProcess``.
        """
        samples, targets = batch
        with torch.no_grad():
            outputs = self.model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        return self.postprocess(outputs, orig_sizes)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Auto-detect legacy formats and reconcile PE shapes at checkpoint load time.

        PTL calls this hook before applying ``checkpoint["state_dict"]`` to the module.  Three normalisation steps are
        applied in order:

        1. **Raw legacy format** — a ``*.pth`` file loaded directly by
           ``Trainer`` (e.g. via ``ckpt_path=``).  Recognised by the presence of ``"model"`` without ``"state_dict"``.
           The state dict is rewritten in-place with the ``"model."`` prefix so PTL can apply it normally.

        2. **Positional-embedding interpolation** — when the checkpoint was
           saved at a different image resolution than the current model, the DINOv2 ``position_embeddings`` tensor shape
           will mismatch. :func:`~rfdetr.models.weights.interpolate_position_embeddings` is called to bicubic-resize the
           PE to ``model_config.positional_encoding_size`` before PTL applies the state dict.  Regression fix for
           :issue:`998`.

        3. **Converted format** — a file produced by
           :func:`~rfdetr.training.checkpoint.convert_legacy_checkpoint` that already has ``"state_dict"`` but also
           carries ``"legacy_ema_state_dict"``.  The EMA weights are stashed on ``self._pending_legacy_ema_state`` for
           optional restoration by :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`.

        Note:
            This hook only fires on ``Trainer(ckpt_path=...)`` resume paths. Fresh-train bootstrap from a
            ``pretrain_weights`` checkpoint runs through :func:`~rfdetr.models.weights.load_pretrain_weights` during
            ``__init__`` instead — that helper performs its own PTL ``.ckpt`` normalisation (``state_dict`` → ``model``
            key, ``_orig_mod`` strip) and PE interpolation, so the two code paths intentionally do not share state.

        Args:
            checkpoint: Checkpoint dict passed in by PTL (mutated in-place).
        """
        # Raw legacy .pth: no "state_dict" key — build it from "model".
        if "model" in checkpoint and "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["model"].items()}

        # Interpolate DINOv2 positional embeddings when the checkpoint was saved
        # at a different resolution than the current model.  PTL applies
        # checkpoint["state_dict"] immediately after this hook, so the shapes
        # must already match at this point.  Regression: #998.
        if "state_dict" in checkpoint:
            interpolate_position_embeddings(
                checkpoint["state_dict"],
                self.model_config.positional_encoding_size,
            )

        # Stash legacy EMA weights for RFDETREMACallback.setup(), which restores
        # them into AveragedModel when resuming from converted legacy checkpoints.
        if "legacy_ema_state_dict" in checkpoint:
            self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
            warnings.warn(
                "Checkpoint contains legacy EMA weights (`legacy_ema_state_dict`). "
                "Add RFDETREMACallback to your trainer callbacks to restore them; "
                "without it the stashed weights will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new class count.

        Args:
            num_classes: New number of classes (excluding background).
        """
        self.model.reinitialize_detection_head(num_classes)
