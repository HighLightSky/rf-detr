# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Functions to get params dict."""

from typing import Any, cast

from torch import nn

from rfdetr.models.backbone import Joiner
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: parameter name.
        lr_decay_rate: base lr decay rate.
        num_layers: number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    # NOTE: near-duplicate of get_dinov2_lr_decay_rate in models/backbone/backbone.py (same formula,
    # different layer-key pattern: this matches ".blocks.", that matches ".layer.").
    # If updating this formula, update the sibling too.
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    logger.debug(f"name: {name}, lr_decay: {lr_decay_rate ** (num_layers + 1 - layer_id)}")
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Calculate weight decay rate for different ViT parameters.

    Args:
        name: parameter name.
        weight_decay_rate: base weight decay rate.

    Returns:
        weight decay rate for the given parameter.
    """
    if ("gamma" in name) or ("pos_embed" in name) or ("rel_pos" in name) or ("bias" in name) or ("norm" in name):
        weight_decay_rate = 0.0
    logger.debug(f"name: {name}, weight_decay rate: {weight_decay_rate}")
    return weight_decay_rate


def get_param_dict(args: Any, model_without_ddp: nn.Module) -> list[dict[str, Any]]:
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = cast("Any", model_without_ddp.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [p for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad]

    decoder_param_lr_pairs = [{"params": param, "lr": args.lr * args.lr_component_decay} for param in decoder_params]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters()
        if (n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [{"params": param, "lr": args.lr} for param in other_params]

    final_param_dicts = other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs

    return final_param_dicts


def get_projection_head_param_dict(sscl_loss: nn.Module | None, lr: float) -> list[dict[str, Any]]:
    """收集 SSCL 投影头可训练参数为独立参数组。

    投影头挂在 ``sscl_loss`` 子模块上而非 model（LWDETR）内部，
    ``get_param_dict`` 只扫描 model 的 ``named_parameters()``，因此这里单独
    收集。未启用投影头、无投影头子模块或参数已冻结时返回空列表。

    Args:
        sscl_loss: SSCL 损失模块（可选含 ``projection_head`` 子模块），
            可为 ``None``（未启用 SSCL 时）。
        lr: 投影头参数组的学习率。

    Returns:
        投影头参数组列表（``[{"params": [...], "lr": lr}]``）；无可训练投影头
        参数时为空列表。
    """
    if sscl_loss is None:
        return []
    head = getattr(sscl_loss, "projection_head", None)
    if head is None:
        return []
    params = [p for p in head.parameters() if p.requires_grad]
    if not params:
        return []
    return [{"params": params, "lr": lr}]


def get_semantic_head_param_dict(semantic_residual: nn.Module | None, lr: float) -> list[dict[str, Any]]:
    """收集语义分类头可学习参数（α/θ）为独立参数组。

    SemanticResidual 挂在 LWDETR 内部（``model.semantic_residual``），
    ``get_param_dict`` 会把其参数收进 ``other_params``（lr=args.lr 主组）；
    调用方须先按参数 id 从 ``get_param_dict`` 结果中过滤，再追加本组，避免
    同一参数被两个参数组重复更新。冻结的 novel 类 θ（requires_grad=False）与
    冻结 α 自动被排除。

    Args:
        semantic_residual: 语义残差模块（可为 ``None``，未启用语义头时）。
        lr: 语义参数组学习率（通常高于主组，标量参数需足够 LR 才能学出来）。

    Returns:
        语义参数组列表（``[{"params": [...], "lr": lr}]``）；无可训练参数时为
        空列表。
    """
    if semantic_residual is None:
        return []
    params = [p for p in semantic_residual.parameters() if p.requires_grad]
    if not params:
        return []
    return [{"params": params, "lr": lr}]
