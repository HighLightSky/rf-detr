# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多模态原型离线产物：保存、加载与校验。

离线脚本 ``stage0_build_proto_guidance.py`` 构建两类原型产物：
- 视觉原型 ``P_v [C, M, d]``（backbone/projector 特征空间按 GT box 提取，
  每类 M 个槽位子原型，槽位无效掩码 ``valid_slots [C, M]``）；
- 文本原型 ``P_t_clip [C, text_dim]``（CLIP 多提示词平均，未投影）。

训练时 ``ProtoGuidance.build`` 加载产物并 ``copy_`` 进 buffer。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def save_proto_artifacts(
    path: str | Path,
    visual_prototypes: Tensor,
    valid_slots: Tensor,
    text_prototypes: Tensor,
    class_names: list[str],
    meta: dict[str, Any],
) -> None:
    """保存多模态原型离线产物。

    Args:
        path: 输出文件路径（``.pt`` 后缀）。
        visual_prototypes: 视觉原型 ``[C, M, d]``。
        valid_slots: 槽位有效掩码 ``[C, M]``（bool）。
        text_prototypes: CLIP 文本原型 ``[C, text_dim]``。
        class_names: 类别名称列表（长度 = C）。
        meta: 元信息（dataset、num_classes、hidden_dim、checkpoint 等）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "visual_prototypes": visual_prototypes.detach().cpu(),
            "valid_slots": valid_slots.detach().cpu(),
            "text_prototypes": text_prototypes.detach().cpu(),
            "class_names": list(class_names),
            "meta": dict(meta),
        },
        path,
    )
    logger.info(
        f"多模态原型产物已保存到: {path}（视觉原型形状: {tuple(visual_prototypes.shape)}）"
    )


def load_proto_artifacts(path: str | Path) -> dict[str, Any]:
    """加载多模态原型离线产物。

    Args:
        path: ``save_proto_artifacts`` 保存的文件路径。

    Returns:
        ``{"visual_prototypes": Tensor, "valid_slots": Tensor,
        "text_prototypes": Tensor, "class_names": list[str], "meta": dict}``。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        KeyError: 当文件中缺少必需键时抛出。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"多模态原型产物文件不存在: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = ("visual_prototypes", "valid_slots", "text_prototypes", "class_names", "meta")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"文件 {path} 缺少必需键: {missing}")
    return data


def validate_proto_artifacts(
    data: dict[str, Any],
    *,
    num_classes: int | None = None,
    hidden_dim: int | None = None,
    text_dim: int | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """校验产物形状与元信息，返回规范化后的产物字典。

    Args:
        data: ``load_proto_artifacts`` 返回的字典。
        num_classes: 期望类别数（不匹配时抛错）。
        hidden_dim: 期望特征维度（不匹配时抛错）。
        text_dim: 期望文本维度（不匹配时抛错）。
        dataset: 期望数据集名（元信息里可选的 dataset 键，不匹配时仅警告）。

    Returns:
        规范化字典（含 ``visual_prototypes``/``valid_slots``/``text_prototypes``/
        ``class_names``/``meta``）。

    Raises:
        ValueError: 形状或类别数不匹配时抛出。
    """
    visual_prototypes = data["visual_prototypes"]
    valid_slots = data["valid_slots"]
    text_prototypes = data["text_prototypes"]
    class_names = data["class_names"]

    if visual_prototypes.ndim != 3:
        raise ValueError(f"视觉原型必须是 3 维 [C, M, d]，收到 {tuple(visual_prototypes.shape)}")
    if valid_slots.shape != visual_prototypes.shape[:2]:
        raise ValueError(
            f"槽位掩码形状 {tuple(valid_slots.shape)} 与视觉原型前两维 {tuple(visual_prototypes.shape[:2])} 不一致"
        )
    if not torch.isfinite(visual_prototypes).all():
        raise ValueError("视觉原型含 NaN/Inf。")
    if num_classes is not None and visual_prototypes.shape[0] != num_classes:
        raise ValueError(
            f"类别数不一致: 产物 {visual_prototypes.shape[0]}，期望 {num_classes}"
        )
    if hidden_dim is not None and visual_prototypes.shape[2] != hidden_dim:
        raise ValueError(
            f"特征维度不一致: 产物 {visual_prototypes.shape[2]}，期望 {hidden_dim}"
        )
    if text_dim is not None and text_prototypes.shape[1] != text_dim:
        raise ValueError(
            f"文本维度不一致: 产物 {text_prototypes.shape[1]}，期望 {text_dim}"
        )
    if len(class_names) != visual_prototypes.shape[0]:
        raise ValueError(
            f"类别名称数 {len(class_names)} 与视觉原型类别数 {visual_prototypes.shape[0]} 不一致"
        )

    meta = data.get("meta", {})
    if dataset is not None and meta.get("dataset") not in (None, dataset):
        logger.warning(
            f"产物数据集 {meta.get('dataset')!r} 与期望 {dataset!r} 不一致，"
            "请确认原型与训练数据对齐。"
        )
    return data
