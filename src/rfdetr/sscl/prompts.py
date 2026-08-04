# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLIP 类别文本提示词加载模块。

各类别的 CLIP 文本提示词统一存放在 ``sscl/prompts/`` 目录下的 YAML 文件中，
本模块负责加载与校验，供离线构建 CLIP 语义相似度矩阵使用（见 semantic_matrix.py）。

每个 YAML 文件包含两部分：
- ``class_names``: ``class_id -> 类别名称`` 映射，用于日志与矩阵验证输出。
- ``class_prompts``: ``class_id -> [prompt_1, prompt_2, ...]`` 映射，用于编码。

提示词编写原则（各数据集通用）：
- 使用完整名称而非缩写（CLIP 对缩写理解不稳定）。
- 描述可见外观特征，并统一使用遥感/俯视视角，与检测任务视角一致。
- 同一类别提供多个模板，编码时取平均以增强 CLIP 对类别语义的稳定理解。

类别索引与各数据集标注中的 ``class_id`` 保持一致：
- SHWX: 0-24（见 shwx.yaml）。
- DIOR: 1-20（见 dior.yaml，与 COCO 标注中的 category id 一致）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _coerce_int_keys(data: dict[Any, Any]) -> dict[int, Any]:
    """将 YAML 解析出的 dict 键统一转为 int，兼容字符串与数字键。

    Args:
        data: YAML 解析出的原始 dict（键可能为 int 或 str）。

    Returns:
        键均为 int 的新 dict。
    """
    return {int(key): value for key, value in data.items()}


def load_class_prompts(dataset: str) -> tuple[dict[int, str], dict[int, list[str]]]:
    """从 YAML 文件加载指定数据集的类别名称与提示词。

    Args:
        dataset: 数据集名称，对应 ``sscl/prompts/<dataset>.yaml`` 文件。

    Returns:
        ``(class_names, class_prompts)`` 二元组：
        - class_names: ``{class_id: 类别名称}`` 映射。
        - class_prompts: ``{class_id: [prompt_1, prompt_2, ...]}`` 映射。

    Raises:
        FileNotFoundError: 当对应的 YAML 文件不存在时抛出。
        ValueError: 当 YAML 根节点不是映射、缺少必要字段、
            两个字段类别索引不一致或存在无提示词的类别时抛出。
    """
    yaml_path = _PROMPTS_DIR / f"{dataset}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"提示词 YAML 文件不存在: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"提示词 YAML 根节点必须是映射: {yaml_path}")
    if "class_names" not in data or "class_prompts" not in data:
        raise ValueError(f"提示词 YAML 必须包含 class_names 与 class_prompts 字段: {yaml_path}")

    class_names = _coerce_int_keys(data["class_names"])
    class_prompts = _coerce_int_keys(data["class_prompts"])

    # 两个字段的类别索引必须完全一致
    if set(class_names.keys()) != set(class_prompts.keys()):
        diff = sorted(set(class_names.keys()) ^ set(class_prompts.keys()))
        raise ValueError(f"类别索引不一致，差异项: {diff}")

    # 每个类别至少需要 1 条提示词
    empty = [cid for cid, prompts in class_prompts.items() if not prompts]
    if empty:
        raise ValueError(f"以下类别没有提示词: {empty}")

    return class_names, class_prompts


# 模块加载时从对应 YAML 读取各数据集提示词，保持向后兼容
SHWX_CLASS_NAMES, SHWX_CLASS_PROMPTS = load_class_prompts("shwx")
DIOR_CLASS_NAMES, DIOR_CLASS_PROMPTS = load_class_prompts("dior")

__all__ = [
    "SHWX_CLASS_NAMES",
    "SHWX_CLASS_PROMPTS",
    "DIOR_CLASS_NAMES",
    "DIOR_CLASS_PROMPTS",
    "load_class_prompts",
]
