# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""实验配置加载与路径解析（train/test/predict 三模板共享）。

yaml 结构为自定义三段式：

.. code-block:: yaml

    _template: {class_counts: auto}   # 模板专用键（train.py 消费）
    model:     {variant: medium, num_classes: 25, ...}  # 模型构造 kwargs
    train:     {dataset_dir: ..., ...}  # 100% 透传为 model.train(**kwargs)
    test:      {dataset: shwx, checkpoint: ..., ...}    # test.py 消费
    predict:   {checkpoint: ..., image: ..., ...}       # predict.py 消费

规则：

- ``train:`` 段键名原封不动作为 ``model.train(**kwargs)`` 的参数（零映射表），
  ``TrainConfig`` 的 ``extra="forbid"`` 天然校验拼写错误；
- 相对路径一律以项目根解析，绝对路径原样（与旧脚本 ``project_root / X`` 一致）；
- ``--set train.sscl_lambda=0.2`` 支持点路径标量覆盖（yaml < --set < 专用 CLI 参数）；
- ``aug_config`` 只接受预设名（AERIAL/CONSERVATIVE/AGGRESSIVE/INDUSTRIAL/none），
  ``AUG_AERIAL`` 等是嵌套 dict 无法直接写在 yaml 里，由本模块映射。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.datasets.aug_configs import (  # noqa: E402
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)
from rfdetr.variants import (  # noqa: E402
    RFDETRLarge,
    RFDETRLargeDeprecated,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
)

# 模型注册表：{名称: (变体类, 默认分辨率)}。
# 注意 large 有两代：RFDETRLarge（新版，train_LoRA 用）与 RFDETRLargeDeprecated
# （旧版，train.py 曾用），默认分辨率取各自 config 默认值。
MODEL_REGISTRY: dict[str, tuple[type, int]] = {
    "nano": (RFDETRNano, 384),
    "small": (RFDETRSmall, 512),
    "medium": (RFDETRMedium, 576),
    "large": (RFDETRLarge, 704),
    "large_deprecated": (RFDETRLargeDeprecated, 560),
}

# 增广预设名 → 实际配置 dict（rfdetr.datasets.aug_configs 的预设）
AUG_PRESETS: dict[str, dict] = {
    "AERIAL": AUG_AERIAL,
    "CONSERVATIVE": AUG_CONSERVATIVE,
    "AGGRESSIVE": AUG_AGGRESSIVE,
    "INDUSTRIAL": AUG_INDUSTRIAL,
    "none": {},
}


class ConfigError(ValueError):
    """实验配置文件错误（结构缺失 / 预设名未知 / 覆盖路径无效）。"""


def load_config(path: str | Path) -> dict[str, Any]:
    """加载并校验实验 yaml 配置。

    Args:
        path: yaml 文件路径。

    Returns:
        解析后的配置字典（顶层含 ``_template``/``model``/``train``/``test``/``predict``）。

    Raises:
        FileNotFoundError: yaml 文件不存在。
        ConfigError: 顶层结构不是字典。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"配置文件顶层必须是字典: {path}")
    return cfg


def resolve_paths(root: Path, value: Any) -> Any:
    """递归解析相对路径：字符串相对路径（不以 ``/`` 开头）→ ``root / path``。

    Args:
        root: 相对路径解析基准（项目根）。
        value: 任意值（str / 容器 / 其他类型，原样返回）。

    Returns:
        解析后的值。
    """
    if isinstance(value, str):
        if value.startswith("/") or not value:  # 绝对路径或空串原样保留
            return value
        return str(root / value)
    if isinstance(value, dict):
        return {k: resolve_paths(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(root, v) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve_paths(root, v) for v in value)
    return value


def _parse_override_value(raw: str) -> Any:
    """解析 ``--set`` 覆盖值（bool/float/int/null，其余按字符串）。

    Args:
        raw: 命令行传入的原始字符串。

    Returns:
        解析后的值。
    """
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    if lowered == "inf":
        return float("inf")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """按点路径覆盖配置：``train.sscl_lambda=0.2`` → ``cfg["train"]["sscl_lambda"]``。

    Args:
        cfg: 加载后的配置字典（原地修改）。
        overrides: ``--set`` 参数列表，每项形如 ``段名.键名.子键名=值``。

    Raises:
        ConfigError: 覆盖路径无效（段名或键不存在）。
    """
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set 参数须形如 train.sscl_lambda=0.2，得到: {item}")
        path, raw_value = item.split("=", 1)
        keys = path.split(".")
        if not keys or keys[0] not in cfg:
            raise ConfigError(f"--set 覆盖的段不存在: {keys[0]}（可选段: {list(cfg.keys())}）")
        node = cfg
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"--set 覆盖的路径不存在: {path}")
            node = node[key]
        node[keys[-1]] = _parse_override_value(raw_value)
    return cfg


def build_model_kwargs(model_section: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """从 ``model:`` 段构建模型构造 kwargs，弹出模板专用键 ``variant``。

    Args:
        model_section: yaml 的 ``model:`` 段（可为 ``None``/空，默认 medium）。

    Returns:
        ``(model_kwargs, variant_name)``：``model_kwargs`` 为变体类构造参数
        （不含 ``variant``），``variant_name`` 用于查 ``MODEL_REGISTRY``。
    """
    section = dict(model_section or {})
    variant = section.pop("variant", "medium")
    if variant not in MODEL_REGISTRY:
        raise ConfigError(
            f"未知模型变体: {variant}（可选: {', '.join(MODEL_REGISTRY)}）"
        )
    return section, variant


def build_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 ``train:`` 段构建 ``model.train(**kwargs)`` 参数。

    处理：相对路径解析（以项目根为基准）+ ``aug_config`` 预设名映射。

    Args:
        cfg: 加载后的配置字典。

    Returns:
        直接传给 ``model.train`` 的 kwargs 字典。

    Raises:
        ConfigError: 缺 ``train:`` 段 / ``aug_config`` 预设名未知。
    """
    train_section = cfg.get("train")
    if train_section is None:
        raise ConfigError("配置文件缺少 train: 段（训练模板必须提供）")
    train_kwargs = resolve_paths(PROJECT_ROOT, dict(train_section))

    # aug_config 只接受预设名（AUG_* 是嵌套 dict 无法写在 yaml 里）
    aug_name = train_kwargs.get("aug_config")
    if aug_name is not None and aug_name not in AUG_PRESETS:
        raise ConfigError(
            f"aug_config 只支持预设名: {', '.join(AUG_PRESETS)}（得到: {aug_name}）"
        )
    if aug_name is not None:
        train_kwargs["aug_config"] = AUG_PRESETS[aug_name]
    return train_kwargs


def build_test_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 ``test:`` 段构建 test.py 模板使用的参数（相对路径已解析）。

    Args:
        cfg: 加载后的配置字典。

    Returns:
        ``test:`` 段解析后的字典。

    Raises:
        ConfigError: 缺 ``test:`` 段。
    """
    test_section = cfg.get("test")
    if test_section is None:
        raise ConfigError("配置文件缺少 test: 段（测试模板必须提供）")
    return resolve_paths(PROJECT_ROOT, dict(test_section))


def build_predict_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 ``predict:`` 段构建 predict.py 模板使用的参数（相对路径已解析）。

    Args:
        cfg: 加载后的配置字典。

    Returns:
        ``predict:`` 段解析后的字典。

    Raises:
        ConfigError: 缺 ``predict:`` 段。
    """
    predict_section = cfg.get("predict")
    if predict_section is None:
        raise ConfigError("配置文件缺少 predict: 段（推理模板必须提供）")
    return resolve_paths(PROJECT_ROOT, dict(predict_section))
