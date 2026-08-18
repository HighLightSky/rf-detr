# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""推理配置中的 FFT 一致性插件开关测试。"""

from __future__ import annotations

import pytest

from scripts.predict import _build_reason_plugin_kwargs


def test_reason_plugin_is_disabled_by_default() -> None:
    """未配置插件或明确关闭时不向模型传插件参数。"""
    assert _build_reason_plugin_kwargs({}) == {}
    assert _build_reason_plugin_kwargs({"reason_plugin": {"enabled": False}}) == {}


def test_reason_plugin_config_is_forwarded_when_enabled() -> None:
    """开启插件时正确转换类别和候选框阈值参数。"""
    kwargs = _build_reason_plugin_kwargs(
        {
            "reason_plugin": {
                "enabled": True,
                "checkpoint": "/tmp/reason_plugin.pth",
                "class_ids": [24, "3"],
                "conf_low": "0.1",
            }
        }
    )
    assert kwargs == {
        "reason_plugin": "/tmp/reason_plugin.pth",
        "reason_class_ids": (24, 3),
        "reason_conf_low": 0.1,
    }


def test_reason_plugin_requires_checkpoint_when_enabled() -> None:
    """开启插件但未提供 checkpoint 时给出配置错误。"""
    with pytest.raises(ValueError, match="checkpoint"):
        _build_reason_plugin_kwargs({"reason_plugin": {"enabled": True}})


def test_reason_plugin_accepts_all_class_and_default_threshold() -> None:
    """null 类别和阈值配置透传为插件的默认行为。"""
    kwargs = _build_reason_plugin_kwargs(
        {
            "reason_plugin": {
                "enabled": True,
                "checkpoint": "plugin.pth",
                "class_ids": None,
                "conf_low": None,
            }
        }
    )
    assert kwargs["reason_class_ids"] is None
    assert kwargs["reason_conf_low"] is None
