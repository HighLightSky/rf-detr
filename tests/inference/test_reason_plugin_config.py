# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""推理配置中的 FFT 一致性插件开关测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import eval_lib
from scripts.predict import _build_reason_plugin_kwargs, _resolve_predict_settings


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


def test_predict_settings_are_entirely_read_from_config() -> None:
    """推理所需参数只接受配置文件中的 predict 段。"""
    checkpoint, conf, output_dir, image = _resolve_predict_settings(
        {
            "checkpoint": "/tmp/model.pth",
            "conf": 0.3,
            "output_dir": "/tmp/predictions",
            "image": "/tmp/image.jpg",
        }
    )

    assert checkpoint == "/tmp/model.pth"
    assert conf == 0.3
    assert output_dir == Path("/tmp/predictions")
    assert image == "/tmp/image.jpg"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            {"conf": 0.25, "output_dir": "out", "image": "image.jpg"},
            "checkpoint",
            id="missing-checkpoint",
        ),
        pytest.param(
            {"checkpoint": "model.pth", "conf": 0.25, "image": "image.jpg"},
            "output_dir",
            id="missing-output-dir",
        ),
        pytest.param(
            {"checkpoint": "model.pth", "conf": 0.25, "output_dir": "out"},
            "image",
            id="missing-image",
        ),
        pytest.param(
            {
                "checkpoint": "model.pth",
                "conf": 1.1,
                "output_dir": "out",
                "image": "image.jpg",
            },
            "predict.conf",
            id="invalid-conf",
        ),
    ],
)
def test_predict_settings_require_complete_valid_config(
    config: dict[str, object], message: str
) -> None:
    """缺失或非法预测配置在执行前报出明确错误。"""
    with pytest.raises(ValueError, match=message):
        _resolve_predict_settings(config)


def test_label_comparison_config_requires_label_directory_when_enabled() -> None:
    """标签对比模式开启时必须配置 YOLO 标签目录。"""
    config = eval_lib.LabelComparisonCfg.from_config(
        {"enabled": True, "labels_dir": "/tmp/labels", "iou_threshold": 0.4}
    )

    assert config == eval_lib.LabelComparisonCfg(Path("/tmp/labels"), 0.4)

    with pytest.raises(ValueError, match="labels_dir"):
        eval_lib.LabelComparisonCfg.from_config({"enabled": True})
