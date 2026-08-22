# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""阶段 0 原型构建脚本的数据集配置校验测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_stage0_module() -> ModuleType:
    """按文件路径加载带连字符目录下的阶段 0 脚本。"""
    script_path = Path(__file__).resolve().parents[2] / "src/scripts/semantic_experiments/stage0_build_proto_guidance.py"
    spec = importlib.util.spec_from_file_location("stage0_build_proto_guidance_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage0_rejects_dataset_class_count_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段 0 应在 CUDA 前拒绝类别数不匹配的数据集目录。"""
    (tmp_path / "data.yaml").write_text("nc: 25\n", encoding="utf-8")
    module = _load_stage0_module()
    monkeypatch.setattr(module, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(module, "NUM_CLASSES", 26)

    with pytest.raises(ValueError, match="类别数与原型配置不一致"):
        module._validate_dataset_num_classes()
