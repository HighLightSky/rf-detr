# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""跨原型空间几何诊断测试。"""

from __future__ import annotations

import pytest
import torch

from rfdetr.sscl.prototype_diagnostics import prototype_geometry, prototype_relation_alignment


def test_geometry_detects_collapsed_prototypes() -> None:
    """完全塌缩原型应具有类间余弦 1 和有效秩 1。"""
    prototypes = torch.ones(4, 2, 8)
    valid = torch.ones(4, 2, dtype=torch.bool)

    stats = prototype_geometry(prototypes, valid)

    assert float(stats["offdiag_cos_mean"]) == pytest.approx(1.0)
    assert float(stats["effective_rank"]) == pytest.approx(1.0)


def test_relation_alignment_accepts_different_feature_dimensions() -> None:
    """两个维度不同但类关系相同的空间应得到接近 1 的关系对齐度。"""
    first = torch.eye(4)
    second = torch.cat([torch.eye(4), torch.zeros(4, 3)], dim=-1)

    alignment = prototype_relation_alignment(first, second)

    assert float(alignment) == pytest.approx(1.0, abs=1e-6)


def test_relation_alignment_rejects_class_count_mismatch() -> None:
    """类别数不同的两个原型空间不能比较。"""
    with pytest.raises(ValueError, match="类别数"):
        prototype_relation_alignment(torch.randn(3, 4), torch.randn(4, 5))
