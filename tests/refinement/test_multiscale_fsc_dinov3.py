"""DINOv3 多尺度 FSC 分类头测试。"""

from __future__ import annotations

import pytest
import torch

from rfdetr.refinement.two_stage_plugin import _context_scales_from_metadata
from scripts.refinement.train_fsc_dinov3_head import FSCDinoV3Head, normalize_context_scales


def test_normalize_context_scales_keeps_ordered_unique_positive_values() -> None:
    """训练侧保留尺度顺序并拒绝重复与非法值。"""
    assert normalize_context_scales([1.5, 2.0, 3.0]) == (1.5, 2.0, 3.0)
    with pytest.raises(ValueError, match="重复"):
        normalize_context_scales([1.5, 1.5])
    with pytest.raises(ValueError, match="正数"):
        normalize_context_scales([0.0])


def test_context_scales_from_metadata_prefers_checkpoint_multiscale_settings() -> None:
    """推理侧优先使用 checkpoint 固化的多尺度配置。"""
    assert _context_scales_from_metadata({"context_scales": [1.5, 2.0, 3.0]}, 2.0) == (1.5, 2.0, 3.0)


def test_context_scales_from_metadata_supports_legacy_single_scale_head() -> None:
    """旧单尺度 checkpoint 仍回退到原有 context_scale。"""
    assert _context_scales_from_metadata({"context_scale": 2.5}, 2.0) == (2.5,)


def test_multiscale_head_accepts_concatenated_features() -> None:
    """三尺度拼接后的特征维度可被分类头正常处理。"""
    head = FSCDinoV3Head(feature_dim=12)
    assert head(torch.zeros((3, 12))).shape == (3, 2)
    with pytest.raises(ValueError, match="features"):
        head(torch.zeros((3, 4)))
