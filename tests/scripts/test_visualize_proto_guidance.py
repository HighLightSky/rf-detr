"""多模态原型热力图纯函数测试。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.visualize_proto_guidance import (
    colorize_overlay,
    enhance_margin_contrast,
    fuse_level_maps,
    normalize_margin,
    normalize_positive,
    split_flattened_scores,
    target_class_margin,
)


def test_split_flattened_scores_restores_multiple_levels() -> None:
    """展平 token 必须按每层起始索引恢复为二维网格。"""
    values = np.arange(4 + 6, dtype=np.float32)
    levels = split_flattened_scores(values, [(2, 2), (2, 3)], [0, 4])
    np.testing.assert_array_equal(levels[0], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(levels[1], [[4, 5, 6], [7, 8, 9]])


def test_split_flattened_scores_rejects_bad_start_index() -> None:
    """错误的 level_start_index 必须被显式拒绝。"""
    with pytest.raises(ValueError, match="level_start_index"):
        split_flattened_scores(np.arange(4, dtype=np.float32), [(2, 2)], [1])


def test_target_class_margin_excludes_target_class() -> None:
    """margin 的竞争最大值不能再次包含目标类别。"""
    logits = np.asarray([[5.0, 2.0, 4.0], [1.0, 3.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(target_class_margin(logits, 0), [1.0, -2.0])


def test_normalization_handles_constant_and_nonfinite_values() -> None:
    """常数图与非有限值不能产生 NaN/Inf。"""
    constant = normalize_positive(np.ones((2, 2), dtype=np.float32))
    mixed = normalize_margin(np.asarray([[np.nan, 1.0], [np.inf, -1.0]], dtype=np.float32))
    assert np.array_equal(constant, np.zeros((2, 2), dtype=np.float32))
    assert np.isfinite(mixed).all()


def test_margin_normalization_stretches_negative_only_maps() -> None:
    """全负 margin 也必须覆盖明显的蓝色动态范围。"""
    values = np.asarray([[-0.30, -0.20], [-0.10, -0.01]], dtype=np.float32)
    normalized = normalize_margin(values)
    assert normalized.min() < -0.9
    assert normalized.max() > -0.1
    assert np.all(normalized <= 0.0)


def test_margin_contrast_preserves_sign_and_increases_midrange() -> None:
    """对比度增强不能改变 margin 符号，并应拉开中间值。"""
    values = np.asarray([-0.25, 0.0, 0.25], dtype=np.float32)
    enhanced = enhance_margin_contrast(values)
    assert enhanced[0] < -0.25
    assert enhanced[1] == 0.0
    assert enhanced[2] > 0.25


def test_fuse_level_maps_and_overlay_preserve_output_size() -> None:
    """多层融合和叠加结果必须保持目标图像尺寸。"""
    fused = fuse_level_maps([np.zeros((2, 2)), np.ones((1, 1))], (8, 6))
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    overlay = colorize_overlay(image, fused, alpha=0.45)
    signed_overlay = colorize_overlay(image, np.zeros((10, 12), dtype=np.float32), alpha=0.45, signed=True)
    assert fused.shape == (8, 6)
    assert overlay.shape == image.shape
    assert signed_overlay.shape == image.shape
