# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""seam_cut 拼接缝检测模块单元测试。

覆盖：
- ``_axis_seams`` 的方向归一化（水平缝沿行投票、垂直缝沿列投票）与间距过滤；
- ``detect_seams`` 在合成拼接图/无拼接图/条纹纹理图上的检测行为；
- ``seam_tiles`` 图块组合与超限图块滑窗兜底。
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.seam_cut import (
    SEAM_MIN_SPACING,
    _axis_seams,
    _jump_profile,
    detect_seams,
    seam_tiles,
)


def _texture(shape: tuple[int, int], mean: float, rng: np.random.Generator) -> np.ndarray:
    """生成高斯噪声纹理块（亮度钳位到 uint8 范围）。

    Args:
        shape: 块 (高, 宽)。
        mean: 亮度均值（不同均值 = 不同源图的辐射差异）。
        rng: numpy 随机源。

    Returns:
        ``(H, W)`` uint8 灰度块。
    """
    noise = rng.normal(mean, 6, size=shape)
    return np.clip(noise, 0, 255).astype(np.uint8)


class TestAxisSeams:
    """_axis_seams 方向归一化与缝判定。"""

    def test_horizontal_votes_over_columns(self):
        """水平缝（axis=0）应按行投票：缝行整行跳变，consist 高。"""
        rng = np.random.default_rng(0)
        img = _texture((600, 800), 100, rng)
        # 第 300 行处贴一条 30 亮度的横带（行间跳变 ~70，稳超 mean_jump 门槛）
        img[300] = 30
        prof, valid = _jump_profile(img, axis=0)
        seams = _axis_seams(prof, valid, axis=0)
        assert any(abs(s - 300) <= 1 for s in seams), f"应检出水平缝 300，实际 {seams}"

    def test_vertical_votes_over_rows(self):
        """垂直缝（axis=1）应按列投票：缝列整列跳变，consist 高。"""
        rng = np.random.default_rng(1)
        img = _texture((600, 800), 100, rng)
        img[:, 500] = 30
        prof, valid = _jump_profile(img, axis=1)
        seams = _axis_seams(prof, valid, axis=1)
        assert any(abs(s - 500) <= 1 for s in seams), f"应检出垂直缝 500，实际 {seams}"

    def test_local_edge_not_seam(self):
        """局部内容边缘（只在部分行跳变）不应通过一致性门槛。"""
        rng = np.random.default_rng(2)
        img = _texture((600, 800), 100, rng)
        # 垂直边缘只出现在前 100 行（consist = 100/600 < 0.6）
        img[:100, 400] = 90
        prof, valid = _jump_profile(img, axis=1)
        seams = _axis_seams(prof, valid, axis=1)
        assert seams == []

    def test_min_spacing_filters_dense_cluster(self):
        """密集候选（周期性纹理）经间距过滤后只保留一条。"""
        rng = np.random.default_rng(3)
        img = _texture((600, 800), 100, rng)
        # 每 50px 一条亮列（周期纹理）→ 密集候选
        img[:, ::50] = 90
        prof, valid = _jump_profile(img, axis=1)
        seams = _axis_seams(prof, valid, axis=1)
        if len(seams) > 1:
            assert min(b - a for a, b in zip(seams[:-1], seams[1:])) >= SEAM_MIN_SPACING, (
                f"密集簇应被间距过滤，实际 {seams}"
            )

    def test_black_padding_excluded(self):
        """黑底填充区不参与统计：纯黑图像无缝。"""
        img = np.zeros((500, 700), dtype=np.uint8)
        prof, valid = _jump_profile(img, axis=1)
        assert not valid.any()
        assert _axis_seams(prof, valid, axis=1) == []


class TestDetectSeams:
    """detect_seams 端到端行为。"""

    def test_single_row_two_tiles(self):
        """单行两块拼接：检出 1 条垂直缝、无水平缝。"""
        rng = np.random.default_rng(4)
        left = _texture((600, 500), 80, rng)
        right = _texture((600, 700), 160, rng)
        img = np.concatenate([left, right], axis=1)
        seam_ys, xs_per_band = detect_seams(img)
        assert seam_ys == []
        assert len(xs_per_band) == 1
        assert any(abs(x - 500) <= 1 for x in xs_per_band[0]), f"应检出垂直缝 500，实际 {xs_per_band}"

    def test_two_rows_three_tiles(self):
        """两行拼接（行边界 + 各行独立垂直缝）：水平缝 1 条，两条带垂直缝各 1 条。"""
        rng = np.random.default_rng(5)
        row1 = np.concatenate(
            [_texture((600, 500), 80, rng), _texture((600, 600), 160, rng)], axis=1
        )
        row2 = np.concatenate(
            [_texture((700, 550), 120, rng), _texture((700, 550), 40, rng)], axis=1
        )
        img = np.concatenate([row1, row2], axis=0)  # 行边界 = 600，两行等宽 1100
        seam_ys, xs_per_band = detect_seams(img)
        assert any(abs(y - 600) <= 1 for y in seam_ys), f"应检出水平缝 600，实际 {seam_ys}"
        assert len(xs_per_band) == 2
        assert any(abs(x - 500) <= 1 for x in xs_per_band[0]), f"带0 应检出垂直缝 500，实际 {xs_per_band[0]}"
        assert any(abs(x - 550) <= 1 for x in xs_per_band[1]), f"带1 应检出垂直缝 550，实际 {xs_per_band[1]}"

    def test_no_seams_uniform_image(self):
        """无拼接的均匀图像：无缝检出。"""
        img = np.full((400, 600), 120, dtype=np.uint8)
        seam_ys, xs_per_band = detect_seams(img)
        assert seam_ys == []
        assert all(xs == [] for xs in xs_per_band)

    def test_no_seams_plain_noise(self):
        """纯随机噪声（无拼接）：不应误检。"""
        rng = np.random.default_rng(6)
        img = _texture((600, 800), 100, rng)
        seam_ys, xs_per_band = detect_seams(img)
        assert seam_ys == []
        assert all(xs == [] for xs in xs_per_band)


class TestSeamTiles:
    """seam_tiles 图块组合与兜底。"""

    def test_band_column_combination(self):
        """条带 × 列区间组合图块：互不重叠、铺满图像、无重叠标记。"""
        image_size = (1000, 1200)  # (宽, 高)；各图块边长 ≤1200 不触发兜底
        tiles = seam_tiles(image_size, [600], [[400, 900], []], resolution=1024, overlap=256)
        # 带0: [0,400),[400,900),[900,1000); 带1: 整带 [0,1000)
        assert tiles == [
            (0, 0, 400, 600, 0),
            (400, 0, 900, 600, 0),
            (900, 0, 1000, 600, 0),
            (0, 600, 1000, 1200, 0),
        ], f"图块组合不符，实际 {tiles}"

    def test_oversized_tile_falls_back_to_grid(self):
        """超限图块（任一边 > max_tile_side）内部展开滑窗网格。"""
        image_size = (1400, 1000)
        tiles = seam_tiles(image_size, [], [[1400]], resolution=1024, overlap=256, max_tile_side=1200)
        # 带内唯一图块 1400x1000 > 1200 → 滑窗: 轴长 1400, stride=768 → 原点 [0, 376]
        xs = {x for x, _y, _x1, _y1, ov in tiles}
        assert 0 in xs and 376 in xs and len(xs) == 2, f"超限图块应展开滑窗，实际 {tiles}"
        assert all(ov == 256 for _x, _y, _x1, _y1, ov in tiles)

    def test_small_tile_no_overlap(self):
        """正常尺寸图块标记 overlap=0（整块直连）。"""
        image_size = (1000, 800)
        tiles = seam_tiles(image_size, [], [[500]], resolution=1024, overlap=256)
        assert tiles == [(0, 0, 500, 800, 0), (500, 0, 1000, 800, 0)]

    def test_black_bottom_band_empty(self):
        """底部黑区（无内容带）仍会产出图块（由推理侧自然无预测）。"""
        image_size = (1000, 500)
        tiles = seam_tiles(image_size, [300], [[400], []], resolution=1024, overlap=256)
        assert tiles == [(0, 0, 400, 300, 0), (400, 0, 1000, 300, 0), (0, 300, 1000, 500, 0)]
