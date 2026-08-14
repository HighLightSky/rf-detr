# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切流水线纯函数测试：letterbox 预处理、坐标映射与裁切。

覆盖 ``large_cut_pipeline.py`` 的核心几何函数（不依赖 GPU/模型）：letterbox 尺寸、坐标往返映射、裁窗 clamp。
"""

import numpy as np
import pytest

from scripts.large_cut_pipeline import crop_with_padding, letterbox_resize, map_boxes_to_original


class TestLetterboxResize:
    """``letterbox_resize`` 等比缩放 + 黑边 padding。"""

    def test_long_edge_target(self):
        """长边（1000）缩放到 704，短边（500）按同比例。"""
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox_resize(image, target=704)
        assert padded.shape == (704, 704, 3)
        assert scale == pytest.approx(704 / 1000)
        # 缩放后 704x352，上下 padding 各 176，左右 0
        assert pad_y == 176
        assert pad_x == 0

    def test_padding_symmetric(self):
        """方形图无 padding；奇数 padding 时右/下多 1px（对称优先）。"""
        # 700x700 方形图（H=W=700）：scale=704/700，缩放后仍为 704x704，无 pad
        image = np.zeros((700, 700, 3), dtype=np.uint8)
        _, _, pad_x, pad_y = letterbox_resize(image, target=704)
        assert (pad_x, pad_y) == (0, 0)

        # 宽 700 高 705（np shape (705, 700)）：new_w=699（pad_w=5 → pad_x=2，右 3px）
        image = np.zeros((705, 700, 3), dtype=np.uint8)
        _, _, pad_x, pad_y = letterbox_resize(image, target=704)
        assert (pad_x, pad_y) == (2, 0)

    def test_padding_is_black(self):
        """Padding 区域为黑（0），与原图内容区（白色）可区分。"""
        image = np.full((500, 1000, 3), 255, dtype=np.uint8)  # 全白原图（H=500, W=1000）
        padded, _, pad_x, pad_y = letterbox_resize(image, target=704)
        assert pad_x == 0
        # 内容区从 y=pad_y 开始：上一行是 padding 区（黑），该行是内容区（白）
        assert padded[pad_y - 1, 0, 0] == 0
        assert padded[pad_y, 0, 0] == 255


class TestMapBoxesToOriginal:
    """``map_boxes_to_original`` 往返映射。"""

    def test_roundtrip_within_1px(self):
        """原图框 → letterbox → 映射回来误差 < 1px。"""
        # 原图 2000x1000，letterbox 到 704：scale=0.352, pad_x=0, pad_y=176
        width, height = 2000, 1000
        scale = 704 / max(width, height)
        pad_y = (704 - int(1000 * scale)) // 2

        box_orig = np.array([[100.0, 50.0, 500.0, 300.0]])
        # 手工构造 letterbox 空间坐标
        box_lb = np.array(
            [
                [
                    box_orig[0, 0] * scale,
                    box_orig[0, 1] * scale + pad_y,
                    box_orig[0, 2] * scale,
                    box_orig[0, 3] * scale + pad_y,
                ]
            ]
        )
        mapped = map_boxes_to_original(box_lb, scale, 0, pad_y, width, height)
        assert np.allclose(mapped, box_orig, atol=1.0)

    def test_clip_and_drop_degenerate(self):
        """越界框被 clip，完全落在 padding 区的退化框被剔除。"""
        # 原图 W=2000 H=1000，letterbox 704：scale=0.352, pad_x=0, pad_y=176
        width, height = 2000, 1000
        scale = 704 / 2000
        pad_y = (704 - int(1000 * scale)) // 2
        # 有效框 + 越界框（x1 映射后 > 2000）+ 全落在 padding 区（y < pad_y）的退化框
        boxes = np.array(
            [
                [100.0, 250.0, 300.0, 400.0],
                [690.0, 250.0, 720.0, 400.0],
                [10.0, 10.0, 100.0, 100.0],
            ]
        )
        mapped = map_boxes_to_original(boxes, scale, 0, pad_y, width, height)
        assert mapped.shape[0] == 2
        # 越界框 x1 被 clip 到 width
        clipped = mapped[1]
        assert clipped[2] == width
        assert clipped[0] < clipped[2]

    def test_empty_input(self):
        """空框数组直接返回。"""
        mapped = map_boxes_to_original(np.zeros((0, 4)), 1.0, 0, 0, 100, 100)
        assert mapped.shape == (0, 4)


class TestCropWithPadding:
    """``crop_with_padding`` 裁切与外扩。"""

    def test_padding_expands_box(self):
        """框四周外扩 32px。"""
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        crop, crop_xyxy = crop_with_padding(image, (100, 100, 300, 200), padding=32)
        assert crop_xyxy == (68, 68, 332, 232)
        assert crop.shape[:2] == (164, 264)  # (232-68) x (332-68)

    def test_clamp_to_image_boundary(self):
        """贴边框 clamp 到图像边界，不外溢。"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, crop_xyxy = crop_with_padding(image, (0, 0, 50, 50), padding=32)
        assert crop_xyxy == (0, 0, 82, 82)
        crop, crop_xyxy = crop_with_padding(image, (80, 80, 100, 100), padding=32)
        assert crop_xyxy == (48, 48, 100, 100)

    def test_zero_padding(self):
        """Padding=0 时裁窗即原框。"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, crop_xyxy = crop_with_padding(image, (10, 10, 40, 40), padding=0)
        assert crop_xyxy == (10, 10, 40, 40)
        assert crop.shape[:2] == (30, 30)
