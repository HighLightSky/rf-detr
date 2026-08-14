# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tiling 模块单元测试：滑窗原点网格、小图/大图拆分、按类别 NMS、中心归属合并。

对应里程碑 1（朴素滑窗切分 + 按类别 NMS 去重基线）与里程碑 2（center 策略：
中心归属 + 极严格安全合并）的纯函数部分，无 GPU 依赖；``tile_predict_records``
的端到端行为由评测回归验证（test_shwx_large.yaml 配置）。
"""

import dataclasses

import cv2
import numpy as np
import pytest
import torch

from scripts import eval_lib, expcfg
from scripts.tiling import (
    _check_tile_strategy,
    _TileDataset,
    _fragment_rescue_sides,
    _rescue_accept,
    _rescue_crop_bounds,
    _select_rescue_candidate,
    apply_nms,
    merge_center_duplicates,
    split_image_paths,
    tile_core_bounds,
    tile_origins,
)


class TestTileOrigins:
    """滑窗原点网格生成。"""

    def test_image_equal_to_tile(self):
        """图像恰等于 tile 时只生成单块。"""
        assert tile_origins((1024, 1024), 1024, 256) == [(0, 0)]

    def test_both_axes_larger(self):
        """1100×1100、overlap=256（stride=768）：每轴原点 [0, 76] → 2×2 网格。"""
        assert tile_origins((1100, 1100), 1024, 256) == [
            (0, 0),
            (76, 0),
            (0, 76),
            (76, 76),
        ]

    def test_short_axis_single_origin(self):
        """2000×700：短轴（700）只有原点 0，长轴多原点 → 单行多列。"""
        origins = tile_origins((2000, 700), 1024, 512)
        assert len({x for x, _ in origins}) == 3
        assert len({y for _, y in origins}) == 1

    def test_overlap_zero_adjacent(self):
        """Overlap=0 时原点等距无缝（stride=tile_size）。"""
        assert tile_origins((2048, 2048), 1024, 0) == [
            (0, 0),
            (1024, 0),
            (0, 1024),
            (1024, 1024),
        ]

    @pytest.mark.parametrize("overlap", [1024, 1025, -1])
    def test_overlap_invalid(self, overlap):
        """Overlap 越界（>= tile_size 或负数）抛 ValueError。"""
        with pytest.raises(ValueError):
            tile_origins((2000, 2000), 1024, overlap)

    @pytest.mark.parametrize(
        "image_size",
        [(2000, 2000), (12000, 12000), (3000, 1500), (5000, 700), (700, 5000), (1100, 1024)],
    )
    @pytest.mark.parametrize("overlap", [0, 128, 256, 512])
    def test_coverage_no_gap(self, image_size, overlap):
        """覆盖性质：原点不越界、相邻原点差 <= stride、长轴末原点贴边、无缝隙。"""
        width, height = image_size
        origins = tile_origins(image_size, 1024, overlap)
        assert origins, f"{image_size} overlap={overlap} 不应为空"
        stride = 1024 - overlap
        for seq, dim in ((sorted({x for x, _ in origins}), width), (sorted({y for _, y in origins}), height)):
            if dim > 1024:
                # 长轴：末原点贴到 dim - tile，相邻差不超过 stride（无缝隙）
                assert seq[-1] == dim - 1024
                for a, b in zip(seq, seq[1:]):
                    assert 0 < b - a <= stride
            else:
                assert seq == [0]


class TestSplitImagePaths:
    """小图/大图分流判定。"""

    def test_split_by_resolution(self, tmp_path):
        """Max(w,h) <= 分辨率归小图，否则归大图（含单轴超限与刚超限边界）。"""
        paths = [tmp_path / f"img{i}.jpg" for i in range(4)]
        size_map = {
            "img0": (800, 800),
            "img1": (1024, 1024),
            "img2": (1024, 1025),
            "img3": (2000, 700),
        }
        small, large = split_image_paths(paths, size_map, 1024)
        assert [p.stem for p in small] == ["img0", "img1"]
        assert [p.stem for p in large] == ["img2", "img3"]

    def test_all_small_returns_empty_large(self, tmp_path):
        """全部小图时大图列表为空（切分路径不触发）。"""
        paths = [tmp_path / "a.jpg"]
        size_map = {"a": (800, 800)}
        small, large = split_image_paths(paths, size_map, 1024)
        assert len(small) == 1
        assert large == []

    def test_keeps_input_order(self, tmp_path):
        """分组保持输入顺序（与评测按文件名排序的语义一致）。"""
        paths = [tmp_path / f"c{i}.jpg" for i in range(3)]
        size_map = {"c0": (1000, 1000), "c1": (2000, 2000), "c2": (900, 900)}
        small, large = split_image_paths(paths, size_map, 1024)
        assert [p.stem for p in small] == ["c0", "c2"]
        assert [p.stem for p in large] == ["c1"]


class TestTileDatasetSeamBounds:
    """seam 矩形在数据集层必须保留真实边界。"""

    def test_large_seam_tile_keeps_true_extent(self, tmp_path):
        """真实宽高大于 1024 的 seam 图块，不得被默认裁成 1024 前缀。"""
        image_path = tmp_path / "seam.jpg"
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        for x in range(image.shape[1]):
            image[:, x, :] = x % 256
        cv2.imwrite(str(image_path), image)

        dataset = _TileDataset(
            [image_path],
            {"seam": (1200, 800)},
            resolution=1024,
            overlap=256,
            origins_map={"seam": [(0, 0, 1100, 700, 0)]},
        )

        assert len(dataset) == 1
        stem, x0, y0, tile_w, tile_h, core, tile = dataset[0]
        assert stem == "seam"
        assert (x0, y0) == (0, 0)
        assert (tile_w, tile_h) == (1100, 700)
        assert core == (0, 0, 1100, 700)
        assert tile.shape == (3, 1024, 1024)


class TestApplyNms:
    """按类别 batched NMS 去重。"""

    def test_high_iou_keep_highest_score(self):
        """同类高 IoU 重叠框：保留分数最高者。"""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.5])
        keep = apply_nms(boxes, labels, scores, 0.5)
        assert keep.tolist() == [0]

    def test_low_iou_keep_both(self):
        """同类但 IoU 低于阈值（相邻目标）：两个都保留。"""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.5])
        keep = apply_nms(boxes, labels, scores, 0.5)
        assert sorted(keep.tolist()) == [0, 1]

    def test_different_classes_no_suppression(self):
        """不同类别的同位置框互不抑制（类间独立）。"""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        labels = torch.tensor([0, 1])
        scores = torch.tensor([0.9, 0.8])
        keep = apply_nms(boxes, labels, scores, 0.5)
        assert sorted(keep.tolist()) == [0, 1]

    def test_low_score_overlap_suppressed(self):
        """低分数重叠框（IoU 0.64 > 阈值）被高分数框抑制。"""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.3, 0.9])
        keep = apply_nms(boxes, labels, scores, 0.5)
        assert keep.tolist() == [1]

    def test_empty_input(self):
        """空输入返回空索引。"""
        boxes = torch.zeros((0, 4))
        labels = torch.zeros(0, dtype=torch.int64)
        scores = torch.zeros(0)
        keep = apply_nms(boxes, labels, scores, 0.5)
        assert keep.numel() == 0


class TestTileCoreBounds:
    """tile 核心区边界：halo 只在有邻居的一侧剥离，覆盖并集无空洞。"""

    def test_middle_tile(self):
        """双轴滑动中块：两侧各剥离 halo。"""
        assert tile_core_bounds((2560, 2560), (768, 768), 1024, 256) == (896, 896, 1664, 1664)

    def test_first_tile_reaches_left_edge(self):
        """首块核心延伸到图左/上边缘（防边缘条带漏检）。"""
        assert tile_core_bounds((2560, 2560), (0, 0), 1024, 256) == (0, 0, 896, 896)

    def test_last_tile_reaches_right_edge(self):
        """末块核心延伸到图右/下边缘（防边缘条带漏检）。"""
        assert tile_core_bounds((2560, 2560), (1536, 1536), 1024, 256) == (1664, 1664, 2560, 2560)

    def test_clamped_last_tile_overlap_strip(self):
        """夹取末块与前一块核心存在重叠条带（合并兜底的前提）。"""
        assert tile_core_bounds((2048, 2048), (1024, 1024), 1024, 256) == (1152, 1152, 2048, 2048)
        assert tile_core_bounds((2048, 2048), (768, 768), 1024, 256) == (896, 896, 1664, 1664)
        assert 1664 - 1152 == 512  # 重叠条带 [1152, 1664] 宽度 512px

    def test_short_axis_keeps_full_content(self):
        """短轴（单块轴）核心 = 全部内容，halo 不剥离（不误丢边缘目标）。"""
        assert tile_core_bounds((2000, 700), (0, 0), 1024, 256) == (0, 0, 896, 700)
        assert tile_core_bounds((2000, 700), (976, 0), 1024, 256) == (1104, 0, 2000, 700)

    def test_overlap_zero_full_tile(self):
        """overlap=0 时核心 = 整块内容。"""
        assert tile_core_bounds((2048, 2048), (0, 0), 1024, 0) == (0, 0, 1024, 1024)
        assert tile_core_bounds((2048, 2048), (1024, 1024), 1024, 0) == (1024, 1024, 2048, 2048)

    def test_odd_overlap_one_px_overlap(self):
        """奇数 overlap：相邻核心重叠恰 1px（歧义条带由合并兜底）。"""
        assert tile_core_bounds((2560, 2560), (0, 0), 1024, 255) == (0, 0, 897, 897)
        assert tile_core_bounds((2560, 2560), (769, 0), 1024, 255) == (896, 0, 1666, 897)

    def test_both_axes_within_resolution(self):
        """双轴都不超过分辨率：单块核心 = 全部内容。"""
        assert tile_core_bounds((900, 900), (0, 0), 1024, 256) == (0, 0, 900, 900)
        assert tile_core_bounds((1024, 1024), (0, 0), 1024, 256) == (0, 0, 1024, 1024)

    @pytest.mark.parametrize("image_size", [(2048, 2048), (2560, 2560), (5000, 700), (2000, 900)])
    @pytest.mark.parametrize("overlap", [0, 128, 255, 256, 512])
    def test_coverage_union_no_hole(self, image_size, overlap):
        """核心并集 = 全图（无空洞）——抽样像素点性质校验。"""
        width, height = image_size
        cores = [
            tile_core_bounds(image_size, (x0, y0), 1024, overlap) for x0, y0 in tile_origins(image_size, 1024, overlap)
        ]
        # 点查方式（而非集合抽样）：覆盖区间起点不与验证网格对齐时也能正确判定
        for x in range(0, width, 8):
            for y in range(0, height, 8):
                assert any(x_lo <= x < x_hi and y_lo <= y < y_hi for x_lo, y_lo, x_hi, y_hi in cores), (
                    f"{width}x{height} overlap={overlap} 核心空洞在 ({x},{y})"
                )


class TestMergeCenterDuplicates:
    """极严格安全合并：只抑制跨 tile 的同一目标重复框。"""

    def test_cross_tile_duplicate_suppressed(self):
        """跨 tile 近同框（IoU 0.96、中心距/面积比满足）→ 低分被抑制。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [2.0, 2.0, 102.0, 102.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.7])
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert keep.tolist() == [0]

    def test_same_tile_never_suppressed(self):
        """同 tile 的近同框永不合并（相邻真实目标免疫）。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [2.0, 2.0, 102.0, 102.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.7])
        tiles = torch.tensor([[0, 0], [0, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert sorted(keep.tolist()) == [0, 1]

    def test_iou_below_threshold_kept(self):
        """IoU 0.85 < 0.9（其余条件满足）→ 都保留（IoU 是主闸门）。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [4.0, 4.0, 104.0, 104.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.7])
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert sorted(keep.tolist()) == [0, 1]

    def test_different_classes_never_suppressed(self):
        """跨类别同框不同 tile → 都保留。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]])
        labels = torch.tensor([0, 1])
        scores = torch.tensor([0.9, 0.8])
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert sorted(keep.tolist()) == [0, 1]

    def test_adjacent_real_objects_kept(self):
        """相邻真实目标（IoU≈0.7 < 0.9）→ 都保留（召回保护）。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [18.0, 0.0, 118.0, 100.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.8])
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert sorted(keep.tolist()) == [0, 1]

    def test_clamped_last_tile_overlap_strip(self):
        """夹取末块重叠条带场景：同一目标被两块检出 → 低分被抑制。"""
        boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0], [101.0, 101.0, 201.0, 201.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.6, 0.95])
        tiles = torch.tensor([[768, 768], [1024, 1024]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert keep.tolist() == [1]

    def test_score_order_independent(self):
        """降序贪心：高分框输入顺序靠后也保留，低分被抑制。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [2.0, 2.0, 102.0, 102.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.7, 0.95])  # 低分在前输入
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert keep.tolist() == [1]

    def test_extreme_area_ratio_kept(self):
        """含包关系面积比 0.25（IoU 必然 < 0.9）→ 都保留（整体安全）。"""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [25.0, 25.0, 75.0, 75.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.8])
        tiles = torch.tensor([[0, 0], [768, 0]])
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert sorted(keep.tolist()) == [0, 1]

    def test_empty_input(self):
        """空输入返回空索引。"""
        boxes = torch.zeros((0, 4))
        labels = torch.zeros(0, dtype=torch.int64)
        scores = torch.zeros(0)
        tiles = torch.zeros((0, 2), dtype=torch.int64)
        keep = merge_center_duplicates(boxes, labels, scores, tiles)
        assert keep.numel() == 0


class TestTileStrategyValidation:
    """tile_strategy 策略名校验。"""

    def test_valid_strategies(self):
        """合法策略名不抛错。"""
        _check_tile_strategy("nms")
        _check_tile_strategy("center")
        _check_tile_strategy("center_rescue")

    @pytest.mark.parametrize("strategy", ["", "NMS", "nms_plus", "rescue"])
    def test_invalid_strategies_raise(self, strategy):
        """非法策略名抛 ValueError。"""
        with pytest.raises(ValueError):
            _check_tile_strategy(strategy)


class TestFragmentRescueSides:
    """残框触边识别：触 tile 边且该边非图像边界。"""

    def test_touches_right_edge(self):
        """框右缘 == tile 右缘（tile 右侧非图像边）→ 仅 right。"""
        sides = _fragment_rescue_sides((100.0, 100.0, 1024.0, 400.0), (0, 0), (1024, 1024), (2000, 1200))
        assert sides == (False, True, False, False)

    def test_touches_left_edge(self):
        """框左缘 == tile 左缘（tile 左侧非图像边）→ 仅 left。"""
        sides = _fragment_rescue_sides((0.0, 100.0, 500.0, 400.0), (768, 0), (1024, 1024), (2000, 1200))
        assert sides == (True, False, False, False)

    def test_tolerance_boundary(self):
        """贴内距离 == tol 判定为触边；tol+0.01 不触边。"""
        # 框右缘贴 tile 右缘内 2px（= tol）→ 触边
        assert _fragment_rescue_sides((100.0, 100.0, 1022.0, 400.0), (0, 0), (1024, 1024), (2000, 1200)) == (
            False,
            True,
            False,
            False,
        )
        # 框右缘贴内 2.01px（> tol）→ 不触边
        assert _fragment_rescue_sides((100.0, 100.0, 1021.99, 400.0), (0, 0), (1024, 1024), (2000, 1200)) == (
            False,
            False,
            False,
            False,
        )

    def test_touching_image_edge_not_fragment(self):
        """框贴 tile 右缘但该边是图像右缘（末块）→ 完整目标，非残框。"""
        # tile x 范围 [976, 2000]，框右缘贴 2000（= 图像右缘）→ 非残框
        sides = _fragment_rescue_sides((1100.0, 100.0, 2000.0, 400.0), (976, 0), (1024, 1024), (2000, 1200))
        assert sides == (False, False, False, False)

    def test_first_tile_left_is_image_edge(self):
        """首块左缘 == 图像左缘 → 非残框。"""
        sides = _fragment_rescue_sides((0.0, 100.0, 500.0, 400.0), (0, 0), (1024, 1024), (2000, 1200))
        assert sides == (False, False, False, False)

    def test_short_axis_single_tile_never_fragment(self):
        """短轴单块（内容 == 图像宽）→ 该轴永不判定为残框。"""
        sides = _fragment_rescue_sides((0.0, 100.0, 700.0, 400.0), (0, 0), (2000, 700), (2000, 700))
        assert sides == (False, False, False, False)

    def test_corner_touches_two_axes(self):
        """双轴角部触边（目标在 tile 角被切）→ left + top。"""
        sides = _fragment_rescue_sides((0.0, 0.0, 500.0, 500.0), (768, 176), (1024, 1024), (2000, 1200))
        assert sides == (True, False, True, False)

    def test_no_touch(self):
        """框完全在 tile 内部 → 全部 False。"""
        sides = _fragment_rescue_sides((300.0, 300.0, 700.0, 700.0), (0, 0), (1024, 1024), (2000, 1200))
        assert sides == (False, False, False, False)


class TestRescueCropBounds:
    """重裁块生成：触边轴内侧边锚定，不触边轴中心对齐，夹取安全。"""

    R = 1024

    def test_right_touch_anchors_inner_edge(self):
        """右触边：块 [f0-margin, f0+R-margin]，完整包含 D ≤ R-margin 的目标。"""
        fragment = (300.0, 100.0, 1024.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (False, True, False, False), (2000, 1200), 1024)
        assert x0 == 300 - 24
        assert w == 1024
        # 目标 [300, 300+D] 完整落入 [276, 1300] ⟺ D ≤ 1000
        assert 300 + 1000 <= x0 + 1024

    def test_left_touch_anchors_inner_edge(self):
        """左触边：块 [f1-(R-margin), f1+margin]，夹取到图像内。"""
        # f1=500 时 start = 500-1000 = -500 → 夹取到 0
        fragment = (0.0, 100.0, 500.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (True, False, False, False), (2000, 1200), 1024)
        assert x0 == 0
        assert w == 1024
        assert x0 + w >= 500  # 残框右缘被包含
        # f1=1500 时 start = 500，无夹取
        fragment = (1000.0, 100.0, 1500.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (True, False, False, False), (2000, 1200), 1024)
        assert x0 == 1500 - (1024 - 24)
        assert w == 1024
        # 目标 [1500-D, 1500] 完整落入 ⟺ D ≤ 1000
        assert x0 <= 1500 - 1000

    def test_untouched_axis_center_aligned(self):
        """不触边轴中心对齐，块完整包含残框（越界时夹取到 0）。"""
        fragment = (300.0, 300.0, 1024.0, 700.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (False, True, False, False), (2000, 1200), 1024)
        # y 轴不触边：中心对齐 = (300+700)/2 - 512 = -212 → 夹取到 0
        assert y0 == 0
        assert y0 + h >= 700  # 残框下缘被包含

    def test_clamp_to_image_bounds(self):
        """夹取：start < 0 → 0；start > dim-R → dim-R；内容可能 < R。"""
        # 左触边且内侧边靠近图像左缘 → start 夹到 0
        fragment = (0.0, 100.0, 100.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (True, False, False, False), (2000, 1200), 1024)
        assert x0 == 0
        assert w == 1024
        # 右触边且内侧边靠近图像右缘 → start 夹到 dim-R
        fragment = (1800.0, 100.0, 2000.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (False, True, False, False), (2000, 1200), 1024)
        assert x0 == 2000 - 1024
        assert w == 1024

    def test_short_axis_untouched_clamps_to_zero(self):
        """短轴 (dim < R) 不触边：start 夹到 0（回归：不能夹成负数）。"""
        fragment = (300.0, 100.0, 1024.0, 600.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (False, True, False, False), (2000, 700), 1024)
        assert y0 == 0
        assert h == 700

    def test_both_sides_touch_center_aligned(self):
        """双轴同触（目标跨度 > R）：中心对齐兜底。"""
        fragment = (0.0, 100.0, 2000.0, 500.0)
        x0, y0, w, h = _rescue_crop_bounds(fragment, (True, True, False, False), (2000, 1200), 1024)
        assert x0 == int(round((0 + 2000) / 2 - 512))  # 中心对齐 = 488，未夹取
        assert w == 1024

    @pytest.mark.parametrize("dim", [2000, 3000, 5000])
    def test_crop_contains_full_target(self, dim):
        """属性测试：真实一致构造（残框 = 目标∩tile、sides 按目标超出方向），
        新块完整包含目标（D ∈ [256, 1000]）。"""
        tile = (0, 0, 1024, 1024)  # tile 内容范围（全图坐标）
        for d in (256, 512, 900, 1000):
            for target in (
                (300.0, 300.0, 300.0 + d, 300.0 + d),  # 目标在 tile 内部
                (900.0, 300.0, 900.0 + d, 300.0 + d),  # 目标向右超出
                (300.0, 900.0, 300.0 + d, 900.0 + d),  # 目标向下超出
                (900.0, 900.0, 900.0 + d, 900.0 + d),  # 目标向右下超出
            ):
                if target[2] > dim or target[3] > dim:
                    continue  # 目标超出图像范围
                # 残框 = 目标 ∩ tile（真实语义：postprocess clamp 后）
                fragment = (
                    max(target[0], tile[0]),
                    max(target[1], tile[1]),
                    min(target[2], tile[2]),
                    min(target[3], tile[3]),
                )
                if fragment[2] <= fragment[0] or fragment[3] <= fragment[1]:
                    continue
                # 触边方向 = 目标超出 tile 的方向
                sides = (
                    target[0] < tile[0],
                    target[2] > tile[2],
                    target[1] < tile[1],
                    target[3] > tile[3],
                )
                x0, y0, w, h = _rescue_crop_bounds(fragment, sides, (dim, dim), 1024)
                assert x0 <= target[0] and x0 + w >= target[2], (
                    f"D={d} target={target} sides={sides} x 轴未包含: 块 [{x0}, {x0 + w}]"
                )
                assert y0 <= target[1] and y0 + h >= target[3], (
                    f"D={d} target={target} sides={sides} y 轴未包含: 块 [{y0}, {y0 + h}]"
                )


class TestRescueAccept:
    """重检候选接受判定：完整包含 + 向缺口方向延伸。"""

    FRAG = (300.0, 300.0, 1024.0, 700.0)
    SIDES_RIGHT = (False, True, False, False)

    def test_contains_and_extends_accepted(self):
        """完整包含残框且向右延伸 → 接受。"""
        assert _rescue_accept((276.0, 276.0, 1300.0, 724.0), self.FRAG, self.SIDES_RIGHT)

    def test_no_extension_rejected(self):
        """框 == 残框（无延伸，假触边）→ 拒绝，保留残框兜底。"""
        assert not _rescue_accept((298.0, 298.0, 1026.0, 702.0), self.FRAG, self.SIDES_RIGHT)

    def test_overlap_without_contain_rejected(self):
        """只延伸但不完整包含（缺口区相邻真实目标）→ 拒绝。"""
        # 候选只覆盖残框左半 + 向右延伸
        assert not _rescue_accept((276.0, 300.0, 1300.0, 500.0), self.FRAG, self.SIDES_RIGHT)

    def test_min_gain_boundary(self):
        """延伸 == min_gain 接受；min_gain - ε 拒绝。"""
        assert _rescue_accept((276.0, 276.0, 1028.0, 724.0), self.FRAG, self.SIDES_RIGHT)
        assert not _rescue_accept((276.0, 276.0, 1027.99, 724.0), self.FRAG, self.SIDES_RIGHT)

    def test_both_axes_must_extend(self):
        """双轴触边：两轴都必须延伸。"""
        frag = (0.0, 0.0, 500.0, 500.0)
        sides = (True, False, True, False)
        # 只向左延伸、不向上延伸 → 拒绝
        assert not _rescue_accept((-200.0, 2.0, 500.0, 500.0), frag, sides)
        # 两轴都延伸 → 接受
        assert _rescue_accept((-200.0, -200.0, 500.0, 500.0), frag, sides)


class TestSelectRescueCandidate:
    """重检候选选择：同类 + accept + 分数最高（平局面积最大）。"""

    FRAG = (300.0, 300.0, 1024.0, 700.0)
    SIDES = (False, True, False, False)

    def test_picks_highest_score(self):
        """多候选取分数最高者。"""
        candidates = [
            (276.0, 276.0, 1300.0, 724.0),  # 合格，低分
            (276.0, 276.0, 1400.0, 724.0),  # 合格，高分
        ]
        best = _select_rescue_candidate(candidates, [0, 0], [0.5, 0.9], self.FRAG, 0, self.SIDES)
        assert best == (candidates[1], 0, 0.9)

    def test_score_tie_prefers_larger_area(self):
        """平局取面积最大。"""
        candidates = [
            (276.0, 276.0, 1300.0, 724.0),
            (276.0, 276.0, 1400.0, 724.0),
        ]
        best = _select_rescue_candidate(candidates, [0, 0], [0.9, 0.9], self.FRAG, 0, self.SIDES)
        assert best[0] == candidates[1]

    def test_wrong_class_rejected(self):
        """不同类别不参与替换。"""
        candidates = [(276.0, 276.0, 1300.0, 724.0)]
        best = _select_rescue_candidate(candidates, [1], [0.9], self.FRAG, 0, self.SIDES)
        assert best is None

    def test_no_qualified_returns_none(self):
        """无合格候选 → None（调用方保留残框）。"""
        assert _select_rescue_candidate([], [], [], self.FRAG, 0, self.SIDES) is None


class TestFilterPostprocessResults:
    """置信度阈值过滤（整图路径与切分路径共用的公共函数）。"""

    def _result(self, boxes, labels, scores):
        return [{"boxes": boxes, "labels": labels, "scores": scores}]

    def test_strict_greater_than(self):
        """严格大于：等于阈值的框被过滤。"""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.25, 0.26])
        filtered = eval_lib.filter_postprocess_results(self._result(boxes, labels, scores), 0.25)
        boxes_out, labels_out, scores_out = filtered[0]
        assert boxes_out.shape[0] == 1
        assert [round(v, 4) for v in scores_out.tolist()] == [0.26]

    def test_class_threshold_fallback(self):
        """命中逐类阈值的类用类阈值，未命中回退全局阈值。"""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]])
        labels = torch.tensor([0, 1, 1])
        scores = torch.tensor([0.1, 0.2, 0.6])
        # 类别 1 阈值 0.5：score 0.6 保留、0.2 过滤；类别 0 回退全局 0.25：0.1 过滤
        filtered = eval_lib.filter_postprocess_results(self._result(boxes, labels, scores), 0.25, {1: 0.5})
        boxes_out, labels_out, scores_out = filtered[0]
        assert labels_out.tolist() == [1]
        assert [round(v, 4) for v in scores_out.tolist()] == [0.6]

    def test_empty_thresholds_dict_falls_back_to_global(self):
        """空逐类阈值字典 = 全部用全局阈值。"""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        labels = torch.tensor([0])
        scores = torch.tensor([0.3])
        filtered = eval_lib.filter_postprocess_results(self._result(boxes, labels, scores), 0.25, {})
        assert [round(v, 4) for v in filtered[0][2].tolist()] == [0.3]

    def test_all_filtered_returns_empty_tensors(self):
        """全部被过滤时返回空张量（shape 保持 (0, 4) / (0,)）。"""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        labels = torch.tensor([0])
        scores = torch.tensor([0.1])
        filtered = eval_lib.filter_postprocess_results(self._result(boxes, labels, scores), 0.25)
        boxes_out, labels_out, scores_out = filtered[0]
        assert boxes_out.shape == (0, 4)
        assert labels_out.shape == (0,)
        assert scores_out.shape == (0,)

    def test_multi_image_output(self):
        """多图输入返回与输入等长的列表。"""
        r1 = self._result(torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64), torch.zeros(0))
        r2 = self._result(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0]),
            torch.tensor([0.9]),
        )
        filtered = eval_lib.filter_postprocess_results(r1 + r2, 0.25)
        assert len(filtered) == 2
        assert [round(v, 4) for v in filtered[1][2].tolist()] == [0.9]


class TestInferenceCfgTilingFields:
    """InferenceCfg 滑窗字段：默认值 + yaml 透传（test.py 的构造路径）。"""

    def test_default_values(self):
        """切分默认关闭（tile_overlap=0），合并策略默认 nms（里程碑 1 基线）。"""
        cfg = eval_lib.InferenceCfg()
        assert cfg.tile_overlap == 0
        assert cfg.tile_nms_iou == 0.5
        assert cfg.tile_batch_size is None
        assert cfg.tile_strategy == "nms"

    def test_yaml_passthrough(self):
        """Yaml 的 test 段新键经 build_test_kwargs 透传到 InferenceCfg（与 test.py 同构）。"""
        kwargs = expcfg.build_test_kwargs(
            {
                "test": {
                    "dataset": "shwx",
                    "tile_overlap": 256,
                    "tile_nms_iou": 0.6,
                    "tile_batch_size": 16,
                    "tile_strategy": "center",
                }
            }
        )
        infer_fields = {f.name for f in dataclasses.fields(eval_lib.InferenceCfg)}
        infer = eval_lib.InferenceCfg(**{k: v for k, v in kwargs.items() if k in infer_fields})
        assert infer.tile_overlap == 256
        assert infer.tile_nms_iou == 0.6
        assert infer.tile_batch_size == 16
        assert infer.tile_strategy == "center"


class TestEffectiveNumWorkers:
    """切分模式对应的实际 DataLoader worker 数。"""

    def test_seam_mode_forces_serial_workers(self):
        """seam 模式主进程做完缝检测后不再 fork DataLoader worker。"""
        infer = eval_lib.InferenceCfg(num_workers=12, tile_cut_mode="seam")
        assert eval_lib._effective_num_workers(infer, seam_mode=True) == 0

    def test_grid_mode_preserves_workers(self):
        """普通滑窗模式保留配置中的 worker 数。"""
        infer = eval_lib.InferenceCfg(num_workers=12, tile_cut_mode="grid")
        assert eval_lib._effective_num_workers(infer, seam_mode=False) == 12
