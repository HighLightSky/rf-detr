# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tiling 模块单元测试：滑窗原点网格、小图/大图拆分、按类别 NMS。

对应里程碑 1（朴素滑窗切分 + 按类别 NMS 去重基线）的纯函数部分，无 GPU 依赖；
``tile_predict_records`` 的端到端行为由评测回归验证（test_shwx_large.yaml 配置）。
"""

import dataclasses

import pytest
import torch

from scripts import eval_lib, expcfg
from scripts.tiling import apply_nms, split_image_paths, tile_origins


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
        """切分默认关闭（tile_overlap=0），NMS 阈值与 tile 批量有合理默认。"""
        cfg = eval_lib.InferenceCfg()
        assert cfg.tile_overlap == 0
        assert cfg.tile_nms_iou == 0.5
        assert cfg.tile_batch_size is None

    def test_yaml_passthrough(self):
        """Yaml 的 test 段新键经 build_test_kwargs 透传到 InferenceCfg（与 test.py 同构）。"""
        kwargs = expcfg.build_test_kwargs(
            {"test": {"dataset": "shwx", "tile_overlap": 256, "tile_nms_iou": 0.6, "tile_batch_size": 16}}
        )
        infer_fields = {f.name for f in dataclasses.fields(eval_lib.InferenceCfg)}
        infer = eval_lib.InferenceCfg(**{k: v for k, v in kwargs.items() if k in infer_fields})
        assert infer.tile_overlap == 256
        assert infer.tile_nms_iou == 0.6
        assert infer.tile_batch_size == 16
