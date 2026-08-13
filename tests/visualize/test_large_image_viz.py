# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图切割结果对比可视化（左 GT / 右 Predict）单元测试。

覆盖 ``visualization.detection.save_large_image_visualizations``：随机抽样
确定性、左右拼接图生成、缩放后框坐标换算、无框大图不崩溃。无 GPU 依赖。
"""

from pathlib import Path

import cv2
import numpy as np

from val.competition_metrics import BoxRecord
from visualization.detection import save_large_image_visualizations


def _make_image(path: Path, width: int = 2000, height: int = 1200) -> None:
    """生成一张纯色测试大图（BGR，目标用绿色，红色保留给网格线判定）。"""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (600, 500), (0, 255, 0), -1)  # 一个"目标"
    cv2.imwrite(str(path), img)


def _make_records(image_id: str) -> tuple[list[BoxRecord], list[BoxRecord]]:
    """构造该大图的 GT 与预测框记录（像素坐标）。"""
    gts = [
        BoxRecord(image_id=image_id, class_id=0, xyxy=(300.0, 200.0, 600.0, 500.0)),
        BoxRecord(image_id=image_id, class_id=1, xyxy=(1000.0, 100.0, 1200.0, 300.0)),
    ]
    preds = [
        BoxRecord(image_id=image_id, class_id=0, xyxy=(300.0, 200.0, 600.0, 500.0), score=0.92),
        BoxRecord(image_id=image_id, class_id=1, xyxy=(1000.0, 100.0, 1200.0, 300.0), score=0.61),
        BoxRecord(image_id=image_id, class_id=2, xyxy=(1500.0, 800.0, 1700.0, 950.0), score=0.45),
    ]
    return gts, preds


class TestSaveLargeImageVisualizations:
    """大图对比可视化：抽样、拼接、缩放坐标。"""

    def test_generates_side_by_side_file(self, tmp_path):
        """输出左 GT / 右 Predict 的左右拼接图，尺寸为展示尺寸两倍宽。"""
        image_path = tmp_path / "big.jpg"
        _make_image(image_path)
        gts, preds = _make_records("big")
        out_dir = tmp_path / "viz"
        save_large_image_visualizations(
            ["big"],
            gts,
            preds,
            [image_path],
            {0: "MS", 1: "A1_SU-35", 2: "FSC"},
            out_dir,
            count=1,
            seed=0,
        )
        assert (out_dir / "big.jpg").exists()
        combined = cv2.imread(str(out_dir / "big.jpg"))
        # 长边 2000 > 展示上限 1600 → 缩放后宽 = 1600/2000*2000，拼接后两倍宽
        assert combined.shape[1] == 2 * int(round(2000 * 1600 / 2000))
        assert combined.shape[0] == 30 + int(round(1200 * 1600 / 2000))  # 标题条 30px

    def test_no_scale_when_small(self, tmp_path):
        """长边不超过展示上限时不缩放。"""
        image_path = tmp_path / "small_big.jpg"
        _make_image(image_path, width=1024, height=1024)
        gts, preds = _make_records("small_big")
        out_dir = tmp_path / "viz2"
        save_large_image_visualizations(
            ["small_big"],
            gts,
            preds,
            [image_path],
            {0: "MS", 1: "A1_SU-35", 2: "FSC"},
            out_dir,
            count=1,
        )
        combined = cv2.imread(str(out_dir / "small_big.jpg"))
        assert combined.shape[1] == 2 * 1024
        assert combined.shape[0] == 30 + 1024

    def test_seed_determinism(self, tmp_path):
        """固定种子抽样结果可复现；不同种子抽样可能不同。"""
        image_paths = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            _make_image(p, width=1100, height=1100)
            image_paths.append(p)
        records = [(f"img{i}",) + _make_records(f"img{i}") for i in range(5)]
        gts = [g for _, gs, _ in records for g in gs]
        preds = [p for _, _, ps in records for p in ps]

        out1 = tmp_path / "seed1"
        out2 = tmp_path / "seed2"
        save_large_image_visualizations(
            [f"img{i}" for i in range(5)],
            gts,
            preds,
            image_paths,
            {0: "MS", 1: "A1_SU-35", 2: "FSC"},
            out1,
            count=2,
            seed=42,
        )
        save_large_image_visualizations(
            [f"img{i}" for i in range(5)],
            gts,
            preds,
            image_paths,
            {0: "MS", 1: "A1_SU-35", 2: "FSC"},
            out2,
            count=2,
            seed=42,
        )
        assert sorted(p.name for p in out1.glob("*.jpg")) == sorted(p.name for p in out2.glob("*.jpg"))
        assert len(list(out1.glob("*.jpg"))) == 2

    def test_no_predictions_still_saves(self, tmp_path):
        """大图无预测框时右面板为空图，不崩溃。"""
        image_path = tmp_path / "nopred.jpg"
        _make_image(image_path, width=1100, height=1100)
        gts, _ = _make_records("nopred")
        out_dir = tmp_path / "viz3"
        save_large_image_visualizations(
            ["nopred"],
            gts,
            [],
            [image_path],
            {0: "MS", 1: "A1_SU-35"},
            out_dir,
            count=1,
        )
        assert (out_dir / "nopred.jpg").exists()

    def test_count_exceeds_pool(self, tmp_path):
        """Count 大于候选池时取全部。"""
        image_path = tmp_path / "one.jpg"
        _make_image(image_path, width=1100, height=1100)
        gts, _ = _make_records("one")
        out_dir = tmp_path / "viz4"
        save_large_image_visualizations(
            ["one"],
            gts,
            [],
            [image_path],
            {0: "MS", 1: "A1_SU-35"},
            out_dir,
            count=10,
        )
        assert len(list(out_dir.glob("*.jpg"))) == 1

    def test_tile_grid_drawn_with_tile_size(self, tmp_path):
        """传入 tile_size/tile_overlap 时右面板叠加红色滑窗网格，输出尺寸不变。"""
        image_path = tmp_path / "grid.jpg"
        _make_image(image_path, width=2048, height=1024)
        gts, preds = _make_records("grid")
        out_dir = tmp_path / "viz_grid"
        save_large_image_visualizations(
            ["grid"],
            gts,
            preds,
            [image_path],
            {0: "MS", 1: "A1_SU-35", 2: "FSC"},
            out_dir,
            count=1,
            tile_size=1024,
            tile_overlap=256,
        )
        assert (out_dir / "grid.jpg").exists()
        combined = cv2.imread(str(out_dir / "grid.jpg"))
        # 2048 宽 > 展示上限 1600 → 缩放，右面板含网格但拼接尺寸与不画网格时一致
        display_w = int(round(2048 * 1600 / 2048))
        assert combined.shape[1] == 2 * display_w
        # 右侧面板存在红色像素（网格线 COLOR_FP = (0,0,255)）
        right = combined[30:, display_w:, :]
        red_mask = (right[:, :, 2] > 200) & (right[:, :, 0] < 100) & (right[:, :, 1] < 100)
        assert red_mask.any(), "右面板未检测到红色网格线"

    def test_no_grid_without_tile_size(self, tmp_path):
        """不传 tile_size 时右面板不绘制网格线（纯预测框）。"""
        image_path = tmp_path / "nogrid.jpg"
        _make_image(image_path, width=1100, height=1100)
        gts, _ = _make_records("nogrid")
        out_dir = tmp_path / "viz_nogrid"
        save_large_image_visualizations(
            ["nogrid"],
            gts,
            [],
            [image_path],
            {0: "MS", 1: "A1_SU-35"},
            out_dir,
            count=1,
        )
        combined = cv2.imread(str(out_dir / "nogrid.jpg"))
        right = combined[30:, combined.shape[1] // 2 :, :]
        red_mask = (right[:, :, 2] > 200) & (right[:, :, 0] < 100) & (right[:, :, 1] < 100)
        assert not red_mask.any(), "未传 tile_size 时不应出现红色网格线"
