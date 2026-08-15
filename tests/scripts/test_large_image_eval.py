# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图切分评估配置与图像分类逻辑的回归测试。

覆盖：
- ``LargeImageCfg`` 默认值与字段；
- ``_classify_large_images`` 在 image_size_map 可用/不可用（PIL 回退）两条路径；
- ``build_test_report`` 在提供大图耗时统计时输出平均/最大时长行。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts import eval_lib


def _write_image(path: Path, width: int, height: int) -> None:
    """写一张纯色测试图。"""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_large_image_cfg_defaults() -> None:
    """``LargeImageCfg`` 默认值应与 test.yaml 注释口径一致。"""
    cfg = eval_lib.LargeImageCfg()
    assert cfg.min_side == 2000
    assert cfg.boundary_checkpoint is None
    assert cfg.boundary_resolution == 704
    assert cfg.boundary_conf == 0.25
    assert cfg.detector_conf == 0.25
    assert cfg.padding == 32
    assert cfg.nms_iou == 0.0
    assert cfg.batch_size == 8
    assert cfg.num_workers == 4


def test_classify_large_images_with_size_map() -> None:
    """有 ``image_size_map`` 时按长边阈值直接判定，不读图像。"""
    image_paths = [Path("/tmp/small.jpg"), Path("/tmp/large.jpg")]
    size_map = {"small": (800, 600), "large": (3000, 2000)}
    large_ids = eval_lib._classify_large_images(image_paths, size_map, min_side=2000)
    assert large_ids == {"large"}
    assert eval_lib._classify_large_images(image_paths, size_map, min_side=0) == set()


def test_classify_large_images_pil_fallback(tmp_path: Path) -> None:
    """无 ``image_size_map`` 时用 PIL 头信息读取尺寸判定。"""
    small = tmp_path / "small.jpg"
    large = tmp_path / "large.jpg"
    _write_image(small, 800, 600)
    _write_image(large, 3000, 2000)
    large_ids = eval_lib._classify_large_images([small, large], None, min_side=2000)
    assert large_ids == {"large"}


def test_build_test_report_includes_large_image_timing(tmp_path: Path) -> None:
    """提供大图耗时统计时，报告输出平均/最大时长行。"""
    dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, output_dir="output/x")
    infer = eval_lib.InferenceCfg()
    lines = eval_lib.build_test_report(
        dataset_name="shwx",
        checkpoint_path=Path("/tmp/ckpt.pth"),
        test_image_paths=[],
        gt_records=[],
        pred_records=[],
        throughput=10.0,
        timed_images=100,
        gpu_util=None,
        eval_results={"all": eval_lib.EvalResult(1, 0, 0), "groups": {}},
        group_macro={},
        total_macro={
            "avg_tp": 1.0,
            "avg_fp": 0.0,
            "avg_fn": 0.0,
            "recall": 1.0,
            "fdr": 0.0,
            "precision": 1.0,
        },
        per_class_results={"groups": {}},
        dataset=dataset,
        infer=infer,
        large_errors_dir=tmp_path / "large_errors",
        large_image_stats={"count": 4.0, "avg": 1.25, "max": 2.0, "total": 5.0},
    )
    assert any("大图目标检测" in line and "平均 1.25s" in line and "最大 2.00s" in line for line in lines)
