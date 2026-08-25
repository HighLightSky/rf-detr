# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""FP/FN 可视化保存逻辑：大图错误单独目录的回归测试。

覆盖：
- 小图仍按原有 ``FP/{类名}``、``FN/{类名}`` 结构保存；
- 大图（传入 ``large_image_ids``）错误可视化每张只保存一次到
  ``large_errors_dir/{image_id}.jpg``（不区分 FP/FN、不按类分）；
- 大图可视化叠加裁窗边界（``large_crop_boxes``）不报错；
- 不传大图参数时行为与原来完全一致（全部写入 FP/FN 目录）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts import eval_lib
from val.competition_metrics import BoxRecord
from visualization.detection import (
    _format_error_label,
    clear_vis_dirs,
    save_fp_fn_visualizations,
    save_label_comparison_visualizations,
)


def _write_test_image(path: Path, width: int, height: int) -> None:
    """写一张纯色测试图（BGR）。"""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _records(image_id: str, class_id: int) -> list[BoxRecord]:
    """构造一条测试 BoxRecord。"""
    return [BoxRecord(image_id=image_id, class_id=class_id, xyxy=(10.0, 10.0, 30.0, 30.0), score=0.9)]


def test_large_image_errors_routed_to_separate_dir(tmp_path: Path) -> None:
    """大图 FP/FN 应保存到 large_errors 目录，小图保持原目录。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    small_path = image_dir / "small.jpg"
    large_path = image_dir / "large.jpg"
    _write_test_image(small_path, 100, 100)
    _write_test_image(large_path, 2100, 800)  # 长边 2100 ≥ 阈值 2000

    fp_dir = tmp_path / "FP"
    fn_dir = tmp_path / "FN"
    large_dir = tmp_path / "large_errors"
    class_names = {0: "MS", 1: "QHS"}

    fp_images = {0: {"small"}, 1: {"large"}}
    fn_images = {0: {"small"}, 1: {"large"}}
    fp_boxes = {"small": _records("small", 0), "large": _records("large", 1)}
    fn_boxes = {"small": _records("small", 0), "large": _records("large", 1)}
    tp_preds: dict[str, list[BoxRecord]] = {}
    all_gt: list[BoxRecord] = []

    clear_vis_dirs(fp_dir, fn_dir, class_names, large_errors_dir=large_dir)
    save_fp_fn_visualizations(
        fp_images,
        fn_images,
        fp_boxes,
        fn_boxes,
        tp_preds,
        all_gt,
        [small_path, large_path],
        class_names,
        fp_dir,
        fn_dir,
        large_image_ids={"large"},
        large_errors_dir=large_dir,
    )

    # 小图：保持原有目录结构
    assert (fp_dir / "MS" / "small.jpg").exists()
    assert (fn_dir / "MS" / "small.jpg").exists()
    # 大图：不再写入小图目录
    assert not (fp_dir / "QHS" / "large.jpg").exists()
    assert not (fn_dir / "QHS" / "large.jpg").exists()
    # 大图：每张只保存一次，不区分 FP/FN、不按类分
    assert (large_dir / "large.jpg").exists()
    assert not (large_dir / "FP").exists()
    assert not (large_dir / "FN").exists()
    assert not (large_dir / "QHS").exists()


def test_large_image_saved_once_across_classes(tmp_path: Path) -> None:
    """大图在多个类、同时有 FP 和 FN 时，只保存一张。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    large_path = image_dir / "large.jpg"
    _write_test_image(large_path, 2100, 800)

    fp_dir = tmp_path / "FP"
    fn_dir = tmp_path / "FN"
    large_dir = tmp_path / "large_errors"
    class_names = {0: "MS", 1: "QHS"}

    # 同一张大图在 MS、QHS 两个类都有 FP 和 FN
    fp_images = {0: {"large"}, 1: {"large"}}
    fn_images = {0: {"large"}, 1: {"large"}}
    fp_boxes = {"large": _records("large", 0) + _records("large", 1)}
    fn_boxes = {"large": _records("large", 0) + _records("large", 1)}

    clear_vis_dirs(fp_dir, fn_dir, class_names, large_errors_dir=large_dir)
    save_fp_fn_visualizations(
        fp_images,
        fn_images,
        fp_boxes,
        fn_boxes,
        {},
        [],
        [large_path],
        class_names,
        fp_dir,
        fn_dir,
        large_image_ids={"large"},
        large_errors_dir=large_dir,
    )

    assert (large_dir / "large.jpg").exists()
    assert len(list(large_dir.glob("*.jpg"))) == 1


def test_large_image_visualization_accepts_crop_boxes(tmp_path: Path) -> None:
    """大图可视化传入裁窗坐标时不报错且正常保存。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    large_path = image_dir / "large.jpg"
    _write_test_image(large_path, 2100, 800)

    fp_dir = tmp_path / "FP"
    fn_dir = tmp_path / "FN"
    large_dir = tmp_path / "large_errors"
    class_names = {0: "MS"}

    fp_images = {0: {"large"}}
    fn_images = {0: {"large"}}
    fp_boxes = {"large": _records("large", 0)}
    fn_boxes = {"large": _records("large", 0)}

    clear_vis_dirs(fp_dir, fn_dir, class_names, large_errors_dir=large_dir)
    save_fp_fn_visualizations(
        fp_images,
        fn_images,
        fp_boxes,
        fn_boxes,
        {},
        [],
        [large_path],
        class_names,
        fp_dir,
        fn_dir,
        large_image_ids={"large"},
        large_errors_dir=large_dir,
        large_crop_boxes={"large": [(50.0, 50.0, 1050.0, 850.0)]},
    )

    assert (large_dir / "large.jpg").exists()


def test_default_behavior_unchanged_without_large_params(tmp_path: Path) -> None:
    """不传大图参数时，所有图像（含大图）仍保存到原 FP/FN 目录。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    large_path = image_dir / "large.jpg"
    _write_test_image(large_path, 2100, 800)

    fp_dir = tmp_path / "FP"
    fn_dir = tmp_path / "FN"
    class_names = {0: "MS"}

    fp_images = {0: {"large"}}
    fn_images = {0: {"large"}}
    fp_boxes = {"large": _records("large", 0)}
    fn_boxes = {"large": _records("large", 0)}

    clear_vis_dirs(fp_dir, fn_dir, class_names)
    save_fp_fn_visualizations(
        fp_images,
        fn_images,
        fp_boxes,
        fn_boxes,
        {},
        [],
        [large_path],
        class_names,
        fp_dir,
        fn_dir,
    )

    assert (fp_dir / "MS" / "large.jpg").exists()
    assert (fn_dir / "MS" / "large.jpg").exists()


def test_error_labels_include_prediction_confidence() -> None:
    """FP 使用预测置信度，FN 明确标注不存在对应预测置信度。"""
    fp = BoxRecord("sample", 0, (10.0, 10.0, 30.0, 30.0), score=0.876)
    fn = BoxRecord("sample", 0, (10.0, 10.0, 30.0, 30.0))

    assert _format_error_label(fp, "FP", {0: "MS"}) == "MS(FP) conf=0.876"
    assert _format_error_label(fn, "FN", {0: "MS"}) == "MS(FN) conf=N/A"


def test_label_comparison_saves_every_image_and_marks_errors_red(tmp_path: Path) -> None:
    """标签对比模式为每张图输出对照图，并将 FP/FN 绘制为红色。"""
    image_path = tmp_path / "sample.jpg"
    _write_test_image(image_path, 100, 100)
    output_dir = tmp_path / "comparison"
    class_names = {0: "MS"}
    gt_records = [BoxRecord("sample", 0, (10.0, 10.0, 30.0, 30.0))]
    fp_boxes = {"sample": _records("sample", 0)}
    fn_boxes = {"sample": gt_records}

    saved_images = save_label_comparison_visualizations(
        fp_boxes=fp_boxes,
        fn_boxes=fn_boxes,
        tp_preds={},
        all_gt=gt_records,
        image_paths=[image_path],
        class_names=class_names,
        output_dir=output_dir,
    )

    comparison = cv2.imread(str(output_dir / "sample.jpg"))
    assert saved_images == 1
    assert comparison is not None
    assert comparison[40, 10, 2] > 200
    assert comparison[40, 10, 2] > comparison[40, 10, 0] * 3
    assert comparison[40, 110, 2] > 200
    assert comparison[40, 110, 2] > comparison[40, 110, 0] * 3


def test_eval_lib_loads_yolo_labels_for_comparison(tmp_path: Path) -> None:
    """评测工具可直接加载 YOLO 标签并生成预测对比图。"""
    image_path = tmp_path / "sample.jpg"
    _write_test_image(image_path, 100, 100)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "sample.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    saved_images, fp_count, fn_count = eval_lib.save_yolo_label_comparisons(
        image_paths=[image_path],
        pred_records=[BoxRecord("sample", 0, (10.0, 10.0, 30.0, 30.0), score=0.9)],
        class_names={0: "MS"},
        output_dir=tmp_path / "comparison",
        comparison_cfg=eval_lib.LabelComparisonCfg(labels_dir),
    )

    assert saved_images == 1
    assert fp_count == 0
    assert fn_count == 0
    assert (tmp_path / "comparison" / "sample.jpg").exists()
