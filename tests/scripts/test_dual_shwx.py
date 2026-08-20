# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SHWX 双模型数据缓存和类别映射测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import eval_lib, test_dual_shwx
from scripts.dual_shwx import PAN_CLASSES, RGB_CLASSES, prepare_dual_dataset


def _write_image(path: Path, rgb: tuple[int, int, int]) -> None:
    """写入一个最小测试图像。"""
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[...] = rgb
    Image.fromarray(array, mode="RGB").save(path)


def _write_sample(root: Path, split: str, name: str, class_id: int, rgb: tuple[int, int, int]) -> None:
    """写入一张图像和一个 YOLO 标签。"""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    _write_image(root / "images" / split / f"{name}.jpg", rgb)
    (root / "labels" / split / f"{name}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _make_dataset(root: Path) -> None:
    """构造包含 PAN、RGB 和空标注样本的最小数据集。"""
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"pan_{split}", 0, (80, 80, 80))
        _write_sample(root, split, f"rgb_{split}", 24, (20, 180, 40))
    (root / "images" / "train" / "empty_train.jpg").parent.mkdir(parents=True, exist_ok=True)
    _write_image(root / "images" / "train" / "empty_train.jpg", (90, 90, 90))


def test_prepare_dual_dataset_remaps_labels_and_reuses_cache(tmp_path: Path) -> None:
    """验证局部标签映射、清单计数和缓存复用。"""
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_dataset(dataset_dir)

    manifest = prepare_dual_dataset(dataset_dir, cache_dir, rebuild=True)
    assert manifest.threshold_accuracy == 1.0
    assert len(manifest.records["train"]["pan"]) == 2
    assert len(manifest.records["train"]["rgb"]) == 1
    assert (cache_dir / "pan" / "labels" / "train" / "pan_train.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert (cache_dir / "rgb" / "labels" / "train" / "rgb_train.txt").read_text() == "20 0.5 0.5 0.2 0.2\n"

    cached = prepare_dual_dataset(dataset_dir, cache_dir)
    assert cached.fingerprint == manifest.fingerprint
    assert cached.records == manifest.records


def test_mixed_labels_are_rejected(tmp_path: Path) -> None:
    """验证同一训练图像包含两种域类别时直接报错。"""
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_dataset(dataset_dir)
    label_path = dataset_dir / "labels" / "train" / "mixed_train.txt"
    _write_image(dataset_dir / "images" / "train" / "mixed_train.jpg", (30, 120, 200))
    label_path.write_text("0 0.5 0.5 0.2 0.2\n24 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="同时包含 PAN/RGB"):
        prepare_dual_dataset(dataset_dir, cache_dir, rebuild=True)


def test_test_split_routing_does_not_read_labels(tmp_path: Path) -> None:
    """验证测试 split 按像素路由，即使测试标签与像素模态冲突也不改路由。"""
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_dataset(dataset_dir)
    (dataset_dir / "labels" / "test" / "pan_test.txt").write_text(
        "24 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    manifest = prepare_dual_dataset(dataset_dir, cache_dir, rebuild=True)
    assert [Path(item["image"]).stem for item in manifest.records["test"]["pan"]] == ["pan_test"]


def test_public_class_sets_are_disjoint_and_complete() -> None:
    """验证双模型类别集合覆盖 SHWX 25 类且无交集。"""
    assert set(PAN_CLASSES).isdisjoint(RGB_CLASSES)
    assert set(PAN_CLASSES) | set(RGB_CLASSES) == set(range(25))


def test_global_predictions_are_filtered_without_remapping() -> None:
    """验证全局类别权重只保留路由模态的类别且不改变类别索引。"""
    records = [
        eval_lib.BoxRecord("sample", 0, (1.0, 1.0, 3.0, 3.0), 0.9),
        eval_lib.BoxRecord("sample", 3, (2.0, 2.0, 4.0, 4.0), 0.8),
        eval_lib.BoxRecord("sample", 24, (3.0, 3.0, 5.0, 5.0), 0.7),
    ]

    filtered = test_dual_shwx._filter_global_records(records, set(PAN_CLASSES))

    assert [record.class_id for record in filtered] == [0, 3]


def test_merged_predictions_save_fp_fn_visualizations(tmp_path: Path) -> None:
    """验证合并后的全局类别预测复用统一的 FP/FN 可视化。"""
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path, (80, 80, 80))
    output_dir = tmp_path / "output"
    dataset = SimpleNamespace(
        num_classes=1,
        vehicle_class_ids=frozenset(),
        class_names={0: "target"},
    )
    gt_records = [
        eval_lib.BoxRecord("sample", 0, (1.0, 1.0, 3.0, 3.0)),
    ]
    pred_records = [
        eval_lib.BoxRecord("sample", 0, (5.0, 5.0, 7.0, 7.0), 0.9),
    ]

    test_dual_shwx._save_fp_fn_visualizations(
        dataset,
        gt_records,
        pred_records,
        [image_path],
        output_dir,
        save_fp_fn=True,
    )

    assert (output_dir / "FP" / "target" / "sample.jpg").exists()
    assert (output_dir / "FN" / "target" / "sample.jpg").exists()
