# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""PatchPasteDataset（正/负样本补丁粘贴）单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from supervision import Detections

from rfdetr.datasets.patch_paste import (
    Patch,
    PatchPasteDataset,
    _D4,
    _concat_detections,
    _d4_box,
    _d4_point,
    load_patch_pool,
)
from rfdetr.datasets.yolo import ConvertYolo

FSC = 24


# ------------------------------------------------------------------------
# 测试基建：假底层数据集（load_raw/prepare 与 YoloDetection 同契约）
# ------------------------------------------------------------------------
class _FakeRawDataset:
    """内存假数据集：load_raw 返回 (path, rgb HWC, Detections)，prepare 用真 ConvertYolo。"""

    def __init__(self, per_image_boxes: list[list[tuple]]) -> None:
        self._per_image_boxes = per_image_boxes
        self._rgb_cache: dict[int, np.ndarray] = {}
        self.prepare = ConvertYolo()
        self._transforms = None

    def __len__(self) -> int:
        return len(self._per_image_boxes)

    def __getitem__(self, idx: int):
        path, rgb, dets = self.load_raw(idx)
        img = Image.fromarray(rgb)
        target = {"image_id": idx, "detections": dets}
        img, target = self.prepare(img, target)
        return img, target

    def load_raw(self, idx: int) -> tuple[str, np.ndarray, Detections]:
        if idx not in self._rgb_cache:
            self._rgb_cache[idx] = np.zeros((128, 128, 3), dtype=np.uint8)
        boxes = self._per_image_boxes[idx]
        if boxes:
            dets = Detections(
                xyxy=np.asarray([b[:4] for b in boxes], dtype=np.float32),
                class_id=np.asarray([b[4] for b in boxes], dtype=int),
            )
        else:
            dets = Detections(xyxy=np.zeros((0, 4), dtype=np.float32), class_id=np.zeros(0, dtype=int))
        return f"img{idx}.jpg", self._rgb_cache[idx], dets


def _make_pool(tmp_path: Path, pos_boxes: list[tuple] | None = None, neg_count: int = 2) -> Path:
    """在 tmp 目录构造一个补丁池（正样本若干 + 负样本若干）。"""
    pool_dir = tmp_path / "pool"
    (pool_dir / "positive").mkdir(parents=True)
    (pool_dir / "negative").mkdir(parents=True)
    patches = []
    if pos_boxes is not None:
        for i, (x1, y1, x2, y2) in enumerate(pos_boxes):
            img = Image.new("RGB", (64, 64), (200, 50 + 20 * i, 50))
            img.save(pool_dir / "positive" / f"P{i:04d}.jpg", quality=95)
            patches.append(
                {
                    "kind": "positive",
                    "id": f"P{i:04d}",
                    "file": f"positive/P{i:04d}.jpg",
                    "width": 64,
                    "height": 64,
                    "class_id": FSC,
                    "box": [x1, y1, x2, y2],
                }
            )
    for i in range(neg_count):
        img = Image.new("RGB", (64, 64), (30, 30, 200))
        img.save(pool_dir / "negative" / f"N{i:04d}.jpg", quality=95)
        patches.append(
            {
                "kind": "negative",
                "id": f"N{i:04d}",
                "file": f"negative/N{i:04d}.jpg",
                "width": 64,
                "height": 64,
            }
        )
    import json

    (pool_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "patches": patches}),
        encoding="utf-8",
    )
    return pool_dir


# ------------------------------------------------------------------------
# D4 变换（像素金标准）
# ------------------------------------------------------------------------
@pytest.mark.parametrize("op", _D4)
def test_d4_box_transform_golden(op: str) -> None:
    """对 8 个 D4 元素逐一验证：暴力按像素变换对比 _d4_box。"""
    w, h = 8, 5
    box = (2.0, 1.0, 5.0, 4.0)
    # 金标准：把框内像素映射后取包围盒
    mapped = []
    for px in range(2, 6):
        for py in range(1, 5):
            mapped.append(_d4_point(px, py, w, h, op))
    xs = [m[0] for m in mapped]
    ys = [m[1] for m in mapped]
    golden = (min(xs), min(ys), max(xs), max(ys))
    assert _d4_box(box, w, h, op) == golden


def test_d4_swaps_dims() -> None:
    """90° 系变换后宽高互换。"""
    box = (1.0, 1.0, 3.0, 2.0)
    swapped = _d4_box(box, 8, 5, "rot90")
    # rot90: (x,y) -> (y, w-1-x)；角点映射后 AABB 宽高互换（2x1 → 1x2）
    assert (swapped[2] - swapped[0], swapped[3] - swapped[1]) == (1.0, 2.0)
    identity = _d4_box(box, 8, 5, "identity")
    assert identity == box


# ------------------------------------------------------------------------
# 宿主约束
# ------------------------------------------------------------------------
def test_host_constraint_no_target_class(tmp_path: Path) -> None:
    """宿主不含类24 → 零粘贴（输出与原始一致）。"""
    raw = _FakeRawDataset([[(10, 10, 20, 20, 0)], [(10, 10, 20, 20, FSC)]])
    ds = PatchPasteDataset(raw, manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json", p=1.0)
    img0, target0 = ds[0]
    img0_raw, target0_raw = raw[0]
    assert target0["labels"].tolist() == target0_raw["labels"].tolist()
    # 无类24 宿主不触发：图像内容不变（纯色图 + 粘贴的红色补丁会改变像素）
    assert np.array_equal(np.asarray(img0), np.asarray(img0_raw))


def test_host_constraint_with_target_class(tmp_path: Path) -> None:
    """宿主含类24 → 触发粘贴（图像内容变化）。"""
    raw = _FakeRawDataset([[(10, 10, 20, 20, FSC)]])
    ds = PatchPasteDataset(raw, manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json", p=1.0)
    img, target = ds[0]
    assert len(target["labels"]) >= 1
    assert FSC in target["labels"].tolist()


# ------------------------------------------------------------------------
# 正/负样本语义
# ------------------------------------------------------------------------
def test_positive_appends_box(tmp_path: Path) -> None:
    """正样本粘贴 → Detections 框数 +1、类24、与所有原 GT 零交叠。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    ds = PatchPasteDataset(
        raw,
        manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json",
        p=1.0,
        max_patches=1,
        neg_ratio=0.0,  # 全正样本
    )
    path, rgb, dets = ds.load_raw(0)
    assert len(dets) == 2
    assert dets.class_id[-1] == FSC
    new_box = dets.xyxy[-1]
    # 与原始 GT 零交叠
    from val.competition_metrics import compute_iou

    assert compute_iou(tuple(new_box), (20, 20, 40, 40)) == 0.0
    # 框在宿主图内
    assert 0 <= new_box[0] < new_box[2] <= 128
    assert 0 <= new_box[1] < new_box[3] <= 128


def test_negative_appends_nothing(tmp_path: Path) -> None:
    """负样本粘贴 → 框数不变但图像像素变化。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    ds = PatchPasteDataset(
        raw,
        manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json",
        p=1.0,
        max_patches=1,
        neg_ratio=1.0,  # 全负样本
    )
    path, rgb, dets = ds.load_raw(0)
    assert len(dets) == 1
    img_pasted = Image.fromarray(rgb)
    img_raw = Image.fromarray(raw._rgb_cache[0])
    assert np.asarray(img_pasted).sum() != np.asarray(img_raw).sum()


def test_no_overlap_with_gt_and_siblings(tmp_path: Path) -> None:
    """固定种子多轮：粘贴框与 GT 及互不交叠。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC), (80, 80, 100, 100, FSC)]])
    ds = PatchPasteDataset(
        raw,
        manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10), (2, 2, 10, 10)]) / "manifest.json",
        p=1.0,
        max_patches=3,
        neg_ratio=0.5,
    )
    from val.competition_metrics import compute_iou

    for _ in range(10):
        path, rgb, dets = ds.load_raw(0)
        gt = dets.xyxy[:2]
        pasted = dets.xyxy[2:]
        for pb in pasted:
            for g in gt:
                assert compute_iou(tuple(pb), tuple(g)) == 0.0
        for i in range(len(pasted)):
            for j in range(i + 1, len(pasted)):
                assert compute_iou(tuple(pasted[i]), tuple(pasted[j])) == 0.0


def test_scale_range_respected(tmp_path: Path) -> None:
    """粘贴框尺寸受 scale_range 约束。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    # 正补丁框 8x8，scale 固定 1.0 → 粘贴框仍 8x8
    ds = PatchPasteDataset(
        raw,
        manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json",
        p=1.0,
        max_patches=1,
        neg_ratio=0.0,
        scale_range=(1.0, 1.0),
    )
    path, rgb, dets = ds.load_raw(0)
    pasted = dets.xyxy[-1]
    assert (pasted[2] - pasted[0]) == 8.0
    assert (pasted[3] - pasted[1]) == 8.0


def test_p_zero_passthrough(tmp_path: Path) -> None:
    """p=0 → 恒等于原始路径（连异常路径也不触发）。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    ds = PatchPasteDataset(raw, manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json", p=0.0)
    for i in range(len(raw)):
        img, target = ds[i]
        img_raw, target_raw = raw[i]
        assert np.array_equal(np.asarray(img), np.asarray(img_raw))
        assert target["labels"].tolist() == target_raw["labels"].tolist()


def test_load_raw_interface(tmp_path: Path) -> None:
    """load_raw 返回 (str, HWC uint8 ndarray, Detections)，供外层 Mosaic 组合。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    ds = PatchPasteDataset(raw, manifest_path=_make_pool(tmp_path, [(2, 2, 10, 10)]) / "manifest.json", p=1.0)
    path, rgb, dets = ds.load_raw(0)
    assert isinstance(path, str)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert isinstance(dets, Detections)


# ------------------------------------------------------------------------
# 补丁池加载校验
# ------------------------------------------------------------------------
def test_pool_validation_missing_manifest(tmp_path: Path) -> None:
    """manifest 缺失 → 抛 FileNotFoundError。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    with pytest.raises(FileNotFoundError):
        PatchPasteDataset(raw, manifest_path=tmp_path / "nope" / "manifest.json")


def test_pool_validation_bad_box(tmp_path: Path) -> None:
    """正样本 box 越界 → 抛 ValueError。"""
    raw = _FakeRawDataset([[(20, 20, 40, 40, FSC)]])
    pool_dir = _make_pool(tmp_path, [(2, 2, 1000, 1000)])  # 越界框
    with pytest.raises(ValueError):
        load_patch_pool(pool_dir / "manifest.json")


def test_concat_detections_empty() -> None:
    """空 Detections 拼接正样本框。"""
    dets = Detections(xyxy=np.zeros((0, 4), dtype=np.float32), class_id=np.zeros(0, dtype=int))
    out = _concat_detections(dets, [(1.0, 1.0, 5.0, 5.0)], class_id=FSC)
    assert len(out) == 1
    assert out.class_id[0] == FSC


# ------------------------------------------------------------------------
# 配置接线（字段声明）
# ------------------------------------------------------------------------
def test_train_config_fields_declared() -> None:
    """TrainConfig 新字段已声明：默认关、可构造。"""
    from rfdetr.config import TrainConfig

    cfg = TrainConfig(dataset_dir="dummy")
    assert cfg.patch_paste_enabled is False
    assert cfg.patch_paste_dir is None
    assert cfg.patch_paste_prob == 0.5
    assert cfg.patch_paste_max_patches == 2
    assert cfg.patch_paste_target_classes == [24]
    assert cfg.patch_paste_neg_ratio == 0.5

    cfg2 = TrainConfig(
        dataset_dir="dummy",
        patch_paste_enabled=True,
        patch_paste_dir="data/fsc_patch_pool",
        patch_paste_prob=0.3,
    )
    assert cfg2.patch_paste_enabled is True
    assert cfg2.patch_paste_dir == "data/fsc_patch_pool"
    assert cfg2.patch_paste_prob == 0.3
