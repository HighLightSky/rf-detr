# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切评估脚本测试：BoxRecord → COCO dict 转换与 pycocotools AP 计算。

覆盖 ``eval_large_cut.py`` 的核心纯函数：bbox 转换、COCO dict 结构与 ``compute_coco_ap`` 数值正确性。
"""

import pytest

from scripts.eval_large_cut import (
    boxes_to_coco_gt,
    boxes_to_coco_pred,
    compute_coco_ap,
)
from val.competition_metrics import BoxRecord


def _make_records() -> tuple[list[BoxRecord], list[BoxRecord], list[str], dict]:
    """构造 2 张 100x100 图的小场景：1 个完美匹配 + 1 个错位框 + 1 个漏检。

    - 图 a：gt [0,0,10,10]，pred [0,0,10,10]（score 0.9）→ 完美匹配；
    - 图 b：gt [0,0,10,10]，pred [20,20,30,30]（score 0.8）→ IoU=0 错位框，
      另一个 gt 漏检。

    返回 ``(gt_records, pred_records, image_ids, image_sizes)``。
    """
    gt_records = [
        BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
        BoxRecord(image_id="b", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
    ]
    pred_records = [
        BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.9),
        BoxRecord(image_id="b", class_id=0, xyxy=(20.0, 20.0, 30.0, 30.0), score=0.8),
    ]
    image_ids = ["a", "b"]
    image_sizes = {"a": (100, 100), "b": (100, 100)}
    return gt_records, pred_records, image_ids, image_sizes


class TestBoxesToCocoGt:
    """``boxes_to_coco_gt`` 真值转换。"""

    def test_bbox_format_xyxy_to_xywh(self):
        """Xyxy 转换为 COCO 标准 ``[x, y, w, h]``。"""
        gt_records, _, image_ids, image_sizes = _make_records()
        gt = boxes_to_coco_gt(gt_records, image_ids, image_sizes)

        assert len(gt["images"]) == 2
        assert len(gt["annotations"]) == 2
        ann = gt["annotations"][0]
        assert ann["bbox"] == [0.0, 0.0, 10.0, 10.0]
        assert ann["area"] == pytest.approx(100.0)
        assert ann["iscrowd"] == 0
        assert ann["category_id"] == 0

    def test_categories_single_class(self):
        """Categories 为单类 sample。"""
        gt_records, _, image_ids, image_sizes = _make_records()
        gt = boxes_to_coco_gt(gt_records, image_ids, image_sizes)
        assert gt["categories"] == [{"id": 0, "name": "sample"}]

    def test_images_have_dimensions(self):
        """Images 条目含宽高（来自 image_sizes）。"""
        gt_records, _, image_ids, image_sizes = _make_records()
        gt = boxes_to_coco_gt(gt_records, image_ids, image_sizes)
        image_a = next(img for img in gt["images"] if img["id"] == "a")
        assert image_a["width"] == 100
        assert image_a["height"] == 100


class TestBoxesToCocoPred:
    """``boxes_to_coco_pred`` 预测转换。"""

    def test_contains_score(self):
        """预测 ann 携带 score 字段。"""
        _, pred_records, image_ids, _ = _make_records()
        pred = boxes_to_coco_pred(pred_records, image_ids)
        assert len(pred["annotations"]) == 2
        scores = {ann["image_id"]: ann["score"] for ann in pred["annotations"]}
        assert scores["a"] == pytest.approx(0.9)
        assert scores["b"] == pytest.approx(0.8)


class TestComputeCocoAp:
    """``compute_coco_ap`` 数值正确性。"""

    def test_ap50_perfect_match_wins(self):
        """只有完美匹配（IoU=1.0）被命中时 AP50 = 1.0。"""
        # 2 张图各 1 个 gt、各 1 个完美匹配 pred（无漏检无错位）
        gt_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
            BoxRecord(image_id="b", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
        ]
        pred_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.9),
            BoxRecord(image_id="b", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.8),
        ]
        gt = boxes_to_coco_gt(gt_records, ["a", "b"], {"a": (100, 100), "b": (100, 100)})
        pred = boxes_to_coco_pred(pred_records, ["a", "b"])
        metrics = compute_coco_ap(gt, pred)

        assert metrics["AP50"] == pytest.approx(1.0)
        assert metrics["AP50:95"] == pytest.approx(1.0)

    def test_missing_gt_pulls_ap_down(self):
        """漏检 1/2 的 GT 时 AP50 ≈ 0.505（COCO 101 点插值：recall 上限 0.5）。"""
        gt_records, pred_records, image_ids, image_sizes = _make_records()
        gt = boxes_to_coco_gt(gt_records, image_ids, image_sizes)
        pred = boxes_to_coco_pred(pred_records, image_ids)
        metrics = compute_coco_ap(gt, pred)

        # 高分完美匹配（0.9）排在错位框（0.8）之前：recall 0~0.5 处 precision=1.0，
        # recall>0.5 处为 0 → AP = 51/101 ≈ 0.505（COCO AP 的标准语义）；
        # 完美匹配框在所有 IoU 档命中、错位框全档不命中 → AP50:95 与 AP50 相同
        assert metrics["AP50"] == pytest.approx(0.50495, abs=1e-3)
        assert metrics["AP50:95"] == pytest.approx(0.50495, abs=1e-3)

    def test_loose_box_pulls_down_high_iou_ap(self):
        """中等 IoU 框（命中 0.5 档、不命中 0.75 档）拉低 AP75 但不影响 AP50。"""
        # 图 a：完美匹配；图 b：gt [0,0,10,10]，pred [2,2,11,11]（IoU=64/117≈0.547）
        gt_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
            BoxRecord(image_id="b", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0)),
        ]
        pred_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.9),
            BoxRecord(image_id="b", class_id=0, xyxy=(2.0, 2.0, 11.0, 11.0), score=0.8),
        ]
        gt = boxes_to_coco_gt(gt_records, ["a", "b"], {"a": (100, 100), "b": (100, 100)})
        pred = boxes_to_coco_pred(pred_records, ["a", "b"])
        metrics = compute_coco_ap(gt, pred)

        # 两个框都命中 0.5 档（IoU 0.547 ≥ 0.5）→ AP50 = 1.0
        assert metrics["AP50"] == pytest.approx(1.0)
        # 中等框在 0.75 档计为 FP（recall 上限 0.5）→ AP75 ≈ 0.505
        assert metrics["AP75"] == pytest.approx(0.50495, abs=1e-3)
        assert metrics["AP75"] < metrics["AP50"]

    def test_ap90_drops_when_boxes_loose(self):
        """放宽的框（IoU<0.9 但 >=0.75）在 AP90 档计为 FP：AP90 < AP75。"""
        # 图 a：gt [0,0,100,100]，pred [5,5,95,95]（IoU=(90^2)/(100^2+90^2-90^2)=0.81 → 命中 0.75 档，不命中 0.9 档）
        gt_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(0.0, 0.0, 100.0, 100.0)),
        ]
        pred_records = [
            BoxRecord(image_id="a", class_id=0, xyxy=(5.0, 5.0, 95.0, 95.0), score=0.9),
        ]
        image_ids = ["a"]
        image_sizes = {"a": (200, 200)}
        gt = boxes_to_coco_gt(gt_records, image_ids, image_sizes)
        pred = boxes_to_coco_pred(pred_records, image_ids)
        metrics = compute_coco_ap(gt, pred)

        assert metrics["AP75"] == pytest.approx(1.0)
        assert metrics["AP90"] == pytest.approx(0.0)
        assert metrics["AP50:95"] < 1.0
