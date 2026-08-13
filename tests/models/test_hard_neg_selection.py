# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本选择器（select_hard_negatives_for_image）的单元测试。

不依赖 GPU 与网络，用合成单图数据（Q=10、D=4、C=2）验证：
- IoU 带 [0.0, 0.3] 过滤（纯背景与低 IoU 候选入选，高 IoU/重复检测排除）。
- Hungarian matched query 永不入选（即使是难例带内的最高分）。
- top-k 按目标类别集合内最大 logit 降序、忽略 background 列、stable 并列取小索引。
- score_thresh 下限过滤、空 GT 不崩溃、全 matched 无候选。
- 返回 query 索引而非 detached 特征，供训练侧直接对虚警 query 反传。
"""

from __future__ import annotations

import torch

from rfdetr.sscl.hard_neg_selection import (
    IOU_BAND_HIGH,
    IOU_BAND_LOW,
    select_hard_negatives_for_image,
)
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou

# GT 框（归一化 cxcywh）
_GT_BOX = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
# 查询框：索引 0 = 纯背景（IoU 0），索引 1 = 带内（IoU 0.25），索引 2 = 重复检测（IoU 1.0）
_QUERY_BOXES = torch.tensor(
    [
        [0.1, 0.1, 0.05, 0.05],  # q0: IoU = 0.0 ∈ [0.0, 0.3]（纯背景，带内）
        [0.5, 0.5, 0.1, 0.1],  # q1: IoU = 0.25 ∈ [0.0, 0.3]（难例带内）
        [0.5, 0.5, 0.2, 0.2],  # q2: IoU = 1.0 > 0.3（真实目标的重复检测，带外）
        [0.1, 0.9, 0.1, 0.1],  # q3: IoU = 0.0（远处背景）
        [0.9, 0.9, 0.1, 0.1],  # q4: IoU = 0.0（远处背景）
        [0.2, 0.2, 0.1, 0.1],  # q5: IoU = 0.0（远处背景）
        [0.8, 0.2, 0.1, 0.1],  # q6: IoU = 0.0（远处背景）
        [0.2, 0.8, 0.1, 0.1],  # q7: IoU = 0.0（远处背景）
        [0.7, 0.7, 0.1, 0.1],  # q8: IoU = 0.0（远处背景）
        [0.3, 0.3, 0.1, 0.1],  # q9: IoU = 0.0（远处背景）
    ]
)


def _logits(scores: list[float], num_q: int = 10, num_cls: int = 2) -> torch.Tensor:
    """按给定前景分构造 [Q, C+1] logits（background 列全置高分，验证被忽略）。

    前景分同时写入全部前景列：最大前景 logit = max(col0, col1) = 给定分数，
    保证负分数真正为负（若只写一列，另一列恒 0 会让分数被抬升到 0）。
    """
    logits = torch.zeros(num_q, num_cls + 1)
    for i, s in enumerate(scores):
        logits[i, :num_cls] = s
    logits[:, -1] = 9.0  # background 列高分——选择时必须忽略
    return logits


def _assert_iou_band(iou: float) -> None:
    """断言已知 IoU 值确实落在设计好的带内/带外（防止测试自身假设错误）。"""
    if iou != 0.0 and iou != 1.0:
        assert IOU_BAND_LOW <= iou <= IOU_BAND_HIGH


class TestHardNegSelection:
    """难例选择器的采样规则。"""

    def test_iou_band_filtering(self) -> None:
        """带内（IoU 0.0 纯背景与 0.25）候选入选，重复检测（1.0）排除。"""
        # q0/q1 带内且分数高 → 入选；q2 分数最高（6.0）但 IoU 1.0 必须被带挡住
        scores = [5.0, 4.0, 6.0] + [-5.0] * 7
        logits = _logits(scores)
        # 先验证测试数据自身的 IoU 假设
        iou, _ = box_iou(box_cxcywh_to_xyxy(_QUERY_BOXES), box_cxcywh_to_xyxy(_GT_BOX))
        assert iou[0, 0].item() == 0.0
        assert 0.0 <= iou[1, 0].item() <= 0.3
        assert iou[2, 0].item() == 1.0

        hn_idx, stats = select_hard_negatives_for_image(
            logits, _QUERY_BOXES, _GT_BOX, torch.tensor([], dtype=torch.long), top_k=3
        )
        assert torch.equal(hn_idx, torch.tensor([0, 1]))
        assert stats["n_selected"] == 2.0
        assert stats["n_band"] == 9.0  # 带内：q0、q1、q3-q9（IoU 0.0），q2 排除
        assert stats["n_unmatched"] == 10.0
        assert stats["score_mean"] == 4.5
        assert 0.0 <= stats["iou_mean"] <= 0.3

    def test_matched_excluded(self) -> None:
        """Matched query 即使带内且最高分也永不入选。"""
        # q1 最高分（9.0）且带内但 matched；其余候选分数全部 < score_thresh（-1.0）被过滤
        scores = [-1.0, 9.0, -1.0] + [-1.0] * 7
        logits = _logits(scores)
        hn_idx, stats = select_hard_negatives_for_image(logits, _QUERY_BOXES, _GT_BOX, torch.tensor([1]), top_k=3)
        assert hn_idx.shape[0] == 0  # q1 被 matched 排除后无其他候选
        assert stats["n_selected"] == 0.0

    def test_topk_order_ignores_background_and_stable(self) -> None:
        """Top-k 按最大前景 logit 降序（忽略 background 列），并列时取小索引。"""
        # 构造两个带内候选：q1 前景分 4.0，q3 前景分 5.0（q3 框与 GT 部分
        # 重叠，IoU = 0.25 落在 [0.0, 0.3] 带内）
        boxes = _QUERY_BOXES.clone()
        boxes[3] = torch.tensor([0.5, 0.5, 0.1, 0.1])
        iou, _ = box_iou(box_cxcywh_to_xyxy(boxes[3:4]), box_cxcywh_to_xyxy(_GT_BOX))
        assert IOU_BAND_LOW <= iou.item() <= IOU_BAND_HIGH
        # q1 前景 4.0、q3 前景 5.0；两者 background 列均为 9.0（必须被忽略）
        scores = [0.0, 4.0, 0.0, 5.0] + [0.0] * 6
        logits = _logits(scores)
        hn_idx, _ = select_hard_negatives_for_image(
            logits, boxes, _GT_BOX, torch.tensor([], dtype=torch.long), top_k=1
        )
        assert torch.equal(hn_idx, torch.tensor([3]))  # 分数更高者入选，background 9.0 未干扰

    def test_target_classes_score_only(self) -> None:
        """只按 target_classes 中的类别分数排序，非目标类别高分不应入选优先级。"""
        logits = torch.full((10, 4), -5.0)
        logits[:, -1] = 9.0
        logits[0, 2] = 8.0
        logits[1, 0] = 4.0
        logits[3, 1] = 5.0
        hn_idx, _ = select_hard_negatives_for_image(
            logits,
            _QUERY_BOXES,
            _GT_BOX,
            torch.tensor([], dtype=torch.long),
            top_k=2,
            target_classes=[0, 1],
        )
        assert torch.equal(hn_idx, torch.tensor([3, 1]))

    def test_score_thresh(self) -> None:
        """低于 score_thresh 的候选不选。"""
        scores = [0.0, 0.5, 0.0] + [0.0] * 7  # q1 前景分 0.5
        logits = _logits(scores)
        hn_idx, stats = select_hard_negatives_for_image(
            logits, _QUERY_BOXES, _GT_BOX, torch.tensor([], dtype=torch.long), top_k=3, score_thresh=1.0
        )
        assert hn_idx.shape[0] == 0
        assert stats["n_selected"] == 0.0

    def test_empty_gt_no_crash(self) -> None:
        """空 GT 框：无候选、不崩溃、统计为 0。"""
        hn_idx, stats = select_hard_negatives_for_image(
            _logits([-1.0] * 10),
            _QUERY_BOXES,
            torch.zeros(0, 4),
            torch.tensor([], dtype=torch.long),
            top_k=3,
        )
        assert hn_idx.shape[0] == 0
        # 带下界 0.0：空 GT 时 max_iou 全为 0，10 个 query 都算带内；
        # 难例数仍为 0（分数全部 < score_thresh）
        assert stats["n_band"] == 10.0

    def test_all_matched_no_candidates(self) -> None:
        """全部 query 都被 matched → 无难例。"""
        hn_idx, stats = select_hard_negatives_for_image(
            _logits([4.0] * 10),
            _QUERY_BOXES,
            _GT_BOX,
            torch.arange(10),
            top_k=3,
        )
        assert hn_idx.shape[0] == 0
        assert stats["n_unmatched"] == 0.0
