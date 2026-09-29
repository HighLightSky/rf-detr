"""验证 DIOR query selection 统计口径。"""

import torch

from scripts.analysis.dior_query_selection import measure_instances


def test_measure_instances_separates_coverage_and_selection() -> None:
    """正 proposal 缺失与未进入 top-K 应分开统计。"""
    proposal_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.4, 0.4, 0.6, 0.6],
            [0.8, 0.8, 1.0, 1.0],
        ]
    )
    scores = torch.tensor([0.2, 0.9, 0.8])
    gt_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.4, 0.4, 0.6, 0.6],
            [0.0, 0.8, 0.2, 1.0],
        ]
    )

    rows = measure_instances(proposal_boxes, scores, gt_boxes, ks=(1, 2), iou_thresholds=(0.5,))

    assert [row["covered_iou50"] for row in rows] == [True, True, False]
    assert [row["selected_iou50_k1"] for row in rows] == [False, True, False]
    assert [row["selected_iou50_k2"] for row in rows] == [False, True, False]
    assert [row["best_positive_rank_iou50"] for row in rows] == [3, 1, None]


def test_measure_instances_counts_any_matching_proposal() -> None:
    """同一 GT 只要有一个匹配 proposal 被选中即算命中。"""
    proposal_boxes = torch.tensor(
        [
            [0.1, 0.1, 0.3, 0.3],
            [0.1, 0.1, 0.3, 0.3],
            [0.7, 0.7, 0.9, 0.9],
        ]
    )
    scores = torch.tensor([0.1, 0.8, 0.9])
    gt_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]])

    rows = measure_instances(proposal_boxes, scores, gt_boxes, ks=(1, 2), iou_thresholds=(0.5,))

    assert rows[0]["selected_iou50_k1"] is False
    assert rows[0]["selected_iou50_k2"] is True
    assert rows[0]["best_positive_rank_iou50"] == 2
