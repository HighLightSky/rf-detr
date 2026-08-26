"""人工 FSC 复核候选标签映射测试。"""

from __future__ import annotations

from scripts.refinement.build_fsc_manual_refinement_cache import label_candidate


def test_label_candidate_marks_full_fsc_positive() -> None:
    """与完整 FSC 框充分重叠的候选为正例。"""
    assert label_candidate((0.1, 0.1, 0.5, 0.5), [(0, (0.1, 0.1, 0.5, 0.5))]) == 1


def test_label_candidate_ignores_partial_fsc() -> None:
    """落在 FSC 内但只覆盖局部的候选不得成为负例。"""
    assert label_candidate((0.1, 0.1, 0.2, 0.2), [(0, (0.1, 0.1, 0.7, 0.7))]) is None


def test_label_candidate_marks_manual_confuser_negative() -> None:
    """与人工车辆或固定设施干扰物匹配的候选为负例。"""
    assert label_candidate((0.1, 0.1, 0.4, 0.4), [(1, (0.1, 0.1, 0.4, 0.4))]) == 0
