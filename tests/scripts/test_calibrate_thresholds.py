# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""calibrate_thresholds.py 纯函数的单元测试（不触发推理）。

只测离线选择逻辑（字典序选优），推理管线由 E0 实跑验证。
"""

import pytest

from scripts.evaluation.calibrate_thresholds import _select_best_candidate


def _mk(recall: float, fdr: float) -> dict[str, float]:
    """构造总宏指标字典。"""
    return {"recall": recall, "fdr": fdr, "precision": 1.0 - fdr}


class TestSelectBestCandidate:
    """测试逐类阈值候选点的字典序选优规则。"""

    def test_prefers_gate_ok_with_lowest_fdr(self):
        """能同时过门槛时，选 FDR 最低的点（即使 Recall 略低）。"""
        candidates = [
            ({"a": 0.3}, _mk(0.86, 0.195)),  # 恰好过门槛
            ({"a": 0.4}, _mk(0.87, 0.150)),  # 过门槛且 FDR 最低
            ({"a": 0.5}, _mk(0.88, 0.220)),  # Recall 最高但 FDR 超门槛
        ]

        thresholds, total = _select_best_candidate(candidates)

        assert thresholds == {"a": 0.4}
        assert total["fdr"] == pytest.approx(0.150)

    def test_prefers_highest_recall_within_fdr_gate(self):
        """门槛不可同时满足时，先压住 FDR<=0.195 再选 Recall 最高。"""
        candidates = [
            ({"a": 0.2}, _mk(0.80, 0.19)),  # FDR 过关但 Recall 低
            ({"a": 0.3}, _mk(0.83, 0.19)),  # FDR 过关且 Recall 更高
            ({"a": 0.4}, _mk(0.86, 0.21)),  # Recall 最高但 FDR 超
        ]

        thresholds, total = _select_best_candidate(candidates)

        assert thresholds == {"a": 0.3}
        assert total["recall"] == pytest.approx(0.83)

    def test_falls_back_to_recall_within_loose_fdr(self):
        """FDR<=0.195 全无解时，退到 FDR<=0.25 内选 Recall 最高。"""
        candidates = [
            ({"a": 0.2}, _mk(0.82, 0.24)),
            ({"a": 0.3}, _mk(0.85, 0.23)),  # FDR<=0.25 内 Recall 最高
            ({"a": 0.4}, _mk(0.86, 0.30)),
        ]

        thresholds, total = _select_best_candidate(candidates)

        assert thresholds == {"a": 0.3}

    def test_last_resort_highest_recall(self):
        """全部超 0.25 时兜底选 Recall 最高。"""
        candidates = [
            ({"a": 0.2}, _mk(0.80, 0.30)),
            ({"a": 0.3}, _mk(0.85, 0.35)),
        ]

        thresholds, total = _select_best_candidate(candidates)

        assert thresholds == {"a": 0.3}
