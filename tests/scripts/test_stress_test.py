"""高负荷实验统计逻辑测试。"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from scripts.evaluation.stress_test import _prediction_digest, _summary


class _Prediction:
    """最小化的预测对象替身。"""

    def __init__(self, offset: float = 0.0) -> None:
        self.xyxy = np.asarray([[offset, 1.0, 10.0, 20.0]], dtype=np.float32)
        self.confidence = np.asarray([0.75], dtype=np.float32)
        self.class_id = np.asarray([3], dtype=np.int64)


def test_prediction_digest_is_deterministic() -> None:
    """相同预测对象应产生相同摘要。"""
    assert _prediction_digest(_Prediction()) == _prediction_digest(_Prediction())


def test_prediction_digest_changes_when_box_changes() -> None:
    """框坐标变化应改变摘要。"""
    assert _prediction_digest(_Prediction()) != _prediction_digest(_Prediction(1.0))


def test_summary_reports_peak_and_memory_drift() -> None:
    """汇总应计算连续斜率并识别成功批量和 OOM。"""
    rows = [
        {
            "experiment": "long_run",
            "iteration": 0,
            "gpu_allocated_mb": 100.0,
            "gpu_peak_mb": 110.0,
            "rss_mb": 200.0,
            "digests": "a",
        },
        {
            "experiment": "long_run",
            "iteration": 1,
            "gpu_allocated_mb": 101.0,
            "gpu_peak_mb": 111.0,
            "rss_mb": 201.0,
            "digests": "a",
        },
        {"experiment": "peak_batch", "batch_size": 1, "status": "ok"},
        {"experiment": "peak_batch", "batch_size": 2, "status": "ok"},
        {"experiment": "peak_batch", "batch_size": 4, "status": "oom"},
        {"experiment": "constrained", "memory_fraction": 0.5, "processed": 3, "total": 3, "status": "ok"},
    ]
    args = Namespace(checkpoint="model.pth", device="cpu", image_count=3)
    summary = _summary(rows, args)
    assert summary["peak_max_success_batch"] == 2
    assert summary["peak_first_oom_batch"] == 4
    assert summary["long_gpu_slope_mb_per_iter"] == pytest.approx(1.0)
    assert summary["constrained"][0]["processed"] == 3
