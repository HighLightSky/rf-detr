"""FSC DINOv3 训练样本图集的基础测试。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.refinement.visualize_fsc_dinov3_samples import _candidate_overview, select_rows


def test_select_rows_uses_training_split_and_label() -> None:
    """只返回训练来源中对应内部划分与标签的候选。"""
    payload = {"candidates": [
        {"image": "/tmp/a.jpg", "prediction_index": 0, "split": "train", "label": 1},
        {"image": "/tmp/b.jpg", "prediction_index": 1, "split": "train", "label": 0},
        {"image": "/tmp/c.jpg", "prediction_index": 2, "split": "val", "label": 1},
    ]}

    rows = select_rows(payload, "train", 1, 10)

    assert all(row["split"] == "train" for row in rows)
    assert all(row["label"] == 1 for row in rows)


def test_select_rows_is_deterministic() -> None:
    """同一缓存的图集抽样顺序保持稳定。"""
    payload = {"candidates": [
        {"image": f"/tmp/{index}.jpg", "prediction_index": index, "split": "train", "label": 0}
        for index in range(20)
    ]}

    assert select_rows(payload, "train", 0, 5) == select_rows(payload, "train", 0, 5)


def test_candidate_overview_preserves_edge_geometry() -> None:
    """边缘候选的概览图应使用正方形窗口，不能因拉伸移动目标位置。"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 20, 0] = 255

    overview = _candidate_overview(Image.fromarray(image), [0.0, 10.0, 20.0, 30.0])

    red_column = int(np.asarray(overview)[..., 0].mean(axis=0).argmax())
    assert 74 <= red_column <= 76
