# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试 DINOv3 FSC 二级头的数据划分。"""

import hashlib
import json
from pathlib import Path

from scripts.refinement.train_fsc_dinov3_head import _load_rows


def test_load_rows_uses_only_train_candidates_for_holdout(tmp_path: Path) -> None:
    """内部留出集只能由缓存中的 train 候选构成。"""
    candidates = []
    expected_train = set()
    expected_holdout = set()
    for index in range(20):
        name = f"image-{index}.png"
        row = {"image": f"/data/{name}", "split": "train", "label": index % 2, "xyxy": [0, 0, 1, 1]}
        candidates.append(row)
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        (expected_holdout if int.from_bytes(digest[:8], "big") % 5 == 0 else expected_train).add(name)
    candidates.append({"image": "/data/test-like.png", "split": "val", "label": 0, "xyxy": [0, 0, 1, 1]})
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps(
            {
                "format": "shwx-fsc-verifier-cache-v1",
                "metadata": {"test_split_used": False},
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )

    _, train, holdout = _load_rows(path)

    assert {row["image"].split("/")[-1] for row in train} == expected_train
    assert {row["image"].split("/")[-1] for row in holdout} == expected_holdout
