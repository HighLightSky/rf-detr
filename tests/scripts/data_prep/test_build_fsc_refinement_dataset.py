"""FSC 人工复核数据集构建测试。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.data_prep.build_fsc_refinement_dataset import build_dataset


def _write_source_dataset(root: Path) -> None:
    """创建含一张 FSC 图像和一个非 FSC 标签的最小来源数据集。"""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(root / "images" / "train" / "sample.jpg")
    (root / "labels" / "train" / "sample.txt").write_text(
        "24 0.500000 0.500000 0.400000 0.400000\n12 0.200000 0.200000 0.100000 0.100000\n",
        encoding="utf-8",
    )


def test_build_dataset_keeps_only_fsc_and_appends_review_candidate(tmp_path: Path) -> None:
    """新标签只含 FSC 和待审核硬负例，图像为真实复制。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    cache = tmp_path / "cache.json"
    _write_source_dataset(source)
    cache.write_text(json.dumps({
        "format": "shwx-fsc-verifier-cache-v1",
        "candidates": [{
            "image": str(source / "images" / "train" / "sample.jpg"),
            "split": "train",
            "label": 0,
            "score": 0.8,
            "xyxy": [10, 10, 30, 30],
        }],
    }), encoding="utf-8")

    stats = build_dataset(source, output, cache)

    label_rows = (output / "labels" / "train" / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert stats["splits"]["train"] == {"images": 1, "fsc_instances": 1, "hard_negative_review": 1}
    assert label_rows[0].startswith("0 ")
    assert label_rows[1].startswith("3 ")
    assert not (output / "images" / "train" / "sample.jpg").is_symlink()
    assert (output / "images" / "train" / "sample.jpg").read_bytes() == (source / "images" / "train" / "sample.jpg").read_bytes()
