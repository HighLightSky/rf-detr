# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""从 SHWX train split 构建无测试泄漏的内部验证数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    """解析内部留出数据集构建参数。"""
    parser = argparse.ArgumentParser(description="构建 FSC 一级检测内部留出集")
    parser.add_argument("--source", required=True, help="原始 SHWX 数据集根目录")
    parser.add_argument("--output", required=True, help="派生数据集根目录")
    parser.add_argument("--holdout-modulus", type=int, default=5)
    return parser.parse_args()


def _is_holdout(name: str, modulus: int) -> bool:
    """按图像文件名稳定地划分内部留出集。"""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0


def _count_fsc(label_path: Path) -> int:
    """统计一份 YOLO 标签中的 FSC 实例数。"""
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip() and line.split()[0] == "24")


def _link_split(source: Path, output: Path, names: list[str], split: str) -> dict[str, int]:
    """为指定 split 建立图像软链接并复制其标签。"""
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    for name in names:
        image_path = source / "images" / "train" / name
        label_path = source / "labels" / "train" / f"{Path(name).stem}.txt"
        target_image = image_dir / name
        if not target_image.exists():
            target_image.symlink_to(image_path)
        target_label = label_dir / label_path.name
        if not target_label.exists():
            target_label.symlink_to(label_path)
        counters["images"] += 1
        counters["fsc_instances"] += _count_fsc(label_path)
    return dict(counters)


def main() -> None:
    """构建训练图内部划分，并显式记录测试集从未参与其中。"""
    args = _parse_args()
    if args.holdout_modulus < 2:
        raise ValueError("holdout-modulus 必须不小于 2")
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    source_yaml = source / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"缺少 data.yaml: {source_yaml}")
    names = sorted(path.name for path in (source / "images" / "train").iterdir() if path.is_file())
    if not names:
        raise ValueError("源数据集 train split 为空")
    train_names = [name for name in names if not _is_holdout(name, args.holdout_modulus)]
    val_names = [name for name in names if _is_holdout(name, args.holdout_modulus)]
    if not train_names or not val_names:
        raise ValueError("内部 train 或 val split 为空")
    train_counts = _link_split(source, output, train_names, "train")
    val_counts = _link_split(source, output, val_names, "val")
    source_data: dict[str, Any] = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    output_data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "nc": source_data["nc"],
        "names": source_data["names"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(yaml.safe_dump(output_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    metadata = {
        "source_dataset": str(source),
        "source_split": "train only",
        "holdout_modulus": args.holdout_modulus,
        "train": train_counts,
        "val": val_counts,
        "test_used_for_training": False,
    }
    (output / "derivation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
