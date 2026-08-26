# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""从 SHWX 原始 YOLO 数据集构建 FSC 人工复核数据集。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
FSC_SOURCE_CLASS_ID = 24


def _parse_args() -> argparse.Namespace:
    """解析数据集构建参数。"""
    parser = argparse.ArgumentParser(description="构建只含 FSC 图像的人工复核数据集")
    parser.add_argument("--source", type=Path, required=True, help="原始 SHWX YOLO 数据集")
    parser.add_argument("--output", type=Path, required=True, help="新数据集输出目录")
    parser.add_argument("--candidate-cache", type=Path, required=True, help="一级候选缓存 JSON")
    parser.add_argument("--force", action="store_true", help="允许清空已存在的输出目录")
    return parser.parse_args()


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """按标签文件 stem 查找对应图像。"""
    for suffix in IMAGE_SUFFIXES:
        image = images_dir / f"{stem}{suffix}"
        if image.exists():
            return image
    return None


def _read_fsc_rows(label_path: Path) -> tuple[list[list[str]], Counter[int]]:
    """读取标签并只返回 FSC 行，同时统计被丢弃的原类别。"""
    fsc_rows: list[list[str]] = []
    dropped: Counter[int] = Counter()
    if not label_path.exists():
        return fsc_rows, dropped
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"标签字段数不是 5: {label_path}: {line}")
        class_id = int(fields[0])
        if class_id == FSC_SOURCE_CLASS_ID:
            fsc_rows.append(["0", *fields[1:]])
        else:
            dropped[class_id] += 1
    return fsc_rows, dropped


def _load_candidates(cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    """按图像文件名加载当前缓存中已确认的非 FSC 候选。"""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1":
        raise ValueError("候选缓存格式不是 shwx-fsc-verifier-cache-v1")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("candidates", []):
        if int(row.get("label", 1)) != 0:
            continue
        name = Path(str(row["image"])).name
        candidates.setdefault(name, []).append(row)
    return candidates


def _write_hard_negative_rows(
    image_path: Path,
    candidates: list[dict[str, Any]],
    output_label: Path,
    existing_rows: list[list[str]],
) -> list[dict[str, Any]]:
    """将非 FSC 候选追加为临时复核类别并返回人工复核索引。"""
    from PIL import Image

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    rows = list(existing_rows)
    manifest: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        x0, y0, x1, y1 = (float(value) for value in candidate["xyxy"])
        x0, x1 = max(0.0, min(x0, image_width)), max(0.0, min(x1, image_width))
        y0, y1 = max(0.0, min(y0, image_height)), max(0.0, min(y1, image_height))
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append([
            "3",
            f"{(x0 + x1) / (2 * image_width):.8f}",
            f"{(y0 + y1) / (2 * image_height):.8f}",
            f"{(x1 - x0) / image_width:.8f}",
            f"{(y1 - y0) / image_height:.8f}",
        ])
        manifest.append({
            "image": image_path.name,
            "candidate_index": candidate_index,
            "source_split": candidate.get("split", "unknown"),
            "score": candidate.get("score"),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "initial_label": "hard_negative_review",
            "manual_action": "待人工确认：改为 vehicle_confuser、static_confuser 或删除",
        })
    output_label.write_text("\n".join(" ".join(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return manifest


def build_dataset(source: Path, output: Path, candidate_cache: Path, force: bool = False) -> dict[str, Any]:
    """复制含 FSC 图像并生成 FSC 与临时硬负例标签。"""
    if not source.is_dir():
        raise FileNotFoundError(f"原始数据集不存在: {source}")
    if output.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在，使用 --force 才允许覆盖: {output}")
        shutil.rmtree(output)
    candidates = _load_candidates(candidate_cache)
    stats: dict[str, Any] = {"source": str(source), "output": str(output), "splits": {}, "dropped_original_classes": {}, "missing_images_for_fsc_labels": []}
    hard_manifest: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        copied = 0
        fsc_instances = 0
        hard_negatives = 0
        split_dropped: Counter[int] = Counter()
        for label_path in sorted(source_labels.glob("*.txt")):
            fsc_rows, dropped = _read_fsc_rows(label_path)
            if not fsc_rows:
                continue
            image_path = _find_image(source_images, label_path.stem)
            if image_path is None:
                stats["missing_images_for_fsc_labels"].append(str(label_path))
                continue
            shutil.copy2(image_path, output_images / image_path.name)
            output_label = output_labels / f"{label_path.stem}.txt"
            image_candidates = candidates.get(image_path.name, []) if split != "test" else []
            hard_rows = _write_hard_negative_rows(image_path, image_candidates, output_label, fsc_rows)
            hard_manifest.extend([{**row, "split": split} for row in hard_rows])
            copied += 1
            fsc_instances += len(fsc_rows)
            hard_negatives += len(hard_rows)
            split_dropped.update(dropped)
        stats["splits"][split] = {"images": copied, "fsc_instances": fsc_instances, "hard_negative_review": hard_negatives}
        for class_id, count in split_dropped.items():
            stats["dropped_original_classes"][str(class_id)] = stats["dropped_original_classes"].get(str(class_id), 0) + count
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        "# FSC 人工复核数据集；hard_negative_review 仅为待人工确认的临时类别\n"
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: FSC\n"
        "  1: vehicle_confuser\n"
        "  2: static_confuser\n"
        "  3: hard_negative_review\n",
        encoding="utf-8",
    )
    with (output / "hard_negative_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["image", "candidate_index", "source_split", "score", "x0", "y0", "x1", "y1", "initial_label", "manual_action", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard_manifest)
    stats["hard_negative_manifest_count"] = len(hard_manifest)
    (output / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# FSC 人工复核数据集\n\n"
        "本数据集只复制原始数据集中含 FSC 的图像，原始 25 类标签不会保留。类别 0 是完整 FSC 框；类别 1 和 2 分别是车辆、固定设施干扰物。类别 3 是当前一级候选缓存中与 FSC GT 不匹配的候选，仅作为 `hard_negative_review` 临时标签，必须人工确认后改为类别 1、类别 2 或删除。\n\n"
        "训练只使用 `train`；`val/test` 保留原始划分，禁止把它们并入训练。每张图的标签仍是标准 YOLO 格式：`class cx cy w h`，坐标归一化到 0-1。\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    """构建并报告 FSC 人工复核数据集。"""
    args = _parse_args()
    stats = build_dataset(args.source.resolve(), args.output.resolve(), args.candidate_cache.resolve(), args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
