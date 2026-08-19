# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SHWX 双模型数据准备、模态路由和类别映射工具。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

PAN_CLASSES: tuple[int, ...] = (0, 1, 2, 3)
RGB_CLASSES: tuple[int, ...] = tuple(range(4, 25))
SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CACHE_VERSION = 1


@dataclass(frozen=True)
class DualManifest:
    """双模型数据缓存清单。"""

    dataset_dir: Path
    cache_dir: Path
    threshold: float
    threshold_accuracy: float
    records: dict[str, dict[str, list[dict[str, Any]]]]
    fingerprint: str

    def paths(self, split: str, modality: str) -> list[Path]:
        """返回指定 split 和模态的原始图像路径。"""
        return [Path(item["image"]) for item in self.records[split][modality]]

    def route(self, image_path: Path) -> str:
        """按通道统计将单张图像路由到 PAN 或 RGB。"""
        return "pan" if channel_difference(image_path) <= self.threshold else "rgb"


def channel_difference(image_path: str | Path) -> float:
    """计算图像三个颜色通道的平均绝对差异。"""
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(
        (np.abs(array[..., 0] - array[..., 1]).mean() + np.abs(array[..., 1] - array[..., 2]).mean())
    )


def _image_path(image_dir: Path, stem: str) -> Path:
    """按文件名 stem 查找图像。"""
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到标签对应的图像: {image_dir / stem}")


def _label_classes(label_path: Path) -> set[int]:
    """读取 YOLO 标签中的全局类别集合。"""
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if values:
            classes.add(int(values[0]))
    return classes


def _fingerprint(dataset_dir: Path) -> str:
    """计算影响缓存的源数据指纹。"""
    hasher = hashlib.sha256(f"dual-shwx-v{CACHE_VERSION}".encode())
    for split in SPLITS:
        for root_name in ("images", "labels"):
            root = dataset_dir / root_name / split
            if not root.exists():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_file():
                    continue
                stat = path.stat()
                hasher.update(f"{path.relative_to(dataset_dir)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()[:20]


def _calibrate_threshold(dataset_dir: Path) -> tuple[float, float]:
    """使用 train/val 非空标注图像拟合 PAN/RGB 通道差异阈值。"""
    samples: list[tuple[float, int]] = []
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            classes = _label_classes(label_path)
            if not classes:
                continue
            if classes <= set(PAN_CLASSES):
                target = 0
            elif classes <= set(RGB_CLASSES):
                target = 1
            else:
                raise ValueError(f"图像同时包含 PAN/RGB 类别，无法拆分: {label_path} ({sorted(classes)})")
            samples.append((channel_difference(_image_path(image_dir, label_path.stem)), target))
    if not samples or {target for _, target in samples} != {0, 1}:
        raise ValueError("训练/验证集缺少 PAN 或 RGB 的有效标注，无法校准模态路由阈值")

    candidates = sorted({score for score, _ in samples})
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for left, right in zip(candidates, candidates[1:] + [candidates[-1] + 1e-6]):
        threshold = (left + right) / 2.0
        accuracy = sum((score > threshold) == bool(target) for score, target in samples) / len(samples)
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    if best_accuracy < 0.99:
        raise ValueError(
            f"PAN/RGB 通道路由校准准确率过低: {best_accuracy:.4f}，请检查数据或提供模态清单"
        )
    return float(best_threshold), float(best_accuracy)


def _link_or_copy(source: Path, destination: Path) -> None:
    """在缓存目录创建软链接，软链接不可用时复制文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _write_subset(
    cache_dir: Path,
    split: str,
    modality: str,
    records: Iterable[dict[str, Any]],
    global_to_local: dict[int, int],
    class_names: list[str],
    strict_labels: bool = True,
) -> None:
    """生成单个局部类别 YOLO 数据集。"""
    root = cache_dir / modality
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for item in records:
        source_image = Path(item["image"])
        source_label = Path(item["label"])
        _link_or_copy(source_image, image_dir / source_image.name)
        output_label = label_dir / f"{source_image.stem}.txt"
        lines: list[str] = []
        if source_label.exists():
            for line in source_label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if not values:
                    continue
                global_class = int(values[0])
                if global_class not in global_to_local:
                    if strict_labels:
                        raise ValueError(f"{source_label} 含有不属于 {modality} 的类别 {global_class}")
                    continue
                lines.append(" ".join([str(global_to_local[global_class]), *values[1:]]))
        output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml = (
        "# 双模型局部类别数据集\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names}\n"
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")


def prepare_dual_dataset(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    *,
    rebuild: bool = False,
) -> DualManifest:
    """准备或加载 PAN/RGB 双模型缓存。"""
    dataset_root = Path(dataset_dir).resolve()
    cache_root = Path(cache_dir).resolve()
    dataset_fingerprint = _fingerprint(dataset_root)
    manifest_path = cache_root / "manifest.json"
    if not rebuild and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") == dataset_fingerprint:
            return DualManifest(
                dataset_dir=dataset_root,
                cache_dir=cache_root,
                threshold=float(payload["threshold"]),
                threshold_accuracy=float(payload["threshold_accuracy"]),
                records=payload["records"],
                fingerprint=dataset_fingerprint,
            )

    threshold, threshold_accuracy = _calibrate_threshold(dataset_root)
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        split_records = {"pan": [], "rgb": []}
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = label_dir / f"{image_path.stem}.txt"
            classes = _label_classes(label_path)
            if split != "test" and classes and classes <= set(PAN_CLASSES):
                modality = "pan"
            elif split != "test" and classes and classes <= set(RGB_CLASSES):
                modality = "rgb"
            elif split != "test" and classes:
                raise ValueError(f"图像同时包含 PAN/RGB 类别，无法拆分: {label_path} ({sorted(classes)})")
            else:
                modality = "pan" if channel_difference(image_path) <= threshold else "rgb"
            split_records[modality].append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "score": channel_difference(image_path),
                }
            )
        records[split] = split_records

    if rebuild and cache_root.exists():
        for child in (cache_root / "pan", cache_root / "rgb"):
            if child.exists():
                shutil.rmtree(child)
    cache_root.mkdir(parents=True, exist_ok=True)
    pan_names = ["HM", "LQS", "QHS", "MS"]
    rgb_names = [
        "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16", "A6_TU-160", "A7_E-3",
        "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8", "A12_TU-22", "A13_F-15", "A14_KC-135",
        "A15_F-22", "A16_FA-18", "A17_TU-95", "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC",
    ]
    for split in SPLITS:
        strict_labels = split != "test"
        _write_subset(
            cache_root,
            split,
            "pan",
            records[split]["pan"],
            {c: c for c in PAN_CLASSES},
            pan_names,
            strict_labels=strict_labels,
        )
        _write_subset(
            cache_root,
            split,
            "rgb",
            records[split]["rgb"],
            {c: c - 4 for c in RGB_CLASSES},
            rgb_names,
            strict_labels=strict_labels,
        )
    payload = {
        "version": CACHE_VERSION,
        "fingerprint": dataset_fingerprint,
        "threshold": threshold,
        "threshold_accuracy": threshold_accuracy,
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return DualManifest(dataset_root, cache_root, threshold, threshold_accuracy, records, dataset_fingerprint)


def global_class_names() -> dict[int, str]:
    """返回 SHWX 全局类别名称。"""
    return {
        0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "A1_SU-35", 5: "A2_C-130", 6: "A3_C-17",
        7: "A4_C-5", 8: "A5_F-16", 9: "A6_TU-160", 10: "A7_E-3", 11: "A8_B-52", 12: "A9_P-3C",
        13: "A10_B-1B", 14: "A11_E-8", 15: "A12_TU-22", 16: "A13_F-15", 17: "A14_KC-135",
        18: "A15_F-22", 19: "A16_FA-18", 20: "A17_TU-95", 21: "A18_KC-10", 22: "A19_SU-34",
        23: "A20_SU-24", 24: "FSC",
    }
