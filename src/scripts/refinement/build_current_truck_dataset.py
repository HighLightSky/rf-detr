# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""从当前 SHWX 训练图像和同源 truck 标注构建 26 类派生训练集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """解析派生数据集路径。"""
    parser = argparse.ArgumentParser(description="构建当前数据集的 26 类 FSC/truck 训练集")
    parser.add_argument("--current-dataset", required=True)
    parser.add_argument("--truck-dataset", required=True, help="同源但含 truck=25 标注的数据集")
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--teacher", default=None, help="可选的同源 26 类教师 checkpoint")
    parser.add_argument("--teacher-floor", type=float, default=0.25, help="truck 伪标注最低置信度")
    parser.add_argument("--teacher-nms", type=float, default=0.5)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    return parser.parse_args()


def _is_holdout(name: str, modulus: int) -> bool:
    """按图像名稳定划分当前 train 图像，避免泄漏。"""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0


def _read_label(path: Path) -> list[str]:
    """读取非空 YOLO 标签行。"""
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_split(
    names: list[str],
    split: str,
    current: Path,
    truck: Path,
    output: Path,
    modulus: int,
    teacher_rows: dict[str, list[tuple[float, tuple[float, float, float, float]]]],
) -> dict[str, int]:
    """写入派生 split 的图像软链接和合并标签。"""
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    counts = {"images": 0, "truck_images": 0, "truck_boxes": 0, "pseudo_truck_boxes": 0}
    for name in names:
        source_image = current / "images" / "train" / name
        target_image = image_dir / name
        if not target_image.exists():
            target_image.symlink_to(source_image)
        current_label = _read_label(current / "labels" / "train" / f"{Path(name).stem}.txt")
        old_label = _read_label(truck / "labels" / "train" / f"{Path(name).stem}.txt")
        truck_rows = [row for row in old_label if int(row.split()[0]) == 25]
        pseudo_rows = teacher_rows.get(Path(name).stem, [])
        existing_boxes = []
        for row in truck_rows:
            values = [float(value) for value in row.split()[1:]]
            existing_boxes.append(values)
        pseudo_yolo: list[str] = []
        source_image = current / "images" / "train" / name
        from PIL import Image

        width, height = Image.open(source_image).size
        for score, box in pseudo_rows:
            if any(_iou_xyxy(box, old_box) >= 0.5 for old_box in _yolo_to_xyxy(existing_boxes, width, height)):
                continue
            x0, y0, x1, y1 = box
            pseudo_yolo.append(
                f"25 {((x0 + x1) / 2) / width:.8f} {((y0 + y1) / 2) / height:.8f} "
                f"{(x1 - x0) / width:.8f} {(y1 - y0) / height:.8f}"
            )
        (label_dir / f"{Path(name).stem}.txt").write_text(
            "\n".join(current_label + truck_rows + pseudo_yolo) + "\n", encoding="utf-8"
        )
        counts["images"] += 1
        counts["truck_images"] += int(bool(truck_rows))
        counts["truck_boxes"] += len(truck_rows) + len(pseudo_yolo)
        counts["pseudo_truck_boxes"] += len(pseudo_yolo)
    return counts


def _iou_xyxy(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """计算两个像素框的交并比。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / (area_first + area_second - intersection) if area_first + area_second > intersection else 0.0


def _yolo_to_xyxy(rows: list[list[float]], width: int, height: int) -> list[tuple[float, float, float, float]]:
    """将无类别 YOLO 数值转换为像素框。"""
    return [
        ((x - w * 0.5) * width, (y - h * 0.5) * height, (x + w * 0.5) * width, (y + h * 0.5) * height)
        for x, y, w, h in rows
    ]


def _teacher_rows(
    checkpoint: Path,
    image_paths: list[Path],
    floor: float,
    nms_threshold: float,
    batch_size: int,
) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    """仅对当前 train 图像运行教师并提取 truck 伪标注。"""
    if not 0.0 < floor < 1.0 or not 0.0 < nms_threshold <= 1.0 or batch_size <= 0:
        raise ValueError("teacher-floor 和 teacher-nms 参数范围错误")
    project_root = Path(__file__).resolve().parents[3]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rfdetr import RFDETR

    import numpy as np

    model = RFDETR.from_checkpoint(str(checkpoint))
    output: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        images = []
        from PIL import Image

        for path in paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
        detections_list = model.predict(images, threshold=floor, include_source_image=False)
        for path, detections in zip(paths, detections_list, strict=True):
            candidates = [
                (float(score), tuple(float(value) for value in box))
                for score, box, class_id in zip(detections.confidence, detections.xyxy, detections.class_id, strict=True)
                if int(class_id) == 25 and float(score) >= floor
            ]
            kept: list[tuple[float, tuple[float, float, float, float]]] = []
            for candidate in sorted(candidates, reverse=True):
                if all(_iou_xyxy(candidate[1], chosen[1]) <= nms_threshold for chosen in kept):
                    kept.append(candidate)
            output[path.stem] = kept
        completed = min(start + len(paths), len(image_paths))
        if completed % 500 < batch_size or completed == len(image_paths):
            print(f"[teacher] {completed}/{len(image_paths)}")
    del model
    return output


def main() -> None:
    """构建只由当前 train 图像组成的 26 类派生数据集。"""
    args = _parse_args()
    if args.holdout_modulus < 2:
        raise ValueError("holdout-modulus 必须不小于 2")
    current = Path(args.current_dataset).resolve()
    truck = Path(args.truck_dataset).resolve()
    output = Path(args.output).resolve()
    names = sorted(path.name for path in (current / "images" / "train").iterdir() if path.is_file())
    if not names:
        raise ValueError("当前数据集没有 train 图像")
    train_names = [name for name in names if not _is_holdout(name, args.holdout_modulus)]
    holdout_names = [name for name in names if _is_holdout(name, args.holdout_modulus)]
    teacher_rows = _teacher_rows(
        Path(args.teacher).resolve(),
        [current / "images" / "train" / name for name in names],
        args.teacher_floor,
        args.teacher_nms,
        args.teacher_batch_size,
    ) if args.teacher else {}
    train_counts = _prepare_split(train_names, "train", current, truck, output, args.holdout_modulus, teacher_rows)
    holdout_counts = _prepare_split(holdout_names, "val", current, truck, output, args.holdout_modulus, teacher_rows)
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "nc": 26,
        "names": [
            "HM", "LQS", "QHS", "MS", "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5",
            "A5_F-16", "A6_TU-160", "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B",
            "A11_E-8", "A12_TU-22", "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18",
            "A17_TU-95", "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC", "truck",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(json.dumps(data_yaml, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "current_dataset": str(current),
        "truck_annotation_dataset": str(truck),
        "source_split": "current train only",
        "holdout_modulus": args.holdout_modulus,
        "train": train_counts,
        "holdout": holdout_counts,
        "test_used_for_training": False,
        "teacher": str(Path(args.teacher).resolve()) if args.teacher else None,
        "teacher_floor": args.teacher_floor,
    }
    (output / "derivation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
