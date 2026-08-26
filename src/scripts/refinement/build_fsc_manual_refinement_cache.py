# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""从 FSC 人工复核数据集生成二级 DINOv3 分类头候选缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import iou_xyxy  # noqa: E402
from scripts import eval_lib  # noqa: E402

_FSC_CLASS_ID = 0
_NEGATIVE_CLASS_IDS = {1, 2}
_IGNORE_CLASS_IDS = {3}
_FSC_MATCH_IOU = 0.35
_FSC_PARTIAL_IOU = 0.10
_NEGATIVE_MATCH_IOU = 0.10


def _parse_args() -> argparse.Namespace:
    """解析候选缓存构建参数。"""
    parser = argparse.ArgumentParser(description="生成 FSC 人工复核数据的二级分类候选缓存")
    parser.add_argument("--checkpoint", required=True, help="冻结一级 RF-DETR checkpoint")
    parser.add_argument("--dataset-dir", required=True, help="FSC 人工复核数据集根目录")
    parser.add_argument("--output", required=True, help="候选缓存 JSON 路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--candidate-floor", type=float, default=0.05, help="与部署保持一致的 FSC 候选入口")
    parser.add_argument("--max-candidates-per-image", type=int, default=50)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--base-cache", type=Path, default=None, help="可选：复用已有一级候选缓存，避免重复前向")
    return parser.parse_args()


def _image_paths(directory: Path) -> list[Path]:
    """返回目录内的图像文件。"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _load_annotations(paths: list[Path], labels_dir: Path) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """读取人工复核数据集的 YOLO 标注并裁剪微小边界浮点误差。"""
    grouped: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for path in paths:
        rows: list[tuple[int, tuple[float, float, float, float]]] = []
        label_path = labels_dir / f"{path.stem}.txt"
        with Image.open(path) as image:
            width, height = image.size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"标签字段数不是 5: {label_path}: {line}")
            class_id = int(fields[0])
            if class_id not in {_FSC_CLASS_ID, *_NEGATIVE_CLASS_IDS, *_IGNORE_CLASS_IDS}:
                raise ValueError(f"未知人工复核类别: {label_path}: {class_id}")
            cx, cy, box_width, box_height = (float(value) for value in fields[1:])
            x0 = max(0.0, min((cx - box_width / 2.0) * width, width))
            y0 = max(0.0, min((cy - box_height / 2.0) * height, height))
            x1 = max(0.0, min((cx + box_width / 2.0) * width, width))
            y1 = max(0.0, min((cy + box_height / 2.0) * height, height))
            if x1 <= x0 or y1 <= y0:
                continue
            rows.append((class_id, (x0, y0, x1, y1)))
        grouped[path.stem] = rows
    return grouped


def _center_in_box(candidate: tuple[float, float, float, float], annotation: tuple[float, float, float, float]) -> bool:
    """判断候选中心是否落在人工标注框内。"""
    center_x = (candidate[0] + candidate[2]) / 2.0
    center_y = (candidate[1] + candidate[3]) / 2.0
    return annotation[0] <= center_x <= annotation[2] and annotation[1] <= center_y <= annotation[3]


def label_candidate(
    candidate: tuple[float, float, float, float],
    annotations: list[tuple[int, tuple[float, float, float, float]]],
) -> int | None:
    """将一级 FSC 候选映射为完整 FSC、非 FSC 或忽略。

    完整 FSC 必须达到部署评估一致的 IoU 0.35。与 FSC 有局部重叠的候选视为
    partial FSC 并忽略，避免把真实发射车局部错误训练成负例。
    """
    fsc_boxes = [box for class_id, box in annotations if class_id == _FSC_CLASS_ID]
    if any(iou_xyxy(candidate, box) >= _FSC_MATCH_IOU for box in fsc_boxes):
        return 1
    if any(iou_xyxy(candidate, box) >= _FSC_PARTIAL_IOU or _center_in_box(candidate, box) for box in fsc_boxes):
        return None
    for class_id, box in annotations:
        if class_id in _NEGATIVE_CLASS_IDS and (
            iou_xyxy(candidate, box) >= _NEGATIVE_MATCH_IOU or _center_in_box(candidate, box)
        ):
            return 0
    return None


def _nms(records: list[eval_lib.BoxRecord], threshold: float, limit: int) -> list[eval_lib.BoxRecord]:
    """按一级置信度对 FSC 候选执行 NMS。"""
    kept: list[eval_lib.BoxRecord] = []
    for record in sorted(records, key=lambda item: float(item.score or 0.0), reverse=True):
        if all(iou_xyxy(record.xyxy, chosen.xyxy) <= threshold for chosen in kept):
            kept.append(record)
        if len(kept) >= limit:
            break
    return kept


def _sha256(path: Path) -> str:
    """计算一级权重哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_base_candidates(path: Path, image_stems: set[str]) -> dict[str, list[eval_lib.BoxRecord]]:
    """读取已有缓存中属于人工训练图像的一级 FSC 候选。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("base-cache 格式错误或包含测试集")
    grouped: dict[str, list[eval_lib.BoxRecord]] = defaultdict(list)
    for row in payload.get("candidates", []):
        image_id = str(row["image_id"])
        if image_id not in image_stems:
            continue
        grouped[image_id].append(
            eval_lib.BoxRecord(
                image_id=image_id,
                class_id=24,
                xyxy=tuple(float(value) for value in row["xyxy"]),
                score=float(row.get("score", 0.0)),
            )
        )
    return grouped


def main() -> None:
    """使用训练集人工标注构建无官方测试泄漏的二级候选缓存。"""
    args = _parse_args()
    if not 0.0 <= args.candidate_floor <= 1.0:
        raise ValueError("candidate-floor 必须位于 [0, 1]")
    if args.max_candidates_per_image <= 0 or not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("max-candidates-per-image 必须为正数且 nms-iou 必须位于 (0, 1]")
    checkpoint = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    paths = _image_paths(dataset_dir / "images" / "train")
    if not paths:
        raise ValueError("人工复核训练集为空")
    annotations = _load_annotations(paths, dataset_dir / "labels" / "train")
    if args.base_cache is not None:
        print(f"[i] 复用已有一级候选缓存: {args.base_cache.resolve()}")
        grouped = _load_base_candidates(args.base_cache.resolve(), {path.stem for path in paths})
        throughput, timed_images = 0.0, 0
    else:
        print(f"[i] 加载冻结一级检测器: {checkpoint}")
        model = RFDETR.from_checkpoint(str(checkpoint))
        class_thresholds = {class_id: 1.01 for class_id in range(25)}
        class_thresholds[24] = args.candidate_floor
        predictions, throughput, _, timed_images = eval_lib.predict_batched_to_records(
            model,
            paths,
            device=eval_lib.resolve_device(args.device),
            conf_threshold=1.01,
            class_conf_thresholds=class_thresholds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_classes=25,
            prefetch_factor=2,
            precision="auto",
            copy_prefetch=True,
        )
        del model
        grouped = defaultdict(list)
        for record in predictions:
            if record.class_id == 24:
                grouped[record.image_id].append(record)
    candidates: list[dict[str, Any]] = []
    ignored = Counter()
    for path in paths:
        image_id = path.stem
        for index, prediction in enumerate(_nms(grouped[image_id], args.nms_iou, args.max_candidates_per_image)):
            label = label_candidate(prediction.xyxy, annotations[image_id])
            if label is None:
                ignored[image_id] += 1
                continue
            candidates.append({
                "image": str(path),
                "image_id": image_id,
                "split": "train",
                "prediction_index": index,
                "xyxy": [float(value) for value in prediction.xyxy],
                "score": float(prediction.score or 0.0),
                "label": label,
                "source": "manual_annotation_matched_candidate",
            })
    counts = Counter(int(row["label"]) for row in candidates)
    payload = {
        "format": "shwx-fsc-verifier-cache-v1",
        "metadata": {
            "dataset_dir": str(dataset_dir),
            "detector_checkpoint": str(checkpoint),
            "detector_sha256": _sha256(checkpoint),
            "candidate_floor": args.candidate_floor,
            "nms_iou": args.nms_iou,
            "max_candidates_per_image": args.max_candidates_per_image,
            "base_cache": str(args.base_cache.resolve()) if args.base_cache is not None else None,
            "label_mapping": {"1": "full_fsc", "0": "vehicle_or_static_confuser"},
            "ignored_labels": ["hard_negative_review", "partial_fsc", "unmatched_candidate"],
            "test_split_used": False,
            "throughput": throughput,
            "timed_images": timed_images,
        },
        "split_manifest": {"train": [str(path) for path in paths]},
        "ground_truth": [],
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"labeled_full_fsc": counts[1], "labeled_non_fsc": counts[0], "ignored_candidates": sum(ignored.values()), "images": len(paths)}, ensure_ascii=False))
    print(f"[完成] 候选缓存: {output}")


if __name__ == "__main__":
    main()
