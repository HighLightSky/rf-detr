# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""以冻结 RF-DETR 生成 FSC 二阶段分类器的训练与验证候选缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement.fsc_two_stage import FSCVerifierPolicy, iou_xyxy, label_fsc_candidate  # noqa: E402
from scripts import eval_lib  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析候选缓存构建参数。"""
    parser = argparse.ArgumentParser(description="生成 FSC 二阶段训练候选缓存")
    parser.add_argument("--checkpoint", required=True, help="冻结一级 RF-DETR checkpoint")
    parser.add_argument("--dataset-dir", required=True, help="SHWX 数据集根目录")
    parser.add_argument("--output", required=True, help="候选缓存 JSON 路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--candidate-floor", type=float, default=0.01, help="仅用于挖掘训练硬负样本的 FSC 候选下限")
    parser.add_argument("--final-candidate-floor", type=float, default=0.05, help="部署时固定使用的一级 FSC 候选下限")
    parser.add_argument("--max-candidates-per-image", type=int, default=12, help="每张图保留的 NMS 后 FSC 候选数")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """计算文件 SHA256，用于绑定一级检测器版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(directory: Path) -> list[Path]:
    """读取一个 split 中的常见格式图像。"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _ground_truth(paths: list[Path], labels_dir: Path) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """读取图片对应的全部 GT，供候选标注与负样本安全审计使用。"""
    sizes = eval_lib.build_image_size_map(paths)
    records = eval_lib.load_yolo_labels(labels_dir, sizes)
    grouped: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {path.stem: [] for path in paths}
    for record in records:
        if record.image_id in grouped:
            grouped[record.image_id].append((record.class_id, record.xyxy))
    return grouped


def _nms(records: list[eval_lib.BoxRecord], threshold: float, limit: int) -> list[eval_lib.BoxRecord]:
    """对单张图的 FSC 候选执行置信度优先 NMS。"""
    kept: list[eval_lib.BoxRecord] = []
    for record in sorted(records, key=lambda item: float(item.score or 0.0), reverse=True):
        if all(iou_xyxy(record.xyxy, chosen.xyxy) <= threshold for chosen in kept):
            kept.append(record)
        if len(kept) >= limit:
            break
    return kept


def main() -> None:
    """生成只包含训练集和验证集的 FSC 候选缓存。"""
    args = _parse_args()
    policy = FSCVerifierPolicy(candidate_floor=args.final_candidate_floor)
    policy.validate()
    if args.max_candidates_per_image <= 0:
        raise ValueError("max-candidates-per-image 必须为正数")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou 必须位于 (0, 1]")

    checkpoint = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    split_paths = {split: _image_paths(dataset_dir / "images" / split) for split in ("train", "val")}
    if not split_paths["train"] or not split_paths["val"]:
        raise ValueError("训练集和验证集都必须包含图像")
    gt_by_split = {
        split: _ground_truth(paths, dataset_dir / "labels" / split) for split, paths in split_paths.items()
    }
    all_paths = split_paths["train"] + split_paths["val"]
    split_by_image = {path.stem: split for split, paths in split_paths.items() for path in paths}
    path_by_image = {path.stem: path for path in all_paths}

    print(f"[i] 加载冻结一级检测器: {checkpoint}")
    model = RFDETR.from_checkpoint(str(checkpoint))
    other_class_thresholds = {class_id: 1.01 for class_id in range(25) if class_id != policy.fsc_class_id}
    predictions, throughput, _, timed_images = eval_lib.predict_batched_to_records(
        model,
        all_paths,
        device=eval_lib.resolve_device(args.device),
        conf_threshold=1.01,
        class_conf_thresholds={policy.fsc_class_id: policy.candidate_floor, **other_class_thresholds},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=25,
        prefetch_factor=2,
        precision="auto",
        copy_prefetch=True,
    )
    del model

    grouped_predictions: dict[str, list[eval_lib.BoxRecord]] = defaultdict(list)
    for record in predictions:
        if record.class_id == policy.fsc_class_id and record.image_id in split_by_image:
            grouped_predictions[record.image_id].append(record)

    candidate_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    for split, paths in split_paths.items():
        for path in paths:
            image_id = path.stem
            ground_truth = gt_by_split[split][image_id]
            for class_id, box in ground_truth:
                if class_id == policy.fsc_class_id:
                    gt_rows.append(
                        {
                            "image": str(path),
                            "image_id": image_id,
                            "split": split,
                            "xyxy": [float(value) for value in box],
                            "label": 1,
                            "source": "ground_truth",
                        }
                    )
            for index, prediction in enumerate(_nms(grouped_predictions[image_id], args.nms_iou, args.max_candidates_per_image)):
                candidate_rows.append(
                    {
                        "image": str(path_by_image[image_id]),
                        "image_id": image_id,
                        "split": split,
                        "prediction_index": index,
                        "xyxy": [float(value) for value in prediction.xyxy],
                        "score": float(prediction.score or 0.0),
                        "label": label_fsc_candidate(prediction.xyxy, ground_truth, fsc_class_id=policy.fsc_class_id),
                        "source": "detector_candidate",
                    }
                )

    payload = {
        "format": "shwx-fsc-verifier-cache-v1",
        "metadata": {
            "dataset_dir": str(dataset_dir),
            "detector_checkpoint": str(checkpoint),
            "detector_sha256": _sha256(checkpoint),
            "policy": policy.to_dict(),
            "nms_iou": args.nms_iou,
            "max_candidates_per_image": args.max_candidates_per_image,
            "mining_candidate_floor": args.candidate_floor,
            "throughput": throughput,
            "timed_images": timed_images,
            "test_split_used": False,
        },
        "split_manifest": {split: [str(path) for path in paths] for split, paths in split_paths.items()},
        "ground_truth": gt_rows,
        "candidates": candidate_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    counts: dict[str, int] = defaultdict(int)
    for row in candidate_rows + gt_rows:
        counts[f"{row['split']}:label_{row['label']}:{row['source']}"] += 1
    print(f"[完成] 候选缓存: {output}")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
