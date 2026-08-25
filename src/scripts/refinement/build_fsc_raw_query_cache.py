# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""从冻结 RF-DETR 原始 FSC query 挖掘二级训练候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as transforms_functional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import FSCVerifierPolicy, iou_xyxy, label_fsc_candidate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析原始 query 候选挖掘参数。"""
    parser = argparse.ArgumentParser(description="构建 FSC 原始 query 硬负样本缓存")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=80, help="每图参与 NMS 的原始 FSC query 数")
    parser.add_argument("--max-candidates", type=int, default=12, help="每图 NMS 后保留的候选数")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """计算冻结一级 checkpoint 的内容摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(directory: Path) -> list[Path]:
    """读取 split 中的图像文件。"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _labels(path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    """读取一张 YOLO 标签并转为绝对 xyxy。"""
    if not path.is_file():
        return []
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"标签格式错误: {path}")
        class_id, center_x, center_y, box_width, box_height = (float(value) for value in values)
        result.append(
            (
                int(class_id),
                (
                    (center_x - box_width * 0.5) * width,
                    (center_y - box_height * 0.5) * height,
                    (center_x + box_width * 0.5) * width,
                    (center_y + box_height * 0.5) * height,
                ),
            )
        )
    return result


def _nms(rows: list[tuple[float, tuple[float, float, float, float]]], threshold: float, limit: int) -> list[tuple[float, tuple[float, float, float, float]]]:
    """对单张图的同类 FSC query 做置信度优先 NMS。"""
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for row in sorted(rows, key=lambda item: item[0], reverse=True):
        if all(iou_xyxy(row[1], chosen[1]) <= threshold for chosen in kept):
            kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def _batch_tensors(paths: list[Path], resolution: int, means: list[float], stds: list[float]) -> tuple[Tensor, list[tuple[int, int]]]:
    """读取图像并制作与 RF-DETR predict 一致的方形输入批次。"""
    tensors: list[Tensor] = []
    sizes: list[tuple[int, int]] = []
    for path in paths:
        with Image.open(path) as image:
            source = image.convert("RGB")
            width, height = source.size
            tensor = transforms_functional.to_tensor(source)
        sizes.append((width, height))
        tensors.append(transforms_functional.normalize(transforms_functional.resize(tensor, [resolution, resolution], antialias=False), means, stds))
    return torch.stack(tensors), sizes


def main() -> None:
    """仅从 train/val 生成带 IoU 标签的原始 FSC query 缓存。"""
    args = _parse_args()
    if args.batch_size <= 0 or args.topk <= 0 or args.max_candidates <= 0:
        raise ValueError("batch-size、topk、max-candidates 必须为正数")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou 必须位于 (0, 1]")
    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    split_paths = {split: _paths(dataset / "images" / split) for split in ("train", "val")}
    if not checkpoint.is_file() or not split_paths["train"] or not split_paths["val"]:
        raise ValueError("checkpoint、训练集和验证集必须存在")
    policy = FSCVerifierPolicy()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    detector = RFDETR.from_checkpoint(str(checkpoint))
    core = detector.model.model.to(device).eval()
    grouped = {split: split_paths[split] for split in ("train", "val")}
    rows: list[dict[str, Any]] = []
    for split, paths in grouped.items():
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start : start + args.batch_size]
            batch, sizes = _batch_tensors(chunk, detector.model.resolution, detector.means, detector.stds)
            with torch.inference_mode():
                result = core(batch.to(device))
                scores, indices = result["pred_logits"][:, :, policy.fsc_class_id].sigmoid().topk(args.topk, dim=1)
                selected_boxes = result["pred_boxes"].gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
            for path, (width, height), image_scores, boxes in zip(chunk, sizes, scores.cpu(), selected_boxes.cpu(), strict=True):
                ground_truth = _labels(dataset / "labels" / split / f"{path.stem}.txt", width, height)
                decoded: list[tuple[float, tuple[float, float, float, float]]] = []
                for score, box in zip(image_scores.tolist(), boxes.tolist(), strict=True):
                    center_x, center_y, box_width, box_height = box
                    xyxy = (
                        max(0.0, (center_x - box_width * 0.5) * width),
                        max(0.0, (center_y - box_height * 0.5) * height),
                        min(float(width), (center_x + box_width * 0.5) * width),
                        min(float(height), (center_y + box_height * 0.5) * height),
                    )
                    decoded.append((float(score), xyxy))
                for candidate_index, (score, xyxy) in enumerate(_nms(decoded, args.nms_iou, args.max_candidates)):
                    rows.append(
                        {
                            "image": str(path),
                            "image_id": path.stem,
                            "split": split,
                            "prediction_index": candidate_index,
                            "xyxy": [float(value) for value in xyxy],
                            "score": score,
                            "label": label_fsc_candidate(xyxy, ground_truth, fsc_class_id=policy.fsc_class_id),
                            "source": "raw_query_candidate",
                        }
                    )
            completed = min(start + len(chunk), len(paths))
            if completed == len(paths) or completed % 512 == 0:
                print(f"[{split}] {completed}/{len(paths)}")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['split']}:label_{row['label']}"] += 1
    payload = {
        "format": "shwx-fsc-verifier-cache-v1",
        "metadata": {
            "dataset_dir": str(dataset),
            "detector_checkpoint": str(checkpoint),
            "detector_sha256": _sha256(checkpoint),
            "policy": policy.to_dict(),
            "source": "raw_fsc_queries",
            "topk": args.topk,
            "max_candidates_per_image": args.max_candidates,
            "nms_iou": args.nms_iou,
            "test_split_used": False,
        },
        "split_manifest": {split: [str(path) for path in paths] for split, paths in split_paths.items()},
        "ground_truth": [],
        "candidates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[完成] 原始 FSC query 缓存: {output}")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
