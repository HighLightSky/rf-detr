# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""在固定一级阈值下评估一个或多个 DINOv3 FSC 复核头。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as transforms_functional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import crop_fsc_context, iou_xyxy  # noqa: E402
from scripts import eval_lib  # noqa: E402
from scripts.refinement.train_fsc_dinov3_head import FORMAT, FSCDinoV3Head  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析批量评估参数。"""
    parser = argparse.ArgumentParser(description="评估 DINOv3 FSC 二级复核头")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--verifier", required=True, nargs="+")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--rotation-or", action="store_true", help="使用四方向固定 OR 规则")
    parser.add_argument("--context-scale", type=float, default=None, help="覆盖复核 crop 的上下文比例")
    parser.add_argument("--trusted-floor", type=float, default=None, help="高于该分数的 FSC 候选直接保留")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _nms(records: list[eval_lib.BoxRecord], threshold: float = 0.5) -> list[eval_lib.BoxRecord]:
    """对单张图的 FSC 候选执行置信度优先 NMS。"""
    kept: list[eval_lib.BoxRecord] = []
    for record in sorted(records, key=lambda item: float(item.score or 0.0), reverse=True):
        if all(iou_xyxy(record.xyxy, chosen.xyxy) <= threshold for chosen in kept):
            kept.append(record)
    return kept


def _plain(result: eval_lib.EvalResult) -> dict[str, float | int]:
    """转换评测结果并计算 F1。"""
    f1 = 2 * result.precision * result.recall / max(result.precision + result.recall, 1e-12)
    return {"tp": result.tp, "fp": result.fp, "fn": result.fn, "precision": result.precision, "recall": result.recall, "f1": f1}


def _fsc_metric(gt: list[eval_lib.BoxRecord], predictions: list[eval_lib.BoxRecord]) -> dict[str, float | int]:
    """按 FSC 车辆 IoU=0.35 计算二级结果。"""
    config = eval_lib.EvalConfig(class_to_group={24: "FSC"}, group_iou_thresholds={"FSC": 0.35}, default_iou_threshold=0.35)
    result = eval_lib.evaluate_competition_metrics(
        [record for record in gt if record.class_id == 24],
        [record for record in predictions if record.class_id == 24],
        config,
    )
    return _plain(result["all"])


def _load_verifier(path: Path, device: torch.device) -> tuple[dict[str, Any], Any, FSCDinoV3Head, Any]:
    """加载冻结 DINOv3 主干和二分类头。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"不是 {FORMAT} checkpoint: {path}")
    import timm
    from timm.data import create_transform, resolve_model_data_config

    backbone = timm.create_model(payload["metadata"]["model_name"], pretrained=True, num_classes=0).to(device).eval()
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    head = FSCDinoV3Head(int(payload["feature_dim"])).to(device).eval()
    head.load_state_dict(payload["state_dict"])
    return payload, backbone, head, transform


def _refine(
    paths: list[Path],
    raw: list[eval_lib.BoxRecord],
    payload: dict[str, Any],
    backbone: Any,
    head: FSCDinoV3Head,
    transform: Any,
    device: torch.device,
    rotation_or: bool,
    context_scale: float | None,
    trusted_floor: float | None,
) -> list[eval_lib.BoxRecord]:
    """使用固定 argmax 对一级 FSC 候选执行二级复核。"""
    by_image: dict[str, list[eval_lib.BoxRecord]] = defaultdict(list)
    for record in raw:
        by_image[record.image_id].append(record)
    output: list[eval_lib.BoxRecord] = []
    for path in paths:
        image_records = by_image.get(path.stem, [])
        candidates = _nms([record for record in image_records if record.class_id == 24])
        trusted = [record for record in candidates if trusted_floor is not None and float(record.score or 0.0) >= trusted_floor]
        candidates = [record for record in candidates if record not in trusted]
        kept: list[eval_lib.BoxRecord] = []
        if candidates:
            scale = float(payload["metadata"]["context_scale"]) if context_scale is None else context_scale
            with Image.open(path) as source:
                image = source.convert("RGB")
                crops = [transform(crop_fsc_context(image, record.xyxy, scale, 224)) for record in candidates]
            decisions: list[int] = []
            for start in range(0, len(crops), 64):
                batch = torch.stack(crops[start : start + 64]).to(device)
                with torch.inference_mode():
                    views = (0, 90, 180, 270) if rotation_or else (0,)
                    predictions = [head(backbone(transforms_functional.rotate(batch, angle))).argmax(1) for angle in views]
                    decisions.extend(torch.stack(predictions).any(dim=0).cpu().tolist())
            kept = [record for record, decision in zip(candidates, decisions, strict=True) if decision == 1]
        output.extend(record for record in image_records if record.class_id != 24)
        output.extend(trusted)
        output.extend(kept)
    return output


def main() -> None:
    """在 test split 上输出一级和多个二级头的完整指标。"""
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    paths = sorted(path for path in (dataset_dir / "images/test").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    device = torch.device(eval_lib.resolve_device(args.device))
    detector = RFDETR.from_checkpoint(str(Path(args.detector).resolve()))
    raw, throughput, _, timed = eval_lib.predict_batched_to_records(detector, paths, device=str(device), conf_threshold=args.threshold, batch_size=8, num_workers=0, num_classes=25, prefetch_factor=2, precision="auto", copy_prefetch=True, warmup_batches=1, progress_interval_s=2.0)
    sizes = eval_lib.build_image_size_map(paths)
    gt = eval_lib.load_yolo_labels(dataset_dir / "labels/test", sizes)
    dataset = eval_lib.build_dataset_cfg(data_dir=dataset_dir)
    config = eval_lib.EvalConfig(class_to_group=dataset.class_to_group, group_iou_thresholds=dataset.group_iou_thresholds, default_iou_threshold=0.50)
    report: dict[str, Any] = {"test_images": len(paths), "threshold": args.threshold, "detector_records": len(raw), "throughput_img_s": throughput, "timed_images": timed, "stage1_fsc": _fsc_metric(gt, raw), "stage1_all": _plain(eval_lib.evaluate_competition_metrics(gt, raw, config)["all"]), "verifiers": {}}
    for verifier_path in [Path(value).resolve() for value in args.verifier]:
        payload, backbone, head, transform = _load_verifier(verifier_path, device)
        refined = _refine(paths, raw, payload, backbone, head, transform, device, args.rotation_or, args.context_scale, args.trusted_floor)
        metrics = eval_lib.evaluate_competition_metrics(gt, refined, config)
        report["verifiers"][str(verifier_path)] = {"metadata": payload["metadata"], "records": len(refined), "fsc": _fsc_metric(gt, refined), "all": _plain(metrics["all"]), "groups": {name: _plain(result) for name, result in metrics["groups"].items()}}
        del backbone, head
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
