# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""通过统一配置对比 RF-DETR nano 与 YOLO11m 大图切分模型。

评测分成两部分：

1. ``large-cut`` 600 张独立切分测试图上的边界框 AP50/AP75/AP90；
2. SHWX 真正大图上的边界切分、裁窗检测和坐标回映射端到端结果。

示例：
    uv run python src/scripts/compare_large_cut_models.py \
        --detector-checkpoint output/.../checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.eval_large_cut import boxes_to_coco_gt, boxes_to_coco_pred, compute_coco_ap  # noqa: E402
from scripts.eval_lib import build_image_size_map, read_test_image_paths  # noqa: E402
from scripts.large_cut_pipeline import (  # noqa: E402
    _nms_boxes,
    crop_with_padding,
    infer_detector_on_crops,
    map_boxes_to_original,
    predict_batched_letterbox,
    predict_yolo_boundaries,
)
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    evaluate_competition_metrics,
    load_yolo_labels,
)


def _load_boundary_model(backend: str, checkpoint: Path, resolution: int) -> Any:
    """加载一个边界检测模型。"""
    if backend == "yolo":
        from ultralytics import YOLO

        return YOLO(str(checkpoint))
    from rfdetr import RFDETR

    return RFDETR.from_checkpoint(str(checkpoint), resolution=resolution)


def _predict_boundaries(
    backend: str,
    model: Any,
    image_paths: list[Path],
    device: str,
    resolution: int,
    conf: float,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    """统一调用 RF-DETR 或 YOLO 边界推理。"""
    if backend == "yolo":
        return predict_yolo_boundaries(model, image_paths, device, resolution, conf, batch_size)
    return predict_batched_letterbox(
        model,
        image_paths,
        device=device,
        resolution=resolution,
        conf_threshold=conf,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _boundary_records(
    results: list[dict[str, Any]],
    image_sizes: dict[str, tuple[int, int]],
    backend: str = "rfdetr",
) -> list[BoxRecord]:
    """把边界模型结果转换成单类 BoxRecord。"""
    records: list[BoxRecord] = []
    for result in results:
        image_id = str(result["image_id"])
        width, height = image_sizes[image_id]
        boxes = map_boxes_to_original(
            np.asarray(result["boxes"]),
            float(result["scale"]),
            int(result["pad_x"]),
            int(result["pad_y"]),
            width,
            height,
        )
        if backend == "yolo" and len(boxes):
            boxes = _nms_boxes(boxes, 0.5)
        scores = np.asarray(result["scores"])
        for box, score in zip(boxes, scores[: len(boxes)], strict=False):
            records.append(BoxRecord(image_id, 0, tuple(float(value) for value in box), float(score)))
    return records


def evaluate_boundary_model(
    backend: str,
    checkpoint: Path,
    image_paths: list[Path],
    image_sizes: dict[str, tuple[int, int]],
    gt_records: list[BoxRecord],
    device: str,
    resolution: int,
    conf: float,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, float], float, Any]:
    """评估边界 AP 并返回已加载模型。"""
    model = _load_boundary_model(backend, checkpoint, resolution)
    # 首轮包含模型运行时初始化，单独计入总耗时但 AP 使用同一结果。
    start = time.perf_counter()
    predictions = _predict_boundaries(
        backend, model, image_paths, device, resolution, conf, batch_size, num_workers
    )
    elapsed = time.perf_counter() - start
    pred_records = _boundary_records(predictions, image_sizes, backend)
    image_ids = [path.stem for path in image_paths]
    ap = compute_coco_ap(
        boxes_to_coco_gt(gt_records, image_ids, image_sizes),
        boxes_to_coco_pred(pred_records, image_ids),
    )
    ap["images_per_second"] = len(image_paths) / elapsed if elapsed > 0 else 0.0
    ap["predictions"] = float(len(pred_records))
    return ap, elapsed, model


def _predict_one_boundary(
    backend: str,
    model: Any,
    image_path: Path,
    device: str,
    resolution: int,
    conf: float,
) -> dict[str, Any]:
    """对单张大图运行边界推理，计时口径与单图业务调用一致。"""
    return _predict_boundaries(backend, model, [image_path], device, resolution, conf, 1, 0)[0]


def evaluate_end_to_end(
    backend: str,
    boundary_model: Any,
    image_paths: list[Path],
    detector_checkpoint: Path,
    dataset_dir: Path,
    device: str,
    boundary_resolution: int,
    boundary_conf: float,
    detector_conf: float,
    padding: int,
    detector_batch_size: int,
) -> dict[str, Any]:
    """在 SHWX 大图上评估最终识别指标和单图端到端速度。"""
    from rfdetr import RFDETR

    detector = RFDETR.from_checkpoint(str(detector_checkpoint))
    image_sizes = build_image_size_map(image_paths)
    gt_records = load_yolo_labels(dataset_dir / "labels" / "test", image_sizes)
    measured: list[float] = []
    pred_records: list[BoxRecord] = []

    # 预热 CUDA kernel 和 Ultralytics 编译路径，不把首次初始化抖动当作业务时延。
    warmup_path = image_paths[0]
    warmup_boundary = _predict_one_boundary(backend, boundary_model, warmup_path, device, boundary_resolution, boundary_conf)
    warmup_image = cv2.imread(str(warmup_path))
    if warmup_image is not None:
        warmup_boxes = map_boxes_to_original(
            np.asarray(warmup_boundary["boxes"]),
            float(warmup_boundary["scale"]),
            int(warmup_boundary["pad_x"]),
            int(warmup_boundary["pad_y"]),
            warmup_image.shape[1],
            warmup_image.shape[0],
        )
        if len(warmup_boxes):
            crop, _ = crop_with_padding(warmup_image, tuple(int(v) for v in warmup_boxes[0]), padding)
            detector.predict([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)], threshold=detector_conf, include_source_image=False)

    for image_path in image_paths:
        start = time.perf_counter()
        boundary = _predict_one_boundary(backend, boundary_model, image_path, device, boundary_resolution, boundary_conf)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        boxes = map_boxes_to_original(
            np.asarray(boundary["boxes"]),
            float(boundary["scale"]),
            int(boundary["pad_x"]),
            int(boundary["pad_y"]),
            width,
            height,
        )
        if backend == "yolo" and len(boxes):
            boxes = _nms_boxes(boxes, 0.5)
        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int, int, int]] = []
        for box in boxes:
            crop, offset = crop_with_padding(image_bgr, tuple(int(value) for value in box), padding)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            offsets.append(offset)
        detections = infer_detector_on_crops(detector, crops, detector_conf, batch_size=detector_batch_size)
        for detection, (x0, y0, _, _) in zip(detections, offsets, strict=False):
            for xyxy, score, class_id in zip(detection.xyxy, detection.confidence, detection.class_id, strict=False):
                pred_records.append(
                    BoxRecord(
                        image_path.stem,
                        int(class_id),
                        (float(xyxy[0] + x0), float(xyxy[1] + y0), float(xyxy[2] + x0), float(xyxy[3] + y0)),
                        float(score),
                    )
                )
        measured.append(time.perf_counter() - start)

    config = EvalConfig(
        class_to_group={**{class_id: "ship" for class_id in range(4)}, **{class_id: "aircraft" for class_id in range(4, 24)}, 24: "vehicle"},
        group_iou_thresholds={"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
        default_iou_threshold=0.50,
        class_aware=True,
    )
    result = evaluate_competition_metrics(gt_records, pred_records, config)
    all_result = result["all"]
    report = {
        "images": len(measured),
        "gt_boxes": len(gt_records),
        "pred_boxes": len(pred_records),
        "recall": all_result.recall,
        "fdr": all_result.fdr,
        "precision": all_result.precision,
        "tp": all_result.tp,
        "fp": all_result.fp,
        "fn": all_result.fn,
        "latency_mean_seconds": statistics.mean(measured) if measured else 0.0,
        "latency_median_seconds": statistics.median(measured) if measured else 0.0,
        "latency_p95_seconds": float(np.percentile(measured, 95)) if measured else 0.0,
        "latency_images_per_second": 1.0 / statistics.mean(measured) if measured and statistics.mean(measured) > 0 else 0.0,
    }
    del detector
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 RF-DETR nano 与 YOLO11m 大图切分模型")
    parser.add_argument("--detector-checkpoint", required=True, type=Path, help="SHWX 小图检测器 checkpoint")
    parser.add_argument("--yolo-checkpoint", type=Path, default=Path("data/large_cut/yolo11m-large-cut.pt"))
    parser.add_argument("--rf-checkpoint", type=Path, default=Path("data/large_cut/rf-detr-nano-large-cut.pth"))
    parser.add_argument("--boundary-dataset", type=Path, default=Path("/home/liu/wzt/datasets/large-cut"))
    parser.add_argument("--shwx-dataset", type=Path, default=Path("/home/liu/wzt/datasets/SHWX-dataset-dict-redo-large"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/large-cut-model-comparison"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rf-resolution", type=int, default=704)
    parser.add_argument("--yolo-resolution", type=int, default=640)
    parser.add_argument("--boundary-ap-conf", type=float, default=0.1)
    parser.add_argument("--boundary-conf", type=float, default=0.25)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """运行双模型评测并保存 JSON 报告。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boundary_images = read_test_image_paths(args.boundary_dataset / "images" / "test")
    boundary_sizes = build_image_size_map(boundary_images)
    boundary_gt = load_yolo_labels(args.boundary_dataset / "labels" / "test", boundary_sizes)
    shwx_all_images = read_test_image_paths(args.shwx_dataset / "images" / "test")
    shwx_sizes = build_image_size_map(shwx_all_images)
    shwx_images = [path for path in shwx_all_images if max(shwx_sizes[path.stem]) >= 2000]
    print(f"[i] SHWX 大图样本: {len(shwx_images)}/{len(shwx_all_images)} 张（长边 >= 2000）", flush=True)
    report: dict[str, Any] = {"config": {key: str(value) for key, value in vars(args).items()}}
    for backend, checkpoint in (("rfdetr", args.rf_checkpoint), ("yolo", args.yolo_checkpoint)):
        resolution = args.rf_resolution if backend == "rfdetr" else args.yolo_resolution
        print(f"[i] 评测边界模型: {backend} -> {checkpoint}", flush=True)
        ap, elapsed, boundary_model = evaluate_boundary_model(
            backend, checkpoint, boundary_images, boundary_sizes, boundary_gt,
            args.device, resolution, args.boundary_ap_conf, args.batch_size, 12,
        )
        print(f"[i] {backend} 边界 AP50={ap['AP50']:.4f}, AP90={ap['AP90']:.4f}", flush=True)
        end_to_end = evaluate_end_to_end(
            backend, boundary_model, shwx_images, args.detector_checkpoint, args.shwx_dataset,
            args.device, resolution, args.boundary_conf, args.detector_conf,
            args.padding, args.detector_batch_size,
        )
        report[backend] = {"boundary": {**ap, "elapsed_seconds": elapsed}, "end_to_end": end_to_end}
        print(f"[i] {backend} 大图 Recall={end_to_end['recall']:.4f}, FDR={end_to_end['fdr']:.4f}, "
              f"平均 {end_to_end['latency_mean_seconds']:.3f}s/张", flush=True)
        del boundary_model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 对比报告: {output_path}")


if __name__ == "__main__":
    main()
