"""对单幅 10000×10000 像素影像执行 CPU 大图推理延迟测试。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_APP = PROJECT_ROOT / "deploy" / "app"
if str(DEPLOY_APP) not in sys.path:
    sys.path.insert(0, str(DEPLOY_APP))

from competition.contracts import InferenceTask, RawDetection  # noqa: E402
from competition.detector.decoding import decode_rfdetr_outputs  # noqa: E402
from competition.preprocess.shwx_large_image import ShwxLargeImagePreprocessor  # noqa: E402
from competition.config import PreprocessConfig  # noqa: E402


def _session(model_path: Path) -> Any:
    """创建 CUDA ONNX Runtime 会话。"""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"当前 ONNX Runtime 不可用 CUDAExecutionProvider: {providers}")
    return ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def _run_session(session: Any, tasks: list[InferenceTask], batch_size: int) -> list[RawDetection]:
    """分批执行 ONNX 前向并解码检测结果。"""
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    outputs = []
    for start in range(0, len(tasks), batch_size):
        current = tasks[start : start + batch_size]
        batch = np.stack([task.pixels for task in current]).astype(np.float32) / 255.0
        batch = (batch - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        batch = np.transpose(batch, (0, 3, 1, 2))
        raw = session.run(output_names, {input_name: batch})
        boxes = next(item for item in raw if item.ndim == 3 and item.shape[-1] == 4)
        logits = next(item for item in raw if item.ndim == 3 and item.shape[-1] != 4)
        outputs.extend(decode_rfdetr_outputs(logits, boxes, current))
    return outputs


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="10000×10000 大图 GPU ONNX 延迟测试")
    parser.add_argument("--input", type=Path, required=True, help="输入大图")
    parser.add_argument("--boundary", type=Path, required=True, help="边界 ONNX 模型")
    parser.add_argument("--detector", type=Path, required=True, help="主检测 ONNX 模型")
    parser.add_argument("--output", type=Path, required=True, help="JSON 结果路径")
    parser.add_argument("--batch-size", type=int, default=8, help="主检测批量大小")
    return parser.parse_args()


def main() -> None:
    """完成单幅大图测试并保存各阶段耗时。"""
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须为正整数")
    import torch

    load_t0 = time.perf_counter()
    boundary_session = _session(args.boundary)
    detector_session = _session(args.detector)
    model_load_s = time.perf_counter() - load_t0

    read_t0 = time.perf_counter()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    image = image[:10000, :10000]
    if image.shape[0] < 10000 or image.shape[1] < 10000:
        image = cv2.copyMakeBorder(
            image,
            0,
            10000 - image.shape[0],
            0,
            10000 - image.shape[1],
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_read_s = time.perf_counter() - read_t0

    preprocessor = ShwxLargeImagePreprocessor(
        PreprocessConfig("shwx_large_image", 2000, 0.25, 0, 0.50, 704)
    )
    prep_t0 = time.perf_counter()
    plan = preprocessor.prepare("benchmark_10000", pixels, 1024, 704)
    boundary_prep_s = time.perf_counter() - prep_t0

    torch.cuda.synchronize()
    boundary_t0 = time.perf_counter()
    boundary_detections = _run_session(boundary_session, [plan.boundary_task], 1)
    torch.cuda.synchronize()
    boundary_infer_s = time.perf_counter() - boundary_t0

    expand_t0 = time.perf_counter()
    main_tasks = preprocessor.expand(plan, boundary_detections, 1024)
    crop_expand_s = time.perf_counter() - expand_t0

    torch.cuda.synchronize()
    main_t0 = time.perf_counter()
    main_detections = _run_session(detector_session, main_tasks, args.batch_size)
    torch.cuda.synchronize()
    main_infer_s = time.perf_counter() - main_t0

    post_t0 = time.perf_counter()
    confidence_detections = [item for item in main_detections if item.score >= 0.25]
    postprocess_s = time.perf_counter() - post_t0
    total_s = boundary_prep_s + boundary_infer_s + crop_expand_s + main_infer_s + postprocess_s
    result = {
        "input_size": [int(pixels.shape[1]), int(pixels.shape[0])],
        "device": "CUDAExecutionProvider (RTX 3090)",
        "batch_size": args.batch_size,
        "model_load_seconds": model_load_s,
        "image_read_seconds": image_read_s,
        "boundary_preprocess_seconds": boundary_prep_s,
        "boundary_inference_seconds": boundary_infer_s,
        "crop_expand_seconds": crop_expand_s,
        "main_inference_seconds": main_infer_s,
        "postprocess_seconds": postprocess_s,
        "inference_total_seconds": total_s,
        "boundary_candidates": len(boundary_detections),
        "crop_count": len(main_tasks),
        "raw_detections": len(main_detections),
        "detections_after_threshold": len(confidence_detections),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
