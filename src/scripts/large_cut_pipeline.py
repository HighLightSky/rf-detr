# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切目标检测流水线：边界检测 → 裁切 → 目标检测 → 坐标映射回原图。

完整流程：

1. 原始大图（3000~12000px）等比缩放到 ``--resolution``（默认 704）最长边，
   黑边 padding（与训练数据黑底语义一致）；
2. 边界检测器（rf-detr-nano 矩形预测器，单类 sample）批量预测所有小图矩形；
3. 预测框由 letterbox 空间映射回原图坐标（``--nms-iou`` 可对边界框做 NMS 兜底）；
4. 按每个矩形裁切，四周外扩 ``--padding`` 像素（默认 32，clamp 到图像边界）；
5. 目标检测器（已有 RF-DETR checkpoint，如 SHWX 25 类）批量检测每个裁窗；
6. 检测框坐标 + 裁窗偏移映射回原始大图坐标。

输出：
    - ``output_dir/results.json``：每张原图的裁窗与检测结果（全部为原图坐标）；
    - ``output_dir/visualization/*.jpg``：边界框（青色）+ 检测框（类别色）叠加图。

用法：
    python src/scripts/large_cut_pipeline.py \
        --input /path/to/large_image.jpg \
        --boundary-checkpoint output/0814-large-cut-rfdetr-nano-704-rot90/checkpoint_best_total.pth \
        --detector-checkpoint output/0813-SHWX-rfdetr-medium-baseline-精细标注/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.predict import _draw_detections, _resolve_class_names  # noqa: E402

#: 边界框在可视化中的 BGR 颜色（青色）
BOUNDARY_BOX_COLOR: tuple[int, int, int] = (255, 255, 0)


def letterbox_resize(
    image_bgr: np.ndarray,
    target: int = 704,
) -> tuple[np.ndarray, float, int, int]:
    """等比缩放到 ``target`` 最长边并黑边 padding 到 ``target × target``。

    Args:
        image_bgr: BGR 图像（``cv2.imread`` 输出）。
        target: 目标最长边像素数（须为 32 的倍数，与模型分辨率一致）。

    Returns:
        ``(padded_bgr, scale, pad_x, pad_y)``：padding 后的方形图、缩放比例、
        左右 padding 像素、上下 padding 像素（对称，奇数时右/下多 1px）。
    """
    height, width = image_bgr.shape[:2]
    scale = target / max(height, width)
    new_h = max(round(height * scale), 1)
    new_w = max(round(width * scale), 1)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target - new_w
    pad_h = target - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        top=pad_y,
        bottom=pad_h - pad_y,
        left=pad_x,
        right=pad_w - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # 黑边（与训练数据黑底背景语义一致）
    )
    return padded, scale, pad_x, pad_y


def map_boxes_to_original(
    xyxy: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """把 letterbox 空间的框映射回原图坐标并剔除退化框。

    Args:
        xyxy: ``(N, 4)`` letterbox 空间的 xyxy 框。
        scale: 原图 → letterbox 的缩放比例。
        pad_x: letterbox 左右 padding 像素。
        pad_y: letterbox 上下 padding 像素。
        width: 原图宽度。
        height: 原图高度。

    Returns:
        ``(M, 4)`` 原图坐标 xyxy 框（``M <= N``，越界退化框已被剔除）。
    """
    if xyxy.shape[0] == 0:
        return xyxy
    boxes = (xyxy.astype(np.float64) - np.array([pad_x, pad_y, pad_x, pad_y])) / scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, height)
    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return boxes[valid]


def crop_with_padding(
    image_bgr: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    padding: int = 32,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按矩形裁切图像并四周外扩 padding（clamp 到图像边界）。

    Args:
        image_bgr: BGR 图像。
        box_xyxy: ``(x0, y0, x1, y1)`` 原图坐标的矩形框（像素）。
        padding: 外扩像素数（可为 0）。

    Returns:
        ``(crop_bgr, crop_xyxy)``：裁切图与裁窗在原图中的坐标 ``(x0, y0, x1, y1)``
        （已 clamp，含 padding）。
    """
    image_h, image_w = image_bgr.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    crop_x0 = max(int(x0) - padding, 0)
    crop_y0 = max(int(y0) - padding, 0)
    crop_x1 = min(int(x1) + padding, image_w)
    crop_y1 = min(int(y1) + padding, image_h)
    crop = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    return crop, (crop_x0, crop_y0, crop_x1, crop_y1)


class _InferenceDataset(Dataset):
    """在 DataLoader worker 中读取图像，返回 ``(stem, rgb_tensor, (h, w))``。

    Args:
        image_paths: 待推理图像路径列表。
    """

    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, tuple[int, int]]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C,H,W) uint8 视图
        return image_path.stem, rgb_tensor, (height, width)


def _inference_collate(
    batch: list[tuple[str, torch.Tensor, tuple[int, int]]],
) -> tuple[list[str], list[torch.Tensor], list[tuple[int, int]]]:
    """自定义聚合函数：图像尺寸不同，保持张量列表而非堆叠。"""
    stems = [item[0] for item in batch]
    tensors = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]
    return stems, tensors, orig_sizes


def _worker_init_fn(worker_id: int) -> None:
    """限制 worker 线程数，避免多进程线程争抢过订阅。"""
    del worker_id
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def predict_batched_letterbox(
    model,
    image_paths: list[Path],
    device: str,
    resolution: int,
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
    *,
    square_stretch: bool = False,
) -> list[dict]:
    """批量推理边界检测器（多进程预取 + GPU 批量前向，letterbox 预处理）。

    预处理与训练一致：等比缩放到 ``resolution`` 最长边 + 黑边 padding；
    ``square_stretch=True`` 时退化为直接拉伸到 ``resolution × resolution``
    （与 ``eval_lib.predict_batched_to_records`` 一致），映射时 scale 由
    逐轴比例替代（``pad_x = pad_y = 0``）。

    Args:
        model: 已加载的 RFDETR 实例（边界检测器）。
        image_paths: 待推理图像路径列表。
        device: 推理设备（如 ``"cuda:0"``）。
        resolution: 推理分辨率（须为 32 的倍数，与训练一致）。
        conf_threshold: 置信度阈值。
        batch_size: GPU 单次前向的图像数。
        num_workers: 预取 worker 进程数。
        square_stretch: ``True`` 时用方形拉伸替代 letterbox。

    Returns:
        每张图的 dict 列表：
        ``{"image_id": stem, "boxes": (N,4) letterbox 坐标 xyxy,
        "scores": (N,), "class_ids": (N,), "scale": float, "pad_x": int, "pad_y": int}``。
    """
    dataset = _InferenceDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=_inference_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    # 显式把权重放到目标设备并切换 eval 模式（eval 下解码器仅用单组 query）
    model.model.model = model.model.model.to(device)
    model.model.model.eval()
    model_dtype = next(model.model.model.parameters()).dtype
    means: list[float] = model.means
    stds: list[float] = model.stds

    results: list[dict] = []
    with torch.inference_mode():
        for stems, rgb_tensors, orig_sizes in loader:
            gpu_images: list[torch.Tensor] = []
            scales: list[float] = []
            pad_xy: list[tuple[int, int]] = []
            for tensor, (height, width) in zip(rgb_tensors, orig_sizes):
                img = tensor.to(device, non_blocking=True).to(model_dtype).div_(255.0)
                if square_stretch:
                    # 直接拉伸到方形（scale 语义退化为逐轴比例，padding 为 0）
                    resized = F.resize(img, (resolution, resolution), antialias=False)
                    scales.append(float(width) / resolution)
                    pad_xy.append((0, 0))
                else:
                    scale = resolution / max(height, width)
                    new_h = max(round(height * scale), 1)
                    new_w = max(round(width * scale), 1)
                    resized = F.resize(img, (new_h, new_w), antialias=False)
                    pad_w = resolution - new_w
                    pad_h = resolution - new_h
                    pad_x = pad_w // 2
                    pad_y = pad_h // 2
                    resized = F.pad(resized, (pad_x, pad_y, pad_w - pad_x, pad_h - pad_y), fill=0.0)
                    scales.append(scale)
                    pad_xy.append((pad_x, pad_y))
                gpu_images.append(resized)
            batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)

            predictions = model.model.model(batch_tensor)
            target_sizes = torch.tensor([[resolution, resolution]] * len(stems), device=device)
            post = model.model.postprocess(predictions, target_sizes=target_sizes)

            for stem, result, scale, (pad_x, pad_y) in zip(stems, post, scales, pad_xy):
                keep = result["scores"] > conf_threshold
                results.append(
                    {
                        "image_id": stem,
                        "boxes": result["boxes"][keep].cpu().numpy(),
                        "scores": result["scores"][keep].cpu().numpy(),
                        "class_ids": result["labels"][keep].cpu().numpy(),
                        "scale": scale,
                        "pad_x": pad_x,
                        "pad_y": pad_y,
                    }
                )
    return results


def infer_detector_on_crops(model, crops_rgb: list[np.ndarray], conf: float) -> list:
    """对一组裁窗批量运行目标检测器。

    Args:
        model: 已加载的 RFDETR 实例（目标检测器，如 SHWX 25 类）。
        crops_rgb: RGB 裁窗列表（任意尺寸，内部自动 resize 到模型分辨率）。
        conf: 置信度阈值。

    Returns:
        ``model.predict`` 的 Detections 列表（与 ``crops_rgb`` 一一对应，
        坐标在各自裁窗像素空间）。
    """
    if not crops_rgb:
        return []
    detections = model.predict(crops_rgb, threshold=conf, include_source_image=False)
    if not isinstance(detections, list):
        detections = [detections]
    return detections


def _collect_input_images(input_path: Path) -> list[Path]:
    """收集输入：单文件直接返回，目录则扫描常见图片格式。"""
    if input_path.is_file():
        return [input_path]
    images = sorted([p for p in input_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        raise FileNotFoundError(f"输入目录中未找到图片: {input_path}")
    return images


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="大图裁切目标检测流水线（边界检测 → 裁切+padding → 目标检测 → 映射回原图）"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="原始大图路径，或包含多张图的目录路径",
    )
    parser.add_argument(
        "--boundary-checkpoint",
        type=str,
        required=True,
        help="边界检测器 checkpoint（小图矩形预测器，单类）",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=str,
        required=True,
        help="目标检测器 checkpoint（如 SHWX 25 类检测器）",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=704,
        help="边界检测器输入分辨率（须与边界检测器训练一致，默认 704）",
    )
    parser.add_argument(
        "--boundary-conf",
        type=float,
        default=0.25,
        help="边界框置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="目标检测置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=32,
        help="裁窗外扩像素（默认 32，约边界图 704 的 4.5%%）",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="边界框 NMS 的 IoU 阈值（默认 0 关闭；DETR 后处理无 NMS，>0 时用 cv2.dnn.NMSBoxes 兜底）",
    )
    parser.add_argument(
        "--square-stretch",
        action="store_true",
        help="用方形拉伸替代 letterbox（与训练方形拉伸预处理一致）",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="shwx",
        help="目标检测类别名：shwx/dior/省略，或 {类别id: 名称} JSON 字典",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录（默认与边界检测器 checkpoint 同目录）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="边界检测器 GPU 批量大小（默认 8）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="边界检测器预取 worker 数（默认 4）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推理设备（默认 cuda:0，无 CUDA 自动回退 CPU）",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="不生成可视化叠加图（只写 results.json）",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：加载模型 → 批量边界检测 → 裁切 → 目标检测 → 输出。"""
    from rfdetr import RFDETR

    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.boundary_checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[w] 当前环境未检测到 CUDA，自动改用 CPU 推理", flush=True)
        device = "cpu"
    class_names = _resolve_class_names(args.class_names)

    print(f"[i] 加载边界检测器: {args.boundary_checkpoint}")
    boundary_model = RFDETR.from_checkpoint(args.boundary_checkpoint, resolution=args.resolution)
    print(f"[i] 加载目标检测器: {args.detector_checkpoint}")
    detector_model = RFDETR.from_checkpoint(args.detector_checkpoint)

    image_paths = _collect_input_images(input_path)
    print(f"[i] 待处理大图: {len(image_paths)} 张")

    # ── 1. 批量边界检测（letterbox 预处理）────────────────────────────
    t0 = time.perf_counter()
    pred_results = predict_batched_letterbox(
        boundary_model,
        image_paths,
        device=device,
        resolution=args.resolution,
        conf_threshold=args.boundary_conf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        square_stretch=args.square_stretch,
    )
    print(f"[i] 边界检测完成: {time.perf_counter() - t0:.1f}s")

    all_results: list[dict] = []
    total_detections = 0
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[w] 无法读取图像，跳过: {image_path}")
            continue
        image_h, image_w = image_bgr.shape[:2]
        stem = image_path.stem
        batch_result = next((r for r in pred_results if r["image_id"] == stem), None)
        if batch_result is None:
            continue

        # ── 2. 边界框映射回原图坐标 ─────────────────────────────────
        boxes_orig = map_boxes_to_original(
            batch_result["boxes"],
            batch_result["scale"],
            batch_result["pad_x"],
            batch_result["pad_y"],
            image_w,
            image_h,
        )
        if args.nms_iou > 0 and boxes_orig.shape[0] > 0:
            boxes_orig = _nms_boxes(boxes_orig, args.nms_iou)

        # ── 3-4. 裁切 + padding + 批量目标检测 ───────────────────────
        crops_rgb: list[np.ndarray] = []
        crop_offsets: list[tuple[int, int, int, int]] = []
        for box in boxes_orig:
            crop_bgr, crop_xyxy = crop_with_padding(image_bgr, tuple(int(v) for v in box), args.padding)
            crops_rgb.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            crop_offsets.append(crop_xyxy)

        detections = infer_detector_on_crops(detector_model, crops_rgb, args.detector_conf)

        # ── 5. 检测框映射回原图坐标 ─────────────────────────────────
        crops_json: list[dict] = []
        for box_xyxy, crop_xyxy, det in zip(boxes_orig, crop_offsets, detections):
            crop_x0, crop_y0, _, _ = crop_xyxy
            det_json: list[dict] = []
            for xyxy, score, class_id in zip(det.xyxy, det.confidence, det.class_id):
                x0 = float(xyxy[0]) + crop_x0
                y0 = float(xyxy[1]) + crop_y0
                x1 = float(xyxy[2]) + crop_x0
                y1 = float(xyxy[3]) + crop_y0
                det_json.append(
                    {
                        "class_id": int(class_id),
                        "class_name": class_names.get(int(class_id), str(int(class_id))),
                        "score": float(score),
                        "xyxy": [x0, y0, x1, y1],
                    }
                )
            total_detections += len(det_json)
            crops_json.append(
                {
                    "bbox": [float(v) for v in box_xyxy],
                    "padding": args.padding,
                    "detections": det_json,
                }
            )
        all_results.append({"image": str(image_path), "width": image_w, "height": image_h, "crops": crops_json})

        # ── 可视化（切割矩形青色粗框+编号，检测框类别色）────────────
        if not args.no_vis:
            vis = image_bgr.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(1.2, image_w / 1600))  # 大图字号自适应
            for crop_idx, (box_xyxy, crop_json) in enumerate(zip(boxes_orig, crops_json), start=1):
                x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
                cv2.rectangle(vis, (x0, y0), (x1, y1), BOUNDARY_BOX_COLOR, 3)
                # 编号标签（白字青底，画在框左上角内侧，越界时下移）
                label = f"#{crop_idx}"
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 2)
                label_y = max(y0 + text_h + baseline, text_h + baseline)
                cv2.rectangle(
                    vis,
                    (x0, label_y - text_h - baseline),
                    (x0 + text_w, label_y),
                    BOUNDARY_BOX_COLOR,
                    -1,
                )
                cv2.putText(vis, label, (x0, label_y - baseline), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                # 检测目标框（类别色 + score，复用 predict.py 绘制）
                dets = crop_json["detections"]
                if dets:
                    xyxy_array = np.array([d["xyxy"] for d in dets])
                    score_array = np.array([d["score"] for d in dets])
                    class_id_array = np.array([d["class_id"] for d in dets])
                    _draw_detections(vis, xyxy_array, score_array, class_id_array, class_names)
            vis_dir = output_dir / "visualization"
            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), vis)

        n_tiles = len(boxes_orig)
        print(f"  {stem}: {n_tiles} 个裁窗, {sum(len(c['detections']) for c in crops_json)} 个检测目标")

    # ── 输出 results.json ───────────────────────────────────────────
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[完成] 结果已保存: {results_path}")
    print(
        f"[完成] 共处理 {len(all_results)} 张大图，{sum(len(r['crops']) for r in all_results)} 个裁窗，"
        f"{total_detections} 个检测目标，总耗时 {time.perf_counter() - t0:.1f}s"
    )


def _nms_boxes(boxes_xyxy: np.ndarray, iou_threshold: float) -> np.ndarray:
    """对边界框做 NMS 兜底（cv2.dnn.NMSBoxes，DETR 后处理本身无 NMS）。

    Args:
        boxes_xyxy: ``(N, 4)`` xyxy 框。
        iou_threshold: NMS IoU 阈值。

    Returns:
        抑制后的框（按输入顺序）。
    """
    if boxes_xyxy.shape[0] == 0:
        return boxes_xyxy
    x0, y0, x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    boxes_pixels = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1).astype(np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes_pixels.tolist(),
        [1.0] * len(boxes_pixels),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return boxes_xyxy[:0]
    keep = np.array(indices).reshape(-1)
    return boxes_xyxy[keep]


if __name__ == "__main__":
    main()
