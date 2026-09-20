#!/usr/bin/env python3
"""绘制多模态原型位置引导热力图。

该脚本只支持未优化的 PyTorch checkpoint。它从 ProtoGuidance 的 encoder
dense token 分数恢复多尺度 ``H×W`` 网格，并输出目标类别相似度、类别
margin 以及最终 query selection score 三类叠加图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from rfdetr import RFDETR

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_flattened_scores(
    values: np.ndarray,
    spatial_shapes: Sequence[Sequence[int]],
    level_start_index: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """把展平的多尺度分数恢复成逐层 ``H×W`` 网格。

    Args:
        values: 一维展平分数。
        spatial_shapes: 每层的 ``(H, W)``。
        level_start_index: 可选的每层起始索引，用于校验 flatten 顺序。

    Returns:
        按输入顺序排列的二维分数数组。

    Raises:
        ValueError: 输入不是一维、尺寸非法、起始索引不一致或长度不匹配。
    """
    flat = np.asarray(values)
    if flat.ndim != 1:
        raise ValueError(f"values 必须是一维数组，收到形状 {flat.shape}。")

    shapes: list[tuple[int, int]] = []
    starts: list[int] = []
    cursor = 0
    for shape in spatial_shapes:
        if len(shape) != 2:
            raise ValueError(f"空间形状必须是 (H, W)，收到 {shape!r}。")
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"空间尺寸必须为正数，收到 {(height, width)}。")
        shapes.append((height, width))
        starts.append(cursor)
        cursor += height * width

    if cursor != flat.shape[0]:
        raise ValueError(f"token 数量不匹配：空间形状需要 {cursor}，实际收到 {flat.shape[0]}。")
    if level_start_index is not None:
        provided = [int(value) for value in level_start_index]
        if provided != starts:
            raise ValueError(f"level_start_index 不匹配：期望 {starts}，实际收到 {provided}。")

    return [
        flat[start : start + height * width].reshape(height, width)
        for start, (height, width) in zip(starts, shapes)
    ]


def normalize_positive(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    """使用分位数把非负向热力图稳健归一化到 ``[0, 1]``。"""
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    valid = array[finite]
    lo, hi = np.percentile(valid, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    result = np.zeros_like(array, dtype=np.float32)
    result[finite] = np.clip((valid - lo) / (hi - lo), 0.0, 1.0)
    return result


def normalize_margin(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    """按正负两侧分别拉伸 margin，稳健归一化到 ``[-1, 1]``。

    对于目标类别整体落后于竞争类别的图，负值侧也会使用完整的蓝色
    动态范围，避免整张图因少量正值或绝对最大值而显示成近似单色。
    """
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    result = np.zeros_like(array, dtype=np.float32)
    valid = array[finite]

    negative = valid < 0.0
    if negative.any():
        negative_values = valid[negative]
        negative_lo, negative_hi = np.percentile(negative_values, [lower, upper])
        negative_span = float(negative_hi - negative_lo)
        if not np.isfinite(negative_span) or negative_span <= 1e-6:
            finite_result = result[finite]
            finite_result[negative] = -1.0
            result[finite] = finite_result
        else:
            result_negative = np.clip(
                (negative_values - float(negative_hi)) / negative_span,
                -1.0,
                0.0,
            )
            finite_result = result[finite]
            finite_result[negative] = result_negative
            result[finite] = finite_result

    positive = valid > 0.0
    if positive.any():
        positive_values = valid[positive]
        positive_lo, positive_hi = np.percentile(positive_values, [lower, upper])
        positive_span = float(positive_hi - positive_lo)
        if not np.isfinite(positive_span) or positive_span <= 1e-6:
            finite_result = result[finite]
            finite_result[positive] = 1.0
            result[finite] = finite_result
        else:
            result_positive = np.clip(
                (positive_values - float(positive_lo)) / positive_span,
                0.0,
                1.0,
            )
            finite_result = result[finite]
            finite_result[positive] = result_positive
            result[finite] = finite_result
    return result


def enhance_margin_contrast(values: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    """使用保留符号的幂变换增强 margin 对比度。"""
    if gamma <= 0.0:
        raise ValueError(f"gamma 必须为正数，收到 {gamma}。")
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    result = np.zeros_like(array, dtype=np.float32)
    result[finite] = np.sign(array[finite]) * np.power(np.abs(array[finite]), gamma)
    return np.clip(result, -1.0, 1.0)


def fuse_level_maps(
    level_maps: Sequence[np.ndarray],
    output_size: tuple[int, int],
    *,
    signed: bool = False,
) -> np.ndarray:
    """逐层归一化、上采样并等权融合热力图。"""
    if not level_maps:
        raise ValueError("level_maps 不能为空。")
    output_height, output_width = output_size
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"output_size 必须为正数，收到 {output_size}。")
    resized: list[np.ndarray] = []
    for level_map in level_maps:
        normalized = normalize_margin(level_map) if signed else normalize_positive(level_map)
        resized.append(
            cv2.resize(normalized, (output_width, output_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        )
    return np.mean(np.stack(resized, axis=0), axis=0)


def target_class_margin(proto_logits: np.ndarray, class_id: int) -> np.ndarray:
    """计算目标类别相对于最强竞争类别的 margin。"""
    logits = np.asarray(proto_logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"proto_logits 必须是二维数组，收到形状 {logits.shape}。")
    if not 0 <= class_id < logits.shape[1]:
        raise ValueError(f"class_id 必须在 [0, {logits.shape[1] - 1}] 内，收到 {class_id}。")
    target = logits[:, class_id]
    competitors = np.delete(logits, class_id, axis=1)
    return target if competitors.shape[1] == 0 else target - competitors.max(axis=1)


def read_image(path: Path) -> np.ndarray:
    """以 BGR 格式读取图片，并兼容非 ASCII 文件路径。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def colorize_overlay(
    image_bgr: np.ndarray,
    score_map: np.ndarray,
    *,
    alpha: float,
    signed: bool = False,
) -> np.ndarray:
    """将归一化分数着色并叠加到 BGR 原图。"""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1] 内，收到 {alpha}。")
    height, width = image_bgr.shape[:2]
    if score_map.shape != (height, width):
        score_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
    if signed:
        score_map = enhance_margin_contrast(score_map)
        color_input = np.clip((score_map + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(color_input, _diverging_lut())
    else:
        color_input = np.clip(score_map * 255.0, 0.0, 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(color_input, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def _diverging_lut() -> np.ndarray:
    """构造蓝-白-红的发散色图查找表。"""
    positions = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    blue = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
    white = np.asarray([255.0, 255.0, 255.0], dtype=np.float32)
    red = np.asarray([0.0, 0.0, 255.0], dtype=np.float32)
    lut = np.empty((256, 1, 3), dtype=np.uint8)
    for index, position in enumerate(positions):
        if position <= 0.5:
            color = blue + (white - blue) * (position * 2.0)
        else:
            color = white + (red - white) * ((position - 0.5) * 2.0)
        lut[index, 0] = np.clip(color, 0.0, 255.0).astype(np.uint8)
    return lut


def _image_paths(images: Path) -> list[Path]:
    """解析单张图片或图片目录。"""
    if images.is_file():
        if images.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的图片后缀: {images.suffix}")
        return [images]
    if not images.is_dir():
        raise FileNotFoundError(f"图片路径不存在: {images}")
    paths = sorted(path for path in images.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"目录中没有支持的图片: {images}")
    return paths


def _parse_levels(value: str, count: int) -> list[int]:
    """解析 ``all`` 或逗号分隔的层索引。"""
    if value.strip().lower() == "all":
        return list(range(count))
    try:
        levels = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"levels 必须是 all 或逗号分隔的整数，收到 {value!r}。") from exc
    if not levels or any(level < 0 or level >= count for level in levels):
        raise ValueError(f"level 索引超出范围 [0, {count - 1}]：{levels}")
    return levels


def _save_image(path: Path, image_bgr: np.ndarray) -> None:
    """保存图片并检查编码结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"无法保存图片: {path}")


def visualize_image(
    model: RFDETR,
    image_path: Path,
    output_dir: Path,
    class_id: int,
    *,
    resolution: int | None,
    levels: str,
    save_raw: bool,
    alpha: float,
    checkpoint: Path,
) -> Path:
    """为单张图片生成三类多模态原型热力图。"""
    image_bgr = read_image(image_path)
    shape = (resolution, resolution) if resolution is not None else None
    model.predict(str(image_path), threshold=0.0, shape=shape, include_source_image=False)
    debug = model.get_proto_guidance_debug()
    if debug is None:
        raise RuntimeError("模型没有产生 ProtoGuidance debug 数据，请确认模块已启用。")

    proto_logits = np.asarray(debug["proto_logits"].detach().cpu())[0]
    proto_score = np.asarray(debug["proto_score"].detach().cpu())[0]
    linear_score = np.asarray(debug["linear_score"].detach().cpu())[0]
    select_score = np.asarray(debug["select_score"].detach().cpu())[0]
    spatial_shapes = debug["spatial_shapes"]
    starts = np.asarray(debug["level_start_index"].detach().cpu()).tolist()
    input_size = tuple(int(value) for value in debug["input_size"])
    if proto_logits.ndim != 2:
        raise ValueError(f"proto_logits 必须是 [N, C]，收到 {proto_logits.shape}。")
    if not 0 <= class_id < proto_logits.shape[1]:
        raise ValueError(f"class_id 必须在 [0, {proto_logits.shape[1] - 1}] 内，收到 {class_id}。")

    target = proto_logits[:, class_id]
    margin = target_class_margin(proto_logits, class_id)
    target_levels = split_flattened_scores(target, spatial_shapes, starts)
    margin_levels = split_flattened_scores(margin, spatial_shapes, starts)
    select_levels = split_flattened_scores(select_score, spatial_shapes, starts)
    selected_levels = _parse_levels(levels, len(target_levels))

    model_height, model_width = input_size
    selected_target = [target_levels[index] for index in selected_levels]
    selected_margin = [margin_levels[index] for index in selected_levels]
    selected_select = [select_levels[index] for index in selected_levels]
    target_map = fuse_level_maps(selected_target, (model_height, model_width))
    margin_map = fuse_level_maps(selected_margin, (model_height, model_width), signed=True)
    select_map = fuse_level_maps(selected_select, (model_height, model_width))

    original_size = image_bgr.shape[:2]
    target_map = cv2.resize(target_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    margin_map = cv2.resize(margin_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    select_map = cv2.resize(select_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

    image_output_dir = output_dir / image_path.stem
    _save_image(
        image_output_dir / f"target_c{class_id}_similarity.jpg",
        colorize_overlay(image_bgr, target_map, alpha=alpha),
    )
    _save_image(
        image_output_dir / f"target_c{class_id}_margin.jpg",
        colorize_overlay(image_bgr, margin_map, alpha=alpha, signed=True),
    )
    _save_image(image_output_dir / "selection_score.jpg", colorize_overlay(image_bgr, select_map, alpha=alpha))

    for index in selected_levels:
        level_target = normalize_positive(target_levels[index])
        level_margin = normalize_margin(margin_levels[index])
        level_select = normalize_positive(select_levels[index])
        level_dir = image_output_dir / "levels"
        level_dir.mkdir(parents=True, exist_ok=True)
        level_target = cv2.resize(level_target, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        level_margin = cv2.resize(level_margin, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        level_select = cv2.resize(level_select, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        _save_image(
            level_dir / f"level{index}_target_c{class_id}_similarity.jpg",
            colorize_overlay(image_bgr, level_target, alpha=alpha),
        )
        _save_image(
            level_dir / f"level{index}_target_c{class_id}_margin.jpg",
            colorize_overlay(image_bgr, level_margin, alpha=alpha, signed=True),
        )
        _save_image(
            level_dir / f"level{index}_selection_score.jpg",
            colorize_overlay(image_bgr, level_select, alpha=alpha),
        )

    if save_raw:
        np.savez_compressed(
            image_output_dir / "raw.npz",
            proto_logits=proto_logits,
            proto_score=proto_score,
            linear_score=linear_score,
            select_score=select_score,
            selected_class=np.asarray(debug["selected_class"].detach().cpu())[0],
            spatial_shapes=np.asarray(spatial_shapes, dtype=np.int64),
            level_start_index=np.asarray(starts, dtype=np.int64),
        )
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "class_id": class_id,
        "input_size": list(input_size),
        "original_size": list(original_size),
        "spatial_shapes": [list(shape) for shape in spatial_shapes],
        "level_start_index": starts,
        "levels": selected_levels,
        "normalization": "per_level_signed_margin_p2_p98",
        "alpha": alpha,
    }
    (image_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return image_output_dir


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="未优化的 PyTorch checkpoint")
    parser.add_argument("--images", type=Path, required=True, help="单张图片或图片目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="热力图输出目录")
    parser.add_argument("--class-id", type=int, required=True, help="目标前景类别索引")
    parser.add_argument("--resolution", type=int, default=None, help="可选的正方形推理分辨率")
    parser.add_argument("--levels", default="all", help="all 或逗号分隔的 level 索引，例如 0,1")
    parser.add_argument("--save-raw", action="store_true", help="保存原始 dense 分数到 raw.npz")
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度，默认 0.45")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """加载模型并为输入图片生成热力图。"""
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    if args.resolution is not None and args.resolution <= 0:
        raise ValueError(f"resolution 必须为正数，收到 {args.resolution}。")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1] 内，收到 {args.alpha}。")

    model = RFDETR.from_checkpoint(str(args.checkpoint))
    model.enable_proto_guidance_debug()
    try:
        for image_path in _image_paths(args.images):
            output = visualize_image(
                model,
                image_path,
                args.output_dir,
                args.class_id,
                resolution=args.resolution,
                levels=args.levels,
                save_raw=args.save_raw,
                alpha=args.alpha,
                checkpoint=args.checkpoint,
            )
            print(f"[完成] {image_path} -> {output}")
    finally:
        model.disable_proto_guidance_debug()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
