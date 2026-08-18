# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""统一推理模板：加载 checkpoint 对单张图片或整个目录执行推理。

输出：
- **YOLO 格式预测文件**：``<output_dir>/labels/<图像名>.txt``，每行
  ``class_id x_center y_center width height confidence``（坐标已归一化到 [0,1]），
  与 ``val.competition_metrics.load_yolo_predictions`` 读取的格式完全兼容。
- **推理结果可视化**：``<output_dir>/visualization/<图像名>.jpg``，绘制预测框、
  类别名称与置信度。

用法：
    python src/scripts/predict.py -c configs/experiments/predict_shwx.yaml --image /path/to/image.jpg
    python src/scripts/predict.py --image img.jpg                 # 缺省 -c：内置默认 checkpoint
    python src/scripts/predict.py -c ... --image folder --conf 0.3 --output-dir my_out

配置结构（predict: 段，详见 configs/experiments/README.md）：

.. code-block:: yaml

    predict:
      checkpoint: output/xxx/checkpoint_best_total.pth   # 相对项目根
      conf: 0.25
      output_dir: output/xxx/predict
      image: null              # 通常由 --image 命令行提供
      class_names: shwx        # 内置名 shwx/shwx_truck/dior，或 {label: 名称} 字典
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.sscl.prompts import (  # noqa: E402
    DIOR_CLASS_NAMES,
    SHWX_CLASS_NAMES,
    SHWX_TRUCK_CLASS_NAMES,
)
from scripts import expcfg  # noqa: E402

# 缺省配置（不传 -c 时使用，保持向后兼容）
DEFAULT_CHECKPOINT = "output/0807-SHWX-SSCL-Proj-原型+实例正样本/checkpoint_best_total.pth"
DEFAULT_OUTPUT_DIR = "output/0807-SHWX-SSCL-Proj-原型+实例正样本/predict"
DEFAULT_CONF_THRESHOLD = 0.25

# 细粒度类别名称映射：SHWX 类别 id 0-24 已连续，label 即 class_id
_DEFAULT_CLASS_NAMES: dict[int, str] = {
    label: SHWX_CLASS_NAMES[cid] for label, cid in enumerate(sorted(SHWX_CLASS_NAMES))
}


def _resolve_class_names(value) -> dict[int, str]:
    """解析类别名配置：内置名（shwx/shwx_truck/dior）或 ``{label: 名称}`` 字典。

    Args:
        value: yaml ``predict.class_names`` 的值（``"shwx"``/``"dior"``/字典/省略）。

    Returns:
        ``{label: 名称}`` 映射。
    """
    if isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}
    if value == "dior":
        return {label: DIOR_CLASS_NAMES[cid] for label, cid in enumerate(sorted(DIOR_CLASS_NAMES))}
    if value == "shwx_truck":
        return {label: SHWX_TRUCK_CLASS_NAMES[cid] for label, cid in enumerate(sorted(SHWX_TRUCK_CLASS_NAMES))}
    return _DEFAULT_CLASS_NAMES


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（--image 必填，其余可经 yaml/命令行覆盖）。"""
    parser = argparse.ArgumentParser(description="RF-DETR 统一推理模板（yaml 配置）")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="实验 yaml 配置文件路径；缺省使用内置默认 checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="输入图片路径，或包含图片的目录路径（自动扫描 .jpg/.jpeg/.png 批量推理）",
    )
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值（覆盖 yaml）")
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录（覆盖 yaml）")
    return parser.parse_args()


def _xyxy_to_yolo(xyxy: np.ndarray, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """把 xyxy 像素坐标框转换为 YOLO 归一化格式。

    Args:
        xyxy: ``[x1, y1, x2, y2]`` 像素坐标框。
        image_width: 图像宽度。
        image_height: 图像高度。

    Returns:
        ``(x_center, y_center, width, height)`` 归一化到 [0,1] 的元组。
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    # 裁剪到 [0,1]，避免浮点误差越界
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(width, 0.0), 1.0),
        min(max(height, 0.0), 1.0),
    )


def _box_color(class_id: int) -> tuple[int, int, int]:
    """根据类别 id 生成稳定的 BGR 颜色。

    Args:
        class_id: 类别索引。

    Returns:
        BGR 三元组颜色。
    """
    palette = [
        (0, 140, 255),  # 橙
        (0, 255, 0),  # 绿
        (255, 0, 0),  # 蓝
        (0, 255, 255),  # 黄
        (255, 0, 255),  # 品红
        (255, 255, 0),  # 青
        (0, 128, 255),  # 浅橙
        (128, 0, 255),  # 紫
    ]
    return palette[class_id % len(palette)]


def _draw_detections(
    image: np.ndarray,
    xyxy_array: np.ndarray,
    score_array: np.ndarray,
    class_id_array: np.ndarray,
    class_names: dict[int, str],
) -> np.ndarray:
    """在 BGR 图像上绘制预测框、类别名称与置信度。

    Args:
        image: BGR 图像（原地绘制）。
        xyxy_array: ``(N, 4)`` 的 xyxy 像素坐标框。
        score_array: ``(N,)`` 的置信度。
        class_id_array: ``(N,)`` 的类别索引。
        class_names: ``{类别id: 名称}`` 映射。

    Returns:
        绘制后的图像。
    """
    image_height, image_width = image.shape[:2]
    for xyxy, score, class_id in zip(xyxy_array, score_array, class_id_array):
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        color = _box_color(int(class_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_names.get(int(class_id), str(int(class_id)))} {float(score):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
        # 文字底色，避免与背景混淆；框顶超出画布时放到框内
        label_y1 = max(y1 - text_height - baseline, 0)
        cv2.rectangle(
            image,
            (x1, label_y1),
            (min(x1 + text_width, image_width - 1), label_y1 + text_height + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, label_y1 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _collect_image_paths(image_arg: str | Path) -> list[Path]:
    """把命令行传入的图像路径展开为待推理的图像文件列表。

    传入目录时自动扫描其中的 ``.jpg/.jpeg/.png`` 文件（非递归，按文件名排序）。

    Args:
        image_arg: 图像文件路径，或包含图像的目录路径。

    Returns:
        待推理的图像文件路径列表。

    Raises:
        FileNotFoundError: 当路径不存在，或目录中没有图像文件时抛出。
    """
    path = Path(image_arg)
    if path.is_dir():
        images = sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")) + sorted(path.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"目录中未找到 .jpg/.jpeg/.png 图像: {path}")
        return images
    if not path.exists():
        raise FileNotFoundError(f"输入图片不存在: {path}")
    return [path]


def _infer_image(
    model: RFDETR,
    image_path: Path,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """对单张图片执行推理。

    Args:
        model: 已加载的 RFDETR 实例（任意尺寸，nano/small/medium/large）。
        image_path: 图像路径。
        conf_threshold: 置信度阈值。

    Returns:
        ``(xyxy_array, score_array, class_id_array, image_bgr, (width, height))`` 元组：
        预测框、置信度、类别索引、原图 BGR 数组（用于可视化）与图像尺寸。
    """
    # 读图：cv2 得到 BGR，推理需要 RGB；可视化沿用 BGR 原图
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    width, height = image_bgr.shape[1], image_bgr.shape[0]

    detections = model.predict(image_rgb, threshold=conf_threshold, include_source_image=False)
    if detections is None or len(detections) == 0:
        xyxy_array = np.empty((0, 4), dtype=np.float32)
        score_array = np.empty((0,), dtype=np.float32)
        class_id_array = np.empty((0,), dtype=int)
    else:
        xyxy_array = detections.xyxy
        score_array = detections.confidence
        class_id_array = detections.class_id
    return xyxy_array, score_array, class_id_array, image_bgr, (width, height)


def _save_results(
    image_path: Path,
    xyxy_array: np.ndarray,
    score_array: np.ndarray,
    class_id_array: np.ndarray,
    image_bgr: np.ndarray,
    output_dir: Path,
    class_names: dict[int, str],
) -> tuple[Path, Path]:
    """保存单张图片的 YOLO 结果文件与可视化图像。

    Args:
        image_path: 原图像路径（用于取图像名）。
        xyxy_array: ``(N, 4)`` 的 xyxy 像素坐标框。
        score_array: ``(N,)`` 的置信度。
        class_id_array: ``(N,)`` 的类别索引。
        image_bgr: 原图 BGR 数组。
        output_dir: 结果输出根目录（内含 ``labels/`` 与 ``visualization/`` 子目录）。
        class_names: ``{类别id: 名称}`` 映射（可视化标签用）。

    Returns:
        ``(label_path, vis_path)`` 保存路径二元组。
    """
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visualization"
    labels_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 写 YOLO 格式结果文件：class_id xc yc w h confidence
    label_path = labels_dir / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        for xyxy, score, class_id in zip(xyxy_array, score_array, class_id_array):
            xc, yc, w, h = _xyxy_to_yolo(xyxy, image_bgr.shape[1], image_bgr.shape[0])
            f.write(f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {float(score):.6f}\n")

    # 可视化：绘制预测框与类别标签
    annotated = _draw_detections(image_bgr, xyxy_array, score_array, class_id_array, class_names)
    vis_path = vis_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(str(vis_path), annotated)
    return label_path, vis_path


def main() -> None:
    """加载模型，对单张图片或整个目录中的图片推理并保存 YOLO 结果与可视化。"""
    args = _parse_args()

    # ── 配置来源：yaml（可选）> 命令行覆盖 > 内置默认 ────────────────
    predict_cfg: dict = {}
    if args.config:
        cfg = expcfg.load_config(args.config)
        predict_cfg = expcfg.build_predict_kwargs(cfg)
    checkpoint = predict_cfg.get("checkpoint") or DEFAULT_CHECKPOINT
    conf_threshold = args.conf if args.conf is not None else predict_cfg.get("conf", DEFAULT_CONF_THRESHOLD)
    output_dir = Path(args.output_dir or predict_cfg.get("output_dir") or DEFAULT_OUTPUT_DIR)
    image_arg = args.image or predict_cfg.get("image")
    if not image_arg:
        raise SystemExit("缺少 --image 参数（或 yaml 的 predict.image）: 请输入图片路径或目录")

    class_names = _resolve_class_names(predict_cfg.get("class_names", "shwx"))

    image_paths = _collect_image_paths(image_arg)
    # 传入的是文件 → 单图模式（详细打印）；传入的是目录 → 批量模式（逐张一行进度）
    single_mode = Path(image_arg).is_file()

    # 单图/批量共用同一份模型权重（from_checkpoint 自动推断模型尺寸）
    resolved_checkpoint = expcfg.resolve_paths(expcfg.PROJECT_ROOT, checkpoint)
    print(f"[i] 加载 checkpoint: {resolved_checkpoint}")
    model = RFDETR.from_checkpoint(str(resolved_checkpoint))

    total_detections = 0
    for index, image_path in enumerate(image_paths, start=1):
        xyxy_array, score_array, class_id_array, image_bgr, (width, height) = _infer_image(
            model,
            image_path,
            conf_threshold,
        )
        label_path, vis_path = _save_results(
            image_path,
            xyxy_array,
            score_array,
            class_id_array,
            image_bgr,
            output_dir,
            class_names,
        )
        total_detections += len(xyxy_array)

        if single_mode:
            # ── 单图模式：逐目标详细打印 ──────────────────────────────
            print("=" * 60)
            print(f"图像: {image_path}")
            print(f"尺寸: {width} x {height}")
            print(f"检测目标数: {len(xyxy_array)}  置信度阈值: {conf_threshold}")
            for xyxy, score, class_id in zip(xyxy_array, score_array, class_id_array):
                name = class_names.get(int(class_id), str(int(class_id)))
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                print(f"  [{name:>8s}] score={float(score):.3f}  xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            print("=" * 60)
            print(f"[完成] YOLO 结果: {label_path}")
            print(f"[完成] 可视化:   {vis_path}")
        else:
            # ── 批量模式：逐张一行进度 ────────────────────────────────
            print(f"[{index:>4d}/{len(image_paths)}] {image_path.name}: {len(xyxy_array)} 个目标")

    if not single_mode:
        print("=" * 60)
        print(f"共处理 {len(image_paths)} 张图片，检出 {total_detections} 个目标，置信度阈值: {conf_threshold}")
        print(f"[完成] YOLO 结果目录: {output_dir / 'labels'}")
        print(f"[完成] 可视化目录:   {output_dir / 'visualization'}")


if __name__ == "__main__":
    main()
