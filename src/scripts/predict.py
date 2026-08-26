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
- **标签对比可视化**：配置 ``label_comparison`` 后输出
  ``<output_dir>/label_comparison/<图像名>.jpg``；FP/FN 使用红色标记。

用法：
    python src/scripts/predict.py -c configs/experiments/predict_shwx.yaml

配置结构（predict: 段，详见 configs/experiments/README.md）：

.. code-block:: yaml

    predict:
      checkpoint: output/xxx/checkpoint_best_total.pth   # 相对项目根
      conf: 0.25
      output_dir: output/xxx/predict
      image: /path/to/images
      class_names: shwx        # 内置名 shwx/shwx_truck/dior，或 {label: 名称} 字典
      label_comparison:
        enabled: false
        labels_dir: /path/to/yolo/labels
        iou_threshold: 0.50
      reason_plugin:
        enabled: false         # 默认关闭；开启后才加载插件 checkpoint
        checkpoint: null       # 插件 checkpoint 路径
        class_ids: [24]        # 需要重打分的类别；null 表示全部类别
        conf_low: null         # 候选框下限；null 使用插件 checkpoint 内置值
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from scripts import eval_lib, expcfg  # noqa: E402

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


def _build_reason_plugin_kwargs(predict_cfg: dict[str, object]) -> dict[str, object]:
    """根据 ``predict.reason_plugin`` 配置构造模型推理参数。

    插件默认关闭。关闭时返回空字典，确保调用方不会加载插件或改变普通推理流程。

    Args:
        predict_cfg: yaml ``predict`` 段解析后的配置。

    Returns:
        可直接展开传给 ``RFDETR.predict`` 的插件参数。

    Raises:
        ValueError: 插件开启但缺少 checkpoint，或类别/阈值参数格式错误。
    """
    raw_config = predict_cfg.get("reason_plugin")
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError("predict.reason_plugin 必须是字典配置")
    if not bool(raw_config.get("enabled", False)):
        return {}

    checkpoint = raw_config.get("checkpoint")
    if not checkpoint:
        raise ValueError("predict.reason_plugin.enabled=true 时必须设置 checkpoint")

    class_ids = raw_config.get("class_ids", [24])
    if class_ids is not None:
        if not isinstance(class_ids, (list, tuple)):
            raise ValueError("predict.reason_plugin.class_ids 必须是整数列表或 null")
        try:
            class_ids = tuple(int(class_id) for class_id in class_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("predict.reason_plugin.class_ids 必须只包含整数") from exc

    conf_low = raw_config.get("conf_low")
    if conf_low is not None:
        try:
            conf_low = float(conf_low)
        except (TypeError, ValueError) as exc:
            raise ValueError("predict.reason_plugin.conf_low 必须是数字或 null") from exc
        if not 0.0 <= conf_low <= 1.0:
            raise ValueError("predict.reason_plugin.conf_low 必须位于 [0, 1]")

    return {
        "reason_plugin": checkpoint,
        "reason_class_ids": class_ids,
        "reason_conf_low": conf_low,
    }


def _parse_args() -> argparse.Namespace:
    """解析命令行参数；推理参数全部从配置文件读取。"""
    parser = argparse.ArgumentParser(description="RF-DETR 统一推理模板（yaml 配置）")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="实验 yaml 配置文件路径，推理参数均在其中的 predict 段配置",
    )
    return parser.parse_args()


def _resolve_predict_settings(predict_cfg: dict[str, Any]) -> tuple[str, float, Path, str]:
    """读取并校验推理所需的配置项。

    Args:
        predict_cfg: yaml ``predict`` 段解析后的配置。

    Returns:
        ``(checkpoint, conf_threshold, output_dir, image)``。

    Raises:
        ValueError: 必填项缺失或配置值类型、范围不正确。
    """
    required_values = {
        "checkpoint": predict_cfg.get("checkpoint"),
        "output_dir": predict_cfg.get("output_dir"),
        "image": predict_cfg.get("image"),
    }
    for name, value in required_values.items():
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"predict.{name} 必须在配置文件中设置")

    conf_value = predict_cfg.get("conf")
    if isinstance(conf_value, bool) or not isinstance(conf_value, (int, float)):
        raise ValueError("predict.conf 必须是数字")
    conf_threshold = float(conf_value)
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("predict.conf 必须位于 [0, 1]")

    return (
        str(required_values["checkpoint"]),
        conf_threshold,
        Path(required_values["output_dir"]),
        str(required_values["image"]),
    )


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
    reason_plugin_kwargs: dict[str, object] | None = None,
    two_stage_cfg: eval_lib.TwoStageConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """对单张图片执行推理。

    Args:
        model: 已加载的 RFDETR 实例（任意尺寸，nano/small/medium/large）。
        image_path: 图像路径。
        conf_threshold: 置信度阈值。
        reason_plugin_kwargs: 已解析的插件参数；为空时关闭插件。

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

    predict_kwargs: dict[str, object] = {
        "threshold": min(conf_threshold, two_stage_cfg.candidate_floor) if two_stage_cfg else conf_threshold,
        "include_source_image": False,
    }
    if reason_plugin_kwargs:
        predict_kwargs.update(reason_plugin_kwargs)
    detections = model.predict(image_rgb, **predict_kwargs)
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

    cfg = expcfg.load_config(args.config)
    predict_cfg = expcfg.build_predict_kwargs(cfg)
    checkpoint, conf_threshold, output_dir, image_arg = _resolve_predict_settings(predict_cfg)

    class_names = _resolve_class_names(predict_cfg.get("class_names", "shwx"))
    label_comparison_cfg = eval_lib.LabelComparisonCfg.from_config(predict_cfg.get("label_comparison"))
    if label_comparison_cfg is not None and not label_comparison_cfg.labels_dir.is_dir():
        raise FileNotFoundError(f"YOLO 标签目录不存在: {label_comparison_cfg.labels_dir}")
    reason_plugin_kwargs = _build_reason_plugin_kwargs(predict_cfg)
    two_stage_cfg = eval_lib.TwoStageConfig.from_config(predict_cfg.get("two_stage"))
    if reason_plugin_kwargs and two_stage_cfg is not None:
        raise ValueError("predict.reason_plugin 与 predict.two_stage 不能同时启用")
    ms_nms_config = eval_lib.MsNmsConfig.from_config(predict_cfg.get("ms_nms"))
    ms_nms_total = eval_lib.SuppressionStats()
    if reason_plugin_kwargs:
        print(f"[i] 启用 FFT 一致性插件: {reason_plugin_kwargs['reason_plugin']}")

    image_paths = _collect_image_paths(image_arg)
    # 传入的是文件 → 单图模式（详细打印）；传入的是目录 → 批量模式（逐张一行进度）
    single_mode = Path(image_arg).is_file()

    # 单图/批量共用同一份模型权重（from_checkpoint 自动推断模型尺寸）
    resolved_checkpoint = expcfg.resolve_paths(expcfg.PROJECT_ROOT, checkpoint)
    print(f"[i] 加载 checkpoint: {resolved_checkpoint}")
    model = RFDETR.from_checkpoint(str(resolved_checkpoint))
    class_conf_thresholds = {
        int(class_id): float(threshold)
        for class_id, threshold in (predict_cfg.get("class_conf_thresholds") or {}).items()
    }
    if two_stage_cfg is not None:
        eval_lib._two_stage_collection_thresholds(
            eval_lib.InferenceCfg(conf_threshold=conf_threshold, class_conf_thresholds=class_conf_thresholds),
            two_stage_cfg,
        )
    two_stage_plugin = (
        eval_lib.TwoStagePluginLoader.load(two_stage_cfg, device=eval_lib.resolve_device("cuda:0"))
        if two_stage_cfg is not None
        else None
    )
    two_stage_stats_total = eval_lib.TwoStageStats()

    total_detections = 0
    pred_records: list[eval_lib.BoxRecord] = []
    for index, image_path in enumerate(image_paths, start=1):
        xyxy_array, score_array, class_id_array, image_bgr, (width, height) = _infer_image(
            model,
            image_path,
            conf_threshold,
            reason_plugin_kwargs,
            two_stage_cfg,
        )
        raw_records = [
            eval_lib.BoxRecord(
                image_id=image_path.stem,
                class_id=int(class_id),
                xyxy=tuple(float(value) for value in xyxy),
                score=float(score),
            )
            for xyxy, score, class_id in zip(xyxy_array, score_array, class_id_array)
        ]
        if two_stage_plugin is not None:
            refined_records, stats = two_stage_plugin.refine_records(raw_records, {image_path.stem: image_path})
            two_stage_stats_total.routed += stats.routed
            two_stage_stats_total.candidate_nms_suppressed += stats.candidate_nms_suppressed
            two_stage_stats_total.kept += stats.kept
            two_stage_stats_total.rejected += stats.rejected
            two_stage_stats_total.images += stats.images
            two_stage_stats_total.elapsed_seconds += stats.elapsed_seconds
            two_stage_stats_total.per_image_candidates.update(stats.per_image_candidates)
            refined_records = eval_lib.filter_records_by_thresholds(
                refined_records,
                conf_threshold,
                class_conf_thresholds,
            )
            xyxy_array = np.asarray([record.xyxy for record in refined_records], dtype=np.float32).reshape(-1, 4)
            score_array = np.asarray([record.score for record in refined_records], dtype=np.float32)
            class_id_array = np.asarray([record.class_id for record in refined_records], dtype=int)
            raw_records = refined_records
        filtered_records, image_nms_stats = eval_lib.apply_shwx_ms_nms(raw_records, ms_nms_config)
        ms_nms_total = eval_lib.SuppressionStats(
            input_count=ms_nms_total.input_count + image_nms_stats.input_count,
            output_count=ms_nms_total.output_count + image_nms_stats.output_count,
            same_class_suppressed=ms_nms_total.same_class_suppressed + image_nms_stats.same_class_suppressed,
            cross_class_suppressed=ms_nms_total.cross_class_suppressed + image_nms_stats.cross_class_suppressed,
            ambiguous_cross_class_kept=ms_nms_total.ambiguous_cross_class_kept
            + image_nms_stats.ambiguous_cross_class_kept,
        )
        if ms_nms_config.enabled:
            xyxy_array = np.asarray([record.xyxy for record in filtered_records], dtype=np.float32).reshape(-1, 4)
            score_array = np.asarray([record.score for record in filtered_records], dtype=np.float32)
            class_id_array = np.asarray([record.class_id for record in filtered_records], dtype=int)
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
        pred_records.extend(filtered_records)

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

    if label_comparison_cfg is not None:
        comparison_dir = output_dir / "label_comparison"
        saved_images, fp_count, fn_count = eval_lib.save_yolo_label_comparisons(
            image_paths,
            pred_records,
            class_names,
            comparison_dir,
            label_comparison_cfg,
        )
        print(
            f"[完成] 标签对比图: {comparison_dir}（{saved_images} 张，FP={fp_count}，FN={fn_count}）"
        )

    if ms_nms_config.enabled:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ms_nms_stats.json").write_text(
            json.dumps(ms_nms_total.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[i] MS 保守 NMS: "
            f"输入 {ms_nms_total.input_count}，输出 {ms_nms_total.output_count}，"
            f"同类删除 {ms_nms_total.same_class_suppressed}，"
            f"跨类删除 {ms_nms_total.cross_class_suppressed}，"
            f"歧义保留 {ms_nms_total.ambiguous_cross_class_kept}"
        )

    if two_stage_cfg is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "two_stage_report.json").write_text(
            json.dumps(
                {
                    "backend": two_stage_cfg.backend,
                    "checkpoint": str(two_stage_cfg.checkpoint),
                    "config": {
                        "class_ids": list(two_stage_cfg.class_ids),
                        "candidate_floor": two_stage_cfg.candidate_floor,
                        "candidate_nms_iou": two_stage_cfg.candidate_nms_iou,
                        "context_scale": two_stage_cfg.context_scale,
                        "image_size": two_stage_cfg.image_size,
                        "batch_size": two_stage_cfg.batch_size,
                        "positive_threshold": two_stage_cfg.positive_threshold,
                        "bypass_score": two_stage_cfg.bypass_score,
                        "detector_score_weight": two_stage_cfg.detector_score_weight,
                    },
                    "stats": two_stage_stats_total.to_dict(),
                    "final_count": len(pred_records),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not single_mode:
        print("=" * 60)
        print(f"共处理 {len(image_paths)} 张图片，检出 {total_detections} 个目标，置信度阈值: {conf_threshold}")
        print(f"[完成] YOLO 结果目录: {output_dir / 'labels'}")
        print(f"[完成] 可视化目录:   {output_dir / 'visualization'}")


if __name__ == "__main__":
    main()
