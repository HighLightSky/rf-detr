# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""按 PAN/RGB 模态路由并合并 SHWX 双模型测试结果。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import eval_lib, expcfg  # noqa: E402
from scripts.dual_shwx import PAN_CLASSES, RGB_CLASSES, _label_classes, prepare_dual_dataset  # noqa: E402
from rfdetr import RFDETR  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析测试命令行参数。"""
    parser = argparse.ArgumentParser(description="SHWX PAN/RGB 双模型合并评估")
    parser.add_argument("-c", "--config", required=True, help="双模型测试 YAML 配置路径")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置项")
    return parser.parse_args()


def _load_model(checkpoint: Path, resolution: int | None) -> RFDETR:
    """加载 checkpoint 并恢复模型分辨率。"""
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    kwargs = {"resolution": resolution} if resolution is not None else {}
    return RFDETR.from_checkpoint(str(checkpoint), **kwargs)


def _predict_one_model(
    checkpoint: Path,
    image_paths: list[Path],
    num_classes: int,
    infer: eval_lib.InferenceCfg,
    resolution: int | None,
) -> tuple[list[eval_lib.BoxRecord], float, int]:
    """运行一个局部类别模型并返回预测记录。"""
    if not image_paths:
        return [], 0.0, 0
    model = _load_model(checkpoint, resolution)
    device = eval_lib.resolve_device(infer.device)
    records, throughput, _, timed_images = eval_lib.predict_batched_to_records(
        model,
        image_paths,
        device,
        conf_threshold=infer.conf_threshold,
        class_conf_thresholds=infer.class_conf_thresholds,
        batch_size=infer.batch_size,
        num_workers=infer.num_workers,
        num_classes=num_classes,
        prefetch_factor=infer.prefetch_factor,
        precision=infer.precision,
        compile_model=infer.compile_model,
        copy_prefetch=infer.copy_prefetch,
        warmup_batches=infer.warmup_batches,
        progress_interval_s=infer.progress_interval_s,
        gpu_monitor_enabled=infer.gpu_monitor_enabled,
    )
    del model
    eval_lib.release_cuda_cache(device)
    return records, throughput, timed_images


def _remap_records(records: list[eval_lib.BoxRecord], class_offset: int) -> list[eval_lib.BoxRecord]:
    """把局部类别预测映射回 SHWX 全局类别。"""
    return [
        eval_lib.BoxRecord(
            image_id=record.image_id,
            class_id=record.class_id + class_offset,
            xyxy=record.xyxy,
            score=record.score,
        )
        for record in records
    ]


def _filter_global_records(
    records: list[eval_lib.BoxRecord], allowed_classes: set[int]
) -> list[eval_lib.BoxRecord]:
    """筛选全局 25 类权重在当前模态允许输出的类别。"""
    return [record for record in records if record.class_id in allowed_classes]


def _class_space(section: dict[str, Any], modality: str) -> str:
    """读取局部或全局 checkpoint 的类别空间配置。"""
    value = str(section.get(f"{modality}_checkpoint_class_space", "local")).lower()
    if value not in {"local", "global"}:
        raise ValueError(f"{modality}_checkpoint_class_space 只能是 local 或 global，当前为: {value}")
    return value


def _write_route_audit(manifest: Any, output_path: Path) -> dict[str, Any]:
    """使用测试标签生成路由审计，不参与路由决策。"""
    counts = {"pan": 0, "rgb": 0, "correct": 0, "incorrect": 0, "ambiguous": 0}
    errors: list[str] = []
    for modality in ("pan", "rgb"):
        for item in manifest.records["test"][modality]:
            counts[modality] += 1
            classes = _label_classes(Path(item["label"]))
            expected = "pan" if classes and classes <= set(PAN_CLASSES) else "rgb"
            if not classes:
                counts["ambiguous"] += 1
            elif expected == modality:
                counts["correct"] += 1
            else:
                counts["incorrect"] += 1
                errors.append(str(item["image"]))
    payload = {
        "threshold": manifest.threshold,
        "calibration_accuracy": manifest.threshold_accuracy,
        "counts": counts,
        "errors": errors,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _save_fp_fn_visualizations(
    dataset: eval_lib.DatasetCfg,
    gt_records: list[eval_lib.BoxRecord],
    pred_records: list[eval_lib.BoxRecord],
    test_image_paths: list[Path],
    output_dir: Path,
    *,
    save_fp_fn: bool,
) -> None:
    """按统一比赛匹配口径保存合并预测的 FP/FN 对照图。"""
    if not save_fp_fn:
        return
    print("[i] 正在生成双模型合并结果的 FP/FN 可视化...")
    fp_dir = output_dir / "FP"
    fn_dir = output_dir / "FN"
    eval_lib.clear_vis_dirs(fp_dir, fn_dir, dataset.class_names)
    fp_images, fn_images, fp_boxes, fn_boxes, tp_preds = eval_lib.match_per_image_per_class(
        gt_records,
        pred_records,
        dataset.num_classes,
        dataset.vehicle_class_ids,
    )
    eval_lib.save_fp_fn_visualizations(
        fp_images,
        fn_images,
        fp_boxes,
        fn_boxes,
        tp_preds,
        gt_records,
        test_image_paths,
        dataset.class_names,
        fp_dir,
        fn_dir,
    )
    print("[完成] 双模型 FP/FN 可视化保存完成")


def _save_yolo_predictions(
    pred_records: list[eval_lib.BoxRecord],
    output_dir: Path,
    image_size_map: dict[str, tuple[int, int]],
    *,
    save_yolo_preds: bool,
) -> None:
    """按配置保存双分支合并后的 YOLO 预测。

    Args:
        pred_records: 合并后的全局类别预测。
        output_dir: 评测输出目录。
        image_size_map: 图像尺寸映射。
        save_yolo_preds: 是否写入 YOLO 预测文件。
    """
    if not save_yolo_preds:
        return
    eval_lib.save_yolo_predictions(pred_records, output_dir / "yolo_preds", image_size_map)


def main() -> None:
    """加载双模型、按模态推理并生成统一比赛报告。"""
    args = _parse_args()
    cfg = expcfg.apply_overrides(expcfg.load_config(args.config), args.set)
    section = expcfg.resolve_paths(PROJECT_ROOT, dict(cfg.get("test_dual", {})))
    dataset_dir = Path(section.get("dataset_dir", "/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test")).resolve()
    cache_dir = Path(section.get("cache_dir", dataset_dir / ".rfdetr_dual_cache" / "shwx_v1")).resolve()
    output_dir = Path(section.get("output_dir", "output/dual-shwx-medium-1024/test_dual")).resolve()
    manifest = prepare_dual_dataset(dataset_dir, cache_dir, rebuild=False)
    pan_checkpoint = Path(section.get("pan_checkpoint", output_dir.parent / "pan" / "checkpoint_best_total.pth"))
    rgb_checkpoint = Path(section.get("rgb_checkpoint", output_dir.parent / "rgb" / "checkpoint_best_total.pth"))
    pan_class_space = _class_space(section, "pan")
    rgb_class_space = _class_space(section, "rgb")
    infer_fields = {field.name for field in dataclasses.fields(eval_lib.InferenceCfg)}
    infer = eval_lib.InferenceCfg(**{key: value for key, value in section.items() if key in infer_fields})
    resolution = int(section["resolution"]) if section.get("resolution") else None

    pan_paths = manifest.paths("test", "pan")
    rgb_paths = manifest.paths("test", "rgb")
    pan_records, pan_throughput, pan_timed = _predict_one_model(
        pan_checkpoint, pan_paths, 25 if pan_class_space == "global" else len(PAN_CLASSES), infer, resolution
    )
    rgb_records, rgb_throughput, rgb_timed = _predict_one_model(
        rgb_checkpoint, rgb_paths, 25 if rgb_class_space == "global" else len(RGB_CLASSES), infer, resolution
    )
    pan_global_records = (
        _filter_global_records(pan_records, set(PAN_CLASSES))
        if pan_class_space == "global"
        else pan_records
    )
    rgb_global_records = (
        _filter_global_records(rgb_records, set(RGB_CLASSES))
        if rgb_class_space == "global"
        else _remap_records(rgb_records, 4)
    )
    pred_records = pan_global_records + rgb_global_records

    dataset = eval_lib.build_dataset_cfg("shwx", data_dir=dataset_dir, output_dir=output_dir)
    test_image_paths = eval_lib.read_test_image_paths(dataset.test_image_dir)
    image_size_map = eval_lib.build_image_size_map(test_image_paths)
    gt_records = eval_lib.load_yolo_labels(dataset.label_dir, image_size_map)
    _save_yolo_predictions(
        pred_records,
        output_dir,
        image_size_map,
        save_yolo_preds=bool(section.get("save_yolo_preds", False)),
    )
    metric_config = eval_lib.EvalConfig(
        class_to_group=dataset.class_to_group,
        group_iou_thresholds=dataset.group_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    eval_results = eval_lib.evaluate_competition_metrics(gt_records, pred_records, metric_config)
    per_class_config = eval_lib.EvalConfig(
        class_to_group=dataset.per_class_to_group,
        group_iou_thresholds=dataset.per_class_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    per_class_results = eval_lib.evaluate_competition_metrics(gt_records, pred_records, per_class_config)
    group_macro = eval_lib.compute_group_macro_averages(
        per_class_results["groups"], eval_lib.build_metric_class_to_group(dataset), dataset.class_names
    )
    total_macro = eval_lib.compute_total_metrics(group_macro)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _write_route_audit(manifest, output_dir / "route_audit.json")
    _save_fp_fn_visualizations(
        dataset,
        gt_records,
        pred_records,
        test_image_paths,
        output_dir,
        save_fp_fn=bool(section.get("save_fp_fn", True)),
    )

    cm = eval_lib.build_confusion_matrix(
        gt_records=gt_records,
        pred_records=pred_records,
        num_classes=dataset.num_classes,
        vehicle_class_ids=dataset.vehicle_class_ids,
    )
    eval_lib.plot_confusion_matrix(cm, dataset.class_names, str(output_dir / "confusion_matrix.png"))
    total_timed = pan_timed + rgb_timed
    total_seconds = (pan_timed / pan_throughput if pan_throughput else 0.0) + (
        rgb_timed / rgb_throughput if rgb_throughput else 0.0
    )
    combined_throughput = total_timed / total_seconds if total_seconds else 0.0
    report = eval_lib.build_test_report(
        dataset_name="shwx-dual",
        checkpoint_path=(
            f"PAN={pan_checkpoint} ({pan_class_space}); "
            f"RGB={rgb_checkpoint} ({rgb_class_space})"
        ),
        test_image_paths=test_image_paths,
        gt_records=gt_records,
        pred_records=pred_records,
        throughput=combined_throughput,
        timed_images=total_timed,
        gpu_util=None,
        eval_results=eval_results,
        group_macro=group_macro,
        total_macro=total_macro,
        per_class_results=per_class_results,
        dataset=dataset,
        infer=infer,
        test_resolution=resolution,
    )
    report.extend(
        [
            "模态路由审计",
            f"路由阈值: {manifest.threshold:.6f}",
            f"路由校准准确率: {manifest.threshold_accuracy:.4f}",
            f"测试路由统计: {json.dumps(audit['counts'], ensure_ascii=False)}",
        ]
    )
    eval_lib.write_test_result(report, output_dir / "test_result.txt")
    print(f"[完成] 双模型合并评估完成: {output_dir}")


if __name__ == "__main__":
    main()
