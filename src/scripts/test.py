# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""统一测试入口。

本文件只读取 YAML、处理命令行覆盖并构造评估配置；批量推理、指标计算、混淆矩阵和
FP/FN 可视化全部由 eval_lib.run_evaluation 完成。常用启动方式是
python src/scripts/test.py -c configs/experiments/test_shwx.yaml，或追加
--set test.batch_size=64 覆盖单项配置。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# 项目根目录和模块路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import eval_lib, expcfg  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（-c 必填，位置参数可覆盖 checkpoint）。"""
    parser = argparse.ArgumentParser(description="RF-DETR 统一测试评估模板（yaml 配置）")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="实验 yaml 配置文件路径（configs/experiments/*.yaml）",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default=None,
        help="可选：覆盖 yaml 中的 checkpoint 路径",
    )
    parser.add_argument(
        "--set",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项（可多次），如 --set test.batch_size=64",
    )
    return parser.parse_args()


def main() -> None:
    """加载 yaml 配置并运行完整评估。"""
    args = _parse_args()
    cfg = expcfg.load_config(args.config)
    cfg = expcfg.apply_overrides(cfg, args.set)
    test_cfg = expcfg.build_test_kwargs(cfg)

    # 解析数据集名称、数据目录和评估输出目录。
    dataset = eval_lib.build_dataset_cfg(
        test_cfg.get("dataset") or "shwx",
        output_dir=test_cfg.get("output_dir"),
        data_dir=test_cfg.get("dataset_dir"),
    )
    if test_cfg.get("dataset_dir"):
        print(f"[i] 数据集根目录覆盖为: {dataset.data_dir}")
    if test_cfg.get("output_dir"):
        print(f"[i] 测试输出目录覆盖为: {dataset.exp_output_dir}")

    # 按命令行、YAML、数据集默认值的优先级确定 checkpoint。
    checkpoint_path = Path(test_cfg.get("checkpoint") or dataset.exp_output_dir / dataset.checkpoint_file)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        print(f"[i] 命令行覆盖 checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    # 只把 InferenceCfg 定义的字段传给统一推理 runtime。
    infer_fields = {f.name for f in dataclasses.fields(eval_lib.InferenceCfg)}
    infer = eval_lib.InferenceCfg(**{k: v for k, v in test_cfg.items() if k in infer_fields})

    # 可选地构造推理侧 LA bias。
    la_bias_cfg: eval_lib.LaBiasCfg | None = None
    la_bias_section = test_cfg.get("la_bias")
    if la_bias_section:
        la_fields = {f.name for f in dataclasses.fields(eval_lib.LaBiasCfg)}
        la_bias_cfg = eval_lib.LaBiasCfg(**{k: v for k, v in la_bias_section.items() if k in la_fields})

    reason_plugin_cfg = eval_lib.ReasonPluginCfg.from_config(test_cfg.get("reason_plugin"))
    two_stage_cfg = eval_lib.TwoStageConfig.from_config(test_cfg.get("two_stage"))
    ms_nms_config = eval_lib.MsNmsConfig.from_config(test_cfg.get("ms_nms"))

    # 可选地构造大图边界检测和 crop 队列配置。
    large_image_cfg: eval_lib.LargeImageCfg | None = None
    if test_cfg.get("large_image_min_side") or test_cfg.get("boundary_checkpoint"):
        large_image_cfg = eval_lib.LargeImageCfg(
            min_side=int(test_cfg.get("large_image_min_side", 2000)),
            boundary_checkpoint=test_cfg.get("boundary_checkpoint"),
            boundary_resolution=int(test_cfg.get("boundary_resolution", 704)),
            boundary_conf=float(test_cfg.get("boundary_conf", 0.25)),
            detector_conf=float(test_cfg.get("detector_conf", 0.25)),
            padding=int(test_cfg.get("tiling_padding", 32)),
            nms_iou=float(test_cfg.get("tiling_nms_iou", 0.0)),
            square_stretch=bool(test_cfg.get("tiling_square_stretch", False)),
            batch_size=int(test_cfg.get("tiling_batch_size", 8)),
            num_workers=int(test_cfg.get("tiling_num_workers", 4)),
            max_pending_crops=int(test_cfg.get("max_pending_crops", 128)),
            roi_backend=str(test_cfg.get("roi_backend", "auto")),
            proxy_max_side=(int(test_cfg["proxy_max_side"]) if test_cfg.get("proxy_max_side") else None),
            roi_output_size=(int(test_cfg["roi_output_size"]) if test_cfg.get("roi_output_size") else None),
            roi_queue_size=int(test_cfg.get("roi_queue_size", 128)),
            roi_cache_dir=test_cfg.get("roi_cache_dir"),
            strict_roi_backend=bool(test_cfg.get("strict_roi_backend", False)),
            inference_mode=str(test_cfg.get("large_image_inference_mode", "mixed")),
        )

    # 可选覆盖推理分辨率；省略时使用 checkpoint 记录的分辨率。
    eval_lib.run_evaluation(
        dataset,
        infer,
        checkpoint_path,
        save_fp_fn=bool(test_cfg.get("save_fp_fn", True)),
        save_yolo_preds=bool(test_cfg.get("save_yolo_preds", False)),
        la_bias=la_bias_cfg,
        reason_plugin_cfg=reason_plugin_cfg,
        two_stage_cfg=two_stage_cfg,
        resolution=int(test_cfg["resolution"]) if test_cfg.get("resolution") else None,
        large_image_cfg=large_image_cfg,
        ms_nms_config=ms_nms_config,
    )


if __name__ == "__main__":
    main()
