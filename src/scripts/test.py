# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""统一测试评估模板：按 yaml 配置在测试集上评估 RF-DETR 模型。

评估管线（批量流水线推理 + 比赛评分 + 混淆矩阵 + FP/FN 可视化 + 报告）在
``eval_lib.run_evaluation`` 中实现，本文件只负责把 yaml 配置解析为参数。

用法：
    python src/scripts/test.py -c configs/experiments/test_shwx.yaml
    # 位置参数覆盖 yaml 中的 checkpoint：
    python src/scripts/test.py -c configs/experiments/test_shwx.yaml output/xxx/checkpoint_best_total.pth
    # --set 覆盖任意配置项：
    python src/scripts/test.py -c configs/experiments/test_shwx.yaml --set test.batch_size=64

配置结构（详见 configs/experiments/README.md）：

.. code-block:: yaml

    test:
      dataset: shwx                     # eval_lib.DATASET_CONFIGS 内置名
      checkpoint: output/xxx/checkpoint_best_total.pth   # 相对项目根；省略用数据集默认
      conf_threshold: 0.25
      class_conf_thresholds: {}         # 可整段贴入 calibrate_thresholds 产物
      device: cuda:0
      batch_size: 32
      num_workers: 12
      use_fp16: false
      output_dir: output/xxx-eval       # 测试输出目录；省略=数据集内置 exp_output_dir
      la_bias:                          # 推理侧 LA bias；省略=关闭
        counts_path: output/xxx/class_counts.json
        k: 1.0
        tau: 0.1
        clip: 1.0
      save_fp_fn: true
      save_yolo_preds: false
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# ── 项目路径 ───────────────────────────────────────────────────────
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

    # 数据集配置：dataset 为内置名（shwx/dior，省略默认 shwx），类别语义来自内置配置；
    # yaml 提供 test.dataset_dir 时覆盖数据集根目录（与训练侧 dataset_dir 同一目录模式），
    # test.output_dir 覆盖输出目录
    dataset = eval_lib.build_dataset_cfg(
        test_cfg.get("dataset") or "shwx",
        output_dir=test_cfg.get("output_dir"),
        data_dir=test_cfg.get("dataset_dir"),
    )
    if test_cfg.get("dataset_dir"):
        print(f"[i] 数据集根目录覆盖为: {dataset.data_dir}")
    if test_cfg.get("output_dir"):
        print(f"[i] 测试输出目录覆盖为: {dataset.exp_output_dir}")

    # checkpoint：命令行 > yaml > 数据集默认（exp_output_dir / checkpoint_file）
    checkpoint_path = Path(test_cfg.get("checkpoint") or dataset.exp_output_dir / dataset.checkpoint_file)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        print(f"[i] 命令行覆盖 checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    # 推理参数：只取 InferenceCfg 字段，yaml 中其余键（dataset/checkpoint 等）忽略
    infer_fields = {f.name for f in dataclasses.fields(eval_lib.InferenceCfg)}
    infer = eval_lib.InferenceCfg(**{k: v for k, v in test_cfg.items() if k in infer_fields})

    # 推理侧 LA bias（省略 la_bias 段 = 关闭）
    la_bias_cfg: eval_lib.LaBiasCfg | None = None
    la_bias_section = test_cfg.get("la_bias")
    if la_bias_section:
        la_fields = {f.name for f in dataclasses.fields(eval_lib.LaBiasCfg)}
        la_bias_cfg = eval_lib.LaBiasCfg(**{k: v for k, v in la_bias_section.items() if k in la_fields})

    # 大图切分测试配置：边界检测器（nano）只加载一次，大图走裁切推理
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
        )

    # test.resolution 可选：显式指定推理输入分辨率（须与训练分辨率一致，
    # 例如 nano 704 训练时填 704），省略则用 checkpoint 记录的分辨率
    eval_lib.run_evaluation(
        dataset,
        infer,
        checkpoint_path,
        save_fp_fn=bool(test_cfg.get("save_fp_fn", True)),
        save_yolo_preds=bool(test_cfg.get("save_yolo_preds", False)),
        la_bias=la_bias_cfg,
        resolution=int(test_cfg["resolution"]) if test_cfg.get("resolution") else None,
        large_image_cfg=large_image_cfg,
    )


if __name__ == "__main__":
    main()
