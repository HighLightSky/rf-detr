# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR 训练脚本 —— 使用 Python API，所有参数集中配置。

用法：
    python src/scripts/train.py

修改本文件下方的「训练参数」常量即可切换模型、数据集等配置。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL, AUG_AGGRESSIVE, AUG_CONSERVATIVE, AUG_INDUSTRIAL
from rfdetr.variants import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
MODEL = "medium"          # 可选: nano, small, medium, large

# --- 数据集 ---
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"

# --- 训练超参数 ---
NUM_CLASSES = 25          # 类别数 (SHWX 数据集共 25 类)
EPOCHS = 100              # 训练轮数
BATCH_SIZE = 12            # 每 GPU 的 batch size（medium 模型显存占用较大，设为 4）
NUM_WORKERS = 10           # DataLoader 工作进程数
LR = 1e-4                 # 基础学习率
LR_ENCODER = 1.5e-4       # 编码器（backbone）学习率
WEIGHT_DECAY = 1e-4       # 权重衰减
GRAD_ACCUM_STEPS = 4      # 梯度累积步数（有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS）
CLIP_MAX_NORM = 0.1       # 梯度裁剪

# --- 学习率调度 ---
LR_DROP = 100             # 学习率下降的 epoch 数
WARMUP_EPOCHS = 0.0       # 预热 epoch 数

# --- 数据增广 ---
# 可选预设:
#   None          → 使用默认 torchvision 增广（HorizontalFlip + 多尺度缩放裁剪）
#   AUG_CONSERVATIVE → 保守增广（小数据集推荐，<500 张）
#   AUG_AGGRESSIVE   → 激进增广（大数据集推荐，2000+ 张）
#   AUG_AERIAL       → 航拍/遥感影像（水平/垂直翻转 + 90° 旋转）
#   AUG_INDUSTRIAL   → 工业/检测影像（光照噪声 + 模糊）
#   {}             → 关闭额外增广
AUG_CONFIG = AUG_AERIAL   # SHWX 属航拍数据集，使用航拍预设

# --- 硬件 ---
DEVICE = "cuda"           # 设备: cuda, cuda:0, cpu
DEVICES = 1               # GPU 数量
NUM_NODES = 1             # 节点数

# --- 输出 & 日志 ---
OUTPUT_DIR = "output/0724-shwx-rfdetr_medium"   # 输出目录
TENSORBOARD = True                  # 是否启用 TensorBoard
WANDB = False                       # 是否启用 Wandb

# --- 恢复训练 ---
RESUME = None              # 设为 "output/xxx/last.ckpt" 可从检查点恢复

# ============================================================================
# 模型注册表（无需修改）
# ============================================================================

_MODEL_REGISTRY: dict[str, tuple[type, int]] = {
    "nano": (RFDETRNano, 384),
    "small": (RFDETRSmall, 512),
    "medium": (RFDETRMedium, 640),
    "large": (RFDETRLarge, 768),
}


def _ensure_data_yaml(dataset_dir: str) -> None:
    """若目录只有 dataset.yaml 而无 data.yaml，则创建软链接。"""
    root = Path(dataset_dir)
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        return
    dataset_yaml = root / "dataset.yaml"
    if dataset_yaml.exists():
        data_yaml.symlink_to("dataset.yaml")
        print(f"[信息] 创建软链接: {data_yaml} -> dataset.yaml")
    else:
        print(f"[警告] 数据集中未找到 data.yaml 或 dataset.yaml: {dataset_dir}")


def main() -> None:
    """加载配置并启动训练。"""
    # --- 校验模型和数据集 ---
    if MODEL not in _MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {MODEL}，可选: {list(_MODEL_REGISTRY)}")

    model_cls, default_resolution = _MODEL_REGISTRY[MODEL]
    dataset_dir = str(Path(DATASET_DIR).resolve())

    _ensure_data_yaml(dataset_dir)

    print(f"模型: {MODEL} | 类别数: {NUM_CLASSES} | 分辨率: {default_resolution}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"数据集: {dataset_dir} | 输出: {OUTPUT_DIR}")
    print(f"增广预设: {'无' if AUG_CONFIG is None else AUG_CONFIG}")

    # --- 构建模型 ---
    model = model_cls(num_classes=NUM_CLASSES, resolution=default_resolution)

    # --- 训练 ---
    model.train(
        dataset_dir=dataset_dir,
        dataset_file="yolo",
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        lr_encoder=LR_ENCODER,
        weight_decay=WEIGHT_DECAY,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        clip_max_norm=CLIP_MAX_NORM,
        lr_drop=LR_DROP,
        warmup_epochs=WARMUP_EPOCHS,
        tensorboard=TENSORBOARD,
        wandb=WANDB,
        device=DEVICE,
        devices=DEVICES,
        num_nodes=NUM_NODES,
        resume=RESUME,
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
    )

    print(f"\n训练完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
