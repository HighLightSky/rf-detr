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

from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRLargeDeprecated, RFDETRMedium, RFDETRNano, RFDETRSmall

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
# 实验一：少数类重采样（对照 0805-SHWX-data-expand-rfdetr-baseline，唯一变量为重采样开关）
MODEL = "medium"          # 可选: nano, small, medium, large

# --- 数据集 ---
DATASET_DIR = "/home/liu/wzt/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"            # roboflow：Roboflow COCO 格式 (train/_annotations.coco.json)，还有coco yolo

# --- 训练超参数 ---
NUM_CLASSES = 25          # 类别数
EPOCHS = 100              # 训练轮数
BATCH_SIZE = 16           # 每 GPU 的 batch size（与 0805 基线一致）
NUM_WORKERS = 12           # DataLoader 工作进程数
LR = 1e-4                 # 基础学习率
LR_ENCODER = 1.5e-4       # 编码器（backbone）学习率
WEIGHT_DECAY = 1e-4       # 权重衰减
GRAD_ACCUM_STEPS = 4      # 梯度累积步数（有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 64，与 0805 基线一致）
CLIP_MAX_NORM = 0.1       # 梯度裁剪

# --- 学习率调度 ---
LR_DROP = 60             # 学习率下降的 epoch 数
WARMUP_EPOCHS = 0.0       # 预热 epoch 数

# --- 数据增广 ---
# 可选预设:
#   None          → 使用默认 torchvision 增广（HorizontalFlip + 多尺度缩放裁剪）
#   AUG_CONSERVATIVE → 保守增广（小数据集推荐，<500 张）
#   AUG_AGGRESSIVE   → 激进增广（大数据集推荐，2000+ 张）
#   AUG_AERIAL       → 航拍/遥感影像（水平/垂直翻转 + 90° 旋转）
#   AUG_INDUSTRIAL   → 工业/检测影像（光照噪声 + 模糊）
#   {}             → 关闭额外增广
AUG_CONFIG = AUG_AERIAL   # 遥感、航拍预设

# --- Mosaic 增强 ---
# Mosaic 将 4 张图片拼接为 1 张训练样本，有效提升小目标检测和遥感数据集的性能。
# 推荐值: 0.5 (前 80% 训练阶段开启，最后 20% 关闭)
# 设为 0.0 关闭
MOSAIC_P = 0.8

# --- 少数类重采样（实验性）---
# 将训练集长度扩展 RARE_CLASS_OVERSAMPLE_FACTOR 倍，扩展段循环映射到包含
# RARE_CLASS_OVERSAMPLE_CLASS_IDS 中任意类别的图片，提升稀有类训练参与度
# （SHWX: HM 航母仅 6 框、LQS 两栖舰仅 15 框，训练中几乎不会被采样到）。
RARE_CLASS_OVERSAMPLE = True           # 重采样开关（本实验开启）
RARE_CLASS_OVERSAMPLE_FACTOR = 2       # 倍率（总长度 = 原长 × factor，可做 2/4/8 消融）
RARE_CLASS_OVERSAMPLE_CLASS_IDS = [0, 1]   # 0=HM 航母, 1=LQS 两栖舰

# --- 硬件 ---
DEVICE = "cuda"           # 设备: cuda, cuda:0, cpu
DEVICES = 1               # GPU 数量
NUM_NODES = 1             # 节点数

# --- 输出 & 日志 ---
OUTPUT_DIR = "output/0810-SHWX-rfdetr-medium-rare-oversample"   # 输出目录
TENSORBOARD = True                  # 是否启用 TensorBoard
WANDB = False                       # 是否启用 Wandb

# --- 骨干微调 ---
# 原版默认是全量微调（freeze_encoder=False）。
# 0805（全量微调）vs 0808（冻结骨干）对比实验表明：冻结 DINOv2 骨干后
# ship 类 Recall 从 0.80 暴跌到 0.42，全量微调明显更优。
FREEZE_ENCODER = False

# --- EMA (指数移动平均) ---
USE_EMA = True            # EMA 提升收敛稳定性与最终精度，原版默认开启

# --- 验证 ---
EVAL_INTERVAL = 5         # 每隔 N 个 epoch 验证一次（减少 CPU 阻塞，加快训练）

# --- 恢复训练 ---
# RESUME = "output/0726-DIOR-rfdetr_medium/last.ckpt"
RESUME = ""

# ============================================================================
# 模型注册表（无需修改）
# ============================================================================

_MODEL_REGISTRY: dict[str, tuple[type, int]] = {
    "nano": (RFDETRNano, 384),
    "small": (RFDETRSmall, 512),
    "medium": (RFDETRMedium, 640),
    # 原版 Large（DINOv2-Base 骨干，135M 参数，medium 的 4 倍）。
    # 注意：新版 RFDETRLarge(2026) 的骨干与 medium 相同（ViT-S），只是分辨率 704；
    # 真正更大的骨干是这里的 RFDETRLargeDeprecated（已标记废弃但功能完整）。
    # 原生分辨率 560，位置编码 PE=37 与预训练权重绑定，不能随意改分辨率。
    "large": (RFDETRLargeDeprecated, 560),
}


def main() -> None:
    """加载配置并启动训练。"""
    # --- 校验模型和数据集 ---
    if MODEL not in _MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {MODEL}，可选: {list(_MODEL_REGISTRY)}")

    model_cls, default_resolution = _MODEL_REGISTRY[MODEL]
    dataset_dir = str(Path(DATASET_DIR).resolve())

    print(f"模型: {MODEL} | 类别数: {NUM_CLASSES} | 分辨率: {default_resolution}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"数据集: {dataset_dir} | 输出: {OUTPUT_DIR}")
    print(f"增广预设: {'无' if AUG_CONFIG is None else AUG_CONFIG}")

    # --- 构建模型 ---
    # gradient_checkpointing=True: 用计算换显存，backbone 中间激活值不缓存，
    # 反向传播时重新计算。可将激活值显存从 ~15 GB 降到 ~3 GB。
    model = model_cls(
        num_classes=NUM_CLASSES,
        resolution=default_resolution,
        gradient_checkpointing=True,
        freeze_encoder=FREEZE_ENCODER,
    )

    # --- 训练 ---
    model.train(
        dataset_dir=dataset_dir,
        dataset_file=DATASET_FILE,
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
        eval_interval=EVAL_INTERVAL,
        use_ema=USE_EMA,           # 关闭 EMA 以节省显存
        compute_val_loss=False,    # 关掉验证 loss 计算，省显存，mAP 指标不受影响
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,
        rare_class_oversample=RARE_CLASS_OVERSAMPLE,
        rare_class_oversample_factor=RARE_CLASS_OVERSAMPLE_FACTOR,
        rare_class_oversample_class_ids=RARE_CLASS_OVERSAMPLE_CLASS_IDS,
    )

    print(f"\n训练完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
