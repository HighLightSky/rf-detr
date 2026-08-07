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
from rfdetr.variants import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
MODEL = "medium"  # 可选: nano, small, medium, large

# --- SGM 混合编码器分支（SPM+SGM+融合）---
# 启用后新增 stride-8/16 纹理分支，用于小目标检测（Phase 1：单级 fused P4）。
# 关闭 = 完全等同原模型行为。
USE_SGA = True

# --- 跨尺度交互分支（Phase 2：CFE，尺度内 RepNCSPELAN4 + 尺度间 RGM）---
# 启用需同时：USE_SGA=True、USE_CFE=True、PROJECTOR_SCALE=["P3","P4"]（多级，
# decoder 自动 n_levels=2 + warm-start，CFE 模块随机初始化、其余从 COCO 预训练权重开始）。
# 默认单级 P4（当前 Phase 1 训练），CFE 关闭不影响现有训练。
USE_CFE = True
# PROJECTOR_SCALE = ["P4"]  # CFE 开启时改为 ["P3", "P4"]
PROJECTOR_SCALE = ["P3", "P4"]

# --- SGM 门控/融合变体（P0 修复实验，详见 output/0805-SHWX-SGA-rfdetr/实验报告.md §五）---
# 门控模式: product(原版) | lower_bound(下界保底 [0.5,1]) | residual(残差 1+M) | ones(SPM-only 消融)
SGA_GATE_MODE = "product"
# 融合残差保底: fused = feats[i] + gamma*delta，保留原始 projector 语义特征
SGA_FUSION_RESIDUAL = False
SGA_RESIDUAL_GAMMA = 0.1  # 残差融合系数（起步 0.1 更稳）
# SGM 注意力 logits 初值偏置（>0 使初始注意力≈全通，防早期反向收敛；attn_bias 实验用 +2.0）
SGA_ATTN_BIAS = 0.0

# --- 数据集 ---
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"  # roboflow：Roboflow COCO 格式 (train/_annotations.coco.json)，还有coco yolo

# --- 训练超参数 ---
NUM_CLASSES = 25  # 类别数
EPOCHS = 100  # 训练轮数
BATCH_SIZE = 16  # 每 GPU 的 batch size
NUM_WORKERS = 12  # DataLoader 工作进程数
LR = 1e-4  # 基础学习率
LR_ENCODER = 1.5e-4  # 编码器（backbone）学习率
WEIGHT_DECAY = 1e-4  # 权重衰减
GRAD_ACCUM_STEPS = 4  # 梯度累积步数（有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS）
CLIP_MAX_NORM = 0.1  # 梯度裁剪

# --- 学习率调度 ---
LR_DROP = 70  # 学习率下降的 epoch 数
WARMUP_EPOCHS = 2.0  # 预热 epoch 数

# --- 数据增广 ---
# 可选预设:
#   None          → 使用默认 torchvision 增广（HorizontalFlip + 多尺度缩放裁剪）
#   AUG_CONSERVATIVE → 保守增广（小数据集推荐，<500 张）
#   AUG_AGGRESSIVE   → 激进增广（大数据集推荐，2000+ 张）
#   AUG_AERIAL       → 航拍/遥感影像（水平/垂直翻转 + 90° 旋转）
#   AUG_INDUSTRIAL   → 工业/检测影像（光照噪声 + 模糊）
#   {}             → 关闭额外增广
AUG_CONFIG = AUG_AERIAL  # 遥感、航拍预设

# --- Mosaic 增强 ---
# Mosaic 将 4 张图片拼接为 1 张训练样本，有效提升小目标检测和遥感数据集的性能。
# 推荐值: 0.5 (前 80% 训练阶段开启，最后 20% 关闭)
# 设为 0.0 关闭
MOSAIC_P = 0.8

# --- 硬件 ---
DEVICE = "cuda"  # 设备: cuda, cuda:0, cpu
DEVICES = 1  # GPU 数量
NUM_NODES = 1  # 节点数

# --- 输出 & 日志 ---
OUTPUT_DIR = "output/0806-SHWX-SGA-CFE-rfdetr"  # 输出目录
TENSORBOARD = True  # 是否启用 TensorBoard
WANDB = False  # 是否启用 Wandb

# --- EMA (指数移动平均) ---
USE_EMA = False  # 关闭 EMA，节省约 1 倍模型权重的显存

# --- 验证 ---
EVAL_INTERVAL = 5  # 每隔 N 个 epoch 验证一次（减少 CPU 阻塞，加快训练）

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
    "large": (RFDETRLarge, 768),
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
        use_sga=USE_SGA,
        use_cfe=USE_CFE,
        projector_scale=PROJECTOR_SCALE,
        sga_gate_mode=SGA_GATE_MODE,
        sga_fusion_residual=SGA_FUSION_RESIDUAL,
        sga_residual_gamma=SGA_RESIDUAL_GAMMA,
        sga_attn_bias=SGA_ATTN_BIAS,
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
        use_ema=USE_EMA,  # 关闭 EMA 以节省显存
        compute_val_loss=False,  # 关掉验证 loss 计算，省显存，mAP 指标不受影响
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,
    )

    print(f"\n训练完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
