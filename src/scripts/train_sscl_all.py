# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 全类别实验训练脚本（从 DINOv2 预训练直接开始）。

与 train_sscl.py（在已收敛的 SHWX checkpoint 上做舰船专属微调）不同，
本脚本从 RF-DETR Medium 发布权重（DINOv2 windowed_small 主干 + COCO 解码器）
开始，对 SHWX 做全量微调，并在第 30 轮之后才对**全部 25 个类别**施加
SSCL（语义相似度引导的监督对比学习）损失。

实验目的：验证「全类别 SSCL + 从某 epoch 开启」能否相比不加 SSCL 的全量
微调基线带来改善（尤其 MS/QHS 与易混飞机类）。

对照实验：
    同一脚本将 SSCL_ENABLED 改为 False 即得到无 SSCL 的对照组；
    两组起点、超参、增广完全一致，唯一差异是 SSCL 的开关与起始 epoch。

用法：
    python src/scripts/train_sscl_all.py

前置条件：
    - 已运行 build_semantic_matrix.py 生成 data/semantic_matrix_shwx.pt
"""

from __future__ import annotations

from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
# 不指定 pretrain_weights：使用 RFDETRMedium 默认发布权重
# （DINOv2 windowed_small backbone + COCO decoder），即「DINOv2 预训练起点」。
MODEL = "medium"
NUM_CLASSES = 25

# --- 数据集 ---
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# --- 训练超参数（全量微调，参考 train.py 从预训练起步的标准配置）---
EPOCHS = 100
SSCL_START_EPOCH = 30  # 前 30 轮仅常规损失训练，30 轮之后开始施加 SSCL
BATCH_SIZE = 8  # 每 GPU batch size
GRAD_ACCUM_STEPS = 4  # 有效 batch = 8 × 4 = 32，保证 batch 内有足够同类正样本
NUM_WORKERS = 12
LR = 1e-4  # 基础学习率（非 decoder 部分）
LR_ENCODER = 1.5e-4  # 编码器（backbone）学习率
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1
LR_DROP = 40
WARMUP_EPOCHS = 3.0

# --- 数据增广 ---
AUG_CONFIG = AUG_AERIAL  # 遥感航拍预设
MOSAIC_P = 0.8  # 从预训练全量训练，保留 Mosaic

# --- 硬件 ---
DEVICE = "cuda"
DEVICES = 1
NUM_NODES = 1

# --- 输出 & 日志 ---
OUTPUT_DIR = "output/0802-SHWX-rfdetr_medium_SSCL-all"
TENSORBOARD = True
WANDB = False

# --- EMA ---
USE_EMA = True

# --- 验证 ---
EVAL_INTERVAL = 5

# --- 恢复训练 ---
RESUME = ""

# ============================================================================
# SSCL 配置（全类别实验）
# ============================================================================
SSCL_ENABLED = True
SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_shwx.pt"
SSCL_MATRIX_NORMALIZE = "minmax"  # 语义矩阵后处理: "minmax"（推荐）/"softmax"/"none"
SSCL_LAMBDA = 0.01  # SSCL 损失权重 λ（0.01 ~ 0.05）
SSCL_TAU = 0.1  # 对比学习温度 τ
SSCL_RHO = 0.3  # 语义先验放大系数 ρ（0.2 ~ 0.5）
SSCL_OMEGA_MAX = 2.0  # 负样本语义权重上限
SSCL_ANCHOR_CLASSES = None  # None = 全部 25 个类别都作为 anchor
SSCL_CONFUSING_CLASSES = None  # None = 所有异类负样本都施加语义权重
SSCL_FREEZE_STRATEGY = "none"  # 从预训练直接训练：不冻结，全量微调

# --- 基类蒸馏：本实验关闭（从 COCO 起步没有合理的 SHWX teacher）---
SSCL_DISTILL_ENABLED = False


def main() -> None:
    """构建 SSCL 全类别实验模型并启动训练。"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)
    semantic_matrix_path = str(project_root / SSCL_SEMANTIC_MATRIX_PATH)

    print(f"模型: {MODEL} (SSCL 全类别实验) | 类别数: {NUM_CLASSES}")
    print("起点: RF-DETR Medium 发布权重（DINOv2 backbone + COCO decoder）")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, anchor=全部类别")
    print(f"SSCL 起始 epoch: {SSCL_START_EPOCH}（之前 loss_sscl 权重为 0） | 冻结策略: {SSCL_FREEZE_STRATEGY}")
    print(f"SSCL 语义矩阵: {semantic_matrix_path}")

    # --- 构建模型，使用默认发布权重作为预训练起点 ---
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=True,
    )

    # --- 训练（SSCL 相关参数直接作为 kwargs 传给 train()） ---
    model.train(
        dataset_dir=dataset_dir,
        dataset_file=DATASET_FILE,
        output_dir=output_dir,
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
        use_ema=USE_EMA,
        compute_val_loss=False,  # 关闭验证 loss 计算，省显存
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,
        multi_scale=True,  # 从预训练全量训练，保留多尺度
        # --- SSCL 配置 ---
        sscl_enabled=SSCL_ENABLED,
        sscl_semantic_matrix_path=semantic_matrix_path,
        sscl_matrix_normalize=SSCL_MATRIX_NORMALIZE,
        sscl_lambda=SSCL_LAMBDA,
        sscl_tau=SSCL_TAU,
        sscl_rho=SSCL_RHO,
        sscl_omega_max=SSCL_OMEGA_MAX,
        sscl_anchor_classes=SSCL_ANCHOR_CLASSES,
        sscl_confusing_classes=SSCL_CONFUSING_CLASSES,
        sscl_start_epoch=SSCL_START_EPOCH,
        sscl_freeze_strategy=SSCL_FREEZE_STRATEGY,
        sscl_distill_enabled=SSCL_DISTILL_ENABLED,
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
