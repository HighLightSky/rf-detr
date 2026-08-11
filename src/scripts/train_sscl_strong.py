# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 原型方案参数化微调脚本（0807 实验的增强版）。

与 train_sscl.py（0807 原型+实例正样本配方）完全相同的训练范式：
- 起点：0805 data-expand baseline（已在 SHWX 上收敛）
- 6 epoch 微调、lr 1e-5、conservative 冻结策略、AUG_AERIAL、无 Mosaic
- SSCL 原型模式 + 投影头 + 实例正样本，anchor/confusing 均为舰船四类

唯一差异：SSCL 关键超参与语义矩阵通过环境变量可调，用于两个方向实验：
1. **强作用力**：提高 λ（0.02→0.2）、降低 τ（0.1→0.05）、提高 ρ 与 ω_max
   ——之前 SSCL 权重仅 0.02，相对检测 loss 几乎可忽略；
2. **判别性语义矩阵**：替换为基于遥感判别特征的 CLIP 提示词矩阵
   （prompts/s hwx_disc.yaml 构建）。

用法（三个实验）：
    # A: 强作用力（原矩阵）
    SSCL_LAMBDA=0.2 SSCL_TAU=0.05 SSCL_RHO=0.6 SSCL_OMEGA_MAX=3.0 \
        OUTPUT_DIR=output/0811-SHWX-SSCL-Strong python src/scripts/train_sscl_strong.py

    # B: 判别性 prompt 矩阵（原超参）
    SSCL_MATRIX=data/semantic_matrix_shwx_disc.pt \
        OUTPUT_DIR=output/0811-SHWX-SSCL-DiscPrompt python src/scripts/train_sscl_strong.py

    # C: A+B 组合
    SSCL_LAMBDA=0.2 SSCL_TAU=0.05 SSCL_RHO=0.6 SSCL_OMEGA_MAX=3.0 \
        SSCL_MATRIX=data/semantic_matrix_shwx_disc.pt \
        OUTPUT_DIR=output/0811-SHWX-SSCL-StrongDisc python src/scripts/train_sscl_strong.py
"""

from __future__ import annotations

import os
from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 训练参数（环境变量可覆盖，见用法）
# ============================================================================

MODEL = "medium"
NUM_CLASSES = 25
# 起点：0805 data-expand baseline（与 0807 实验一致）
_BASE_CHECKPOINT = Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth")
DATASET_DIR = "/home/liu/wzt/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"
EPOCHS = 6
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 12
LR = 1e-5
LR_ENCODER = 1e-5
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1
WARMUP_EPOCHS = 0.0
AUG_CONFIG = AUG_AERIAL
MOSAIC_P = 0.0
DEVICE = "cuda"
DEVICES = 1
NUM_NODES = 1
TENSORBOARD = True
WANDB = False
USE_EMA = True
EVAL_INTERVAL = 1
RESUME = ""

# --- SSCL 超参（环境变量可覆盖）---
SSCL_ENABLED = True
# 语义矩阵：B/C 实验指向判别性矩阵
SSCL_SEMANTIC_MATRIX_PATH = os.environ.get("SSCL_MATRIX", "data/semantic_matrix_shwx.pt")
SSCL_MATRIX_NORMALIZE = "minmax"
# 强作用力实验：λ 0.02→0.2（10 倍），τ 0.1→0.05（对比更尖锐），
# ρ 0.3→0.6 + ω_max 2→3（语义权重放大更显著）
SSCL_LAMBDA = float(os.environ.get("SSCL_LAMBDA", "0.02"))
SSCL_TAU = float(os.environ.get("SSCL_TAU", "0.1"))
SSCL_RHO = float(os.environ.get("SSCL_RHO", "0.3"))
SSCL_OMEGA_MAX = float(os.environ.get("SSCL_OMEGA_MAX", "2.0"))
SSCL_ANCHOR_CLASSES = [0, 1, 2, 3]
SSCL_CONFUSING_CLASSES = [0, 1, 2, 3]
SSCL_START_EPOCH = 0
SSCL_FREEZE_STRATEGY = "conservative"
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROTOTYPE_MOMENTUM = 0.99
SSCL_PROTOTYPE_MIN_SAMPLES = 1
SSCL_PROJECTION_ENABLED = True
SSCL_PROJECTION_DIM = 128
SSCL_PROTOTYPE_INSTANCE_POS = True  # 原型 + 实例正样本（0807 配方）
SSCL_DISTILL_ENABLED = False
SSCL_PROTECTED_CLASSES = None

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/0811-SHWX-SSCL-Strong")


def main() -> None:
    """构建 SSCL 微调模型并启动训练。"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)
    semantic_matrix_path = str(project_root / SSCL_SEMANTIC_MATRIX_PATH)

    print(f"模型: {MODEL} (SSCL 微调) | 类别数: {NUM_CLASSES}")
    print(f"基线 checkpoint: {_BASE_CHECKPOINT}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, ω_max={SSCL_OMEGA_MAX}")
    print(f"SSCL 语义矩阵: {semantic_matrix_path}")
    print(f"投影头: dim={SSCL_PROJECTION_DIM} 实例正样本={SSCL_PROTOTYPE_INSTANCE_POS}")

    # --- 构建模型，加载基线 checkpoint 作为微调起点 ---
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=True,
        pretrain_weights=str(_BASE_CHECKPOINT),
    )

    # --- 训练 ---
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
        lr_drop=max(1, EPOCHS),  # 短训练不打步长，保持恒定 LR
        warmup_epochs=WARMUP_EPOCHS,
        tensorboard=TENSORBOARD,
        wandb=WANDB,
        device=DEVICE,
        devices=DEVICES,
        num_nodes=NUM_NODES,
        resume=RESUME,
        eval_interval=EVAL_INTERVAL,
        use_ema=USE_EMA,
        compute_val_loss=False,
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,
        multi_scale=False,
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
        sscl_teacher_checkpoint=None,
        sscl_protected_classes=SSCL_PROTECTED_CLASSES,
        sscl_prototype_enabled=SSCL_PROTOTYPE_ENABLED,
        sscl_prototype_momentum=SSCL_PROTOTYPE_MOMENTUM,
        sscl_prototype_min_samples=SSCL_PROTOTYPE_MIN_SAMPLES,
        sscl_projection_enabled=SSCL_PROJECTION_ENABLED,
        sscl_projection_dim=SSCL_PROJECTION_DIM,
        sscl_prototype_instance_pos=SSCL_PROTOTYPE_INSTANCE_POS,
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
