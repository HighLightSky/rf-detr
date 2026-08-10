# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 难例负样本（hard negatives）微调训练脚本。

与 train_sscl.py（0807 基线配方）完全一致的超参，唯一差异是启用难例负
样本：从每张图选择 Hungarian matching 之外、与任一 GT 的 IoU 落在
[0.1, 0.5] 区间的 unmatched query，按最大前景 logit 取 top-k，作为额外
负样本列追加进原型模式损失的分母（权重 1.0、detach、不进原型库、不 EMA）。
对照实验矩阵：
- 基线：output/0807-SHWX-SSCL-Proj-原型+实例正样本（复用，不重跑）
- 本次：output/0810-SHWX-SSCL-Proj-HardNeg-k3（k=3）
- 本次：output/0810-SHWX-SSCL-Proj-HardNeg-k5（k=5）

用法：
    python src/scripts/train_sscl_hardneg.py            # k=3（默认）
    SSCL_HARD_NEG_TOPK=5 python src/scripts/train_sscl_hardneg.py   # k=5

前置条件：
    - 已运行 build_semantic_matrix.py 生成 data/semantic_matrix_shwx.pt
    - 已有基线 checkpoint（output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth）
"""

from __future__ import annotations

import os
from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
MODEL = "medium"
NUM_CLASSES = 25
# 基线 checkpoint（作为微调起点，与 0807 基线实验一致）
_BASE_CHECKPOINT = Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth")

# 缓存文件加速训练
DATASET_CACHE_MODE = "raw"  # 缓存解码后的原始 RGB 图片，保留 Mosaic/随机增广在线执行
DATASET_CACHE_DIR = None  # None 表示使用 output_dir/dataset_cache
DATASET_CACHE_REBUILD = False

# --- 数据集 ---
DATASET_DIR = "/home/liu/wzt/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# --- 训练超参数（与 0807 基线配方一致：6 epoch 保守微调）---
EPOCHS = 6
BATCH_SIZE = 64  # 每 GPU batch size
GRAD_ACCUM_STEPS = 1  # 有效 batch = 64
NUM_WORKERS = 12
LR = 1e-5  # 低学习率，保护已有权重（解码器部分）
LR_ENCODER = 1e-5  # backbone 已冻结，此处仅为占位
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1
WARMUP_EPOCHS = 0.0

# --- 数据增广 ---
AUG_CONFIG = AUG_AERIAL  # 遥感航拍预设
MOSAIC_P = 0.0  # SSCL 微调阶段关闭 Mosaic，减少特征分布扰动

# --- 硬件 ---
DEVICE = "cuda"
DEVICES = 1
NUM_NODES = 1

# --- 输出 & 日志 ---
# 难例数量 k 由环境变量控制（默认 3，双臂实验用 3 与 5），输出目录随 k 区分
SSCL_HARD_NEG_TOPK = int(os.environ.get("SSCL_HARD_NEG_TOPK", "3"))
OUTPUT_DIR = f"output/0810-SHWX-SSCL-Proj-HardNeg-k{SSCL_HARD_NEG_TOPK}"
TENSORBOARD = True
WANDB = False

# --- EMA ---
USE_EMA = True

# --- 验证 ---
EVAL_INTERVAL = 1

# --- 恢复训练 ---
RESUME = ""

# ============================================================================
# SSCL 配置（与 0807 基线配方一致）
# ============================================================================
SSCL_ENABLED = True
SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_shwx.pt"  # SHWX 25 类矩阵
SSCL_MATRIX_NORMALIZE = "minmax"
SSCL_LAMBDA = 0.02  # SSCL 损失权重 λ
SSCL_TAU = 0.1  # 对比学习温度 τ
SSCL_RHO = 0.3  # 语义先验放大系数 ρ
SSCL_OMEGA_MAX = 2.0  # 负样本语义权重上限
SSCL_ANCHOR_CLASSES = [0, 1, 2, 3]  # 参与计算 sscl 损失的类别
SSCL_CONFUSING_CLASSES = [0, 1, 2, 3]  # 参与充当比较类别的类别
SSCL_START_EPOCH = 0  # 微调场景从第 0 个 epoch 即启用 SSCL
SSCL_FREEZE_STRATEGY = "conservative"  # 在已收敛 checkpoint 上微调，采用保守冻结策略

# --- 类别原型库（原型锚定 SSCL，规避 batch 内同类样本不足导致的零损失）---
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROTOTYPE_MOMENTUM = 0.99
SSCL_PROTOTYPE_MIN_SAMPLES = 1

# --- 投影头 + 实例正样本（对齐 0807 基线：原型+实例正样本）---
SSCL_PROJECTION_ENABLED = True
SSCL_PROJECTION_DIM = 128
SSCL_PROTOTYPE_INSTANCE_POS = True

# --- 难例负样本（本次实验变量）---
SSCL_HARD_NEG_ENABLED = True
# 最大前景 logit 下限（原始 logit）。默认 -2.0（≈ p>0.12）：0.0 在原始
# logit 空间过严（强模型带内候选几乎全被滤掉，每图难例数 < 0.02），
# 诊断脚本实测 -2.0 使每图难例数提升至 0.22（见方案文档 §4.1 诊断发现）。
SSCL_HARD_NEG_SCORE_THRESH = -2.0
SSCL_HARD_NEG_LOG_INTERVAL = 100  # 训练监控采样步间隔（epoch 末输出 train/sscl/*）

# --- 基类蒸馏（关闭，与 0807 基线一致）---
SSCL_DISTILL_ENABLED = False
SSCL_DISTILL_LAMBDA = 0.5
SSCL_DISTILL_TEMPERATURE = 2.0
SSCL_DISTILL_MODE = "mse"
SSCL_TEACHER_CHECKPOINT = str(_BASE_CHECKPOINT)
SSCL_PROTECTED_CLASSES = None


def main() -> None:
    """构建 SSCL 难例微调模型并启动训练。"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)
    semantic_matrix_path = str(project_root / SSCL_SEMANTIC_MATRIX_PATH)

    print(f"模型: {MODEL} (SSCL 难例微调) | 类别数: {NUM_CLASSES}")
    print(f"基线 checkpoint: {_BASE_CHECKPOINT}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, anchor={SSCL_ANCHOR_CLASSES}")
    print(f"SSCL 投影头: dim={SSCL_PROJECTION_DIM}, 实例正样本={SSCL_PROTOTYPE_INSTANCE_POS}")
    print(
        f"SSCL 难例负样本: 启用 (topk={SSCL_HARD_NEG_TOPK}, "
        f"score_thresh={SSCL_HARD_NEG_SCORE_THRESH}, log_interval={SSCL_HARD_NEG_LOG_INTERVAL})"
    )

    # --- 构建模型，加载基线 checkpoint 作为微调起点（与 0807 基线一致）---
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=True,
        pretrain_weights=str(_BASE_CHECKPOINT),  # 从基线 checkpoint 微调
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
        compute_val_loss=False,  # 关闭验证 loss 计算，省显存
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,
        multi_scale=False,  # 关闭多尺度，保证 batch 内特征稳定
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
        sscl_distill_lambda=SSCL_DISTILL_LAMBDA,
        sscl_distill_temperature=SSCL_DISTILL_TEMPERATURE,
        sscl_distill_mode=SSCL_DISTILL_MODE,
        sscl_teacher_checkpoint=SSCL_TEACHER_CHECKPOINT if SSCL_DISTILL_ENABLED else None,
        sscl_protected_classes=SSCL_PROTECTED_CLASSES,
        sscl_prototype_enabled=SSCL_PROTOTYPE_ENABLED,
        sscl_prototype_momentum=SSCL_PROTOTYPE_MOMENTUM,
        sscl_prototype_min_samples=SSCL_PROTOTYPE_MIN_SAMPLES,
        sscl_projection_enabled=SSCL_PROJECTION_ENABLED,
        sscl_projection_dim=SSCL_PROJECTION_DIM,
        sscl_prototype_instance_pos=SSCL_PROTOTYPE_INSTANCE_POS,
        # --- 难例负样本（本次实验变量）---
        sscl_hard_neg_enabled=SSCL_HARD_NEG_ENABLED,
        sscl_hard_neg_topk=SSCL_HARD_NEG_TOPK,
        sscl_hard_neg_score_thresh=SSCL_HARD_NEG_SCORE_THRESH,
        sscl_hard_neg_log_interval=SSCL_HARD_NEG_LOG_INTERVAL,
        dataset_cache_mode=DATASET_CACHE_MODE,
        dataset_cache_dir=DATASET_CACHE_DIR,
        dataset_cache_rebuild=DATASET_CACHE_REBUILD,
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
