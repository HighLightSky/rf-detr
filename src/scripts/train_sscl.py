# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL 语义相似度引导微调训练脚本 SSCL+视觉原型

在原始 RF-DETR checkpoint 基础上，冻结 backbone/encoder/bbox 头，
仅解冻 decoder 最后一层与分类头，加入语义相似度引导的对比学习损失，
缓解 SHWX 舰船细粒度类别（HM/LQS/QHS/MS）的混淆问题。

用法：
    python src/scripts/train_sscl.py

前置条件：
    - 已运行 build_semantic_matrix.py 生成 data/semantic_matrix_shwx.pt
    - 已有基线 checkpoint（默认 output/0724-shwx-rfdetr_medium/checkpoint_best_total.pth）

"""

from __future__ import annotations

from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
MODEL = "medium"
NUM_CLASSES = 25
# 是否启用 SGM 混合编码器分支。默认关闭以保持现有 SSCL 行为不变；
# 注意：SSCL 的冻结策略可能连带冻结 backbone.0.sga.*（随机初始化），
# 建议先用 src/scripts/train.py 跑通 SGM 分支后再在此开启。
USE_SGA = False
# 基线 checkpoint（作为微调起点，同时是蒸馏的 teacher 权重来源）
_BASE_CHECKPOINT = Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth")

# --- 数据集 ---
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
# DATASET_DIR = "/home/liu/datasets/DIOR-rfdetr"
# DIOR 是 Roboflow 布局（train/valid/test 各含 _annotations.coco.json），
# 必须用 "roboflow"；"coco" 走标准 COCO 布局（annotations/instances_*2017.json）。
DATASET_FILE = "yolo"

# --- 训练超参数（阶段 1：SSCL Only，保守微调）---
EPOCHS = 6
BATCH_SIZE = 12  # 每 GPU batch size
GRAD_ACCUM_STEPS = 4  # 有效 batch = 8 × 4 = 32，保证 batch 内有足够同类正样本
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
OUTPUT_DIR = "output/0805-SHWX-data-expand-SSCL"
TENSORBOARD = True
WANDB = False

# --- EMA ---
USE_EMA = True

# --- 验证 ---
EVAL_INTERVAL = 1

# --- 恢复训练 ---
RESUME = ""

# ============================================================================
# SSCL 配置
# ============================================================================
SSCL_ENABLED = True
SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_shwx.pt"
# SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_dior.pt"
SSCL_MATRIX_NORMALIZE = "minmax"  # 语义矩阵后处理: "minmax"（推荐）/"softmax"/"none"
SSCL_LAMBDA = 0.03  # SSCL 损失权重 λ（0.01 ~ 0.05）
SSCL_TAU = 0.1  # 对比学习温度 τ
SSCL_RHO = 0.3  # 语义先验放大系数 ρ（0.2 ~ 0.5）
SSCL_OMEGA_MAX = 2.0  # 负样本语义权重上限
SSCL_ANCHOR_CLASSES = [0, 1, 2, 3]  # 参与计算sscl损失的类别
SSCL_CONFUSING_CLASSES = [0, 1, 2, 3]  # 参与充当比较类别的类别
# SSCL_ANCHOR_CLASSES = None
# SSCL_CONFUSING_CLASSES = None
SSCL_START_EPOCH = 0  # 微调场景从第 0 个 epoch 即启用 SSCL（config 默认 30，此处显式覆盖）
SSCL_FREEZE_STRATEGY = "conservative"  # 在已收敛 checkpoint 上微调，采用保守冻结策略

# --- 类别原型库（原型锚定 SSCL，规避 batch 内同类样本不足导致的零损失）---
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROTOTYPE_MOMENTUM = 0.99  # 原型 EMA 更新系数（0.9 ~ 0.999）
SSCL_PROTOTYPE_MIN_SAMPLES = 1  # 单批同类样本低于该阈值则跳过该类原型更新

# --- 基类蒸馏 ---
SSCL_DISTILL_ENABLED = False
SSCL_DISTILL_LAMBDA = 0.5
SSCL_DISTILL_TEMPERATURE = 2.0
SSCL_DISTILL_MODE = "mse"
SSCL_TEACHER_CHECKPOINT = str(_BASE_CHECKPOINT)
SSCL_PROTECTED_CLASSES = None  # None 表示默认 = 飞机类(4-23) + FSC(24)


def main() -> None:
    """构建 SSCL 微调模型并启动训练。"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)
    semantic_matrix_path = str(project_root / SSCL_SEMANTIC_MATRIX_PATH)

    print(f"模型: {MODEL} (SSCL 微调) | 类别数: {NUM_CLASSES}")
    print(f"基线 checkpoint: {_BASE_CHECKPOINT}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, anchor={SSCL_ANCHOR_CLASSES}")
    print(f"SSCL 语义矩阵: {semantic_matrix_path}")
    print(f"蒸馏: {'启用' if SSCL_DISTILL_ENABLED else '关闭（阶段 2 启用）'}")

    # --- 构建模型，加载基线 checkpoint 作为微调起点 ---
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=True,
        use_sga=USE_SGA,
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
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
