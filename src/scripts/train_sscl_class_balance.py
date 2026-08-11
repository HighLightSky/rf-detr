# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL + 分类损失均衡化训练脚本（0807 同配方消融）。

在 0807-SHWX-SSCL-Proj-原型+实例正样本 的完整配方（RFDETRMedium、640、6 epoch、
SSCL + 原型库 + 投影头 + 实例正样本、从 0805 基线微调）基础上，加入：

- **P0 正样本类均衡 IA-BCE**（CB Loss 风格）：只对正样本 slot 乘类别权重
  ``w_c = clamp((N_ref / max(n_c, n_min)) ** beta, 1.0, w_max)``，
  不降低负样本惩罚，优先保护 FDR。
- **P1 居中截断 Logit Adjustment**（默认关闭）：训练侧对 logit 加
  居中 + 截断 + warmup 的先验 bias。

类别统计在训练前从训练集 labels 自动生成并写入实验目录
（``<output_dir>/class_counts.json``），避免手写类别数出错。

用法：
    python src/scripts/train_sscl_class_balance.py

实验矩阵（改上方常数切换）：
- E1（默认）：P0 beta=0.25, w_max=3, targets=[0,1]（HM/LQS），LA 关
- E2：P0 beta=0.5, w_max=5, targets=[0,1]
- E3：E1 + LA tau=0.1, clip=1.0, warmup=1 epoch

详见 docs/改进方案-SSCL/RF-DETR分类损失均衡化改进方案.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.variants import RFDETRMedium

# 保证可导入同目录的 stat_class_counts（脚本以独立进程运行）
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from scripts.stat_class_counts import write_counts_json  # noqa: E402

# ============================================================================
# 训练参数 —— 在这里修改配置
# ============================================================================

# --- 模型 ---
MODEL = "medium"
NUM_CLASSES = 25
# 基线 checkpoint（与 0807 配方一致：0805 全量训练产物）
_BASE_CHECKPOINT = Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth")

# --- 数据集 ---
DATASET_DIR = "/home/liu/wzt/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# --- 训练超参数（与 0807 配方一致）---
EPOCHS = 6
BATCH_SIZE = 32  # 每 GPU batch size
GRAD_ACCUM_STEPS = 2  # 有效 batch = 64
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
OUTPUT_DIR = "output/0811-SHWX-SSCL-仅验证QNorm-Obj物体性门控"
TENSORBOARD = True
WANDB = False

# --- EMA / 验证 / 恢复 ---
USE_EMA = True
EVAL_INTERVAL = 1
RESUME = ""

# ============================================================================
# SSCL 配置（与 0807 配方一致：原型 + 投影头 + 实例正样本）
# ============================================================================
SSCL_ENABLED = True
SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_shwx_disc.pt"  # SHWX 25 类矩阵
SSCL_MATRIX_NORMALIZE = "minmax"
SSCL_LAMBDA = 0.03  # SSCL 损失权重 λ
SSCL_TAU = 0.1  # 对比学习温度 τ
SSCL_RHO = 0.3  # 语义先验放大系数 ρ
SSCL_OMEGA_MAX = 2.0  # 负样本语义权重上限
SSCL_ANCHOR_CLASSES = [0, 1, 2, 3]  # 参与计算 sscl 损失的类别
SSCL_CONFUSING_CLASSES = [0, 1, 2, 3]  # 参与充当比较类别的类别
SSCL_START_EPOCH = 0
SSCL_FREEZE_STRATEGY = "conservative"

# --- 类别原型库 ---
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROTOTYPE_MOMENTUM = 0.99
SSCL_PROTOTYPE_MIN_SAMPLES = 1

# --- 投影头 ---
SSCL_PROJECTION_ENABLED = True
SSCL_PROJECTION_DIM = 128
SSCL_PROTOTYPE_INSTANCE_POS = True

# --- 基类蒸馏（本组实验关闭）---
SSCL_DISTILL_ENABLED = False
SSCL_DISTILL_LAMBDA = 0.5
SSCL_DISTILL_TEMPERATURE = 2.0
SSCL_DISTILL_MODE = "mse"
SSCL_TEACHER_CHECKPOINT = str(_BASE_CHECKPOINT)
SSCL_PROTECTED_CLASSES = None

# --- 难例负样本 ---
SSCL_HARD_NEG_ENABLED = False
SSCL_HARD_NEG_SCORE_THRESH = -2.0
SSCL_HARD_NEG_LOG_INTERVAL = 100  # 训练监控采样步间隔（epoch 末输出 train/sscl/*）

# ============================================================================
# 分类损失均衡化配置
# ============================================================================
# --- P0 正样本类均衡 IA-BCE ---
CLASS_BALANCE_ENABLED = False
CLASS_BALANCE_BETA = 0.25  # E1：0.25；E2：0.5
CLASS_BALANCE_MAX_WEIGHT = 3.0  # E1：3.0；E2：5.0
CLASS_BALANCE_MIN_COUNT = 10  # 分母下限，防极端小样本类权重过大
CLASS_BALANCE_REF_COUNT = None  # None 自动取 sqrt(N_max * N_min)；也可显式指定
CLASS_BALANCE_TARGET_CLASSES = [0, 1]  # 首发只作用于 HM/LQS；第二轮可扩 [0,1,2]
# --- P1 居中截断 Logit Adjustment（E3 时开启）---
LOGIT_ADJUSTMENT_ENABLED = False
LOGIT_ADJUSTMENT_TAU = 0.1
LOGIT_ADJUSTMENT_BIAS_CLIP = 1.0
LOGIT_ADJUSTMENT_WARMUP_EPOCHS = 1.0


def main() -> None:
    """构建模型并启动训练（训练前自动统计类别数并落盘）。"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)

    # --- 训练前自动统计类别实例数（YOLO 布局：<dataset_dir>/labels/train）---
    labels_dir = Path(dataset_dir) / "labels" / "train"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"未找到训练集标签目录（YOLO 布局）: {labels_dir}")
    counts_json = str(Path(output_dir) / "class_counts.json")
    payload = write_counts_json(labels_dir, counts_json, NUM_CLASSES)
    print(f"类别实例数（自动统计）: {payload['counts']}")
    print(f"n_max={payload['n_max']:.0f}  n_min={payload['n_min']:.0f}  n_ref={payload['n_ref']:.2f}  -> {counts_json}")

    # --- 打印本次实验配置 ---
    print(f"模型: {MODEL} (SSCL + 分类损失均衡化) | 类别数: {NUM_CLASSES}")
    print(f"基线 checkpoint: {_BASE_CHECKPOINT}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, anchor={SSCL_ANCHOR_CLASSES}")
    print(
        f"P0 类均衡: {'启用' if CLASS_BALANCE_ENABLED else '关闭'} "
        f"β={CLASS_BALANCE_BETA}, w_max={CLASS_BALANCE_MAX_WEIGHT}, "
        f"targets={CLASS_BALANCE_TARGET_CLASSES}, counts={counts_json}"
    )
    print(
        f"P1 LA: {'启用' if LOGIT_ADJUSTMENT_ENABLED else '关闭'} "
        f"τ={LOGIT_ADJUSTMENT_TAU}, clip={LOGIT_ADJUSTMENT_BIAS_CLIP}, "
        f"warmup={LOGIT_ADJUSTMENT_WARMUP_EPOCHS} epoch"
    )

    # --- 构建模型，加载基线 checkpoint 作为微调起点 ---
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=True,
        pretrain_weights=str(_BASE_CHECKPOINT),  # 从基线 checkpoint 微调
    )

    # --- 训练（SSCL + 分类损失均衡化参数直接作为 kwargs 传给 train()） ---
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
        sscl_semantic_matrix_path=str(project_root / SSCL_SEMANTIC_MATRIX_PATH),
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
        # --- 分类损失均衡化配置 ---
        class_balance_enabled=CLASS_BALANCE_ENABLED,
        class_balance_counts_path=counts_json,
        class_balance_beta=CLASS_BALANCE_BETA,
        class_balance_max_weight=CLASS_BALANCE_MAX_WEIGHT,
        class_balance_min_count=CLASS_BALANCE_MIN_COUNT,
        class_balance_ref_count=CLASS_BALANCE_REF_COUNT,
        class_balance_target_classes=CLASS_BALANCE_TARGET_CLASSES,
        logit_adjustment_enabled=LOGIT_ADJUSTMENT_ENABLED,
        logit_adjustment_tau=LOGIT_ADJUSTMENT_TAU,
        logit_adjustment_bias_clip=LOGIT_ADJUSTMENT_BIAS_CLIP,
        logit_adjustment_warmup_epochs=LOGIT_ADJUSTMENT_WARMUP_EPOCHS,
        sscl_hard_neg_enabled=SSCL_HARD_NEG_ENABLED,
        sscl_hard_neg_topk=3,
        sscl_hard_neg_score_thresh=SSCL_HARD_NEG_SCORE_THRESH,
        sscl_hard_neg_log_interval=SSCL_HARD_NEG_LOG_INTERVAL,
        qnorm_obj_enabled=True
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
