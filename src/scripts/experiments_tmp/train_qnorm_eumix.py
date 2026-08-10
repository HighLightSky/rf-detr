# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR 实验二训练脚本 —— QNorm-Obj + EUMix（query 范数物体性 + 熵感知校准）。

在「重采样 + SSCL + 视觉原型」方案基础上叠加 QNorm-Obj + EUMix（EW-DETR 论文
§3.4-3.5 的闭集适配，见 docs/改进方案-QNorm-Obj/）：

- 重采样：HM/LQS 稀有类训练参与度提升（与 0810 重采样实验同参数）；
- SSCL + 视觉原型（原型锚定 + 投影头 + 实例正样本）：语义对比学习，**只在
  最后 10 轮启用**（已实验证明前期开启效果不好；原型 EMA 预热不门控为现有
  设计，锚点在生效时就绪）；
- QNorm-Obj + EUMix：解码器 query 范数物体性门控 + 熵感知校准，logit 层改造，
  不加任何辅助监督损失，与 SSCL 路线正交可叠加。

对照实验（唯一变量链）：
    基线（0805-SHWX-data-expand-rfdetr-baseline，无重采样无 SSCL）
  → 重采样（0810-SHWX-rfdetr-medium-rare-oversample，仅重采样）
  → 实验二（本脚本，重采样 + SSCL+原型 + QNorm+EUMix）

用法：
    python src/scripts/experiments_tmp/train_qnorm_eumix.py

前置条件：
    - 已运行 build_semantic_matrix.py 生成 data/semantic_matrix_shwx.pt
"""

from __future__ import annotations

import os
from pathlib import Path

# torch.compile 的 inductor C++ wrapper 需要 -std=c++20 支持（torch>=2.11），
# 系统默认 g++-9 过旧会导致编译静默失败（suppress_errors 吞掉错误，训练背着
# dynamo 开销跑 eager）。系统已装 g++-13，这里强制指定（须在 torch.compile
# 执行前设置，编译时读取）。
os.environ.setdefault("CXX", "/usr/bin/g++-13")

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
DATASET_DIR = "/home/liu/wzt/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# --- 训练超参数（与 0810 重采样实验完全一致，保证唯一对照变量）---
# 2026-08-09 实测（5090 + bf16 + torch.compile）：有效 batch 保持 64（B32×2），
# 优化器/EMA/日志步数减半、GPU 单步 burst 更长、占用更平稳；compile B32 峰值 ~22.5GB。
EPOCHS = 100
BATCH_SIZE = 16          # 每 GPU 的 batch size（实测上限：compile 下 32 安全，40 OOM）
GRAD_ACCUM_STEPS = 4     # 有效 batch = 32 × 2 = 64，与 0810 重采样实验一致
NUM_WORKERS = 20          # DataLoader 工作进程数（20 核 CPU，worker 单线程）
PREFETCH_FACTOR = 6       # 每 worker 预取样本数（默认 2；加深预取缓冲，平滑 decode/mosaic 抖动）
LR = 1e-4                # 基础学习率
LR_ENCODER = 1.5e-4      # 编码器（backbone）学习率
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1
LR_DROP = 60
WARMUP_EPOCHS = 0.0

# --- 数据增广 ---
AUG_CONFIG = AUG_AERIAL   # 遥感、航拍预设
MOSAIC_P = 0.8

# --- 性能优化（5090 利用率）---
# 数据增强后端: "cpu"(Albumentations)/"kornia"(GPU 增强)/"auto"(kornia 可用时自动 GPU)。
# AUG_AERIAL 的全部变换（翻转/旋转/亮度对比度）都有 Kornia 等价物，CPU→GPU 消除最大瓶颈。
AUGMENTATION_BACKEND = "kornia"
# torch.compile 编译整个模型（CUDA 下生效）。
# 2026-08-10 实测澄清：TrainConfig 默认 multi_scale=True + square_resize_div_64=True +
# do_random_resize_via_padding=False → skip_random_resize 只保留最大单一 scale，
# **训练实际固定 800×800（并非脚本旧注释以为的 640 / 多尺度）**。module_model.py 的
# compile 门控已改为按"实际生效 scale 数"判断：单一 scale 时启用 dynamic=False 编译
# （810 基线同样按默认配置跑在 800×800，对照关系不变）。
# 前置修复（均已生效）：
#   - transformer.py:332 torch._shape_as_tensor 不可追踪 → torch.as_tensor(list(shape[2:]))
#   - 实测提速 1.40×（B16@640: 119→85ms/step），显存反而更低（Inductor 融合）
# 注意：旧注释"要求 multi_scale=False"是错误假设；首次训练有数分钟编译开销。
COMPILE = True

# --- 少数类重采样（与 0810 重采样实验同参数）---
RARE_CLASS_OVERSAMPLE = True
RARE_CLASS_OVERSAMPLE_FACTOR = 2       # 倍率
RARE_CLASS_OVERSAMPLE_CLASS_IDS = [0, 1]   # 0=HM 航母, 1=LQS 两栖舰

# --- 硬件 ---
DEVICE = "cuda"
DEVICES = 1
NUM_NODES = 1

# --- 输出 & 日志 ---
OUTPUT_DIR = "output/0809-SHWX-rfdetr-medium-rare-oversample-SSCL-Proj-QNormEUMix"
TENSORBOARD = True
WANDB = False

# --- 骨干微调 ---
FREEZE_ENCODER = False

# --- EMA ---
USE_EMA = True

# --- 验证 ---
# 2026-08-09 实测：单 epoch 训练仅 ~35-40s，而一次验证（672 张双前向 + COCO 匹配）
# 需 30-60s 低占用 → eval_interval=5 时每 4 分钟就有 ~1 分钟 GPU 低占用，是利用率
# 波动的主因。改为 10（配合 eval_ema_only=True 再省一半验证前向）。
EVAL_INTERVAL = 10

# --- 恢复训练 ---
# 支持环境变量 RF_RESUME 覆盖（监控脚本续训用，如 RF_RESUME=<ckpt 路径> 启动）；
# 手动运行时留空即从头训练。
RESUME = os.environ.get("RF_RESUME", "")

# ============================================================================
# SSCL 配置（语义相似度引导对比学习 + 视觉原型）
# ============================================================================
SSCL_ENABLED = True
SSCL_SEMANTIC_MATRIX_PATH = "data/semantic_matrix_shwx.pt"
SSCL_MATRIX_NORMALIZE = "minmax"  # 语义矩阵后处理: "minmax"（推荐）/"softmax"/"none"
SSCL_LAMBDA = 0.02  # SSCL 损失权重 λ（0807 原型+实例正样本同款）
SSCL_TAU = 0.1  # 对比学习温度 τ
SSCL_RHO = 0.3  # 语义先验放大系数 ρ
SSCL_OMEGA_MAX = 2.0  # 负样本语义权重上限
SSCL_ANCHOR_CLASSES = [0, 1, 2, 3]  # 重点 anchor 类别（0807 同款：舰船 0-3）
SSCL_CONFUSING_CLASSES = None  # None = 所有异类负样本都施加语义权重
# SSCL + 视觉原型只在最后 10 轮启用（epoch 90-99）：已实验证明训练前期开启
# SSCL+视觉原型效果不好；前期仅常规检测损失训练，SSCL+原型作为收敛后期
# （低 LR 段）的精修约束。原型 EMA 预热不按 start_epoch 门控（现有设计），
# epoch 90 loss 生效时锚点已就绪。
SSCL_START_EPOCH = 90
SSCL_FREEZE_STRATEGY = "none"  # 从预训练直接训练：不冻结，全量微调

# --- 类别原型库（原型锚定 SSCL，规避 batch 内同类样本不足导致的零损失）---
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROTOTYPE_MOMENTUM = 0.99  # 原型 EMA 更新系数
SSCL_PROTOTYPE_MIN_SAMPLES = 1  # 单批同类样本低于该阈值则跳过该类原型更新

# --- 投影头（把特征投影到低维对比空间再施加对比损失，缓解对共享特征的冲击）---
SSCL_PROJECTION_ENABLED = True
SSCL_PROJECTION_DIM = 128
SSCL_PROTOTYPE_INSTANCE_POS = True  # 原型模式加入同类别实例正样本

# --- 基类蒸馏：本实验关闭（从 COCO 起步没有合理的 SHWX teacher）---
SSCL_DISTILL_ENABLED = False

# ============================================================================
# QNorm-Obj + EUMix 配置（实验二新增，EW-DETR 论文 §3.4-3.5 闭集适配）
# ============================================================================
QNORM_OBJ_ENABLED = True  # 总开关
QNORM_OBJ_TAU = 2.0  # 物体性头温度 τ
QNORM_OBJ_FEATURE_MIX = True  # 特征混合（h_cls = (1-α_mix)·h + α_mix·h_norm）
QNORM_OBJ_GATE = True  # 物体性门控（z_known *= σ(z_obj)，只乘前景列）
QNORM_OBJ_EUMIX = True  # 熵感知校准（背景列扮演 unknown）
QNORM_OBJ_OBJ_HIDDEN_DIM = 64  # 物体性头隐藏维度
QNORM_OBJ_GAMMA_INIT = 1.0  # 熵缺口指数 γ 初始有效值
QNORM_OBJ_ALPHA_INIT = 0.1  # EUMix 混合权重 α 初始有效值（恒等起步，可学习上升）
QNORM_OBJ_LAMBDA_INIT = 0.5  # 前景软抑制强度 λ 初始值


def main() -> None:
    """构建实验二模型并启动训练。"""
    # 本脚本位于 src/scripts/experiments_tmp/ 下，比 src/scripts/ 深一层
    project_root = Path(__file__).resolve().parents[3]
    dataset_dir = str(Path(DATASET_DIR).resolve())
    output_dir = str(project_root / OUTPUT_DIR)
    semantic_matrix_path = str(project_root / SSCL_SEMANTIC_MATRIX_PATH)

    print(f"模型: {MODEL}（实验二：QNorm-Obj + EUMix）| 类别数: {NUM_CLASSES}")
    print("起点: RF-DETR Medium 发布权重（DINOv2 backbone + COCO decoder）")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"重采样: {'开' if RARE_CLASS_OVERSAMPLE else '关'} (factor={RARE_CLASS_OVERSAMPLE_FACTOR}, ids={RARE_CLASS_OVERSAMPLE_CLASS_IDS})")
    print(f"SSCL: λ={SSCL_LAMBDA}, τ={SSCL_TAU}, ρ={SSCL_RHO}, anchor={SSCL_ANCHOR_CLASSES}")
    print(f"SSCL 起始 epoch: {SSCL_START_EPOCH}（最后 10 轮启用） | 冻结策略: {SSCL_FREEZE_STRATEGY}")
    print(
        f"SSCL 投影头: {'启用' if SSCL_PROJECTION_ENABLED else '关闭'} "
        f"(dim={SSCL_PROJECTION_DIM}, 实例正样本={SSCL_PROTOTYPE_INSTANCE_POS})"
    )
    print(
        f"QNorm-Obj: {'启用' if QNORM_OBJ_ENABLED else '关闭'} "
        f"(τ={QNORM_OBJ_TAU}, feature_mix={QNORM_OBJ_FEATURE_MIX}, "
        f"gate={QNORM_OBJ_GATE}, eumix={QNORM_OBJ_EUMIX})"
    )

    # --- 构建模型，使用默认发布权重作为预训练起点 ---
    # gradient_checkpointing=False: 5090 32GB + bf16 显存充足，关闭省去 ~20-30%
    # 重算开销（checkpointing 以计算换显存，显存充裕时纯浪费）。若 OOM 再开启。
    # compile=True: torch.compile 编译模型加速（CUDA + multi_scale=False 下生效）。
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=640,
        gradient_checkpointing=False,
        freeze_encoder=FREEZE_ENCODER,
        compile=COMPILE,
    )

    # --- 训练（SSCL / QNorm 相关参数直接作为 kwargs 传给 train()） ---
    model.train(
        dataset_dir=dataset_dir,
        dataset_file=DATASET_FILE,
        output_dir=output_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
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
        eval_ema_only=True,  # 验证只前向 EMA 模型（USE_EMA=True 时 best 指标本就用 EMA），省一半验证时间
        compute_val_loss=False,  # 关闭验证 loss 计算，省显存
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        augmentation_backend=AUGMENTATION_BACKEND,
        mosaic_p=MOSAIC_P,
        # 多尺度关闭（与 0810 重采样实验一致，保证唯一变量；同时解锁 torch.compile）
        rare_class_oversample=RARE_CLASS_OVERSAMPLE,
        rare_class_oversample_factor=RARE_CLASS_OVERSAMPLE_FACTOR,
        rare_class_oversample_class_ids=RARE_CLASS_OVERSAMPLE_CLASS_IDS,
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
        sscl_prototype_enabled=SSCL_PROTOTYPE_ENABLED,
        sscl_prototype_momentum=SSCL_PROTOTYPE_MOMENTUM,
        sscl_prototype_min_samples=SSCL_PROTOTYPE_MIN_SAMPLES,
        sscl_projection_enabled=SSCL_PROJECTION_ENABLED,
        sscl_projection_dim=SSCL_PROJECTION_DIM,
        sscl_prototype_instance_pos=SSCL_PROTOTYPE_INSTANCE_POS,
        # --- QNorm-Obj + EUMix 配置 ---
        qnorm_obj_enabled=QNORM_OBJ_ENABLED,
        qnorm_obj_tau=QNORM_OBJ_TAU,
        qnorm_obj_feature_mix=QNORM_OBJ_FEATURE_MIX,
        qnorm_obj_gate=QNORM_OBJ_GATE,
        qnorm_obj_eumix=QNORM_OBJ_EUMIX,
        qnorm_obj_obj_hidden_dim=QNORM_OBJ_OBJ_HIDDEN_DIM,
        qnorm_obj_gamma_init=QNORM_OBJ_GAMMA_INIT,
        qnorm_obj_alpha_init=QNORM_OBJ_ALPHA_INIT,
        qnorm_obj_lambda_init=QNORM_OBJ_LAMBDA_INIT,
    )

    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
