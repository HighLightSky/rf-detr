# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义头消融实验配置：BASE_RECIPE（公共超参）+ ABLATIONS（各变体 overrides）。

每个实验只改变一个因素（归因原则），输出目录分离，见方案文档 §5。stage2_train.py
按实验名取配置：``kwargs = BASE_RECIPE + overrides``。

实验矩阵（8 个训练 run）：
- p1  锚点基线：无语义头 = 纯 SSCL+蒸馏（硬约束参照系）
- e1a 完整语义头（残差 M + α·S）
- e2b 仅语义方向（M=1，掩码关闭）→ 掩码贡献 = e1a − e2b
- e1c 仅通道掩码（α=0，语义方向关闭）→ 语义方向贡献 = e1c − p1（或 e1a − e2b 交叉验证）
- e3b α 冻结在初始值 → α 可学习性 = e1a − e3b
- e3c novel 类 α 初始更大（0.5 vs 0.1）→ 差异化初始化 = e1a − e3c
- e4b ω=1（SSCL 语义加权退化为普通 SupCon）→ SSCL 语义加权价值 = e1a − e4b
- e4c SSCL 关闭（蒸馏保留）→ SSCL 独立增益 = e1a − e4c

注：原方案中的"平行 logit（1b）"与"残差 + 掩码关闭（e2b）"数学等价（平行 =
仅叠加 α·S，无通道掩码项），故不单独占一个 run，结构与掩码的消融由 e2b 覆盖。
"""

from __future__ import annotations

from pathlib import Path

from rfdetr.datasets.aug_configs import AUG_AERIAL

# --- 公共路径常量 ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_CHECKPOINT = str(PROJECT_ROOT / "output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth")
SEMANTIC_MATRIX_PATH = str(PROJECT_ROOT / "data/semantic_matrix_shwx.pt")
FSEM_PATH = str(PROJECT_ROOT / "data/fsem_shwx.pt")
CHANNEL_STATS_PATH = str(PROJECT_ROOT / "data/channel_stats_shwx.pt")
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"

# novel 类（少样本舰船）+ base 类（飞机 + FSC）
NOVEL_CLASSES = [0, 1, 2, 3]
BASE_CLASSES = list(range(4, 25))

# --- 公共训练配方（0807 Stage-2 配方 + 蒸馏，见方案文档 §5.2）---
BASE_RECIPE: dict[str, object] = {
    # 数据集
    "dataset_dir": DATASET_DIR,
    "dataset_file": "yolo",
    # 训练超参
    "epochs": 6,
    "batch_size": 32,
    "num_workers": 12,
    "lr": 1e-5,  # decoder / class_embed 组
    "lr_encoder": 1e-5,  # backbone 冻结，占位
    "weight_decay": 1e-4,
    "grad_accum_steps": 2,
    "clip_max_norm": 0.1,
    "lr_drop": 6,  # 恒定 LR（短训练不打步长）
    "warmup_epochs": 0.0,
    "use_ema": True,
    "compute_val_loss": False,
    "eval_interval": 1,
    # 数据增广
    "aug_config": AUG_AERIAL,
    "mosaic_p": 0.0,
    "multi_scale": False,
    # 硬件与日志
    "device": "cuda",
    "devices": 1,
    "num_nodes": 1,
    "tensorboard": True,
    "wandb": False,
    "resume": "",
    # --- SSCL（anchor=舰船 0-3，原型+投影头+实例正样本全开）---
    "sscl_enabled": True,
    "sscl_semantic_matrix_path": SEMANTIC_MATRIX_PATH,
    "sscl_matrix_normalize": "minmax",
    "sscl_lambda": 0.02,
    "sscl_tau": 0.1,
    "sscl_rho": 0.3,
    "sscl_omega_max": 2.0,
    "sscl_anchor_classes": NOVEL_CLASSES,
    "sscl_confusing_classes": NOVEL_CLASSES,
    "sscl_start_epoch": 0,
    "sscl_freeze_strategy": "conservative",
    "sscl_prototype_enabled": True,
    "sscl_prototype_momentum": 0.99,
    "sscl_prototype_min_samples": 1,
    "sscl_projection_enabled": True,
    "sscl_projection_dim": 128,
    "sscl_prototype_instance_pos": True,
    # --- 基类蒸馏（全部实验关闭——前期实验证明蒸馏无收益；字段保留以便将来对比）---
    "sscl_distill_enabled": False,
    "sscl_distill_lambda": 0.5,
    "sscl_distill_temperature": 2.0,
    "sscl_distill_mode": "mse",
    "sscl_teacher_checkpoint": None,
    "sscl_protected_classes": BASE_CLASSES,
    # --- 语义头（默认完整版：残差 M + α·S）---
    "semantic_head_enabled": True,
    "semantic_fsem_path": FSEM_PATH,
    "semantic_channel_stats_path": CHANNEL_STATS_PATH,
    "semantic_mask_enabled": True,
    "semantic_alpha_enabled": True,
    "semantic_alpha_learnable": True,
    "semantic_alpha_init": 0.1,
    "semantic_novel_alpha_init": 0.1,
    "semantic_alpha_max": 2.0,
    "semantic_mask_tau": 1.0,  # 有效 τ = max(该值, d/16)，d=256 时≈16
    "semantic_theta_init": 0.0,  # θ = d + 该值·τ；0 → θ=d（掩码梯度存活，之前 3.0 使 M 饱和）
    "semantic_novel_classes": NOVEL_CLASSES,
    "semantic_frozen_threshold_classes": NOVEL_CLASSES,
    "semantic_lr": 1e-4,
    "semantic_align_classes": NOVEL_CLASSES,
    "semantic_monitor_log_interval": 100,
}

# 每个实验的输出目录后缀与 overrides（只改一个因素）
ABLATIONS: dict[str, dict[str, object]] = {
    "p1": {
        "output_suffix": "p1-anchor",
        # 该实验复用已存在的 0807-SHWX-SSCL-Proj-原型+实例正样本 checkpoint（配置
        # 完全一致：6 epochs、lr=1e-5、conservative、原型+投影头+实例正样本、
        # λ=0.02、start=0、无蒸馏），eval_ablation 的 CHECKPOINT_OVERRIDES 处理，
        # 因此**不需要运行本实验**。
        "desc": "锚点基线：无语义头（纯 SSCL，复用 0807-Proj checkpoint，不运行）",
        "overrides": {"semantic_head_enabled": False},
    },
    "e1a": {
        "output_suffix": "e1a-full",
        "desc": "完整语义头（残差 M + α·S）",
        "overrides": {},
    },
    "e2b": {
        "output_suffix": "e2b-maskoff",
        "desc": "仅语义方向（M=1 掩码关闭）",
        "overrides": {"semantic_mask_enabled": False},
    },
    "e1c": {
        "output_suffix": "e1c-alpha0",
        "desc": "仅通道掩码（α=0 语义方向关闭）",
        "overrides": {"semantic_alpha_enabled": False, "semantic_alpha_learnable": False},
    },
    "e3b": {
        "output_suffix": "e3b-alphafix",
        "desc": "α 冻结在初始值（0.1）",
        "overrides": {"semantic_alpha_learnable": False},
    },
    "e3c": {
        "output_suffix": "e3c-novelalpha",
        "desc": "novel 类 α 初始更大（0.5 vs base 0.1）",
        "overrides": {"semantic_novel_alpha_init": 0.5},
    },
    "e4b": {
        "output_suffix": "e4b-omega1",
        "desc": "SSCL 语义加权退化（ω≡1，SupCon）",
        "overrides": {"sscl_rho": 0.0},
    },
    "e4c": {
        "output_suffix": "e4c-nosscl",
        "desc": "SSCL 关闭（仅语义头，验证 SSCL 独立增益）",
        "overrides": {"sscl_enabled": False, "semantic_head_enabled": True},
    },
}

# 默认实验（不带参数运行时）
DEFAULT_EXPERIMENT = "e1a"
