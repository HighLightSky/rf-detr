# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR LoRA 微调训练脚本 —— 冻结 DINO 主干，使用 LoRA 适配器进行参数高效微调。

用法：
    python src/scripts/train_LoRA.py

训练策略：
    - 冻结 DINOv2 编码器（freeze_encoder=True），仅训练解码器和 LoRA 适配器
    - 在 backbone 的注意力投影层（q_proj, v_proj, k_proj 等）注入 LoRA/DoRA 适配器
    - DoRA（Weight-Decomposed Low-Rank Adaptation）分解预训练权重为方向+幅度，
      仅微调幅度分量，相比普通 LoRA 有更好的微调效果
    - 配合 Mosaic 数据增强，提升小目标检测性能
    - 针对军事舰船/飞行器数据集使用遥感航拍增广预设

显存参考：
    - 冻结 backbone 可节省约 60-70% 的编码器显存
    - LoRA 适配器参数量仅为 backbone 的 ~1%，训练显存大幅降低
    - 单卡 24 GB (RTX 3090/4090) 可轻松运行 medium 模型 + batch_size=16

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
# 可选: nano, small, medium, large
# LoRA 微调时推荐使用 small 或 medium，在精度和显存之间取得平衡
MODEL = "medium"

# --- 数据集 ---
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"            # YOLO 格式数据集（包含 data.yaml）

# --- 训练超参数 ---
NUM_CLASSES = 25          # 类别数（4 类舰船 + 20 类飞行器 + 1 类其他）
EPOCHS = 60               # 训练轮数（LoRA 微调收敛快，60 epochs 通常够用）

# ⚠️ batch_size 不要设太大！原因：
#   - group_detr=13 让训练时每张图产生 300×13=3900 个 query
#   - loss_boxes 中 generalized_box_iou 计算全量 pairwise [N,N] 矩阵（O(N²) 显存）
#   - 设 8，通过 grad_accum_steps 补偿有效 batch size
BATCH_SIZE = 12            # 每 GPU 的 batch size
NUM_WORKERS = 8           # DataLoader 工作进程数（Mosaic 重CPU，过大反而争抢）
LR = 2e-4                 # 基础学习率（解码器部分）
LR_ENCODER = 1e-4         # 编码器学习率（LoRA 适配器参数）
WEIGHT_DECAY = 1e-4       # 权重衰减
GRAD_ACCUM_STEPS = 8      # 梯度累积步数（有效 batch = 8 × 8 = 64）
CLIP_MAX_NORM = 0.1       # 梯度裁剪

# --- 学习率调度 ---
LR_DROP = 40              # 学习率下降的 epoch 数
WARMUP_EPOCHS = 3.0       # 预热 epoch 数（LoRA 微调建议使用少量 warmup 稳定初期训练）

# --- 数据增广 ---
# 可选预设:
#   None          → 使用默认 torchvision 增广（HorizontalFlip + 多尺度缩放裁剪）
#   AUG_CONSERVATIVE → 保守增广（小数据集推荐，<500 张）
#   AUG_AGGRESSIVE   → 激进增广（大数据集推荐，2000+ 张）
#   AUG_AERIAL       → 航拍/遥感影像（水平/垂直翻转 + 90° 旋转）
#   AUG_INDUSTRIAL   → 工业/检测影像（光照噪声 + 模糊）
#   {}             → 关闭额外增广
AUG_CONFIG = AUG_AERIAL   # 军事舰船/飞行器 → 遥感航拍预设

# --- Mosaic 增强 ---
# Mosaic 将 4 张图片拼接为 1 张训练样本，有效提升小目标检测和遥感数据集的性能。
# 推荐值: 0.5 ~ 1.0（前 80% 训练阶段开启，最后 20% 关闭）
# 设为 0.0 关闭
MOSAIC_P = 0.8

# --- 硬件 ---
DEVICE = "cuda"           # 设备: cuda, cuda:0, cpu
DEVICES = 1               # GPU 数量
NUM_NODES = 1             # 节点数

# --- DataLoader 性能调优 ---
# Mosaic + Albumentations 增广都在 CPU 上执行，以下参数减少 CPU 阻塞，提升 GPU 利用率
PERSISTENT_WORKERS = True   # 复用 worker 进程，避免每个 epoch 重建（减少 fork 开销）
PREFETCH_FACTOR = 4         # 每个 worker 预取 batch 数（减少 GPU 等待数据）
PIN_MEMORY = True           # 锁页内存，加速 CPU→GPU 数据传输

# --- 多尺度训练 ---
# multi_scale=True 会随机放大输入分辨率（640 最高到 ~800），增加显存压力
MULTI_SCALE = True

# --- 输出 & 日志 ---
# 使用脚本所在的项目根目录作为基准，避免从不同目录运行导致输出路径不一致
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = str(_PROJECT_ROOT / "output/0728-SHWX-rfdetr_medium_LoRA")   # 输出目录
TENSORBOARD = True                  # 是否启用 TensorBoard
WANDB = False                       # 是否启用 Wandb

# --- EMA (指数移动平均) ---
# 对模型权重做指数移动平均，提升模型的泛化能力和稳定性
USE_EMA = True

# --- 验证 ---
EVAL_INTERVAL = 5         # 每隔 N 个 epoch 验证一次（减少 CPU 阻塞，加快训练）

# --- 恢复训练 ---
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
    """构建 LoRA 微调模型并启动训练。"""
    # --- 校验模型 ---
    if MODEL not in _MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {MODEL}，可选: {list(_MODEL_REGISTRY)}")

    model_cls, default_resolution = _MODEL_REGISTRY[MODEL]
    dataset_dir = str(Path(DATASET_DIR).resolve())

    print(f"模型: {MODEL} (LoRA 微调) | 类别数: {NUM_CLASSES} | 分辨率: {default_resolution}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} (有效={BATCH_SIZE * GRAD_ACCUM_STEPS}) | Epochs: {EPOCHS}")
    print(f"数据集: {dataset_dir} | 输出: {OUTPUT_DIR}")
    print(f"增广预设: {'无' if AUG_CONFIG is None else AUG_CONFIG}")
    print(f"Mosaic: p={MOSAIC_P} | 冻结Backbone + DoRA LoRA")

    # --- 构建模型（冻结 backbone + LoRA 微调） ---
    # freeze_encoder=True: 冻结 DINOv2 主干网络的所有参数，不参与梯度更新
    # backbone_lora=True:  在 backbone 的注意力层注入 LoRA/DoRA 适配器，
    #                       仅训练这些低秩适配器，参数量约为 backbone 的 ~1%
    # gradient_checkpointing=True: 用计算换显存，反向传播时重新计算激活值
    model = model_cls(
        num_classes=NUM_CLASSES,
        resolution=default_resolution,
        freeze_encoder=True,         # 冻结 DINO 主干网络
        backbone_lora=True,          # 注入 LoRA 适配器
        gradient_checkpointing=True, # 节省显存
    )

    # --- 训练 ---
    # 注意：LoRA 微调模式下，编码器学习率 lr_encoder 仅作用于 LoRA 适配器参数，
    # 因为 backbone 原始权重已被冻结（requires_grad=False），不会被优化器更新。
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
        use_ema=USE_EMA,           # EMA 提升模型泛化能力
        compute_val_loss=False,    # 关掉验证 loss 计算，省显存，mAP 指标不受影响
        aug_config=AUG_CONFIG if AUG_CONFIG is not None else {},
        mosaic_p=MOSAIC_P,         # Mosaic 增强，提升小目标和遥感场景性能
        # --- 性能优化 ---
        persistent_workers=PERSISTENT_WORKERS,  # 复用 DataLoader worker 进程，减少 CPU 开销
        prefetch_factor=PREFETCH_FACTOR,        # 预取 batch，减少 GPU 等待时间
        pin_memory=PIN_MEMORY,                  # 锁页内存加速 CPU→GPU 传输
        multi_scale=MULTI_SCALE,                # 关闭多尺度，减少显存压力和 CPU 开销
    )

    print(f"\n训练完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
