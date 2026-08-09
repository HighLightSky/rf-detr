# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""阶段 0 第一步：用 Stage-1（0805 全量微调）模型在 base 类上收集 matched query 特征。

语义头方案要求 f_sem 与通道 TF-IDF 统计都建立在"已适配 SHWX"的特征空间上，
且训练数据**只含 base 类**（飞机 4-23 + FSC 24，绝不含少样本舰船类 HM/LQS/QHS/MS），
避免少样本类噪声把对齐带歪。

本脚本复用训练侧完全一致的组件，保证收集到的特征与在线训练时的分布一致：
- ``RFDETRMedium`` 构造 + ``pretrain_weights=0805 checkpoint``；
- ``RFDETRDataModule`` 构建训练集 DataLoader（关闭增广，特征更稳定）；
- ``build_criterion_from_config`` 构建与训练一致的 Hungarian matcher，在 eval
  输出上做匹配（eval 模式 group_detr=1），提取 matched query 特征。

用法：
    python src/scripts/ret-sscl/stage0_collect_features.py

输出：
    data/fsem_collect_0805.pt —— 含 ``features``/``labels``/``class_names``/
    ``num_instances_per_class``/``checkpoint`` 等键，供 stage0_train_fsem.py 消费。
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm  # 注意：必须用 tqdm.auto，不能用 from tqdm import tqdm

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.training import RFDETRDataModule
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 配置 —— 在这里修改
# ============================================================================

# Stage-1 全量微调 checkpoint（已适配 SHWX 的骨干权重）
BASE_CHECKPOINT = str(
    Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth").resolve()
)

# SHWX 数据集（YOLO 布局）
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# base 类 = 飞机(4-23) + FSC(24)，与方案文档一致；少样本舰船类(0-3)不参与收集
BASE_CLASSES = list(range(4, 25))

NUM_CLASSES = 25
RESOLUTION = 640
BATCH_SIZE = 16
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [对齐修复] 关键开关：Stage-2 训练用的是 train 模式（group_detr=13）+ 航拍增广，
# 而旧版用 eval 模式 + 无增广收集，导致 f_sem 方向与训练特征分布不对齐（E1a 观测到
# align_cos≈0）。开启 TRAIN_MODE 后按与 Stage-2 一致的分布收集，保证语义方向落得准。
TRAIN_MODE = True  # True: 与 Stage-2 训练分布一致（train 模式 + 增广）；False: 旧版行为
AUG_CONFIG = AUG_AERIAL  # 与 Stage-2 训练相同的增广（旋转90°/翻转/亮度）

# 收集结果输出路径（供 stage0_train_fsem.py 读取）
OUTPUT_FILE = str(Path("data/fsem_collect_0805_train_aug.pt").resolve())


def main() -> None:
    """执行 base 类 matched query 特征收集并保存。"""
    project_root = Path(__file__).resolve().parents[2]

    print(f"加载 Stage-1 checkpoint: {BASE_CHECKPOINT}")
    # 构造模型（关闭 gradient_checkpointing：仅前向，无内存压力，避免 eval+no_grad 下的不确定行为）
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=RESOLUTION,
        gradient_checkpointing=False,
        pretrain_weights=BASE_CHECKPOINT,
    )
    net = model.model.model  # 底层 LWDETR 模块
    # [对齐修复] TRAIN_MODE=True 时用 train 模式（group_detr=13）收集，与 Stage-2
    # 训练分布一致；False 则用 eval 模式（旧版行为）
    if TRAIN_MODE:
        net.train()
        print(f"收集模式: train（group_detr={model.model_config.group_detr}）+ 航拍增广")
    else:
        net.eval()
        print("收集模式: eval（单组查询，无增广）")
    net.to(DEVICE)

    # 构建训练配置：与 Stage-2 一致的增广（AUG_AERIAL、mosaic 0、多尺度关）
    train_config = model.get_train_config(
        dataset_dir=DATASET_DIR,
        dataset_file=DATASET_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        aug_config=AUG_CONFIG,
        mosaic_p=0.0,
        multi_scale=False,
    )

    # 构建 DataModule 与 criterion（含与训练一致的 Hungarian matcher）
    datamodule = RFDETRDataModule(model.model_config, train_config)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    criterion, _ = build_criterion_from_config(model.model_config, train_config)
    # train 模式下 matcher 用 group_detr（13 组），与训练时的匹配一致
    group_detr = int(model.model_config.group_detr) if TRAIN_MODE else 1
    print(f"训练集 batch 数: {len(loader)}（base 类 {BASE_CLASSES}，matcher group_detr={group_detr}）")

    # 收集 matched query 特征
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for samples, targets in tqdm(loader, desc="收集 base 类 matched 特征"):
        samples = samples.to(DEVICE)
        targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = net(samples, targets)
            # 与 criterion forward 一致：matcher 只作用于不含 aux_outputs 的输出
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            indices = criterion.matcher(outputs_without_aux, targets, group_detr=group_detr)
            # 提取 matched foreground query 特征（与 module_model._extract_matched_query_features 同路径）
            batch_idx, src_idx = criterion._get_src_permutation_idx(indices)
            features = outputs["hs"][batch_idx, src_idx]
            labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # 仅保留 base 类的特征
        keep = torch.isin(labels, torch.tensor(BASE_CLASSES, device=labels.device))
        if keep.any():
            all_features.append(features[keep].float().cpu())
            all_labels.append(labels[keep].cpu())

    features_all = torch.cat(all_features, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    print(f"收集完成: {features_all.shape[0]} 个 base 类实例，特征维度 {features_all.shape[1]}")

    # 统计每类实例数（校验类分布健康度）
    class_names = model.class_names
    num_per_class = {int(c): int((labels_all == c).sum().item()) for c in BASE_CLASSES}
    for c in BASE_CLASSES:
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  类 {c} ({name}): {num_per_class[c]} 个实例")
    if any(n == 0 for n in num_per_class.values()):
        print("警告: 存在 0 实例的 base 类，建议检查数据划分与匹配情况。")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features_all,
            "labels": labels_all,
            "class_names": class_names,
            "checkpoint": BASE_CHECKPOINT,
            "num_instances_per_class": num_per_class,
        },
        OUTPUT_FILE,
    )
    print(f"结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
