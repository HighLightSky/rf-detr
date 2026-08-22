# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""阶段 0 诊断：验证 novel（舰船）类语义方向是否与舰船特征对齐。

背景：f_sem 只用 base 类（飞机+FSC）训练，其"文本→方向"映射对飞机类有效
（训练监控 align_cos_base≈+0.45），但套用到舰船文本时可能落到舰船特征空间
即失效（训练监控 align_cos_novel≈−0.06）。本脚本按与 Stage-2 一致的分布
（train 模式 + 航拍增广）收集舰船 matched 特征，计算：

    cos(mean(h_c), s_c)          —— 同类对齐度（应为正且显著）
    mean_{c'≠c} cos(mean(h_c), s_c') —— 跨类对齐度（应远低于同类）
    gap = 同类 − 跨类

若舰船类同类对齐度 ≈ 0 或为负 → 坐实"文本派生舰船方向不可用"的根因。

用法：
    python src/scripts/semantic_experiments/stage0_diag_novel_align.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm  # 注意：必须用 tqdm.auto

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.sscl.fsem import evaluate_alignment, load_fsem_artifacts
from rfdetr.training import RFDETRDataModule
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 配置
# ============================================================================

BASE_CHECKPOINT = str(Path("output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth").resolve())
FSEM_PATH = str(Path("data/fsem_shwx.pt").resolve())
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"
DATASET_FILE = "yolo"

# 舰船 novel 类（QHS/MS 样本充足，HM/LQS 少）
NOVEL_CLASSES = [0, 1, 2, 3]
NUM_CLASSES = 25
RESOLUTION = 640
BATCH_SIZE = 16
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 Stage-2 训练一致：train 模式（group_detr=13）+ 航拍增广
TRAIN_MODE = True
AUG_CONFIG = AUG_AERIAL


def main() -> None:
    """收集舰船 matched 特征并计算与 s_ship 的对齐度。"""
    print(f"加载 Stage-1 checkpoint: {BASE_CHECKPOINT}")
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=RESOLUTION,
        gradient_checkpointing=False,
        pretrain_weights=BASE_CHECKPOINT,
    )
    net = model.model.model
    if TRAIN_MODE:
        net.train()
        print(f"收集模式: train（group_detr={model.model_config.group_detr}）+ 航拍增广")
    else:
        net.eval()
    net.to(DEVICE)

    train_config = model.get_train_config(
        dataset_dir=DATASET_DIR,
        dataset_file=DATASET_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        aug_config=AUG_CONFIG,
        mosaic_p=0.0,
        multi_scale=False,
    )
    datamodule = RFDETRDataModule(model.model_config, train_config)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    criterion, _ = build_criterion_from_config(model.model_config, train_config)
    group_detr = int(model.model_config.group_detr) if TRAIN_MODE else 1

    # 收集舰船 matched 特征
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for samples, targets in tqdm(loader, desc="收集舰船 matched 特征"):
        samples = samples.to(DEVICE)
        targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = net(samples, targets)
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            indices = criterion.matcher(outputs_without_aux, targets, group_detr=group_detr)
            batch_idx, src_idx = criterion._get_src_permutation_idx(indices)
            features = outputs["hs"][batch_idx, src_idx]
            labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        keep = torch.isin(labels, torch.tensor(NOVEL_CLASSES, device=labels.device))
        if keep.any():
            all_features.append(features[keep].float().cpu())
            all_labels.append(labels[keep].cpu())

    features_all = torch.cat(all_features, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    print(f"收集完成: {features_all.shape[0]} 个舰船实例（维度 {features_all.shape[1]}）")

    # 按类分组
    class_names = model.class_names
    features_by_class = {c: features_all[labels_all == c] for c in NOVEL_CLASSES}
    for c in NOVEL_CLASSES:
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  类 {c} ({name}): {features_by_class[c].shape[0]} 个实例")

    # 加载当前 S 矩阵并计算对齐
    fsem = load_fsem_artifacts(FSEM_PATH)
    s_matrix = fsem["S"].float()
    print(f"S 矩阵形状: {tuple(s_matrix.shape)}")

    report = evaluate_alignment(features_by_class, s_matrix)
    print("\n===== 舰船 novel 类对齐诊断 =====")
    for c, v in sorted(report["per_class"].items()):
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  类 {c} ({name}): 同类cos={v['align']:.4f} 跨类均值cos={v['cross_mean']:.4f} gap={v['gap']:.4f}")
    print(f"  均值: mean_align={report['mean_align']:.4f} mean_gap={report['mean_gap']:.4f}")

    # 判定
    print("\n===== 根因判定 =====")
    if report["mean_align"] < 0.1:
        print("❌ 舰船同类对齐度 ≈ 0/负 → 文本派生的舰船语义方向不可用（坐实根因）")
        print("  建议：舰船方向改数据派生（类中心/原型），或放弃文本锚定舰船方向")
    elif report["mean_align"] >= 0.3 and report["mean_gap"] >= 0.2:
        print("✅ 舰船对齐度达标 → 阶段 0 门控应扩展到 novel 类，问题在别处")
    else:
        print("⚠️ 舰船对齐度介于中间 → 需结合训练监控判断")


if __name__ == "__main__":
    main()
