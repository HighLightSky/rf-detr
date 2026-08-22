# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""阶段 0：离线构建多模态原型产物（视觉 + 文本）。

供 ``ProtoGuidance`` 模块（多模态原型引导 query selection）消费。视觉原型
从"已适配 RF-DETR 的 backbone/projector 特征空间"（hidden_dim=256，P4 单
尺度）按 GT 框 masked average pooling 提取实例特征，每类经余弦 k-means
聚类为 ``num_slots`` 个槽位子原型（实例不足时槽位标记无效）；文本原型用
CLIP 对遥感提示词（``sscl/prompts/shwx.yaml``）编码取平均，保持原始 768 维
（投影在训练期由 ``ProtoGuidance`` 学习）。

与训练侧一致的组件（对齐修复的教训，见 stage0_collect_features.py）：
- ``RFDETRMedium`` 构造 + ``pretrain_weights=0805 checkpoint``（与训练相同的
  骨干/projector 权重，保证特征空间一致）；
- ``RFDETRDataModule`` 构建训练集 DataLoader（与训练相同的数据分布）。

用法：
    python src/scripts/semantic_experiments/stage0_build_proto_guidance.py

输出：
    data/proto_guidance_shwx.pt —— 含 ``visual_prototypes [C, M, d]``、
    ``valid_slots [C, M]``、``text_prototypes [C, 768]``、``class_names``、
    ``meta`` 等键，由 ``ProtoGuidance.build`` 加载。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
import yaml
from tqdm.auto import tqdm  # 注意：必须用 tqdm.auto，不能用 from tqdm import tqdm

from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.sscl.prompts import load_class_prompts
from rfdetr.sscl.proto_guidance.artifacts import save_proto_artifacts
from rfdetr.sscl.semantic_matrix import encode_class_text_embeddings
from rfdetr.training import RFDETRDataModule
from rfdetr.variants import RFDETRMedium

# ============================================================================
# 配置 —— 在这里修改
# ============================================================================

# 以下配置可通过环境变量覆盖，默认值保持旧 25 类实验行为。
PROTO_DATASET = os.environ.get("RFDETR_PROTO_DATASET", "shwx")
BASE_CHECKPOINT = str(
    Path(
        os.environ.get(
            "RFDETR_PROTO_CHECKPOINT",
            "output/0813-SHWX-rfdetr-medium-baseline-精细标注/checkpoint_best_total.pth",
        )
    ).resolve()
)

# SHWX 数据集（YOLO 布局；与 configs/experiments/*.yaml 的 dataset_dir 一致）
DATASET_DIR = os.environ.get(
    "RFDETR_PROTO_DATASET_DIR",
    "/home/liu/wzt/datasets/SHWX-dataset-dict-redo",
)
DATASET_FILE = "yolo"

NUM_CLASSES = int(os.environ.get("RFDETR_PROTO_NUM_CLASSES", "25"))
RESOLUTION = int(os.environ.get("RFDETR_PROTO_RESOLUTION", "640"))
BATCH_SIZE = int(os.environ.get("RFDETR_PROTO_BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("RFDETR_PROTO_NUM_WORKERS", "8"))
DEVICE = os.environ.get("RFDETR_PROTO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 每类视觉子原型槽位数 M（与 proto_guidance_num_slots 配置一致）
NUM_SLOTS = 10
# 余弦 k-means 聚类迭代轮数
CLUSTER_ITERS = 20

# 输出路径（训练配置 proto_guidance_artifacts_path 引用该文件）
OUTPUT_FILE = str(
    Path(
        os.environ.get(
            "RFDETR_PROTO_OUTPUT",
            f"data/proto_guidance_{PROTO_DATASET}.pt",
        )
    ).resolve()
)

CLASS_NAMES, CLASS_PROMPTS = load_class_prompts(PROTO_DATASET)
if len(CLASS_NAMES) != NUM_CLASSES or set(CLASS_NAMES) != set(range(NUM_CLASSES)):
    raise ValueError(
        f"提示词类别与 NUM_CLASSES 不一致: dataset={PROTO_DATASET!r}, "
        f"类别={sorted(CLASS_NAMES)}, NUM_CLASSES={NUM_CLASSES}"
    )


def _cluster_slots(
    features: torch.Tensor, num_slots: int, iters: int = CLUSTER_ITERS
) -> tuple[torch.Tensor, torch.Tensor]:
    """按余弦相似度把实例特征聚类为槽位子原型。

    Args:
        features: 某类实例特征 ``[N, d]``。
        num_slots: 目标槽位数 M。
        iters: k-means 迭代轮数。

    Returns:
        ``(centroids, valid)``：中心 ``[num_slots, d]``（不足部分补零），
        有效掩码 ``[num_slots]``（bool）。
    """
    n, d = features.shape
    k = min(num_slots, n)
    centroids = torch.zeros(num_slots, d, device=features.device)
    valid = torch.zeros(num_slots, dtype=torch.bool, device=features.device)
    if n == 0:
        return centroids, valid

    # 初始化：均匀抽样的 K 个实例（随机种子固定，产物可复现）
    perm = torch.randperm(n)[:k]
    centers = features[perm].clone()
    norm_features = F.normalize(features, dim=-1)
    for _ in range(iters):
        sim = norm_features @ F.normalize(centers, dim=-1).T  # [N, K]
        assign = sim.argmax(dim=1)
        for kk in range(k):
            mask = assign == kk
            if bool(mask.any()):
                centers[kk] = features[mask].mean(dim=0)
    centroids[:k] = centers
    valid[:k] = True
    return centroids, valid


def _validate_dataset_num_classes() -> None:
    """在 CUDA 前校验数据集类别数，避免标签错误表现为设备端断言。"""
    data_yaml_path = Path(DATASET_DIR) / "data.yaml"
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml_path}")
    with data_yaml_path.open(encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    if not isinstance(data_config, dict):
        raise ValueError(f"数据集配置必须是映射: {data_yaml_path}")
    dataset_num_classes = data_config.get("nc")
    if dataset_num_classes != NUM_CLASSES:
        raise ValueError(
            f"数据集类别数与原型配置不一致: data.yaml nc={dataset_num_classes}, "
            f"NUM_CLASSES={NUM_CLASSES}, DATASET_DIR={DATASET_DIR}"
        )


def main() -> None:
    """执行视觉原型提取、文本原型编码与产物保存。"""
    _validate_dataset_num_classes()
    print(f"加载 backbone checkpoint: {BASE_CHECKPOINT}")
    print(f"原型数据集目录: {DATASET_DIR}")
    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=RESOLUTION,
        gradient_checkpointing=False,
        pretrain_weights=BASE_CHECKPOINT,
    )
    net = model.model.model  # 底层 LWDETR 模块
    net.eval()
    net.to(DEVICE)

    train_config = model.get_train_config(
        dataset_dir=DATASET_DIR,
        dataset_file=DATASET_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        aug_config=AUG_AERIAL,
        mosaic_p=0.0,
        multi_scale=False,
    )
    datamodule = RFDETRDataModule(model.model_config, train_config)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    print(f"训练集 batch 数: {len(loader)}")

    # 收集每类实例特征（backbone/projector 特征空间，P4 单尺度）
    features_by_class: dict[int, list[torch.Tensor]] = {c: [] for c in range(NUM_CLASSES)}
    for samples, targets in tqdm(loader, desc="提取 backbone 特征"):
        samples = samples.to(DEVICE)
        targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            features, _, _ = net.backbone(samples)
        feat = features[0].tensors  # [B, d, H, W]（P4 单尺度）
        bs, d, h, w = feat.shape
        for img_idx in range(bs):
            boxes = targets[img_idx].get("boxes")
            labels = targets[img_idx].get("labels")
            if boxes is None or labels is None or boxes.numel() == 0:
                continue
            img_feat = feat[img_idx]  # [d, H, W]
            for box, label in zip(boxes, labels, strict=False):
                class_id = int(label.item())
                if class_id >= NUM_CLASSES:
                    continue
                cx, cy, bw, bh = box.tolist()
                x0 = max(int((cx - bw / 2) * w), 0)
                y0 = max(int((cy - bh / 2) * h), 0)
                x1 = min(int((cx + bw / 2) * w) + 1, w)
                y1 = min(int((cy + bh / 2) * h) + 1, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                region = img_feat[:, y0:y1, x0:x1]
                features_by_class[class_id].append(region.mean(dim=(1, 2)))

    # 每类聚类为槽位子原型
    visual_prototypes = torch.zeros(NUM_CLASSES, NUM_SLOTS, d)
    valid_slots = torch.zeros(NUM_CLASSES, NUM_SLOTS, dtype=torch.bool)
    for class_id in range(NUM_CLASSES):
        feats = features_by_class[class_id]
        if not feats:
            print(f"  警告: 类别 {class_id} 没有实例特征，槽位全部无效")
            continue
        class_features = torch.stack(feats)
        centroids, valid = _cluster_slots(class_features, NUM_SLOTS)
        visual_prototypes[class_id] = centroids
        valid_slots[class_id] = valid
        print(f"  类别 {class_id} ({CLASS_NAMES.get(class_id, '?')}): "
              f"{class_features.shape[0]} 个实例 -> {int(valid.sum())} 个槽位")

    # 文本原型：CLIP 对遥感提示词编码取平均（多 prompt 平均，768 维不归一化）
    print("编码 CLIP 文本原型 ...")
    text_prototypes = encode_class_text_embeddings(CLASS_PROMPTS, device=DEVICE)

    meta = {
        "dataset": PROTO_DATASET,
        "num_classes": NUM_CLASSES,
        "hidden_dim": int(d),
        "text_dim": int(text_prototypes.shape[1]),
        "num_slots": NUM_SLOTS,
        "resolution": RESOLUTION,
        "checkpoint": BASE_CHECKPOINT,
        "cluster_iters": CLUSTER_ITERS,
        "note": "视觉原型: backbone/projector P4 特征 + GT box masked avg pool + 余弦 k-means",
    }
    class_names = [CLASS_NAMES.get(c, f"c{c}") for c in range(NUM_CLASSES)]
    save_proto_artifacts(
        OUTPUT_FILE,
        visual_prototypes=visual_prototypes,
        valid_slots=valid_slots,
        text_prototypes=text_prototypes,
        class_names=class_names,
        meta=meta,
    )
    print(f"完成: {OUTPUT_FILE}（视觉原型 {tuple(visual_prototypes.shape)}，"
          f"有效槽位 {int(valid_slots.sum())}/{NUM_CLASSES * NUM_SLOTS}）")


if __name__ == "__main__":
    main()
