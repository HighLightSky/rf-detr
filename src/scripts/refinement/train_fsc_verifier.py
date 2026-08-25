# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练 FSC/非FSC 二阶段视觉复核器，不读取测试集。"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rfdetr.refinement.fsc_two_stage import FSCVerifier, FSCVerifierPolicy, crop_fsc_context, crop_transform
from val.competition_metrics import BoxRecord, EvalConfig, evaluate_competition_metrics


def _parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练 FSC/非FSC 二阶段复核器")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--context-scale", type=float, default=None, help="覆盖缓存的上下文裁剪倍数")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖缓存的二级输入尺寸")
    parser.add_argument("--architecture", choices=["resnet18", "mobilenet_v3_small"], default="resnet18")
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="困难样本聚焦系数，0 表示普通交叉熵")
    parser.add_argument("--hard-negative-checkpoint", default=None, help="上一轮二级模型，用于识别训练集难负样本")
    parser.add_argument("--hard-negative-repeat", type=int, default=4, help="第一轮仍判错的训练负样本重复次数")
    return parser.parse_args()


def _load_cache(path: Path) -> dict[str, Any]:
    """读取并校验 FSC 候选缓存。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1":
        raise ValueError("不是 shwx-fsc-verifier-cache-v1 缓存")
    if payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("二级训练缓存不得包含测试集")
    return payload


class _CropDataset(Dataset[tuple[Tensor, int]]):
    """将一级候选和 GT FSC 框转换为二级分类 crop。"""

    def __init__(self, rows: list[dict[str, Any]], policy: FSCVerifierPolicy, training: bool) -> None:
        """初始化样本行、固定裁剪策略与图像变换。"""
        self.rows = rows
        self.policy = policy
        self.transform = crop_transform(training=training)

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """读取一个候选上下文 crop 与二分类标签。"""
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            crop = crop_fsc_context(
                image,
                row["xyxy"],
                context_scale=self.policy.context_scale,
                output_size=self.policy.image_size,
            )
        return self.transform(crop), int(row["label"])


class _FocalLoss(nn.Module):
    """对卡车等高混淆候选提高训练权重的多类 focal loss。"""

    def __init__(self, gamma: float) -> None:
        """初始化困难样本聚焦系数。"""
        super().__init__()
        if gamma < 0:
            raise ValueError("focal-gamma 不能为负数")
        self.gamma = gamma

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """计算每个样本按预测概率缩放后的交叉熵。"""
        cross_entropy = nn.functional.cross_entropy(logits, labels, reduction="none")
        if self.gamma == 0:
            return cross_entropy.mean()
        probability = torch.exp(-cross_entropy)
        return ((1.0 - probability).pow(self.gamma) * cross_entropy).mean()


def _worker_init(_: int) -> None:
    """避免图像加载 worker 争抢 CPU 线程。"""
    torch.set_num_threads(1)


def _loader(
    dataset: _CropDataset,
    batch_size: int,
    num_workers: int,
    *,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """构造稳定的训练或验证数据加载器。"""
    settings: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": _worker_init,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        settings["prefetch_factor"] = 2
    if sampler is None:
        settings["shuffle"] = False
    else:
        settings["sampler"] = sampler
    return DataLoader(dataset, **settings)


def _balanced_loader(dataset: _CropDataset, batch_size: int, num_workers: int, seed: int) -> DataLoader[tuple[Tensor, Tensor]]:
    """按 FSC 与非FSC 类别均衡抽取训练样本。"""
    counts = Counter(int(row["label"]) for row in dataset.rows)
    if set(counts) != {0, 1}:
        raise ValueError(f"训练集必须同时包含 FSC 和非FSC 候选，实际为 {dict(counts)}")
    weights = [1.0 / counts[int(row["label"])] for row in dataset.rows]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(weights), batch_size),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return _loader(dataset, batch_size, num_workers, sampler=sampler)


def _classification_metrics(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> dict[str, float]:
    """计算验证候选的二级分类准确率与两类 F1。"""
    matrix = torch.zeros((2, 2), dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            for actual, estimated in zip(labels, predicted, strict=True):
                matrix[int(actual), int(estimated)] += 1
    values: dict[str, float] = {"accuracy": float(matrix.diag().sum() / max(matrix.sum(), 1))}
    for class_id, name in ((1, "fsc"), (0, "non_fsc")):
        tp = float(matrix[class_id, class_id])
        precision = tp / max(float(matrix[:, class_id].sum()), 1.0)
        recall = tp / max(float(matrix[class_id, :].sum()), 1.0)
        values[f"{name}_f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
    return values


def _detection_f1(verifier: FSCVerifier, cache: dict[str, Any], device: torch.device) -> dict[str, float]:
    """在验证集一级候选上评估固定 argmax 复核的端到端 FSC 指标。"""
    rows = [row for row in cache["candidates"] if row["split"] == "val"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image_id"], []).append(row)
    predictions: list[BoxRecord] = []
    for image_rows in grouped.values():
        with Image.open(image_rows[0]["image"]) as image:
            probabilities = verifier.predict_probabilities(image, [row["xyxy"] for row in image_rows]).detach().cpu()
        for row, probability in zip(image_rows, probabilities, strict=True):
            if int(probability.argmax().item()) == 1:
                predictions.append(BoxRecord(row["image_id"], 24, tuple(row["xyxy"]), float(row["score"])))
    ground_truth = [
        BoxRecord(row["image_id"], 24, tuple(row["xyxy"])) for row in cache["ground_truth"] if row["split"] == "val"
    ]
    metrics = evaluate_competition_metrics(
        ground_truth,
        predictions,
        EvalConfig(class_to_group={24: "FSC"}, group_iou_thresholds={"FSC": 0.35}, default_iou_threshold=0.35),
    )["groups"]["FSC"]
    return {"recall": metrics.recall, "precision": metrics.precision, "f1": 2 * metrics.precision * metrics.recall / max(metrics.precision + metrics.recall, 1e-8)}


def _repeat_hard_negatives(
    rows: list[dict[str, Any]],
    checkpoint: str | None,
    repeat: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    """仅用训练集候选找出上一轮仍误判为 FSC 的负样本并重复采样。"""
    if checkpoint is None or repeat <= 1:
        return rows, 0
    verifier = FSCVerifier.from_checkpoint(checkpoint, device=device)
    negatives = [row for row in rows if int(row["label"]) == 0 and row["source"] == "detector_candidate"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in negatives:
        grouped.setdefault(row["image_id"], []).append(row)
    hard: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        with Image.open(image_rows[0]["image"]) as image:
            probabilities = verifier.predict_probabilities(image, [row["xyxy"] for row in image_rows]).detach().cpu()
        hard.extend(row for row, probability in zip(image_rows, probabilities, strict=True) if int(probability.argmax()) == 1)
    return rows + hard * (repeat - 1), len(hard)


def main() -> None:
    """仅使用训练集更新权重，并在验证集选择最佳训练轮次。"""
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.freeze_epochs < 0:
        raise ValueError("epochs、batch-size 必须为正数，freeze-epochs 不能为负数")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cache_path = Path(args.cache).resolve()
    cache = _load_cache(cache_path)
    policy = FSCVerifierPolicy.from_mapping(cache["metadata"]["policy"])
    policy = replace(policy, architecture=args.architecture)
    if args.context_scale is not None or args.image_size is not None:
        policy = replace(
            policy,
            context_scale=policy.context_scale if args.context_scale is None else args.context_scale,
            image_size=policy.image_size if args.image_size is None else args.image_size,
        )
        policy.validate()
    if args.hard_negative_repeat <= 0:
        raise ValueError("hard-negative-repeat 必须为正数")
    train_rows = [row for row in cache["candidates"] + cache["ground_truth"] if row["split"] == "train"]
    val_rows = [row for row in cache["candidates"] + cache["ground_truth"] if row["split"] == "val"]
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    train_rows, hard_negative_count = _repeat_hard_negatives(
        train_rows,
        args.hard_negative_checkpoint,
        args.hard_negative_repeat,
        device,
    )
    train_dataset = _CropDataset(train_rows, policy, training=True)
    val_dataset = _CropDataset(val_rows, policy, training=False)
    if not train_dataset or not val_dataset:
        raise ValueError("训练集和验证集都必须包含二级分类样本")
    train_loader = _balanced_loader(train_dataset, args.batch_size, args.num_workers, args.seed)
    val_loader = _loader(val_dataset, args.batch_size, args.num_workers)
    verifier = FSCVerifier(policy=policy, pretrained=True).to(device)
    backbone = (
        verifier.expert.features
        if hasattr(verifier.expert, "features")
        else nn.ModuleList(
            [verifier.expert.conv1, verifier.expert.bn1, verifier.expert.layer1, verifier.expert.layer2, verifier.expert.layer3, verifier.expert.layer4]
        )
    )
    head = verifier.expert.classifier if hasattr(verifier.expert, "classifier") else verifier.expert.fc
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": 3e-4},
            {"params": backbone.parameters(), "lr": 3e-5},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_epochs, 1))
    criterion = _FocalLoss(args.focal_gamma)
    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            for parameter in backbone.parameters():
                parameter.requires_grad = True
        verifier.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(verifier.expert(images.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        if epoch >= args.freeze_epochs:
            scheduler.step()
        verifier.eval()
        class_metrics = _classification_metrics(verifier.expert, val_loader, device)
        detection_metrics = _detection_f1(verifier, cache, device)
        report = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "classification": class_metrics,
            "detection": detection_metrics,
        }
        history.append(report)
        score = (detection_metrics["f1"], detection_metrics["recall"], class_metrics["fsc_f1"])
        if score > (best_score, -1.0, -1.0):
            best_score = detection_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in verifier.expert.state_dict().items()}
        print(
            f"[fsc-verifier] epoch={epoch + 1}/{args.epochs} loss={report['train_loss']:.4f} "
            f"val_f1={detection_metrics['f1']:.4f} val_recall={detection_metrics['recall']:.4f}"
        )

    assert best_state is not None
    result = FSCVerifier(policy=policy, pretrained=False)
    result.expert.load_state_dict(best_state)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache": str(cache_path),
        "cache_metadata": cache["metadata"],
        "seed": args.seed,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "focal_gamma": args.focal_gamma,
        "hard_negative_checkpoint": args.hard_negative_checkpoint,
        "hard_negative_count": hard_negative_count,
        "hard_negative_repeat": args.hard_negative_repeat,
        "train_counts": dict(Counter(int(row["label"]) for row in train_rows)),
        "val_counts": dict(Counter(int(row["label"]) for row in val_rows)),
        "selection": "验证集固定 argmax 端到端 FSC F1，不搜索分类阈值",
        "history": history,
    }
    checkpoint = output_dir / "fsc_two_stage_verifier.pth"
    torch.save(result.checkpoint_payload(metadata), checkpoint)
    (output_dir / "training_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 二级复核器: {checkpoint}")


if __name__ == "__main__":
    main()
