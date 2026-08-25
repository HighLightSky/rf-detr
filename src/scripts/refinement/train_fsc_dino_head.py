# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用冻结 RF-DETR/DINOv2 特征训练 FSC 二分类头。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from rfdetr import RFDETR
from rfdetr.refinement import FSCDinoHead, FSCVerifierPolicy, crop_fsc_context, crop_transform, pool_dino_features


def _parse_args() -> argparse.Namespace:
    """解析 DINOv2 分类头训练参数。"""
    parser = argparse.ArgumentParser(description="训练冻结 RF-DETR DINOv2 FSC 分类头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--context-scale", type=float, default=None, help="覆盖二阶段上下文边长倍数")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 DINO crop 尺寸")
    parser.add_argument("--negative-weight", type=float, default=1.0, help="固定的非 FSC 损失权重")
    parser.add_argument("--pooling", choices=("avg", "avgmax"), default="avg", help="DINO 空间池化方式")
    return parser.parse_args()


def _load_cache(path: Path) -> dict[str, Any]:
    """读取无测试集候选缓存。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("DINOv2 训练需要不含测试集的候选缓存")
    return payload


def _extract(
    encoder: nn.Module,
    rows: list[dict[str, Any]],
    policy: FSCVerifierPolicy,
    device: torch.device,
    batch_size: int,
    pooling: str,
) -> tuple[Tensor, Tensor]:
    """从候选上下文 crop 提取四尺度 DINOv2 全局平均池化特征。"""
    transform = crop_transform(training=False)
    features: list[Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tensors: list[Tensor] = []
        for row in chunk:
            with Image.open(row["image"]) as image:
                crop = crop_fsc_context(image, row["xyxy"], policy.context_scale, policy.image_size)
            tensors.append(transform(crop))
        with torch.inference_mode():
            output = encoder(torch.stack(tensors).to(device))
            pooled = pool_dino_features(output, pooling)
        features.append(pooled.float().cpu())
        labels.extend(int(row["label"]) for row in chunk)
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """计算固定 argmax 的二分类指标。"""
    predicted = logits.argmax(dim=1)
    tp = ((predicted == 1) & (labels == 1)).sum().float()
    fp = ((predicted == 1) & (labels == 0)).sum().float()
    fn = ((predicted == 0) & (labels == 1)).sum().float()
    precision = float(tp / (tp + fp).clamp_min(1.0))
    recall = float(tp / (tp + fn).clamp_min(1.0))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def main() -> None:
    """训练 DINOv2 分类头并按验证集固定 argmax F1 选择权重。"""
    args = _parse_args()
    cache = _load_cache(Path(args.cache).resolve())
    policy = FSCVerifierPolicy.from_mapping(cache["metadata"]["policy"])
    if args.context_scale is not None or args.image_size is not None:
        policy = replace(
            policy,
            context_scale=args.context_scale if args.context_scale is not None else policy.context_scale,
            image_size=args.image_size if args.image_size is not None else policy.image_size,
        )
    if args.negative_weight <= 0:
        raise ValueError("negative-weight 必须为正数")
    train_rows = [row for row in cache["candidates"] + cache["ground_truth"] if row["split"] == "train"]
    val_rows = [row for row in cache["candidates"] + cache["ground_truth"] if row["split"] == "val"]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    detector = RFDETR.from_checkpoint(args.detector)
    detector_core = detector.model.model
    encoder = detector_core.backbone[0].encoder.to(device).eval()
    train_x, train_y = _extract(encoder, train_rows, policy, device, args.batch_size, args.pooling)
    val_x, val_y = _extract(encoder, val_rows, policy, device, args.batch_size, args.pooling)
    head = FSCDinoHead(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([args.negative_weight, 1.0], device=device, dtype=torch.float32)
    )
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x.to(device))
        loss = criterion(logits, train_y.to(device))
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metrics = _metrics(head(val_x.to(device)), val_y.to(device))
        report = {"epoch": epoch + 1, "loss": float(loss.detach()), **metrics}
        history.append(report)
        if best_metrics is None or (metrics["f1"], metrics["recall"]) > (best_metrics["f1"], best_metrics["recall"]):
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    assert best_state is not None and best_metrics is not None
    result = FSCDinoHead(train_x.shape[1])
    result.load_state_dict(best_state)
    metadata = {
        "cache": str(Path(args.cache).resolve()),
        "detector": str(Path(args.detector).resolve()),
        "policy": policy.to_dict(),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "selection": "验证集固定 argmax，不搜索阈值",
        "negative_weight": args.negative_weight,
        "pooling": args.pooling,
        "best": best_metrics,
        "history": history,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint_payload(metadata), output)
    output.with_name("dino_training_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] DINOv2 FSC 分类头: {output}")
    print(json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
