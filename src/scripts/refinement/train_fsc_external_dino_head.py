# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用独立的 DINOv2 ViT-L/14 特征训练 FSC 二级分类头。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from dataclasses import replace

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.refinement import FSCDinoHead, FSCVerifierPolicy, crop_fsc_context, crop_transform  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析外部 DINOv2 训练参数。"""
    parser = argparse.ArgumentParser(description="训练外部 DINOv2 ViT-L FSC 分类头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repo", required=True, help="本地 DINOv2 源码目录")
    parser.add_argument("--backbone-checkpoint", required=True, help="DINOv2 ViT-L/14 权重")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--context-scale", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--include-ground-truth",
        action="store_true",
        help="将 GT FSC 框作为额外正样本；默认只使用一级候选，保证验证口径一致",
    )
    return parser.parse_args()


def _load_rows(
    path: Path,
    include_ground_truth: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """读取不含测试集的 train/val 候选。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("候选缓存必须是未使用测试集的版本")
    rows = list(payload["candidates"])
    if include_ground_truth:
        rows.extend(payload["ground_truth"])
    return payload, [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "val"]


def _features(backbone: nn.Module, rows: list[dict[str, Any]], policy: FSCVerifierPolicy, device: torch.device, batch_size: int) -> tuple[Tensor, Tensor]:
    """提取 CLS 与 patch 平均池化特征。"""
    transform = crop_transform(training=False)
    feature_chunks: list[Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        crops: list[Tensor] = []
        for row in chunk:
            with Image.open(row["image"]) as image:
                crops.append(transform(crop_fsc_context(image, row["xyxy"], policy.context_scale, policy.image_size)))
        with torch.inference_mode():
            output = backbone.forward_features(torch.stack(crops).to(device))
            pooled = torch.cat((output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)), dim=1)
        feature_chunks.append(pooled.float().cpu())
        labels.extend(int(row["label"]) for row in chunk)
    return torch.cat(feature_chunks), torch.tensor(labels, dtype=torch.long)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """计算固定 argmax 的 FSC 分类指标。"""
    predicted = logits.argmax(dim=1)
    tp = ((predicted == 1) & (labels == 1)).sum().float()
    fp = ((predicted == 1) & (labels == 0)).sum().float()
    fn = ((predicted == 0) & (labels == 1)).sum().float()
    precision = float(tp / (tp + fp).clamp_min(1.0))
    recall = float(tp / (tp + fn).clamp_min(1.0))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def main() -> None:
    """训练并保存冻结外部 DINOv2 特征头。"""
    args = _parse_args()
    if args.negative_weight <= 0:
        raise ValueError("negative-weight 必须为正数")
    cache, train_rows, val_rows = _load_rows(Path(args.cache).resolve(), args.include_ground_truth)
    policy = FSCVerifierPolicy.from_mapping(cache["metadata"]["policy"])
    if args.context_scale is not None or args.image_size is not None:
        policy = replace(
            policy,
            context_scale=args.context_scale if args.context_scale is not None else policy.context_scale,
            image_size=args.image_size if args.image_size is not None else policy.image_size,
        )
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from dinov2.hub.backbones import dinov2_vitl14_reg

    backbone = dinov2_vitl14_reg(pretrained=False)
    payload = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(payload)
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = _features(backbone, train_rows, policy, device, args.batch_size)
    val_x, val_y = _features(backbone, val_rows, policy, device, args.batch_size)
    head = FSCDinoHead(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.negative_weight, 1.0], device=device))
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x.to(device)), train_y.to(device))
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metrics = _metrics(head(val_x.to(device)), val_y.to(device))
        history.append({"epoch": epoch + 1, "loss": float(loss.detach()), **metrics})
        if best_metrics is None or (metrics["f1"], metrics["recall"]) > (best_metrics["f1"], best_metrics["recall"]):
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    assert best_state is not None and best_metrics is not None
    result = FSCDinoHead(train_x.shape[1])
    result.load_state_dict(best_state)
    metadata = {
        "cache": str(Path(args.cache).resolve()),
        "repo": str(Path(args.repo).resolve()),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "policy": policy.to_dict(),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "negative_weight": args.negative_weight,
        "include_ground_truth": args.include_ground_truth,
        "selection": "验证集固定 argmax，不搜索阈值",
        "best": best_metrics,
        "history": history,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint_payload(metadata), output)
    output.with_name("external_dino_training_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 外部 DINOv2 FSC 分类头: {output}")
    print(json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
