# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练旋转不变的外部 DINOv2 FSC 二级头。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as transforms_functional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.refinement import FSCDinoHead, FSCVerifierPolicy, crop_fsc_context, crop_transform  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析旋转增强训练参数。"""
    parser = argparse.ArgumentParser(description="训练旋转不变 DINOv2 FSC 头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    return parser.parse_args()


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """读取不含测试集的候选缓存。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("缓存格式错误或包含测试集")
    rows = payload["candidates"]
    return payload, [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def _extract(backbone: nn.Module, rows: list[dict[str, Any]], policy: FSCVerifierPolicy, device: torch.device, batch_size: int) -> tuple[Tensor, Tensor]:
    """提取四方向旋转的平均 DINO 特征。"""
    transform = crop_transform(False)
    features: list[Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        crops: list[Tensor] = []
        for row in chunk:
            with Image.open(row["image"]) as image:
                crops.append(transform(crop_fsc_context(image, row["xyxy"], policy.context_scale, policy.image_size)))
        rotated = torch.stack([transforms_functional.rotate(torch.stack(crops), angle) for angle in (0, 90, 180, 270)])
        with torch.inference_mode():
            batch_features: list[Tensor] = []
            for view in rotated:
                output = backbone.forward_features(view.to(device))
                batch_features.append(torch.cat((output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)), dim=1))
            features.append(torch.stack(batch_features).mean(dim=0).float().cpu())
        labels.extend(int(row["label"]) for row in chunk)
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """计算固定 argmax 指标。"""
    prediction = logits.argmax(1)
    tp = ((prediction == 1) & (labels == 1)).sum().float()
    fp = ((prediction == 1) & (labels == 0)).sum().float()
    fn = ((prediction == 0) & (labels == 1)).sum().float()
    precision = float(tp / (tp + fp).clamp_min(1))
    recall = float(tp / (tp + fn).clamp_min(1))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def main() -> None:
    """训练并保存旋转不变分类头。"""
    args = _parse_args()
    cache, train_rows, val_rows = _load(Path(args.cache).resolve())
    if args.negative_weight <= 0:
        raise ValueError("negative-weight 必须为正数")
    policy = FSCVerifierPolicy.from_mapping(cache["metadata"]["policy"])
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from dinov2.hub.backbones import dinov2_vitl14_reg

    backbone = dinov2_vitl14_reg(pretrained=False)
    backbone.load_state_dict(torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True))
    backbone.to(device).eval()
    train_x, train_y = _extract(backbone, train_rows, policy, device, args.batch_size)
    val_x, val_y = _extract(backbone, val_rows, policy, device, args.batch_size)
    head = FSCDinoHead(2048).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.negative_weight, 1.0], device=device))
    best: dict[str, float] | None = None
    state: dict[str, Tensor] | None = None
    for _ in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x.to(device)), train_y.to(device))
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metrics = _metrics(head(val_x.to(device)), val_y.to(device))
        if best is None or (metrics["f1"], metrics["precision"]) > (best["f1"], best["precision"]):
            best = metrics
            state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best is not None and state is not None
    result = FSCDinoHead(2048)
    result.load_state_dict(state)
    metadata = {
        "repo": str(Path(args.repo).resolve()),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "policy": policy.to_dict(),
        "tta_rotations": [0, 90, 180, 270],
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "negative_weight": args.negative_weight,
        "selection": "验证集固定 argmax，不搜索阈值",
        "best": best,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint_payload(metadata), output)
    print(json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
