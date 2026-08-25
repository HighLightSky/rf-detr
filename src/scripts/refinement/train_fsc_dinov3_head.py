# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用冻结 DINOv3 特征训练 FSC 二级复核头。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.refinement import crop_fsc_context  # noqa: E402

FORMAT = "shwx-fsc-dinov3-head-v1"


class FSCDinoV3Head(nn.Module):
    """在冻结 DINOv3 特征上判别 FSC 与非 FSC。"""

    def __init__(self, feature_dim: int) -> None:
        """初始化二分类头。"""
        super().__init__()
        self.feature_dim = feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        """输出非 FSC、FSC logits。"""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features 必须为 [N, {self.feature_dim}]")
        return self.head(features)


def _parse_args() -> argparse.Namespace:
    """解析 DINOv3 训练参数。"""
    parser = argparse.ArgumentParser(description="训练 DINOv3 FSC 二级复核头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--negative-weight", type=float, default=4.0)
    return parser.parse_args()


def _load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """从缓存 train 候选构建图像名哈希内部留出集。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("候选缓存格式错误或已使用测试集")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in payload["candidates"]:
        if row["split"] != "train":
            continue
        digest = hashlib.sha256(Path(row["image"]).name.encode("utf-8")).digest()
        (holdout if int.from_bytes(digest[:8], "big") % 5 == 0 else train).append(row)
    return payload, train, holdout


def _features(
    model: nn.Module,
    transform: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """批量提取 DINOv3 全局图像描述子。"""
    features: list[Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        crops: list[Tensor] = []
        for row in chunk:
            with Image.open(row["image"]) as image:
                crops.append(transform(crop_fsc_context(image, row["xyxy"], context_scale=2.0, output_size=224)))
        with torch.inference_mode():
            features.append(model(torch.stack(crops).to(device)).float().cpu())
        labels.extend(int(row["label"]) for row in chunk)
        if start % 512 == 0:
            print(f"[特征] {start}/{len(rows)}")
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def _metric(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """计算固定 argmax 的 precision、recall 和 F1。"""
    pred = logits.argmax(1)
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def main() -> None:
    """只从当前数据集训练候选二级复核头。"""
    args = _parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.negative_weight <= 0:
        raise ValueError("batch-size、epochs 和 negative-weight 必须为正数")
    import timm
    from timm.data import create_transform, resolve_model_data_config

    cache, train_rows, holdout_rows = _load_rows(Path(args.cache).resolve())
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    train_x, train_y = _features(backbone, transform, train_rows, device, args.batch_size)
    holdout_x, holdout_y = _features(backbone, transform, holdout_rows, device, args.batch_size)
    head = FSCDinoV3Head(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.negative_weight, 1.0], device=device))
    best, state = None, None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x.to(device)), train_y.to(device))
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metric = _metric(head(holdout_x.to(device)), holdout_y.to(device))
        history.append({"epoch": epoch + 1, "loss": float(loss.detach()), **metric})
        if best is None or (metric["f1"], metric["precision"], metric["recall"]) > (
            best["f1"],
            best["precision"],
            best["recall"],
        ):
            best, state = metric, {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best is not None and state is not None
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache": str(Path(args.cache).resolve()),
        "model_name": args.model_name,
        "source_split": "cache train only, image-name SHA256 internal holdout",
        "test_used_for_training": False,
        "context_scale": 2.0,
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "negative_weight": args.negative_weight,
        "selection": "仅内部留出集固定 argmax，不搜索阈值",
        "best": best,
        "history": history,
    }
    torch.save({"format": FORMAT, "feature_dim": head.feature_dim, "state_dict": state, "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
