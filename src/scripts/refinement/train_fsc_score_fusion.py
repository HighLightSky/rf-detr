# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练冻结 FSC 视觉复核器与一级分数的学习型融合头。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from rfdetr.refinement import FSCScoreFusion, FSCVerifier


def _parse_args() -> argparse.Namespace:
    """解析融合训练参数。"""
    parser = argparse.ArgumentParser(description="训练 FSC 二级视觉与 detector 分数融合头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    return parser.parse_args()


def _load_cache(path: Path) -> dict[str, Any]:
    """读取候选缓存并确认未包含测试 split。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("融合训练需要不含测试集的 FSC 候选缓存")
    return payload


def _features(verifier: FSCVerifier, rows: list[dict[str, Any]], device: torch.device) -> tuple[Tensor, Tensor]:
    """批量提取候选 crop 的视觉概率和一级分数。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["image_id"], []).append(row)
    feature_parts: list[Tensor] = []
    labels: list[int] = []
    for image_rows in grouped.values():
        with Image.open(image_rows[0]["image"]) as image:
            probabilities = verifier.predict_probabilities(image, [row["xyxy"] for row in image_rows]).detach()
        scores = torch.tensor([float(row["score"]) for row in image_rows], dtype=torch.float32)
        feature_parts.append(FSCScoreFusion.features(probabilities.cpu(), scores))
        labels.extend(int(row["label"]) for row in image_rows)
    return torch.cat(feature_parts).to(device), torch.tensor(labels, dtype=torch.long, device=device)


def _metrics(prediction: Tensor, labels: Tensor) -> dict[str, float]:
    """计算固定 argmax 的分类指标。"""
    prediction, labels = prediction.detach(), labels.detach()
    tp = ((prediction == 1) & (labels == 1)).sum().float()
    fp = ((prediction == 1) & (labels == 0)).sum().float()
    fn = ((prediction == 0) & (labels == 1)).sum().float()
    precision = float(tp / (tp + fp).clamp_min(1.0))
    recall = float(tp / (tp + fn).clamp_min(1.0))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def main() -> None:
    """训练并保存融合头。"""
    args = _parse_args()
    cache_path = Path(args.cache).resolve()
    cache = _load_cache(cache_path)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    verifier = FSCVerifier.from_checkpoint(args.verifier, device=device)
    train_rows = [row for row in cache["candidates"] if row["split"] == "train"]
    val_rows = [row for row in cache["candidates"] if row["split"] == "val"]
    train_x, train_y = _features(verifier, train_rows, device)
    val_x, val_y = _features(verifier, val_rows, device)
    fusion = FSCScoreFusion().to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        fusion.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(fusion.head(train_x), train_y)
        loss.backward()
        optimizer.step()
        fusion.eval()
        metrics = _metrics(fusion.predict(val_x[:, :2], torch.sigmoid(val_x[:, 2])), val_y)
        report = {"epoch": epoch + 1, "loss": float(loss.detach()), **metrics}
        history.append(report)
        if best_metrics is None or (metrics["f1"], metrics["recall"]) > (best_metrics["f1"], best_metrics["recall"]):
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in fusion.state_dict().items()}
    assert best_state is not None and best_metrics is not None
    result = FSCScoreFusion()
    result.load_state_dict(best_state)
    metadata = {
        "cache": str(cache_path),
        "verifier": str(Path(args.verifier).resolve()),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "selection": "验证集固定 argmax，不搜索阈值",
        "best": best_metrics,
        "history": history,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint_payload(metadata), output)
    output.with_name("fusion_training_report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 学习型融合头: {output}")
    print(json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
