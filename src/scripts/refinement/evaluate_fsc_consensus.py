# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""在 train/val 候选上评估二级共识规则，不读取测试集。"""

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

from rfdetr.refinement import FSCDinoHead, FSCVerifierPolicy, crop_fsc_context, crop_transform, fsc_consensus_decision  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="评估 FSC 共识二阶段头")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--single-head", required=True)
    parser.add_argument("--rotation-head", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取验证候选并拒绝测试 split。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1" or payload.get("metadata", {}).get("test_split_used"):
        raise ValueError("缓存格式错误或包含测试集")
    return payload, [row for row in payload["candidates"] if row["split"] == "val"]


def _features(backbone: nn.Module, rows: list[dict[str, Any]], policy: FSCVerifierPolicy, device: torch.device, batch_size: int) -> tuple[Tensor, Tensor]:
    """提取验证候选的单视图和四方向旋转特征。"""
    transform = crop_transform(False)
    single_parts: list[Tensor] = []
    rotation_parts: list[Tensor] = []
    labels: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        crops: list[Tensor] = []
        for row in chunk:
            with Image.open(row["image"]) as image:
                crops.append(transform(crop_fsc_context(image, row["xyxy"], policy.context_scale, policy.image_size)))
        batch = torch.stack(crops).to(device)
        with torch.inference_mode():
            def _extract(images: Tensor) -> Tensor:
                output = backbone.forward_features(images)
                return torch.cat((output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)), dim=1)

            single_parts.append(_extract(batch).cpu())
            rotation_parts.append(torch.stack([_extract(transforms_functional.rotate(batch, angle)) for angle in (0, 90, 180, 270)]).mean(0).cpu())
        labels.extend(int(row["label"]) for row in chunk)
    return torch.cat(single_parts), torch.cat(rotation_parts), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    """输出固定 argmax 共识规则的验证指标。"""
    args = _parse_args()
    cache, rows = _load(Path(args.cache).resolve())
    policy = FSCVerifierPolicy.from_mapping(cache["metadata"]["policy"])
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from dinov2.hub.backbones import dinov2_vitl14_reg

    backbone = dinov2_vitl14_reg(pretrained=False)
    backbone.load_state_dict(torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True))
    backbone.to(device).eval()
    single_payload = torch.load(args.single_head, map_location="cpu", weights_only=False)
    rotation_payload = torch.load(args.rotation_head, map_location="cpu", weights_only=False)
    single = FSCDinoHead(int(single_payload["feature_dim"])).to(device)
    rotation = FSCDinoHead(int(rotation_payload["feature_dim"])).to(device)
    single.load_state_dict(single_payload["state_dict"]); rotation.load_state_dict(rotation_payload["state_dict"])
    single.eval(); rotation.eval()
    single_features, rotation_features, labels = _features(backbone, rows, policy, device, args.batch_size)
    with torch.inference_mode():
        prediction = fsc_consensus_decision(
            single(single_features.to(device)),
            rotation(rotation_features.to(device)),
        ).cpu() == 1
    tp = int((prediction & (labels == 1)).sum())
    fp = int((prediction & (labels == 0)).sum())
    fn = int((~prediction & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    print(json.dumps({"split": "val", "test_split_used": False, "candidate_count": len(rows), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
