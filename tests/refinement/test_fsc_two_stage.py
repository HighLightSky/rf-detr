# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""发射车两阶段复核器的核心测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image
from torch import nn

from rfdetr.refinement.fsc_two_stage import (
    FSCVerifier,
    FSCVerifierPolicy,
    crop_fsc_context,
    fsc_consensus_decision,
    label_fsc_candidate,
    pool_dino_features,
)


class _FixedExpert(nn.Module):
    """输出固定二分类 logits 的测试网络。"""

    def __init__(self, logits: tuple[float, float]) -> None:
        """初始化固定输出。"""
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """按输入 batch 大小返回 logits。"""
        return self.logits.unsqueeze(0).expand(images.shape[0], -1)


def test_crop_fsc_context_returns_square_rgb_for_degenerate_box() -> None:
    """退化框也必须生成固定大小 RGB 上下文图。"""
    crop = crop_fsc_context(np.zeros((5, 7), dtype=np.uint8), (2.0, 2.0, 2.0, 2.0), output_size=32)

    assert crop.mode == "RGB"
    assert crop.size == (32, 32)


def test_label_fsc_candidate_uses_fsc_iou_threshold() -> None:
    """只有达到车辆评测 IoU 门槛的 FSC 候选才是正样本。"""
    gt = [(24, (0.0, 0.0, 10.0, 10.0)), (8, (20.0, 20.0, 30.0, 30.0))]

    assert label_fsc_candidate((0.0, 0.0, 10.0, 10.0), gt) == 1
    assert label_fsc_candidate((20.0, 20.0, 30.0, 30.0), gt) == 0
    assert label_fsc_candidate((7.0, 7.0, 17.0, 17.0), gt) == 0


def test_verifier_rejects_non_fsc_without_score_threshold_tuning() -> None:
    """二级复核器应按类别 argmax 拒绝非 FSC，而非重设 detector 分数阈值。"""
    verifier = FSCVerifier(policy=FSCVerifierPolicy(), pretrained=False)
    verifier.expert = _FixedExpert((4.0, 0.0))
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    boxes = np.asarray([[1, 1, 8, 8], [2, 2, 9, 9]], dtype=np.float32)
    scores = np.asarray([0.91, 0.70], dtype=np.float32)
    class_ids = np.asarray([24, 5], dtype=np.int64)

    out_boxes, out_scores, out_classes, audit = verifier.refine_image(image, boxes, scores, class_ids)

    assert out_boxes.shape == (1, 4)
    assert out_scores.tolist() == pytest.approx([0.70])
    assert out_classes.tolist() == [5]
    assert audit == {"routed_fsc": 1, "kept": 1, "rejected_non_fsc": 1, "unchanged": 1}


def test_verifier_checkpoint_roundtrip_preserves_candidate_floor(tmp_path: Path) -> None:
    """保存后的复核器须保留固定候选地板和元数据。"""
    verifier = FSCVerifier(policy=FSCVerifierPolicy(candidate_floor=0.05), pretrained=False)
    path = tmp_path / "verifier.pth"
    torch.save(verifier.checkpoint_payload({"dataset": "SHWX"}), path)

    restored = FSCVerifier.from_checkpoint(path)

    assert restored.policy.candidate_floor == 0.05
    assert restored.checkpoint_metadata["dataset"] == "SHWX"


def test_pool_dino_features_supports_spatial_average_and_maximum() -> None:
    """DINO 多尺度池化必须保持批次维和可预测的特征宽度。"""
    outputs = [torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)]

    pooled = pool_dino_features(outputs, "avgmax")

    assert pooled.shape == (2, 6)
    assert pooled[:, :3].shape == pooled[:, 3:].shape
    assert torch.all(pooled[:, 3:] >= pooled[:, :3])


def test_fsc_consensus_requires_both_fixed_argmax_decisions() -> None:
    """共识规则只能保留两个独立头均判为 FSC 的候选。"""
    single = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
    rotation = torch.tensor([[0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])

    decision = fsc_consensus_decision(single, rotation)

    assert decision.tolist() == [1, 0, 0]
