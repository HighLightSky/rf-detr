# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义方向投影 f_sem（FSemProjection）与产物存取、对齐校验的单元测试。

不依赖 GPU 与网络，验证：
- FSemProjection 前向输出形状。
- save/load_fsem_artifacts 往返一致。
- evaluate_alignment 对"特征与语义方向对齐"与"随机方向"能区分。
"""

from __future__ import annotations

import torch

from rfdetr.sscl.fsem import (
    FSemProjection,
    evaluate_alignment,
    load_fsem_artifacts,
    save_fsem_artifacts,
)


def test_fsem_projection_shape() -> None:
    """前向输出形状正确：``[N, 768] -> [N, 256]``。"""
    proj = FSemProjection(text_dim=768, hidden_dim=512, out_dim=256)
    x = torch.randn(4, 768)
    out = proj(x)
    assert out.shape == (4, 256)


def test_fsem_artifacts_round_trip(tmp_path) -> None:
    """Save/load 往返后 S 矩阵与权重一致。"""
    path = tmp_path / "fsem.pt"
    proj = FSemProjection(text_dim=768, hidden_dim=512, out_dim=256)
    s = torch.randn(25, 256)
    s = torch.nn.functional.normalize(s, dim=-1)
    save_fsem_artifacts(path, s, proj.state_dict(), {"class_names": ["a"] * 25, "epoch": 3})

    data = load_fsem_artifacts(path)
    assert torch.allclose(data["S"], s)
    assert data["meta"]["epoch"] == 3
    # 权重可恢复
    proj2 = FSemProjection(text_dim=768, hidden_dim=512, out_dim=256)
    proj2.load_state_dict(data["fsem_state_dict"])


def test_evaluate_alignment_distinguishes_aligned_from_random() -> None:
    """对齐校验：特征对齐语义方向时 gap 显著为正，随机特征时接近 0。"""
    torch.manual_seed(0)
    d = 16
    classes = [0, 1, 2, 3, 4]
    s = torch.nn.functional.normalize(torch.randn(len(classes), d), dim=-1)

    # 对齐特征：特征 = 语义方向 + 小噪声
    aligned = {}
    for c in classes:
        base = s[c].unsqueeze(0)
        noise = 0.1 * torch.randn(8, d)
        aligned[c] = torch.nn.functional.normalize(base + noise, dim=-1)
    report = evaluate_alignment(aligned, s)
    assert report["mean_align"] > 0.7
    assert report["mean_gap"] > 0.2

    # 随机特征：gap 接近 0
    random = {c: torch.nn.functional.normalize(torch.randn(8, d), dim=-1) for c in classes}
    report_random = evaluate_alignment(random, s)
    assert report_random["mean_gap"] < 0.2
