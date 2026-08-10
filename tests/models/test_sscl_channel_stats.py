# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""通道 TF-IDF 统计（channel_stats）的单元测试。

不依赖 GPU，验证：
- _exact_2means_activation_mask 的确定性与正确性（均值高的簇为激活簇）。
- compute_channel_tfidf 结果确定性、每类 rank 是 1..d 的置换。
- build_mask_from_rank 随 θ 单调（θ 越大保留通道越多）。
- save/load_channel_stats 往返一致。
"""

from __future__ import annotations

import torch

from rfdetr.sscl.channel_stats import (
    _batch_2means_activation_mask,
    build_mask_from_rank,
    compute_channel_tfidf,
    load_channel_stats,
    save_channel_stats,
)


def test_2means_mask_marks_higher_cluster() -> None:
    """批量精确 2-means：均值较高的簇被标记为激活簇，结果确定。"""
    values = torch.tensor([[0.1, 0.2, 0.9, 1.0, 0.3, 0.8]])
    mask1 = _batch_2means_activation_mask(values)[0]
    mask2 = _batch_2means_activation_mask(values)[0]
    assert torch.equal(mask1, mask2)  # 确定性
    # 高值通道应被激活
    assert mask1[2].item() and mask1[3].item() and mask1[5].item()
    assert not mask1[0].item() and not mask1[1].item() and not mask1[4].item()


def test_channel_tfidf_rank_is_permutation_and_deterministic() -> None:
    """每类 rank 是 1..d 的置换，且重复计算得到相同结果。"""
    torch.manual_seed(0)
    d = 16
    s = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    features = {c: torch.randn(20, d) for c in range(4)}
    stats1 = compute_channel_tfidf(features, s)
    stats2 = compute_channel_tfidf(features, s)
    assert torch.equal(stats1.rank, stats2.rank)  # 确定性
    assert stats1.rank.shape == (4, d)
    for c in range(4):
        # rank 是 1..d 的置换（每个值恰好出现一次）
        assert torch.sort(stats1.rank[c])[0].tolist() == list(range(1, d + 1))
    # score 非负（TF≥0, IDF 可能为正）
    assert bool((stats1.score >= 0).all())


def test_build_mask_monotonic_in_theta() -> None:
    """Θ 越大，掩码值越大（保留更多通道）。"""
    rank = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    m_low = build_mask_from_rank(rank, torch.tensor([4.0, 8.0]), mask_tau=1.0)
    m_high = build_mask_from_rank(rank, torch.tensor([8.0, 12.0]), mask_tau=1.0)
    assert bool((m_high > m_low).all())
    assert m_low.shape == rank.shape


def test_channel_stats_round_trip(tmp_path) -> None:
    """save/load_channel_stats 往返一致。"""
    torch.manual_seed(1)
    d = 16
    s = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    stats = compute_channel_tfidf({c: torch.randn(10, d) for c in range(4)}, s)
    path = tmp_path / "channel_stats.pt"
    save_channel_stats(path, stats)
    loaded = load_channel_stats(path)
    assert torch.equal(loaded.rank, stats.rank)
    assert torch.allclose(loaded.tf, stats.tf)
    assert loaded.counts == stats.counts
