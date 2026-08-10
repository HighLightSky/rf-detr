# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义分类头 SemanticResidual 的单元测试。

不依赖 GPU 与网络，验证：
- forward 输出形状、mask/alpha 两个开关的独立开关行为。
- α 在 forward 内 clamp 到 [0, alpha_max]。
- build() 从离线产物装配：α 初始化（novel 类用 novel_alpha_init）、θ 冻结类生效。
- attach_from_checkpoint 从模型 state_dict 重建语义头（评估路径）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rfdetr.sscl.channel_stats import save_channel_stats
from rfdetr.sscl.fsem import save_fsem_artifacts
from rfdetr.sscl.semantic_head import SemanticResidual, attach_from_checkpoint

_C = 5
_D = 16


def _build_cfg(**overrides: object) -> SimpleNamespace:
    """构造语义头装配所需的配置命名空间（默认值 = 完整语义头配置）。"""
    cfg = {
        "semantic_fsem_path": "/tmp/_fake_fsem.pt",
        "semantic_channel_stats_path": "/tmp/_fake_stats.pt",
        "semantic_mask_enabled": True,
        "semantic_alpha_enabled": True,
        "semantic_alpha_learnable": True,
        "semantic_alpha_init": 0.1,
        "semantic_novel_alpha_init": 0.5,
        "semantic_alpha_max": 2.0,
        "semantic_mask_tau": 1.0,
        "semantic_theta_init": 3.0,
        "semantic_novel_classes": [0, 1],
        "semantic_frozen_threshold_classes": [0, 1],
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _make_artifacts(tmp_path) -> tuple[str, str]:
    """生成最小 fsem 与通道统计产物文件，返回路径。"""
    torch.manual_seed(0)
    s = torch.nn.functional.normalize(torch.randn(_C, _D), dim=-1)
    fsem_path = tmp_path / "fsem.pt"
    save_fsem_artifacts(fsem_path, s, {"linear1.weight": torch.zeros(4, 4)}, {"class_names": []})

    rank = torch.arange(1, _D + 1, dtype=torch.float32).unsqueeze(0).repeat(_C, 1)
    score = torch.randn(_C, _D)
    idf = torch.randn(_D)
    tf = torch.rand(_C, _D)
    stats_path = tmp_path / "stats.pt"
    save_channel_stats(stats_path, SimpleNamespace(tf=tf, rank=rank, score=score, idf=idf, counts={}, meta={}))
    return str(fsem_path), str(stats_path)


def test_forward_shape_and_switches(tmp_path) -> None:
    """Forward 输出形状正确，且 mask/alpha 开关各自生效。"""
    fsem_path, stats_path = _make_artifacts(tmp_path)
    cfg = _build_cfg(semantic_fsem_path=fsem_path, semantic_channel_stats_path=stats_path)
    head = SemanticResidual.build(cfg, _C, _D)
    # 用随机 S/rank 覆盖（不影响形状/开关测试）
    head.S.copy_(torch.nn.functional.normalize(torch.randn(_C, _D), dim=-1))
    head.rank.copy_(torch.arange(1, _D + 1, dtype=torch.float32).unsqueeze(0).repeat(_C, 1))

    hs = torch.randn(2, 8, _D)
    w = torch.randn(_C, _D)
    delta, stats = head(hs, w)
    assert delta.shape == (2, 8, _C)
    assert stats["sem_delta"].shape == (2, 8, _C)
    assert stats["M"].shape == (_C, _D)

    # mask 关闭 → 掩码增量为 0（delta == sem_delta）
    head.mask_enabled = False
    delta_no_mask, _ = head(hs, w)
    assert torch.allclose(delta_no_mask, head.last_stats["sem_delta"])

    # alpha 关闭 → 语义增量为 0（delta == mask_delta）
    head.alpha_enabled = False
    delta_no_alpha, _ = head(hs, w)
    assert torch.equal(delta_no_alpha, torch.zeros_like(delta_no_alpha))


def test_alpha_clamped_in_forward() -> None:
    """Α 在 forward 内被 clamp 到 [0, alpha_max]，且不修改原始参数。"""
    head = SemanticResidual(_C, _D, alpha_max=2.0, mask_tau=1.0)
    head.S.copy_(torch.nn.functional.normalize(torch.randn(_C, _D), dim=-1))
    with torch.no_grad():
        head.alpha.fill_(5.0)  # 超过上限
    hs = torch.randn(1, 4, _D)
    _, stats = head(hs, torch.randn(_C, _D))
    assert bool((stats["alpha"] <= 2.0).all())
    # 原始参数未被修改
    assert float(head.alpha.max()) == 5.0


def test_build_init_and_freeze(tmp_path) -> None:
    """Build() 装配：S/rank 载入、novel 类 α 更大、novel 类 θ 冻结。"""
    fsem_path, stats_path = _make_artifacts(tmp_path)
    cfg = _build_cfg(semantic_fsem_path=fsem_path, semantic_channel_stats_path=stats_path)
    head = SemanticResidual.build(cfg, _C, _D)
    # S 与 rank 已载入（形状正确、非全零）
    assert head.S.shape == (_C, _D)
    assert head.rank.shape == (_C, _D)
    assert bool((head.rank != 0).any())
    # novel 类 (0,1) α = 0.5，其他类 α = 0.1
    assert float(head.alpha[0]) == pytest.approx(0.5)
    assert float(head.alpha[2]) == pytest.approx(0.1)
    # novel 类 (0,1) θ 冻结（梯度置零 mask 标记），其他类可学习
    assert bool(head._frozen_theta_mask[0])
    assert bool(head._frozen_theta_mask[1])
    assert not bool(head._frozen_theta_mask[2])

    # 验证冻结机制：走一次真实 backward，冻结类 θ 梯度应为 0，非冻结类非 0
    head.set_frozen_theta_classes([0, 1])
    hs = torch.randn(1, 4, _D)
    delta, _ = head(hs, torch.randn(_C, _D))
    delta.sum().backward()
    assert head.theta.grad is not None
    assert float(head.theta.grad[0]) == 0.0
    assert float(head.theta.grad[1]) == 0.0
    assert float(head.theta.grad[2]) != 0.0


def test_attach_from_checkpoint_round_trip(tmp_path) -> None:
    """attach_from_checkpoint：从 state_dict 重建语义头并加载子状态。"""
    head = SemanticResidual(_C, _D)
    head.S.copy_(torch.nn.functional.normalize(torch.randn(_C, _D), dim=-1))
    sub_state = head.state_dict()
    full_state = {f"semantic_residual.{k}": v for k, v in sub_state.items()}

    # 模拟一个带语义头键的模型 state_dict
    class Stub:
        pass

    model = Stub()
    assert attach_from_checkpoint(model, full_state) is True
    assert model.semantic_residual is not None
    # 重建后的 S 与原始一致（buffer 无损）
    assert torch.allclose(model.semantic_residual.S, head.S)
    # 无语义头键时返回 False 且不挂载
    model2 = Stub()
    assert attach_from_checkpoint(model2, {"class_embed.weight": torch.zeros(1, 1)}) is False
    assert not hasattr(model2, "semantic_residual")
