# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ProtoGuidance（多模态原型引导）模块单元测试。

不依赖 GPU 与网络，验证：
- position_score 输出形状与 target_classes 限定。
- 近恒等初始化：lambda=0 时 select 分数恒等于线性分数；gate bias 初始
  时内容扰动有界（相对范数 < 3%）。
- 梯度流到全部可学习参数（含 fusion 投影与融合权重）。
- 子开关短路恒等：content_enabled=False 时 tgt 原样通过。
- warmup 插值正确。
- artifacts 保存/加载往返；gated 融合构建抛 NotImplementedError。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.sscl.proto_guidance.artifacts import load_proto_artifacts, save_proto_artifacts
from rfdetr.sscl.proto_guidance.fusion import GatedProtoFusion, build_fusion
from rfdetr.sscl.proto_guidance.guidance import ProtoGuidance

# 测试维度：B=2、N=64 token、d=16、C=5 前景类、M=4 槽位
_B, _N, _D, _C, _M = 2, 64, 16, 5, 4


def _make_artifacts(
    num_classes: int = _C, num_slots: int = _M, hidden_dim: int = _D, text_dim: int = 8
) -> dict[str, object]:
    """构造随机离线产物字典（与 artifacts 格式一致）。

    Returns:
        ``{"visual_prototypes", "valid_slots", "text_prototypes", "class_names", "meta"}``。
    """
    generator = torch.Generator().manual_seed(0)
    visual = torch.randn(num_classes, num_slots, hidden_dim, generator=generator)
    valid = torch.ones(num_classes, num_slots, dtype=torch.bool)
    text = torch.randn(num_classes, text_dim, generator=generator)
    return {
        "visual_prototypes": visual,
        "valid_slots": valid,
        "text_prototypes": text,
        "class_names": [f"c{c}" for c in range(num_classes)],
        "meta": {"dataset": "test", "num_classes": num_classes, "hidden_dim": hidden_dim},
    }


def _make_module(**overrides: object) -> ProtoGuidance:
    """构造测试用模块（随机初始化，不加载产物）。

    Args:
        **overrides: 覆盖 ProtoGuidance 构造参数。

    Returns:
        ProtoGuidance 实例。
    """
    kwargs: dict[str, object] = {
        "num_classes": _C,
        "hidden_dim": _D,
        "text_dim": 8,
        "num_slots": _M,
        "warmup_epochs": 2.0,
    }
    kwargs.update(overrides)
    return ProtoGuidance(**kwargs)  # type: ignore[arg-type]


def _init_from_artifacts(module: ProtoGuidance, data: dict[str, object]) -> ProtoGuidance:
    """把产物 copy 进模块 buffer（模拟 build 的加载过程）。"""
    module.visual_bank.prototypes.copy_(data["visual_prototypes"])  # type: ignore[arg-type]
    module.visual_bank.slot_valid_mask.copy_(data["valid_slots"])  # type: ignore[arg-type]
    module.P_t_clip.copy_(data["text_prototypes"])  # type: ignore[arg-type]
    return module


class TestPositionScore:
    def test_shapes(self) -> None:
        """position_score 输出形状与类别数一致。"""
        module = _make_module()
        mem = torch.randn(_B, _N, _D)
        proto_logits, proto_score, selected_class = module.position_score(mem)
        assert proto_logits.shape == (_B, _N, _C)
        assert proto_score.shape == (_B, _N)
        assert selected_class.shape == (_B, _N)
        assert proto_logits.dtype == mem.dtype

    def test_target_classes_limits_score(self) -> None:
        """target_classes 限定位置证据来源，且分数保持有限。"""
        module = _make_module(target_classes=[])  # 全类
        mem = torch.randn(_B, _N, _D)
        _, proto_score_all, _ = module.position_score(mem)
        assert proto_score_all.shape == (_B, _N)
        # 限定 [0,1]：临时改 target_classes（同一投影，结果可比）
        module.target_classes = [0, 1]
        _, proto_score_subset, _ = module.position_score(mem)
        assert torch.isfinite(proto_score_subset).all()
        assert not torch.allclose(proto_score_subset, proto_score_all)

    def test_temperature_uses_inverse_scaling(self) -> None:
        """更小温度应放大分类 logits，但不改变位置 margin 和预测类别。"""
        module = _init_from_artifacts(_make_module(tau_p=0.1), _make_artifacts())
        warmer = _init_from_artifacts(_make_module(tau_p=0.2), _make_artifacts())
        warmer.load_state_dict(module.state_dict(), strict=False)
        memory = torch.randn(_B, _N, _D)

        logits_cold, score_cold, class_cold = module.position_score(memory)
        logits_warm, score_warm, class_warm = warmer.position_score(memory)

        assert torch.allclose(logits_cold, logits_warm * 2.0, atol=1e-5)
        assert torch.allclose(score_cold, score_warm, atol=1e-6)
        assert torch.equal(class_cold, class_warm)


class TestNearIdentity:
    def test_lambda_zero_is_identity_score(self) -> None:
        """lambda=0（warmup 起点）时 select 分数 = 线性分数。"""
        module = _make_module(lambda_pos_init=0.0, lambda_pos_max=0.0)
        linear_score = torch.randn(_B, _N)
        _, proto_score, _ = module.position_score(torch.randn(_B, _N, _D))
        select = linear_score + module.lambda_pos_effective() * proto_score
        assert torch.allclose(select, linear_score, atol=1e-6)

    def test_content_disabled_passthrough(self) -> None:
        """content_enabled=False 时 enhance_content 原样返回（恒等短路）。"""
        module = _make_module(content_enabled=False)
        tgt = torch.randn(_B, 8, _D)
        selected = torch.randint(0, _C, (_B, 8))
        out = module.enhance_content(tgt, selected)
        assert torch.equal(out, tgt)

    def test_content_perturbation_bounded(self) -> None:
        """gate bias 初始 logit(0.05) 时内容扰动有界（相对范数 < 3%）。"""
        module = _make_module(content_enabled=True, gamma_content_init=0.05, gamma_content_max=0.05)
        tgt = torch.randn(_B, 8, _D)
        selected = torch.randint(0, _C, (_B, 8))
        out = module.enhance_content(tgt, selected)
        rel = (out - tgt).norm(dim=-1).mean() / (tgt.norm(dim=-1).mean() + 1e-6)
        assert float(rel) < 0.03

    def test_content_confidence_zero_is_identity(self) -> None:
        """原型分类无置信度时，即使内容分支开启也不得注入原型方向。"""
        module = _init_from_artifacts(
            _make_module(
                content_enabled=True,
                gamma_content_init=1.0,
                gamma_content_max=1.0,
                warmup_epochs=0.0,
            ),
            _make_artifacts(),
        )
        tgt = torch.randn(_B, 8, _D)
        selected = torch.randint(0, _C, (_B, 8))
        confidence = torch.zeros(_B, 8)

        enhanced = module.enhance_content(tgt, selected, confidence)

        assert torch.equal(enhanced, tgt)

    def test_warmup_interpolation(self) -> None:
        """warmup 插值：epoch 0 → init，epoch >= warmup → max。"""
        module = _make_module(lambda_pos_init=0.1, lambda_pos_max=0.9, warmup_epochs=2.0)
        module.current_epoch = 0.0
        assert module.lambda_pos_effective() == pytest.approx(0.1)
        module.current_epoch = 1.0
        assert module.lambda_pos_effective() == pytest.approx(0.5)
        module.current_epoch = 5.0
        assert module.lambda_pos_effective() == pytest.approx(0.9)


class TestGradientFlow:
    def test_gradients_reach_all_trainable_params(self) -> None:
        """位置与内容分支的损失回传到全部可学习参数（content 分支需开启才能走 gate）。"""
        module = _init_from_artifacts(_make_module(content_enabled=True), _make_artifacts())
        mem = torch.randn(_B, _N, _D, requires_grad=True)
        proto_logits, _, selected = module.position_score(mem)
        tgt = torch.randn(_B, 8, _D)
        enhanced = module.enhance_content(tgt, selected[:, :8])
        loss = proto_logits.mean() + enhanced.mean()
        loss.backward()
        trainable = [name for name, p in module.named_parameters() if p.requires_grad]
        assert trainable, "模块应至少有一个可学习参数"
        for name in trainable:
            param = dict(module.named_parameters())[name]
            assert param.grad is not None, f"参数 {name} 未收到梯度"
            assert float(param.grad.abs().sum()) > 0.0, f"参数 {name} 梯度为零"


class TestArtifacts:
    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """产物保存/加载往返一致。"""
        data = _make_artifacts()
        path = tmp_path / "proto_test.pt"
        save_proto_artifacts(
            path,
            visual_prototypes=data["visual_prototypes"],  # type: ignore[arg-type]
            valid_slots=data["valid_slots"],  # type: ignore[arg-type]
            text_prototypes=data["text_prototypes"],  # type: ignore[arg-type]
            class_names=data["class_names"],  # type: ignore[arg-type]
            meta=data["meta"],  # type: ignore[arg-type]
        )
        loaded = load_proto_artifacts(path)
        assert torch.allclose(loaded["visual_prototypes"], data["visual_prototypes"])  # type: ignore[arg-type]
        assert torch.equal(loaded["valid_slots"], data["valid_slots"])  # type: ignore[arg-type]

    def test_build_missing_artifacts_returns_none(self) -> None:
        """产物缺失时 build 返回 None（恒等降级）。"""
        module = ProtoGuidance.build(num_classes=_C, hidden_dim=_D, artifacts_path="/nonexistent/x.pt")
        assert module is None


class TestFusion:
    def test_gated_not_implemented(self) -> None:
        """gated 融合 v1 未实现，构建抛 NotImplementedError。"""
        with pytest.raises(NotImplementedError):
            build_fusion("gated", hidden_dim=_D)
        with pytest.raises(NotImplementedError):
            GatedProtoFusion(_D)

    def test_simple_fusion_shapes(self) -> None:
        """simple 融合输出 [C, M, d]，无效槽位置零。"""
        module = _init_from_artifacts(_make_module(), _make_artifacts())
        p_mm, valid = module.fused_prototypes()
        assert p_mm.shape == (_C, _M, _D)
        assert valid.shape == (_C, _M)
        # 有效槽位 L2 范数 ≈ 1
        norms = p_mm[valid].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestCollectStats:
    def test_stats_keys_and_overlap(self) -> None:
        """collect_stats 输出键齐全，overlap 与手工计算一致。"""
        module = _make_module()
        proto_logits = torch.randn(_B, 8, _C)
        linear_logits = torch.randn(_B, 8, _C + 1)
        topk = torch.randint(0, _N, (_B, 8))
        linear_topk = topk.clone()
        selected = torch.randint(0, _C, (_B, 8))
        stats = module.collect_stats(proto_logits, linear_logits, topk, linear_topk, selected)
        expected = {
            "topk_overlap",
            "proto_selected_ratio",
            "lambda_effective",
            "proto_logits_pmax_mean",
            "proto_logits_entropy_mean",
            "selected_class_hist",
        }
        assert expected.issubset(stats.keys())
        assert float(stats["topk_overlap"].mean()) == pytest.approx(1.0)
        assert stats["selected_class_hist"].shape == (_C,)
        assert all(v.dtype == torch.float32 for v in stats.values())

    def test_overlap_is_set_intersection_not_rank_equality(self) -> None:
        """top-k 顺序变化但集合相同时，重叠率仍应为 1。"""
        module = _init_from_artifacts(_make_module(), _make_artifacts())
        proto_logits = torch.randn(1, 4, _C)
        linear_logits = torch.randn(1, 4, _C + 1)
        topk = torch.tensor([[1, 2, 3, 4]])
        linear_topk = torch.tensor([[4, 3, 2, 1]])
        selected = torch.randint(0, _C, (1, 4))

        stats = module.collect_stats(proto_logits, linear_logits, topk, linear_topk, selected)

        assert float(stats["topk_overlap"].item()) == pytest.approx(1.0)
        assert float(stats["proto_selected_ratio"].item()) == pytest.approx(0.0)
        assert "prototype_offdiag_cos_mean" in stats
        assert "prototype_effective_rank" in stats
