# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""QNorm-Obj + EUMix（QNormObjectness）的单元测试。

不依赖 GPU 与网络，验证：
- 前向输出形状与输入一致（含单层 decoder 边界）。
- 近恒等初始化：起步 logits 相对输入仅有界扰动（门控/特征混合不破坏预训练行为）。
- 梯度流到全部 11 个可学习参数。
- 子开关语义：eumix 关 → 背景列不变；gate 关 → 前景 logits 不被物体性缩放；
  feature_mix 关 → 直接使用输入 logits。
- 全子开关关闭时抛 ValueError；build() 从 TrainConfig 装配并校验开关映射。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.config import TrainConfig
from rfdetr.sscl.qnorm_obj import QNormObjectness

# 测试维度：L=6 层、B=2、Q=8、d=16（低于真实 256 以保持 CPU 测试轻量）、C=5 前景类
_L, _B, _Q, _D, _C = 6, 2, 8, 16, 5


def _make_inputs(layers: int = _L) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造与模块重算一致的输入（模拟 class_embed(hs)）。

    Args:
        layers: decoder 层数。

    Returns:
        ``(hs, z_cls, w, b)`` 四元组：hidden states、分类 logits（由
        ``hs @ w.T + b`` 构造，与 feature_mix 开启时模块重算路径一致）、
        分类头权重/偏置。
    """
    generator = torch.Generator().manual_seed(0)
    hs = torch.randn(layers, _B, _Q, _D, generator=generator)
    w = torch.randn(_C + 1, _D, generator=generator) * 0.1
    b = torch.randn(_C + 1, generator=generator) * 0.1
    z_cls = hs @ w.T + b
    return hs, z_cls, w, b


def _make_module(**overrides: object) -> QNormObjectness:
    """构造测试用模块，默认全开关。

    Args:
        **overrides: 覆盖 QNormObjectness 构造参数。

    Returns:
        QNormObjectness 实例。
    """
    kwargs: dict[str, object] = {"hidden_dim": _D, "num_classes": _C}
    kwargs.update(overrides)
    return QNormObjectness(**kwargs)  # type: ignore[arg-type]


def test_forward_shape_preserved() -> None:
    """前向输出形状与输入 logits 全层栈一致。"""
    hs, z_cls, w, b = _make_inputs()
    z_out, stats = _make_module()(hs, z_cls, w, b)
    assert z_out.shape == z_cls.shape
    # 统计字典包含监控键与标量参数（全部 detach）
    for key in ("obj", "p_max", "p_obj_bg", "alpha_mix", "alpha", "gamma", "lambda_suppress", "b_obj"):
        assert key in stats
    assert stats["obj"].shape == (_B, _Q)
    assert stats["p_max"].shape == (_B, _Q)
    assert not stats["obj"].requires_grad


def test_forward_single_layer() -> None:
    """单层 decoder（aux 关闭的边界情形）形状正确。"""
    hs, z_cls, w, b = _make_inputs(layers=1)
    z_out, _ = _make_module()(hs, z_cls, w, b)
    assert z_out.shape == z_cls.shape == (1, _B, _Q, _C + 1)


def test_near_identity_init() -> None:
    """近恒等初始化：起点 logits 相对输入仅有界扰动。

    门控初始 σ(z_obj)≈0.98、α_mix=0（特征混合恒等）、EUMix 混合权重 α=0.1
    （背景列 90% 保持输入）——起步行为≈基线，不扰动预训练权重下的匹配。
    前景/背景列分别给出宽松界（含门控 2% 缩放与前景软抑制 ≤λ·1）。
    """
    hs, z_cls, w, b = _make_inputs()
    z_out, stats = _make_module()(hs, z_cls, w, b)
    fg_err = (z_out[..., :_C] - z_cls[..., :_C]).abs().max().item()
    bg_err = (z_out[..., _C:] - z_cls[..., _C:]).abs().max().item()
    # 初始有效 α = σ(logit⁻¹(0.1)) = 0.1
    assert abs(stats["alpha"].item() - 0.1) < 1e-5
    assert stats["alpha_mix"].item() == 0.0
    assert fg_err < 1.0  # 门控 2% + 软抑制 ≤0.5
    assert bg_err < 2.0  # 混合权重 10% 贡献


def test_grad_flows_to_all_params() -> None:
    """标准检测损失式反向传播：全部 11 个可学习参数均收到梯度。"""
    hs, z_cls, w, b = _make_inputs()
    module = _make_module()
    z_out, _ = module(hs, z_cls, w, b)
    z_out.sum().backward()
    named = {name: param for name, param in module.named_parameters()}
    assert len(named) == 11
    for name, param in named.items():
        assert param.grad is not None, f"参数 {name} 未收到梯度"
        assert param.grad.abs().sum().item() > 0.0, f"参数 {name} 梯度为零"


def test_eumix_off_background_column_unchanged() -> None:
    """eumix=False：背景列与输入完全一致（仅特征混合路径生效）。"""
    hs, z_cls, w, b = _make_inputs()
    z_out, _ = _make_module(eumix=False)(hs, z_cls, w, b)
    # α_mix=0 时特征混合重算 = class_embed(hs) = 输入（浮点精度内一致）
    assert (z_out[..., _C:] - z_cls[..., _C:]).abs().max().item() < 1e-5


def test_gate_off_no_objectness_scaling() -> None:
    """gate=False：前景 logits 不被物体性缩放（仅剩软抑制的有界扰动）。"""
    hs, z_cls, w, b = _make_inputs()
    z_out, stats = _make_module(gate=False)(hs, z_cls, w, b)
    # 软抑制 ≤ λ·p_obj_bg ≤ 0.5；无 gate 缩放
    diff = (z_out[..., :_C] - z_cls[..., :_C]).abs()
    assert diff.max().item() < 0.6
    assert stats["obj"].shape == (_B, _Q)


def test_gate_suppresses_low_objectness() -> None:
    """门控方向性：物体性压到 ≈0 时，前景 logits 被整体压塌（低物体性背景框被压）。"""
    hs, z_cls, w, b = _make_inputs()
    module = _make_module()
    z_ref, _ = module(hs, z_cls, w, b)  # 正常门（σ(z_obj)≈0.98）下的前景
    with torch.no_grad():
        module.obj_head[-1].bias.fill_(-20.0)  # σ(z_obj) → 0，门关闭
    z_suppressed, _ = module(hs, z_cls, w, b)
    ref_scale = z_ref[..., :_C].abs().mean().item()
    suppressed_scale = z_suppressed[..., :_C].abs().mean().item()
    assert suppressed_scale < 0.3 * ref_scale


def test_feature_mix_off_uses_input_logits() -> None:
    """feature_mix=False：输出直接来自输入 logits（不按 hs 重算）。"""
    hs, z_cls, w, b = _make_inputs()
    module = _make_module(feature_mix=False)
    z_out1, _ = module(hs, z_cls, w, b)
    # 输入 logits 整体平移 1.0（保持 hs 不变）→ 前景输出随动 ≈ gate(0.98)
    z2 = z_cls + 1.0
    z_out2, _ = module(hs, z2, w, b)
    delta = (z_out2[..., :_C] - z_out1[..., :_C]).abs()
    assert (delta - 0.98).abs().max().item() < 0.3  # 软抑制随 p_max 变化的扰动 ≤0.5


def test_all_switches_off_raises() -> None:
    """feature_mix/gate/eumix 全关时抛 ValueError（模块无意义）。"""
    with pytest.raises(ValueError):
        _make_module(feature_mix=False, gate=False, eumix=False)


def test_alpha_init_out_of_range_raises() -> None:
    """alpha_init 越界（≤0 或 ≥1）时抛 ValueError。"""
    with pytest.raises(ValueError):
        _make_module(alpha_init=0.0)
    with pytest.raises(ValueError):
        _make_module(alpha_init=1.0)


def test_build_from_train_config() -> None:
    """build() 从 TrainConfig 装配：开关映射正确，关闭时抛 ValueError。"""
    cfg = TrainConfig(
        dataset_dir="dummy",
        qnorm_obj_enabled=True,
        qnorm_obj_tau=4.0,
        qnorm_obj_obj_hidden_dim=32,
    )
    module = QNormObjectness.build(cfg, num_classes=_C, hidden_dim=_D)
    assert module.tau == 4.0
    assert module.feature_mix and module.gate and module.eumix
    assert module.obj_head[0].out_features == 32

    cfg_off = TrainConfig(dataset_dir="dummy", qnorm_obj_enabled=False)
    with pytest.raises(ValueError):
        QNormObjectness.build(cfg_off, num_classes=_C, hidden_dim=_D)
