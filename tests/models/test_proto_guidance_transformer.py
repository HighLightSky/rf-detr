# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ProtoGuidance 与 Transformer 集成的回归测试。

用小尺寸 Transformer（two_stage、group_detr=2）挂载 ProtoGuidance，验证：
- 返回元组第 5 位（proto_logits_ts）形状为 [bs, num_queries*group, C]。
- 近恒等初始化：lambda 小时 topk_overlap >= 0.98（初始不扰乱 query selection）。
- 位置/内容双关时输出与原版完全一致（恒等短路）。
- lambda 增大后原型分数真实改变 top-k（overlap < 1）。
- 无模块时返回元组第 5 位为 None（原版行为兼容）。
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from rfdetr.models.transformer import Transformer
from rfdetr.sscl.proto_guidance.guidance import ProtoGuidance

# 测试维度：d=8、C=5 前景类、M=3 槽位、单尺度 8x8、N=8 queries、G=2 组
_BS, _D, _C, _M = 2, 8, 5, 3
_NQ, _G, _H, _W = 8, 2, 8, 8


def _make_transformer(**overrides: object) -> Transformer:
    """构造测试用小 Transformer（two_stage + group_detr，CPU 可跑）。

    仿照 LWDETR.__init__ 装配 two_stage 的 ``enc_out_class_embed``/
    ``enc_out_bbox_embed``（原代码在 LWDETR 侧装配，Transformer 自身只声明 None）。

    Args:
        **overrides: 覆盖 Transformer 构造参数。

    Returns:
        Transformer 实例。
    """
    kwargs: dict[str, object] = {
        "d_model": _D,
        "sa_nhead": 2,
        "ca_nhead": 2,
        "num_queries": _NQ,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "group_detr": _G,
        "two_stage": True,
        "num_feature_levels": 1,
        "dec_n_points": 1,
        "bbox_reparam": True,
        "return_intermediate_dec": True,  # decoder 返回 (hs, references) 二元组
    }
    kwargs.update(overrides)
    transformer = Transformer(**kwargs)  # type: ignore[arg-type]
    if transformer.two_stage:
        class_embed = nn.Linear(_D, _C + 1)  # 前景 C + background 1
        bbox_embed = nn.Linear(_D, 4)
        transformer.enc_out_class_embed = nn.ModuleList(
            [copy.deepcopy(class_embed) for _ in range(transformer.group_detr)]
        )
        transformer.enc_out_bbox_embed = nn.ModuleList(
            [copy.deepcopy(bbox_embed) for _ in range(transformer.group_detr)]
        )
    if not getattr(transformer.decoder, "lite_refpoint_refine", False):
        # 非 lite 模式 decoder 用 bbox_embed 做 refpoints refine（LWDETR 侧装配行为）
        transformer.decoder.bbox_embed = nn.Linear(_D, 4)
    return transformer


def _make_proto_guidance(**overrides: object) -> ProtoGuidance:
    """构造测试用 ProtoGuidance，并注入有效的随机离线原型。"""
    kwargs: dict[str, object] = {
        "num_classes": _C,
        "hidden_dim": _D,
        "text_dim": 4,
        "num_slots": _M,
        "warmup_epochs": 0.0,  # 直接到上限，便于控制 lambda
    }
    kwargs.update(overrides)
    module = ProtoGuidance(**kwargs)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        module.visual_bank.prototypes.copy_(
            torch.randn(_C, _M, _D, generator=generator)
        )
        module.visual_bank.slot_valid_mask.fill_(True)
        module.P_t_clip.copy_(torch.randn(_C, 4, generator=generator))
    return module


def _forward(transformer: Transformer) -> tuple:
    """执行一次前向，返回 transformer 输出元组。"""
    generator = torch.manual_seed(0)
    srcs = [torch.randn(_BS, _D, _H, _W, generator=generator)]
    masks = [torch.zeros(_BS, _H, _W, dtype=torch.bool)]
    poss = [torch.randn(_BS, _D, _H, _W, generator=generator)]
    refpoint_embed = torch.randn(_NQ * _G, 4, generator=generator)
    query_feat = torch.randn(_NQ * _G, _D, generator=generator)
    return transformer(srcs, masks, poss, refpoint_embed, query_feat)


def _proto_logits_of(out: tuple) -> torch.Tensor | None:
    """取返回元组第 5 位（proto_logits_ts）。"""
    return out[4]


class TestTransformerIntegration:
    def test_return_slot_shape(self) -> None:
        """模块挂载后返回元组第 5 位形状为 [bs, nq*G, C]。"""
        transformer = _make_transformer()
        transformer.proto_guidance = _make_proto_guidance()
        out = _forward(transformer)
        proto_logits = _proto_logits_of(out)
        assert proto_logits is not None
        assert proto_logits.shape == (_BS, _NQ * _G, _C)

    def test_dense_tokens_use_inference_group_only(self) -> None:
        """训练态 dense token 只保留 group 0，确保与 eval 推理路径一致。"""
        transformer = _make_transformer()
        transformer.proto_guidance_dense_loss_enabled = True
        transformer.proto_guidance = _make_proto_guidance()
        out = _forward(transformer)
        dense = out[5]
        assert dense is not None
        assert dense["pred_proto_logits_dense"].shape == (_BS, _H * _W, _C)

    def test_without_module_returns_none(self) -> None:
        """无模块时第 5 位为 None（原版行为兼容）。"""
        transformer = _make_transformer()
        out = _forward(transformer)
        assert _proto_logits_of(out) is None

    def test_near_identity_topk_overlap(self) -> None:
        """lambda 很小（0.05）时 topk_overlap >= 0.98（近恒等起步）。"""
        transformer = _make_transformer()
        module = _make_proto_guidance(
            lambda_pos_init=0.05,
            lambda_pos_max=0.05,
            position_enabled=True,
            content_enabled=False,
        )
        transformer.proto_guidance = module
        _forward(transformer)
        assert module.last_stats is not None
        assert float(module.last_stats["topk_overlap"].mean()) >= 0.98

    def test_large_lambda_changes_topk(self) -> None:
        """lambda 足够大时原型分数真实改变 top-k（overlap < 1）。"""
        transformer = _make_transformer()
        module = _make_proto_guidance(
            lambda_pos_init=10.0,
            lambda_pos_max=10.0,
            position_enabled=True,
            content_enabled=False,
        )
        transformer.proto_guidance = module
        _forward(transformer)
        assert module.last_stats is not None
        assert float(module.last_stats["topk_overlap"].mean()) < 1.0

    def test_disabled_is_identical_to_baseline(self) -> None:
        """位置/内容双关时前向输出与原版完全一致（恒等短路，同一实例挂模块前后对比）。"""
        transformer = _make_transformer()
        # 先记录原版输出（无模块）
        baseline_out = _forward(transformer)
        # 同一实例挂载双关模块后再次前向
        transformer.proto_guidance = _make_proto_guidance(
            position_enabled=False, content_enabled=False
        )
        disabled_out = _forward(transformer)

        for idx, (base, mod) in enumerate(zip(baseline_out, disabled_out)):
            if idx == 4:
                # 第 5 位 proto_logits_ts 是新输出：模块存在时总会记录（供内容增强/监控），
                # 与原版 None 本就不同，不属于"恒等短路"承诺范围。
                continue
            if base is None or mod is None:
                assert base is mod, "None 位置必须一致"
            else:
                assert torch.allclose(base, mod, atol=1e-6), "关闭模块时输出必须与原版一致"

    def test_content_enabled_changes_tgt(self) -> None:
        """内容增强开启后输出与原版不同（注入生效，同一实例对比）。"""
        transformer = _make_transformer()
        baseline_out = _forward(transformer)
        transformer.proto_guidance = _make_proto_guidance(
            position_enabled=False,
            content_enabled=True,
            gamma_content_init=1.0,
            gamma_content_max=1.0,
        )
        enhanced_out = _forward(transformer)

        base_hs = baseline_out[0]
        enh_hs = enhanced_out[0]
        assert base_hs is not None and enh_hs is not None
        assert not torch.allclose(base_hs, enh_hs, atol=1e-6), "内容增强应改变 decoder 输出"
