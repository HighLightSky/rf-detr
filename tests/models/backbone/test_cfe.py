# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""跨尺度交互编码器（CFE）的单元与集成测试。

不联网：模块级测试直接构造模块；Backbone 级测试使用 ``load_dinov2_weights=False``。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.cfe import (
    ChannelGate,
    CrossScaleEncoder,
    ReciprocalGuidanceModule,
    RepNCSPELAN4,
    SCDown,
    SpatialGate,
)
from rfdetr.utilities.tensors import NestedTensor


class TestGates:
    """空间/通道注意力门控测试。"""

    def test_spatial_gate_shape_and_range(self) -> None:
        """SpatialGate 输出 [B,C,H,W]（expand_as），值域 [0,1]。"""
        w = SpatialGate()(torch.rand(2, 32, 8, 8))
        assert w.shape == (2, 32, 8, 8)
        assert float(w.detach().min()) >= 0.0
        assert float(w.detach().max()) <= 1.0

    def test_channel_gate_shape_and_range(self) -> None:
        """ChannelGate 输出 [B,C,H,W]（expand_as），值域 [0,1]。"""
        w = ChannelGate(32)(torch.rand(2, 32, 8, 8))
        assert w.shape == (2, 32, 8, 8)
        assert float(w.detach().min()) >= 0.0
        assert float(w.detach().max()) <= 1.0


class TestReciprocalGuidanceModule:
    """RGM 互导融合测试。"""

    def test_output_shape_and_backward(self) -> None:
        """两个 [B,32,H,W] 输入 → [B,64,H,W]（交叉相加），可反向。"""
        rgm = ReciprocalGuidanceModule([32, 32], 64)
        out = rgm([torch.rand(2, 32, 8, 8), torch.rand(2, 32, 8, 8)])
        assert out.shape == (2, 64, 8, 8)
        out.sum().backward()
        assert all(p.grad is not None for p in rgm.parameters())

    def test_output_channel_projection(self) -> None:
        """ouc != 2C 时尾接 1×1 投影到 ouc。"""
        rgm = ReciprocalGuidanceModule([32, 32], 32)
        out = rgm([torch.rand(2, 32, 8, 8), torch.rand(2, 32, 8, 8)])
        assert out.shape == (2, 32, 8, 8)


class TestBlocks:
    """RepNCSPELAN4 / SCDown 基础块测试。"""

    def test_repncspelan4_shape(self) -> None:
        """RepNCSPELAN4(64→32) 保持分辨率。"""
        out = RepNCSPELAN4(64, 32, 64, 32, n=2)(torch.rand(2, 64, 16, 16))
        assert out.shape == (2, 32, 16, 16)

    def test_scdown_shape(self) -> None:
        """SCDown 下采样 2×。"""
        out = SCDown(32, 32)(torch.rand(2, 32, 16, 16))
        assert out.shape == (2, 32, 8, 8)


class TestCrossScaleEncoder:
    """CrossScaleEncoder 编排器测试。"""

    def test_p3_p4_preserves_shapes(self) -> None:
        """P3+P4 输入两级，输出同形状（128 输入：P3=16×16、P4=8×8）。"""
        enc = CrossScaleEncoder(["P3", "P4"], hidden_dim=32)
        feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 32, 8, 8)]
        out = enc(feats)
        assert len(out) == 2
        assert out[0].shape == (2, 32, 16, 16)
        assert out[1].shape == (2, 32, 8, 8)

    def test_use_encoder_preserves_shapes(self) -> None:
        """use_encoder=True 时最深级自注意力，输出形状不变。"""
        enc = CrossScaleEncoder(["P3", "P4"], hidden_dim=32, use_encoder=True)
        feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 32, 8, 8)]
        out = enc(feats)
        assert [tuple(o.shape) for o in out] == [(2, 32, 16, 16), (2, 32, 8, 8)]

    def test_single_level_rejected(self) -> None:
        """单级构造应抛 ValueError（跨尺度需要 ≥2 级）。"""
        with pytest.raises(ValueError, match="至少 2 个金字塔等级"):
            CrossScaleEncoder(["P4"])

    def test_backward_all_params_grad(self) -> None:
        """前向可反向，全部 CFE 参数得到梯度。"""
        enc = CrossScaleEncoder(["P3", "P4"], hidden_dim=32)
        enc.train()
        feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 32, 8, 8)]
        out = enc(feats)
        sum(o.sum() for o in out).backward()
        assert all(p.grad is not None for p in enc.parameters())

    def test_gradient_checkpointing_equivalent_and_backward(self) -> None:
        """训练态开关检查点对输出无影响（同权重同输入），且可反向、梯度齐全。"""
        enc = CrossScaleEncoder(["P3", "P4"], hidden_dim=32, gradient_checkpointing=True)
        enc.train()
        feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 32, 8, 8)]
        # 同一实例、同一权重：先关检查点再开检查点（训练态 BN 用 batch 统计，输出确定）
        enc.gradient_checkpointing = False
        out_plain = enc(feats)
        enc.gradient_checkpointing = True
        out_ckpt = enc(feats)
        assert [tuple(o.shape) for o in out_ckpt] == [tuple(o.shape) for o in out_plain]
        for a, b in zip(out_ckpt, out_plain):
            assert torch.allclose(a, b, atol=1e-5)
        sum(o.sum() for o in out_ckpt).backward()
        assert all(p.grad is not None for p in enc.parameters())


def _build_backbone(
    projector_scale: list[str] = ["P3", "P4"],
    use_sga: bool = True,
    use_cfe: bool = False,
) -> Backbone:
    """构造一个不联网的 Backbone（随机初始化窗口化 DINOv2，hidden_dim=32 以加速测试）。"""
    return Backbone(
        name="dinov2_windowed_small",
        out_feature_indexes=[3, 6, 9, 12],
        projector_scale=projector_scale,
        out_channels=32,
        load_dinov2_weights=False,
        target_shape=(128, 128),
        patch_size=16,
        num_windows=2,
        positional_encoding_size=8,
        use_sga=use_sga,
        use_cfe=use_cfe,
    )


def _nested_tensor(batch: int = 2, size: int = 128) -> NestedTensor:
    """构造一个无 padding 的 NestedTensor（patch16×num_windows2=32 可整除）。"""
    return NestedTensor(
        tensors=torch.rand(batch, 3, size, size),
        mask=torch.zeros(batch, size, size, dtype=torch.bool),
    )


class TestBackboneCFE:
    """Backbone 接入 CFE 后的前向/导出行为测试。"""

    def test_use_cfe_false_is_noop(self) -> None:
        """use_cfe=False 时 self.cfe 为 None，named_parameters 无 cfe 前缀。"""
        backbone = _build_backbone(use_cfe=False)
        assert backbone.cfe is None
        assert not any("cfe" in n for n, _ in backbone.named_parameters())

    def test_use_cfe_true_shapes_match_disabled(self) -> None:
        """use_cfe=True 时 self.cfe 非空，且前向输出级数/形状与禁用态一致。"""
        off = _build_backbone(use_cfe=False)
        on = _build_backbone(use_cfe=True)
        assert on.cfe is not None
        assert any("cfe" in n for n, _ in on.named_parameters())

        out_on, _ = on(_nested_tensor())
        out_off, _ = off(_nested_tensor())
        assert len(out_on) == len(out_off) == 2
        assert [tuple(f.tensors.shape) for f in out_on] == [(2, 32, 16, 16), (2, 32, 8, 8)]
        assert [tuple(f.tensors.shape) for f in out_off] == [(2, 32, 16, 16), (2, 32, 8, 8)]

    def test_forward_export_arity_and_shapes(self) -> None:
        """forward_export 返回两级特征，CFE 开启时形状正常。"""
        on = _build_backbone(use_cfe=True)
        x = torch.rand(2, 3, 128, 128)
        feats, masks, cross = on.forward_export(x)
        assert cross is None
        assert len(feats) == len(masks) == 2
        assert feats[0].shape == (2, 32, 16, 16)
        assert feats[1].shape == (2, 32, 8, 8)
        assert masks[0].shape == (2, 16, 16)
        assert masks[1].shape == (2, 8, 8)

    def test_export_path_with_cfe(self) -> None:
        """backbone.export() 后 forward_export 走 CFE 分支仍正常。"""
        backbone = _build_backbone(use_cfe=True)
        backbone.export()
        x = torch.rand(2, 3, 128, 128)
        feats, masks, cross = backbone.forward_export(x)
        assert len(feats) == 2
        assert cross is None


class TestCFEConfig:
    """use_cfe 配置字段与校验测试。"""

    def test_default_false(self) -> None:
        """默认关闭 CFE。"""
        from rfdetr.config import RFDETRMediumConfig

        assert RFDETRMediumConfig().use_cfe is False

    def test_requires_use_sga(self) -> None:
        """use_cfe=True 但 use_sga=False 应抛 ValueError。"""
        from rfdetr.config import RFDETRMediumConfig

        with pytest.raises(ValueError, match="use_sga=True"):
            RFDETRMediumConfig(use_cfe=True, use_sga=False)

    def test_requires_multi_level(self) -> None:
        """use_cfe=True + use_sga=True 但单级 projector_scale 应抛 ValueError。"""
        from rfdetr.config import RFDETRMediumConfig

        with pytest.raises(ValueError, match="至少 2 个金字塔等级"):
            RFDETRMediumConfig(use_cfe=True, use_sga=True)  # 默认单级 ["P4"]

    def test_valid_multi_level_warns(self) -> None:
        """use_cfe=True + use_sga=True + P3/P4 可构造，并触发预训练兼容警告。"""
        import warnings

        from rfdetr.config import PretrainWeightsCompatibilityWarning, RFDETRMediumConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mc = RFDETRMediumConfig(use_cfe=True, use_sga=True, projector_scale=["P3", "P4"], num_classes=5)
        assert mc.use_cfe is True
        assert any(issubclass(w.category, PretrainWeightsCompatibilityWarning) for w in caught)

    def test_namespace_forwards_cfe(self) -> None:
        """_namespace_from_configs 应透传 use_cfe 与 cfe_* 字段。"""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import RFDETRMediumConfig, TrainConfig

        mc = RFDETRMediumConfig(use_cfe=True, use_sga=True, projector_scale=["P3", "P4"], num_classes=5)
        tc = TrainConfig(dataset_dir="/tmp")
        ns = _namespace_from_configs(mc, tc)
        assert ns.use_cfe is True
        assert ns.cfe_act == "silu"


class TestCFEIntegration:
    """从 config 端到端构建 + 参数分组测试。"""

    def test_build_model_use_cfe_flag(self) -> None:
        """build_model_from_config：use_cfe=True 时 backbone.cfe 非空，False 时为空。"""
        from rfdetr.config import RFDETRMediumConfig
        from rfdetr.models import build_model_from_config

        for flag in (True, False):
            mc = RFDETRMediumConfig(use_cfe=flag, use_sga=True, projector_scale=["P3", "P4"], num_classes=5)
            model = build_model_from_config(mc)
            cfe = model.backbone[0].cfe
            assert (cfe is not None) == flag

    def test_cfe_params_get_full_lr(self) -> None:
        """param_groups 应把 backbone.0.cfe.* 参数置于满 LR 组（lr == args.lr）。"""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import RFDETRMediumConfig, TrainConfig
        from rfdetr.models import build_model_from_config
        from rfdetr.training.param_groups import get_param_dict

        mc = RFDETRMediumConfig(use_cfe=True, use_sga=True, projector_scale=["P3", "P4"], num_classes=5)
        tc = TrainConfig(dataset_dir="/tmp")
        ns = _namespace_from_configs(mc, tc)
        model = build_model_from_config(mc, tc)

        name2param = dict(model.named_parameters())
        cfe_names = [n for n in name2param if "cfe" in n]
        assert cfe_names, "构建的模型缺少 cfe 参数"

        groups = get_param_dict(ns, model)
        for name in cfe_names:
            param = name2param[name]
            group = next(g for g in groups if g["params"] is param)
            assert group["lr"] == ns.lr, f"{name} lr={group['lr']} 应等于满 LR {ns.lr}"
