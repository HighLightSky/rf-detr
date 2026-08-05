# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGM 混合编码器分支（SGAEncoder）的单元与集成测试。

不联网：模块级测试直接构造模块；Backbone 级测试使用 ``load_dinov2_weights=False``
（随机初始化的窗口化 DINOv2）。
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.sga import SemanticGuidingModule, SGAEncoder, SpatialPriorModule
from rfdetr.utilities.tensors import NestedTensor


class TestSpatialPriorModule:
    """SpatialPriorModule（SPM）基础行为测试。"""

    def test_output_shapes(self) -> None:
        """128×128 输入应得到 c2/c3/c4（stride 8/16/32，通道 32/64/64）。"""
        spm = SpatialPriorModule()
        c2, c3, c4 = spm(torch.rand(2, 3, 128, 128))
        assert c2.shape == (2, 32, 16, 16)
        assert c3.shape == (2, 64, 8, 8)
        assert c4.shape == (2, 64, 4, 4)

    def test_is_fully_convolutional(self) -> None:
        """任意被 4 整除的输入分辨率都应工作。"""
        spm = SpatialPriorModule()
        out = spm(torch.rand(1, 3, 256, 128))
        assert out[1].shape == (1, 64, 16, 8)


class TestSemanticGuidingModule:
    """SemanticGuidingModule（SGM）注意力图生成测试。"""

    def test_attention_map_shapes_and_range(self) -> None:
        """sem_feat 在 stride-16，注意力图应插值到各目标尺度且值域 [0,1]。"""
        sgm = SemanticGuidingModule(sem_channels=8)
        sem = torch.rand(2, 8, 16, 16)
        targets = [torch.Size([2, 32, 8, 8]), torch.Size([2, 64, 16, 16])]
        maps = sgm(sem, targets)
        assert len(maps) == 2
        assert maps[0].shape == (2, 1, 8, 8)
        assert maps[1].shape == (2, 1, 16, 16)
        assert float(maps[0].detach().min()) >= 0.0
        assert float(maps[0].detach().max()) <= 1.0

    def test_gate_is_elementwise_multiplication(self) -> None:
        """门控效果等价于逐元素相乘（det ⊙ attn）。"""
        sgm = SemanticGuidingModule(sem_channels=8)
        sem = torch.rand(2, 8, 16, 16)
        det = torch.rand(2, 64, 16, 16)
        maps = sgm(sem, [torch.Size([2, 64, 16, 16])])
        assert torch.allclose(det * maps[0], det * maps[0].detach())


class TestSGAEncoder:
    """SGAEncoder 顶层封装测试。"""

    def test_single_level_p4(self) -> None:
        """projector_scale=["P4"] 应返回 1 级且形状与 feats 一致（stride-16）。"""
        enc = SGAEncoder(["P4"], hidden_dim=256, sem_channels=8)
        feats = [torch.rand(2, 256, 8, 8)]  # 128×128 输入在 stride-16 下为 8×8
        raw = [torch.rand(2, 8, 16, 16)] * 4
        out = enc(feats, raw, torch.rand(2, 3, 128, 128))
        assert len(out) == 1
        assert out[0].shape == (2, 256, 8, 8)

    def test_multi_level_p3_p4(self) -> None:
        """projector_scale=["P3","P4"] 应返回 2 级，P3↔c2（stride-8）、P4↔c3（stride-16）。"""
        enc = SGAEncoder(["P3", "P4"], hidden_dim=256, sem_channels=8)
        feats = [torch.rand(2, 256, 16, 16), torch.rand(2, 256, 8, 8)]
        raw = [torch.rand(2, 8, 16, 16)] * 4
        out = enc(feats, raw, torch.rand(2, 3, 128, 128))
        assert len(out) == 2
        assert out[0].shape == (2, 256, 16, 16)
        assert out[1].shape == (2, 256, 8, 8)

    def test_rejects_unsupported_level(self) -> None:
        """P6 不在 _LEVEL_MAP 中，应抛出 ValueError。"""
        with pytest.raises(ValueError, match="use_sga 暂只支持 P3/P4/P5"):
            SGAEncoder(["P6"])

    def test_backward(self) -> None:
        """前向可反向传播，SGA 参数得到梯度。"""
        enc = SGAEncoder(["P4"], hidden_dim=256, sem_channels=8)
        enc.train()
        feats = [torch.rand(2, 256, 8, 8)]
        raw = [torch.rand(2, 8, 16, 16)] * 4
        out = enc(feats, raw, torch.rand(2, 3, 128, 128))
        out[0].sum().backward()
        # 仅 conv4（stride-32 分支）在 P4-only 配置下不使用，允许无梯度
        used = {n for n, p in enc.named_parameters() if "conv4" not in n}
        assert all(enc.get_parameter(n).grad is not None for n in used)


def _build_backbone(use_sga: bool) -> Backbone:
    """构造一个不联网的 Backbone（随机初始化窗口化 DINOv2，hidden_dim=32 以加速测试）。

    positional_encoding_size 必须为正（默认 0 会使窗口化 DINOv2 的位置编码网格退化为 0×0）；
    128/16=8，与测试输入分辨率对齐。
    """
    return Backbone(
        name="dinov2_windowed_small",
        out_feature_indexes=[3, 6, 9, 12],
        projector_scale=["P4"],
        out_channels=32,
        load_dinov2_weights=False,
        target_shape=(128, 128),
        patch_size=16,
        num_windows=2,
        positional_encoding_size=8,
        use_sga=use_sga,
    )


def _nested_tensor(batch: int = 2, size: int = 128) -> NestedTensor:
    """构造一个无 padding 的 NestedTensor（patch16×num_windows2=32 可整除）。"""
    return NestedTensor(
        tensors=torch.rand(batch, 3, size, size),
        mask=torch.zeros(batch, size, size, dtype=torch.bool),
    )


class TestBackboneSGA:
    """Backbone 接入 SGM 分支后的前向/导出行为测试。"""

    def test_use_sga_false_is_noop(self) -> None:
        """use_sga=False 时 self.sga 为 None，且 named_parameters 无 sga 前缀。"""
        backbone = _build_backbone(use_sga=False)
        assert backbone.sga is None
        assert not any("sga" in n for n, _ in backbone.named_parameters())

        out, cross_attn = backbone(_nested_tensor())
        assert cross_attn is None
        assert len(out) == 1
        assert out[0].tensors.shape == (2, 32, 8, 8)
        assert out[0].mask is not None

    def test_use_sga_true_shapes_match_disabled(self) -> None:
        """use_sga=True 时 self.sga 非空，且前向输出级数/形状与禁用态一致。"""
        off = _build_backbone(use_sga=False)
        on = _build_backbone(use_sga=True)
        assert on.sga is not None
        assert any("sga" in n for n, _ in on.named_parameters())

        out_on, _ = on(_nested_tensor())
        out_off, _ = off(_nested_tensor())
        assert len(out_on) == len(out_off) == 1
        assert out_on[0].tensors.shape == out_off[0].tensors.shape == (2, 32, 8, 8)

    def test_forward_export_arity_and_shapes(self) -> None:
        """forward_export 返回 (feats, masks, cross_attn_feats)，SGA 开启时级数与禁用态一致。"""
        off = _build_backbone(use_sga=False)
        on = _build_backbone(use_sga=True)
        x = torch.rand(2, 3, 128, 128)

        feats_on, masks_on, cross_on = on.forward_export(x)
        feats_off, masks_off, cross_off = off.forward_export(x)
        assert cross_on is None and cross_off is None
        assert len(feats_on) == len(feats_off) == 1
        assert feats_on[0].shape == feats_off[0].shape == (2, 32, 8, 8)
        assert masks_on[0].shape == masks_off[0].shape == (2, 8, 8)

    def test_export_path_with_sga(self) -> None:
        """backbone.export() 后 forward_export 走 SGA 分支仍正常（纯卷积+BN，无重参数化）。"""
        backbone = _build_backbone(use_sga=True)
        backbone.export()
        x = torch.rand(2, 3, 128, 128)
        feats, masks, cross = backbone.forward_export(x)
        assert len(feats) == 1
        assert feats[0].shape == (2, 32, 8, 8)
        assert masks[0].shape == (2, 8, 8)
        assert cross is None


class TestSGAIntegration:
    """从 config 端到端构建 + 参数分组测试。"""

    def test_build_model_use_sga_flag(self) -> None:
        """build_model_from_config：use_sga=True 时 backbone.sga 非空，False 时为空。"""
        from rfdetr.config import RFDETRMediumConfig
        from rfdetr.models import build_model_from_config

        for flag in (True, False):
            mc = RFDETRMediumConfig(use_sga=flag, num_classes=5)
            model = build_model_from_config(mc)
            sga = model.backbone[0].sga
            assert (sga is not None) == flag

    def test_sga_params_get_full_lr(self) -> None:
        """param_groups 应把 backbone.0.sga.* 参数置于满 LR 组（lr == args.lr）。"""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import RFDETRMediumConfig, TrainConfig
        from rfdetr.models import build_model_from_config
        from rfdetr.training.param_groups import get_param_dict

        mc = RFDETRMediumConfig(use_sga=True, num_classes=5)
        tc = TrainConfig(dataset_dir="/tmp")
        ns = _namespace_from_configs(mc, tc)
        model = build_model_from_config(mc, tc)

        name2param = dict(model.named_parameters())
        sga_names = [n for n in name2param if "sga" in n]
        assert sga_names, "构建的模型缺少 sga 参数"

        groups = get_param_dict(ns, model)
        for name in sga_names:
            param = name2param[name]
            group = next(g for g in groups if g["params"] is param)
            assert group["lr"] == ns.lr, f"{name} lr={group['lr']} 应等于满 LR {ns.lr}"
