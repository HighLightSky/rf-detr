# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# 跨尺度交互编码器（Cross Fused Encoder，CFE），参考 SKYDET：
#   - RGM（Reciprocal Guidance Module）：语义/纹理互导融合（交叉相加公式）
#   - RepNCSPELAN4：尺度内交互（CSP + 重参数块，YOLOv9 结构）
#   - SCDown：轻量下采样（1×1 + depthwise 3×3 s2）
#   - 可选 TransformerEncoder：最深级自注意力（对应 SKYDET use_encoder_idx）
"""跨尺度交互编码器（CFE）。

作用于 SGA 融合后的多级特征，在自顶向下 FPN + 自底向上 PAN 中用 RGM 对相邻级做
双向互导，并用 RepNCSPELAN4 做尺度内交互，从而让 DINOv2 语义与 SPM 纹理信息
跨尺度流通。结构与超参照 SKYDET 实际代码（``engine/skydet/cross_fused_encoder.py``）。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# P 级 ↔ SPM 输出 (c2, c3, c4) 的映射（patch16 下 stride 对齐：8/16/32）
_LEVEL_MAP: dict[str, int] = {"P3": 0, "P4": 1, "P5": 2}
_STRIDE_MAP: dict[str, int] = {"P3": 8, "P4": 16, "P5": 32}


def _act(name: str | None) -> nn.Module:
    """把激活名解析为 nn.Module；未知/None 返回恒等。

    Args:
        name: 激活名（"silu"/"gelu"/"relu"）或 None。

    Returns:
        对应的激活模块。
    """
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.Identity()


class _ConvBnAct(nn.Module):
    """卷积 + BatchNorm + 可选激活 的基础块。

    归一化用 BatchNorm2d（照 SKYDET CFE 的 ConvNormLayer 语义），激活由 ``_act`` 解析。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: str | None = "silu",
    ) -> None:
        """初始化基础块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            kernel_size: 卷积核大小，padding 自动对齐。
            stride: 卷积步长。
            groups: 分组卷积（depthwise 时 = out_channels）。
            act: 激活名（"silu"/"gelu"/"relu"/None）。
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _act(act)

    def forward(self, x: Tensor) -> Tensor:
        """前向：act(bn(conv(x)))。"""
        return cast(Tensor, self.act(self.bn(self.conv(x))))


class SpatialGate(nn.Module):
    """空间注意力门控（CBAM 式）。

    输入特征，输出 [B,C,H,W] 的空间权重（逐通道广播）。公式（SKYDET）：
    ``M = σ( f^{k×k}([AvgPool_c(x); MaxPool_c(x)]) )``，然后 ``expand_as(x)``。
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """初始化空间注意力。

        Args:
            kernel_size: 卷积核大小（3 或 7），默认 7。
        """
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.compress = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B,C,H,W) → 空间权重 [B,C,H,W]，值域 [0,1]。"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = self.compress(torch.cat([avg_out, max_out], dim=1))
        return cast(Tensor, torch.sigmoid(w).expand_as(x))


class ChannelGate(nn.Module):
    """通道注意力门控（SE 式）。

    输入特征，输出 [B,C,H,W] 的通道权重（逐空间广播）。公式（SKYDET）：
    ``M = σ( W2·δ(W1·GAP(x)) )``，reduction r=16，然后 ``expand_as(x)``。
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:
        """初始化通道注意力。

        Args:
            channel: 输入通道数。
            reduction: 压缩比，默认 16。
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B,C,H,W) → 通道权重 [B,C,H,W]，值域 [0,1]。"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return cast(Tensor, y.expand_as(x))


class ReciprocalGuidanceModule(nn.Module):
    """互导融合模块（RGM）：语义特征与空间纹理特征交叉相加互导。

    x0 = 深层/语义，x1 = 浅层/纹理（方向不可调换，照 SKYDET 消融结论）。
    与论文公式不同，实际代码为**交叉相加**：
        w_spatial = SpatialGate(x0)；w_channel = ChannelGate(x1)
        out = cat([x0 + x1·w_spatial, x1 + x0·w_channel], dim=1) → (可选) 1×1 投影
    """

    def __init__(
        self,
        inc: tuple[int, int],
        ouc: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        """初始化 RGM。

        Args:
            inc: 两个输入分支的通道数 (in_c0, in_c1)。
            ouc: 输出通道数。
            reduction: 通道注意力压缩比。
            spatial_kernel: 空间注意力卷积核大小。
        """
        super().__init__()
        in_c0, in_c1 = inc
        # 通道不一致时先 1×1 对齐到 in_c1
        self.adjust_conv = nn.Identity()
        if in_c0 != in_c1:
            self.adjust_conv = _ConvBnAct(in_c0, in_c1, kernel_size=1, act=None)
            in_c0 = in_c1

        self.spatial_gate = SpatialGate(spatial_kernel)
        self.channel_gate = ChannelGate(in_c1, reduction)

        final_in = in_c1 * 2
        # 输出通道与 2C 不一致时尾接 1×1 投影
        self.conv1x1 = _ConvBnAct(final_in, ouc, kernel_size=1, act=None) if final_in != ouc else nn.Identity()

    def forward(self, x: list[Tensor]) -> Tensor:
        """x = [x0, x1]，两个 (B,C,H,W) 同分辨率特征。

        返回 (B, ouc, H, W)。
        """
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        w_spatial = self.spatial_gate(x0)  # 语义 → 空间权重
        w_channel = self.channel_gate(x1)  # 纹理 → 通道权重
        # 交叉相加：x0 吸收被空间过滤的纹理，x1 吸收被通道选择的语义
        out = torch.cat([x0 + x1 * w_spatial, x1 + x0 * w_channel], dim=1)
        return cast(Tensor, self.conv1x1(out))


class VGGBlock(nn.Module):
    """VGG 风格块：Conv3×3 + Conv1×1 求和（可重参数化）。

    本库 forward_export 走普通卷积，无需 deploy 融合，故只保留训练形态。
    """

    def __init__(self, ch_in: int, ch_out: int, act: str | None = "silu") -> None:
        """初始化 VGG 块。

        Args:
            ch_in: 输入通道数。
            ch_out: 输出通道数。
            act: 激活名。
        """
        super().__init__()
        self.conv1 = _ConvBnAct(ch_in, ch_out, kernel_size=3, act=None)
        self.conv2 = _ConvBnAct(ch_in, ch_out, kernel_size=1, act=None)
        self.act = _act(act)

    def forward(self, x: Tensor) -> Tensor:
        """前向：act(conv3×3(x) + conv1×1(x))。"""
        return cast(Tensor, self.act(self.conv1(x) + self.conv2(x)))


class CSPLayer(nn.Module):
    """CSP 层：1×1 分支 + bottleneck 分支求和后 1×1 融合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str | None = "silu",
        bottletype: type[nn.Module] = VGGBlock,
    ) -> None:
        """初始化 CSP 层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            num_blocks: bottleneck 块数量。
            expansion: 隐藏通道膨胀系数。
            act: 激活名。
            bottletype: bottleneck 块类型。
        """
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = _ConvBnAct(in_channels, hidden, kernel_size=1, act=act)
        self.conv2 = _ConvBnAct(in_channels, hidden, kernel_size=1, act=act)
        self.bottlenecks = nn.Sequential(*[bottletype(hidden, hidden, act=act) for _ in range(num_blocks)])
        self.conv3 = (
            _ConvBnAct(hidden, out_channels, kernel_size=1, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向：conv3(bottlenecks(conv1(x)) + conv2(x))。"""
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return cast(Tensor, self.conv3(self.bottlenecks(x1) + x2))


class RepNCSPELAN4(nn.Module):
    """RepNCSPELAN4 块（YOLOv9 CSP-ELAN）：尺度内交互主体。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 3,
        act: str | None = "silu",
    ) -> None:
        """初始化 RepNCSPELAN4。

        Args:
            c1: 输入通道数。
            c2: 输出通道数。
            c3: 中间通道数（cv1 输出，随后切半）。
            c4: CSP 隐藏通道数。
            n: CSPLayer 内 bottleneck 数量。
            act: 激活名。
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = _ConvBnAct(c1, c3, kernel_size=1, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, act=act),
            _ConvBnAct(c4, c4, kernel_size=3, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, act=act),
            _ConvBnAct(c4, c4, kernel_size=3, act=act),
        )
        self.cv4 = _ConvBnAct(c3 + 2 * c4, c2, kernel_size=1, act=act)

    def forward(self, x: Tensor) -> Tensor:
        """前向：cv1 切两半，cv2/cv3 串接，cat 后 cv4 融合。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return cast(Tensor, self.cv4(torch.cat(y, 1)))


class SCDown(nn.Module):
    """轻量下采样：1×1 通道变换 + depthwise 3×3 s2。"""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, act: str | None = None) -> None:
        """初始化 SCDown。

        Args:
            c1: 输入通道数。
            c2: 输出通道数。
            k: depthwise 卷积核大小。
            s: 下采样步长（默认 2）。
            act: 激活名（默认 None = 恒等）。
        """
        super().__init__()
        self.cv1 = _ConvBnAct(c1, c2, kernel_size=1, act=act)
        self.cv2 = _ConvBnAct(c2, c2, kernel_size=k, stride=s, groups=c2, act=act)

    def forward(self, x: Tensor) -> Tensor:
        """前向：depthwise s2(cv1(x))。"""
        return cast(Tensor, self.cv2(self.cv1(x)))


class _TransformerEncoderLayer(nn.Module):
    """单层 Transformer 编码器（post-norm）：自注意力 + FFN。

    对应 SKYDET CFE 中最深级的自注意力（use_encoder_idx）。为简化实现与导出，
    未加位置编码（默认关闭该功能）。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """初始化编码器层。

        Args:
            d_model: 特征维度。
            nhead: 注意力头数。
            dim_feedforward: FFN 隐藏维度。
            dropout: dropout 率。
            activation: FFN 激活名。
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = _act(activation)

    def forward(self, src: Tensor) -> Tensor:
        """src: (B, L, C) → 同形状。"""
        # 自注意力 + 残差 + LayerNorm
        attn_out, _ = self.self_attn(src, src, value=src)
        src = self.norm1(src + self.dropout(attn_out))
        # FFN + 残差 + LayerNorm
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = self.norm2(src + self.dropout(ffn_out))
        return src


class CrossScaleEncoder(nn.Module):
    """跨尺度交互编码器（CFE）。

    作用于 SGA 融合后的多级特征，做：
        1. 自顶向下 FPN：lateral 1×1 → upsample 2× → RGM → RepNCSPELAN4
        2. 自底向上 PAN：SCDown → RGM → RepNCSPELAN4
        3. （可选）最深级 TransformerEncoder 自注意力
    输出与输入同构（长度/通道/分辨率不变），从而对 decoder/mask 零改动。
    """

    def __init__(
        self,
        projector_scale: list[str],
        hidden_dim: int = 256,
        use_encoder: bool = False,
        gradient_checkpointing: bool = False,
        num_encoder_layers: int = 1,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = "silu",
    ) -> None:
        """初始化跨尺度交互编码器。

        Args:
            projector_scale: P 级列表（P3/P4/P5，需至少 2 级）。
            hidden_dim: 特征通道数（与 SGA 输出一致）。
            use_encoder: 是否在最深级叠加 TransformerEncoder。
            gradient_checkpointing: 训练时是否用激活检查点（重算前向换显存）。
            num_encoder_layers: 自注意力层数。
            expansion: CSPLayer 通道膨胀系数。
            depth_mult: RepNCSPELAN4 深度倍率。
            act: 激活名。
        """
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if len(projector_scale) < 2:
            raise ValueError(f"跨尺度交互需要至少 2 个金字塔等级，收到 {projector_scale}")
        unsupported = set(projector_scale) - set(_LEVEL_MAP)
        if unsupported:
            raise ValueError(f"CFE 暂只支持 P3/P4/P5，收到不支持的等级: {unsupported}")

        self.projector_scale = list(projector_scale)
        self.strides = [_STRIDE_MAP[lvl] for lvl in projector_scale]
        n_levels = len(projector_scale)

        # 可选：最深级自注意力（照 SKYDET use_encoder_idx）
        self.encoder: nn.Module | None = (
            nn.ModuleList([_TransformerEncoderLayer(hidden_dim, activation=act) for _ in range(num_encoder_layers)])
            if use_encoder
            else None
        )

        # 自顶向下 FPN：lateral 1×1 → upsample → RGM → RepNCSPELAN4
        self.lateral_convs: nn.ModuleList = nn.ModuleList()
        self.fpn_fusion: nn.ModuleList = nn.ModuleList()
        self.fpn_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(n_levels - 1):
            self.lateral_convs.append(_ConvBnAct(hidden_dim, hidden_dim, kernel_size=1, act=None))
            self.fpn_fusion.append(ReciprocalGuidanceModule([hidden_dim, hidden_dim], hidden_dim * 2))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                    act=act,
                )
            )

        # 自底向上 PAN：SCDown → RGM → RepNCSPELAN4
        self.downsample_convs: nn.ModuleList = nn.ModuleList()
        self.pan_fusion: nn.ModuleList = nn.ModuleList()
        self.pan_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(n_levels - 1):
            self.downsample_convs.append(SCDown(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_fusion.append(ReciprocalGuidanceModule([hidden_dim, hidden_dim], hidden_dim * 2))
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                    act=act,
                )
            )

    def _apply_encoder(self, feats: list[Tensor], idx: int) -> None:
        """对第 idx 级做自注意力（就地改写 feats[idx]）。"""
        if self.encoder is None:
            return
        feat = feats[idx]
        b, c, h, w = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        for layer in self.encoder:
            src = layer(src)
        feats[idx] = src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """feats: SGA 融合后的多级特征（hidden_dim，各级 stride 递增）。

        返回与 feats 同构的互导融合后特征列表。开启 ``gradient_checkpointing`` 时，
        训练阶段把内部前向包进 ``torch.utils.checkpoint``：CFE 的中间激活（尤其 P3 级，
        全网分辨率最高）不缓存，反向时重算前向换取显存。非 reentrant 模式支持
        ``list[Tensor]`` 返回值，且内部不会改写输入张量，可安全检查点化。
        """
        assert len(feats) == len(self.projector_scale)
        if self.gradient_checkpointing and self.training:
            # 逐级张量传入 checkpoint；返回时统一转回 list
            return list(torch.utils.checkpoint.checkpoint(self._forward_impl, *feats, use_reentrant=False))
        return self._forward_impl(*feats)

    def _forward_impl(self, *feats: Tensor) -> list[Tensor]:
        """实际前向实现：自顶向下 FPN + 自底向上 PAN + RGM 双向互导。

        Args:
            feats: 各级特征，与 ``forward`` 的 list 参数逐项对应（checkpoint 要求逐张量传入）。

        Returns:
            与输入同构的互导融合后特征列表。
        """
        proj = list(feats)

        # 可选：最深级自注意力
        if self.encoder is not None:
            self._apply_encoder(proj, len(proj) - 1)

        # 自顶向下 FPN：从最深级向下融合
        inner_outs: list[Tensor] = [proj[-1]]
        for idx in range(len(self.projector_scale) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj[idx - 1]
            lateral = self.lateral_convs[len(self.projector_scale) - 1 - idx](feat_high)
            upsample = F.interpolate(lateral, scale_factor=2.0, mode="nearest")
            fused = self.fpn_fusion[len(self.projector_scale) - 1 - idx]([upsample, feat_low])
            inner_outs.insert(0, self.fpn_blocks[len(self.projector_scale) - 1 - idx](fused))

        # 自底向上 PAN：从最浅级向下采样融合
        outs: list[Tensor] = [inner_outs[0]]
        for idx in range(len(self.projector_scale) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            down = self.downsample_convs[idx](feat_low)
            fused = self.pan_fusion[idx]([down, feat_high])
            outs.append(self.pan_blocks[idx](fused))

        return outs
