# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# SGM 混合编码器分支（Semantic Guiding Adapter），参考 SKYDET：
#   - SPM（Spatial Prior Module）：轻量卷积下采样，提供 DINOv2(patch16) 拿不到的原生 stride-8 高频纹理
#   - SGM（Semantic Guiding Module）：取 DINOv2 最深语义特征生成空间注意力图，对 SPM 特征做语义门控
#   - 融合层：cat([语义, 门控纹理]) → Conv1×1+BN+GELU+Conv3×3+BN（照 SKYDET 实际 SGA 配方）
"""SGM 混合编码器分支。

结构细节与超参均照 SKYDET 实际代码（``engine/backbone/Semantic_Guiding_Adapter.py``），
融合层采用 SGA 原版 concat+conv 配方（而非 CFE 中的 RGM，RGM 留待 Phase 3 引入）。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class _ConvBnAct(nn.Module):
    """卷积 + BatchNorm + 可选激活 的基础块（照 SKYDET SGA，全部普通 Conv+BN）。

    归一化默认 BatchNorm2d（与 SKYDET 的 SyncBN 等价，单卡/小 batch 安全；
    多卡大 batch 训练时可换 SyncBN）。代码库 ``get_activation`` 不支持 GELU
    （见 projector.py），故此处内联 ``nn.GELU``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: str | None = "gelu",
    ) -> None:
        """初始化基础块。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            kernel_size: 卷积核大小，padding 自动对齐。
            stride: 卷积步长。
            act: 激活类型，"gelu" 用 GELU，其余（含 None）用恒等。
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act == "gelu" else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """前向：act(bn(conv(x)))。"""
        return cast(Tensor, self.act(self.bn(self.conv(x))))


class SpatialPriorModule(nn.Module):
    """空间先验分支：对原始图像逐级下采样，提供 stride-8/16/32 的原生高频纹理。

    结构照 SKYDET 的 ``SpatialPriorModulev2``（inplanes=16，全部普通 Conv+BN+GELU，无 CSP）：

        stem:  Conv3×3 s2 (3→16) + BN + GELU + MaxPool3×3 s2   → stride4
        conv2: Conv3×3 s2 (16→32) + BN                         → c2, stride8,  32ch
        conv3: GELU + Conv3×3 s2 (32→64) + BN                  → c3, stride16, 64ch
        conv4: GELU + Conv3×3 s2 (64→64) + BN                  → c4, stride32, 64ch

    返回 (c2, c3, c4)，参数量约 0.5M，相对 medium 总量可忽略。
    """

    def __init__(self, in_channels: int = 3) -> None:
        """初始化空间先验分支。

        Args:
            in_channels: 输入图像通道数，默认 3（RGB）。
        """
        super().__init__()
        # stride4：卷积 + MaxPool 各降一半
        self.stem = nn.Sequential(
            _ConvBnAct(in_channels, 16, kernel_size=3, stride=2, act="gelu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # stride8
        self.conv2 = _ConvBnAct(16, 32, kernel_size=3, stride=2, act=None)
        # 预激活顺序（先 GELU 后卷积）照 SKYDET 配方
        # stride16
        self.conv3 = nn.Sequential(nn.GELU(), _ConvBnAct(32, 64, kernel_size=3, stride=2, act=None))
        # stride32
        self.conv4 = nn.Sequential(nn.GELU(), _ConvBnAct(64, 64, kernel_size=3, stride=2, act=None))

    def forward(self, x: Tensor) -> list[Tensor]:
        """x: (B,3,H,W)，H/W 需被 4 整除（实际由 backbone 保证被 32 整除）。

        返回 (c2, c3, c4)，分辨率分别为 H/8、H/16、H/32，通道分别为 32、64、64。
        """
        h = self.stem(x)
        c2 = self.conv2(h)  # stride8
        c3 = self.conv3(c2)  # stride16
        c4 = self.conv4(c3)  # stride32
        return [c2, c3, c4]


class SemanticGuidingModule(nn.Module):
    """语义引导模块：取 DINOv2 最深特征作为唯一引导源，蒸馏多尺度空间注意力图。

    G_SGM = Conv3×3(sem→sem//4) + BN + GELU + Conv1×1(sem//4→1)，输出 1 通道 logits。
    对每个目标尺度：先上采样 logits 再 Sigmoid（顺序照实际代码），得到 [B,1,H_i,W_i] 注意力图。
    """

    def __init__(self, sem_channels: int) -> None:
        """初始化语义引导模块。

        Args:
            sem_channels: DINOv2 最深特征的通道数（small=384）。
        """
        super().__init__()
        self.attention_generator = nn.Sequential(
            nn.Conv2d(sem_channels, sem_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sem_channels // 4),
            nn.GELU(),
            nn.Conv2d(sem_channels // 4, 1, kernel_size=1),
        )

    def forward(self, sem_feat: Tensor, target_shapes: list[torch.Size]) -> list[Tensor]:
        """sem_feat: (B, C_sem, H/16, W/16)；返回与 target_shapes 对应的 [B,1,H_i,W_i] 注意力图。"""
        score = self.attention_generator(sem_feat)  # (B,1,H/16,W/16)
        maps: list[Tensor] = []
        for shape in target_shapes:
            m = F.interpolate(score, size=shape[2:], mode="bilinear", align_corners=False)
            maps.append(torch.sigmoid(m))  # 先上采样、后 Sigmoid
        return maps


class SGAEncoder(nn.Module):
    """SGM 混合编码器分支：SPM + SGM 门控 + concat 融合（照 SKYDET SGA 配方）。

    输入 projector 输出的 feats、DINOv2 原始多深度特征 raw_feats 与输入图像 x，
    返回与 feats 完全同构的融合后特征列表（通道数 hidden_dim、分辨率不变），
    从而对 decoder / mask / 位置编码零改动。
    """

    # P 级 ↔ SPM 输出 (c2, c3, c4) 的映射（patch16 下 stride 对齐：8/16/32）
    _LEVEL_MAP: dict[str, int] = {"P3": 0, "P4": 1, "P5": 2}
    _SPM_CHANNELS: tuple[int, int, int] = (32, 64, 64)

    def __init__(
        self,
        projector_scale: list[str],
        hidden_dim: int = 256,
        sem_channels: int = 384,
    ) -> None:
        """初始化 SGM 混合编码器分支。

        Args:
            projector_scale: 需要融合的 P 级列表（P3/P4/P5），与 decoder 的 num_feature_levels 对应。
            hidden_dim: 输出通道数（与 projector 输出对齐）。
            sem_channels: DINOv2 最深特征通道数（用于 SGM 注意力生成）。
        """
        super().__init__()
        unsupported = set(projector_scale) - set(self._LEVEL_MAP)
        if unsupported:
            raise ValueError(f"use_sga 暂只支持 P3/P4/P5 的融合，收到不支持的等级: {unsupported}")
        self.projector_scale = list(projector_scale)
        self.spm = SpatialPriorModule(in_channels=3)
        self.sgm = SemanticGuidingModule(sem_channels=sem_channels)
        # 每个待融合的 P 级：cat([feats[i](hidden_dim), 门控 SPM(spm_ch)]) → 融合块（1×1 → BN → GELU → 3×3 → BN）
        self.fusion_layers: nn.ModuleList = nn.ModuleList()
        for lvl in self.projector_scale:
            spm_ch = self._SPM_CHANNELS[self._LEVEL_MAP[lvl]]
            self.fusion_layers.append(
                nn.Sequential(
                    _ConvBnAct(hidden_dim + spm_ch, hidden_dim, kernel_size=1, act="gelu"),
                    _ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, act=None),
                )
            )

    def forward(self, feats: list[Tensor], raw_feats: list[Tensor], x: Tensor) -> list[Tensor]:
        """feats: 各 P 级 projector 输出（语义）；raw_feats: DINOv2 多深度特征；x: 输入图像 (B,3,H,W)。

        每级融合流程（对应 SKYDET SGA 公式 (2)(3)(4)）：
            ① SGM 语义门控：det' = det ⊙ M_att
            ② concat：cat([语义 feats[i], det'])
            ③ 融合层：Conv1×1+BN+GELU+Conv3×3+BN → hidden_dim

        语义来源用 feats[i]（projector 输出，已处目标尺度、复用预训练权重），
        替代 SKYDET 中"把 raw ViT 特征逐尺度插值对齐"的步骤——RF-DETR 特有的合理简化；
        门控源仍为 raw_feats[-1]（最深 DINOv2 特征）。
        """
        assert len(feats) == len(self.projector_scale) == len(self.fusion_layers)
        spm_outs = self.spm(x)  # (c2, c3, c4)
        # 为每个待融合的 P 级生成对应尺度的注意力图
        target_shapes = [spm_outs[self._LEVEL_MAP[lvl]].shape for lvl in self.projector_scale]
        attn_maps = self.sgm(raw_feats[-1], target_shapes)
        out: list[Tensor] = []
        for i, lvl in enumerate(self.projector_scale):
            det = spm_outs[self._LEVEL_MAP[lvl]]
            gated_det = det * attn_maps[i]  # ① 语义门控
            fused = self.fusion_layers[i](torch.cat([feats[i], gated_det], dim=1))  # ②③ concat 融合
            out.append(fused)
        return out
