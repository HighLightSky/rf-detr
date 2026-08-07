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

    def __init__(self, sem_channels: int, init_logit_bias: float = 0.0) -> None:
        """初始化语义引导模块。

        Args:
            sem_channels: DINOv2 最深特征的通道数（small=384）。
            init_logit_bias: 注意力 logits 的初值偏置（>0 使初始 sigmoid 注意力≈全通，
                防止早期就向「目标处抑制」方向收敛；P0 修复实验的 attn_bias 变体用 +2.0）。
        """
        super().__init__()
        self.attention_generator = nn.Sequential(
            nn.Conv2d(sem_channels, sem_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sem_channels // 4),
            nn.GELU(),
            nn.Conv2d(sem_channels // 4, 1, kernel_size=1),
        )
        if init_logit_bias != 0.0:
            # 末层 1×1 卷积自带 bias，直接置为初值偏置（只改初值、不增参数）
            final_conv = self.attention_generator[3]
            assert isinstance(final_conv, nn.Conv2d)
            final_conv.bias.data.fill_(init_logit_bias)

    def forward(self, sem_feat: Tensor, target_shapes: list[torch.Size]) -> list[Tensor]:
        """sem_feat: (B, C_sem, H/16, W/16)；返回与 target_shapes 对应的 [B,1,H_i,W_i] 注意力图。"""
        score = self.attention_generator(sem_feat)  # (B,1,H/16,W/16)
        maps: list[Tensor] = []
        for shape in target_shapes:
            m = F.interpolate(score, size=shape[2:], mode="bilinear", align_corners=False)
            maps.append(torch.sigmoid(m))  # 先上采样、后 Sigmoid
        return maps


class _SemanticFilmBlock(nn.Module):
    """语义条件残差调制块（文档 §3.2 公式）。

    用 DINO 语义特征生成有界的通道-空间调制（乘法范围 [0.5,1.5]），对 SPM 高频细节做
    有限幅度增强/减弱，再由可学习标量 α_s（初值 1e-3）通过残差路径接入 projector 语义基线，
    保证训练初期新分支几乎不改变预训练 DINO 主路径（预训练主路径近似不变，新分支逐步介入）。

        D = GN(Conv1x1(C_s))                       # C_s = SPM 输出（spm_ch → hidden_dim）
        S = GN(Conv1x1(V_s))                       # V_s = projector 特征（hidden_dim）
        G = 0.5 * tanh(Conv1x1(GELU(Conv3x3(S))))  # 通道-空间调制，乘法范围 [0.5,1.5]
        U = D * (1 + G)
        F = V_s + α_s * Conv3x3(GELU(U))           # 残差接入，α_s init=1e-3

    归一化统一用 GroupNorm（文档 §3.2：避免小 batch 下 SPM 的 BN 统计与 DINO 特征分布失配）。
    不启用 beta 加性偏移，避免语义分支直接生成与真实细节无关的伪纹理。
    """

    def __init__(self, spm_ch: int, hidden_dim: int, alpha_init: float = 1e-3, gn_groups: int = 32) -> None:
        """初始化语义条件残差调制块。

        Args:
            spm_ch: SPM 特征通道数（P3=32 / P4=64）。
            hidden_dim: 输出通道数（与 projector 语义特征对齐）。
            alpha_init: 可学习残差系数 α_s 初值（默认 1e-3）。
            gn_groups: GroupNorm 分组数（默认 32，hidden_dim=256 时每组 8 通道）。
        """
        super().__init__()
        self.d_proj = nn.Sequential(
            nn.Conv2d(spm_ch, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, hidden_dim),
        )
        self.s_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, hidden_dim),
        )
        self.g_mod = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.alpha = nn.Parameter(torch.full((), float(alpha_init)))

    def forward(self, v: Tensor, c: Tensor) -> Tensor:
        """按 §3.2 公式计算语义条件残差融合。

        Args:
            v: projector 语义特征（B, hidden_dim, H, W）。
            c: SPM 高频细节特征（B, spm_ch, H, W），与 v 同分辨率。

        Returns:
            融合后特征（B, hidden_dim, H, W）。
        """
        d = self.d_proj(c)
        s = self.s_proj(v)
        g = 0.5 * self.g_mod(s)  # 有界调制 [0.5, 1.5]
        u = d * (1.0 + g)
        return v + self.alpha * self.fuse(u)


class SGAEncoder(nn.Module):
    """SGM 混合编码器分支：SPM + SGM 门控 + concat 融合（照 SKYDET SGA 配方）。

    输入 projector 输出的 feats、DINOv2 原始多深度特征 raw_feats 与输入图像 x，
    返回与 feats 完全同构的融合后特征列表（通道数 hidden_dim、分辨率不变），
    从而对 decoder / mask / 位置编码零改动。

    融合方式（``fusion_mode``）：
        - ``concat``：原版 SGM 门控 + concat+conv 融合（SKYDET 配方）；
        - ``semantic_film``：语义条件残差调制（§3.2），无单通道 SGM 门控。
    """

    # P 级 ↔ SPM 输出 (c2, c3, c4) 的映射（patch16 下 stride 对齐：8/16/32）
    _LEVEL_MAP: dict[str, int] = {"P3": 0, "P4": 1, "P5": 2}
    _SPM_CHANNELS: tuple[int, int, int] = (32, 64, 64)

    # SGM 门控模式可选值（P0 修复实验）：
    #   product     ：det * M_att（原版，目标处可能被压到 0）
    #   lower_bound ：det * (0.5 + 0.5*M_att)，有效门控范围 [0.5,1]，目标处至少保留一半 SPM
    #   residual    ：det + det*M_att，保底更强（SPM 幅值放大至 1~2 倍）
    #   ones        ：det（忽略注意力，仅测 SPM 分支，用于消融）
    _GATE_MODES: frozenset[str] = frozenset({"product", "lower_bound", "residual", "ones"})
    # 融合方式可选值
    _FUSION_MODES: frozenset[str] = frozenset({"concat", "semantic_film"})

    def __init__(
        self,
        projector_scale: list[str],
        hidden_dim: int = 256,
        sem_channels: int = 384,
        gate_mode: str = "product",
        fusion_residual: bool = False,
        residual_gamma: float = 0.1,
        attn_bias: float = 0.0,
        fusion_mode: str = "concat",
        residual_alpha_init: float = 1e-3,
    ) -> None:
        """初始化 SGM 混合编码器分支。

        Args:
            projector_scale: 需要融合的 P 级列表（P3/P4/P5），与 decoder 的 num_feature_levels 对应。
            hidden_dim: 输出通道数（与 projector 输出对齐）。
            sem_channels: DINOv2 最深特征通道数（用于 SGM 注意力生成）。
            gate_mode: SGM 门控模式（product/lower_bound/residual/ones），见 ``_GATE_MODES``。
            fusion_residual: 是否用残差融合（fused = feats[i] + gamma*delta），保留 projector 语义基线。
            residual_gamma: 残差融合系数（默认 0.1，起步更稳）。
            attn_bias: SGM 注意力 logits 初值偏置（>0 使初始注意力≈全通，传给
                SemanticGuidingModule；默认 0.0 = 原版从 sigmoid(0)=0.5 起步）。
            fusion_mode: 融合方式（concat/semantic_film），见 ``_FUSION_MODES``。
            residual_alpha_init: semantic_film 各 P 级可学习残差系数 α_s 的初值（默认 1e-3）。
        """
        super().__init__()
        unsupported = set(projector_scale) - set(self._LEVEL_MAP)
        if unsupported:
            raise ValueError(f"use_sga 暂只支持 P3/P4/P5 的融合，收到不支持的等级: {unsupported}")
        if gate_mode not in self._GATE_MODES:
            raise ValueError(f"不支持的门控模式: {gate_mode}，可选: {sorted(self._GATE_MODES)}")
        if fusion_mode not in self._FUSION_MODES:
            raise ValueError(f"不支持的融合方式: {fusion_mode}，可选: {sorted(self._FUSION_MODES)}")
        self.projector_scale = list(projector_scale)
        self.gate_mode = gate_mode
        self.fusion_residual = fusion_residual
        self.residual_gamma = residual_gamma
        self.fusion_mode = fusion_mode
        self.residual_alpha_init = residual_alpha_init
        self.spm = SpatialPriorModule(in_channels=3)
        self.sgm = SemanticGuidingModule(sem_channels=sem_channels, init_logit_bias=attn_bias)
        # 每个待融合的 P 级，按融合方式构建对应模块：
        #   concat      ：cat([feats[i](hidden_dim), 门控 SPM(spm_ch)]) → 融合块（1×1 → BN → GELU → 3×3 → BN）
        #   semantic_film：§3.2 语义条件残差调制块（GN + 通道-空间调制 + 可学习残差 α_s）
        self.fusion_layers: nn.ModuleList = nn.ModuleList()
        self.film_blocks: nn.ModuleList = nn.ModuleList()
        for lvl in self.projector_scale:
            spm_ch = self._SPM_CHANNELS[self._LEVEL_MAP[lvl]]
            if fusion_mode == "semantic_film":
                self.film_blocks.append(_SemanticFilmBlock(spm_ch, hidden_dim, alpha_init=residual_alpha_init))
            else:
                self.fusion_layers.append(
                    nn.Sequential(
                        _ConvBnAct(hidden_dim + spm_ch, hidden_dim, kernel_size=1, act="gelu"),
                        _ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, act=None),
                    )
                )

    @staticmethod
    def _apply_gate(det: Tensor, m: Tensor, gate_mode: str) -> Tensor:
        """按门控模式把 SGM 注意力图作用到 SPM 特征上（P0 修复变体）。

        Args:
            det: SPM 空间先验特征（B, C_spm, H, W）。
            m: SGM 输出的 sigmoid 注意力图（B, 1, H, W），与 det 同分辨率。
            gate_mode: 门控模式，见 ``_GATE_MODES``。

        Returns:
            门控后的特征，形状与 det 相同。

        Raises:
            ValueError: gate_mode 不在 ``_GATE_MODES`` 中。
        """
        if gate_mode == "product":
            return det * m
        if gate_mode == "lower_bound":
            # 下界门控：即使注意力在目标处接近 0，也至少保留一半 SPM 细节
            return det * (0.5 + 0.5 * m)
        if gate_mode == "residual":
            # 残差门控：SPM 幅值放大至 1~2 倍，保底更强
            return det + det * m
        if gate_mode == "ones":
            # SPM-only 消融：忽略注意力，全量保留 SPM
            return det
        raise ValueError(f"不支持的门控模式: {gate_mode}，可选: {sorted(SGAEncoder._GATE_MODES)}")

    def forward(self, feats: list[Tensor], raw_feats: list[Tensor], x: Tensor) -> list[Tensor]:
        """feats: 各 P 级 projector 输出（语义）；raw_feats: DINOv2 多深度特征；x: 输入图像 (B,3,H,W)。

        每级融合流程（由 ``fusion_mode`` 决定）：
            concat（原版，SKYDET SGA 公式 (2)(3)(4)）：
                ① SGM 语义门控：det' = gate(det, M_att)（模式由 ``gate_mode`` 决定）
                ② concat：cat([语义 feats[i], det'])
                ③ 融合层：Conv1×1+BN+GELU+Conv3×3+BN → delta
                ④ （可选）残差融合：fused = feats[i] + gamma*delta，保底原始语义特征
            semantic_film（§3.2）：
                无单通道 SGM；对每个 P 级用 _SemanticFilmBlock 做有界语义调制 + 残差接入。

        语义来源用 feats[i]（projector 输出，已处目标尺度、复用预训练权重），
        替代 SKYDET 中"把 raw ViT 特征逐尺度插值对齐"的步骤——RF-DETR 特有的合理简化。
        """
        assert len(feats) == len(self.projector_scale)
        spm_outs = self.spm(x)  # (c2, c3, c4)
        out: list[Tensor] = []

        if self.fusion_mode == "semantic_film":
            # 无单通道 SGM：DINO 语义通过各 film block 的 S_s 路径做有界调制，SPM 细节不被硬门控
            for i, lvl in enumerate(self.projector_scale):
                det = spm_outs[self._LEVEL_MAP[lvl]]
                out.append(self.film_blocks[i](feats[i], det))
            return out

        # ── concat 模式（原版行为，含 SGM 门控）────────────────────────────
        # 为每个待融合的 P 级生成对应尺度的注意力图
        target_shapes = [spm_outs[self._LEVEL_MAP[lvl]].shape for lvl in self.projector_scale]
        # 所有门控模式都调用 SGM，保证注意力 hook（analyze_sga.py）仍能收集注意力图
        attn_maps = self.sgm(raw_feats[-1], target_shapes)
        for i, lvl in enumerate(self.projector_scale):
            det = spm_outs[self._LEVEL_MAP[lvl]]
            gated_det = self._apply_gate(det, attn_maps[i], self.gate_mode)  # ① 语义门控（变体）
            delta = self.fusion_layers[i](torch.cat([feats[i], gated_det], dim=1))  # ②③ concat 融合
            # ④ 残差融合：保留 projector 语义基线，SGA 只学增量；默认关闭与原版行为一致
            out.append(feats[i] + self.residual_gamma * delta if self.fusion_residual else delta)
        return out
