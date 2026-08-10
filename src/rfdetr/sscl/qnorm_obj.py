# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""QNorm-Obj + EUMix：query 范数物体性门控与熵感知校准（EW-DETR 闭集适配）。

依据论文 EW-DETR（docs/参考论文/EW-DETR.md §3.4-3.5、Appendix A）：

- **QNorm-Obj**（论文 Eq.7-10）：DETR 解码器 query 中，匹配到真实物体的 query
  特征 L2 范数显著高于背景 query（Appendix C）。本模块把"方向（语义）"与
  "幅度（物体性）"解耦：
    ① 特征混合：h_cls = (1-α_mix)·h + α_mix·LN(h)/‖LN(h)‖₂ 后重算分类 logits；
    ② 物体性头：z_obj = f_obj(‖h‖₂)/τ，obj = σ(z_obj)。
- **物体性门控**（本方案的闭集用法）：z_known *= obj，只乘前景类列、背景列
  不动——高分类分但低物体性的背景框被压向背景，无任何辅助监督损失。
- **EUMix**（论文 Eq.16-23 的闭集适配：背景类扮演 unknown 角色）：
    ③ p_max = max σ(z_known)；g = (1-p_max)^γ（γ=softplus(θ_γ)，Eq.17）；
       p_obj_bg = obj·g（物体性驱动的背景证据，Eq.18）；
       闭集适配的 **logit 空间混合**（替代论文 Eq.19-22 的概率空间混合，原因见
       _calibrate_layer 内注释）：z_bg_new = (1-α)·(z_bg + b_obj) + α·logit(p_obj_bg)，
       α=σ(θ_α) 初始 0.1 → 起点 z_bg_new ≈ z_bg（恒等起步，可学习上升）；
       z_known -= λ·p_obj_bg（前景软抑制，论文 §3.5 末段）。

全部参数通过标准检测损失隐式训练（matched query 需要高 logit → 门必须高），
不引入任何辅助损失。近恒等初始化（gate≈0.98、α_mix=0）保证起点行为≈基线，
不扰动 Hungarian matching。

设计模式与 :class:`rfdetr.sscl.semantic_head.SemanticResidual` 一致：
独立 nn.Module 挂到模型（``model.qnorm_obj``），在 lwdetr 前向中
``outputs_class = class_embed(hs)`` 之后调用，逐 decoder 层统一校准
（含 aux 层）；只改 logits、不碰 ``hs``，与 SSCL 损失（读 hs）正交。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.config import TrainConfig

logger = get_logger()

# logit 反变换的数值保护区间（p 被 clamp 到该区间后再取 logit，避免 log(0)/log(1)）
_LOGIT_EPS = 1e-6


def _softplus_inv(y: float) -> float:
    """softplus 的反函数 ``softplus⁻¹(y) = log(e^y − 1)``。

    Args:
        y: 目标有效值（须 > 0）。

    Returns:
        softplus 反函数值，作为参数初始值。
    """
    return math.log(math.expm1(y))


def _logit(p: Tensor) -> Tensor:
    """logit 反变换 ``logit(p) = log(p/(1−p))``（p 须在 (0,1) 内）。

    Args:
        p: 概率张量。

    Returns:
        对应 logit 张量。
    """
    return torch.log(p / (1.0 - p))


class QNormObjectness(nn.Module):
    """QNorm-Obj + EUMix 模块（在线运行，无需离线产物）。

    Args:
        hidden_dim: decoder hidden dim d。
        num_classes: 前景类数 C（不含末列 background）。
        obj_hidden_dim: 物体性头隐藏维度。
        tau: 物体性头温度 τ。
        feature_mix: 是否启用特征混合（论文 Eq.7-9）。
        gate: 是否启用物体性门控（z_known *= σ(z_obj)）。
        eumix: 是否启用熵感知校准（论文 Eq.16-23）。
        gamma_init: 熵缺口指数 γ 的初始有效值（θ_γ 初始化为 softplus⁻¹(该值)）。
        alpha_init: EUMix 背景混合权重 α 的初始**有效值**（θ_α 初始化为 logit⁻¹(该值)）。
            须在 (0,1) 内；默认 0.1 → 起点 z_bg_new ≈ z_bg（恒等起步，训练中可上升）。
        lambda_init: 前景软抑制强度 λ 初始值。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        obj_hidden_dim: int = 64,
        tau: float = 2.0,
        feature_mix: bool = True,
        gate: bool = True,
        eumix: bool = True,
        gamma_init: float = 1.0,
        alpha_init: float = 0.1,
        lambda_init: float = 0.5,
    ) -> None:
        super().__init__()
        if not (feature_mix or gate or eumix):
            raise ValueError("QNormObjectness 至少启用 feature_mix/gate/eumix 中的一个。")
        if gamma_init <= 0.0:
            raise ValueError(f"gamma_init 必须 > 0，当前为 {gamma_init}。")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError(f"alpha_init 必须在 (0,1) 内，当前为 {alpha_init}。")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.tau = tau
        self.feature_mix = feature_mix
        self.gate = gate
        self.eumix = eumix

        # --- QNorm-Obj 组件 ---
        # 特征混合系数 α_mix（clamp [0,1]；初始 0 → h_cls ≡ h，起点与基线一致）
        self.alpha_mix = nn.Parameter(torch.zeros(()))
        # 模块内独立 LayerNorm（不碰 decoder 的最终 norm，互不影响）
        self.ln = nn.LayerNorm(hidden_dim)
        # 物体性头 f_obj：输入为 query 特征 L2 范数（标量，论文 Eq.10）
        self.obj_head = nn.Sequential(
            nn.Linear(1, obj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obj_hidden_dim, 1),
        )
        # 初始 bias = 4·τ → z_obj 初始 ≈ 4 → σ(z_obj) ≈ 0.98，门近恒等。
        # Linear 默认初始化下 weight 贡献远小于 bias，起步时门≈1 不扰动匹配。
        with torch.no_grad():
            self.obj_head[-1].bias.fill_(4.0 * tau)

        # --- EUMix 标量参数（论文 Eq.17/19/20 + 前景软抑制）---
        self.theta_gamma = nn.Parameter(torch.tensor(_softplus_inv(gamma_init)))
        # θ_α 初始化为 logit⁻¹(alpha_init)：有效 α = σ(θ_α) = alpha_init
        self.theta_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))
        self.b_obj = nn.Parameter(torch.zeros(()))
        self.theta_lambda = nn.Parameter(torch.tensor(lambda_init))

        # 最近一次前向的统计信息（detach，供训练监控读取，不驻留计算图）
        self.last_stats: dict[str, Any] = {}

    def _calibrate_layer(self, h: Tensor, z: Tensor, w: Tensor, b: Tensor, c: int) -> tuple[Tensor, dict[str, Any]]:
        """对单层 decoder 输出施加 QNorm-Obj + EUMix 校准。

        Args:
            h: 该层 hidden states ``[B, Q, d]``。
            z: 该层分类 logits ``[B, Q, C+1]``（末列 background）。
            w: ``class_embed.weight`` ``[C+1, d]``（引用，前向传入）。
            b: ``class_embed.bias`` ``[C+1]``。
            c: 前景类数。

        Returns:
            ``(z_calibrated, stats)`` 二元组：
            - z_calibrated: 校准后的 logits ``[B, Q, C+1]``。
            - stats: 该层监控统计（全部 detach）。
        """
        # --- QNorm-Obj：范数与方向解耦 ---
        h_ln = self.ln(h)
        norm = h_ln.norm(dim=-1, keepdim=True)  # [B, Q, 1]，LN 后的范数（去维度尺度差异）
        # 物体性 logit（论文 Eq.10）：f_obj(‖h‖₂)/τ
        z_obj = self.obj_head(norm) / self.tau  # [B, Q, 1]
        obj = torch.sigmoid(z_obj).squeeze(-1)  # [B, Q]，物体性分数

        if self.feature_mix:
            # 特征混合（论文 Eq.7-9）：方向向量 + 凸组合，重算分类 logits
            alpha_mix = self.alpha_mix.clamp(min=0.0, max=1.0)
            h_norm = h_ln / (norm + _LOGIT_EPS)  # 单位方向向量（Eq.7）
            h_cls = (1.0 - alpha_mix) * h + alpha_mix * h_norm  # Eq.8
            z = h_cls @ w.T + b  # Eq.9
        else:
            alpha_mix = self.alpha_mix.detach().clamp(min=0.0, max=1.0)

        z_known = z[..., :c]
        z_bg = z[..., c : c + 1]

        # --- 物体性门控：只乘前景类列，背景列不动 ---
        if self.gate:
            z_known = z_known * obj.unsqueeze(-1)

        # --- EUMix 熵感知校准（闭集适配：背景列扮演 unknown）---
        p_max = torch.zeros_like(obj)
        p_obj_bg = torch.zeros_like(obj)
        if self.eumix:
            gamma = F.softplus(self.theta_gamma)  # Eq.17 中的可学习 γ
            alpha = torch.sigmoid(self.theta_alpha)  # Eq.20 混合权重
            lam = self.theta_lambda.clamp(min=0.0)  # 前景软抑制强度
            # 已知类最大置信度（Eq.16）
            p_known = torch.sigmoid(z_known)
            p_max = p_known.max(dim=-1).values  # [B, Q]
            # 校准缺口（Eq.17）：已知类全不确定时 g→1，某类自信时 g→0
            g = (1.0 - p_max).clamp(min=0.0) ** gamma
            # 物体性驱动的背景证据（Eq.18）："有东西但说不清是啥"→背景
            p_obj_bg = (obj if self.gate else torch.ones_like(obj)) * g
            # 闭集适配的 logit 空间混合（替代论文 Eq.19-22 的概率空间混合）：
            # 开集场景 unknown logit 从未受过监督、概率空间混合可自由塑形；闭集
            # 场景背景列是训练过的、logits 已校准，概率空间混合会把极端 z_bg
            # 硬压缩到 ≈logit(α)（Eq.22 的 logit 逆变换非线性所致），起步即扰动
            # 预训练行为。改为 logit 空间凸组合：
            #   分类器驱动的背景估计 = z_bg + b_obj（Eq.19 的 logit 形式，精确可逆）
            #   物体性驱动的背景估计 = logit(p_obj_bg)
            #   z_bg_new = (1-α)·(z_bg + b_obj) + α·logit(p_obj_bg)
            # α 初始很小（0.1）→ 起点 z_bg_new ≈ z_bg（精确恒等起步），训练中 α
            # 可学习上升，让物体性证据按损失信号参与背景校准。
            z_obj_bg = _logit(p_obj_bg.clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS))
            z_bg_new = ((1.0 - alpha) * (z_bg.squeeze(-1) + self.b_obj) + alpha * z_obj_bg).unsqueeze(-1)
            # 前景软抑制（论文 §3.5 末段）：物体性高但已知类不确定 → 压低前景
            z_known = z_known - lam * p_obj_bg.unsqueeze(-1)
        else:
            z_bg_new = z_bg

        stats = {
            "obj": obj.detach(),
            "p_max": p_max.detach(),
            "p_obj_bg": p_obj_bg.detach(),
        }
        return torch.cat([z_known, z_bg_new], dim=-1), stats

    def forward(self, hs: Tensor, z_cls: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """对全层分类 logits 栈施加 QNorm-Obj + EUMix 校准。

        Args:
            hs: decoder hidden states 全层栈 ``[L, B, Q, d]``。
            z_cls: 分类 logits 全层栈 ``[L, B, Q, C+1]``（末列 background）。
            w: ``class_embed.weight`` ``[C+1, d]``（引用，前向传入）。
            b: ``class_embed.bias`` ``[C+1]``。

        Returns:
            ``(z_calibrated, stats)`` 二元组：
            - z_calibrated: 校准后的 logits 全层栈，形状与 ``z_cls`` 一致。
            - stats: 监控统计字典（全部 detach），含标量参数与最后一层逐 query 量。
        """
        c = self.num_classes
        z_layers = []
        for l_idx in range(z_cls.shape[0]):
            z_l, layer_stats = self._calibrate_layer(hs[l_idx], z_cls[l_idx], w, b, c)
            z_layers.append(z_l)
        z_out = torch.stack(z_layers, dim=0)

        # 汇总统计（全部 detach，供监控读取）
        alpha_mix = self.alpha_mix.detach().clamp(min=0.0, max=1.0)
        gamma = F.softplus(self.theta_gamma).detach()
        alpha = torch.sigmoid(self.theta_alpha).detach()
        lam = self.theta_lambda.detach().clamp(min=0.0)
        self.last_stats = {
            **layer_stats,
            "alpha_mix": alpha_mix,
            "alpha": alpha,
            "gamma": gamma,
            "lambda_suppress": lam,
            "b_obj": self.b_obj.detach(),
        }
        return z_out, self.last_stats

    @classmethod
    def build(cls, cfg: "TrainConfig", num_classes: int, hidden_dim: int) -> "QNormObjectness":
        """从训练配置装配 QNorm-Obj + EUMix 模块。

        Args:
            cfg: 训练配置（``qnorm_obj_*`` 字段）。
            num_classes: 前景类数（不含 background 列）。
            hidden_dim: decoder hidden dim。

        Returns:
            装配完成的 ``QNormObjectness`` 实例。

        Raises:
            ValueError: 当 qnorm_obj_enabled 为 False 时抛出（防御性校验）。
        """
        if not cfg.qnorm_obj_enabled:
            raise ValueError("启用 QNorm-Obj 时必须设置 qnorm_obj_enabled=True。")
        head = cls(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            obj_hidden_dim=int(cfg.qnorm_obj_obj_hidden_dim),
            tau=float(cfg.qnorm_obj_tau),
            feature_mix=bool(cfg.qnorm_obj_feature_mix),
            gate=bool(cfg.qnorm_obj_gate),
            eumix=bool(cfg.qnorm_obj_eumix),
            gamma_init=float(cfg.qnorm_obj_gamma_init),
            alpha_init=float(cfg.qnorm_obj_alpha_init),
            lambda_init=float(cfg.qnorm_obj_lambda_init),
        )
        return head

    def describe_freeze(self) -> str:
        """返回子开关与关键超参的摘要字符串（用于装配日志）。

        Returns:
            人类可读的配置摘要。
        """
        return (
            f"feature_mix={self.feature_mix}, gate={self.gate}, eumix={self.eumix}, "
            f"τ={self.tau}, obj_head_hidden={self.obj_head[0].out_features}, "
            f"初始有效 γ=softplus(θ_γ)≈{F.softplus(self.theta_gamma).item():.3f}, "
            f"α=σ(θ_α)≈{torch.sigmoid(self.theta_alpha).item():.3f}"
        )
