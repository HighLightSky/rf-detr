# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL（语义相关驱动监督对比学习）损失。

SSCL 在 RF-DETR decoder 最后一层输出的 matched foreground query features
上施加类别分离约束。与普通监督对比学习的区别在于：对语义相似（易混淆）
的不同类别负样本，根据 CLIP 类别语义相似度矩阵赋予更强的分离权重，
使容易混淆的类别在 query feature 空间中被拉得更开。

损失公式（对归一化特征 u = h / ||h||）：

    L_SSCL = -1/|A| sum_{i in A} log(
        sum_{j in P(i)} exp(u_i^T u_j / tau)
        /
        (sum_{j in P(i)} exp(u_i^T u_j / tau)
         + sum_{j in N(i)} exp(w_ij * u_i^T u_j / tau))
    )

其中 P(i)/N(i) 分别为与 anchor i 同类别/异类别的样本集合，
w_ij = clamp(1 + rho * S[y_i, y_j], 1, omega_max) 为语义权重，
S 为 CLIP 类别语义相似度矩阵，rho 控制放大强度，omega_max 为权重上限。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class SSCLLoss(nn.Module):
    """语义相关性引导的监督对比学习损失。

    Args:
        semantic_matrix: CLIP 类别语义相似度矩阵 ``[C, C]``，
            对角线为 1，值为余弦相似度（约在 ``[-1, 1]``）。
        tau: 对比学习温度系数，越小对相似度越敏感。
        rho: 语义先验对负样本强度的放大系数。
        omega_max: 负样本语义权重上限，避免训练不稳定。
        anchor_classes: 参与对比的 anchor 类别索引列表。为 ``None`` 时
            使用全部类别作为 anchor。
        confusing_classes: 易混负样本类别索引列表。为 ``None`` 时对所有
            异类负样本施加语义权重；指定时仅对属于这些类别的负样本施加
            语义权重，其余负样本权重保持 1.0（普通负样本）。
        class_names: 类别名称列表（可选），用于日志输出。
    """

    def __init__(
        self,
        semantic_matrix: Tensor,
        tau: float = 0.1,
        rho: float = 0.3,
        omega_max: float = 2.0,
        anchor_classes: list[int] | None = None,
        confusing_classes: list[int] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        # 注册为非可训练 buffer，随模型迁移设备并在 checkpoint 中保存
        self.register_buffer("semantic_matrix", semantic_matrix.float().clone())
        self.tau = tau
        self.rho = rho
        self.omega_max = omega_max
        self.anchor_classes = anchor_classes
        self.confusing_classes = confusing_classes
        self.class_names = class_names

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """计算 SSCL 损失。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``，
                即 decoder 最后一层输出中与 GT 匹配的 query 特征。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。

        Returns:
            标量 SSCL 损失。当没有有效 anchor（如 batch 内正样本不足）时
            返回 0 损失的张量，不产生梯度。

        Raises:
            ValueError: 当 ``features`` 与 ``labels`` 长度不一致时抛出。
        """
        num_fg = features.shape[0]
        if num_fg < 2:
            # 少于 2 个前景样本时无法构成正负样本对，返回零损失
            return features.sum() * 0.0
        if labels.shape[0] != num_fg:
            raise ValueError(f"features 与 labels 数量不一致: {num_fg} vs {labels.shape[0]}")

        # 归一化特征并计算余弦相似度（带温度）
        u = F.normalize(features, dim=-1)
        sim = u @ u.T / self.tau  # [N, N]

        # 同类别掩码（排除自身），用于区分正负样本
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        self_identity = torch.eye(num_fg, dtype=torch.bool, device=labels.device)
        same_class = same_class & ~self_identity

        # 构建负样本语义权重矩阵 w_ij = clamp(1 + rho * S[y_i, y_j], 1, omega_max)
        # semantic_matrix[labels] 形状 [N, C]，再按 labels 索引列得到 [N, N]
        pair_sem = self.semantic_matrix[labels][:, labels]  # [N, N] S[y_i, y_j]
        neg_weight = 1.0 + self.rho * pair_sem

        if self.confusing_classes is not None:
            # 仅对易混负样本（类别在 confusing_classes 中）施加语义放大，
            # 其余负样本保持权重 1.0（普通负样本，不放大分离强度）
            confusing_mask = torch.as_tensor(
                [label.item() in self.confusing_classes for label in labels],
                dtype=torch.bool,
                device=labels.device,
            )
            confusing_pair = confusing_mask.unsqueeze(0) & ~same_class  # [N, N]
            neg_weight = torch.where(
                confusing_pair,
                neg_weight,
                torch.ones_like(neg_weight),
            )

        neg_weight = neg_weight.clamp(min=1.0, max=self.omega_max)
        # 正样本对权重恒为 1（不放大同类吸引力），自身忽略
        weight = torch.where(same_class, torch.ones_like(neg_weight), neg_weight)
        weight = weight.masked_fill(self_identity, 0.0)

        # 构造加权 logits：正样本为 sim，负样本为 w * sim
        logits = sim * weight  # [N, N]

        # 数值稳定的 logsumexp 形式
        neg_inf = torch.finfo(logits.dtype).min
        pos_logits = torch.where(same_class, logits, torch.tensor(neg_inf, device=logits.device))
        denom_logits = torch.where(~self_identity, logits, torch.tensor(neg_inf, device=logits.device))

        log_numerator = torch.logsumexp(pos_logits, dim=1)  # [N] log sum_pos exp(sim)
        log_denominator = torch.logsumexp(denom_logits, dim=1)  # [N] log(sum_pos + sum_neg w*sim)

        # 每个 anchor 的损失 = log_denominator - log_numerator
        loss_per_anchor = log_denominator - log_numerator  # [N]

        # anchor 过滤：仅 anchor_classes 中且存在至少一个同类正样本的样本
        if self.anchor_classes is not None:
            anchor_mask = torch.as_tensor(
                [label.item() in self.anchor_classes for label in labels],
                dtype=torch.bool,
                device=labels.device,
            )
        else:
            anchor_mask = torch.ones(num_fg, dtype=torch.bool, device=labels.device)
        anchor_mask = anchor_mask & same_class.any(dim=1)

        if not anchor_mask.any():
            # 没有有效 anchor（batch 内同类正样本不足），返回零损失。
            # 注意不能使用 loss_per_anchor.sum() * 0.0：当某 anchor 无正样本时
            # log_numerator 为 -inf，inf * 0 = nan。
            # 使用 features.sum() * 0.0 保持计算图连接（利于 DDP 各参数收到梯度）
            if self.anchor_classes is not None and self.class_names is not None:
                anchor_names = [self.class_names[c] for c in self.anchor_classes if c < len(self.class_names)]
                logger.debug(f"SSCL 当前 batch 无有效 anchor（类别: {anchor_names}），损失为 0")
            return features.sum() * 0.0

        loss = loss_per_anchor[anchor_mask].mean()
        return loss
