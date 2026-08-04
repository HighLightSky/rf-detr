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
        prototype_mode: 是否启用类别原型锚定模式（构造期决定，不可运行期
            切换）。开启时正样本为本类原型、负样本为全部类别原型，每个样本
            恒有正负锚点，摆脱 batch 内同类样本不足导致的零损失问题；默认
            ``False`` 使用实例对实例模式（原逻辑，零改动保留）。
        prototype_momentum: 原型模式下的 EMA 更新系数
            ``p <- m*p + (1-m)*batch_mean``。
        prototype_min_samples: 原型模式下单次 batch 中某类样本数低于该
            阈值时跳过该类原型更新（防噪声）。
        prototype_sync_ddp: 是否在 DDP 多卡时先 ``all_gather`` 各 rank 特征
            再更新原型，保证各 rank 原型一致（单卡无需，默认关闭）。
        hidden_dim: 原型特征维度。为 ``None`` 时首次更新惰性确定维度
            （生产环境建议传入，见 ``PrototypeBank``）。
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
        prototype_mode: bool = False,
        prototype_momentum: float = 0.99,
        prototype_min_samples: int = 1,
        prototype_sync_ddp: bool = False,
        hidden_dim: int | None = None,
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
        # 原型模式：构造期决定是否创建类别原型库（不可运行期切换）
        self.prototype_mode = prototype_mode
        if prototype_mode:
            # 延迟导入，避免与 prototype_bank 模块产生循环导入
            from rfdetr.sscl.prototype_bank import PrototypeBank

            self.prototype_bank = PrototypeBank(
                num_classes=semantic_matrix.shape[0],
                hidden_dim=hidden_dim,
                momentum=prototype_momentum,
                min_samples=prototype_min_samples,
                sync_distributed=prototype_sync_ddp,
            )

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
        if labels.shape[0] != num_fg:
            raise ValueError(f"features 与 labels 数量不一致: {num_fg} vs {labels.shape[0]}")
        # 原型模式 dispatch 必须在实例模式的 num_fg < 2 零损失判断之前：
        # 原型模式下"每类仅 1 个样本"恰恰是要规避 batch 影响的核心场景，
        # 不能被提前拦截为零损失。
        if self.prototype_mode:
            return self._prototype_forward(features, labels)
        if num_fg < 2:
            # 少于 2 个前景样本时无法构成正负样本对，返回零损失
            return features.sum() * 0.0

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

    def _prototype_forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """原型锚定模式下的 SSCL 损失。

        正样本为 anchor 与本类原型的余弦相似度（本类原型有效则恒存在），
        负样本为 anchor 与全部有效类别原型的余弦相似度（按语义权重加权），
        从而每个 anchor 都有稳定的正负锚点，彻底摆脱 batch 内同类样本的构成。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。

        Returns:
            标量 SSCL 损失。无有效原型或无有效 anchor 时返回零损失张量。
        """
        num_fg = features.shape[0]
        if num_fg == 0:
            return features.sum() * 0.0

        # 获取归一化原型与有效掩码（本类原型无效的 anchor 不参与损失）
        proto_norm, valid_proto = self.prototype_bank.get_normalized_prototypes()
        if not valid_proto.any():
            # 尚无任何有效原型，返回零损失并保持图连接（利于 DDP）
            return features.sum() * 0.0

        # anchor 过滤：anchor_classes 约束 ∩ 本类原型有效
        if self.anchor_classes is not None:
            anchor_mask = torch.as_tensor(
                [label.item() in self.anchor_classes for label in labels],
                dtype=torch.bool,
                device=labels.device,
            )
        else:
            anchor_mask = torch.ones(num_fg, dtype=torch.bool, device=labels.device)
        anchor_mask = anchor_mask & valid_proto[labels]

        if not anchor_mask.any():
            # 无有效 anchor（anchor 类原型尚未建立），返回零损失并保持图连接
            return features.sum() * 0.0

        # 归一化特征并计算与全部类别原型的余弦相似度（带温度）
        u = F.normalize(features, dim=-1)
        sim = u @ proto_norm.T / self.tau  # [N, C]

        # 语义权重矩阵 w_ic = clamp(1 + rho * S[y_i, c], 1, omega_max)
        # semantic_matrix[labels] 形状 [N, C]，即 S[y_i, c]
        pair_sem = self.semantic_matrix[labels]
        weight = 1.0 + self.rho * pair_sem

        if self.confusing_classes is not None:
            # 仅对易混类别的原型列施加语义放大，其余列保持权重 1.0
            confusing_col = torch.as_tensor(
                [c in self.confusing_classes for c in range(proto_norm.shape[0])],
                dtype=torch.bool,
                device=labels.device,
            )
            weight = torch.where(
                confusing_col.unsqueeze(0),
                weight,
                torch.ones_like(weight),
            )

        weight = weight.clamp(min=1.0, max=self.omega_max)
        # 正样本（本类原型）列权重恒为 1，确保分母中的正项 = 分子，loss >= 0
        pos_mask = F.one_hot(labels, num_classes=proto_norm.shape[0]).bool()
        weight = torch.where(pos_mask, torch.ones_like(weight), weight)

        # 数值稳定的 logsumexp：无效原型列置 -inf，不参与分母
        neg_inf = torch.finfo(sim.dtype).min
        denom_logits = torch.where(
            valid_proto.unsqueeze(0),
            sim * weight,
            torch.tensor(neg_inf, device=sim.device),
        )
        pos_logits = sim.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [N]

        log_denominator = torch.logsumexp(denom_logits, dim=-1)  # [N]
        # 每个 anchor 的损失 = log(denom) - logit(本类原型) = log(1 + Σ_neg/pos) >= 0
        loss_per_anchor = log_denominator - pos_logits  # [N]

        loss = loss_per_anchor[anchor_mask].mean()
        return loss

    def update_prototypes(self, features: Tensor, labels: Tensor) -> None:
        """更新类别原型库（仅原型模式生效，实例模式为 no-op）。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``
                （内部会 ``detach``，不参与反向传播）。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。
        """
        if self.prototype_mode:
            self.prototype_bank.update(features, labels)
