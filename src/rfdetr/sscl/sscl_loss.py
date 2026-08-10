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

投影头（可选）：当构造时传入 ``projection_dim``，输入特征先经
ProjectionHead 映射到低维对比空间再计算损失，原型库同步建立在投影空间，
缓解对比压力对共享特征（同时喂给 class_embed 与 bbox_embed）的直接冲击。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.sscl.projection import ProjectionHead
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
        projection_dim: 投影头输出维度（对比空间维度）。为 ``None`` 时不启用
            投影头（保持原行为）；非 ``None`` 时把特征先投影到该低维空间再
            计算损失，原型库维度也改为 ``projection_dim``，且要求 ``hidden_dim``
            非 None（作为投影头输入维度）。
        prototype_instance_pos: 原型模式下是否加入同类别实例正样本。开启时
            正样本 = 本类原型 ∪ 同类别实例（对齐论文 Eq.9），用真实同类实例
            锚定冷启动（投影头随机初始化时给投影学习提供稳定的正向引力），
            负样本仍为全部有效原型（语义加权）。默认关闭保持原行为。
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
        projection_dim: int | None = None,
        prototype_instance_pos: bool = False,
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
        # 投影头：非 None 时启用，把特征映射到低维对比空间再算损失。
        # 需显式传入 hidden_dim 作为投影头输入维度（生产路径恒传，见 module_model）。
        self.projection_dim = projection_dim
        if projection_dim is not None:
            if hidden_dim is None:
                raise ValueError("启用投影头（projection_dim）时必须同时传入 hidden_dim 作为投影头输入维度。")
            self.projection_head = ProjectionHead(hidden_dim, projection_dim)
        # 原型模式：构造期决定是否创建类别原型库（不可运行期切换）
        self.prototype_mode = prototype_mode
        # 实例正样本开关：开启时原型模式正样本 = 本类原型 ∪ 同类别实例
        self.prototype_instance_pos = prototype_instance_pos
        if prototype_mode:
            # 延迟导入，避免与 prototype_bank 模块产生循环导入
            from rfdetr.sscl.prototype_bank import PrototypeBank

            self.prototype_bank = PrototypeBank(
                num_classes=semantic_matrix.shape[0],
                hidden_dim=projection_dim if projection_dim is not None else hidden_dim,
                momentum=prototype_momentum,
                min_samples=prototype_min_samples,
                sync_distributed=prototype_sync_ddp,
            )

    def _project(self, features: Tensor) -> Tensor:
        """把特征投影到对比空间；未启用投影头时恒等返回。

        Args:
            features: 输入特征 ``[*, in_dim]``（启用投影头时为 decoder hidden
                dim，未启用时为任意维度）。

        Returns:
            投影后的特征 ``[*, proj_dim]``；未启用投影头时返回原特征。
        """
        if self.projection_dim is None:
            return features
        return self.projection_head(features)

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
        hard_neg_features: Tensor | None = None,
    ) -> Tensor:
        """计算 SSCL 损失。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``，
                即 decoder 最后一层输出中与 GT 匹配的 query 特征。启用投影头时
                内部会先投影到对比空间再计算损失。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。
            hard_neg_features: 已 detach 的难例负样本特征 ``[K, hidden_dim]``
                （仅原型模式生效，实例模式忽略）。内部投影到同一对比空间后
                作为额外分母列追加（权重 1.0、无类别身份、无语义加权）。

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
        # 不能被提前拦截为零损失。原型模式在 _prototype_forward 内部自行投影。
        if self.prototype_mode:
            return self._prototype_forward(features, labels, hard_neg_features=hard_neg_features)
        # 实例模式：先投影到对比空间（未启用投影头时 _project 恒等）
        features = self._project(features)
        if num_fg < 2:
            # 少于 2 个前景样本时无法构成正负样本对，返回零损失。
            # 用投影后的特征保持计算图连接（利于 DDP 各参数收到梯度，含投影头）
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

    def _prototype_forward(
        self,
        features: Tensor,
        labels: Tensor,
        hard_neg_features: Tensor | None = None,
    ) -> Tensor:
        """原型锚定模式下的 SSCL 损失。

        正样本为 anchor 与本类原型的余弦相似度（本类原型有效则恒存在），
        负样本为 anchor 与全部有效类别原型的余弦相似度（按语义权重加权），
        从而每个 anchor 都有稳定的正负锚点，彻底摆脱 batch 内同类样本的构成。
        启用投影头时特征先投影到对比空间，原型库也建立在投影空间；
        启用实例正样本时正样本 = 本类原型 ∪ 同类别实例（对齐论文 Eq.9）；
        传入难例负样本时额外追加为分母列（权重 1.0），补齐"前景-背景边界"
        负样本维度。分子不变，故 ``loss >= 0`` 且 ``loss(含难例) >= loss(不含)``
        恒成立（分母单调增）。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。
            hard_neg_features: 已 detach 的难例负样本特征 ``[K, hidden_dim]``，
                为 ``None`` 或 K=0 时不追加（与基线行为完全一致）。

        Returns:
            标量 SSCL 损失。无有效原型或无有效 anchor 时返回零损失张量。
        """
        # 先投影到对比空间（未启用投影头时 _project 恒等），使正/负样本与
        # 原型同处一个几何空间；零损失返回沿用投影后的特征以保持图连接
        features = self._project(features)
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
        # 本类原型相似度（正样本之一）
        pos_logits = sim.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [N]

        # 难例负样本：追加为额外分母列（权重 1.0，无语义权重、无类别身份）。
        # 已 detach（难例方向不产生梯度，只推离 anchor）；K=0 时 no-op。
        # 在实例正样本分支之前追加，保证实例正样本也出现在难例路径的分母中；
        # 分母列序变为 [C 原型, K 难例, N 实例正样本]。
        if hard_neg_features is not None and hard_neg_features.shape[0] > 0:
            u_hn = F.normalize(self._project(hard_neg_features.detach()), dim=-1)  # [K, D]
            hn_sim = u @ u_hn.T / self.tau  # [N, K]
            # 数值安全：非有限（NaN/Inf）列置 -inf，等价于不参与分母 logsumexp
            finite = torch.isfinite(hn_sim).all(dim=0)  # [K]
            hn_sim = torch.where(
                finite.unsqueeze(0),
                hn_sim,
                torch.tensor(neg_inf, device=hn_sim.device),
            )
            denom_logits = torch.cat([denom_logits, hn_sim], dim=-1)  # [N, C+K]

        if self.prototype_instance_pos:
            # 同类别实例正样本：正样本 = 本类原型 ∪ 同类别实例（对齐论文 Eq.9）。
            # 真实同类实例从第一步提供"ground-truth"引力，锚定随机初始化投影头
            # 的冷启动，负样本仍为全部有效原型（语义加权）。
            sim_inst = u @ u.T / self.tau  # [N, N]
            self_identity = torch.eye(sim_inst.shape[0], dtype=torch.bool, device=sim_inst.device)
            same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_identity  # [N, N]
            pos_inst = sim_inst.masked_fill(~same_class, neg_inf)  # [N, N]
            # 分子 = logsumexp(本类原型, 同类别实例)；分母 = 原型项 + 实例正样本项。
            # 所有正项都在分母中（本类原型权重 1、实例正样本权重 1），保证 loss >= 0。
            num_logits = torch.cat([pos_logits.unsqueeze(-1), pos_inst], dim=-1)  # [N, 1+N]
            log_numerator = torch.logsumexp(num_logits, dim=-1)  # [N]
            denom_logits = torch.cat([denom_logits, pos_inst], dim=-1)  # [N, C+N]
            log_denominator = torch.logsumexp(denom_logits, dim=-1)  # [N]
        else:
            log_numerator = pos_logits  # [N]
            log_denominator = torch.logsumexp(denom_logits, dim=-1)  # [N]

        # 每个 anchor 的损失 = log(denom) - log(正样本) >= 0
        loss_per_anchor = log_denominator - log_numerator  # [N]

        loss = loss_per_anchor[anchor_mask].mean()
        return loss

    def hardness_stats(
        self,
        matched_features: Tensor,
        hard_neg_features: Tensor,
        random_features: Tensor | None = None,
    ) -> dict[str, float]:
        """难例硬度诊断：投影空间内三组特征与类别原型的平均余弦相似度。

        验证"难例是否真的硬"：难例比随机未匹配更贴近类别原型
        （``hn_vs_random_gap > 0``）、且比 matched 特征更贴（
        ``hn_vs_matched_gap > 0``）时，"难例代表像目标但不是目标的区域"
        的假设成立。全程 ``no_grad``，返回 CPU 标量字典，不产生损失。

        Args:
            matched_features: matched foreground query features ``[N_fg, D]``。
            hard_neg_features: 难例负样本特征 ``[K, D]``。
            random_features: 随机未匹配对照特征 ``[K', D]``（可选）。

        Returns:
            余弦统计字典（原始余弦，未除以温度，便于解读）：
            ``hn_proto_cos``/``matched_proto_cos`` 恒有；
            传入 ``random_features`` 时另有 ``random_proto_cos`` 与
            ``hn_vs_random_gap``/``hn_vs_matched_gap``。
            非原型模式、无有效原型或难例为空时返回空字典。
        """
        if not self.prototype_mode:
            return {}
        with torch.no_grad():
            proto_norm, valid = self.prototype_bank.get_normalized_prototypes()
            if not valid.any():
                return {}
            if hard_neg_features.shape[0] == 0:
                return {}

            def _mean_cos(feat: Tensor) -> float:
                u = F.normalize(self._project(feat), dim=-1)
                return float((u @ proto_norm.T).mean().item())

            stats = {
                "hn_proto_cos": _mean_cos(hard_neg_features),
                "matched_proto_cos": _mean_cos(matched_features),
            }
            if random_features is not None and random_features.shape[0] > 0:
                stats["random_proto_cos"] = _mean_cos(random_features)
                stats["hn_vs_random_gap"] = stats["hn_proto_cos"] - stats["random_proto_cos"]
                stats["hn_vs_matched_gap"] = stats["hn_proto_cos"] - stats["matched_proto_cos"]
            return stats

    def update_prototypes(self, features: Tensor, labels: Tensor) -> None:
        """更新类别原型库（仅原型模式生效，实例模式为 no-op）。

        启用投影头时先把特征投影到对比空间再更新，保证原型库与对比损失
        处于同一几何空间；未启用时恒等。

        注意：**只喂 matched features，绝不可传入难例特征**——难例刻画的是
        "背景/干扰分布"而非类别稳定中心，EMA 进原型库会污染类中心空间。

        Args:
            features: matched foreground query features ``[N_fg, hidden_dim]``
                （内部会 ``detach``，不参与反向传播）。
            labels: 每个 query 匹配到的 GT 类别标签 ``[N_fg]``。
        """
        if self.prototype_mode:
            self.prototype_bank.update(self._project(features), labels)
