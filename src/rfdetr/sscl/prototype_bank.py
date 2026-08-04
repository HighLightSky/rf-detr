# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""类别原型库（class prototype bank）。

以 EMA 统计量维护每个类别的特征原型，供原型锚定 SSCL 损失使用：
- 正样本 = anchor 与本类原型的余弦相似度（只要有本类原型就恒存在）；
- 负样本 = anchor 与全部类别原型的余弦相似度（按语义权重加权）。
从而彻底摆脱 batch 内同类正样本是否凑齐带来的零损失问题。

原型以 ``register_buffer`` 保存（随模型迁移设备并写入 checkpoint），无梯度，
更新在 ``torch.no_grad()`` 内完成，不参与反向传播。原型的"训练"本质是
随特征分布被动滑动的统计量，而非可学习的参数。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class PrototypeBank(nn.Module):
    """类别特征原型库（EMA 统计量，无梯度）。

    Args:
        num_classes: 类别总数。
        hidden_dim: 特征维度。为 ``None`` 时惰性初始化：首次 ``update``
            时按输入特征形状确定维度（仅建议测试或未知维度场景使用，
            生产环境应显式传入维度以避免惰性 ``resize_`` 与编译/续训交互）。
        momentum: EMA 更新系数 ``p <- m*p + (1-m)*batch_mean``。
        min_samples: 单次 batch 中某类样本数低于该阈值时跳过该类更新
            （默认 1，使少样本场景首个样本即建立原型）。
        sync_distributed: 是否在分布式多卡时先 ``all_gather`` 各 rank 的
            ``(features, labels)`` 再聚合更新，保证各 rank 原型一致
            （``register_buffer`` 不会被 DDP 自动同步，多卡时需显式开启）。

    Attributes:
        prototypes: 原型矩阵 ``[C, D]``。
        num_updates: 每类的更新次数 ``[C]``（``> 0`` 表示已初始化）。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int | None = None,
        momentum: float = 0.99,
        min_samples: int = 1,
        sync_distributed: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.min_samples = min_samples
        self.sync_distributed = sync_distributed
        dim = hidden_dim if hidden_dim is not None else 0
        self.register_buffer("prototypes", torch.zeros(num_classes, dim))
        self.register_buffer("num_updates", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor) -> None:
        """用一批已检测前景特征按类聚合更新原型（EMA）。

        Args:
            features: 已检测前景特征 ``[N, D]``（内部 ``detach``，不建图）。
            labels: 对应类别标签 ``[N]``。

        Raises:
            ValueError: 已初始化的原型维度与输入特征维度不一致时抛出。
        """
        features = features.detach()
        labels = labels.detach()
        if features.shape[0] == 0:
            return
        # 分布式多卡：先把各 rank 特征汇聚，保证各 rank 原型统计一致
        if self.sync_distributed:
            from rfdetr.utilities.distributed import all_gather

            gathered = all_gather((features, labels))
            features = torch.cat([g[0] for g in gathered], dim=0)
            labels = torch.cat([g[1] for g in gathered], dim=0)
        # 惰性初始化特征维度（仅未初始化时允许改变形状）
        if self.prototypes.shape[1] != features.shape[1]:
            if self.prototypes.numel() != 0:
                raise ValueError(
                    f"原型特征维度不一致: 已有 hidden_dim={self.prototypes.shape[1]}, "
                    f"输入 hidden_dim={features.shape[1]}"
                )
            self.prototypes.resize_(self.num_classes, features.shape[1])
        for c in labels.unique():
            c_idx = int(c.item())
            idx = labels == c
            count = int(idx.sum().item())
            if count < self.min_samples:
                continue
            batch_mean = features[idx].mean(dim=0)  # [D]
            if not torch.isfinite(batch_mean).all():
                # 输入特征污染：跳过本次更新，避免毒化原型
                logger.debug(f"PrototypeBank: 类别 {c_idx} 本批特征含 NaN/Inf，跳过更新")
                continue
            if not torch.isfinite(self.prototypes[c_idx]).all():
                # 历史原型已污染（NaN/Inf）：重置为未初始化状态，下次重新建立
                logger.debug(f"PrototypeBank: 类别 {c_idx} 历史原型含 NaN/Inf，重置")
                self.prototypes[c_idx].zero_()
                self.num_updates[c_idx] = 0
            if int(self.num_updates[c_idx]) == 0:
                # 首次出现：直接赋类内均值
                self.prototypes[c_idx].copy_(batch_mean)
            else:
                # 之后：EMA 平滑
                self.prototypes[c_idx].mul_(self.momentum).add_(
                    batch_mean, alpha=1.0 - self.momentum
                )
            self.num_updates[c_idx] += 1

    @torch.no_grad()
    def get_normalized_prototypes(self) -> tuple[Tensor, Tensor]:
        """返回行归一化原型与有效掩码。

        Returns:
            ``(proto_norm, valid)`` 元组：
            - proto_norm: 行归一化原型 ``[C, D]``，无效行置 0。
            - valid: ``[C]`` 布尔掩码，表示该类别原型已初始化且数值有效
              （``num_updates > 0`` 且行有限且模长非零）。
        """
        protos = self.prototypes
        if protos.shape[1] == 0:
            return protos, self.num_updates > 0
        row_norm = protos.norm(dim=-1)
        valid = (
            (self.num_updates > 0)
            & torch.isfinite(protos).all(dim=-1)
            & (row_norm > 1e-6)
        )
        # 逐行除以行模长：row_norm 需 unsqueeze(-1) 才能沿最后一维广播
        proto_norm = protos / row_norm.unsqueeze(-1).clamp_min(1e-6)
        proto_norm = proto_norm.where(valid.unsqueeze(-1), torch.zeros_like(proto_norm))
        return proto_norm, valid
