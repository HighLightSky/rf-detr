# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多 slot 类原型库。

原型以 EMA 统计量维护，供原型锚定 SSCL 使用。与旧版单原型不同，这里允许
部分类别拥有多个 slot，以容纳同一类内部的多模态外观；其余类别仍保持单 slot。

设计约束：
- 原型与更新计数都保存在 buffer 中，随 checkpoint 保存/加载；
- slot 级更新仅影响该类的对应 slot，不把背景/难例塞进类中心；
- 旧 checkpoint 仍可加载：单原型状态会自动映射到 slot 0。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class SlotPrototypeBank(nn.Module):
    """类别多 slot 原型库（EMA 统计量，无梯度）。

    Args:
        num_classes: 类别总数。
        hidden_dim: 特征维度。为 ``None`` 时惰性初始化：首次 ``update``
            时按输入特征形状确定维度。
        momentum: EMA 更新系数 ``p <- m*p + (1-m)*batch_mean``。
        min_samples: 单次 batch 中某个 slot 的样本数低于该阈值时跳过该 slot。
        sync_distributed: 是否在 DDP 时先 all_gather 各 rank 特征再聚合更新。
        max_slots: 每个类别的最大 slot 数。``1`` 表示退化为单原型。
        multi_slot_classes: 允许启用多 slot 的类别索引列表。列表外类别仅使用
            slot 0，其余 slot 固定无效。

    Attributes:
        prototypes: 原型矩阵 ``[C, K, D]``。
        num_updates: 每类总更新次数 ``[C]``（兼容旧代码）。
        slot_num_updates: 每个 slot 的更新次数 ``[C, K]``。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int | None = None,
        momentum: float = 0.99,
        min_samples: int = 1,
        sync_distributed: bool = False,
        max_slots: int = 1,
        multi_slot_classes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if max_slots < 1:
            raise ValueError(f"max_slots 必须 >= 1，收到 {max_slots}。")

        self.num_classes = num_classes
        self.max_slots = max_slots
        self.momentum = momentum
        self.min_samples = min_samples
        self.sync_distributed = sync_distributed
        self.multi_slot_classes = sorted(set(multi_slot_classes or []))

        dim = hidden_dim if hidden_dim is not None else 0
        self.register_buffer("prototypes", torch.zeros(num_classes, max_slots, dim))
        self.register_buffer("num_updates", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("slot_num_updates", torch.zeros(num_classes, max_slots, dtype=torch.long))
        self.register_buffer("slot_valid_mask", self._build_slot_valid_mask(), persistent=True)

    def _build_slot_valid_mask(self) -> Tensor:
        """根据配置构建 slot 有效掩码。"""
        mask = torch.zeros(self.num_classes, self.max_slots, dtype=torch.bool)
        mask[:, 0] = True
        if self.max_slots == 1:
            return mask

        for class_id in self.multi_slot_classes:
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(f"multi_slot_classes 含非法类别索引: {class_id}")
            mask[class_id, :] = True
        return mask

    def _resize_if_needed(self, hidden_dim: int, slot_count: int | None = None) -> None:
        """按需重建 buffer 形状。"""
        target_slots = slot_count if slot_count is not None else self.max_slots
        if target_slots != self.max_slots:
            self.max_slots = target_slots
            self.slot_num_updates = torch.zeros(
                self.num_classes,
                self.max_slots,
                dtype=torch.long,
                device=self.num_updates.device,
            )
            self.slot_valid_mask = self._build_slot_valid_mask().to(self.num_updates.device)
        if self.prototypes.shape[1] != self.max_slots or self.prototypes.shape[2] != hidden_dim:
            new_proto = torch.zeros(
                self.num_classes,
                self.max_slots,
                hidden_dim,
                device=self.prototypes.device,
                dtype=self.prototypes.dtype,
            )
            self.prototypes = new_proto
        if self.slot_num_updates.shape != (self.num_classes, self.max_slots):
            self.slot_num_updates = torch.zeros(
                self.num_classes,
                self.max_slots,
                dtype=torch.long,
                device=self.num_updates.device,
            )

    @staticmethod
    def _cosine_scores(features: Tensor, prototypes: Tensor) -> Tensor:
        """计算特征与原型的余弦相似度矩阵。"""
        return F.normalize(features, dim=-1) @ F.normalize(prototypes, dim=-1).T

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor) -> None:
        """用一批已检测前景特征按类更新 slot 原型（EMA）。

        Args:
            features: 已检测前景特征 ``[N, D]``。
            labels: 对应类别标签 ``[N]``。
        """
        features = features.detach()
        labels = labels.detach()
        if features.shape[0] == 0:
            return

        if self.sync_distributed:
            from rfdetr.utilities.distributed import all_gather

            gathered = all_gather((features, labels))
            features = torch.cat([g[0] for g in gathered], dim=0)
            labels = torch.cat([g[1] for g in gathered], dim=0)

        if self.prototypes.shape[-1] != features.shape[-1]:
            if self.prototypes.shape[-1] != 0:
                raise ValueError(
                    f"原型特征维度不一致: 已有 hidden_dim={self.prototypes.shape[-1]}, "
                    f"输入 hidden_dim={features.shape[-1]}"
                )
            self._resize_if_needed(int(features.shape[-1]))

        for c in labels.unique(sorted=True):
            class_id = int(c.item())
            class_mask = labels == c
            class_features = features[class_mask]
            if class_features.shape[0] < self.min_samples:
                continue
            if self._update_one_class(class_id, class_features):
                self.num_updates[class_id] += 1

    def _update_one_class(self, class_id: int, class_features: Tensor) -> bool:
        """更新单个类别的所有有效 slot。"""
        valid_slots = self.slot_valid_mask[class_id].nonzero(as_tuple=False).flatten()
        if valid_slots.numel() == 0:
            return False

        class_slot_updates = self.slot_num_updates[class_id, valid_slots]
        initialized = valid_slots[class_slot_updates > 0].tolist()
        uninitialized = [
            int(slot.item())
            for slot in valid_slots
            if int(self.slot_num_updates[class_id, slot].item()) == 0
        ]
        updated = False

        # 先为尚未初始化的 slot 选种子，优先把和已有 slot 最不相似的样本分出去。
        while uninitialized:
            if not initialized:
                seed_slot = uninitialized.pop(0)
                seed_feature = class_features.mean(dim=0)
            else:
                if class_features.shape[0] <= len(initialized):
                    break
                seed_slot, seed_feature = self._pick_diverse_seed(class_id, class_features, initialized, uninitialized)
                if seed_slot is None or seed_feature is None:
                    break
                uninitialized.remove(seed_slot)

            if not torch.isfinite(seed_feature).all():
                logger.debug(f"PrototypeBank: 类别 {class_id} 的种子特征含 NaN/Inf，跳过初始化")
                continue
            self.prototypes[class_id, seed_slot].copy_(seed_feature)
            self.slot_num_updates[class_id, seed_slot] += 1
            initialized.append(seed_slot)
            updated = True

        # 归一化后按最近邻把当前 batch 的样本分配给各个已启用 slot。
        active_slots = valid_slots[self.slot_num_updates[class_id, valid_slots] > 0]
        if active_slots.numel() == 0:
            return updated
        slot_proto = self.prototypes[class_id, active_slots]
        if slot_proto.shape[-1] == 0:
            return updated
        scores = self._cosine_scores(class_features, slot_proto)
        assign = active_slots[scores.argmax(dim=1)]

        for slot_id in active_slots.tolist():
            slot_mask = assign == slot_id
            count = int(slot_mask.sum().item())
            if count < self.min_samples:
                continue
            batch_mean = class_features[slot_mask].mean(dim=0)
            if not torch.isfinite(batch_mean).all():
                logger.debug(
                    f"PrototypeBank: 类别 {class_id} 的 slot {slot_id} 本批特征含 NaN/Inf，跳过更新"
                )
                continue
            if int(self.slot_num_updates[class_id, slot_id].item()) == 0:
                self.prototypes[class_id, slot_id].copy_(batch_mean)
            else:
                self.prototypes[class_id, slot_id].mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)
            self.slot_num_updates[class_id, slot_id] += 1
            updated = True
        return updated

    def _pick_diverse_seed(
        self,
        class_id: int,
        class_features: Tensor,
        initialized: list[int],
        uninitialized: list[int],
    ) -> tuple[int | None, Tensor | None]:
        """为新的 slot 选一个尽量不同的种子样本。"""
        if not uninitialized:
            return None, None
        if class_features.shape[0] == 0:
            return None, None

        feat_norm = F.normalize(class_features, dim=-1)
        if initialized:
            init_proto = self.prototypes[class_id, initialized]
            init_proto = F.normalize(init_proto, dim=-1)
            diversity = (feat_norm @ init_proto.T).max(dim=1).values
        else:
            diversity = torch.zeros(class_features.shape[0], device=class_features.device)

        seed_idx = int(torch.argmin(diversity).item())
        seed_slot = uninitialized[0]
        return seed_slot, class_features[seed_idx]

    @torch.no_grad()
    def get_normalized_slot_prototypes(self) -> tuple[Tensor, Tensor]:
        """返回 slot 级归一化原型与有效掩码。"""
        protos = self.prototypes
        slot_valid = self.slot_valid_mask & (self.slot_num_updates > 0) & torch.isfinite(protos).all(dim=-1)
        if protos.shape[-1] == 0:
            return protos, slot_valid

        row_norm = protos.norm(dim=-1)
        slot_valid = slot_valid & (row_norm > 1e-6)
        proto_norm = protos / row_norm.unsqueeze(-1).clamp_min(1e-6)
        proto_norm = proto_norm.where(slot_valid.unsqueeze(-1), torch.zeros_like(proto_norm))
        return proto_norm, slot_valid

    @torch.no_grad()
    def get_normalized_prototypes(self) -> tuple[Tensor, Tensor]:
        """返回按类别聚合后的归一化原型与有效掩码。

        该接口保留给旧分析脚本与兼容代码使用；多 slot 情况下返回同类 slot
        的均值并再做一次归一化。
        """
        slot_norm, slot_valid = self.get_normalized_slot_prototypes()
        class_valid = slot_valid.any(dim=-1)
        if slot_norm.shape[-1] == 0:
            return slot_norm.new_zeros(self.num_classes, 0), class_valid

        class_proto = slot_norm.new_zeros(self.num_classes, slot_norm.shape[-1])
        for class_id in range(self.num_classes):
            valid_slots = slot_valid[class_id]
            if not valid_slots.any():
                continue
            class_vec = slot_norm[class_id][valid_slots].mean(dim=0)
            norm = class_vec.norm().clamp_min(1e-6)
            class_proto[class_id] = class_vec / norm
        return class_proto, class_valid

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """兼容旧 checkpoint 的单原型状态。"""
        proto_key = prefix + "prototypes"
        num_key = prefix + "num_updates"
        slot_key = prefix + "slot_num_updates"
        mask_key = prefix + "slot_valid_mask"

        loaded_proto = state_dict.get(proto_key)
        loaded_mask = state_dict.get(mask_key)
        loaded_slots = None
        if loaded_mask is not None:
            loaded_slots = int(loaded_mask.shape[1])
        elif loaded_proto is not None:
            loaded_slots = 1 if loaded_proto.ndim == 2 else int(loaded_proto.shape[1])

        if loaded_proto is not None:
            target_dim = int(loaded_proto.shape[-1])
            target_slots = int(loaded_slots or self.max_slots)
            if self.prototypes.shape[-1] == 0 and target_dim > 0:
                self._resize_if_needed(target_dim, target_slots)
            elif self.prototypes.shape[-1] not in (0, target_dim):
                raise ValueError(
                    f"原型维度不一致: 当前 hidden_dim={self.prototypes.shape[-1]}, "
                    f"checkpoint hidden_dim={target_dim}"
                )

            if loaded_proto.ndim == 2 and self.prototypes.ndim == 3:
                mapped = self.prototypes.new_zeros(self.num_classes, self.max_slots, target_dim)
                mapped[:, 0, :] = loaded_proto
                state_dict[proto_key] = mapped
            elif (
                loaded_proto.ndim == 3
                and self.prototypes.ndim == 3
                and loaded_proto.shape[1] != self.prototypes.shape[1]
            ):
                target_slots = max(self.prototypes.shape[1], loaded_proto.shape[1])
                if target_slots != self.prototypes.shape[1]:
                    self._resize_if_needed(target_dim, target_slots)
                mapped = self.prototypes.new_zeros(self.num_classes, self.max_slots, target_dim)
                copy_slots = min(loaded_proto.shape[1], self.max_slots)
                mapped[:, :copy_slots, :] = loaded_proto[:, :copy_slots, :]
                state_dict[proto_key] = mapped

        if slot_key not in state_dict and num_key in state_dict:
            loaded_num = state_dict[num_key]
            if loaded_num.ndim == 1:
                mapped = self.slot_num_updates.new_zeros(self.num_classes, self.max_slots)
                mapped[:, 0] = loaded_num
                state_dict[slot_key] = mapped

        if mask_key not in state_dict:
            if self.slot_valid_mask.shape[1] != self.max_slots:
                self.slot_valid_mask = self._build_slot_valid_mask().to(self.slot_valid_mask.device)
            state_dict[mask_key] = self.slot_valid_mask.clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


# 兼容旧导入路径：``PrototypeBank`` 仍然可用，但实现已经升级为多 slot。
PrototypeBank = SlotPrototypeBank
