# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义分类头训练监控：汇总 α/θ/掩码/贡献占比/对齐/梯度 到 TensorBoard。

新可学习参数 α_c（每类混合系数）与 θ_c（每类掩码阈值）是否真的在起作用，需要
在训练过程中持续观察。本模块是纯 CPU 侧的累加器：module_model 在每个采样步把
当前 batch 的统计喂进来，epoch 结束时统一通过 ``pl_module.log_dict`` 输出到
``train/sem/*`` 前缀下，输出后清空累加器。

监控维度：
- α/θ 逐类曲线与 novel/base 聚合（验证注入强度分化、θ 冻结是否生效）。
- 掩码激活率（M>0.5 的通道占比，novel 应从 ~95% 收窄）。
- 语义/掩码增量对 logits 的贡献占比（novel 应显著高于 base）。
- 每类 matched 特征均值与 s_c 的对齐余弦（验证特征是否向语义方向靠拢）。
- α/θ 梯度范数（验证可学习参数是否真在学；冻结参数应恒为 0）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# 采样节流：每 N 步采样一次（由 module_model 控制调用频率，本模块只负责累加）
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# 指标前缀
_PREFIX = "train/sem"


class SemanticMonitor:
    """语义头监控累加器。

    Args:
        class_names: 全部类别名称列表（按类别索引）。
        novel_classes: novel 类（舰船 0-3）索引列表。
        base_classes: base 类（飞机+FSC）索引列表。
        align_classes: 需要逐类输出对齐余弦的类别（默认 novel 类）。
        num_classes: 类别数 C。
    """

    def __init__(
        self,
        class_names: list[str],
        novel_classes: list[int],
        base_classes: list[int],
        align_classes: list[int] | None = None,
        num_classes: int = 25,
    ) -> None:
        self.class_names = class_names
        self.novel_classes = novel_classes
        self.base_classes = base_classes
        self.align_classes = align_classes if align_classes is not None else novel_classes
        self.num_classes = num_classes
        # 累加器：metric_name -> 该指标样本张量列表（或标量列表）
        self._acc: dict[str, list[Any]] = {}
        self._step_count = 0

    def _push(self, key: str, value: Any) -> None:
        """把一个样本值压入对应累加器。"""
        self._acc.setdefault(key, []).append(value)

    def update(self, step_stats: dict[str, Any]) -> None:
        """累加一个采样步的统计。

        Args:
            step_stats: 含 ``alpha``/``theta``/``M``/``ratio_sem``/``ratio_mask``/
                ``align``（可选）键的字典；各键为 ``[C]`` 或 ``[C, d]`` 张量。
        """
        self._step_count += 1
        alpha = step_stats["alpha"].float().detach().cpu()
        theta = step_stats["theta"].float().detach().cpu()
        m = step_stats["M"].float().detach().cpu()
        ratio_sem = step_stats["ratio_sem"].float().detach().cpu()
        ratio_mask = step_stats["ratio_mask"].float().detach().cpu()

        # --- α 聚合 ---
        self._push("alpha", alpha)
        # α 处于下边界（≤0，前向被 clamp 为 0）的比例 = 语义项对该类被关闭的比例。
        # E1a 中该指标从 1.0 反映"语义注入被优化器禁用"，是最关键的负向信号。
        self._push("alpha_clamped_ratio", float((alpha <= 0.0).float().mean().item()))
        # --- θ 聚合 ---
        self._push("theta", theta)
        # --- 掩码激活率：M>0.5 的通道占比 ---
        active = (m > 0.5).float().mean(dim=-1)  # [C] 每类激活率
        self._push("mask_mean", float(m.mean().item()))
        self._push("mask_active", active)
        # --- 贡献占比（novel/base 分组） ---
        self._push("ratio_sem", ratio_sem)
        self._push("ratio_mask", ratio_mask)
        # --- 对齐余弦（逐类，可选） ---
        if "align" in step_stats and step_stats["align"] is not None:
            self._push("align", step_stats["align"].float().detach().cpu())

    def update_grad_norms(self, alpha_grad: Tensor | None, theta_grad: Tensor | None) -> None:
        """累加 α/θ 的梯度范数（在 ``on_after_backward`` 中调用）。

        Args:
            alpha_grad: α 的梯度 ``[C]``（无梯度时为 ``None``）。
            theta_grad: θ 的梯度 ``[C]``（无梯度时为 ``None``）。
        """
        if alpha_grad is not None:
            self._push("alpha_grad", alpha_grad.float().detach().cpu())
        if theta_grad is not None:
            self._push("theta_grad", theta_grad.float().detach().cpu())

    def _mean_of(self, key: str) -> Tensor | None:
        """计算某指标的样本均值（张量），无样本时返回 None。"""
        values = self._acc.get(key)
        if not values:
            return None
        return torch.stack([v if v.dim() > 0 else v.unsqueeze(0) for v in values]).mean(dim=0)

    def _group_mean(self, per_class: Tensor | None, classes: list[int]) -> float:
        """计算某些类别的均值。"""
        if per_class is None:
            return 0.0
        ids = [c for c in classes if c < per_class.numel()]
        return float(per_class[ids].mean().item()) if ids else 0.0

    def on_train_epoch_end(self, pl_module: Any) -> None:
        """Epoch 结束时输出全部监控指标到 ``train/sem/*`` 并清空累加器。

        Args:
            pl_module: LightningModule（用于 ``log_dict``）。
        """
        if self._step_count == 0:
            return
        metrics: dict[str, Any] = {}

        alpha_mean = self._mean_of("alpha")
        theta_mean = self._mean_of("theta")
        mask_active = self._mean_of("mask_active")
        ratio_sem = self._mean_of("ratio_sem")
        ratio_mask = self._mean_of("ratio_mask")
        align = self._mean_of("align")

        # α 逐类（novel 4 类 + 抽样 base 类，避免 25 条噪声）
        for c in self.align_classes:
            if alpha_mean is not None and c < alpha_mean.numel():
                name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                metrics[f"{_PREFIX}/alpha/{name}"] = float(alpha_mean[c])
        # α 聚合
        if alpha_mean is not None:
            metrics[f"{_PREFIX}/alpha_mean_novel"] = self._group_mean(alpha_mean, self.novel_classes)
            metrics[f"{_PREFIX}/alpha_mean_base"] = self._group_mean(alpha_mean, self.base_classes)
            metrics[f"{_PREFIX}/alpha_std"] = float(alpha_mean.std().item())
            metrics[f"{_PREFIX}/alpha_min"] = float(alpha_mean.min().item())
            metrics[f"{_PREFIX}/alpha_max"] = float(alpha_mean.max().item())

        # θ 逐类与聚合
        for c in self.align_classes:
            if theta_mean is not None and c < theta_mean.numel():
                name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                metrics[f"{_PREFIX}/theta/{name}"] = float(theta_mean[c])
        if theta_mean is not None:
            metrics[f"{_PREFIX}/theta_mean_novel"] = self._group_mean(theta_mean, self.novel_classes)
            metrics[f"{_PREFIX}/theta_mean_base"] = self._group_mean(theta_mean, self.base_classes)
            metrics[f"{_PREFIX}/theta_std"] = float(theta_mean.std().item())

        # 掩码激活率
        m_mean = self._acc.get("mask_mean")
        if m_mean:
            metrics[f"{_PREFIX}/mask_mean"] = float(sum(m_mean) / len(m_mean))
        if mask_active is not None:
            metrics[f"{_PREFIX}/mask_active_ratio_novel"] = self._group_mean(mask_active, self.novel_classes)
            metrics[f"{_PREFIX}/mask_active_ratio_base"] = self._group_mean(mask_active, self.base_classes)
            for c in self.align_classes:
                if c < mask_active.numel():
                    name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                    metrics[f"{_PREFIX}/mask_active_ratio/{name}"] = float(mask_active[c])

        # 贡献占比（novel/base）
        if ratio_sem is not None:
            metrics[f"{_PREFIX}/ratio_sem_novel"] = self._group_mean(ratio_sem, self.novel_classes)
            metrics[f"{_PREFIX}/ratio_sem_base"] = self._group_mean(ratio_sem, self.base_classes)
        if ratio_mask is not None:
            metrics[f"{_PREFIX}/ratio_mask_novel"] = self._group_mean(ratio_mask, self.novel_classes)
            metrics[f"{_PREFIX}/ratio_mask_base"] = self._group_mean(ratio_mask, self.base_classes)

        # 对齐余弦
        if align is not None:
            metrics[f"{_PREFIX}/align_cos_mean"] = float(align.mean().item())
            metrics[f"{_PREFIX}/align_cos_novel_mean"] = self._group_mean(align, self.novel_classes)
            metrics[f"{_PREFIX}/align_cos_base_mean"] = self._group_mean(align, self.base_classes)
            for c in self.align_classes:
                if c < align.numel():
                    name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                    metrics[f"{_PREFIX}/align_cos/{name}"] = float(align[c])
            # novel 类的"同类−跨类"gap：用 (align - align.mean()) 近似（跨类均值用全体均值）
            novel_ids = [c for c in self.novel_classes if c < align.numel()]
            if novel_ids:
                novel_align = align[novel_ids]
                metrics[f"{_PREFIX}/align_gap_novel"] = float(novel_align.mean().item() - align.mean().item())

        # 梯度范数：_mean_of 已给出逐类梯度均值 [C]，逐类范数 = |grad|，聚合取 mean
        alpha_grad = self._mean_of("alpha_grad")
        theta_grad = self._mean_of("theta_grad")
        if alpha_grad is not None:
            per_class_norm = alpha_grad.abs()  # [C]
            metrics[f"{_PREFIX}/alpha_grad_norm"] = float(per_class_norm.mean().item())
            metrics[f"{_PREFIX}/alpha_grad_norm_novel"] = self._group_mean(per_class_norm, self.novel_classes)
            metrics[f"{_PREFIX}/alpha_grad_norm_base"] = self._group_mean(per_class_norm, self.base_classes)
        if theta_grad is not None:
            per_class_norm = theta_grad.abs()  # [C]
            metrics[f"{_PREFIX}/theta_grad_norm"] = float(per_class_norm.mean().item())

        # 饱和报警
        clamped = self._acc.get("alpha_clamped_ratio")
        if clamped:
            metrics[f"{_PREFIX}/alpha_clamped_ratio"] = float(sum(clamped) / len(clamped))

        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        logger.debug(f"[SemHead] 已输出 {len(metrics)} 个训练监控指标到 train/sem/*")
        self._acc.clear()
        self._step_count = 0
