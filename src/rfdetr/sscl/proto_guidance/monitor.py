# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多模态原型引导训练监控：汇总 top-k 重叠/带入比例/判别性/gate/梯度 到 TensorBoard。

判断原型引导模块是否真的在起作用（而非被训练成 no-op），需要持续观察：
- ``topk_overlap``：合并分数与纯线性分数的 top-k 重叠率，长期接近 1 说明
  位置分支没有改变 query selection；
- ``proto_selected_ratio``：原型 residual 新带入 top-k 的 query 比例；
- ``lambda_effective``：位置权重的有效量级（相对线性分数标准差）；
- 原型 logits 判别性（max/熵）与选中类别分布；
- 内容 gate 均值（E3 起）与原型参数梯度范数（冻结应恒 0）。

本模块是纯 CPU 侧累加器（仿 SemanticMonitor）：module_model 在每个采样步
把统计喂进来，epoch 结束时经 ``pl_module.log_dict`` 输出到 ``train/proto/*``
前缀下，输出后清空累加器。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# 指标前缀
_PREFIX = "train/proto"


class ProtoGuidanceMonitor:
    """原型引导监控累加器。

    Args:
        class_names: 全部类别名称列表（按类别索引）。
        watch_classes: 需要逐类输出的类别（默认全部，通常传 target 类）。
    """

    def __init__(self, class_names: list[str], watch_classes: list[int] | None = None) -> None:
        self.class_names = list(class_names)
        self.watch_classes = sorted(watch_classes or list(range(len(class_names))))
        # 累加器：metric_name -> 样本列表
        self._acc: dict[str, list[Any]] = {}
        self._step_count = 0

    def _push(self, key: str, value: Any) -> None:
        """把一个样本值压入对应累加器。"""
        self._acc.setdefault(key, []).append(value)

    def update(self, step_stats: dict[str, Tensor]) -> None:
        """累加一个采样步的统计。

        Args:
            step_stats: ``ProtoGuidance.collect_stats`` 的返回字典
                （``[bs]`` 或 ``[C]`` 张量，自动 detach + cpu）。
        """
        self._step_count += 1
        for key, value in step_stats.items():
            self._push(key, value.detach().cpu())

    def update_grad_norms(self, param_grads: dict[str, float]) -> None:
        """累加原型模块各参数的梯度范数（在 ``on_after_backward`` 中调用）。

        Args:
            param_grads: ``{参数名: 梯度 L2 范数}``（无梯度参数传 0 或不传）。
        """
        for name, norm in param_grads.items():
            self._push(f"grad_{name}", float(norm))

    def _mean_of(self, key: str) -> Tensor | None:
        """计算某指标的样本均值（按样本数取平均），无样本时返回 None。"""
        values = self._acc.get(key)
        if not values:
            return None
        stacked = torch.stack([v if v.dim() > 0 else v.unsqueeze(0) for v in values])
        return stacked.mean(dim=0)

    def _sum_of(self, key: str) -> Tensor | None:
        """计算某指标的总和（用于直方图类计数），无样本时返回 None。"""
        values = self._acc.get(key)
        if not values:
            return None
        return torch.stack([v if v.dim() > 0 else v.unsqueeze(0) for v in values]).sum(dim=0)

    def on_train_epoch_end(self, pl_module: Any) -> None:
        """Epoch 结束时输出全部监控指标到 ``train/proto/*`` 并清空累加器。

        Args:
            pl_module: LightningModule（用于 ``log_dict``）。
        """
        if self._step_count == 0:
            return
        metrics: dict[str, Any] = {}

        # --- top-k 有效性（核心判据） ---
        overlap = self._mean_of("topk_overlap")
        if overlap is not None:
            metrics[f"{_PREFIX}/topk_overlap"] = float(overlap.mean().item())
        ratio = self._mean_of("proto_selected_ratio")
        if ratio is not None:
            metrics[f"{_PREFIX}/proto_selected_ratio"] = float(ratio.mean().item())

        # --- 有效 lambda（量级匹配） ---
        lam = self._mean_of("lambda_effective")
        if lam is not None:
            metrics[f"{_PREFIX}/lambda_effective"] = float(lam.mean().item())

        # --- 原型判别性 ---
        pmax = self._mean_of("proto_logits_pmax_mean")
        if pmax is not None:
            metrics[f"{_PREFIX}/proto_logits_pmax_mean"] = float(pmax.mean().item())
        ent = self._mean_of("proto_logits_entropy_mean")
        if ent is not None:
            metrics[f"{_PREFIX}/proto_logits_entropy_mean"] = float(ent.mean().item())

        # --- 选中类别分布（逐 watch 类 + 总体） ---
        hist = self._sum_of("selected_class_hist")
        if hist is not None:
            total = float(hist.sum().item())
            if total > 0:
                for c in self.watch_classes:
                    if c < hist.numel():
                        name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                        metrics[f"{_PREFIX}/selected_ratio/{name}"] = float(hist[c].item()) / total

        # --- 内容 gate（E3 起） ---
        gate = self._mean_of("gate_mean")
        if gate is not None:
            metrics[f"{_PREFIX}/gate_mean"] = float(gate.mean().item())

        # --- 原型参数梯度范数（冻结应恒 0） ---
        grad_keys = [k for k in self._acc if k.startswith("grad_")]
        for key in grad_keys:
            values = self._acc[key]
            mean_norm = float(sum(values) / len(values))
            metrics[f"{_PREFIX}/{key}"] = mean_norm

        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        logger.debug(f"[ProtoGuidance] 已输出 {len(metrics)} 个训练监控指标到 train/proto/*")
        self._acc.clear()
        self._step_count = 0
