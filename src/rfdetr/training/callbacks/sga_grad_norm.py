# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGA 分支梯度范数监控回调。

SGA 分支（SPM/SGM/融合层）为随机初始化，训练中最大的风险是塌缩成 no-op（近恒等）。
本回调在每个优化器步记录 ``backbone.0.sga.*`` 参数的梯度 L2 范数为 scalar
（键名 ``grad_norm/{名称}``），便于在 TensorBoard 中监控新分支是否真的在学。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn

if TYPE_CHECKING:
    from rfdetr.training.module_model import RFDETRModelModule


class SgaGradNormCallback(Callback):
    """记录 SGA 分支参数的梯度 L2 范数。

    Args:
        prefix: 参数名中用于识别 SGA 分支的子串，默认 ``"sga"``。
        首次优化器步时惰性收集参数，之后每个优化器步记录一次。
    """

    def __init__(self, prefix: str = "sga") -> None:
        super().__init__()
        self._prefix = prefix
        self._named_params: list[tuple[str, nn.Parameter]] | None = None

    def _collect(self, pl_module: LightningModule) -> None:
        """惰性收集 SGA 分支的、requires_grad 的参数。"""
        module = cast("RFDETRModelModule", pl_module)
        self._named_params = [
            (name, param)
            for name, param in module.model.named_parameters()
            if self._prefix in name and param.requires_grad
        ]

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Any,
        optimizer_idx: int = 0,
    ) -> None:
        """在优化器步之前记录各 SGA 参数的梯度 L2 范数。

        Args:
            trainer: Lightning Trainer 实例。
            pl_module: ``RFDETRModelModule`` 实例。
            optimizer: 当前优化器（未使用）。
            optimizer_idx: 优化器索引（未使用）。
        """
        if self._named_params is None:
            self._collect(pl_module)
        if not self._named_params:
            return  # 非 SGA 模型，无参可记

        for name, param in self._named_params:
            grad = param.grad
            if grad is None:
                continue  # 该参数本步无梯度（如 Phase 1 下未使用的 conv4）
            norm = grad.detach().float().norm(2)
            # 去掉 backbone.0. 前缀、点号换斜杠，便于 TensorBoard 分组显示
            short_name = name.replace("backbone.0.", "").replace(".", "/")
            pl_module.log(
                f"grad_norm/{short_name}",
                norm,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
