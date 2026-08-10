# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本训练监控累加器。

在 module_model 的 SSCL 回调中按 ``sscl_hard_neg_log_interval`` 步间隔喂入
CPU 标量统计，epoch 末统一输出到 ``train/sscl/*`` 前缀后清空（仿
``SemanticMonitor`` 的累加-冲刷模式）。

指标：
- ``hn_count`` / ``hn_fill_rate``：每图平均难例数、IoU 带内未匹配候选占比
  （填充率过低说明难例不常见，需要放宽采样规则）；
- ``hn_proto_cos`` / ``random_proto_cos`` / ``matched_proto_cos``：难例 /
  随机未匹配 / matched 三组特征与类别原型的平均余弦（"难例是否硬"的直接
  证据）；
- ``hn_vs_random_gap`` / ``hn_vs_matched_gap``：难例相对随机未匹配与
  matched 的余弦差距（> 0 表示难例更贴类中心，即"硬"成立）。
"""

from __future__ import annotations

from typing import Any

from rfdetr.utilities.logger import get_logger

logger = get_logger()

_PREFIX = "train/sscl"


class HardNegMonitor:
    """难例负样本训练监控累加器。

    不产生损失、不污染 loss_dict；update 只收 CPU 标量，epoch 末输出均值后清空。
    """

    def __init__(self) -> None:
        self._acc: dict[str, list[float]] = {}
        self._step_count = 0

    def update(self, step_stats: dict[str, float]) -> None:
        """累加一个采样步的标量统计。

        Args:
            step_stats: 难例统计字典（键与输出的 ``train/sscl/*`` 同名）。
        """
        self._step_count += 1
        for key, value in step_stats.items():
            self._acc.setdefault(key, []).append(float(value))

    def on_train_epoch_end(self, pl_module: Any) -> None:
        """Epoch 末输出均值到 ``train/sscl/*`` 并清空累加器。

        Args:
            pl_module: Lightning 模块（提供 ``log_dict``）。
        """
        if self._step_count == 0:
            return
        metrics = {f"{_PREFIX}/{k}": sum(v) / len(v) for k, v in self._acc.items()}
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
        logger.debug(f"[SSCL-HN] 已输出 {len(metrics)} 个难例监控指标到 {_PREFIX}/*")
        self._acc.clear()
        self._step_count = 0
