# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""基类 logit 蒸馏损失。

teacher（原始 RF-DETR checkpoint，完全冻结）只监督基类（飞机类 + FSC），
舰船类 logits 通道不参与蒸馏。原因：SSCL 的目标是重塑舰船类的特征空间
边界，若 teacher 同时蒸馏舰船类，会把学生"锚定"到 teacher 已经混乱的
舰船边界上，与 SSCL 的分离目标直接冲突。

蒸馏作用在所有 query（matched + background）的基类通道上，以同时保护
基类检测置信度和 background 抑制（避免基类 FP 增加）。

默认使用 MSE logit 蒸馏（第一版推荐，工程风险低）；也支持伯努利 KL
蒸馏（温度缩放的 sigmoid 软标签），梯度信息更柔和。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class BaseClassDistillLoss(nn.Module):
    """基类 logit 蒸馏损失，仅对受保护类别通道进行蒸馏。

    Args:
        protected_classes: 受保护类别索引列表（默认飞机类 + FSC，
            即索引 4-24，舰船类 0-3 不参与蒸馏）。
        temperature: 蒸馏温度，soft label 模式下软化标签分布。
        mode: 蒸馏方式：
            - ``"mse"``: logit 蒸馏，直接计算学生-教师 logits 的 MSE（默认）。
            - ``"kl"``: 伯努利 KL 蒸馏，对温度缩放的 sigmoid 软标签求 KL。
    """

    def __init__(
        self,
        protected_classes: list[int],
        temperature: float = 2.0,
        mode: str = "mse",
    ) -> None:
        super().__init__()
        self.protected_classes = list(protected_classes)
        self.temperature = temperature
        self.mode = mode

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """对受保护类别通道计算蒸馏损失。

        Args:
            student_logits: 学生模型分类 logits ``[B, Q, C]``。
            teacher_logits: 教师模型分类 logits ``[B, Q, C]``。

        Returns:
            标量蒸馏损失。

        Raises:
            ValueError: 当 ``mode`` 不是 ``"mse"`` 或 ``"kl"`` 时抛出。
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(f"学生与教师 logits 形状不一致: {student_logits.shape} vs {teacher_logits.shape}")

        # 仅取受保护类别通道；舰船类通道不参与蒸馏
        student_protected = student_logits[..., self.protected_classes]
        teacher_protected = teacher_logits[..., self.protected_classes]

        if self.mode == "mse":
            loss = F.mse_loss(student_protected, teacher_protected)
        elif self.mode == "kl":
            # 伯努利 KL：将每个受保护类别 logit 视为独立二分类分布，
            # 对温度缩放的 sigmoid 概率计算 KL(student || teacher)
            p_student = torch.sigmoid(student_protected / self.temperature)
            p_teacher = torch.sigmoid(teacher_protected / self.temperature)
            eps = torch.finfo(p_student.dtype).eps
            loss = F.kl_div(
                torch.log(p_student.clamp(min=eps)),
                p_teacher.clamp(min=eps, max=1.0 - eps),
                reduction="batchmean",
            )
        else:
            raise ValueError(f"不支持的蒸馏方式: {self.mode}，可选: 'mse', 'kl'")

        return loss
