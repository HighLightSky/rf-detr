# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SSCL（语义相似度引导的监督对比学习）模块包。

SSCL（Semantic Correlation-Driven Supervised Contrastive Learning）通过 CLIP
离线构建的类别语义相似度矩阵，指导 decoder 输出的 matched foreground query
features 上的对比学习，缓解遥感数据集中舰船细粒度类别的分类混淆问题。

核心设计原则：
- CLIP 只做离线：仅用于构造类别语义相似度矩阵，不参与在线训练或推理。
- 模块化编程：各功能独立成文件，尽量不改动或少改动 RF-DETR 原有代码。
- 渐进式实验：通过多个阶段（矩阵构建 → SSCL 训练 → 基类蒸馏）逐步验证。

主要模块：
- prompts: 各类别的 CLIP 文本提示词。
- semantic_matrix: CLIP 语义相似度矩阵的构建、保存与验证。
- sscl_loss: SSCL 对比学习损失。
- distill_loss: 基类 logit 蒸馏损失。
"""

from __future__ import annotations

from rfdetr.sscl.distill_loss import BaseClassDistillLoss
from rfdetr.sscl.semantic_matrix import (
    build_semantic_similarity_matrix,
    load_semantic_matrix,
    normalize_semantic_matrix,
    save_semantic_matrix,
    validate_matrix,
)
from rfdetr.sscl.sscl_loss import SSCLLoss

__all__ = [
    "SSCLLoss",
    "BaseClassDistillLoss",
    "build_semantic_similarity_matrix",
    "save_semantic_matrix",
    "load_semantic_matrix",
    "normalize_semantic_matrix",
    "validate_matrix",
]
