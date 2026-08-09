# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义分类头改进（SemanticResidual）的离线准备与消融实验脚本。

本目录存放语义分类头实验的全部训练/准备入口，按阶段组织：

- ``stage0_collect_features.py``：离线收集 base 类 matched query 特征。
- ``stage0_train_fsem.py``：离线训练 f_sem + 计算 TF-IDF 通道统计 + 对齐校验。
- ``stage2_train.py``：Stage-2 消融训练入口（配合 ``ablation_configs.py``）。
- ``eval_ablation.py``：批量评估 9 个消融实验并生成对比表。

设计见 ``docs/改进方案-SSCL/RF-DETR语义分类头改进方案.md``。
"""
