# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""探针/诊断分析脚本包。

收编原散落在 ``src/scripts/`` 根目录的探针与归因分析脚本：

- ``linear_probe.py``：骨干/query 特征线性探针实验（被 probe_head/prototype_probe 复用）；
- ``probe_head.py``：均衡线性头作为舰船小类分类器的端到端验证；
- ``prototype_probe.py``：SSCL EMA 原型即分类器的离线验证；
- ``analyze_fn_decomposition.py`` / ``analyze_fp_decomposition.py``：漏检/虚警成因分解。

所有脚本通过 ``from scripts import eval_lib`` 复用统一评估管线
（推理、数据集配置、比赛指标）。
"""
