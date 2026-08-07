# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGA 门控/融合变体小规模实验包。

用于验证实验报告（output/0805-SHWX-SGA-rfdetr/实验报告.md §五）提出的 P0 修复方向：
在短训规模下对比 baseline / SPM-only / fixed-SGA（下界门控、残差门控）各变体，
确认修复方向正确后再进行全量微调。入口为 ``run.py``，公共逻辑见 ``common.py``。
"""
