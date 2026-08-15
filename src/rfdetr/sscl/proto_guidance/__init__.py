# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多模态原型引导模块包。

参考《Multi-modal Prototype Guided Few-shot Object Detection》(ACM MM'25)
的思想，在 RF-DETR 的 two-stage query selection 处用多模态原型
（视觉 DINOv2 + 文本 CLIP 融合）引导位置选择，并对选中 query 做内容增强。
详见 ``docs/改进方案-dinov2-proto/RF-DETR-DINOv2多模态原型引导方案.md``。

模块：
- guidance: ``ProtoGuidance`` 主模块（位置打分/内容增强/监控统计）。
- fusion: 视觉-文本原型融合（v1 为简化加权融合）。
- artifacts: 离线产物保存/加载/校验。
- monitor: 训练监控累加器（``train/proto/*``）。
"""

from __future__ import annotations

from rfdetr.sscl.proto_guidance.artifacts import (
    load_proto_artifacts,
    save_proto_artifacts,
    validate_proto_artifacts,
)
from rfdetr.sscl.proto_guidance.fusion import (
    GatedProtoFusion,
    SimpleProtoFusion,
    build_fusion,
)
from rfdetr.sscl.proto_guidance.guidance import ProtoGuidance
from rfdetr.sscl.proto_guidance.monitor import ProtoGuidanceMonitor

__all__ = [
    "ProtoGuidance",
    "SimpleProtoFusion",
    "GatedProtoFusion",
    "build_fusion",
    "ProtoGuidanceMonitor",
    "save_proto_artifacts",
    "load_proto_artifacts",
    "validate_proto_artifacts",
]
