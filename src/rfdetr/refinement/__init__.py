# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""两阶段候选框复核组件。"""

from rfdetr.refinement.fsc_two_stage import (
    FSCScoreFusion,
    FSCDinoHead,
    FSCMultiViewHead,
    FSCFeatureGeometryHead,
    FSCEnsembleHead,
    FSCVerifier,
    FSCVerifierPolicy,
    crop_fsc_context,
    crop_transform,
    iou_xyxy,
    label_fsc_candidate,
    pool_dino_features,
)

__all__ = [
    "FSCScoreFusion",
    "FSCDinoHead",
    "FSCMultiViewHead",
    "FSCFeatureGeometryHead",
    "FSCEnsembleHead",
    "FSCVerifier",
    "FSCVerifierPolicy",
    "crop_fsc_context",
    "crop_transform",
    "iou_xyxy",
    "label_fsc_candidate",
    "pool_dino_features",
]
