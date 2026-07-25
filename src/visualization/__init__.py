# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""训练过程和检测结果可视化工具包。

提供训练指标曲线绘制、FP/FN 错误分析可视化和混淆矩阵功能。
"""

from visualization.detection import (
    build_confusion_matrix,
    clear_vis_dirs,
    load_image,
    match_per_image_per_class,
    plot_confusion_matrix,
    save_fp_fn_visualizations,
)
from visualization.training import (
    plot_loss_metrics,
    plot_lr_schedule,
    plot_map_metrics,
    plot_metrics,
    plot_per_class_ap,
)

__all__ = [
    "build_confusion_matrix",
    "clear_vis_dirs",
    "load_image",
    "match_per_image_per_class",
    "plot_confusion_matrix",
    "plot_loss_metrics",
    "plot_lr_schedule",
    "plot_map_metrics",
    "plot_metrics",
    "plot_per_class_ap",
    "save_fp_fn_visualizations",
]
