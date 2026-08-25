# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试评估库，提供配置解析、统一批量推理和评测结果生成。

test.py、分析脚本和语义实验脚本都复用本模块。标准图片由 CPU worker 解码并组成
固定批次，GPU 负责异步拷贝、预处理、detector 前向和阈值筛选；大图和 reason plugin
也使用同一套批量 detector 接口。最后统一生成 BoxRecord 并计算比赛指标。
"""

from __future__ import annotations

import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# 项目根目录和模块路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 类别名称统一来自 sscl/prompts/*.yaml，保证与语义矩阵的类别索引一致
from rfdetr import RFDETR  # noqa: E402
from rfdetr.sscl.prompts import (  # noqa: E402
    DIOR_CLASS_NAMES,
    SHWX_CLASS_NAMES,
    SHWX_TRUCK_CLASS_NAMES,
)
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    EvalResult,
    evaluate_competition_metrics,
    load_yolo_labels,
)
from visualization.detection import (  # noqa: E402
    build_confusion_matrix,
    clear_vis_dirs,
    match_per_image_per_class,
    plot_confusion_matrix,
    save_label_comparison_visualizations,
    save_fp_fn_visualizations,
)


def _label_keyed_names(names_by_id: dict[int, str]) -> dict[int, str]:
    """把 {类别id: 名称} 转成 {label: 名称}。

    label 是按类别 id 排序后的连续索引，与训练时的类别 remap 一致。

    Args:
        names_by_id: {类别id: 名称} 映射。

    Returns:
        {label: 名称} 映射。
    """
    return {label: names_by_id[cid] for label, cid in enumerate(sorted(names_by_id.keys()))}


# 各数据集配置集中维护，新增数据集时沿用现有字段。
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "shwx": {
        "data_dir": "/home/liu/wzt/datasets/SHWX-dataset-dict",
        "image_dir": "images/test",
        "label_format": "yolo",
        "label_dir": "labels/test",
        "exp_output_dir": "output/0811-SHWX-SSCL-仅验证QNorm-Obj物体性门控",
        "checkpoint_file": "checkpoint_best_total.pth",
        "num_classes": 25,
        "vehicle_class_ids": {24},  # FSC 发射车，比赛规则按车辆目标 IoU=0.35
        "metric_excluded_class_ids": set(),
        "class_names": _label_keyed_names(SHWX_CLASS_NAMES),
        # 大类分组：25 类 → 3 个大类（舰船/飞机/车辆）
        "class_to_group": {
            **{class_id: "ship" for class_id in range(0, 4)},
            **{class_id: "aircraft" for class_id in range(4, 24)},
            24: "vehicle",
        },
        "group_iou_thresholds": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
    },
    "shwx_truck": {
        "data_dir": "/home/liu/wzt/datasets/SHWX-dataset-dict-redo-truck",
        "image_dir": "images/test",
        "label_format": "yolo",
        "label_dir": "labels/test",
        "exp_output_dir": "output/0820-SHWX-26class-truck-eval",
        "checkpoint_file": "checkpoint_best_total.pth",
        "num_classes": 26,
        "vehicle_class_ids": {24},  # 车辆类指标只保留 FSC，truck 采用常规 IoU=0.50
        "metric_excluded_class_ids": {25},  # truck 只作辅助类别，不进入大类/总指标
        "class_names": _label_keyed_names(SHWX_TRUCK_CLASS_NAMES),
        # 26 类实验：舰船 0-3、飞机 4-23、车辆 24-25。
        "class_to_group": {
            **{class_id: "ship" for class_id in range(0, 4)},
            **{class_id: "aircraft" for class_id in range(4, 24)},
            24: "vehicle",
            25: "vehicle",
        },
        "group_iou_thresholds": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
    },
    "dior": {
        "data_dir": "/home/liu/wzt/datasets/DIOR-rfdetr",
        "image_dir": "test",
        "label_format": "coco",
        "annotation_file": "test/_annotations.coco.json",
        "exp_output_dir": "output/0804-DIOR-rfdetr_medium_SSCL",
        "checkpoint_file": "checkpoint_best_regular.pth",
        "num_classes": 20,
        "vehicle_class_ids": set(),  # DIOR 无比赛特殊 IoU 规则，全部按 0.50
        "metric_excluded_class_ids": set(),
        "class_names": _label_keyed_names(DIOR_CLASS_NAMES),
        # DIOR 无舰船/飞机/车辆大类分组，所有类别归为单组 "all"
        "class_to_group": {class_id: "all" for class_id in range(20)},
        "group_iou_thresholds": {"all": 0.50},
    },
}


@dataclass(frozen=True)
class DatasetCfg:
    """数据集配置（``DATASET_CONFIGS`` 条目解析后的形态，只读）。

    Attributes:
        name: 数据集名（如 ``"shwx"``）。
        data_dir: 数据集根目录。
        test_image_dir: 测试图像目录（``data_dir / image_dir``）。
        label_format: 标签格式（``"yolo"`` / ``"coco"``）。
        label_dir: YOLO 标签目录（YOLO 格式时有效）。
        annotation_file: COCO 标注文件路径（COCO 格式时有效）。
        exp_output_dir: 实验输出目录（测试报告/可视化写入处）。
        checkpoint_file: 默认 checkpoint 相对文件名。
        num_classes: 类别数。
        vehicle_class_ids: 车辆类别 id 集合（比赛规则按 IoU=0.35）。
        metric_excluded_class_ids: 不参与大类与总指标计算的辅助类别 ID 集合。
        class_names: ``{label: 名称}`` 映射。
        class_to_group: ``{类别 id: 大类名}`` 映射。
        group_iou_thresholds: ``{大类名: IoU 阈值}`` 映射。
        per_class_to_group: 逐类分组映射（每个类独立成组，用于输出逐类指标）。
        per_class_iou_thresholds: 逐类 IoU 阈值（``{类别名: 阈值}``）。
    """

    name: str
    data_dir: Path
    test_image_dir: Path
    label_format: str
    label_dir: Path | None
    annotation_file: Path | None
    exp_output_dir: Path
    checkpoint_file: str
    num_classes: int
    vehicle_class_ids: frozenset[int]
    metric_excluded_class_ids: frozenset[int]
    class_names: dict[int, str]
    class_to_group: dict[int, str]
    group_iou_thresholds: dict[str, float]
    per_class_to_group: dict[int, str]
    per_class_iou_thresholds: dict[str, float]


def build_dataset_cfg(
    name: str = "shwx",
    root: Path | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> DatasetCfg:
    """按名称构建数据集配置（相对路径以项目根解析）。

    Args:
        name: ``DATASET_CONFIGS`` 中的数据集名（``"shwx"`` / ``"dior"``）。
        root: 相对路径解析基准，默认项目根。
        output_dir: 可选：覆盖 ``exp_output_dir``（测试报告/可视化输出目录）。
            传相对路径时以 *root* 为基准解析；``None`` 用 ``DATASET_CONFIGS``
            内置值。各测试实验通过 yaml 的 ``test.output_dir`` 传此参数，避免
            多次评估互相覆盖输出。
        data_dir: 可选：覆盖内置 ``data_dir``（数据集根目录）。与训练侧
            ``dataset_dir`` 配置模式对应——测试集跟随训练数据集的目录，
            例如重新标注后的数据集路径通过 yaml 的 ``test.dataset_dir`` 传入；
            ``None`` 用 ``DATASET_CONFIGS`` 内置值。类别语义（类名/大类分组/
            IoU 阈值）仍由 *name* 指定的内置配置提供。

    Returns:
        解析后的 DatasetCfg。

    Raises:
        KeyError: 数据集名不在 ``DATASET_CONFIGS`` 中。
    """
    root = root or PROJECT_ROOT
    cfg = DATASET_CONFIGS[name]
    data_dir = Path(data_dir) if data_dir is not None else Path(cfg["data_dir"])
    label_dir = data_dir / cfg["label_dir"] if cfg.get("label_dir") else None
    annotation_file = data_dir / cfg["annotation_file"] if cfg.get("annotation_file") else None
    class_names = cfg["class_names"]
    class_to_group = cfg["class_to_group"]
    # 相对路径（无 / 前缀）以 root 为基准解析，绝对路径原样——与 expcfg.resolve_paths 规则一致
    if output_dir is not None:
        exp_output_dir = Path(output_dir)
        if not str(output_dir).startswith("/"):
            exp_output_dir = root / exp_output_dir
    else:
        exp_output_dir = root / cfg["exp_output_dir"]
    return DatasetCfg(
        name=name,
        data_dir=data_dir,
        test_image_dir=data_dir / cfg["image_dir"],
        label_format=cfg["label_format"],
        label_dir=label_dir,
        annotation_file=annotation_file,
        exp_output_dir=exp_output_dir,
        checkpoint_file=cfg["checkpoint_file"],
        num_classes=cfg["num_classes"],
        vehicle_class_ids=frozenset(cfg["vehicle_class_ids"]),
        metric_excluded_class_ids=frozenset(cfg.get("metric_excluded_class_ids", set())),
        class_names=class_names,
        class_to_group=class_to_group,
        group_iou_thresholds=cfg["group_iou_thresholds"],
        # 细粒度类别分组映射：每个类独立成组，用于输出逐类指标
        per_class_to_group={class_id: name_ for class_id, name_ in class_names.items()},
        per_class_iou_thresholds={
            name_: 0.35 if class_id in cfg["vehicle_class_ids"] else 0.50 for class_id, name_ in class_names.items()
        },
    )


@dataclass
class InferenceCfg:
    """推理参数（原 ``CONF_THRESHOLD``/``DEVICE``/``BATCH_SIZE``/``NUM_WORKERS`` 等 模块级常量的收敛形态）。

    Attributes:
        device: 推理设备（如 ``"cuda:0"``；无 CUDA 时自动回退 CPU）。
        conf_threshold: 全局置信度阈值（默认 0.25）。
        class_conf_thresholds: 逐类置信度阈值 ``{类别id: 阈值}``（默认全 0.25）。
        batch_size: GPU 单次前向的图像数。
        num_workers: CPU 预取 worker 进程数（建议等于 CPU 核数）。
        prefetch_factor: 每个 worker 在内存中预取的数据批数。
        precision: 推理精度；``auto`` 在 CUDA 上优先选择 BF16，否则选择 FP16。
        compile_model: 是否使用 ``torch.compile`` 优化 detector 前向。
        copy_prefetch: 是否使用 CUDA copy stream 预取下一批数据。
        warmup_batches: 不计入测速的真实 warmup 批次数。
        progress_interval_s: 进度输出的最小时间间隔；小于等于 0 时关闭。
        gpu_monitor_enabled: 是否启用后台 GPU 利用率采样。
    """

    device: str = "cuda:0"
    conf_threshold: float = 0.25
    class_conf_thresholds: dict[int, float] = field(default_factory=dict)
    batch_size: int = 32
    num_workers: int = 12
    prefetch_factor: int = 3
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    compile_model: bool = False
    copy_prefetch: bool = True
    warmup_batches: int = 1
    progress_interval_s: float = 1.0
    gpu_monitor_enabled: bool = False


@dataclass(frozen=True)
class ReasonPluginCfg:
    """测试侧 FFT 一致性插件配置。

    Attributes:
        checkpoint: 已训练插件 checkpoint 路径。
        class_ids: 需要重打分的类别；``None`` 表示全部类别。
        conf_low: 低置信候选下限；``None`` 使用 checkpoint 内置值。
    """

    checkpoint: str | Path
    class_ids: tuple[int, ...] | None = (24,)
    conf_low: float | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> ReasonPluginCfg | None:
        """解析 ``test.reason_plugin`` 配置；未配置或关闭时返回 ``None``。

        Args:
            config: yaml 中的插件配置段。

        Returns:
            启用后的插件配置，或 ``None``。

        Raises:
            ValueError: 配置类型、checkpoint、类别或阈值不合法。
        """
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("test.reason_plugin 必须是字典配置")
        if not bool(config.get("enabled", False)):
            return None

        checkpoint = config.get("checkpoint")
        if not checkpoint:
            raise ValueError("test.reason_plugin.enabled=true 时必须设置 checkpoint")

        class_ids = config.get("class_ids", [24])
        if class_ids is not None:
            if not isinstance(class_ids, (list, tuple)):
                raise ValueError("test.reason_plugin.class_ids 必须是整数列表或 null")
            try:
                class_ids = tuple(int(class_id) for class_id in class_ids)
            except (TypeError, ValueError) as exc:
                raise ValueError("test.reason_plugin.class_ids 必须只包含整数") from exc

        conf_low = config.get("conf_low")
        if conf_low is not None:
            try:
                conf_low = float(conf_low)
            except (TypeError, ValueError) as exc:
                raise ValueError("test.reason_plugin.conf_low 必须是数字或 null") from exc
            if not 0.0 <= conf_low <= 1.0:
                raise ValueError("test.reason_plugin.conf_low 必须位于 [0, 1]")

        return cls(checkpoint=checkpoint, class_ids=class_ids, conf_low=conf_low)


@dataclass(frozen=True)
class LabelComparisonCfg:
    """YOLO 标签对比可视化配置。

    Attributes:
        labels_dir: YOLO 格式标签目录。
        iou_threshold: 同类别预测和真实框判定为 TP 的 IoU 阈值。
    """

    labels_dir: Path
    iou_threshold: float = 0.50

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> LabelComparisonCfg | None:
        """解析 ``predict.label_comparison`` 配置。

        Args:
            config: yaml 中的标签对比配置段。

        Returns:
            开启后的标签对比配置，未配置或关闭时返回 ``None``。

        Raises:
            ValueError: 配置类型、标签目录或 IoU 阈值不合法。
        """
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("predict.label_comparison 必须是字典配置")
        if not bool(config.get("enabled", False)):
            return None

        labels_dir = config.get("labels_dir")
        if not isinstance(labels_dir, (str, Path)) or not str(labels_dir).strip():
            raise ValueError("predict.label_comparison.enabled=true 时必须设置 labels_dir")

        iou_threshold = config.get("iou_threshold", 0.50)
        if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
            raise ValueError("predict.label_comparison.iou_threshold 必须是数字")
        iou_threshold = float(iou_threshold)
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("predict.label_comparison.iou_threshold 必须位于 [0, 1]")
        return cls(labels_dir=Path(labels_dir), iou_threshold=iou_threshold)


@dataclass
class LargeImageCfg:
    """大图切分测试配置（边界检测 → 裁切 → 目标检测 → 映射回原图）。

    Attributes:
        min_side: 大图判定阈值（长边像素数），长边 ≥ 该值的图像走切分流程。
        boundary_checkpoint: 边界检测器 checkpoint 路径（切分流程必需）。
        boundary_backend: 边界模型后端，支持 ``rfdetr`` 与 ``yolo``。
        boundary_resolution: 边界检测器输入分辨率（须与训练一致）。
        boundary_conf: 边界框置信度阈值。
        detector_conf: 裁窗目标检测置信度阈值。
        padding: 裁窗外扩像素。
        nms_iou: 边界框 NMS IoU 阈值（0 关闭）。
        square_stretch: 边界检测用方形拉伸替代 letterbox。
        batch_size: 边界检测器 GPU 批量大小。
        num_workers: 边界检测器预取 worker 数。
        max_pending_crops: crop 元数据队列允许缓存的最大数量。
        roi_backend: 大图 ROI 读取后端，支持 auto、pyvips、opencv。
        proxy_max_side: 边界检测 proxy 最长边。
        roi_output_size: ROI 输出尺寸，None 使用主检测分辨率。
        roi_queue_size: ROI 像素队列上限。
        roi_cache_dir: 可选 proxy 缓存目录。
        strict_roi_backend: 严格要求 pyvips 时失败。
    """

    min_side: int = 2000
    boundary_checkpoint: str | None = None
    boundary_backend: str = "rfdetr"
    boundary_resolution: int = 704
    boundary_conf: float = 0.25
    detector_conf: float = 0.25
    padding: int = 32
    nms_iou: float = 0.0
    square_stretch: bool = False
    batch_size: int = 8
    num_workers: int = 4
    max_pending_crops: int = 128
    roi_backend: str = "auto"
    proxy_max_side: int | None = None
    roi_output_size: int | None = None
    roi_queue_size: int = 128
    roi_cache_dir: str | None = None
    strict_roi_backend: bool = False


@dataclass
class LaBiasCfg:
    """推理侧 Logit Adjustment bias 配置（原 ``LOGIT_ADJUSTMENT_BIAS_*`` 常量）。

    由 ``class_counts.json``（stat_class_counts.py 产物）按训练侧同配方重建
    logit bias；设置后 ``score' = sigmoid(logit - k * bias)``。默认重建参数
    （beta/max_weight/min_count/ref_count/target_classes）须与训练侧
    ``class_balance_*`` 配方一致。

    Attributes:
        counts_path: ``class_counts.json`` 路径。
        k: 推理侧扣减系数（0/0.5/1）。
        tau: 与训练侧 ``logit_adjustment_tau`` 一致。
        clip: 与训练侧 ``logit_adjustment_bias_clip`` 一致。
        beta: 与训练侧 ``class_balance_beta`` 一致。
        max_weight: 与训练侧 ``class_balance_max_weight`` 一致。
        min_count: 与训练侧 ``class_balance_min_count`` 一致。
        ref_count: 与训练侧 ``class_balance_ref_count`` 一致（None 自动取）。
        target_classes: 与训练侧 ``class_balance_target_classes`` 一致。
    """

    counts_path: str | Path
    k: float = 1.0
    tau: float = 0.1
    clip: float = 1.0
    beta: float = 0.25
    max_weight: float = 3.0
    min_count: int = 10
    ref_count: float | None = None
    target_classes: list[int] | None = None

    def build_bias_tensor(self, num_logit_classes: int, device: str) -> torch.Tensor:
        """按训练侧配方从 ``class_counts.json`` 重建 logit bias 张量。

        Args:
            num_logit_classes: 分类头输出通道数（``num_classes + 1``，含背景槽位）。
            device: 目标设备。

        Returns:
            ``(num_logit_classes,)`` 的 bias 张量（背景槽位补 0）。
        """
        from rfdetr.models.criterion import SetCriterion

        counts = torch.as_tensor(
            json.loads(Path(self.counts_path).read_text(encoding="utf-8"))["counts"],
            dtype=torch.float32,
        )
        _, la_bias = SetCriterion._build_class_balance_buffers(
            counts=counts,
            beta=self.beta,
            max_weight=self.max_weight,
            min_count=self.min_count,
            ref_count=self.ref_count,
            target_classes=self.target_classes,
            tau=self.tau,
            bias_clip=self.clip,
        )
        # 补齐分类头的背景槽位 bias，保证 logits 长度与输出通道一致。
        if la_bias.numel() < num_logit_classes:
            la_bias = torch.cat([la_bias, torch.zeros(num_logit_classes - la_bias.numel(), dtype=la_bias.dtype)])
        return la_bias.to(device)


# 工具函数。


def read_test_image_paths(image_dir: Path) -> list[Path]:
    """从测试图像目录扫描图像路径列表（按文件名排序）。"""
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"测试图像目录中未找到 .jpg/.png 文件: {image_dir}")
    return image_paths


def build_image_size_map(image_paths: list[Path]) -> dict[str, tuple[int, int]]:
    """读取测试集图像尺寸，返回 {image_id: (width, height)}。"""
    image_size_map: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        image_size_map[image_path.stem] = (width, height)
        del image
    return image_size_map


def load_coco_labels(annotation_path: str | Path) -> list[BoxRecord]:
    """读取 Roboflow COCO 标注文件，生成真实框记录。

    COCO 的 ``bbox`` 为 ``[x, y, w, h]``（像素坐标），此处转换为 ``xyxy``。
    类别 ``category_id`` 按 id 排序映射为连续 label（0..C-1），
    与训练时 ``remap_category_ids=True`` 的映射（build_roboflow_from_coco）保持一致，
    保证真实框类别与模型预测类别处于同一 label 空间。

    Args:
        annotation_path: ``_annotations.coco.json`` 文件路径。

    Returns:
        BoxRecord 列表。
    """
    import json

    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    # category_id → 连续 label（按 id 排序，与训练 remap 一致）
    cat_ids = sorted(int(cat["id"]) for cat in data["categories"])
    cat2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    # image_id → 文件名（Roboflow 布局：图像与标注文件在同一目录）
    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    records: list[BoxRecord] = []
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        records.append(
            BoxRecord(
                image_id=Path(img_id_to_file[ann["image_id"]]).stem,
                class_id=cat2label[ann["category_id"]],
                xyxy=(float(x), float(y), float(x) + float(w), float(y) + float(h)),
            )
        )
    return records


def save_yolo_predictions(
    pred_records: list[BoxRecord],
    output_dir: Path,
    image_size_map: dict[str, tuple[int, int]],
) -> None:
    """把预测框保存为 YOLO 格式 txt（每图一个文件，与 load_yolo_predictions 兼容）。

    每行 ``class_id cx cy w h conf``，坐标为按图像尺寸归一化的 cxcywh，
    置信度追加在末尾（GT 标签无 conf 列，预测文件带 conf 列，可被
    ``val.competition_metrics.load_yolo_predictions`` 读取）。

    Args:
        pred_records: 预测框记录列表（xyxy 像素坐标）。
        output_dir: 输出目录（自动创建）。
        image_size_map: {image_id: (width, height)} 映射，用于归一化。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[BoxRecord]] = {}
    for r in pred_records:
        by_image.setdefault(r.image_id, []).append(r)
    for image_id, records in by_image.items():
        w, h = image_size_map[image_id]
        lines: list[str] = []
        for r in sorted(records, key=lambda rec: rec.score or 0.0, reverse=True):
            x0, y0, x1, y1 = r.xyxy
            cx = (x0 + x1) / 2 / w
            cy = (y0 + y1) / 2 / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            lines.append(f"{r.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {r.score:.6f}")
        (output_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[i] YOLO 预测已保存: {output_dir}（{len(by_image)} 个文件）")


def save_yolo_label_comparisons(
    image_paths: list[Path],
    pred_records: list[BoxRecord],
    class_names: dict[int, str],
    output_dir: str | Path,
    comparison_cfg: LabelComparisonCfg,
) -> tuple[int, int, int]:
    """加载 YOLO 标签并保存预测对比可视化。

    所有输入图像均会输出一张左右对照图。左侧为真实标签，右侧为模型预测；
    FP 和 FN 均以红色突出，其中 FP 显示预测置信度，FN 显示 ``conf=N/A``。

    Args:
        image_paths: 已完成预测的图像路径列表。
        pred_records: 本次预测生成的检测框记录。
        class_names: 类别 ID 到名称的映射字典。
        output_dir: 对比图输出目录。
        comparison_cfg: 标签路径和匹配阈值配置。

    Returns:
        ``(saved_images, fp_count, fn_count)``：保存图片数量、FP 框数量和 FN
        框数量。

    Raises:
        FileNotFoundError: 标签目录不存在。
    """
    if not comparison_cfg.labels_dir.is_dir():
        raise FileNotFoundError(f"YOLO 标签目录不存在: {comparison_cfg.labels_dir}")

    image_size_map = build_image_size_map(image_paths)
    gt_records = load_yolo_labels(comparison_cfg.labels_dir, image_size_map)
    configured_class_ids = set(class_names)
    unknown_class_ids = {record.class_id for record in gt_records + pred_records} - configured_class_ids
    if unknown_class_ids:
        raise ValueError(f"标签或预测包含未配置类别 ID: {sorted(unknown_class_ids)}")
    num_classes = max(class_names, default=-1) + 1
    fp_images, fn_images, fp_boxes, fn_boxes, tp_preds = match_per_image_per_class(
        gt_records,
        pred_records,
        num_classes,
        frozenset(),
        default_iou_threshold=comparison_cfg.iou_threshold,
    )
    saved_images = save_label_comparison_visualizations(
        fp_boxes,
        fn_boxes,
        tp_preds,
        gt_records,
        image_paths,
        class_names,
        output_dir,
    )
    fp_count = sum(len(boxes) for boxes in fp_boxes.values())
    fn_count = sum(len(boxes) for boxes in fn_boxes.values())
    return saved_images, fp_count, fn_count


def resolve_device(device: str) -> str:
    """根据当前环境解析实际推理设备。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[w] 当前环境未检测到 CUDA，自动改用 CPU 推理", flush=True)
        return "cpu"
    return device


def format_cuda_memory(device: str) -> str:
    """格式化当前 CUDA 显存占用信息。"""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "CUDA=N/A"

    device_index = torch.cuda.current_device()
    allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
    return f"CUDA={allocated_gb:.2f}G/{reserved_gb:.2f}G"


def print_progress(index: int, total: int, start_time: float, device: str) -> None:
    """在同一行实时打印推理进度。"""
    elapsed = time.perf_counter() - start_time
    speed = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / speed if speed > 0 else 0.0
    percent = index / total * 100 if total > 0 else 100.0

    print(
        f"\r[i] 推理进度 {index:>5d}/{total:<5d} "
        f"{percent:6.2f}% | {speed:5.2f} img/s | "
        f"ETA {remaining:7.1f}s | {format_cuda_memory(device)}",
        end="",
        flush=True,
    )


def release_cuda_cache(device: str) -> None:
    """按需释放 Python 引用和 PyTorch CUDA 缓存。"""
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_eval_line(name: str, result: EvalResult) -> str:
    """把单组比赛评测结果格式化为一行（控制台与结果报告共用）。

    Args:
        name: 组名（如 ``all``/``ship`` 或类别名）。
        result: 单组比赛评测结果。

    Returns:
        格式化后的评测结果行。
    """
    return (
        f"{name:<11s}TP={result.tp:<7d}FP={result.fp:<7d}FN={result.fn:<7d}"
        f"Recall={result.recall:.4f} FDR={result.fdr:.4f} Precision={result.precision:.4f}"
    )


def print_eval_result(name: str, result: EvalResult) -> None:
    """打印单组比赛评测结果。"""
    print(format_eval_line(name, result))


def format_macro_line(name: str, macro: Mapping[str, float]) -> str:
    """把大类下小类指标的 macro 平均结果格式化为一行。

    Args:
        name: 大类名（如 ``ship``/``aircraft``/``vehicle``）或 ``total``。
        macro: 含 ``avg_tp``/``avg_fp``/``avg_fn``/``recall``/``fdr``/``precision``
            六项的 macro 平均指标。

    Returns:
        格式化后的 macro 平均结果行。
    """
    return (
        f"{name:<11s}avgTP={macro['avg_tp']:.2f} avgFP={macro['avg_fp']:6.2f} "
        f"avgFN={macro['avg_fn']:6.2f} avgRecall={macro['recall']:.4f} "
        f"avgFDR={macro['fdr']:.4f} avgPrecision={macro['precision']:.4f}"
    )


def compute_group_macro_averages(
    per_class_results: Mapping[str, EvalResult],
    class_to_group: Mapping[int, str],
    class_names: Mapping[int, str],
) -> dict[str, dict[str, float]]:
    """计算每个大类下小类指标的平均值（macro 平均）。

    对大类中的每个小类，先按比赛口径计算各项指标（TP/FP/FN、召回率、虚警率、
    精确率），再直接对同大类下所有小类的指标取算术平均，而不是先累计样本数再
    计算指标。例如船的召回率 = 四型船（驱护舰、航母、两栖船、民船）召回率的
    平均值，飞机与车辆同理。

    Args:
        per_class_results: ``{类别名: EvalResult}`` 的逐类评估结果。
        class_to_group: ``{类别 id: 大类名}`` 映射。
        class_names: ``{类别 id: 类别名}`` 映射。

    Returns:
        ``{大类名: {"avg_tp": ..., "avg_fp": ..., "avg_fn": ..., "recall": ...,
        "fdr": ..., "precision": ...}}``，大类顺序按其内最小类别 id 升序排列
        （即舰船→飞机→车辆）。
    """
    # 把 {类别id: 大类} 反转为 {大类: [类别id]}，类别 id 升序
    group_to_class_ids: dict[str, list[int]] = defaultdict(list)
    for class_id, group_name in sorted(class_to_group.items()):
        group_to_class_ids[group_name].append(class_id)

    # 按大类内最小类别 id 排序大类，保证舰船→飞机→车辆的展示顺序
    group_macro: dict[str, dict[str, float]] = {}
    for group_name, class_ids in sorted(
        group_to_class_ids.items(),
        key=lambda item: min(item[1]),
    ):
        # 测试集中不存在（无真实框也无预测）的小类取全零结果，保持按全部小类平均
        class_results = [
            per_class_results.get(class_names[class_id], EvalResult(tp=0, fp=0, fn=0)) for class_id in class_ids
        ]
        num_classes = len(class_results)
        group_macro[group_name] = {
            "avg_tp": sum(result.tp for result in class_results) / num_classes,
            "avg_fp": sum(result.fp for result in class_results) / num_classes,
            "avg_fn": sum(result.fn for result in class_results) / num_classes,
            "recall": sum(result.recall for result in class_results) / num_classes,
            "fdr": sum(result.fdr for result in class_results) / num_classes,
            "precision": sum(result.precision for result in class_results) / num_classes,
        }
    return group_macro


def compute_total_metrics(group_macro: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """计算总指标：各大类平均指标再取算术平均（即（船+飞机+车辆）/3）。

    Args:
        group_macro: ``compute_group_macro_averages`` 的返回结果。

    Returns:
        与单大类 macro 相同键结构的六项总指标。
    """
    metric_keys = ("avg_tp", "avg_fp", "avg_fn", "recall", "fdr", "precision")
    num_groups = len(group_macro)
    if num_groups == 0:
        return {key: 0.0 for key in metric_keys}
    return {key: sum(group[key] for group in group_macro.values()) / num_groups for key in metric_keys}


def filter_auxiliary_metric_records(
    records: list[BoxRecord],
    excluded_class_ids: frozenset[int],
) -> list[BoxRecord]:
    """过滤不参与大类和总指标计算的辅助类别记录。

    Args:
        records: 原始真实框或预测框记录。
        excluded_class_ids: 辅助类别 ID 集合。

    Returns:
        去除辅助类别后的记录列表；逐类指标、混淆矩阵和 FP/FN 分析仍使用原始列表。
    """
    return [record for record in records if record.class_id not in excluded_class_ids]


def build_metric_class_to_group(dataset: DatasetCfg) -> dict[int, str]:
    """构建排除辅助类别后的大类映射。

    Args:
        dataset: 数据集评测配置。

    Returns:
        仅包含参与大类 macro 与总指标计算类别的分组映射。
    """
    return {
        class_id: group_name
        for class_id, group_name in dataset.class_to_group.items()
        if class_id not in dataset.metric_excluded_class_ids
    }


def build_test_report(
    *,
    dataset_name: str,
    checkpoint_path: Path,
    test_image_paths: list[Path],
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    throughput: float,
    timed_images: int,
    gpu_util: float | None,
    eval_results: dict[str, EvalResult | dict[str, EvalResult]],
    group_macro: dict[str, dict[str, float]],
    total_macro: dict[str, float],
    per_class_results: dict[str, EvalResult | dict[str, EvalResult]],
    dataset: DatasetCfg,
    infer: InferenceCfg,
    test_resolution: int | None = None,
    large_errors_dir: Path | None = None,
    large_image_stats: dict[str, float] | None = None,
) -> list[str]:
    """组装 test_result.txt 报告文本行列表。

    Args:
        dataset_name: 数据集名（如 ``"shwx"``）。
        checkpoint_path: 权重文件路径。
        test_image_paths: 测试图像路径列表。
        gt_records: 真实框记录列表。
        pred_records: 预测框记录列表。
        throughput: 稳态推理吞吐（img/s）。
        timed_images: 参与稳态计时的图像数。
        gpu_util: 推理期间 GPU 平均利用率（%），采样失败时为 ``None``。
        eval_results: 比赛指标评估结果（``all`` + 各大大类）。
        group_macro: 每个大类下小类指标的 macro 平均结果。
        total_macro: 各大类平均指标再取平均的总指标。
        per_class_results: 细粒度逐类评估结果。
        dataset: 数据集配置。
        infer: 推理参数。
        test_resolution: 实际生效的测试输入分辨率。
        large_errors_dir: 大图错误可视化目录；非 ``None`` 时写入报告。
        large_image_stats: 大图切分目标检测的耗时统计（``count``/``avg``/
            ``max``/``total``，秒）；非 ``None`` 时写入报告。

    Returns:
        报告文本行列表。
    """
    sep = "=" * 80
    dash = "-" * 80
    report_lines: list[str] = []

    # 写入报告头和运行环境信息。
    report_lines.extend(
        [
            sep,
            "RF-DETR 测试结果报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"数据集: {dataset_name}",
            f"权重: {checkpoint_path}",
            f"测试分辨率: {test_resolution if test_resolution is not None else '未记录'}",
            f"数据集目录: {dataset.data_dir}",
            sep,
        ]
    )
    if dataset.metric_excluded_class_ids:
        excluded_names = "，".join(
            f"{dataset.class_names.get(class_id, str(class_id))}({class_id})"
            for class_id in sorted(dataset.metric_excluded_class_ids)
        )
        report_lines.append(f"大类与总指标排除的辅助类别: {excluded_names}")

    # 写入测试数据、阈值和 IoU 配置。
    report_lines.extend(
        [
            f"测试图像数: {len(test_image_paths)}",
            f"真实框数: {len(gt_records)}",
            f"预测框数: {len(pred_records)}",
            f"置信度阈值: {infer.conf_threshold}"
            + (
                "；逐类 "
                + "，".join(
                    f"{dataset.class_names.get(cid, str(cid))}={thr:.2f}"
                    for cid, thr in sorted(infer.class_conf_thresholds.items())
                )
                if infer.class_conf_thresholds
                else ""
            ),
            "IoU 阈值: " + "，".join(f"{key}={value:.2f}" for key, value in dataset.group_iou_thresholds.items()),
            sep,
        ]
    )

    # 写入批量推理吞吐和大图耗时。
    report_lines.append("推理测速结果")
    report_lines.append(
        f"GPU 批量大小: {infer.batch_size}  |  CPU 预取 worker 数: {infer.num_workers}"
        f"  |  精度: {infer.precision}  |  compile: {infer.compile_model}"
    )
    elapsed = timed_images / throughput if throughput > 0 else 0.0
    report_lines.append(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {elapsed:.1f}s）")
    if gpu_util is not None:
        report_lines.append(f"推理期间 GPU 平均利用率: {gpu_util:.1f}%")
    if large_image_stats is not None:
        report_lines.append(
            f"大图目标检测（裁切推理）: {int(large_image_stats['count'])} 张 | "
            f"平均 {large_image_stats['avg']:.2f}s | 最大 {large_image_stats['max']:.2f}s | "
            f"合计 {large_image_stats['total']:.2f}s"
        )
        if "detector_seconds" in large_image_stats:
            report_lines.append(
                f"统一 detector 阶段（小图+crop 混合）: {float(large_image_stats['detector_seconds']):.2f}s"
            )
        if "boundary_seconds" in large_image_stats:
            end_to_end = float(large_image_stats.get("end_to_end_seconds", large_image_stats["total"]))
            count = max(float(large_image_stats["count"]), 1.0)
            report_lines.append(
                "大图端到端计时（边界检测+裁切+小图推理，边界批量均摊）: "
                f"平均 {end_to_end / count:.2f}s | 合计 {end_to_end:.2f}s | "
                f"边界模型合计 {float(large_image_stats['boundary_seconds']):.2f}s"
            )
        if "proxy_seconds" in large_image_stats:
            report_lines.append(f"proxy 读取合计: {float(large_image_stats['proxy_seconds']):.2f}s")
        if "fallback_count" in large_image_stats:
            report_lines.append(f"ROI 后端 fallback 图像数: {int(large_image_stats['fallback_count'])}")
    report_lines.append(sep)

    # 写入比赛指标和大类聚合结果。
    report_lines.append("比赛指标评估结果（测试集）")
    report_lines.append(format_eval_line("all", eval_results["all"]))
    for group_name, group_result in eval_results["groups"].items():
        report_lines.append(format_eval_line(group_name, group_result))
    report_lines.append(dash)

    # 写入各大类的小类 macro 平均结果。
    report_lines.append("每个大类下小类指标的平均值（macro 平均）")
    report_lines.append(dash)
    for group_name, macro in group_macro.items():
        report_lines.append(format_macro_line(group_name, macro))
    report_lines.append(dash)

    # 写入所有大类 macro 的总平均结果。
    report_lines.append("总指标（各大类平均指标再取算术平均，即（船+飞机+车辆）/3）")
    report_lines.append(dash)
    report_lines.append(format_macro_line("total", total_macro))
    report_lines.append(dash)

    # 写入细粒度类别指标。
    report_lines.append("细粒度类别指标")
    report_lines.append(dash)
    for class_name in sorted(per_class_results["groups"].keys()):
        report_lines.append(format_eval_line(class_name, per_class_results["groups"][class_name]))
    report_lines.append(sep)

    # 写入混淆矩阵和 FP/FN 文件路径。
    report_lines.append("生成的可视化文件")
    report_lines.append(f"混淆矩阵: {dataset.exp_output_dir / 'confusion_matrix.png'}")
    report_lines.append(f"FP 可视化目录: {dataset.exp_output_dir / 'FP'}")
    report_lines.append(f"FN 可视化目录: {dataset.exp_output_dir / 'FN'}")
    if large_errors_dir is not None:
        report_lines.append(f"大图错误可视化目录: {large_errors_dir}")

    return report_lines


def write_test_result(report_lines: list[str], output_path: Path) -> None:
    """把测试结果报告写入实验文件夹中的 test_result.txt。

    Args:
        report_lines: 报告文本行列表。
        output_path: 输出文件路径（一般为 ``EXP_OUTPUT_DIR / "test_result.txt"``）。
    """
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[完成] 测试结果已保存至: {output_path}")


# 统一批量推理：CPU 预取、CUDA 拷贝和 GPU 前向并行执行。


class _InferenceDataset(Dataset):
    """在 DataLoader 子进程 worker 中读取并轻量预处理的测试图像数据集。

    每个样本返回 ``(image_id, rgb_tensor, (height, width))``。标准检测路径会在
    worker 中将图像调整为固定分辨率；reason plugin 路径保留原始尺寸以便裁剪 patch。
    两种路径都返回连续的 uint8 RGB CHW 张量，测试集只遍历一轮。

    Args:
        image_paths: 测试图像路径列表。
    """

    def __init__(self, image_paths: list[Path], resize_resolution: int | None = None) -> None:
        self.image_paths = image_paths
        self.resize_resolution = resize_resolution

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, tuple[int, int]]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (C,H,W) uint8 连续张量
        if self.resize_resolution is not None:
            rgb_tensor = F.resize(
                rgb_tensor,
                (self.resize_resolution, self.resize_resolution),
                antialias=False,
            )
        return image_path.stem, rgb_tensor, (height, width)


def _inference_collate(
    batch: list[tuple[str, torch.Tensor, tuple[int, int]]],
) -> tuple[list[str], list[torch.Tensor], list[tuple[int, int]]]:
    """自定义聚合函数：图像原始尺寸不同，保持为张量列表而非堆叠。"""
    stems = [item[0] for item in batch]
    tensors = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]
    return stems, tensors, orig_sizes


def _worker_init_fn(worker_id: int) -> None:
    """限制每个 worker 进程的线程数，避免多进程线程争抢导致过订阅。

    Args:
        worker_id: DataLoader 分配的 worker 编号（未使用）。
    """
    del worker_id
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class _GpuUtilMonitor:
    """后台线程周期性采样 GPU 利用率，用于评估推理期间 GPU 是否满载。

    Args:
        interval: 采样间隔（秒）。
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """启动后台采样线程。"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample_once())
            except Exception:
                pass
            self._stop.wait(self.interval)

    @staticmethod
    def _sample_once() -> int:
        """读取一次 GPU 利用率（%）。

        优先使用 ``torch.cuda.utilization``（需要 nvidia-ml-py），不可用时回退到 ``nvidia-smi`` 命令行。
        """
        try:
            return int(torch.cuda.utilization())
        except Exception:
            pass
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0].strip())

    def stop(self) -> None:
        """停止采样线程。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def average_utilization(self) -> float | None:
        """返回采样期间的 GPU 平均利用率（%）。"""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)


def _rescore_reason_candidates(
    reason_plugin: Any,
    class_embed_weight: torch.Tensor,
    class_names: list[str],
    source_image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
    class_conf_thresholds: Mapping[int, float] | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在最终阈值筛选前，对批量推理的一张图执行插件重打分。

    不同类别可能设置不同阈值，因此按最终阈值将插件目标类别分组。每组只修改
    对应类别的候选分数，其他类别保持基线分数，随后由调用方统一执行逐类筛选。

    Args:
        reason_plugin: 已加载的 FFT 一致性插件。
        class_embed_weight: 冻结检测器的类别嵌入矩阵。
        class_names: 按类别索引排序的类别名称。
        source_image: 当前图像的原始 RGB uint8 数组。
        boxes: 候选框像素坐标数组。
        scores: 候选框置信度数组。
        class_ids: 候选类别数组。
        conf_threshold: 默认最终置信度阈值。
        class_conf_thresholds: 可选逐类阈值表。
        device: 插件推理设备。

    Returns:
        保持原候选顺序的 ``(boxes, scores, class_ids)``；仅目标类别的低分
        候选分数可能被修改。
    """
    if boxes.size == 0:
        return boxes, scores, class_ids

    configured_ids = reason_plugin.config.reason_class_ids
    present_ids = {int(class_id) for class_id in class_ids}
    target_ids = present_ids if configured_ids is None else present_ids.intersection(configured_ids)
    if not target_ids:
        return boxes, scores, class_ids

    adjusted_scores = scores.astype(np.float32, copy=True)
    if hasattr(reason_plugin, "predict_detections_batch"):
        threshold_array = np.asarray(
            [class_conf_thresholds.get(int(label), conf_threshold) for label in class_ids]
            if class_conf_thresholds
            else np.full(scores.shape, conf_threshold, dtype=np.float32),
            dtype=np.float32,
        )
        out_boxes, out_scores, out_class_ids = reason_plugin.predict_detections_batch(
            [
                {
                    "image": source_image,
                    "boxes": boxes,
                    "scores": scores,
                    "classes": class_ids,
                }
            ],
            class_names,
            class_embed_weight,
            device,
            target_thresholds=[threshold_array],
            reason_class_ids=tuple(target_ids),
            filter_final=False,
        )[0]
        if out_boxes.shape != boxes.shape or not np.array_equal(out_class_ids, class_ids):
            raise RuntimeError("reason_plugin 在未筛选模式下必须保持候选框顺序与类别不变")
        adjusted_scores[:] = out_scores
        return boxes, adjusted_scores, class_ids

    threshold_groups: dict[float, list[int]] = {}
    for class_id in target_ids:
        threshold = (
            float(class_conf_thresholds.get(class_id, conf_threshold)) if class_conf_thresholds else conf_threshold
        )
        threshold_groups.setdefault(threshold, []).append(class_id)

    for target_conf, grouped_ids in threshold_groups.items():
        out_boxes, out_scores, out_class_ids = reason_plugin.predict_detections(
            source_image=source_image,
            candidate_boxes=boxes,
            candidate_scores=adjusted_scores,
            candidate_classes=class_ids,
            class_names=class_names,
            class_embed_weight=class_embed_weight,
            device=device,
            target_conf=target_conf,
            reason_class_ids=tuple(grouped_ids),
            filter_final=False,
        )
        if out_boxes.shape != boxes.shape or not np.array_equal(out_class_ids, class_ids):
            raise RuntimeError("reason_plugin 在未筛选模式下必须保持候选框顺序与类别不变")
        group_mask = np.isin(class_ids, grouped_ids)
        adjusted_scores[group_mask] = out_scores[group_mask]

    return boxes, adjusted_scores, class_ids


class _InferenceRuntime:
    """统一管理 detector 的设备、精度、编译和 CUDA 预取运行时。"""

    def __init__(
        self,
        model: RFDETR,
        device: str,
        resolution: int,
        batch_size: int,
        precision: Literal["auto", "fp32", "fp16", "bf16"],
        compile_model: bool,
        copy_prefetch: bool,
    ) -> None:
        self.device = torch.device(device)
        self.resolution = resolution
        self.batch_size = batch_size
        self.copy_prefetch = copy_prefetch and self.device.type == "cuda"
        self.model = model.model.model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.autocast_dtype = self._resolve_precision(precision)
        self.compiled = False
        if compile_model and self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, dynamic=False)  # type: ignore[assignment]
                self.compiled = True
            except Exception as exc:
                print(f"[w] torch.compile 失败，使用同一运行时的 eager 前向: {exc}", flush=True)
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.copy_prefetch else None
        self._ready_event: torch.cuda.Event | None = None
        self.means = torch.as_tensor(model.means, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        self.stds = torch.as_tensor(model.stds, device=self.device, dtype=torch.float32).view(1, -1, 1, 1)

    def _resolve_precision(self, precision: Literal["auto", "fp32", "fp16", "bf16"]) -> torch.dtype | None:
        """解析运行精度；CPU 始终使用 FP32。"""
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError(f"不支持的推理精度: {precision}")
        if self.device.type != "cuda" or precision == "fp32":
            return None
        if precision == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("当前 CUDA 设备不支持 BF16")
            return torch.bfloat16
        return None

    @property
    def autocast_context(self) -> Any:
        """返回 detector 前向使用的 autocast 上下文。"""
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def stage_batch(
        self,
        rgb_tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """在 copy stream 上完成 H2D 和 resize，并返回已就绪的 batch。"""
        stream_context = torch.cuda.stream(self.copy_stream) if self.copy_stream is not None else contextlib.nullcontext()
        with stream_context:
            fixed_shape = all(tensor.shape[1:] == (self.resolution, self.resolution) for tensor in rgb_tensors)
            if fixed_shape:
                staged_batch = torch.stack(rgb_tensors).to(self.device, non_blocking=True).to(torch.float32).div_(255.0)
                staged = list(staged_batch.unbind(0))
            else:
                staged = [
                    F.resize(
                        tensor.to(self.device, non_blocking=True).to(torch.float32).div_(255.0),
                        (self.resolution, self.resolution),
                        antialias=False,
                    )
                    for tensor in rgb_tensors
                ]
            if self.copy_stream is not None:
                self._ready_event = torch.cuda.Event()
                self._ready_event.record(self.copy_stream)
        if self.copy_stream is not None and self._ready_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready_event)
        return staged

    def forward(self, images: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
        """完成归一化和 detector 前向，必要时对尾 batch 做零填充。"""
        valid_count = len(images)
        if valid_count == 0:
            raise ValueError("推理 batch 不能为空")
        if self.compiled and valid_count < self.batch_size:
            padding = images[0].new_zeros((self.batch_size - valid_count, *images[0].shape))
            images = images + list(padding)
        batch_tensor = (torch.stack(images) - self.means) / self.stds
        with torch.inference_mode(), self.autocast_context:
            try:
                predictions = self.model(batch_tensor)
            except Exception:
                if not self.compiled:
                    raise
                print("[w] 编译 detector 前向失败，回退同一 runtime 的未编译前向", flush=True)
                self.model = getattr(self.model, "_orig_mod", self.model)
                self.compiled = False
                predictions = self.model(batch_tensor)
        if valid_count != len(images):
            predictions = {
                key: value[:valid_count] if isinstance(value, torch.Tensor) and value.shape[0] == len(images) else value
                for key, value in predictions.items()
            }
        return predictions, valid_count


def predict_batched_to_records(
    model: RFDETR,
    image_paths: list[Path],
    device: str,
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
    class_conf_thresholds: Mapping[int, float] | None = None,
    *,
    la_bias: LaBiasCfg | torch.Tensor | None = None,
    num_classes: int = 25,
    prefetch_factor: int = 3,
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto",
    compile_model: bool = False,
    copy_prefetch: bool = True,
    warmup_batches: int = 1,
    progress_interval_s: float = 1.0,
    gpu_monitor_enabled: bool = False,
    reason_plugin: Any | None = None,
    reason_class_embed: torch.Tensor | None = None,
    reason_class_names: list[str] | None = None,
) -> tuple[list[BoxRecord], float, float | None, int]:
    """执行一次完整批量推理并返回预测框、稳态吞吐和 GPU 采样结果。

    worker 负责解码和必要的固定尺寸调整，runtime 负责异步 H2D、归一化、detector
    前向、LA bias、后处理和批量导出。首批真实数据可作为 warmup，但仍会计入预测结果。

    Args:
        model: 已加载的 RFDETR 实例（任意尺寸，nano/small/medium/large）。
        image_paths: 测试图像路径列表。
        device: 推理设备（如 ``"cuda:0"``）。
        conf_threshold: 全局置信度阈值（未在 ``class_conf_thresholds`` 中
            列出的类别回退到此值）。
        batch_size: GPU 单次前向的图像数。
        num_workers: 预取 worker 进程数。
        class_conf_thresholds: 逐类置信度阈值表 ``{类别id: 阈值}``；为
            ``None`` 或空字典时所有类别统一使用 ``conf_threshold``。
        la_bias: 推理侧 LA bias：``LaBiasCfg``（由 class_counts.json 按训练侧
            配方重建）或已构建好的 ``torch.Tensor``；``None`` 表示不生效。
        num_classes: 类别数（用于把 bias 补齐到分类头输出通道数）。
        prefetch_factor: 每个 worker 在内存中预取的数据批数。
        precision: 推理精度；``auto`` 在 CUDA 上优先使用 BF16，否则使用 FP16。
        compile_model: 是否编译 detector 前向。
        copy_prefetch: 是否使用 CUDA copy stream 预取 H2D 和 resize。
        warmup_batches: 不计入吞吐的真实 warmup 批次数。
        progress_interval_s: 进度输出的最小时间间隔。
        gpu_monitor_enabled: 是否启用 GPU 利用率采样。
        reason_plugin: 可选的已加载 FFT 一致性插件；``None`` 时保持普通批量推理。
        reason_class_embed: 插件交叉注意力使用的冻结检测器类别嵌入矩阵。
        reason_class_names: 按类别索引排序的名称，用于插件调用。

    Returns:
        ``(pred_records, steady_throughput, gpu_util, timed_images)``：
        ``pred_records`` 为 BoxRecord 列表；``steady_throughput`` 为稳态吞吐
        （img/s）；``gpu_util`` 为推理期间 GPU 平均利用率（%），采样失败时为
        ``None``；``timed_images`` 为参与稳态计时（剔除预热批）的图像数。
    """
    if reason_plugin is not None and (reason_class_embed is None or reason_class_names is None):
        raise ValueError("reason_plugin 需要 reason_class_embed 和 reason_class_names")
    resolution = int(model.model.resolution)
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际为 {batch_size}")
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches 不能为负数，实际为 {warmup_batches}")

    # 按训练侧配方构建 LA bias，并在 postprocess 前修正 logits。
    bias_tensor: torch.Tensor | None = None
    bias_k: float = 1.0
    if la_bias is not None:
        if isinstance(la_bias, torch.Tensor):
            bias_tensor = la_bias
        else:
            bias_tensor = la_bias.build_bias_tensor(num_classes + 1, device)
            bias_k = la_bias.k
            print(f"[i] 推理侧 LA bias 生效: {la_bias.counts_path}（k={la_bias.k}）")

    dataset = _InferenceDataset(image_paths, resize_resolution=resolution if reason_plugin is None else None)
    loader_context = "spawn" if num_workers > 0 and str(device).startswith("cuda") else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=_inference_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        multiprocessing_context=loader_context,
    )

    # 初始化统一 runtime，接管设备、精度、编译和 CUDA 拷贝流。
    runtime = _InferenceRuntime(
        model=model,
        device=device,
        resolution=resolution,
        batch_size=batch_size,
        precision=precision,
        compile_model=compile_model,
        copy_prefetch=copy_prefetch,
    )
    if bias_tensor is not None:
        bias_tensor = bias_tensor.to(device=runtime.device, dtype=torch.float32)

    gpu_monitor = _GpuUtilMonitor(1.0) if gpu_monitor_enabled and runtime.device.type == "cuda" else None
    if gpu_monitor is not None:
        gpu_monitor.start()

    pred_records: list[BoxRecord] = []
    total = len(image_paths)  # 测试图像总数（单轮）
    timed_images = 0
    warmup_images = 0
    bench_start = time.perf_counter()
    warmup_count = 0
    last_progress = bench_start
    # 在推理设备上预建全局和逐类置信度阈值表。
    threshold_table = torch.full(
        (num_classes + 1,),
        float(conf_threshold),
        device=runtime.device,
        dtype=torch.float32,
    )
    if class_conf_thresholds:
        for class_id, threshold in class_conf_thresholds.items():
            if 0 <= int(class_id) < threshold_table.numel():
                threshold_table[int(class_id)] = float(threshold)

    # 每轮批量执行 H2D、预处理、前向、后处理和结果导出。
    for stems, rgb_tensors, orig_sizes in loader:
        staged_images = runtime.stage_batch(rgb_tensors)
        predictions, valid_count = runtime.forward(staged_images)
        if bias_tensor is not None:
            # LA bias 必须在 postprocess 和 top-k 之前生效。
            adjusted_logits = predictions["pred_logits"] - bias_k * bias_tensor.to(
                dtype=predictions["pred_logits"].dtype
            )
            predictions = {**predictions, "pred_logits": adjusted_logits}
        target_sizes = torch.tensor(orig_sizes, device=runtime.device)
        results = model.model.postprocess(predictions, target_sizes=target_sizes)

        if reason_plugin is None:
            # 普通检测只在 GPU 筛选候选，随后每个 batch 统一拷贝到 CPU。
            selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for result in results:
                keep = result["scores"] > threshold_table[result["labels"].long()]
                selected.append((result["boxes"][keep], result["scores"][keep], result["labels"][keep]))
            if selected:
                flat_boxes = torch.cat([item[0] for item in selected], dim=0).float().cpu().numpy()
                flat_scores = torch.cat([item[1] for item in selected], dim=0).float().cpu().numpy()
                flat_labels = torch.cat([item[2] for item in selected], dim=0).cpu().numpy()
                offset = 0
                for stem, item in zip(stems, selected):
                    count = int(item[0].shape[0])
                    for xyxy, class_id, score in zip(
                        flat_boxes[offset : offset + count],
                        flat_labels[offset : offset + count],
                        flat_scores[offset : offset + count],
                    ):
                        pred_records.append(
                            BoxRecord(
                                image_id=stem,
                                class_id=int(class_id),
                                xyxy=tuple(float(v) for v in xyxy),
                                score=float(score),
                            )
                        )
                    offset += count
        else:
            # reason plugin 跨当前 detector batch 合并候选 pair 后只前向一次。
            plugin_samples: list[dict[str, np.ndarray]] = []
            plugin_thresholds: list[np.ndarray] = []
            for rgb_tensor, result in zip(rgb_tensors, results):
                class_ids = result["labels"].long().cpu().numpy()
                plugin_samples.append(
                    {
                        "image": np.ascontiguousarray(rgb_tensor.permute(1, 2, 0).numpy()),
                        "boxes": result["boxes"].float().cpu().numpy(),
                        "scores": result["scores"].float().cpu().numpy(),
                        "classes": class_ids,
                    }
                )
                plugin_thresholds.append(
                    np.asarray(
                        [class_conf_thresholds.get(int(label), conf_threshold) for label in class_ids]
                        if class_conf_thresholds
                        else np.full(class_ids.shape, conf_threshold, dtype=np.float32),
                        dtype=np.float32,
                    )
                )
            if hasattr(reason_plugin, "predict_detections_batch"):
                plugin_outputs = reason_plugin.predict_detections_batch(
                    plugin_samples,
                    reason_class_names,
                    reason_class_embed,
                    device,
                    target_thresholds=plugin_thresholds,
                    reason_class_ids=reason_plugin.config.reason_class_ids,
                    filter_final=False,
                )
            else:
                plugin_outputs = [
                    _rescore_reason_candidates(
                        reason_plugin,
                        reason_class_embed,
                        reason_class_names,
                        sample["image"],
                        sample["boxes"],
                        sample["scores"],
                        sample["classes"],
                        conf_threshold,
                        class_conf_thresholds,
                        device,
                    )
                    for sample in plugin_samples
                ]
            for stem, (boxes, scores, class_ids), per_class_thr in zip(stems, plugin_outputs, plugin_thresholds):
                keep = scores > per_class_thr
                for xyxy, class_id, score in zip(boxes[keep], class_ids[keep], scores[keep]):
                    pred_records.append(
                        BoxRecord(
                            image_id=stem,
                            class_id=int(class_id),
                            xyxy=tuple(float(v) for v in xyxy),
                            score=float(score),
                        )
                    )

        if warmup_count < warmup_batches:
            warmup_count += 1
            warmup_images += valid_count
            if warmup_count == warmup_batches:
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                bench_start = time.perf_counter()
            continue
        timed_images += valid_count
        now = time.perf_counter()
        if progress_interval_s > 0 and now - last_progress >= progress_interval_s:
            print_progress(timed_images, max(total - warmup_images, 1), bench_start, device)
            last_progress = now

    if gpu_monitor is not None:
        gpu_monitor.stop()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    if progress_interval_s > 0:
        print()

    steady_elapsed = time.perf_counter() - bench_start
    steady_throughput = timed_images / steady_elapsed if steady_elapsed > 0 else 0.0
    return pred_records, steady_throughput, gpu_monitor.average_utilization() if gpu_monitor else None, timed_images


def predict_mixed_to_records(
    model: RFDETR,
    image_paths: list[Path],
    crop_sources: list[tuple[str, Path | np.ndarray, tuple[int, int, int, int]]],
    device: str,
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
    crop_conf_threshold: float | None = None,
    max_pending_crops: int = 128,
    class_conf_thresholds: Mapping[int, float] | None = None,
    *,
    la_bias: LaBiasCfg | torch.Tensor | None = None,
    num_classes: int = 25,
    prefetch_factor: int = 3,
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto",
    compile_model: bool = False,
    copy_prefetch: bool = True,
    warmup_batches: int = 1,
    progress_interval_s: float = 1.0,
    gpu_monitor_enabled: bool = False,
    reason_plugin: Any | None = None,
    reason_class_embed: torch.Tensor | None = None,
    reason_class_names: list[str] | None = None,
    roi_backend: str = "auto",
    roi_output_size: int | None = None,
    roi_cache_dir: str | Path | None = None,
    strict_roi_backend: bool = False,
    roi_queue_size: int = 128,
) -> tuple[list[BoxRecord], float, float | None, int, float]:
    """将普通图片和大图 crop 放入同一个 detector runtime 批量推理。"""
    if reason_plugin is not None and (reason_class_embed is None or reason_class_names is None):
        raise ValueError("reason_plugin 需要 reason_class_embed 和 reason_class_names")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际为 {batch_size}")
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches 不能为负数，实际为 {warmup_batches}")
    if not image_paths and not crop_sources:
        return [], 0.0, None, 0, 0.0

    from scripts.large_cut.image_source import create_image_source

    resolution = int(model.model.resolution)
    crop_conf_threshold = conf_threshold if crop_conf_threshold is None else float(crop_conf_threshold)
    max_pending_crops = max(1, int(max_pending_crops))
    roi_queue_size = max(1, int(roi_queue_size))
    bias_tensor: torch.Tensor | None = None
    bias_k = 1.0
    if la_bias is not None:
        if isinstance(la_bias, torch.Tensor):
            bias_tensor = la_bias
        else:
            bias_tensor = la_bias.build_bias_tensor(num_classes + 1, device)
            bias_k = la_bias.k
            print(f"[i] 推理侧 LA bias 生效: {la_bias.counts_path}（k={la_bias.k}）")

    # crop 存在时让 DataLoader 预取半批小图，剩余位置由 crop 填充。
    small_loader_batch = batch_size if not crop_sources else max(1, batch_size // 2)
    dataset = _InferenceDataset(image_paths, resize_resolution=resolution if reason_plugin is None else None)
    loader_context = "spawn" if num_workers > 0 and str(device).startswith("cuda") else None
    loader = DataLoader(
        dataset,
        batch_size=small_loader_batch,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=_inference_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        multiprocessing_context=loader_context,
    )

    # 小图和 crop 共用一个 runtime，避免重复创建 stream 或重复 compile。
    runtime = _InferenceRuntime(
        model=model,
        device=device,
        resolution=resolution,
        batch_size=batch_size,
        precision=precision,
        compile_model=compile_model,
        copy_prefetch=copy_prefetch,
    )
    if bias_tensor is not None:
        bias_tensor = bias_tensor.to(device=runtime.device, dtype=torch.float32)
    gpu_monitor = _GpuUtilMonitor(1.0) if gpu_monitor_enabled and runtime.device.type == "cuda" else None
    if gpu_monitor is not None:
        gpu_monitor.start()

    pred_records: list[BoxRecord] = []
    small_iterator = iter(loader)
    small_buffer: deque[tuple[str, torch.Tensor, tuple[int, int]]] = deque()
    small_done = not bool(image_paths)
    crop_index = 0
    crop_buffer: deque[tuple[str, torch.Tensor, tuple[int, int], tuple[int, int]]] = deque()
    # ROI 解码线程数独立限流，避免多个 OpenCV 全图解码同时放大内存。
    roi_executor = ThreadPoolExecutor(max_workers=max(1, min(num_workers or 1, 4))) if crop_sources else None
    roi_futures: deque[Future[tuple[str, torch.Tensor, tuple[int, int], tuple[int, int]]]] = deque()
    # OpenCV 回退会把整张大图解码到内存；限制缓存数量，避免 215 张大图常驻内存。
    roi_source_cache: OrderedDict[str, Any] = OrderedDict()
    roi_source_cache_limit = max(1, min(4, max_pending_crops // max(batch_size, 1)))
    roi_source_lock = threading.Lock()
    total_samples = len(image_paths) + len(crop_sources)
    timed_samples = 0
    warmup_samples = 0
    warmup_count = 0
    bench_start = time.perf_counter()
    last_progress = bench_start
    loaded_crops = 0
    processed_samples = 0
    last_prepare_progress = bench_start
    print(
        f"[i] 统一 detector 开始: 小图 {len(image_paths)} 张，"
        f"crop {len(crop_sources)} 个，batch={batch_size}",
        flush=True,
    )

    def refill_small() -> None:
        """从 DataLoader 预取下一批小图。"""
        nonlocal small_done
        if small_done:
            return
        try:
            stems, tensors, sizes = next(small_iterator)
        except StopIteration:
            small_done = True
            return
        small_buffer.extend(zip(stems, tensors, sizes, strict=True))

    def refill_crops() -> None:
        """按上限生成待检测 crop，避免一次性创建全部像素数组。"""
        nonlocal crop_index, loaded_crops, last_prepare_progress
        # 只预取约两个 detector batch，避免已完成 future 持有过多 ROI 像素张量。
        limit = min(max_pending_crops, roi_queue_size, max(batch_size * 2, batch_size))

        def load_one(
            item: tuple[str, Path | np.ndarray, tuple[int, int, int, int]],
        ) -> tuple[str, torch.Tensor, tuple[int, int], tuple[int, int]]:
            stem, source_path, crop_xyxy = item
            if isinstance(source_path, np.ndarray):
                # 仅保留测试和外部内存源的直接 ROI 读取，不会进入新的大图切分路径。
                x0, y0, x1, y1 = crop_xyxy
                original_crop_size = (y1 - y0, x1 - x0)
                crop_rgb = np.ascontiguousarray(source_path[y0:y1, x0:x1])
                if reason_plugin is None:
                    target_size = roi_output_size or resolution
                    crop_rgb = cv2.resize(
                        crop_rgb,
                        (target_size, target_size),
                        interpolation=cv2.INTER_AREA,
                    )
            else:
                source_key = str(source_path)
                with roi_source_lock:
                    source = roi_source_cache.get(source_key)
                    if source is None:
                        source = create_image_source(
                            source_path,
                            backend=roi_backend,
                            cache_dir=roi_cache_dir,
                            strict=strict_roi_backend,
                        )
                        roi_source_cache[source_key] = source
                        roi_source_cache.move_to_end(source_key)
                        while len(roi_source_cache) > roi_source_cache_limit:
                            roi_source_cache.popitem(last=False)
                    else:
                        roi_source_cache.move_to_end(source_key)
                crop_rgb, roi_size = source.read_roi(
                    crop_xyxy,
                    output_size=None if reason_plugin is not None else (roi_output_size or resolution),
                )
                # target_sizes 必须使用 resize 前的 ROI 尺寸；归一化框再由 postprocess
                # 直接还原到 ROI 原始坐标，最后才能正确叠加原图 offset。
                original_crop_size = (roi_size[1], roi_size[0])
            return (
                stem,
                torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1),
                original_crop_size,
                (crop_xyxy[0], crop_xyxy[1]),
            )

        while len(crop_buffer) + len(roi_futures) < limit and crop_index < len(crop_sources):
            item = crop_sources[crop_index]
            crop_index += 1
            if roi_executor is None:
                crop_buffer.append(load_one(item))
                loaded_crops += 1
            else:
                roi_futures.append(roi_executor.submit(load_one, item))
        if roi_futures and progress_interval_s > 0:
            print(
                f"\r[i] ROI 预取中: 已提交 {crop_index:>5d}/{len(crop_sources):<5d} "
                f"| 等待首批 {min(batch_size, len(roi_futures)):d} 个 crop",
                end="",
                flush=True,
            )
        # 至少补满一个 detector batch，避免每次前向只拿到一个 crop。
        target_buffer = min(batch_size, len(crop_sources) - crop_index + len(roi_futures))
        while len(crop_buffer) < target_buffer and roi_futures:
            crop_buffer.append(roi_futures.popleft().result())
            loaded_crops += 1
            now = time.perf_counter()
            if progress_interval_s > 0 and now - last_prepare_progress >= progress_interval_s:
                print(
                    f"\r[i] ROI 加载进度 {loaded_crops:>5d}/{len(crop_sources):<5d} "
                    f"| detector 等待 batch | 待处理 {len(roi_futures):d}",
                    end="",
                    flush=True,
                )
                last_prepare_progress = now

    while small_buffer or crop_buffer or roi_futures or crop_index < len(crop_sources) or not small_done:
        if not small_buffer and not small_done:
            refill_small()
        if not crop_buffer and (crop_index < len(crop_sources) or roi_futures):
            refill_crops()

        batch_stems: list[str] = []
        batch_tensors: list[torch.Tensor] = []
        batch_sizes: list[tuple[int, int]] = []
        batch_offsets: list[tuple[int, int] | None] = []
        batch_default_thresholds: list[float] = []
        if small_buffer and crop_buffer:
            take_small = min(len(small_buffer), max(1, batch_size // 2))
        else:
            take_small = min(len(small_buffer), batch_size)
        for _ in range(take_small):
            stem, tensor, size = small_buffer.popleft()
            batch_stems.append(stem)
            batch_tensors.append(tensor)
            batch_sizes.append(size)
            batch_offsets.append(None)
            batch_default_thresholds.append(float(conf_threshold))

        while len(batch_tensors) < batch_size and crop_buffer:
            stem, crop_tensor, crop_size, crop_offset = crop_buffer.popleft()
            batch_stems.append(stem)
            batch_tensors.append(crop_tensor)
            batch_sizes.append(crop_size)
            batch_offsets.append(crop_offset)
            batch_default_thresholds.append(crop_conf_threshold)

        while len(batch_tensors) < batch_size and small_buffer:
            stem, tensor, size = small_buffer.popleft()
            batch_stems.append(stem)
            batch_tensors.append(tensor)
            batch_sizes.append(size)
            batch_offsets.append(None)
            batch_default_thresholds.append(float(conf_threshold))

        if not batch_tensors:
            continue

        # 首个前向可能因 OpenCV 解码或模型预热耗时较长，先输出 batch 起始状态。
        processed_samples += len(batch_tensors)
        if progress_interval_s > 0:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s or processed_samples == len(batch_tensors):
                print(
                    f"\r[i] detector batch 开始: 已取 {processed_samples:>5d}/{total_samples:<5d} "
                    f"| batch={len(batch_tensors):d} | ROI 已加载 {loaded_crops:d}",
                    end="",
                    flush=True,
                )
                last_progress = now

        staged_images = runtime.stage_batch(batch_tensors)
        predictions, valid_count = runtime.forward(staged_images)
        if bias_tensor is not None:
            # LA bias 必须在 postprocess 和 top-k 之前生效。
            adjusted_logits = predictions["pred_logits"] - bias_k * bias_tensor.to(
                dtype=predictions["pred_logits"].dtype
            )
            predictions = {**predictions, "pred_logits": adjusted_logits}
        target_sizes = torch.tensor(batch_sizes, device=runtime.device)
        results = model.model.postprocess(predictions, target_sizes=target_sizes)

        if reason_plugin is None:
            selected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for result, default_threshold in zip(results, batch_default_thresholds, strict=True):
                labels = result["labels"].long()
                keep = result["scores"] > default_threshold
                if class_conf_thresholds:
                    for class_id, threshold in class_conf_thresholds.items():
                        keep = torch.where(labels == int(class_id), result["scores"] > float(threshold), keep)
                selected.append((result["boxes"][keep], result["scores"][keep], result["labels"][keep]))
            if selected:
                flat_boxes = torch.cat([item[0] for item in selected], dim=0).float().cpu().numpy()
                flat_scores = torch.cat([item[1] for item in selected], dim=0).float().cpu().numpy()
                flat_labels = torch.cat([item[2] for item in selected], dim=0).cpu().numpy()
                offset = 0
                for stem, item, crop_offset in zip(batch_stems, selected, batch_offsets, strict=True):
                    count = int(item[0].shape[0])
                    for xyxy, class_id, score in zip(
                        flat_boxes[offset : offset + count],
                        flat_labels[offset : offset + count],
                        flat_scores[offset : offset + count],
                        strict=True,
                    ):
                        box = xyxy.astype(np.float32, copy=True)
                        if crop_offset is not None:
                            box[[0, 2]] += crop_offset[0]
                            box[[1, 3]] += crop_offset[1]
                        pred_records.append(
                            BoxRecord(
                                image_id=stem,
                                class_id=int(class_id),
                                xyxy=tuple(float(value) for value in box),
                                score=float(score),
                            )
                        )
                    offset += count
        else:
            # plugin 在当前混合 batch 中一次处理多张图的候选 pair。
            plugin_samples: list[dict[str, np.ndarray]] = []
            plugin_thresholds: list[np.ndarray] = []
            for tensor, result, default_threshold in zip(batch_tensors, results, batch_default_thresholds, strict=True):
                class_ids = result["labels"].long().cpu().numpy()
                plugin_samples.append(
                    {
                        "image": np.ascontiguousarray(tensor.permute(1, 2, 0).numpy()),
                        "boxes": result["boxes"].float().cpu().numpy(),
                        "scores": result["scores"].float().cpu().numpy(),
                        "classes": class_ids,
                    }
                )
                plugin_thresholds.append(
                    np.asarray(
                        [class_conf_thresholds.get(int(label), default_threshold) for label in class_ids]
                        if class_conf_thresholds
                        else np.full(class_ids.shape, default_threshold, dtype=np.float32),
                        dtype=np.float32,
                    )
                )
            plugin_outputs = reason_plugin.predict_detections_batch(
                plugin_samples,
                reason_class_names,
                reason_class_embed,
                device,
                target_thresholds=plugin_thresholds,
                reason_class_ids=reason_plugin.config.reason_class_ids,
                filter_final=False,
            )
            for stem, (boxes, scores, class_ids), thresholds, crop_offset in zip(
                batch_stems,
                plugin_outputs,
                plugin_thresholds,
                batch_offsets,
                strict=True,
            ):
                keep = scores > thresholds
                for xyxy, class_id, score in zip(boxes[keep], class_ids[keep], scores[keep], strict=True):
                    box = xyxy.astype(np.float32, copy=True)
                    if crop_offset is not None:
                        box[[0, 2]] += crop_offset[0]
                        box[[1, 3]] += crop_offset[1]
                    pred_records.append(
                        BoxRecord(
                            image_id=stem,
                            class_id=int(class_id),
                            xyxy=tuple(float(value) for value in box),
                            score=float(score),
                        )
                    )

        if warmup_count < warmup_batches:
            warmup_count += 1
            warmup_samples += valid_count
            if warmup_count == warmup_batches:
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                bench_start = time.perf_counter()
            continue
        timed_samples += valid_count
        now = time.perf_counter()
        if progress_interval_s > 0 and now - last_progress >= progress_interval_s:
            print_progress(timed_samples, max(total_samples - warmup_samples, 1), bench_start, device)
            last_progress = now

    if gpu_monitor is not None:
        gpu_monitor.stop()
    if roi_executor is not None:
        roi_executor.shutdown(wait=True)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    if progress_interval_s > 0:
        print()
    elapsed = time.perf_counter() - bench_start
    throughput = timed_samples / elapsed if elapsed > 0 else 0.0
    return pred_records, throughput, gpu_monitor.average_utilization() if gpu_monitor else None, timed_samples, elapsed


# 完整评估主流程。


def _classify_large_images(
    test_image_paths: list[Path],
    image_size_map: dict[str, tuple[int, int]] | None,
    min_side: int,
) -> set[str]:
    """按长边阈值判定大图 image_id 集合。

    优先使用已构建的 ``image_size_map``（YOLO 标签数据集）；不可用时用 PIL
    只读文件头获取尺寸（不完整解码，兼容 COCO 等格式）。

    Args:
        test_image_paths: 测试图像路径列表。
        image_size_map: 可选的 ``{image_id: (w, h)}`` 尺寸映射。
        min_side: 大图长边阈值（像素），长边 ≥ 该值的图像视为大图。

    Returns:
        大图 image_id 集合。
    """
    if min_side <= 0:
        return set()
    if image_size_map is not None:
        return {stem for stem, (w, h) in image_size_map.items() if max(w, h) >= min_side}
    from PIL import Image

    large_ids: set[str] = set()
    for img_path in test_image_paths:
        with Image.open(img_path) as pil_img:
            if max(pil_img.size) >= min_side:
                large_ids.add(img_path.stem)
    return large_ids


def run_evaluation(
    dataset: DatasetCfg,
    infer: InferenceCfg,
    checkpoint_path: str | Path,
    *,
    save_fp_fn: bool = True,
    save_yolo_preds: bool = False,
    la_bias: LaBiasCfg | None = None,
    reason_plugin_cfg: ReasonPluginCfg | None = None,
    resolution: int | None = None,
    large_image_cfg: LargeImageCfg | None = None,
) -> None:
    """按比赛口径在测试集上完整评估一个 checkpoint。

    流程：读测试图像与真实框 → 加载模型（含语义头重建）→ 批量流水线推理 →
    比赛评估（大类聚合 + 逐类 + macro 平均 + 总指标）→ 混淆矩阵与 FP/FN 可视化
    → 写 ``test_result.txt`` 到实验目录。

    Args:
        dataset: 数据集配置（``build_dataset_cfg`` 的产物）。
        infer: 推理参数。
        checkpoint_path: 权重文件路径。
        save_fp_fn: 是否保存 FP/FN 可视化。
        save_yolo_preds: 是否输出 YOLO 格式预测框（每图一个 txt，供归因脚本使用）。
        la_bias: 推理侧 LA bias 配置（``None`` 表示不生效）。
        reason_plugin_cfg: FFT 一致性插件配置；``None`` 时保持普通测试流程。
        resolution: 可选：推理输入分辨率。构造参数优先于 checkpoint 记录的
            ``model_config``（例如 nano 以 704 训练时强制 704 推理）；
            ``None`` 用 checkpoint 记录的分辨率。
        large_image_cfg: 大图切分测试配置。传入时，长边 ≥ ``min_side`` 的
            图像走 nano 边界检测切分流程（nano 只加载一次），其余小图仍按
            整图推理；大图 FP/FN 可视化单独保存到 ``output_dir/large_errors/``。
            ``None`` 表示不启用（保持原逻辑，全部整图推理）。
    """
    os.chdir(PROJECT_ROOT)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    test_image_paths = read_test_image_paths(dataset.test_image_dir)
    # YOLO 格式需要图像尺寸把归一化坐标换算成像素；COCO 的 bbox 本身就是像素坐标
    image_size_map = build_image_size_map(test_image_paths) if dataset.label_format == "yolo" else None

    # 大图判定（供推理分流与 FP/FN 可视化共用）
    large_image_ids: set[str] = set()
    if large_image_cfg is not None and large_image_cfg.min_side > 0:
        large_image_ids = _classify_large_images(
            test_image_paths,
            image_size_map,
            large_image_cfg.min_side,
        )
    large_errors_dir: Path | None = dataset.exp_output_dir / "large_errors" if large_image_ids else None
    if large_image_ids:
        print(
            f"[i] 大图切分评估启用：{len(large_image_ids)} 张大图"
            f"（长边≥{large_image_cfg.min_side if large_image_cfg else 0}px）"
        )

    # 读取测试集真实框（按数据集标签格式分派）
    if dataset.label_format == "yolo":
        assert dataset.label_dir is not None
        gt_records = load_yolo_labels(dataset.label_dir, image_size_map)
    elif dataset.label_format == "coco":
        assert dataset.annotation_file is not None
        gt_records = load_coco_labels(dataset.annotation_file)
    else:
        raise ValueError(f"不支持的标签格式: {dataset.label_format}")

    # 边界阶段先独立完成，随后释放边界模型再创建主检测器。
    device = resolve_device(infer.device)
    use_large_branch = large_image_cfg is not None and large_image_ids and large_image_cfg.boundary_checkpoint
    small_paths = [p for p in test_image_paths if p.stem not in large_image_ids] if use_large_branch else []
    large_paths = [p for p in test_image_paths if p.stem in large_image_ids] if use_large_branch else []
    crop_sources: list[tuple[str, Path, tuple[int, int, int, int]]] = []
    crop_boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    large_prepare_stats: dict[str, float] = {}
    large_tiler: Any | None = None
    if use_large_branch:
        from scripts.large_cut.large_image_tiler import LargeImageTiler

        large_tiler = LargeImageTiler(
            large_image_cfg.boundary_checkpoint,
            boundary_backend=large_image_cfg.boundary_backend,
            boundary_resolution=large_image_cfg.boundary_resolution,
            boundary_conf=large_image_cfg.boundary_conf,
            padding=large_image_cfg.padding,
            nms_iou=large_image_cfg.nms_iou,
            square_stretch=large_image_cfg.square_stretch,
            device=device,
            batch_size=large_image_cfg.batch_size,
            num_workers=large_image_cfg.num_workers,
            roi_backend=large_image_cfg.roi_backend,
            proxy_max_side=large_image_cfg.proxy_max_side,
            roi_cache_dir=large_image_cfg.roi_cache_dir,
            strict_roi_backend=large_image_cfg.strict_roi_backend,
            progress_interval_s=infer.progress_interval_s,
        )
        crop_sources, crop_boxes_by_image, large_prepare_stats = large_tiler.prepare_crops(large_paths)

    # 只加载一次 RF-DETR，模型尺寸由 checkpoint 中的 model_name 自动确定。
    print(f"[i] 正在从 {checkpoint_path} 加载 RF-DETR 模型...")
    ckpt_kwargs: dict[str, int] = {}
    if resolution is not None:
        ckpt_kwargs["resolution"] = resolution
        print(f"[i] 推理分辨率覆盖为: {resolution}")
    model = RFDETR.from_checkpoint(str(checkpoint_path), **ckpt_kwargs)
    test_resolution = int(model.model.resolution)
    print(f"[i] 已加载模型: {type(model).__name__} | 分辨率: {int(model.model.resolution)}")
    # from_checkpoint 已用同一份 state dict 完成语义头重建。

    # reason plugin 只加载一次，普通图和大图 crop 共用同一实例并保持 FP32。
    reason_plugin = None
    reason_class_embed: torch.Tensor | None = None
    reason_class_names: list[str] | None = None
    if reason_plugin_cfg is not None:
        from rfdetr.reasoning import PluginLoader

        plugin_path = Path(reason_plugin_cfg.checkpoint)
        if not plugin_path.exists():
            raise FileNotFoundError(f"reason_plugin checkpoint 不存在: {plugin_path}")
        reason_plugin = PluginLoader.load(plugin_path)
        if reason_plugin.num_classes != dataset.num_classes:
            raise ValueError(
                "reason_plugin 的 num_classes 与测试数据集不一致: "
                f"{reason_plugin.num_classes} != {dataset.num_classes}"
            )
        reason_plugin.config.reason_class_ids = reason_plugin_cfg.class_ids
        if reason_plugin_cfg.conf_low is not None:
            reason_plugin.config.conf_low = reason_plugin_cfg.conf_low
        reason_plugin = reason_plugin.to(device).float().eval()
        class_embed = getattr(model.model.model, "class_embed", None)
        if class_embed is None or not hasattr(class_embed, "weight"):
            raise RuntimeError("reason_plugin 需要检测器的 class_embed.weight")
        reason_class_embed = class_embed.weight.detach().to(device=device, dtype=torch.float32)
        decoder_layers = getattr(getattr(reason_plugin, "decoder", None), "layers", ())
        expected_kv_dim = getattr(decoder_layers[0], "kv_dim", None) if decoder_layers else None
        if expected_kv_dim is not None and reason_class_embed.shape[1] != expected_kv_dim:
            raise ValueError(
                "reason_plugin checkpoint expects class embeddings with width "
                f"{expected_kv_dim}, but the detector provides {reason_class_embed.shape[1]}"
            )
        reason_class_names = [dataset.class_names[class_id] for class_id in range(dataset.num_classes)]
        print(
            f"[i] 启用 FFT 一致性插件: {plugin_path} "
            f"（class_ids={reason_plugin.config.reason_class_ids}, conf_low={reason_plugin.config.conf_low}）"
        )

    # 小图和已准备好的 crop 统一进入同一个 detector runtime。
    large_image_stats: dict[str, float] | None = None
    if use_large_branch:
        pred_records, throughput, gpu_util, timed_images, detector_elapsed = predict_mixed_to_records(
            model,
            small_paths,
            crop_sources,
            device,
            conf_threshold=infer.conf_threshold,
            class_conf_thresholds=infer.class_conf_thresholds,
            batch_size=infer.batch_size,
            num_workers=infer.num_workers,
            crop_conf_threshold=large_image_cfg.detector_conf,
            max_pending_crops=large_image_cfg.max_pending_crops,
            roi_backend=large_image_cfg.roi_backend,
            roi_output_size=large_image_cfg.roi_output_size,
            roi_cache_dir=large_image_cfg.roi_cache_dir,
            strict_roi_backend=large_image_cfg.strict_roi_backend,
            roi_queue_size=large_image_cfg.roi_queue_size,
            la_bias=la_bias,
            num_classes=dataset.num_classes,
            prefetch_factor=infer.prefetch_factor,
            precision=infer.precision,
            compile_model=infer.compile_model,
            copy_prefetch=infer.copy_prefetch,
            warmup_batches=infer.warmup_batches,
            progress_interval_s=infer.progress_interval_s,
            gpu_monitor_enabled=infer.gpu_monitor_enabled,
            reason_plugin=reason_plugin,
            reason_class_embed=reason_class_embed,
            reason_class_names=reason_class_names,
        )
        if large_paths:
            end_to_end = (
                large_prepare_stats.get("boundary_seconds", 0.0)
                + large_prepare_stats.get("crop_prepare_seconds", 0.0)
                + detector_elapsed
            )
            average = end_to_end / len(large_paths)
            large_image_stats = {
                "count": float(len(large_paths)),
                "avg": average,
                "max": average,
                "total": end_to_end,
                "boundary_seconds": large_prepare_stats.get("boundary_seconds", 0.0),
                "proxy_seconds": large_prepare_stats.get("proxy_seconds", 0.0),
                "fallback_count": large_prepare_stats.get("fallback_count", 0.0),
                "detector_seconds": detector_elapsed,
                "end_to_end_seconds": end_to_end,
            }
    else:
        if large_image_cfg is not None and large_image_ids and not large_image_cfg.boundary_checkpoint:
            print("[w] 已启用大图切分但未配置 boundary_checkpoint，回退为整图推理（大图会被缩小）。")
        pred_records, throughput, gpu_util, timed_images = predict_batched_to_records(
            model,
            test_image_paths,
            device,
            conf_threshold=infer.conf_threshold,
            class_conf_thresholds=infer.class_conf_thresholds,
            batch_size=infer.batch_size,
            num_workers=infer.num_workers,
            la_bias=la_bias,
            num_classes=dataset.num_classes,
            prefetch_factor=infer.prefetch_factor,
            precision=infer.precision,
            compile_model=infer.compile_model,
            copy_prefetch=infer.copy_prefetch,
            warmup_batches=infer.warmup_batches,
            progress_interval_s=infer.progress_interval_s,
            gpu_monitor_enabled=infer.gpu_monitor_enabled,
            reason_plugin=reason_plugin,
            reason_class_embed=reason_class_embed,
            reason_class_names=reason_class_names,
        )
    del model
    release_cuda_cache(device)

    # 可选保存 YOLO 格式预测，供漏检和虚警归因脚本使用。
    if save_yolo_preds:
        save_yolo_predictions(pred_records, dataset.exp_output_dir / "yolo_preds", image_size_map)

    # 输出标准图和大图的推理吞吐统计。
    print("=" * 80)
    print("推理测速结果")
    print(f"GPU 批量大小: {infer.batch_size}  |  CPU 预取 worker 数: {infer.num_workers}")
    elapsed = timed_images / throughput if throughput > 0 else 0.0
    print(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {elapsed:.1f}s）")
    if gpu_util is not None:
        print(f"推理期间 GPU 平均利用率: {gpu_util:.1f}%")
    if large_image_stats is not None:
        print(
            f"大图目标检测（裁切推理）: {int(large_image_stats['count'])} 张 | "
            f"平均 {large_image_stats['avg']:.2f}s | 最大 {large_image_stats['max']:.2f}s | "
            f"合计 {large_image_stats['total']:.2f}s"
        )
        if "proxy_seconds" in large_image_stats:
            print(f"proxy 读取合计: {float(large_image_stats['proxy_seconds']):.2f}s")
        if "fallback_count" in large_image_stats:
            print(f"ROI 后端 fallback 图像数: {int(large_image_stats['fallback_count'])}")
    print("=" * 80)

    # 按比赛规则计算大类、细类和总指标。
    config = EvalConfig(
        class_to_group=dataset.class_to_group,
        group_iou_thresholds=dataset.group_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    metric_excluded_ids = dataset.metric_excluded_class_ids
    metric_gt_records = filter_auxiliary_metric_records(gt_records, metric_excluded_ids)
    metric_pred_records = filter_auxiliary_metric_records(pred_records, metric_excluded_ids)
    eval_results = evaluate_competition_metrics(metric_gt_records, metric_pred_records, config)

    print("=" * 80)
    print("比赛指标评估结果（测试集）")
    print(f"权重: {checkpoint_path}")
    print(f"测试图像数: {len(test_image_paths)}")
    print(f"真实框数: {len(gt_records)}")
    print(f"预测框数: {len(pred_records)}")
    if metric_excluded_ids:
        excluded_names = "，".join(
            f"{dataset.class_names.get(class_id, str(class_id))}({class_id})"
            for class_id in sorted(metric_excluded_ids)
        )
        print(f"大类与总指标排除的辅助类别: {excluded_names}")
    print(f"置信度阈值: {infer.conf_threshold}")
    if infer.class_conf_thresholds:
        print(
            "逐类阈值: "
            + "，".join(
                f"{dataset.class_names.get(cid, str(cid))}={thr:.2f}"
                for cid, thr in sorted(infer.class_conf_thresholds.items())
            )
        )
    print("IoU 阈值: " + "，".join(f"{key}={value:.2f}" for key, value in dataset.group_iou_thresholds.items()))
    print("=" * 80)

    print_eval_result("all", eval_results["all"])
    for group_name, group_result in eval_results["groups"].items():
        print_eval_result(group_name, group_result)

    # 输出细粒度逐类指标。
    per_class_config = EvalConfig(
        class_to_group=dataset.per_class_to_group,
        group_iou_thresholds=dataset.per_class_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    per_class_results = evaluate_competition_metrics(
        gt_records,
        pred_records,
        per_class_config,
    )

    print("\n" + "-" * 80)
    print("细粒度类别指标")
    print("-" * 80)
    for class_name in sorted(per_class_results["groups"].keys()):
        result = per_class_results["groups"][class_name]
        if result.tp + result.fn > 0:  # 只打印测试集中存在的类别
            print_eval_result(class_name, result)

    # 输出各大类的 macro 平均和总指标。
    group_macro = compute_group_macro_averages(
        per_class_results["groups"],
        build_metric_class_to_group(dataset),
        dataset.class_names,
    )
    total_macro = compute_total_metrics(group_macro)

    print("\n" + "-" * 80)
    print("每个大类下小类指标的平均值（macro 平均）")
    print("-" * 80)
    for group_name, macro in group_macro.items():
        print(format_macro_line(group_name, macro))

    print("\n" + "-" * 80)
    print("总指标（各大类平均指标再取算术平均，即（船+飞机+车辆）/3）")
    print("-" * 80)
    print(format_macro_line("total", total_macro))

    # 生成混淆矩阵可视化。
    print("\n[i] 正在生成混淆矩阵分析图...")
    cm = build_confusion_matrix(
        gt_records=gt_records,
        pred_records=pred_records,
        num_classes=dataset.num_classes,
        vehicle_class_ids=dataset.vehicle_class_ids,
    )
    plot_confusion_matrix(
        matrix=cm,
        class_names=dataset.class_names,
        output_path=str(dataset.exp_output_dir / "confusion_matrix.png"),
    )
    print(f"[完成] 混淆矩阵已保存至: {dataset.exp_output_dir / 'confusion_matrix.png'}")

    # 保存 FP 和 FN 可视化结果。
    if save_fp_fn:
        print("\n[i] 正在生成 FP/FN 可视化...")
        clear_vis_dirs(
            dataset.exp_output_dir / "FP",
            dataset.exp_output_dir / "FN",
            dataset.class_names,
            large_errors_dir=large_errors_dir,
        )
        fp_img, fn_img, fp_box, fn_box, tp_pred = match_per_image_per_class(
            gt_records,
            pred_records,
            dataset.num_classes,
            dataset.vehicle_class_ids,
        )
        save_fp_fn_visualizations(
            fp_img,
            fn_img,
            fp_box,
            fn_box,
            tp_pred,
            gt_records,
            test_image_paths,
            dataset.class_names,
            dataset.exp_output_dir / "FP",
            dataset.exp_output_dir / "FN",
            large_image_ids=large_image_ids,
            large_errors_dir=large_errors_dir,
            large_crop_boxes=crop_boxes_by_image,
        )
        print("[完成] FP/FN 可视化保存完成")

    # 将完整评估报告写入实验目录。
    report_lines = build_test_report(
        dataset_name=dataset.name,
        checkpoint_path=checkpoint_path,
        test_image_paths=test_image_paths,
        gt_records=gt_records,
        pred_records=pred_records,
        throughput=throughput,
        timed_images=timed_images,
        gpu_util=gpu_util,
        eval_results=eval_results,
        group_macro=group_macro,
        total_macro=total_macro,
        per_class_results=per_class_results,
        dataset=dataset,
        infer=infer,
        test_resolution=test_resolution,
        large_errors_dir=large_errors_dir,
        large_image_stats=large_image_stats,
    )
    write_test_result(report_lines, dataset.exp_output_dir / "test_result.txt")
