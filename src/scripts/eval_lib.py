# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试评估库：比赛评分推理管线 + 数据集配置 + 指标计算（由原 test.py 拆分）。

本模块承载原 ``test.py`` 的全部函数与配置，供三个入口使用：

1. **``test.py`` 薄模板**：读 yaml 配置后调用 ``run_evaluation`` 完成完整评估；
2. **``analysis/`` 探针/诊断脚本**：复用 ``read_test_image_paths``、``build_image_size_map``、
   ``predict_batched_to_records`` 等推理函数；
3. **``ret-sscl/`` 消融评估脚本**：复用推理管线与比赛指标计算。

推理阶段使用“多进程预取解码 + GPU 批量前向”的流水线：DataLoader 的多个 worker
进程在后台并行完成图像解码与换色，float 化 / 缩放 / 归一化在 GPU 端批量执行，
从而隐藏 CPU 预处理延迟，让 GPU 持续满载。预测结果与逐张 ``model.predict``
完全一致。

模块级配置已收敛为三个 dataclass：

- ``DatasetCfg``：数据集配置（原 ``DATASET_CONFIGS`` 条目解析后的形态）；
- ``InferenceCfg``：推理参数（原 ``CONF_THRESHOLD``/``DEVICE``/``BATCH_SIZE``/
  ``NUM_WORKERS``/``PREFETCH_FACTOR``/``USE_FP16`` 等模块级常量）；
- ``LaBiasCfg``：推理侧 Logit Adjustment bias 参数（原 ``LOGIT_ADJUSTMENT_BIAS_*``
  模块级常量，替代 calibrate_thresholds 临时改写的 save/restore 模式）。
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 类别名称统一来自 sscl/prompts/*.yaml，保证与语义矩阵的类别索引一致
from rfdetr import RFDETR  # noqa: E402
from rfdetr.sscl.prompts import DIOR_CLASS_NAMES, SHWX_CLASS_NAMES  # noqa: E402
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
    save_fp_fn_visualizations,
)


def _label_keyed_names(names_by_id: dict[int, str]) -> dict[int, str]:
    """把 {类别id: 名称} 转成 {label: 名称}。

    label 为按类别 id 排序后的连续索引（0..C-1），与训练时类别 remap 一致：
    - SHWX: 类别 id 0-24 → label 0-24。
    - DIOR: 类别 id 1-20 → label 0-19。

    Args:
        names_by_id: {类别id: 名称} 映射。

    Returns:
        {label: 名称} 映射。
    """
    return {label: names_by_id[cid] for label, cid in enumerate(sorted(names_by_id.keys()))}


# ══════════════════════════════════════════════════════════════════════
#  各数据集配置 —— 所有参数统一在下方维护，新增数据集照格式添加即可
# ══════════════════════════════════════════════════════════════════════
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
        "class_names": _label_keyed_names(SHWX_CLASS_NAMES),
        # 大类分组：25 类 → 3 个大类（舰船/飞机/车辆）
        "class_to_group": {
            **{class_id: "ship" for class_id in range(0, 4)},
            **{class_id: "aircraft" for class_id in range(4, 24)},
            24: "vehicle",
        },
        "group_iou_thresholds": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
    },
    "dior": {
        "data_dir": "/home/liu/datasets/DIOR-rfdetr",
        "image_dir": "test",
        "label_format": "coco",
        "annotation_file": "test/_annotations.coco.json",
        "exp_output_dir": "output/0804-DIOR-rfdetr_medium_SSCL",
        "checkpoint_file": "checkpoint_best_regular.pth",
        "num_classes": 20,
        "vehicle_class_ids": set(),  # DIOR 无比赛特殊 IoU 规则，全部按 0.50
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
    """推理参数（原 ``CONF_THRESHOLD``/``DEVICE``/``BATCH_SIZE``/``NUM_WORKERS`` 等
    模块级常量的收敛形态）。

    Attributes:
        device: 推理设备（如 ``"cuda:0"``；无 CUDA 时自动回退 CPU）。
        conf_threshold: 全局置信度阈值（默认 0.25）。
        class_conf_thresholds: 逐类置信度阈值 ``{类别id: 阈值}``（默认全 0.25）。
        batch_size: GPU 单次前向的图像数。
        num_workers: CPU 预取 worker 进程数（建议等于 CPU 核数）。
        prefetch_factor: 每个 worker 在内存中预取的数据批数。
        gpu_util_sample_interval: 后台采样 GPU 利用率的时间间隔（秒）。
        use_fp16: 用 FP16 张量核加速推理（RTX 30 系约 2.5 倍提速）。
        tile_overlap: 滑窗切分重叠像素数；``0`` = 关闭切分（大图仍走整图缩放
            路径），``> 0`` = 超分辨率大图切块推理。
        tile_nms_iou: 切分合并后按类别 NMS 的 IoU 阈值（``"nms"`` 策略）。
        tile_batch_size: 切分路径的 tile 批量大小；``None`` = 沿用 ``batch_size``。
        tile_strategy: 大图合并策略：``"nms"`` = 里程碑 1 基线（全部保留 + 按类别
            NMS）；``"center"`` = 里程碑 2（中心归属 + 跨 tile 极严格安全合并）。
        tile_cut_mode: 大图切块方式：``"grid"`` = 滑窗网格切块（现有行为，
            里程碑 1-3）；``"seam"`` = 拼接缝切分（里程碑 4，检测小图拼接缝
            沿缝切割，无重叠、无目标截断，合并固定 center）。
    """

    device: str = "cuda:0"
    conf_threshold: float = 0.25
    class_conf_thresholds: dict[int, float] = field(default_factory=dict)
    batch_size: int = 32
    num_workers: int = 12
    prefetch_factor: int = 3
    gpu_util_sample_interval: float = 0.5
    use_fp16: bool = False
    tile_overlap: int = 0
    tile_nms_iou: float = 0.5
    tile_batch_size: int | None = None
    tile_strategy: str = "nms"
    tile_cut_mode: str = "grid"


def _effective_num_workers(infer: InferenceCfg, seam_mode: bool) -> int:
    """解析评测阶段实际使用的 DataLoader worker 数。

    seam 模式会先在主进程执行大规模 OpenCV/numpy 缝检测。之后若继续 fork
    DataLoader worker，容易继承库内部线程锁状态并死锁，因此 seam 模式强制
    走主进程串行解码；普通整图/滑窗路径保持配置值。

    Args:
        infer: 推理配置。
        seam_mode: 是否启用拼接缝切分模式。

    Returns:
        实际传给 DataLoader 的 ``num_workers``。
    """
    if seam_mode and infer.num_workers > 0:
        return 0
    return infer.num_workers


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
        # 补齐到分类头输出通道数（pred_logits 最后一维 = num_classes + 1，
        # 含背景槽位；counts 只覆盖真实类别），背景槽位 bias 补 0。
        if la_bias.numel() < num_logit_classes:
            la_bias = torch.cat([la_bias, torch.zeros(num_logit_classes - la_bias.numel(), dtype=la_bias.dtype)])
        return la_bias.to(device)


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════


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

    Returns:
        报告文本行列表。
    """
    sep = "=" * 80
    dash = "-" * 80
    report_lines: list[str] = []

    # ── 报告头 ──────────────────────────────────────────────────────
    report_lines.extend(
        [
            sep,
            "RF-DETR 测试结果报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"数据集: {dataset_name}",
            f"权重: {checkpoint_path}",
            sep,
        ]
    )

    # ── 测试数据概况 ────────────────────────────────────────────────
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

    # ── 推理测速结果 ────────────────────────────────────────────────
    report_lines.append("推理测速结果")
    report_lines.append(f"GPU 批量大小: {infer.batch_size}  |  CPU 预取 worker 数: {infer.num_workers}")
    report_lines.append(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {timed_images / throughput:.1f}s）")
    if gpu_util is not None:
        report_lines.append(f"推理期间 GPU 平均利用率: {gpu_util:.1f}%")
    report_lines.append(sep)

    # ── 比赛指标评估结果（大类聚合）──────────────────────────────────
    report_lines.append("比赛指标评估结果（测试集）")
    report_lines.append(format_eval_line("all", eval_results["all"]))
    for group_name, group_result in eval_results["groups"].items():
        report_lines.append(format_eval_line(group_name, group_result))
    report_lines.append(dash)

    # ── 每个大类下小类指标的平均值（macro 平均）──────────────────────
    report_lines.append("每个大类下小类指标的平均值（macro 平均）")
    report_lines.append(dash)
    for group_name, macro in group_macro.items():
        report_lines.append(format_macro_line(group_name, macro))
    report_lines.append(dash)

    # ── 总指标（各大类平均指标再取平均）──────────────────────────────
    report_lines.append("总指标（各大类平均指标再取算术平均，即（船+飞机+车辆）/3）")
    report_lines.append(dash)
    report_lines.append(format_macro_line("total", total_macro))
    report_lines.append(dash)

    # ── 细粒度类别指标 ──────────────────────────────────────────────
    report_lines.append("细粒度类别指标")
    report_lines.append(dash)
    for class_name in sorted(per_class_results["groups"].keys()):
        report_lines.append(format_eval_line(class_name, per_class_results["groups"][class_name]))
    report_lines.append(sep)

    # ── 生成的可视化文件 ────────────────────────────────────────────
    report_lines.append("生成的可视化文件")
    report_lines.append(f"混淆矩阵: {dataset.exp_output_dir / 'confusion_matrix.png'}")
    report_lines.append(f"FP 可视化目录: {dataset.exp_output_dir / 'FP'}")
    report_lines.append(f"FN 可视化目录: {dataset.exp_output_dir / 'FN'}")

    return report_lines


def write_test_result(report_lines: list[str], output_path: Path) -> None:
    """把测试结果报告写入实验文件夹中的 test_result.txt。

    Args:
        report_lines: 报告文本行列表。
        output_path: 输出文件路径（一般为 ``EXP_OUTPUT_DIR / "test_result.txt"``）。
    """
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[完成] 测试结果已保存至: {output_path}")


# ══════════════════════════════════════════════════════════════════════
#  RF-DETR 批量流水线推理（CPU 预取 + GPU 批量前向）
# ══════════════════════════════════════════════════════════════════════


class _InferenceDataset(Dataset):
    """在 DataLoader 子进程 worker 中读取并轻量预处理的测试图像数据集。

    每个样本返回 ``(image_id, rgb_tensor, (height, width))``，其中
    ``rgb_tensor`` 为 ``(C, H, W)`` 的 uint8 RGB 张量（零拷贝视图）。worker 只做
    磁盘解码与 BGR→RGB 换色，float 化、缩放与归一化全部放到 GPU 端批量执行，
    从而把 CPU 负载降到最低、让 GPU 持续饱和。测试集单轮完整推理一遍。

    Args:
        image_paths: 测试图像路径列表。
    """

    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, tuple[int, int]]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C,H,W) uint8 视图
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


def filter_postprocess_results(
    results: list[dict[str, torch.Tensor]],
    conf_threshold: float,
    class_conf_thresholds: Mapping[int, float] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """对 postprocess 输出逐图做置信度阈值过滤，返回 ``(boxes, labels, scores)`` 张量。

    阈值语义与原整图路径内联实现完全一致：命中 ``class_conf_thresholds`` 的类别
    用类阈值，未命中（或未配置逐类阈值）回退到 ``conf_threshold``；比较为严格
    大于（``>``）。返回张量保持在原设备，供整图路径直接转 BoxRecord、切分路径
    偏移坐标后再做 NMS（两路径共用同一阈值配方，保证口径一致）。

    Args:
        results: postprocess 输出，每图一个 dict（keys: boxes/labels/scores）。
        conf_threshold: 全局置信度阈值。
        class_conf_thresholds: 逐类置信度阈值 ``{类别id: 阈值}``；``None`` 或
            空字典时所有类别统一使用 ``conf_threshold``。

    Returns:
        ``[(boxes, labels, scores), ...]`` 列表，长度与 ``results`` 一致；
        均为已过滤（保留 ``scores > thr`` 的行）且未转移设备的张量。
    """
    filtered: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for result in results:
        # 逐类阈值：命中 class_conf_thresholds 的类用类阈值，否则回退全局阈值
        if class_conf_thresholds:
            per_class_thr = torch.tensor(
                [class_conf_thresholds.get(int(label), conf_threshold) for label in result["labels"].tolist()],
                device=result["scores"].device,
            )
        else:
            per_class_thr = conf_threshold
        keep = result["scores"] > per_class_thr
        filtered.append((result["boxes"][keep], result["labels"][keep], result["scores"][keep]))
    return filtered


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
    gpu_util_sample_interval: float = 0.5,
    use_fp16: bool = False,
) -> tuple[list[BoxRecord], float, float | None, int]:
    """批量流水线推理：多进程预取解码 + GPU 批量前向，返回预测框与测速结果。

    相比逐张调用 ``model.predict``，本函数通过以下方式让 GPU 满载：

    1. **多进程预取**：``num_workers`` 个 worker 进程在后台并行完成图像解码与
       换色，与 GPU 计算重叠。
    2. **批量前向**：多张图像组成一个 batch 一次性前向，放大 GPU 上的计算粒度
       （batch=1 时单帧前向约 12ms，batch=16 时每帧仅约 5ms）。
    3. **GPU 端预处理**：float 化、缩放、归一化全部在 GPU 批量执行，worker
       只承担最轻量的解码工作。

    预测结果与 ``model.predict`` 逐像素一致（相同的 uint8→float 转换、
    ``antialias=False`` 缩放与归一化）。测试集完整单轮推理一遍，逐张收集
    预测框；第一批作为流水线预热（触发 worker 启动与 pipeline 填充），其预测框
    仍被收集，但不计入测速吞吐。

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
        gpu_util_sample_interval: 后台采样 GPU 利用率的时间间隔（秒）。
        use_fp16: 用 FP16 张量核加速推理。

    Returns:
        ``(pred_records, steady_throughput, gpu_util, timed_images)``：
        ``pred_records`` 为 BoxRecord 列表；``steady_throughput`` 为稳态吞吐
        （img/s）；``gpu_util`` 为推理期间 GPU 平均利用率（%），采样失败时为
        ``None``；``timed_images`` 为参与稳态计时（剔除预热批）的图像数。
    """
    resolution = int(model.model.resolution)

    # [分类损失均衡化] 可选推理侧 LA bias：由 class_counts.json 按训练侧同配方
    # 重建 logit bias，并在 postprocess/top-k 前修正 logits，确保被 bias 抬升
    # 后才应进入 top-k 的稀有类候选不会被提前丢掉。
    bias_tensor: torch.Tensor | None = None
    bias_k: float = 1.0
    if la_bias is not None:
        if isinstance(la_bias, torch.Tensor):
            bias_tensor = la_bias
        else:
            bias_tensor = la_bias.build_bias_tensor(num_classes + 1, device)
            bias_k = la_bias.k
            print(f"[i] 推理侧 LA bias 生效: {la_bias.counts_path}（k={la_bias.k}）")

    dataset = _InferenceDataset(image_paths)
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
    )

    # 显式把权重放到目标设备并切换 eval 模式（绕过 predict() 的懒加载）。
    # 注意：eval 模式下解码器仅使用单组 query（shape 为 num_queries），而训练模式
    # 会输出全部 group 的 queries，因此 eval() 必须设置，否则输出形状会不一致。
    model.model.model = model.model.model.to(device)
    model.model.model.eval()
    if use_fp16:
        model.model.model = model.model.model.half()
    model_dtype = next(model.model.model.parameters()).dtype
    means: list[float] = model.means
    stds: list[float] = model.stds

    # 预热前向：触发 CUDA 内核编译与 cuDNN autotune，避免计入平均耗时
    with torch.inference_mode():
        dummy = F.normalize(
            torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=model_dtype),
            means,
            stds,
        )
        model.model.model(dummy)
        torch.cuda.synchronize()

    gpu_monitor = _GpuUtilMonitor(gpu_util_sample_interval)
    gpu_monitor.start()

    pred_records: list[BoxRecord] = []
    total = len(image_paths)  # 测试图像总数（单轮）
    timed_images = 0
    warmup_images = 0
    bench_start = time.perf_counter()
    first_batch = True

    with torch.inference_mode():
        for stems, rgb_tensors, orig_sizes in loader:
            # uint8(C,H,W) → float[0,1]，并在 GPU 上缩放至模型分辨率
            # （与 predict() 的 F.to_tensor + F.resize(antialias=False) 完全一致）
            gpu_images = [
                F.resize(
                    tensor.to(device, non_blocking=True).to(model_dtype).div_(255.0),
                    (resolution, resolution),
                    antialias=False,
                )
                for tensor in rgb_tensors
            ]
            batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)

            predictions = model.model.model(batch_tensor)
            if bias_tensor is not None:
                adjusted_logits = predictions["pred_logits"] - bias_k * bias_tensor.to(
                    dtype=predictions["pred_logits"].dtype
                )
                predictions = {**predictions, "pred_logits": adjusted_logits}
            target_sizes = torch.tensor(orig_sizes, device=device)
            results = model.model.postprocess(predictions, target_sizes=target_sizes)

            # 收集预测框（每张测试图只推理一遍，全部收集）
            for stem, (boxes, class_ids, scores) in zip(
                stems, filter_postprocess_results(results, conf_threshold, class_conf_thresholds)
            ):
                for xyxy, class_id, score in zip(boxes.cpu().numpy(), class_ids.cpu().numpy(), scores.cpu().numpy()):
                    pred_records.append(
                        BoxRecord(
                            image_id=stem,
                            class_id=int(class_id),
                            xyxy=tuple(float(v) for v in xyxy),
                            score=float(score),
                        )
                    )

            if first_batch:
                # 第一批用于预热：触发 worker 进程启动与流水线填充，剔除其耗时
                warmup_images = len(stems)
                torch.cuda.synchronize()
                bench_start = time.perf_counter()
                first_batch = False
                continue
            timed_images += len(stems)
            print_progress(timed_images, total - warmup_images, bench_start, device)

    gpu_monitor.stop()
    torch.cuda.synchronize()
    print()

    steady_elapsed = time.perf_counter() - bench_start
    steady_throughput = timed_images / steady_elapsed if steady_elapsed > 0 else 0.0
    return pred_records, steady_throughput, gpu_monitor.average_utilization(), timed_images


# ══════════════════════════════════════════════════════════════════════
#  完整评估主流程（原 test.py 的 __main__）
# ══════════════════════════════════════════════════════════════════════


def run_evaluation(
    dataset: DatasetCfg,
    infer: InferenceCfg,
    checkpoint_path: str | Path,
    *,
    save_fp_fn: bool = True,
    save_yolo_preds: bool = False,
    la_bias: LaBiasCfg | None = None,
    resolution: int | None = None,
    viz_large_count: int = 0,
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
        resolution: 可选：推理输入分辨率。构造参数优先于 checkpoint 记录的
            ``model_config``（例如 nano 以 704 训练时强制 704 推理）；
            ``None`` 用 checkpoint 记录的分辨率。
        viz_large_count: 随机可视化大图数量（切分路径启用时有效）：从大图中
            固定种子抽选 ``viz_large_count`` 张，保存左 GT / 右 Predict 的对比图
            到 ``exp_output_dir/large_viz/``；``0`` = 关闭。
    """
    os.chdir(PROJECT_ROOT)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    test_image_paths = read_test_image_paths(dataset.test_image_dir)
    # YOLO 格式需要图像尺寸把归一化坐标换算成像素；COCO 的 bbox 本身就是像素坐标
    image_size_map = build_image_size_map(test_image_paths) if dataset.label_format == "yolo" else None

    # 读取测试集真实框（按数据集标签格式分派）
    if dataset.label_format == "yolo":
        assert dataset.label_dir is not None
        gt_records = load_yolo_labels(dataset.label_dir, image_size_map)
    elif dataset.label_format == "coco":
        assert dataset.annotation_file is not None
        gt_records = load_coco_labels(dataset.annotation_file)
    else:
        raise ValueError(f"不支持的标签格式: {dataset.label_format}")

    # 加载 RF-DETR 模型并执行批量流水线推理（含测速）。
    # from_checkpoint 自动按 checkpoint 中的 model_name 推断模型尺寸（nano/small/medium/large），
    # 无需手动指定模型类。
    device = resolve_device(infer.device)
    print(f"[i] 正在从 {checkpoint_path} 加载 RF-DETR 模型...")
    ckpt_kwargs: dict[str, int] = {}
    if resolution is not None:
        ckpt_kwargs["resolution"] = resolution
        print(f"[i] 推理分辨率覆盖为: {resolution}")
    model = RFDETR.from_checkpoint(str(checkpoint_path), **ckpt_kwargs)
    print(f"[i] 已加载模型: {type(model).__name__} | 分辨率: {int(model.model.resolution)}")
    # [SemHead] 若 checkpoint 含语义头权重（语义分类头实验），重建语义残差模块，
    # 保证离线推理与训练前向一致（from_checkpoint 不经过 module_model 的装配逻辑）。
    from rfdetr.sscl.semantic_head import attach_from_checkpoint

    _ckpt_sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    attach_from_checkpoint(model.model.model, _ckpt_sd.get("model", _ckpt_sd))
    del _ckpt_sd

    # ── 大图切分推理（tile_overlap > 0 或 tile_cut_mode="seam" 时启用）───
    # 小图（max(w,h) <= 分辨率）走整图批量路径；大图逐张切块推理，两路结果
    # 合并后由下游统一评估。切分关闭时所有图走整图缩放路径（行为与旧版一致，
    # 用于回归对照）。
    # "grid" 模式 = 滑窗网格切块（里程碑 1-3）；"seam" 模式 = 拼接缝切分
    # （里程碑 4：检测小图拼接缝沿缝切割，无重叠、无目标截断）。
    seam_mode = infer.tile_cut_mode == "seam"
    tile_mode = infer.tile_overlap > 0 or seam_mode
    effective_num_workers = _effective_num_workers(infer, seam_mode)
    small_paths = test_image_paths
    large_paths: list[Path] = []
    # 推理分辨率（model 释放后仍供大图对比可视化绘制切分网格线使用）
    tile_resolution = int(model.model.resolution)
    origins_map: dict[str, list[tuple[int, int, int, int, int]]] | None = None
    if tile_mode:
        # 惰性 import：避免 eval_lib ↔ tiling 模块循环依赖
        from scripts.tiling import split_image_paths, tile_predict_records

        # COCO 格式未预计算尺寸映射（label_format != "yolo"），切分路径需要时补算
        if image_size_map is None:
            image_size_map = build_image_size_map(test_image_paths)
        small_paths, large_paths = split_image_paths(test_image_paths, image_size_map, tile_resolution)
        if seam_mode:
            # 拼接缝切分：主进程逐张检测缝并组合图块原点（缝图块无重叠，
            # 超限图块内部展开滑窗网格兜底）
            from scripts.seam_cut import build_seam_origins_map

            origins_map, seam_stats = build_seam_origins_map(
                large_paths,
                image_size_map,
                tile_resolution,
                infer.tile_overlap,
            )
            total_h = sum(ny for ny, _nx in seam_stats.values())
            total_v = sum(nx for _ny, nx in seam_stats.values())
            total_tiles = sum(len(origins) for origins in origins_map.values())
            print(
                f"[i] 拼接缝切分推理: 小图 {len(small_paths)} 张走整图路径; "
                f"大图 {len(large_paths)} 张检测到水平缝 {total_h} 条、垂直缝 {total_v} 条, "
                f"共 {total_tiles} 个图块(超限图块滑窗兜底 overlap={infer.tile_overlap})"
            )
            if effective_num_workers != infer.num_workers:
                print(
                    "[i] seam 模式已在主进程完成 OpenCV/numpy 缝检测，"
                    f"为避免 fork DataLoader worker 死锁，实际 num_workers 从 {infer.num_workers} 降为 0"
                )
        else:
            print(
                f"[i] 滑窗切分推理: 小图 {len(small_paths)} 张走整图路径; "
                f"大图 {len(large_paths)} 张走切分路径(overlap={infer.tile_overlap}, "
                f"策略={infer.tile_strategy}, NMS IoU={infer.tile_nms_iou})"
            )
    pred_records, throughput, gpu_util, timed_images = predict_batched_to_records(
        model,
        small_paths,
        device,
        conf_threshold=infer.conf_threshold,
        class_conf_thresholds=infer.class_conf_thresholds,
        batch_size=infer.batch_size,
        num_workers=effective_num_workers,
        la_bias=la_bias,
        num_classes=dataset.num_classes,
        prefetch_factor=infer.prefetch_factor,
        gpu_util_sample_interval=infer.gpu_util_sample_interval,
        use_fp16=infer.use_fp16,
    )
    if large_paths:
        # LA bias 只构建一次 Tensor 传给切分路径（tiling 不依赖 LaBiasCfg 类型）
        if la_bias is not None:
            bias_tensor = la_bias.build_bias_tensor(dataset.num_classes + 1, device)
            bias_k = la_bias.k
        else:
            bias_tensor, bias_k = None, 1.0
        tile_records, tile_throughput, tile_gpu_util, tile_timed = tile_predict_records(
            model,
            large_paths,
            image_size_map,
            device,
            resolution=int(model.model.resolution),
            overlap=infer.tile_overlap,
            nms_iou=infer.tile_nms_iou,
            conf_threshold=infer.conf_threshold,
            batch_size=infer.tile_batch_size or infer.batch_size,
            num_workers=effective_num_workers,
            class_conf_thresholds=infer.class_conf_thresholds,
            la_bias=bias_tensor,
            la_bias_k=bias_k,
            prefetch_factor=infer.prefetch_factor,
            use_fp16=infer.use_fp16,
            gpu_util_sample_interval=infer.gpu_util_sample_interval,
            tile_strategy="center" if seam_mode else infer.tile_strategy,
            origins_map=origins_map,
        )
        pred_records += tile_records
        print(
            f"[i] 切分推理吞吐: {tile_throughput:.1f} img/s  "
            f"（{tile_timed} 张大图 / {tile_timed / tile_throughput:.1f}s）"
        )
    del model
    release_cuda_cache(device)

    # ── YOLO 格式预测输出（供漏检/虚警归因统计）────────────────────────
    if save_yolo_preds:
        save_yolo_predictions(pred_records, dataset.exp_output_dir / "yolo_preds", image_size_map)

    # ── 大图切割结果对比可视化（随机抽选 N 张大图，左 GT / 右 Predict）───
    if viz_large_count > 0 and large_paths:
        from visualization.detection import save_large_image_visualizations

        save_large_image_visualizations(
            [p.stem for p in large_paths],
            gt_records,
            pred_records,
            test_image_paths,
            dataset.class_names,
            dataset.exp_output_dir / "large_viz",
            viz_large_count,
            tile_size=None if seam_mode else tile_resolution,
            tile_overlap=infer.tile_overlap,
            tile_origins_map=origins_map,
            num_classes=dataset.num_classes,
            vehicle_class_ids=set(dataset.vehicle_class_ids),
        )

    # ── 推理测速结果 ─────────────────────────────────────────────────
    print("=" * 80)
    print("推理测速结果")
    print(f"GPU 批量大小: {infer.batch_size}  |  CPU 预取 worker 数: {effective_num_workers}")
    # timed_images=0（如全部图像走切分路径）时 throughput 为 0，避免除零
    if throughput > 0:
        print(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {timed_images / throughput:.1f}s）")
    else:
        print(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张，耗时统计见切分路径）")
    if gpu_util is not None:
        print(f"推理期间 GPU 平均利用率: {gpu_util:.1f}%")
    print("=" * 80)

    # ── 按比赛规则评估：大类（舰船/飞机/车辆）─────────────────────────
    config = EvalConfig(
        class_to_group=dataset.class_to_group,
        group_iou_thresholds=dataset.group_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)

    print("=" * 80)
    print("比赛指标评估结果（测试集）")
    print(f"权重: {checkpoint_path}")
    print(f"测试图像数: {len(test_image_paths)}")
    print(f"真实框数: {len(gt_records)}")
    print(f"预测框数: {len(pred_records)}")
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

    # ── 细粒度逐类指标 ───────────────────────────────────────────────
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

    # ── 每个大类下小类指标的平均值（macro 平均）与总指标 ──────────────
    group_macro = compute_group_macro_averages(
        per_class_results["groups"],
        dataset.class_to_group,
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

    # ── 混淆矩阵可视化 ───────────────────────────────────────────────
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

    # ── FP / FN 可视化保存 ───────────────────────────────────────────
    if save_fp_fn:
        print("\n[i] 正在生成 FP/FN 可视化...")
        clear_vis_dirs(dataset.exp_output_dir / "FP", dataset.exp_output_dir / "FN", dataset.class_names)
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
        )
        print("[完成] FP/FN 可视化保存完成")

    # ── 保存测试结果报告到实验文件夹 ─────────────────────────────────
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
        infer=replace(infer, num_workers=effective_num_workers),
    )
    write_test_result(report_lines, dataset.exp_output_dir / "test_result.txt")
