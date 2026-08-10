# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用比赛评分方案在测试集上评估 RF-DETR 模型，并测量 GPU 满载时的推理吞吐。

推理阶段使用“多进程预取解码 + GPU 批量前向”的流水线：DataLoader 的多个 worker 进程在后台并行完成图像解码与换色，float 化 / 缩放 / 归一化在 GPU 端批量执行， 从而隐藏 CPU 预处理延迟，让
GPU 持续满载。预测结果与逐张 ``model.predict`` 完全一致。
"""

import gc
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# ══════════════════════════════════════════════════════════════════════
#  数据集选择 —— 切换数据集只需改这里
# ══════════════════════════════════════════════════════════════════════
DATASET = "shwx"  # 可选: "shwx"（YOLO 格式）| "dior"（Roboflow COCO 格式）
SAVE_FD_FN = True  # 是否保存FN/FD可视化

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 类别名称统一来自 sscl/prompts/*.yaml，保证与语义矩阵的类别索引一致
from rfdetr.sscl.prompts import DIOR_CLASS_NAMES, SHWX_CLASS_NAMES  # noqa: E402


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
        "exp_output_dir": "output/0810-SHWX-SSCL-Proj-HardNeg-k3-iou03",
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
        # 注意：checkpoint_best_total.pth 因 val mAP 未超过 epoch0 基线，实际存的是基线权重；
        # 真正训练过的模型在 checkpoint_best_regular.pth（epoch 6）。
        "checkpoint_file": "checkpoint_best_regular.pth",
        "num_classes": 20,
        "vehicle_class_ids": set(),  # DIOR 无比赛特殊 IoU 规则，全部按 0.50
        "class_names": _label_keyed_names(DIOR_CLASS_NAMES),
        # DIOR 无舰船/飞机/车辆大类分组，所有类别归为单组 "all"
        "class_to_group": {class_id: "all" for class_id in range(20)},
        "group_iou_thresholds": {"all": 0.50},
    },
}

# ── 激活所选数据集，展开为主流程使用的常量 ────────────────────────────
_cfg = DATASET_CONFIGS[DATASET]
DATA_DIR = Path(_cfg["data_dir"])
TEST_IMAGE_DIR = DATA_DIR / _cfg["image_dir"]
LABEL_FORMAT = _cfg["label_format"]
LABEL_DIR = DATA_DIR / _cfg["label_dir"] if _cfg.get("label_dir") else None
COCO_ANNOTATION = DATA_DIR / _cfg["annotation_file"] if _cfg.get("annotation_file") else None
EXP_OUTPUT_DIR = PROJECT_ROOT / _cfg["exp_output_dir"]
CHECKPOINT_PATH = EXP_OUTPUT_DIR / _cfg["checkpoint_file"]
NUM_CLASSES = _cfg["num_classes"]
VEHICLE_CLASS_IDS = _cfg["vehicle_class_ids"]
CLASS_NAMES: dict[int, str] = _cfg["class_names"]
CLASS_TO_GROUP: dict[int, str] = _cfg["class_to_group"]
GROUP_IOU_THRESHOLDS: dict[str, float] = _cfg["group_iou_thresholds"]

# 细粒度类别分组映射：每个类独立成组，用于输出逐类指标
PER_CLASS_TO_GROUP: dict[int, str] = {class_id: name for class_id, name in CLASS_NAMES.items()}
PER_CLASS_IOU_THRESHOLDS: dict[str, float] = {
    name: 0.35 if class_id in VEHICLE_CLASS_IDS else 0.50 for class_id, name in CLASS_NAMES.items()
}

# ── 推理配置 ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
DEVICE = "cuda:0"  # 使用 GPU 推理；无 CUDA 时脚本会自动回退到 CPU

# ── 推理性能配置（GPU 满载单轮推理）──────────────────────────────────
BATCH_SIZE = 32  # GPU 单次前向处理的图像数；越大 GPU 利用率越高，受显存限制
NUM_WORKERS = 12  # CPU 预取 worker 进程数（建议等于 CPU 核数）
PREFETCH_FACTOR = 3  # 每个 worker 在内存中预取的数据批数（需保证 num_workers × prefetch ≥ batch_size）
GPU_UTIL_SAMPLE_INTERVAL = 0.5  # 后台采样 GPU 利用率的时间间隔（秒）
USE_FP16 = False  # 用 FP16 张量核加速推理（RTX 30 系约 2.5 倍提速）；对精度敏感时保持 False

# ── 可视化保存配置 ───────────────────────────────────────────────────
FP_DIR = EXP_OUTPUT_DIR / "FP"  # FP 可视化保存根目录
FN_DIR = EXP_OUTPUT_DIR / "FN"  # FN 可视化保存根目录

# ── YOLO 格式预测输出（供漏检/虚警归因统计脚本使用）──────────────────
SAVE_YOLO_PREDS = False  # 是否输出 YOLO 格式预测框（每图一个 txt：cls_id cx cy w h conf）
YOLO_PREDS_DIR = EXP_OUTPUT_DIR / "yolo_preds"  # YOLO 预测保存目录

from rfdetr import RFDETRMedium  # noqa: E402
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
) -> list[str]:
    """组装 test_result.txt 报告文本行列表。

    Args:
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
            f"数据集: {DATASET}",
            f"权重: {CHECKPOINT_PATH}",
            sep,
        ]
    )

    # ── 测试数据概况 ────────────────────────────────────────────────
    report_lines.extend(
        [
            f"测试图像数: {len(test_image_paths)}",
            f"真实框数: {len(gt_records)}",
            f"预测框数: {len(pred_records)}",
            f"置信度阈值: {CONF_THRESHOLD}",
            "IoU 阈值: " + "，".join(f"{key}={value:.2f}" for key, value in GROUP_IOU_THRESHOLDS.items()),
            sep,
        ]
    )

    # ── 推理测速结果 ────────────────────────────────────────────────
    report_lines.append("推理测速结果")
    report_lines.append(f"GPU 批量大小: {BATCH_SIZE}  |  CPU 预取 worker 数: {NUM_WORKERS}")
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
    report_lines.append(f"混淆矩阵: {EXP_OUTPUT_DIR / 'confusion_matrix.png'}")
    report_lines.append(f"FP 可视化目录: {FP_DIR}")
    report_lines.append(f"FN 可视化目录: {FN_DIR}")

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


def predict_batched_to_records(
    model: "RFDETRMedium",
    image_paths: list[Path],
    device: str,
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
) -> tuple[list[BoxRecord], float, float | None]:
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
        model: 已加载的 RFDETRMedium 实例。
        image_paths: 测试图像路径列表。
        device: 推理设备（如 ``"cuda:0"``）。
        conf_threshold: 置信度阈值。
        batch_size: GPU 单次前向的图像数。
        num_workers: 预取 worker 进程数。

    Returns:
        ``(pred_records, steady_throughput, gpu_util, timed_images)``：
        ``pred_records`` 为 BoxRecord 列表；``steady_throughput`` 为稳态吞吐
        （img/s）；``gpu_util`` 为推理期间 GPU 平均利用率（%），采样失败时为
        ``None``；``timed_images`` 为参与稳态计时（剔除预热批）的图像数。
    """
    resolution = int(model.model.resolution)
    dataset = _InferenceDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
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
    if USE_FP16:
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

    gpu_monitor = _GpuUtilMonitor(GPU_UTIL_SAMPLE_INTERVAL)
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
            target_sizes = torch.tensor(orig_sizes, device=device)
            results = model.model.postprocess(predictions, target_sizes=target_sizes)

            # 收集预测框（每张测试图只推理一遍，全部收集）
            for stem, result in zip(stems, results):
                keep = result["scores"] > conf_threshold
                boxes = result["boxes"][keep].cpu().numpy()
                class_ids = result["labels"][keep].cpu().numpy()
                scores = result["scores"][keep].cpu().numpy()
                for xyxy, class_id, score in zip(boxes, class_ids, scores):
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
#  主流程
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    # 可选参数：`python test.py <checkpoint.pth>` 覆盖配置里的 checkpoint
    if len(sys.argv) > 1:
        override = Path(sys.argv[1]).resolve()
        if not override.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {override}")
        CHECKPOINT_PATH = override
        print(f"[i] 命令行覆盖 checkpoint: {CHECKPOINT_PATH}")

    test_image_paths = read_test_image_paths(TEST_IMAGE_DIR)
    # YOLO 格式需要图像尺寸把归一化坐标换算成像素；COCO 的 bbox 本身就是像素坐标
    image_size_map = build_image_size_map(test_image_paths) if LABEL_FORMAT == "yolo" else None

    # 读取测试集真实框（按数据集标签格式分派）
    if LABEL_FORMAT == "yolo":
        gt_records = load_yolo_labels(LABEL_DIR, image_size_map)
    elif LABEL_FORMAT == "coco":
        gt_records = load_coco_labels(COCO_ANNOTATION)
    else:
        raise ValueError(f"不支持的标签格式: {LABEL_FORMAT}")

    # 加载 RF-DETR 模型并执行批量流水线推理（含测速）
    device = resolve_device(DEVICE)
    print(f"[i] 正在从 {CHECKPOINT_PATH} 加载 RF-DETR 模型...")
    model = RFDETRMedium.from_checkpoint(str(CHECKPOINT_PATH))
    # [SemHead] 若 checkpoint 含语义头权重（语义分类头实验），重建语义残差模块，
    # 保证离线推理与训练前向一致（from_checkpoint 不经过 module_model 的装配逻辑）。
    from rfdetr.sscl.semantic_head import attach_from_checkpoint

    _ckpt_sd = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    attach_from_checkpoint(model.model.model, _ckpt_sd.get("model", _ckpt_sd))
    del _ckpt_sd
    pred_records, throughput, gpu_util, timed_images = predict_batched_to_records(
        model,
        test_image_paths,
        device,
        conf_threshold=CONF_THRESHOLD,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    del model
    release_cuda_cache(device)

    # ── YOLO 格式预测输出（供漏检/虚警归因统计）────────────────────────
    if SAVE_YOLO_PREDS:
        save_yolo_predictions(pred_records, YOLO_PREDS_DIR, image_size_map)

    # ── 推理测速结果 ─────────────────────────────────────────────────
    print("=" * 80)
    print("推理测速结果")
    print(f"GPU 批量大小: {BATCH_SIZE}  |  CPU 预取 worker 数: {NUM_WORKERS}")
    print(f"推理吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {timed_images / throughput:.1f}s）")
    if gpu_util is not None:
        print(f"推理期间 GPU 平均利用率: {gpu_util:.1f}%")
    print("=" * 80)

    # ── 按比赛规则评估：大类（舰船/飞机/车辆）─────────────────────────
    config = EvalConfig(
        class_to_group=CLASS_TO_GROUP,
        group_iou_thresholds=GROUP_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)

    print("=" * 80)
    print("比赛指标评估结果（测试集）")
    print(f"权重: {CHECKPOINT_PATH}")
    print(f"测试图像数: {len(test_image_paths)}")
    print(f"真实框数: {len(gt_records)}")
    print(f"预测框数: {len(pred_records)}")
    print(f"置信度阈值: {CONF_THRESHOLD}")
    print("IoU 阈值: 车辆=0.35，其他目标=0.50")
    print("=" * 80)

    print_eval_result("all", eval_results["all"])
    for group_name, group_result in eval_results["groups"].items():
        print_eval_result(group_name, group_result)

    # ── 细粒度逐类指标 ───────────────────────────────────────────────
    per_class_config = EvalConfig(
        class_to_group=PER_CLASS_TO_GROUP,
        group_iou_thresholds=PER_CLASS_IOU_THRESHOLDS,
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
        CLASS_TO_GROUP,
        CLASS_NAMES,
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
        num_classes=NUM_CLASSES,
        vehicle_class_ids=VEHICLE_CLASS_IDS,
    )
    plot_confusion_matrix(
        matrix=cm,
        class_names=CLASS_NAMES,
        output_path=str(EXP_OUTPUT_DIR / "confusion_matrix.png"),
    )
    print(f"[完成] 混淆矩阵已保存至: {EXP_OUTPUT_DIR / 'confusion_matrix.png'}")

    # ── FP / FN 可视化保存 ───────────────────────────────────────────
    if SAVE_FD_FN:
        print("\n[i] 正在生成 FP/FN 可视化...")
        clear_vis_dirs(FP_DIR, FN_DIR, CLASS_NAMES)
        fp_img, fn_img, fp_box, fn_box, tp_pred = match_per_image_per_class(
            gt_records,
            pred_records,
            NUM_CLASSES,
            VEHICLE_CLASS_IDS,
        )
        save_fp_fn_visualizations(
            fp_img,
            fn_img,
            fp_box,
            fn_box,
            tp_pred,
            gt_records,
            test_image_paths,
            CLASS_NAMES,
            FP_DIR,
            FN_DIR,
        )
        print("[完成] FP/FN 可视化保存完成")

    # ── 保存测试结果报告到实验文件夹 ─────────────────────────────────
    report_lines = build_test_report(
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
    )
    write_test_result(report_lines, EXP_OUTPUT_DIR / "test_result.txt")
