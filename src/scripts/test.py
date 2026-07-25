# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用比赛评分方案在测试集上评估 RF-DETR 模型。"""

import gc
import os
import sys
import time
from pathlib import Path

import cv2
import torch

# ── 路径配置 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path("/home/liu/datasets/SHWX-dataset-dict")
TEST_IMAGE_DIR = DATA_DIR / "images" / "test"
LABEL_DIR = DATA_DIR / "labels" / "test"
EXP_OUTPUT_DIR = PROJECT_ROOT / "output" / "0724-shwx-rfdetr_medium"
CHECKPOINT_PATH = EXP_OUTPUT_DIR / "checkpoint_best_total.pth"

# ── 推理配置 ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
DEVICE = "cuda:0"  # 使用 GPU 推理；无 CUDA 时脚本会自动回退到 CPU
CUDA_CACHE_CLEAR_INTERVAL = 10  # 每隔若干张图像释放一次 PyTorch CUDA 缓存

# ── 可视化保存配置 ───────────────────────────────────────────────────
FP_DIR = EXP_OUTPUT_DIR / "FP"  # FP 可视化保存根目录
FN_DIR = EXP_OUTPUT_DIR / "FN"  # FN 可视化保存根目录

# ── 比赛指标配置 ───────────────────────────────────────────────────
NUM_CLASSES = 25
VEHICLE_CLASS_IDS = {24}  # FSC 发射车，比赛规则按车辆目标 IoU=0.35

# 25 个细粒度类别名称
CLASS_NAMES: dict[int, str] = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS",
    4: "A1_SU-35", 5: "A2_C-130", 6: "A3_C-17", 7: "A4_C-5",
    8: "A5_F-16", 9: "A6_TU-160", 10: "A7_E-3", 11: "A8_B-52",
    12: "A9_P-3C", 13: "A10_B-1B", 14: "A11_E-8", 15: "A12_TU-22",
    16: "A13_F-15", 17: "A14_KC-135", 18: "A15_F-22", 19: "A16_FA-18",
    20: "A17_TU-95", 21: "A18_KC-10", 22: "A19_SU-34", 23: "A20_SU-24",
    24: "FSC",
}

# 大类分组映射：25 类 → 3 个大类（舰船/飞机/车辆）
CLASS_TO_GROUP: dict[int, str] = {
    **{class_id: "ship" for class_id in range(0, 4)},
    **{class_id: "aircraft" for class_id in range(4, 24)},
    24: "vehicle",
}
GROUP_IOU_THRESHOLDS: dict[str, float] = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}

# 细粒度类别分组映射：每个类独立成组，用于输出逐类指标
PER_CLASS_TO_GROUP: dict[int, str] = {class_id: name for class_id, name in CLASS_NAMES.items()}
PER_CLASS_IOU_THRESHOLDS: dict[str, float] = {
    name: 0.35 if class_id in VEHICLE_CLASS_IDS else 0.50
    for class_id, name in CLASS_NAMES.items()
}


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def print_eval_result(name: str, result: EvalResult) -> None:
    """打印单组比赛评测结果。"""
    print(
        f"{name:<10s} "
        f"TP={result.tp:<6d} "
        f"FP={result.fp:<6d} "
        f"FN={result.fn:<6d} "
        f"Recall={result.recall:.4f} "
        f"FDR={result.fdr:.4f} "
        f"Precision={result.precision:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
#  RF-DETR 推理
# ══════════════════════════════════════════════════════════════════════


def predict_rfdetr_to_records(
    model: "RFDETRMedium",
    image_paths: list[Path],
    device: str,
    conf_threshold: float,
) -> list[BoxRecord]:
    """执行 RF-DETR 推理并转换为比赛评测需要的 BoxRecord。

    逐张处理图像，显式控制 CUDA 缓存以保持显存稳定。

    Args:
        model: 已加载的 RFDETRMedium 实例。
        image_paths: 测试图像路径列表。
        device: 推理设备（如 ``"cuda:0"``）。
        conf_threshold: 置信度阈值。

    Returns:
        BoxRecord 列表，每个元素为一个预测框。
    """
    pred_records: list[BoxRecord] = []
    start_time = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调用 RF-DETR 推理，返回 supervision Detections 对象
        detections = model.predict(image_rgb, threshold=conf_threshold)

        if detections.xyxy is not None and len(detections.xyxy) > 0:
            boxes = detections.xyxy.tolist()
            class_ids = detections.class_id.tolist() if detections.class_id is not None else []
            scores = detections.confidence.tolist() if detections.confidence is not None else []
            for xyxy, class_id, score in zip(boxes, class_ids, scores):
                pred_records.append(
                    BoxRecord(
                        image_id=image_path.stem,
                        class_id=int(class_id),
                        xyxy=tuple(float(v) for v in xyxy),
                        score=float(score),
                    )
                )

        del image, image_rgb, detections
        if index % CUDA_CACHE_CLEAR_INTERVAL == 0:
            release_cuda_cache(device)

        print_progress(index, len(image_paths), start_time, device)

    release_cuda_cache(device)
    print()
    return pred_records


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    test_image_paths = read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = build_image_size_map(test_image_paths)

    # 读取测试集真实框
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)

    # 加载 RF-DETR 模型并执行推理
    device = resolve_device(DEVICE)
    print(f"[i] 正在从 {CHECKPOINT_PATH} 加载 RF-DETR 模型...")
    model = RFDETRMedium.from_checkpoint(str(CHECKPOINT_PATH))
    with torch.inference_mode():
        pred_records = predict_rfdetr_to_records(
            model, test_image_paths, device, conf_threshold=CONF_THRESHOLD,
        )
    del model
    release_cuda_cache(device)

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
        gt_records, pred_records, per_class_config,
    )

    print("\n" + "-" * 80)
    print("细粒度类别指标")
    print("-" * 80)
    for class_name in sorted(per_class_results["groups"].keys()):
        result = per_class_results["groups"][class_name]
        if result.tp + result.fn > 0:  # 只打印测试集中存在的类别
            print_eval_result(class_name, result)

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
    print("\n[i] 正在生成 FP/FN 可视化...")
    clear_vis_dirs(FP_DIR, FN_DIR, CLASS_NAMES)
    fp_img, fn_img, fp_box, fn_box, tp_pred = match_per_image_per_class(
        gt_records, pred_records, NUM_CLASSES, VEHICLE_CLASS_IDS,
    )
    save_fp_fn_visualizations(
        fp_img, fn_img, fp_box, fn_box, tp_pred,
        gt_records, test_image_paths,
        CLASS_NAMES, FP_DIR, FN_DIR,
    )
    print("[完成] FP/FN 可视化保存完成")
