# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用比赛评分方案在测试集上评估 RF-DETR 模型，并测量 GPU 满载时的推理吞吐。

推理阶段使用“多进程预取解码 + GPU 批量前向”的流水线：DataLoader 的多个 worker
进程在后台并行完成图像解码与换色，float 化 / 缩放 / 归一化在 GPU 端批量执行，
从而隐藏 CPU 预处理延迟，让 GPU 持续满载。预测结果与逐张 ``model.predict``
完全一致。
"""

import gc
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# ── 路径配置 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path("/home/liu/datasets/SHWX-dataset-dict")
TEST_IMAGE_DIR = DATA_DIR / "images" / "test"
LABEL_DIR = DATA_DIR / "labels" / "test"
EXP_OUTPUT_DIR = PROJECT_ROOT / "output" / "0802-SHWX-rfdetr_medium_SSCL-all"
CHECKPOINT_PATH = EXP_OUTPUT_DIR / "checkpoint_best_total.pth"

# ── 推理配置 ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
DEVICE = "cuda:0"  # 使用 GPU 推理；无 CUDA 时脚本会自动回退到 CPU

# ── 推理性能配置（用于 GPU 满载测速）──────────────────────────────────
BATCH_SIZE = 32          # GPU 单次前向处理的图像数；越大 GPU 利用率越高，受显存限制
NUM_WORKERS = 12         # CPU 预取 worker 进程数（建议等于 CPU 核数）
PREFETCH_FACTOR = 3      # 每个 worker 在内存中预取的数据批数（需保证 num_workers × prefetch ≥ batch_size）
BENCH_REPEAT = 10        # 推理测速重复轮数，用于获得稳定的稳态吞吐（仅计时，不重复统计）
GPU_UTIL_SAMPLE_INTERVAL = 0.5  # 后台采样 GPU 利用率的时间间隔（秒）
USE_FP16 = False        # 用 FP16 张量核加速推理（RTX 30 系约 2.5 倍提速）；对精度敏感时保持 False

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
#  RF-DETR 批量流水线推理（CPU 预取 + GPU 批量前向）
# ══════════════════════════════════════════════════════════════════════


class _InferenceDataset(Dataset):
    """在 DataLoader 子进程 worker 中读取并轻量预处理的测试图像数据集。

    每个样本返回 ``(image_id, rgb_tensor, (height, width), repeat)``，其中
    ``rgb_tensor`` 为 ``(C, H, W)`` 的 uint8 RGB 张量（零拷贝视图）。worker 只做
    磁盘解码与 BGR→RGB 换色，float 化、缩放与归一化全部放到 GPU 端批量执行，
    从而把 CPU 负载降到最低、让 GPU 持续饱和。

    数据集通过 ``repeats`` 参数把同一批测试图像重复拼接成一条连续流，测速时
    无需反复重新迭代 DataLoader（避免每轮之间的流水线排空/填充气泡）。

    Args:
        image_paths: 测试图像路径列表。
        repeats: 测速重复轮数；``repeat == 0`` 的样本用于收集预测框。
    """

    def __init__(self, image_paths: list[Path], repeats: int) -> None:
        self.image_paths = image_paths
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.image_paths) * self.repeats

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, tuple[int, int], int]:
        repeat = index // len(self.image_paths)
        image_path = self.image_paths[index % len(self.image_paths)]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C,H,W) uint8 视图
        return image_path.stem, rgb_tensor, (height, width), repeat


def _inference_collate(
    batch: list[tuple[str, torch.Tensor, tuple[int, int], int]],
) -> tuple[list[str], list[torch.Tensor], list[tuple[int, int]], list[int]]:
    """自定义聚合函数：图像原始尺寸不同，保持为张量列表而非堆叠。"""
    stems = [item[0] for item in batch]
    tensors = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]
    repeats = [item[3] for item in batch]
    return stems, tensors, orig_sizes, repeats


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

        优先使用 ``torch.cuda.utilization``（需要 nvidia-ml-py），不可用时回退到
        ``nvidia-smi`` 命令行。
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
    ``antialias=False`` 缩放与归一化）。测速通过对同一批测试图像连续重复推理
    ``BENCH_REPEAT`` 轮取稳态吞吐：``repeat == 0`` 的样本用于收集预测框，
    其余轮仅计时。数据集在 DataLoader 内部拼接为连续流，避免每轮之间流水线
    排空/填充带来的气泡。

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
    dataset = _InferenceDataset(image_paths, repeats=BENCH_REPEAT)
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
    total = len(image_paths) * BENCH_REPEAT  # 连续流总图像数（含重复轮）
    timed_images = 0
    warmup_images = 0
    bench_start = time.perf_counter()
    first_batch = True

    with torch.inference_mode():
        for stems, rgb_tensors, orig_sizes, repeats in loader:
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

            # 只在 repeat == 0 的样本上收集预测框（测速轮无需 GPU→CPU 同步，保持流水线满载）
            for stem, repeat, result in zip(stems, repeats, results):
                if repeat != 0:
                    continue
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

    test_image_paths = read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = build_image_size_map(test_image_paths)

    # 读取测试集真实框
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)

    # 加载 RF-DETR 模型并执行批量流水线推理（含测速）
    device = resolve_device(DEVICE)
    print(f"[i] 正在从 {CHECKPOINT_PATH} 加载 RF-DETR 模型...")
    model = RFDETRMedium.from_checkpoint(str(CHECKPOINT_PATH))
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

    # ── 推理测速结果 ─────────────────────────────────────────────────
    print("=" * 80)
    print("推理测速结果")
    print(f"GPU 批量大小: {BATCH_SIZE}  |  CPU 预取 worker 数: {NUM_WORKERS}  |  测速重复轮数: {BENCH_REPEAT}")
    print(f"稳态吞吐: {throughput:.1f} img/s  （{timed_images} 张 / {timed_images / throughput:.1f}s）")
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
