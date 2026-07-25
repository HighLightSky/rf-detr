"""使用比赛评分方案在测试集上评估模型。"""

import os
import shutil
import sys
import gc
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ── 路径配置 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path("/home/liu/datasets/SHWX-dataset-dict")
TEST_IMAGE_DIR = DATA_DIR / "images" / "test"
LABEL_DIR = DATA_DIR / "labels" / "test"
WEIGHT_PATH = PROJECT_ROOT / "experiments" / "0718-yolo12m-baseline" / "weights" / "best.pt"

# ── 推理配置 ───────────────────────────────────────────────────────
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
DEVICE = "cuda:0"  # 使用 GPU 推理；无 CUDA 时脚本会自动回退到 CPU
PREDICT_BATCH_SIZE = 1
CUDA_CACHE_CLEAR_INTERVAL = 10  # 每隔若干张图像释放一次 PyTorch CUDA 缓存

# ── 可视化保存配置 ───────────────────────────────────────────────────
EXP_DIR = Path(__file__).resolve().parent  # 当前实验目录
FP_DIR = EXP_DIR / "FP"                     # FP 可视化保存根目录
FN_DIR = EXP_DIR / "FN"                     # FN 可视化保存根目录

# ── 比赛指标配置 ───────────────────────────────────────────────────
NUM_CLASSES = 25
VEHICLE_CLASS_IDS = {24}  # FSC 发射车，比赛规则按车辆目标 IoU=0.35

# 25 个细粒度类别名称
CLASS_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS",
    4: "A1_SU-35", 5: "A2_C-130", 6: "A3_C-17", 7: "A4_C-5",
    8: "A5_F-16", 9: "A6_TU-160", 10: "A7_E-3", 11: "A8_B-52",
    12: "A9_P-3C", 13: "A10_B-1B", 14: "A11_E-8", 15: "A12_TU-22",
    16: "A13_F-15", 17: "A14_KC-135", 18: "A15_F-22", 19: "A16_FA-18",
    20: "A17_TU-95", 21: "A18_KC-10", 22: "A19_SU-34", 23: "A20_SU-24",
    24: "FSC",
}

# 大类分组映射：25 类 → 3 个大类（舰船/飞机/车辆）
CLASS_TO_GROUP = {
    **{class_id: "ship" for class_id in range(0, 4)},
    **{class_id: "aircraft" for class_id in range(4, 24)},
    24: "vehicle",
}
GROUP_IOU_THRESHOLDS = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}

# 细粒度类别分组映射：每个类独立成组，用于输出逐类指标
PER_CLASS_TO_GROUP = {class_id: name for class_id, name in CLASS_NAMES.items()}
PER_CLASS_IOU_THRESHOLDS = {
    name: 0.35 if class_id in VEHICLE_CLASS_IDS else 0.50
    for class_id, name in CLASS_NAMES.items()
}


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    EvalResult,
    compute_iou,
    evaluate_competition_metrics,
    load_yolo_labels,
)


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


def print_progress(index: int, total: int, start_time: float, device: str):
    """在同一行实时打印推理进度。"""
    elapsed = time.perf_counter() - start_time
    speed = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / speed if speed > 0 else 0.0
    percent = index / total * 100 if total > 0 else 100.0

    # 使用 \r 刷新同一行，flush=True 确保 PowerShell 中能及时看到进度
    print(
        f"\r[i] 推理进度 {index:>5d}/{total:<5d} "
        f"{percent:6.2f}% | {speed:5.2f} img/s | "
        f"ETA {remaining:7.1f}s | {format_cuda_memory(device)}",
        end="",
        flush=True,
    )


def release_cuda_cache(device: str):
    """按需释放 Python 引用和 PyTorch CUDA 缓存。"""
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_to_records(model: YOLO, image_paths: list[Path], device: str) -> list[BoxRecord]:
    """执行 YOLO 推理并转换为比赛评测需要的 BoxRecord。"""
    pred_records: list[BoxRecord] = []
    start_time = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        # 逐张读取并传入 ndarray，避免 Ultralytics 批量 source 管线自动创建输出目录
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            device=device,
            batch=PREDICT_BATCH_SIZE,
            verbose=False,
            stream=False,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show=False,
        )
        result = results[0]

        if result.boxes is not None:
            # result.boxes 是 GPU tensor 包装对象，立刻转为 CPU list，避免持有显存引用
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            class_ids = result.boxes.cls.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            for xyxy, class_id, score in zip(boxes, class_ids, scores):
                pred_records.append(
                    BoxRecord(
                        image_id=image_path.stem,
                        class_id=int(class_id),
                        xyxy=tuple(float(value) for value in xyxy),
                        score=float(score),
                    )
                )

        # Results 对象包含原图和 tensor，单张处理完后立刻删除引用
        del result, results, image
        if index % CUDA_CACHE_CLEAR_INTERVAL == 0:
            release_cuda_cache(device)

        print_progress(index, len(image_paths), start_time, device)

    release_cuda_cache(device)
    print()
    return pred_records


def print_eval_result(name: str, result: EvalResult):
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
#  FP / FN 可视化 — 按类别保存标注图和预测图
# ══════════════════════════════════════════════════════════════════════

# BGR 颜色常量
COLOR_GT = (255, 0, 0)       # 蓝色 — 真实框
COLOR_TP = (0, 255, 0)       # 绿色 — 正确预测
COLOR_FP = (0, 0, 255)       # 红色 — 虚警
COLOR_FN_GT = (0, 165, 255)  # 橙色 — 漏检的真实框


def clear_vis_dirs():
    """清空之前的 FP / FN 可视化目录。"""
    for d in [FP_DIR, FN_DIR]:
        if d.exists():
            shutil.rmtree(d)
    # 预创建 25 个子类文件夹
    for cls_name in CLASS_NAMES.values():
        (FP_DIR / cls_name).mkdir(parents=True, exist_ok=True)
        (FN_DIR / cls_name).mkdir(parents=True, exist_ok=True)


def match_per_image_per_class(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
) -> tuple[
    dict[int, set[str]],  # fp_images: class_id → {image_ids with FP}
    dict[int, set[str]],  # fn_images: class_id → {image_ids with FN}
    dict[str, list[BoxRecord]],  # fp_boxes: image_id → FP 预测框
    dict[str, list[BoxRecord]],  # fn_boxes: image_id → FN 真实框
    dict[str, list[BoxRecord]],  # tp_preds: image_id → TP 预测框
]:
    """
    对每张图、每个类执行 class-aware 一对一匹配，返回 FP/FN 详情。

    匹配逻辑与 competition_metrics._match_single_image_group 完全一致：
    - 按置信度降序匹配
    - 每个 GT 最多匹配一个 pred，每个 pred 最多匹配一个 GT
    - class_aware：pred.class_id 必须等于 gt.class_id
    - 车辆 IoU=0.35，其他 IoU=0.50
    """
    # 按 (image_id, class_id) 分组
    gt_by_image_class: dict[str, dict[int, list[BoxRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pred_by_image_class: dict[str, dict[int, list[BoxRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for gt in gt_records:
        gt_by_image_class[gt.image_id][gt.class_id].append(gt)
    for pred in pred_records:
        pred_by_image_class[pred.image_id][pred.class_id].append(pred)

    all_image_ids = set(gt_by_image_class) | set(pred_by_image_class)

    fp_images: dict[int, set[str]] = defaultdict(set)
    fn_images: dict[int, set[str]] = defaultdict(set)
    fp_boxes: dict[str, list[BoxRecord]] = defaultdict(list)
    fn_boxes: dict[str, list[BoxRecord]] = defaultdict(list)
    tp_preds: dict[str, list[BoxRecord]] = defaultdict(list)

    for image_id in all_image_ids:
        for cls_id in range(NUM_CLASSES):
            gts = gt_by_image_class[image_id].get(cls_id, [])
            preds = pred_by_image_class[image_id].get(cls_id, [])

            if not gts and not preds:
                continue

            iou_threshold = 0.35 if cls_id in VEHICLE_CLASS_IDS else 0.50

            # 按置信度降序排列预测框
            sorted_preds = sorted(
                preds, key=lambda r: r.score if r.score is not None else 0.0, reverse=True
            )

            matched_gt: set[int] = set()  # 已匹配的 GT 索引
            fp_indices: set[int] = set()  # FP 预测框索引

            for pi, pred in enumerate(sorted_preds):
                best_gt_idx = -1
                best_iou = 0.0
                for gi, gt in enumerate(gts):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(pred.xyxy, gt.xyxy)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_gt_idx < 0:
                    fp_indices.add(pi)  # 未匹配 → FP
                else:
                    matched_gt.add(best_gt_idx)  # 匹配成功 → TP

            # 收集 FP 预测框
            for pi in fp_indices:
                fp_boxes[image_id].append(sorted_preds[pi])
                fp_images[cls_id].add(image_id)

            # 收集 FN 真实框（未被匹配的 GT）
            for gi, gt in enumerate(gts):
                if gi not in matched_gt:
                    fn_boxes[image_id].append(gt)
                    fn_images[cls_id].add(image_id)

            # 收集 TP 预测框
            for pi, pred in enumerate(sorted_preds):
                if pi not in fp_indices:
                    tp_preds[image_id].append(pred)

    return fp_images, fn_images, fp_boxes, fn_boxes, tp_preds


def _draw_box_label(
    img: "cv2.Mat",
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
):
    """在图像上绘制矩形框和文字标签。"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 标签背景
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def load_image(image_id: str, test_image_paths: list[Path]) -> "cv2.Mat | None":
    """根据 image_id 加载原始图像。"""
    for p in test_image_paths:
        if p.stem == image_id:
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def save_fp_fn_visualizations(
    fp_images: dict[int, set[str]],
    fn_images: dict[int, set[str]],
    fp_boxes: dict[str, list[BoxRecord]],
    fn_boxes: dict[str, list[BoxRecord]],
    tp_preds: dict[str, list[BoxRecord]],
    all_gt: list[BoxRecord],
    test_image_paths: list[Path],
):
    """
    按类别保存 FP/FN 可视化图像。

    保存结构:
        FP/{类名}/labeled_{image_id}.jpg  — 带 GT 标注的原始图
        FP/{类名}/pred_{image_id}.jpg     — 带预测框的图（TP 绿色 / FP 红色）
        FN/{类名}/labeled_{image_id}.jpg  — 带 GT 标注的原始图
        FN/{类名}/pred_{image_id}.jpg     — 带预测框的图
    """
    # 按 image_id 索引 GT 记录
    gt_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for gt in all_gt:
        gt_by_image[gt.image_id].append(gt)

    # 统计
    total_fp_images = 0
    total_fn_images = 0

    # ── 保存 FP 可视化 ──────────────────────────────────────────────
    for cls_id, image_ids in sorted(fp_images.items()):
        cls_name = CLASS_NAMES[cls_id]
        for image_id in sorted(image_ids):
            img = load_image(image_id, test_image_paths)
            if img is None:
                continue

            gts = gt_by_image.get(image_id, [])

            # 标注图：只画 GT 框
            labeled = img.copy()
            for gt in gts:
                x1, y1, x2, y2 = map(int, gt.xyxy)
                _draw_box_label(labeled, x1, y1, x2, y2, CLASS_NAMES[gt.class_id], COLOR_GT)

            # 预测图：TP 绿色 + FP 红色
            predicted = img.copy()
            for tp in tp_preds.get(image_id, []):
                x1, y1, x2, y2 = map(int, tp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, CLASS_NAMES[tp.class_id], COLOR_TP)
            for fp in fp_boxes.get(image_id, []):
                x1, y1, x2, y2 = map(int, fp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2,
                                f"{CLASS_NAMES[fp.class_id]}(FP)", COLOR_FP)

            cv2.imwrite(str(FP_DIR / cls_name / f"labeled_{image_id}.jpg"), labeled)
            cv2.imwrite(str(FP_DIR / cls_name / f"pred_{image_id}.jpg"), predicted)
            total_fp_images += 1

    # ── 保存 FN 可视化 ──────────────────────────────────────────────
    for cls_id, image_ids in sorted(fn_images.items()):
        cls_name = CLASS_NAMES[cls_id]
        for image_id in sorted(image_ids):
            img = load_image(image_id, test_image_paths)
            if img is None:
                continue

            gts = gt_by_image.get(image_id, [])
            fn_for_img = {b.class_id for b in fn_boxes.get(image_id, [])}

            # 标注图：GT 框，FN 用橙色高亮
            labeled = img.copy()
            for gt in gts:
                color = COLOR_FN_GT if gt.class_id in fn_for_img else COLOR_GT
                label = f"{CLASS_NAMES[gt.class_id]}(FN)" if gt.class_id in fn_for_img else CLASS_NAMES[gt.class_id]
                x1, y1, x2, y2 = map(int, gt.xyxy)
                _draw_box_label(labeled, x1, y1, x2, y2, label, color)

            # 预测图：TP 绿色
            predicted = img.copy()
            for tp in tp_preds.get(image_id, []):
                x1, y1, x2, y2 = map(int, tp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, CLASS_NAMES[tp.class_id], COLOR_TP)
            # 也画出 FP（帮助理解为什么漏检）
            for fp in fp_boxes.get(image_id, []):
                x1, y1, x2, y2 = map(int, fp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2,
                                f"{CLASS_NAMES[fp.class_id]}(FP)", COLOR_FP)

            cv2.imwrite(str(FN_DIR / cls_name / f"labeled_{image_id}.jpg"), labeled)
            cv2.imwrite(str(FN_DIR / cls_name / f"pred_{image_id}.jpg"), predicted)
            total_fn_images += 1

    # 打印统计
    print(f"\n[i] FP 可视化: {total_fp_images} 张图像 → {FP_DIR}")
    for cls_id in sorted(fp_images.keys()):
        print(f"    {CLASS_NAMES[cls_id]:12s}: {len(fp_images[cls_id])} 张")
    print(f"[i] FN 可视化: {total_fn_images} 张图像 → {FN_DIR}")
    for cls_id in sorted(fn_images.keys()):
        print(f"    {CLASS_NAMES[cls_id]:12s}: {len(fn_images[cls_id])} 张")



if __name__ == "__main__":
    # 切换到项目根目录，保证所有相对路径解析一致
    os.chdir(PROJECT_ROOT)

    test_image_paths = read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = build_image_size_map(test_image_paths)

    # 读取测试集真实框，YOLO 标注为归一化 xywh，需要按图像尺寸转换为 xyxy
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)

    # 执行模型推理，得到像素级预测框
    device = resolve_device(DEVICE)
    model = YOLO(str(WEIGHT_PATH))
    model.to(device)
    with torch.inference_mode():
        pred_records = predict_to_records(model, test_image_paths, device)
    del model
    release_cuda_cache(device)

    # 按比赛规则评估：车辆 IoU=0.35，其他目标 IoU=0.50
    config = EvalConfig(
        class_to_group=CLASS_TO_GROUP,
        group_iou_thresholds=GROUP_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    eval_results = evaluate_competition_metrics(gt_records, pred_records, config)

    print("=" * 80)
    print("比赛指标评估结果（测试集）")
    print(f"权重: {WEIGHT_PATH}")
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

    # ── FP / FN 可视化保存 ───────────────────────────────────────────
    print(f"\n[i] 正在生成 FP/FN 可视化...")
    clear_vis_dirs()
    fp_img, fn_img, fp_box, fn_box, tp_pred = match_per_image_per_class(
        gt_records, pred_records,
    )
    save_fp_fn_visualizations(
        fp_img, fn_img, fp_box, fn_box, tp_pred,
        gt_records, test_image_paths,
    )
    print(f"[✓] FP/FN 可视化保存完成")
