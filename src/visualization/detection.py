# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""检测结果可视化函数。

提供 FP/FN 错误分析可视化（标注框叠加）和混淆矩阵风格的 TP/FP/FN 统计图。
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

# 配置 matplotlib 中文字体支持（优先使用 Noto Sans CJK SC）
try:
    import matplotlib
    from matplotlib.font_manager import fontManager

    _CJK_FONT_CANDIDATES = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "AR PL UMing CN",
    ]
    _available_fonts = {f.name for f in fontManager.ttflist}
    for _font_name in _CJK_FONT_CANDIDATES:
        if _font_name in _available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
except ImportError:
    pass

from val.competition_metrics import BoxRecord, compute_iou

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ── BGR 颜色常量 ─────────────────────────────────────────────────────
COLOR_GT = (255, 0, 0)  # 蓝色 — 真实框
COLOR_TP = (0, 255, 0)  # 绿色 — 正确预测
COLOR_FP = (0, 0, 255)  # 红色 — 虚警
COLOR_FN_GT = (0, 165, 255)  # 橙色 — 漏检的真实框

# ── PIL 中文字体候选路径 ─────────────────────────────────────────────
# cv2 的 HERSHEY 字体不支持中文，绘制标题条时用 PIL 渲染；按优先级依次尝试，
# 全部缺失时回退到 PIL 默认字体。
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def clear_vis_dirs(
    fp_dir: str | Path,
    fn_dir: str | Path,
    class_names: dict[int, str],
) -> None:
    """清空之前的 FP / FN 可视化目录并预创建子类文件夹。

    Args:
        fp_dir: FP 可视化保存根目录。
        fn_dir: FN 可视化保存根目录。
        class_names: 类别 ID 到名称的映射字典。
    """
    fp_dir = Path(fp_dir)
    fn_dir = Path(fn_dir)
    for d in [fp_dir, fn_dir]:
        if d.exists():
            shutil.rmtree(d)
    for cls_name in class_names.values():
        (fp_dir / cls_name).mkdir(parents=True, exist_ok=True)
        (fn_dir / cls_name).mkdir(parents=True, exist_ok=True)


def match_per_image_per_class(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    num_classes: int,
    vehicle_class_ids: set[int],
) -> tuple[
    dict[int, set[str]],
    dict[int, set[str]],
    dict[str, list[BoxRecord]],
    dict[str, list[BoxRecord]],
    dict[str, list[BoxRecord]],
]:
    """对每张图、每个类执行 class-aware 一对一匹配，返回 FP/FN 详情。

    匹配逻辑与 ``competition_metrics._match_single_image_group`` 完全一致：
    - 按置信度降序匹配
    - 每个 GT 最多匹配一个 pred，每个 pred 最多匹配一个 GT
    - class_aware：pred.class_id 必须等于 gt.class_id
    - 车辆 IoU=0.35，其他 IoU=0.50

    Args:
        gt_records: 真实标注框记录列表。
        pred_records: 预测框记录列表。
        num_classes: 总类别数。
        vehicle_class_ids: 车辆类别的类别 ID 集合（使用 IoU=0.35）。

    Returns:
        ``(fp_images, fn_images, fp_boxes, fn_boxes, tp_preds)`` 五元组：
        - fp_images: class_id → {存在 FP 的 image_id 集合}
        - fn_images: class_id → {存在 FN 的 image_id 集合}
        - fp_boxes: image_id → FP 预测框列表
        - fn_boxes: image_id → FN 真实框列表
        - tp_preds: image_id → TP 预测框列表
    """
    # 按 (image_id, class_id) 分组
    gt_by_image_class: dict[str, dict[int, list[BoxRecord]]] = defaultdict(lambda: defaultdict(list))
    pred_by_image_class: dict[str, dict[int, list[BoxRecord]]] = defaultdict(lambda: defaultdict(list))
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
        for cls_id in range(num_classes):
            gts = gt_by_image_class[image_id].get(cls_id, [])
            preds = pred_by_image_class[image_id].get(cls_id, [])

            if not gts and not preds:
                continue

            iou_threshold = 0.35 if cls_id in vehicle_class_ids else 0.50

            # 按置信度降序排列预测框
            sorted_preds = sorted(preds, key=lambda r: r.score if r.score is not None else 0.0, reverse=True)

            matched_gt: set[int] = set()
            fp_indices: set[int] = set()

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
                    fp_indices.add(pi)
                else:
                    matched_gt.add(best_gt_idx)

            # 收集 FP 预测框
            for pi in fp_indices:
                fp_boxes[image_id].append(sorted_preds[pi])
                fp_images[cls_id].add(image_id)

            # 收集 FN 真实框
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
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """在图像上绘制矩形框和文字标签。"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _draw_box_label_thin(
    img: "cv2.Mat",
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """绘制细框小字版本（线宽 1、字号 0.3、标签条更薄）。

    供大图可视化使用：大图缩放后框数量可达数百个，粗框大字会互相遮挡； 细框小字保证密集框场景下可读性。
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    cv2.rectangle(img, (x1, y1 - th - 2), (x1 + tw + 1, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def _draw_tile_grid(
    img: "cv2.Mat",
    image_size: tuple[int, int],
    tile_size: int,
    overlap: int,
    scale: float,
) -> None:
    """在图上绘制滑窗切分网格线（红色细框），直观展示大图如何被切成小块。

    每个 tile 的画一个红色细线矩形（线宽 1）；tile 原点由 ``tile_origins``
    按 ``(tile_size, overlap)`` 生成，坐标乘 *scale* 换算到展示尺寸。

    Args:
        img: 待绘制的图像（原地修改，BGR）。
        image_size: 原图 (宽, 高)。
        tile_size: 滑窗边长（= 模型输入分辨率）。
        overlap: 滑窗重叠像素数。
        scale: 原图坐标到展示尺寸的缩放系数。
    """
    # 惰性 import：visualization ↔ scripts 存在潜在循环依赖（tiling 顶层引用
    # eval_lib，eval_lib 顶层引用本模块），运行时才加载即安全
    from scripts.tiling import tile_origins

    for x0, y0 in tile_origins(image_size, tile_size, overlap):
        x1, y1 = int(round(x0 * scale)), int(round(y0 * scale))
        x2 = int(round(min(x0 + tile_size, image_size[0]) * scale))
        y2 = int(round(min(y0 + tile_size, image_size[1]) * scale))
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FP, 1)


def load_image(image_id: str, test_image_paths: list[Path]) -> "cv2.Mat | None":
    """根据 image_id 加载原始图像。

    Args:
        image_id: 图像文件名（不含扩展名）。
        test_image_paths: 测试图像路径列表。

    Returns:
        加载的图像矩阵，未找到时返回 None。
    """
    for p in test_image_paths:
        if p.stem == image_id:
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def _compose_side_by_side(
    left: "cv2.Mat",
    right: "cv2.Mat",
    left_label: str,
    right_label: str,
) -> "cv2.Mat":
    """把左右两张图水平拼接为一张对比图，顶部各带中文标题条。

    cv2 自带的 HERSHEY 字体不支持中文，因此标题条改用 PIL 绘制后再转回 BGR。
    左右两张图来自同一张原始图像的不同标注版本，尺寸一致，可直接水平拼接。

    Args:
        left: 左面板图像（BGR）。
        right: 右面板图像（BGR）。
        left_label: 左面板标题文字。
        right_label: 右面板标题文字。

    Returns:
        水平拼接后的对比图（BGR），宽度为左右两图宽度之和。
    """
    from PIL import Image, ImageDraw, ImageFont

    def _with_title_bar(img: "cv2.Mat", label: str) -> "cv2.Mat":
        """给单张图顶部加一个深色标题条并绘制中文标题。"""
        header_h = 30
        height, width = img.shape[:2]
        canvas = np.full((height + header_h, width, 3), 38, dtype=np.uint8)
        canvas[header_h:, :, :] = img

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = None
        for path in _CJK_FONT_PATHS:
            try:
                font = ImageFont.truetype(path, 18)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        draw.text((10, 6), label, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    left_panel = _with_title_bar(left, left_label)
    right_panel = _with_title_bar(right, right_label)
    return np.hstack((left_panel, right_panel))


def save_fp_fn_visualizations(
    fp_images: dict[int, set[str]],
    fn_images: dict[int, set[str]],
    fp_boxes: dict[str, list[BoxRecord]],
    fn_boxes: dict[str, list[BoxRecord]],
    tp_preds: dict[str, list[BoxRecord]],
    all_gt: list[BoxRecord],
    test_image_paths: list[Path],
    class_names: dict[int, str],
    fp_dir: str | Path,
    fn_dir: str | Path,
) -> None:
    """按类别保存 FP/FN 可视化图像。

    每张图只保存一张**左右拼接对比图**：左侧为带 GT 标注的原始图，右侧为带
    预测框的图，两张图各自顶部带中文标题条，方便上下/左右对照查看。保存结构::

        {fp_dir}/{类名}/{image_id}.jpg  — 左：GT 标注图；右：预测图（TP 绿色 / FP 红色）
        {fn_dir}/{类名}/{image_id}.jpg  — 左：GT 标注图（FN 橙色）；右：预测图（TP 绿色 / FP 红色）

    Args:
        fp_images: class_id → 存在 FP 的 image_id 集合。
        fn_images: class_id → 存在 FN 的 image_id 集合。
        fp_boxes: image_id → FP 预测框列表。
        fn_boxes: image_id → FN 真实框列表。
        tp_preds: image_id → TP 预测框列表。
        all_gt: 所有真实标注框记录。
        test_image_paths: 测试图像路径列表。
        class_names: 类别 ID 到名称的映射字典。
        fp_dir: FP 可视化保存根目录。
        fn_dir: FN 可视化保存根目录。
    """
    fp_dir = Path(fp_dir)
    fn_dir = Path(fn_dir)

    # 按 image_id 索引 GT 记录
    gt_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for gt in all_gt:
        gt_by_image[gt.image_id].append(gt)

    total_fp_images = 0
    total_fn_images = 0

    # ── 保存 FP 可视化 ──────────────────────────────────────────────
    for cls_id, image_ids in sorted(fp_images.items()):
        cls_name = class_names[cls_id]
        for image_id in sorted(image_ids):
            img = load_image(image_id, test_image_paths)
            if img is None:
                continue

            gts = gt_by_image.get(image_id, [])

            # 标注图：只画 GT 框
            labeled = img.copy()
            for gt in gts:
                x1, y1, x2, y2 = map(int, gt.xyxy)
                _draw_box_label(labeled, x1, y1, x2, y2, class_names[gt.class_id], COLOR_GT)

            # 预测图：TP 绿色 + FP 红色
            predicted = img.copy()
            for tp in tp_preds.get(image_id, []):
                x1, y1, x2, y2 = map(int, tp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, class_names[tp.class_id], COLOR_TP)
            for fp in fp_boxes.get(image_id, []):
                x1, y1, x2, y2 = map(int, fp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, f"{class_names[fp.class_id]}(FP)", COLOR_FP)

            # 左右拼接对比图：左 GT 标注、右预测框（TP 绿 / FP 红），保存为单张
            combined = _compose_side_by_side(labeled, predicted, "GT（真实框）", "Pred（模型预测）")
            cv2.imwrite(str(fp_dir / cls_name / f"{image_id}.jpg"), combined)
            total_fp_images += 1

    # ── 保存 FN 可视化 ──────────────────────────────────────────────
    for cls_id, image_ids in sorted(fn_images.items()):
        cls_name = class_names[cls_id]
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
                label = f"{class_names[gt.class_id]}(FN)" if gt.class_id in fn_for_img else class_names[gt.class_id]
                x1, y1, x2, y2 = map(int, gt.xyxy)
                _draw_box_label(labeled, x1, y1, x2, y2, label, color)

            # 预测图：TP 绿色
            predicted = img.copy()
            for tp in tp_preds.get(image_id, []):
                x1, y1, x2, y2 = map(int, tp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, class_names[tp.class_id], COLOR_TP)
            for fp in fp_boxes.get(image_id, []):
                x1, y1, x2, y2 = map(int, fp.xyxy)
                _draw_box_label(predicted, x1, y1, x2, y2, f"{class_names[fp.class_id]}(FP)", COLOR_FP)

            # 左右拼接对比图：左 GT 标注（FN 橙色）、右预测框（TP 绿 / FP 红），保存为单张
            combined = _compose_side_by_side(labeled, predicted, "GT（真实框）", "Pred（模型预测）")
            cv2.imwrite(str(fn_dir / cls_name / f"{image_id}.jpg"), combined)
            total_fn_images += 1

    # 打印统计
    print(f"\n[i] FP 可视化: {total_fp_images} 张图像 → {fp_dir}")
    for cls_id in sorted(fp_images.keys()):
        print(f"    {class_names[cls_id]:12s}: {len(fp_images[cls_id])} 张")
    print(f"[i] FN 可视化: {total_fn_images} 张图像 → {fn_dir}")
    for cls_id in sorted(fn_images.keys()):
        print(f"    {class_names[cls_id]:12s}: {len(fn_images[cls_id])} 张")


def save_large_image_visualizations(
    image_ids: list[str],
    all_gt: list[BoxRecord],
    pred_records: list[BoxRecord],
    test_image_paths: list[Path],
    class_names: dict[int, str],
    output_dir: str | Path,
    count: int,
    seed: int = 0,
    tile_size: int | None = None,
    tile_overlap: int = 0,
) -> None:
    """随机抽选大图，保存左 GT / 右 Predict 的左右拼接对比图（展示切分结果）。

    大图（超过模型分辨率、走滑窗切分的图像）直接显示尺寸过大（最大 12000²），
    先缩放到固定展示长边（1600px）再画框，框坐标按同比例缩放。框用细线小字
    （``_draw_box_label_thin``）绘制，避免密集框互相遮挡：左面板 GT 蓝色框
    （类名），右面板预测绿色框（类名 + 置信度），并可选叠加**红色滑窗网格线**
    （``tile_size``/``tile_overlap`` 非空时），直观展示大图如何被切成小块送入
    检测。保存结构::

        {output_dir}/{image_id}.jpg

    Args:
        image_ids: 大图 image_id 列表（随机抽样的候选池）。
        all_gt: 全部真实框记录（按 image_id 过滤出所选大图）。
        pred_records: 全部预测框记录（按 image_id 过滤）。
        test_image_paths: 测试图像路径列表。
        class_names: 类别 ID 到名称的映射字典。
        output_dir: 输出目录（自动创建）。
        count: 随机抽选的大图数量；大于候选池时取全部。
        seed: 随机抽样种子（固定值保证多次运行可视化可复现）。
        tile_size: 滑窗边长（= 模型输入分辨率）；``None`` = 不绘制分割网格线。
        tile_overlap: 滑窗重叠像素数（与切分推理的 overlap 一致，保证网格与
            实际切块位置吻合）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 固定种子抽样，保证多次运行选中的大图一致
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(image_ids), min(count, len(image_ids))))

    # 按 image_id 索引 GT 与预测记录
    gt_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for gt in all_gt:
        gt_by_image[gt.image_id].append(gt)
    pred_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for pred in pred_records:
        pred_by_image[pred.image_id].append(pred)

    display_max = 1600  # 展示长边上限：大图直接显示过大，缩放后画框
    for image_id in chosen:
        img = load_image(image_id, test_image_paths)
        if img is None:
            continue
        height, width = img.shape[:2]
        scale = min(1.0, display_max / max(height, width))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # 缩放后框坐标需按同比例换算
        def _scaled_xyxy(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
            """把原图坐标的框换算到展示尺寸（乘缩放系数）。"""
            return tuple(int(round(v * scale)) for v in box)

        # 左面板：GT 蓝色框（类名），细框小字
        labeled = img.copy()
        for gt in gt_by_image.get(image_id, []):
            x1, y1, x2, y2 = _scaled_xyxy(gt.xyxy)
            _draw_box_label_thin(labeled, x1, y1, x2, y2, class_names[gt.class_id], COLOR_GT)

        # 右面板：预测绿色框（类名 + 置信度），细框小字；先画红色滑窗网格线
        predicted = img.copy()
        if tile_size is not None:
            _draw_tile_grid(predicted, (width, height), tile_size, tile_overlap, scale)
        for pred in pred_by_image.get(image_id, []):
            x1, y1, x2, y2 = _scaled_xyxy(pred.xyxy)
            label = f"{class_names[pred.class_id]} {pred.score:.2f}"
            _draw_box_label_thin(predicted, x1, y1, x2, y2, label, COLOR_TP)

        # 左右拼接对比图：左 GT 标注、右预测框（含切分网格）
        combined = _compose_side_by_side(labeled, predicted, "GT（真实框）", "Pred（模型预测）")
        cv2.imwrite(str(output_dir / f"{image_id}.jpg"), combined)
        print(
            f"[i] 大图对比可视化: {image_id}（{len(gt_by_image.get(image_id, []))} GT / "
            f"{len(pred_by_image.get(image_id, []))} Pred）→ {output_dir / f'{image_id}.jpg'}"
        )
        del img, labeled, predicted, combined

    print(f"[i] 大图对比可视化完成: {len(chosen)} 张 → {output_dir}")


def build_confusion_matrix(
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    num_classes: int,
    vehicle_class_ids: set[int],
) -> "np.ndarray":
    """构建 YOLO 风格的混淆矩阵。

    对每张图像执行 class-agnostic IoU 匹配（按置信度降序贪婪匹配），
    记录预测类别与真实类别的对应关系。

    矩阵形状为 ``(num_classes + 1, num_classes + 1)``：
    - 行（Y 轴）：Predicted class
    - 列（X 轴）：True class
    - 最后一行 (index ``num_classes``)：Background FN（漏检真实框）
    - 最后一列 (index ``num_classes``)：Background FP（误检预测框）
    - 对角线：正确预测（TP）

    Args:
        gt_records: 真实标注框记录列表。
        pred_records: 预测框记录列表。
        num_classes: 总类别数。
        vehicle_class_ids: 车辆类别的类别 ID 集合（使用 IoU=0.35）。

    Returns:
        ``(num_classes + 1, num_classes + 1)`` 的混淆矩阵（整数计数）。
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.float64)

    # 按 image_id 分组
    gt_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    pred_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for gt in gt_records:
        gt_by_image[gt.image_id].append(gt)
    for pred in pred_records:
        pred_by_image[pred.image_id].append(pred)

    all_image_ids = set(gt_by_image) | set(pred_by_image)

    for image_id in all_image_ids:
        gts = gt_by_image.get(image_id, [])
        preds = pred_by_image.get(image_id, [])

        if not preds:
            # 无预测 → 所有 GT 都是 FN
            for gt in gts:
                matrix[num_classes, gt.class_id] += 1
            continue
        if not gts:
            # 无 GT → 所有预测都是 FP
            for pred in preds:
                matrix[pred.class_id, num_classes] += 1
            continue

        # 按置信度降序排列预测框
        sorted_preds = sorted(preds, key=lambda r: r.score if r.score is not None else 0.0, reverse=True)

        matched_gt_indices: set[int] = set()

        for pred in sorted_preds:
            best_gt_idx = -1
            best_iou = 0.0

            for gi, gt in enumerate(gts):
                if gi in matched_gt_indices:
                    continue
                # 根据 GT 类别确定 IoU 阈值
                iou_threshold = 0.35 if gt.class_id in vehicle_class_ids else 0.50
                iou = compute_iou(pred.xyxy, gt.xyxy)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_gt_idx < 0:
                # 未匹配任何 GT → Background FP
                matrix[pred.class_id, num_classes] += 1
            else:
                # 匹配成功 → 记录预测类别 vs 真实类别
                gt_class = gts[best_gt_idx].class_id
                matrix[pred.class_id, gt_class] += 1
                matched_gt_indices.add(best_gt_idx)

        # 未匹配的 GT → Background FN
        for gi, gt in enumerate(gts):
            if gi not in matched_gt_indices:
                matrix[num_classes, gt.class_id] += 1

    return matrix


def plot_confusion_matrix(
    matrix: "np.ndarray",
    class_names: dict[int, str],
    output_path: str | None = None,
    normalize: bool = True,
) -> "Figure":
    """绘制 YOLO 风格的混淆矩阵热力图。

    行（Y 轴）= Predicted，列（X 轴）= True。
    最后一行/列为 Background FN/FP。

    Args:
        matrix: ``build_confusion_matrix()`` 返回的混淆矩阵。
        class_names: 类别 ID 到名称的映射字典。
        output_path: 可选的图像保存路径，为 ``None`` 时仅返回 Figure 不保存。
        normalize: 是否按列归一化（默认 ``True``）。

    Returns:
        matplotlib Figure 对象。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for confusion matrix plotting. Install with: pip install matplotlib"
        ) from exc

    nc = len(class_names)
    names = [class_names[i] for i in range(nc)]
    xticklabels = names + ["background FP"]
    yticklabels = names + ["background FN"]

    # 列归一化
    if normalize:
        col_sum = matrix.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1  # 避免除零
        array = matrix / col_sum
    else:
        array = matrix.copy()

    # 抑制极小值避免标注杂乱
    array_display = array.copy()
    array_display[array_display < 0.005] = np.nan

    fig, ax = plt.subplots(figsize=(max(12, nc * 0.45), max(10, nc * 0.4)), tight_layout=True)

    im = ax.imshow(array_display, cmap="Blues", aspect="auto")

    # 标注每个非零格子的数值
    for i in range(nc + 1):
        for j in range(nc + 1):
            val = array[i, j]
            if val > 0.005:
                # 根据背景亮度选择文字颜色
                color = "white" if val > 0.5 else "black"
                text = f"{val:.2f}" if normalize else str(int(matrix[i, j]))
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    # 设置颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Normalized Count" if normalize else "Count", fontsize=10)

    ax.set_xticks(range(nc + 1))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax.set_yticks(range(nc + 1))
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # 隐藏边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 在 background 行列处绘制分割线
    ax.axhline(y=nc - 0.5, color="black", linewidth=1.5)
    ax.axvline(x=nc - 0.5, color="black", linewidth=1.5)

    if output_path is not None:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    return fig
