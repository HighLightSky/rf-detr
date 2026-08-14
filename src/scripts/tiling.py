# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图滑窗切分推理模块（里程碑 1：朴素滑窗 + 按类别 NMS 去重基线）。

比赛测试集包含超过模型分辨率（1024）的大图（最大 12000px 量级），无法整图
直接推理。本模块提供：

1. ``split_image_paths``：按图像尺寸把小图（整图直连）与大图（滑窗切分）分流。
2. ``tile_origins``：生成滑窗原点网格（带 overlap，末块夹取贴边，无缝隙）。
3. ``apply_nms``：按类别分组的 NMS 去重（类间不互相抑制）。
4. ``tile_predict_records``：大图切块批量推理（worker 预取流水线），坐标映射
   回全图后按策略合并（nms / center / center_rescue）。

切分路径采用与整图路径（``predict_batched_to_records``）同构的 worker 预取
流水线：DataLoader 的多个 worker 进程并行完成大图解码与切块，主线程把来自
**不同大图**的 tile 混合成 GPU 批量持续前向——避免逐张串行解码大图时
（单张 12000² 解码耗时数秒）GPU 空转。每个 worker 只缓存当前一张解码大图
（~400MB），tile 按图连续排列使缓存命中率最优。

朴素基线阶段接受两个已知缺陷（记录为后续里程碑的对照）：
- 跨块目标被切开时可能产生双侧残缺框（由 NMS 及评测口径兜底）；
- NMS 可能把重叠区内两个真实相邻目标误合并（后续用 core/halo 中心归属替代）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from scripts.eval_lib import (
    _GpuUtilMonitor,
    _worker_init_fn,
    filter_postprocess_results,
    print_progress,
)
from val.competition_metrics import BoxRecord

if TYPE_CHECKING:
    from rfdetr import RFDETR


def _axis_origins(dim: int, tile: int, stride: int) -> list[int]:
    """生成单轴滑窗原点序列。

    序列为 0, stride, 2*stride, …, 末位夹取到 ``dim - tile``，保证滑窗覆盖到
    图像右/下边缘；相邻起点差不超过 stride，不存在缝隙。

    Args:
        dim: 图像在该轴上的尺寸（像素）。
        tile: 滑窗在该轴上的尺寸（像素）。
        stride: 步长（像素，= tile - overlap）。

    Returns:
        原点列表；``dim <= tile`` 时仅返回 ``[0]``（该轴无需滑动，单块覆盖，
        短轴不足 tile 的部分由调用方按实际内容尺寸处理）。
    """
    if dim <= tile:
        return [0]
    last = dim - tile
    origins: list[int] = []
    pos = 0
    while True:
        origins.append(pos)
        if pos >= last:
            break
        pos = min(pos + stride, last)
    return origins


def tile_origins(image_size: tuple[int, int], tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """生成滑窗原点网格（行优先）。

    Args:
        image_size: 图像 (宽, 高)（像素）。
        tile_size: 滑窗边长（应等于模型输入分辨率）。
        overlap: 相邻滑窗重叠像素数，须满足 ``0 <= overlap < tile_size``。

    Returns:
        按行优先排列的 ``(x0, y0)`` 原点列表；任一轴尺寸不超过 tile_size 时
        该轴仅原点 0。每块实际内容尺寸为 ``(min(x0+tile, W)-x0, min(y0+tile, H)-y0)``，
        长轴恒为 tile_size（原点已夹取），短轴可能不足 tile_size（由调用方按内容
        尺寸传 ``target_sizes`` 映射坐标）。

    Raises:
        ValueError: overlap 超出 ``[0, tile_size)`` 范围。
    """
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap 必须在 [0, {tile_size}) 内，实际为 {overlap}")
    width, height = image_size
    stride = tile_size - overlap
    xs = _axis_origins(width, tile_size, stride)
    ys = _axis_origins(height, tile_size, stride)
    return [(x, y) for y in ys for x in xs]


def split_image_paths(
    image_paths: list[Path],
    image_size_map: Mapping[str, tuple[int, int]],
    resolution: int,
) -> tuple[list[Path], list[Path]]:
    """按图像尺寸把测试图像拆分为小图/大图两组。

    Args:
        image_paths: 全部测试图像路径。
        image_size_map: ``{stem: (width, height)}`` 尺寸映射（build_image_size_map 产物）。
        resolution: 模型输入分辨率；``max(w, h) <= resolution`` 视为小图（整图直连）。

    Returns:
        ``(小图列表, 大图列表)``；大图 ``max(w, h) > resolution``，需要滑窗切分。
    """
    small: list[Path] = []
    large: list[Path] = []
    for image_path in image_paths:
        width, height = image_size_map[image_path.stem]
        if max(width, height) <= resolution:
            small.append(image_path)
        else:
            large.append(image_path)
    return small, large


def apply_nms(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """按类别分组执行 NMS，返回保留框索引。

    同一类别的重叠框（IoU > 阈值）只保留分数最高者；不同类别之间不互相抑制，
    保证相邻但属于不同类别的目标（如大船+小船）不会被误删。

    Args:
        boxes: ``(N, 4)`` xyxy 全图坐标张量。
        labels: ``(N,)`` 类别 id 张量（int64，决定按类分组）。
        scores: ``(N,)`` 置信度张量。
        iou_threshold: IoU 阈值。

    Returns:
        保留框索引（按分数降序）。
    """
    # 逐类别分别执行 NMS（等价于 batched_nms 的按类分组语义，类间不互相抑制）。
    # 注意：不直接用 torchvision.ops.batched_nms —— 其 coordinate-trick 路径在
    # torchvision 0.27.x 上报 "dets should have the same type as scores"（数值类型
    # 相同也会触发），逐类循环规避该问题且对 25 类场景开销可忽略。
    keep_parts: list[torch.Tensor] = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        class_keep = torchvision.ops.nms(boxes[class_mask], scores[class_mask], iou_threshold)
        keep_parts.append(class_indices[class_keep])
    if not keep_parts:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    keep = torch.cat(keep_parts)
    # 按分数降序排列（与 batched_nms 的返回顺序约定一致）
    return keep[torch.argsort(scores[keep], descending=True)]


# ── center 策略（里程碑 2）常量 ─────────────────────────────────────────
# 极严格安全合并阈值：只抑制"真正同一目标的跨 tile 重复预测"。IoU > 0.9
# 在数学上已蕴含面积比 < 1.11 且近等大框中心距很小，后两个条件是保险丝；
# "不同 tile"门槛保证同 tile 内相邻真实目标永不合并。
CENTER_MERGE_IOU = 0.9
CENTER_MERGE_CENTER_DIST_FRAC = 0.1
CENTER_MERGE_AREA_RATIO_MIN = 0.5
CENTER_MERGE_AREA_RATIO_MAX = 2.0

# ── center_rescue 策略（里程碑 3）常量 ─────────────────────────────────
# 触边残框定向重裁：重裁块从残框内侧边向缺口方向取 R 宽（留 margin），
# 目标跨度 D ≤ R - margin 时必然完整落入新块。
RESCUE_MARGIN = 24  # 重裁块内侧边外留白；R - margin = 1000 ≥ 目标跨度上限
RESCUE_TOUCH_TOL = 2.0  # 触边判定容差：postprocess clamp 后精确贴边，容忍预估贴内几像素
RESCUE_CONTAIN_TOL = 2.0  # "候选框包含残框"容差（浮点噪声）
RESCUE_MIN_GAIN = 4.0  # 候选框向缺口方向最小延伸量；低于此视为未救回，保留残框
RESCUE_SENTINEL_BASE = 1 << 20  # 重检框唯一哨兵 tile origin 基址（与真实 origin 永不冲突）

# 支持的大图合并策略：
#   "nms"（里程碑 1 基线）= 全部保留 + 按类别 NMS；
#   "center"（里程碑 2）= 中心归属 + 跨 tile 极严格安全合并；
#   "center_rescue"（里程碑 3）= center + 触边残框定向重裁（救回被切大目标）。
TILE_STRATEGIES = ("nms", "center", "center_rescue")


def _check_tile_strategy(strategy: str) -> None:
    """校验大图合并策略名。

    Args:
        strategy: ``"nms"`` / ``"center"`` / ``"center_rescue"``（见 ``TILE_STRATEGIES``）。

    Raises:
        ValueError: 策略名不在支持列表内。
    """
    if strategy not in TILE_STRATEGIES:
        raise ValueError(f"tile_strategy 必须是 {'/'.join(TILE_STRATEGIES)}，实际为 {strategy!r}")


def _axis_core_bounds(dim: int, origin: int, resolution: int, halo: int) -> tuple[int, int]:
    """计算单轴的核心区间 ``[lo, hi)``（全图坐标）。

    halo 只在**存在邻居的一侧**剥离：

    - 首块/短轴（``origin == 0``）：左侧无邻居 → ``lo = origin``；
    - 末块（``content_end == dim``）：右侧无邻居 → ``hi = dim``（延伸到图边缘）；
    - 中块：两侧都剥离 halo。

    该约定保证所有块核心的并集 = 全图（无空洞），图像边缘条带内的目标中心
    不会被结构性丢弃（naive 公式 ``[origin+halo, origin+R-halo]`` 会在首末块
    留下边缘空洞造成漏检）。

    Args:
        dim: 图像在该轴上的尺寸。
        origin: 该块在该轴上的原点。
        resolution: 滑窗边长（= 模型输入分辨率）。
        halo: 单侧重叠余量（= overlap // 2）。

    Returns:
        ``(lo, hi)`` 核心区间（lo 含、hi 不含）。
    """
    content_end = min(origin + resolution, dim)
    lo = origin + halo if origin > 0 else origin
    hi = content_end - halo if content_end < dim else content_end
    return lo, hi


def tile_core_bounds(
    image_size: tuple[int, int],
    origin: tuple[int, int],
    resolution: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    """计算单个 tile 的核心区（全图坐标，lo 含、hi 不含）。

    核心区 = 该 tile 独享归属的区域：预测框中心落在核心区内的框才被保留
    （center 策略）。相邻 tile 的核心区恰好相邻铺满全图（无缝隙），每个目标
    的中心至多落在一个核心区内 → 跨 tile 重复预测在源头消除。

    已知特性：夹取贴边的末块与前一块的核心区可能大范围重叠（最坏
    ``resolution - 2*overlap`` px），该条带内目标被两块同时保留，由
    ``merge_center_duplicates`` 兜底（两块都完整看到目标，框近乎相同）。

    Args:
        image_size: 图像 (宽, 高)。
        origin: tile 原点 ``(x0, y0)``。
        resolution: 滑窗边长（= 模型输入分辨率）。
        overlap: 滑窗重叠像素数。

    Returns:
        ``(x_lo, y_lo, x_hi, y_hi)`` 核心区边界（lo 含、hi 不含）。
    """
    width, height = image_size
    x0, y0 = origin
    halo = overlap // 2
    x_lo, x_hi = _axis_core_bounds(width, x0, resolution, halo)
    y_lo, y_hi = _axis_core_bounds(height, y0, resolution, halo)
    return x_lo, y_lo, x_hi, y_hi


def merge_center_duplicates(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    tile_origins: torch.Tensor,
) -> torch.Tensor:
    """极严格安全合并：仅抑制跨 tile 的同一目标重复框，返回保留索引。

    中心归属（core/halo）已结构性消除大部分重复预测；本函数兜剩余噪声源：
    预测噪声把同一目标的中心推入相邻核心区、奇数 overlap 的 1px 歧义条带、
    夹取末块与前一块核心区的大范围重叠条带。只抑制**同时满足全部条件**的
    框对：

    - 来自不同 tile（``tile_origins`` 不同）；
    - 同类别；
    - IoU > ``CENTER_MERGE_IOU``；
    - 中心距离 < ``CENTER_MERGE_CENTER_DIST_FRAC`` × min(两框对角线)；
    - 面积比 ∈ (``CENTER_MERGE_AREA_RATIO_MIN``, ``CENTER_MERGE_AREA_RATIO_MAX``) 严格。

    同一 tile 内的框对永不合并——相邻真实目标（港口大船+小船）完全免疫，
    这是与 NMS 的本质安全差异（NMS 会误合并重叠区内两个真实相邻目标）。

    Args:
        boxes: ``(N, 4)`` xyxy 全图坐标张量。
        labels: ``(N,)`` 类别 id 张量（int64）。
        scores: ``(N,)`` 置信度张量。
        tile_origins: ``(N, 2)`` 每框所属 tile 原点（int64）。

    Returns:
        保留框索引（按分数降序），与 ``apply_nms`` 的返回序一致。
    """
    tile_origins = tile_origins.to(boxes.device)
    keep_parts: list[torch.Tensor] = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        class_boxes = boxes[indices]
        if class_boxes.shape[0] == 0:
            continue
        class_scores = scores[indices]
        class_tiles = tile_origins[indices]

        # 两两条件矩阵（类内一次性向量化）
        iou = torchvision.ops.box_iou(class_boxes, class_boxes)
        same_tile = (class_tiles[:, None, 0] == class_tiles[None, :, 0]) & (
            class_tiles[:, None, 1] == class_tiles[None, :, 1]
        )
        centers = (class_boxes[:, :2] + class_boxes[:, 2:]) * 0.5
        center_dist = ((centers[:, None] - centers[None, :]) ** 2).sum(-1).sqrt()
        diag = ((class_boxes[:, 2] - class_boxes[:, 0]) ** 2 + (class_boxes[:, 3] - class_boxes[:, 1]) ** 2).sqrt()
        min_diag = torch.minimum(diag[:, None], diag[None, :])
        areas = (class_boxes[:, 2] - class_boxes[:, 0]) * (class_boxes[:, 3] - class_boxes[:, 1])
        area_ratio = areas[:, None] / areas[None, :]
        pair = (iou > CENTER_MERGE_IOU) & ~same_tile
        pair &= center_dist < CENTER_MERGE_CENTER_DIST_FRAC * min_diag
        pair &= (area_ratio > CENTER_MERGE_AREA_RATIO_MIN) & (area_ratio < CENTER_MERGE_AREA_RATIO_MAX)

        # 分数降序贪心：已被保留的更高分框满足全部条件则抑制
        order = torch.argsort(class_scores, descending=True)
        kept = torch.zeros(class_boxes.shape[0], dtype=torch.bool, device=class_boxes.device)
        for i in order:
            if not pair[kept, i].any():
                kept[i] = True
        keep_parts.append(indices[kept])
    if not keep_parts:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    keep = torch.cat(keep_parts)
    return keep[torch.argsort(scores[keep], descending=True)]


def _fragment_rescue_sides(
    fragment_xyxy: tuple[float, float, float, float],
    tile_origin: tuple[int, int],
    tile_content: tuple[int, int],
    image_size: tuple[int, int],
    tol: float = RESCUE_TOUCH_TOL,
) -> tuple[bool, bool, bool, bool]:
    """判定残框触到所在 tile 的哪些边（全图坐标）。

    残框 = 预测框触到 tile 的内容边界（容差 *tol*），且该边**不是图像边界**
    （首/末块或短轴单块的 tile 边即图像边，贴边的框可能是完整目标）。触边
    是"目标被切"的可靠信号，但 *tol* 引入少量假阳性（完整目标紧贴 tile 边），
    安全性由 ``_rescue_accept`` 的 acceptance 闸门兜底。

    Args:
        fragment_xyxy: 残框 (x0, y0, x1, y1)（全图坐标）。
        tile_origin: 该框所属 tile 的原点 (tx0, ty0)。
        tile_content: tile 内容尺寸 (tw, th)（短轴可能不足 R）。
        image_size: 图像 (宽, 高)。
        tol: 触边容差（像素）。

    Returns:
        ``(left, right, top, bottom)``：残框触到的边；任何一边为 True 即需重裁。
    """
    fx0, fy0, fx1, fy1 = fragment_xyxy
    tx0, ty0 = tile_origin
    tw, th = tile_content
    width, height = image_size
    left = (fx0 - tx0 <= tol) and (tx0 > 0)
    right = ((tx0 + tw) - fx1 <= tol) and (tx0 + tw < width)
    top = (fy0 - ty0 <= tol) and (ty0 > 0)
    bottom = ((ty0 + th) - fy1 <= tol) and (ty0 + th < height)
    return left, right, top, bottom


def _rescue_crop_bounds(
    fragment_xyxy: tuple[float, float, float, float],
    rescue_sides: tuple[bool, bool, bool, bool],
    image_size: tuple[int, int],
    resolution: int,
    margin: int = RESCUE_MARGIN,
) -> tuple[int, int, int, int]:
    """生成残框的重裁块范围 ``(x0, y0, w, h)``（全图坐标）。

    每轴独立：触边轴以残框**内侧边**为基准向缺口方向取 R 宽（留 margin，
    目标跨度 D ≤ R - margin 时必然完整落入）；不触边轴中心对齐（残框在该轴
    跨度 ≤ R，必被包含）；双轴同触（目标跨度 > R）中心对齐兜底（只能部分
    延伸——已知极限）。块夹取到图像范围，夹取后内容可能 < R（由调用方
    np.pad 补到 R×R）。

    Args:
        fragment_xyxy: 残框 (x0, y0, x1, y1)（全图坐标）。
        rescue_sides: ``_fragment_rescue_sides`` 的判定结果 (left, right, top, bottom)。
        image_size: 图像 (宽, 高)。
        resolution: 滑窗边长（= 模型输入分辨率）。
        margin: 内侧边外留白。

    Returns:
        ``(x0, y0, w, h)`` 重裁块范围（w/h 为内容尺寸，可能 < resolution）。
    """
    fx0, fy0, fx1, fy1 = fragment_xyxy
    left, right, top, bottom = rescue_sides
    width, height = image_size
    block = resolution - margin  # 触边轴可容纳的目标跨度上限

    def axis(f0: float, f1: float, lo_touch: bool, hi_touch: bool, dim: int) -> tuple[int, int]:
        """计算单轴 (start, 内容宽)。"""
        if lo_touch and not hi_touch:
            start = f1 - block  # 目标向 -∞ 方向延伸：内侧边 f1 锚定，右侧留 margin
        elif hi_touch and not lo_touch:
            start = f0 - margin  # 目标向 +∞ 方向延伸：内侧边 f0 锚定，左侧留 margin
        else:
            start = (f0 + f1) / 2 - resolution / 2  # 不触边/双轴同触：中心对齐
        # 夹取只把块往目标方向推，不丢目标；上界 max(dim - R, 0) 保证短轴
        # (dim < R) 时夹到 0 而非负数
        start = min(max(start, 0), max(dim - resolution, 0))
        content = min(start + resolution, dim) - start
        return int(round(start)), int(content)

    x0, w = axis(fx0, fx1, left, right, width)
    y0, h = axis(fy0, fy1, top, bottom, height)
    return x0, y0, w, h


def _rescue_accept(
    candidate_xyxy: tuple[float, float, float, float],
    fragment_xyxy: tuple[float, float, float, float],
    rescue_sides: tuple[bool, bool, bool, bool],
    min_gain: float = RESCUE_MIN_GAIN,
    contain_tol: float = RESCUE_CONTAIN_TOL,
) -> bool:
    """判定重检候选框是否为残框的完整版（三闸门，缺一不可）。

    1. 候选框**完整包含**残框（容差 *contain_tol*）——排除缺口区相邻真实目标
       （其框与残框有重叠但不可能包含大残框）；
    2. 候选框向**每个缺口方向延伸 ≥ min_gain**——"真的看到了 tile 外像素"的
       证据；假触边（完整目标贴 tile 边）时重检框≈残框、无延伸 → 拒绝；
    3. 同类别由 ``_select_rescue_candidate`` 保证。

    Args:
        candidate_xyxy: 候选框 (x0, y0, x1, y1)（全图坐标）。
        fragment_xyxy: 残框 (x0, y0, x1, y1)。
        rescue_sides: 残框触边方向 (left, right, top, bottom)。
        min_gain: 向缺口方向的最小延伸量。
        contain_tol: 包含判定容差。

    Returns:
        是否接受该候选为完整框。
    """
    c0, c1, c2, c3 = candidate_xyxy
    f0, f1, f2, f3 = fragment_xyxy
    contains = c0 <= f0 + contain_tol and c2 >= f2 - contain_tol and c1 <= f1 + contain_tol and c3 >= f3 - contain_tol
    if not contains:
        return False
    left, right, top, bottom = rescue_sides
    return (
        (not right or c2 >= f2 + min_gain)
        and (not left or c0 <= f0 - min_gain)
        and (not bottom or c3 >= f3 + min_gain)
        and (not top or c1 <= f1 - min_gain)
    )


def _select_rescue_candidate(
    candidates_xyxy: list[tuple[float, float, float, float]],
    candidates_labels: list[int],
    candidates_scores: list[float],
    fragment_xyxy: tuple[float, float, float, float],
    fragment_label: int,
    rescue_sides: tuple[bool, bool, bool, bool],
) -> tuple[tuple[float, float, float, float], int, float] | None:
    """从重检块候选框中选出残框的完整版。

    过滤条件：同类别 + ``_rescue_accept`` 通过；多候选取**分数最高**者
    （平局取面积最大）；无合格候选返回 ``None``（调用方保留残框兜底）。

    Args:
        candidates_xyxy: 重检块内的候选框（全图坐标）。
        candidates_labels: 候选框类别 id。
        candidates_scores: 候选框置信度。
        fragment_xyxy: 残框 (x0, y0, x1, y1)。
        fragment_label: 残框类别 id。
        rescue_sides: 残框触边方向。

    Returns:
        ``(box, label, score)`` 或 ``None``。
    """
    best: tuple[tuple[float, float, float, float], int, float] | None = None
    best_area = 0.0
    for box, label, score in zip(candidates_xyxy, candidates_labels, candidates_scores):
        if label != fragment_label:
            continue
        if not _rescue_accept(box, fragment_xyxy, rescue_sides):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or score > best[2] or (score == best[2] and area > best_area):
            best = (box, label, score)
            best_area = area
    return best


@dataclass
class _Fragment:
    """触边残框记录（center_rescue 定向重裁的输入）。

    Attributes:
        box: 残框 (4,) 全图坐标张量（center 过滤前的原始框）。
        label: 残框类别 id（标量张量）。
        score: 残框置信度（标量张量）。
        part_idx: 所属 tile 在 ``per_image[stem]`` 中的 part 位置。
        row_in_part: 该框在 part 内（center 过滤后）的行号；``-1`` = 未被保留。
        sides: 触边方向 (left, right, top, bottom)。
        retained: center 过滤后是否仍保留在 ``per_image`` 中。
    """

    box: torch.Tensor
    label: torch.Tensor
    score: torch.Tensor
    part_idx: int
    row_in_part: int
    sides: tuple[bool, bool, bool, bool]
    retained: bool


class _TileDataset(Dataset):
    """大图切块数据集：worker 内解码大图并切片返回 tile 张量。

    每个 worker 持有一个大小为 1 的图片缓存：tile 按图像连续排列（行优先），
    同一张图的连续 tile 只解码一次，切换到新图像时才重新解码（大图单张解码
    耗时数秒，缓存把单张 196 块的解码次数从 196 次降到 1 次）。tile 为
    1024² 量级（约 3MB），worker 内存峰值 = 1 张解码大图（~400MB）。
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size_map: Mapping[str, tuple[int, int]],
        resolution: int,
        overlap: int,
    ) -> None:
        """构建 tile 样本列表（按图像行优先、图内行优先排列）。

        Args:
            image_paths: 大图路径列表。
            image_size_map: ``{stem: (width, height)}`` 尺寸映射。
            resolution: 模型输入分辨率（= tile 边长）。
            overlap: 滑窗重叠像素数。
        """
        self._items: list[tuple[str, Path, int, int, int, int, tuple[int, int, int, int]]] = []
        self._resolution = resolution
        for image_path in image_paths:
            width, height = image_size_map[image_path.stem]
            for x0, y0 in tile_origins((width, height), resolution, overlap):
                tile_w = min(x0 + resolution, width) - x0
                tile_h = min(y0 + resolution, height) - y0
                # 核心区在主进程计算一次（center 策略的中心归属过滤用），随 pickle 进 worker
                core = tile_core_bounds((width, height), (x0, y0), resolution, overlap)
                self._items.append((image_path.stem, image_path, x0, y0, tile_w, tile_h, core))
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        """返回 tile 总数。"""
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, int, int, int, int, tuple[int, int, int, int], torch.Tensor]:
        """解码并切片第 *index* 个 tile。

        Returns:
            ``(stem, x0, y0, tile_w, tile_h, core, tile_tensor)``：core 为全图
            坐标核心区 ``(x_lo, y_lo, x_hi, y_hi)``；tile_tensor 为
            ``(C, tile_h, tile_w)`` uint8（零拷贝的连续视图）。
        """
        stem, image_path, x0, y0, tile_w, tile_h, core = self._items[index]
        image = self._cache.get(stem)
        if image is None:
            # 缓存只保留当前一张大图（tile 按图连续，换图才重新解码）
            self._cache.clear()
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {image_path}")
            self._cache[stem] = image
        tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
        # 短轴不足 resolution 的块用边缘复制 pad 到 R×R：worker 混批时保证
        # batch 内所有 tile 同尺寸可 stack；内容尺寸（tile_w/tile_h）仍传给
        # target_sizes 做坐标映射，pad 区域不参与有效预测
        if tile_h != self._resolution or tile_w != self._resolution:
            tile = np.pad(
                tile,
                ((0, self._resolution - tile_h), (0, self._resolution - tile_w), (0, 0)),
                mode="edge",
            )
        # ascontiguousarray：numpy 切片是视图，转 torch 需要连续内存（3MB 拷贝可忽略）
        tile = np.ascontiguousarray(tile)
        return stem, x0, y0, tile_w, tile_h, core, torch.from_numpy(tile).permute(2, 0, 1)


def _tile_collate(
    batch: list[tuple[str, int, int, int, int, tuple[int, int, int, int], torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 tile 样本批化为 ``(stems, origins, sizes, cores, tensors)``。

    Args:
        batch: ``_TileDataset.__getitem__`` 的样本列表。

    Returns:
        ``stems`` 为图像名列表；``origins`` 为 ``(N, 2)`` tile 原点 ``(x0, y0)``；
        ``sizes`` 为 ``(N, 2)`` 内容尺寸 ``(tile_w, tile_h)``；``cores`` 为
        ``(N, 4)`` 全图坐标核心区 ``(x_lo, y_lo, x_hi, y_hi)``（center 策略用）；
        ``tensors`` 为 ``(N, 3, R, R)`` uint8 张量。
    """
    stems = [item[0] for item in batch]
    origins = torch.tensor([[item[1], item[2]] for item in batch], dtype=torch.int64)
    sizes = torch.tensor([[item[3], item[4]] for item in batch], dtype=torch.int64)
    cores = torch.tensor([item[5] for item in batch], dtype=torch.int64)
    tensors = torch.stack([item[6] for item in batch])
    return stems, origins, sizes, cores, tensors


def tile_predict_records(
    model: RFDETR,
    image_paths: list[Path],
    image_size_map: Mapping[str, tuple[int, int]],
    device: str,
    resolution: int,
    overlap: int,
    nms_iou: float,
    conf_threshold: float,
    batch_size: int,
    num_workers: int,
    *,
    class_conf_thresholds: Mapping[int, float] | None = None,
    la_bias: torch.Tensor | None = None,
    la_bias_k: float = 1.0,
    prefetch_factor: int = 3,
    use_fp16: bool = False,
    gpu_util_sample_interval: float = 0.5,
    tile_strategy: str = "nms",
) -> tuple[list[BoxRecord], float, float | None, int]:
    """大图滑窗切分推理：worker 预取切块，tile 混合批量前向，按策略合并。

    与整图路径（``predict_batched_to_records``）同构的多进程预取流水线：
    ``num_workers`` 个 worker 并行完成大图解码与切块（每 worker 缓存当前一张
    大图，内存峰值约 400MB/worker），主线程把来自**不同大图**的 tile 混合成
    ``batch_size`` 的 GPU 批量持续前向——解码与推理重叠，GPU 不空转。预测框
    经置信度阈值过滤 → 偏移到全图坐标（GPU 上完成）→ 逐张按策略合并：

    - ``"nms"``（里程碑 1 基线）：全部保留 + 按类别 NMS 去重；
    - ``"center"``（里程碑 2）：只保留预测框中心落在块核心区（core/halo）内的
      框（重复预测在源头消除），再用 ``merge_center_duplicates`` 跨 tile 极严格
      安全合并兜预测噪声。

    与整图路径的数值一致性：tile 预处理走完全相同的 ``F.resize(antialias=False)
    + F.normalize(means, stds)`` 代码路径，坐标经 ``postprocess(target_sizes=块
    内容尺寸)`` 映射回块坐标再平移。

    Args:
        model: 已加载的 RFDETR 实例（内部同整图路径：直连 ``model.model.model``）。
        image_paths: 大图路径列表（``split_image_paths`` 的分组结果）。
        image_size_map: ``{stem: (width, height)}`` 尺寸映射。
        device: 推理设备（如 ``"cuda:0"``）。
        resolution: 模型输入分辨率（= tile 边长）。
        overlap: 滑窗重叠像素数（须满足 ``0 <= overlap < resolution``）。
        nms_iou: ``"nms"`` 策略下合并后按类别 NMS 的 IoU 阈值。
        conf_threshold: 全局置信度阈值（未在 ``class_conf_thresholds`` 中列出的
            类别回退到此值）。
        batch_size: tile 批量大小（GPU 单次前向的 tile 数）。
        num_workers: 大图解码/切块 worker 进程数（0 = 主进程内串行）。
        class_conf_thresholds: 逐类置信度阈值表；``None``/空 → 全部用全局阈值。
        la_bias: 已构建好的 logit bias 张量（调用方用 ``LaBiasCfg.build_bias_tensor``
            构建）；``None`` 不生效。
        la_bias_k: LA bias 扣减系数。
        prefetch_factor: 每个 worker 在内存中预取的 tile 批数。
        use_fp16: FP16 推理。
        gpu_util_sample_interval: 后台采样 GPU 利用率的时间间隔（秒）。
        tile_strategy: 大图合并策略（``"nms"`` / ``"center"`` / ``"center_rescue"``，
            见 ``TILE_STRATEGIES``）；``"center_rescue"`` = center + 触边残框
            定向重裁（救回被滑窗切开的较大目标）。

    Returns:
        ``(pred_records, steady_throughput, gpu_util, timed_images)``，与整图路径
        返回形态一致：``pred_records`` 为 BoxRecord 列表（全图坐标）；
        ``steady_throughput`` 为稳态吞吐（张大图/秒）；``gpu_util`` 为 GPU 平均
        利用率（%），采样失败时为 ``None``；``timed_images`` 为参与计时的图像数。
    """
    _check_tile_strategy(tile_strategy)
    model.model.model = model.model.model.to(device)
    model.model.model.eval()
    if use_fp16:
        model.model.model = model.model.model.half()
    model_dtype = next(model.model.model.parameters()).dtype
    means: list[float] = model.means
    stds: list[float] = model.stds

    # 预热前向：触发 CUDA 内核编译与 cuDNN autotune（与整图路径同构）
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

    # worker 预取流水线：worker 并行解码大图并切片，主线程持续取 tile 批量
    dataset = _TileDataset(image_paths, image_size_map, resolution, overlap)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=_tile_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    pred_records: list[BoxRecord] = []
    # 每张大图累计的 (全图坐标框, 类别, 分数, tile 原点) 张量列表，
    # 循环结束后逐张按策略合并（tile 原点供 center 策略的安全合并判断跨 tile）
    per_image: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    # center_rescue：触边残框记录（center 过滤前收集，供重裁阶段使用）
    fragments: dict[str, list[_Fragment]] = {}
    stem_to_path = {image_path.stem: image_path for image_path in image_paths}
    total_tiles = len(dataset)
    tiles_done = 0
    bench_start = time.perf_counter()

    with torch.inference_mode():
        for stems, origins, sizes, cores, tensors in loader:
            # 预处理与整图路径逐位一致：uint8(C,H,W) → GPU → 除 255 → 缩放到
            # (resolution, resolution)（整块时恒等；短轴不足的块会被拉伸，坐标
            # 由 target_sizes 按内容尺寸映射回来，语义与整图路径一致）
            gpu_images = [
                F.resize(
                    tensor.to(device, non_blocking=True).to(model_dtype).div_(255.0),
                    (resolution, resolution),
                    antialias=False,
                )
                for tensor in tensors
            ]
            batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)
            predictions = model.model.model(batch_tensor)
            if la_bias is not None:
                adjusted_logits = predictions["pred_logits"] - la_bias_k * la_bias.to(
                    dtype=predictions["pred_logits"].dtype
                )
                predictions = {**predictions, "pred_logits": adjusted_logits}
            # target_sizes 按每块内容尺寸 (高, 宽)
            target_sizes = torch.stack([sizes[:, 1], sizes[:, 0]], dim=1).to(device)
            results = model.model.postprocess(predictions, target_sizes=target_sizes)
            offsets = torch.cat([origins.to(device), origins.to(device)], dim=1)
            for i, (stem, (boxes, labels, scores)) in enumerate(
                zip(
                    stems,
                    filter_postprocess_results(results, conf_threshold, class_conf_thresholds),
                )
            ):
                # 偏移到全图坐标（GPU 上完成，合并后再统一转 CPU）
                offset_boxes = boxes + offsets[i]
                # center_rescue：center 过滤**之前**收集触边残框（过滤后才知
                # retained）——可见部分过小（中心落 halo）的残框会被 center
                # 丢弃，pre-center 收集可一并救回
                pending_sides: list[tuple[int, tuple[bool, bool, bool, bool]]] = []
                if tile_strategy == "center_rescue":
                    image_w, image_h = image_size_map[stem]
                    tx0, ty0 = origins[i].tolist()
                    tw, th = sizes[i].tolist()
                    for r in range(offset_boxes.shape[0]):
                        sides = _fragment_rescue_sides(
                            tuple(offset_boxes[r].float().tolist()),
                            (tx0, ty0),
                            (tw, th),
                            (image_w, image_h),
                        )
                        if any(sides):
                            pending_sides.append((r, sides))
                # center 过滤会重新绑定 labels/scores，残框记录需引用过滤前的原始行
                raw_labels, raw_scores = labels, scores
                keep = None
                if tile_strategy in ("center", "center_rescue"):
                    # 中心归属：只保留预测框中心落在该块核心区（core/halo）内的框。
                    # 核心区恰好铺满全图（无缝隙），每个目标的中心至多落在一个
                    # 核心区内 → 跨 tile 重复预测在源头消除（两策略逐位一致）
                    cx = (offset_boxes[:, 0] + offset_boxes[:, 2]) * 0.5
                    cy = (offset_boxes[:, 1] + offset_boxes[:, 3]) * 0.5
                    core = cores[i].to(device)
                    keep = (cx >= core[0]) & (cx < core[2]) & (cy >= core[1]) & (cy < core[3])
                    offset_boxes, labels, scores = (
                        offset_boxes[keep],
                        labels[keep],
                        scores[keep],
                    )
                # tile 标签按该块框数展开成 (N_i, 2)（移到 GPU，与框同设备）：
                # 合并时 cat 后与 boxes 行对齐
                per_image.setdefault(stem, []).append(
                    (
                        offset_boxes,
                        labels,
                        scores,
                        origins[i].expand(offset_boxes.shape[0], -1).to(device),
                    )
                )
                if tile_strategy == "center_rescue" and pending_sides:
                    part_idx = len(per_image[stem]) - 1
                    frag_list = fragments.setdefault(stem, [])
                    for r, sides in pending_sides:
                        # 注意：labels/scores 在 center 过滤后已被索引，残框记录
                        # 需引用过滤前的原始行——offset_boxes/labels/scores 在
                        # 过滤分支内被重新绑定，此处用过滤前的引用
                        retained = bool(keep[r]) if keep is not None else False
                        frag_list.append(
                            _Fragment(
                                box=boxes[r] + offsets[i],
                                label=raw_labels[r],
                                score=raw_scores[r],
                                part_idx=part_idx,
                                row_in_part=int(keep[:r].sum()) if retained else -1,
                                sides=sides,
                                retained=retained,
                            )
                        )
            tiles_done += len(stems)
            print_progress(tiles_done, total_tiles, bench_start, device)

    gpu_monitor.stop()
    torch.cuda.synchronize()
    print()

    # ── center_rescue：触边残框定向重裁 ──────────────────────────────
    # 被切目标的完整框在**任何原有 tile 里都不存在**（overlap < D 时邻居块
    # 同样只看到残片），只能以残框内侧边为基准重裁重检。rescue_results 为
    # stem → [None 或 (box, label, score)]（按残框索引），None = 重检失败。
    # 重检前向必须在 inference_mode 内：块外前向会构建 autograd graph，
    # 激活全量驻留显存（主循环已释放，重检阶段不包会 OOM）。
    rescue_results: dict[str, list[tuple[tuple[float, float, float, float], int, float] | None]] = {}
    if tile_strategy == "center_rescue" and fragments:
        # 阶段 B：逐图解码 + 切重检块（仅解码有残框的图，一次切出全部重检块）。
        # 只重裁 retained 的残框（中心在 core 的贴边框才是真残框：中心在 halo
        # 的贴边框多为小目标预测误差——真被切目标的中心必落在相邻 tile 的
        # core 内，由那块触发重裁，无需在此处理）
        crop_items: list[tuple[str, int, int, int, int, int, torch.Tensor]] = []
        for stem, frags in fragments.items():
            image = cv2.cvtColor(cv2.imread(str(stem_to_path[stem])), cv2.COLOR_BGR2RGB)
            for fi, frag in enumerate(frags):
                if not frag.retained:
                    continue
                x0, y0, w, h = _rescue_crop_bounds(
                    tuple(frag.box.float().tolist()),
                    frag.sides,
                    image_size_map[stem],
                    resolution,
                )
                crop = image[y0 : y0 + h, x0 : x0 + w]
                # 短轴不足 R 时与主循环相同的 np.pad(edge) 补到 R×R
                if h != resolution or w != resolution:
                    crop = np.pad(
                        crop,
                        ((0, resolution - h), (0, resolution - w), (0, 0)),
                        mode="edge",
                    )
                crop_items.append(
                    (
                        stem,
                        fi,
                        x0,
                        y0,
                        w,
                        h,
                        torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1),
                    )
                )
            del image
        # 阶段 C：批量前向（预处理与主循环逐位一致；必须在 inference_mode 内，
        # 块外前向会构建 autograd graph 导致激活驻留显存 OOM）
        rescue_results = {stem: [None] * len(frags) for stem, frags in fragments.items()}
        with torch.inference_mode():
            for start in range(0, len(crop_items), batch_size):
                batch = crop_items[start : start + batch_size]
                gpu_images = [
                    F.resize(
                        tensor.to(device, non_blocking=True).to(model_dtype).div_(255.0),
                        (resolution, resolution),
                        antialias=False,
                    )
                    for _, _, _, _, _, _, tensor in batch
                ]
                batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)
                predictions = model.model.model(batch_tensor)
                if la_bias is not None:
                    adjusted_logits = predictions["pred_logits"] - la_bias_k * la_bias.to(
                        dtype=predictions["pred_logits"].dtype
                    )
                    predictions = {**predictions, "pred_logits": adjusted_logits}
                target_sizes = torch.tensor([[h, w] for _, _, _, _, w, h, _ in batch], device=device)
                results = model.model.postprocess(predictions, target_sizes=target_sizes)
                for k, (stem, fi, cx0, cy0, _, _, _) in enumerate(batch):
                    # 阶段 D：候选选择（同类别 + 完整包含残框 + 向缺口方向延伸）
                    cand_boxes, cand_labels, cand_scores = filter_postprocess_results(
                        [results[k]], conf_threshold, class_conf_thresholds
                    )[0]
                    offset = torch.tensor([cx0, cy0, cx0, cy0], device=cand_boxes.device)
                    cand_full = cand_boxes + offset
                    frag = fragments[stem][fi]
                    rescue_results[stem][fi] = _select_rescue_candidate(
                        [tuple(b.tolist()) for b in cand_full],
                        [int(l) for l in cand_labels.tolist()],
                        [float(s) for s in cand_scores.tolist()],
                        tuple(frag.box.float().tolist()),
                        int(frag.label),
                        frag.sides,
                    )

    # 逐张合并所有 tile 的框 → 按策略去重 → 转 BoxRecord
    # center_rescue：被替换的残框行先行移除，重检完整框带唯一哨兵 origin 追加，
    # 与 tile 框统一过一次安全合并（rescue-rescue 去重、rescue-tile 由面积比守卫）
    sentinel = RESCUE_SENTINEL_BASE
    for stem, parts in per_image.items():
        if tile_strategy == "center_rescue" and stem in fragments:
            # 重检成功的残框行跳过集合（按 part 内行号）
            skip_rows: dict[int, set[int]] = {}
            boxes_list: list[torch.Tensor] = []
            labels_list: list[torch.Tensor] = []
            scores_list: list[torch.Tensor] = []
            tiles_list: list[torch.Tensor] = []
            for fi, frag in enumerate(fragments[stem]):
                if not frag.retained:
                    continue  # 非真残框（预测误差贴边），已由 center 过滤，不处理
                result = rescue_results[stem][fi]
                if result is not None:
                    # 重检成功：残框行被替换为完整框
                    skip_rows.setdefault(frag.part_idx, set()).add(frag.row_in_part)
                    box, label, score = result
                    boxes_list.append(torch.tensor([box], device=device))
                    labels_list.append(torch.tensor([label], device=device))
                    scores_list.append(torch.tensor([score], device=device))
                    tiles_list.append(torch.tensor([[sentinel, sentinel]], device=device))
                    sentinel += 1
                # 重检失败：残框保留在 parts 中（不跳过），部分召回兜底
            for pi, part in enumerate(parts):
                skip = skip_rows.get(pi, set())
                for row in range(part[0].shape[0]):
                    if row in skip:
                        continue
                    boxes_list.append(part[0][row : row + 1])
                    labels_list.append(part[1][row : row + 1])
                    scores_list.append(part[2][row : row + 1])
                    tiles_list.append(part[3][row : row + 1])
            boxes = torch.cat(boxes_list)
            labels = torch.cat(labels_list)
            scores = torch.cat(scores_list)
            # part 的 tile 标签来自 collate（CPU），重检框的哨兵标签在 GPU，
            # 统一移到 GPU 再合并
            tile_ids = torch.cat(tiles_list).to(device)
            keep = merge_center_duplicates(boxes, labels, scores, tile_ids)
        else:
            boxes = torch.cat([part[0] for part in parts])
            labels = torch.cat([part[1] for part in parts])
            scores = torch.cat([part[2] for part in parts])
            if tile_strategy == "nms":
                keep = apply_nms(boxes, labels, scores, nms_iou)
            else:  # center：跨 tile 极严格安全合并（同 tile 内相邻真实目标永不合并）
                tile_ids = torch.cat([part[3] for part in parts]).to(boxes.device)
                keep = merge_center_duplicates(boxes, labels, scores, tile_ids)
        for xyxy, class_id, score in zip(
            boxes[keep].cpu().numpy(),
            labels[keep].cpu().numpy(),
            scores[keep].cpu().numpy(),
        ):
            pred_records.append(
                BoxRecord(
                    image_id=stem,
                    class_id=int(class_id),
                    xyxy=tuple(float(v) for v in xyxy),
                    score=float(score),
                )
            )

    # ── 重裁统计 ────────────────────────────────────────────────────
    if tile_strategy == "center_rescue" and fragments:
        total_frags = sum(1 for frags in fragments.values() for frag in frags if frag.retained)
        success = sum(
            1
            for stem, frags in fragments.items()
            for fi, frag in enumerate(frags)
            if frag.retained and rescue_results[stem][fi] is not None
        )
        print(
            f"[i] center_rescue: {len(fragments)} 图 / {total_frags} 真残框"
            f"（另有 {sum(len(frags) for frags in fragments.values()) - total_frags} 个非真残框已忽略）→ "
            f"{success} 重裁成功，{total_frags - success} 失败保留残框兜底"
        )

    steady_elapsed = time.perf_counter() - bench_start
    timed_images = len(set(per_image) | set(fragments))
    steady_throughput = timed_images / steady_elapsed if steady_elapsed > 0 else 0.0
    return pred_records, steady_throughput, gpu_monitor.average_utilization(), timed_images
