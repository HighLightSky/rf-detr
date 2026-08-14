# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图拼接缝检测与无缝切块模块（里程碑 4）。

比赛大图 = 小图行式无缝拼接而成（目标框完整落在单个小图内、不跨拼接缝），
因此可以检测拼接缝并**沿缝切割**：图块 = 源小图，无重叠、无目标截断，整块
缩放 1024 推理（与全小图基线路径同构），理论上指标回到全小图基线水平。

缝的两类可观测特征（基于 hstrip 实测）：

- 辐射跳变：相邻源图亮度/色调差，被 JPEG 平滑后较弱（列梯度约 2x 基线）；
- 内容切换：缝两侧来自不同源图，相邻列相关性骤降（实测可到负相关）。

本模块用**统计断点 + 全高/全宽一致性投票**检测缝（无模型）：

1. ``_jump_profile``：行/列间差分矩阵 + 黑底填充区排除掩码；
2. ``_axis_seams``：跳变阈值 = 全图中位数差分 × ``JUMP_THR_MULT``；候选位置
   要求"沿另一轴持续跳变"（一致性 ≥ ``CONSISTENCY_THR``）且平均跳变高于基线
   ——内容边缘（道路/建筑轮廓）只在小段内跳变，一致性低，被自然滤除；
3. 近邻候选（间距 < ``SEAM_MIN_GAP``）合并取最强。

结构约定：先检测**水平缝**（全宽行边界）得到行条带；每条带内再独立检测
**垂直缝**——因为每行内部的小图边界位置互不相同，"全高的垂直线"会切穿
其他行的小图，必须按条带检测（与朴素滑窗网格的本质差异）。

推理图块由 ``seam_tiles`` 组合：条带 × 列区间；每个图块都保留显式
``(x0, y0, x1, y1)`` 边界，不再只传原点，避免 seam 模式下把大于
``resolution`` 的源小图块悄悄裁成 1024×1024 的前缀。图块任一边超过
``MAX_TILE_SIDE``（缝检测漏缝时可能跨两个源图）时，图块内部退回滑窗网格
（``tile_overlap`` 由调用方传入，复用里程碑 2 的 center 合并机制兜底）。
"""

from __future__ import annotations

import cv2
import numpy as np

# ── 缝检测参数（基于 hstrip 100 张真缝 ground truth 标定）─────────────
JUMP_THR_MULT = 2.0  # 跳变阈值 = 全图中位数差分 × 该系数
CONSISTENCY_THR = 0.6  # 一致性投票通过比例（真缝 0.76-1.0，内容边缘簇 ≤0.6）
MEAN_JUMP_THR_MULT = 10.0  # 平均跳变门槛 = 全图中位数差分 × 该系数
# （真缝 ≥13.5 倍，周期性纹理假缝簇仅 8-9 倍，10 倍干净分离）
SEAM_MIN_GAP = 8  # 近邻候选合并间距（像素）
SEAM_MIN_SPACING = 480  # 缝最小间距：任一候选与更强的候选间距小于该值则剔除。
# 真缝间距 ≥ 最小图块边长（源图 700px × 缩放下限 0.8 ≈ 560px），而周期性纹理
# 假缝簇间距仅 20-80px——按强度降序贪心即可把簇内非最强候选全部剔除
BLACK_LEVEL = 10  # 黑底填充判定阈值：任一像素 ≤ 该值即排除出差分统计
MAX_TILE_SIDE = 1200  # 图块任一边超过该值 → 内部展开滑窗网格兜底


def _jump_profile(gray: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """计算指定轴方向的差分剖面（uint8 减法，零拷贝级开销）。

    Args:
        gray: ``(H, W)`` uint8 灰度图。
        axis: ``0`` = 行间差分（水平缝，返回 ``(H-1, W)``）；``1`` = 列间差分
            （垂直缝，返回 ``(H, W-1)``）。

    Returns:
        ``(jumps, valid)``：jumps 为 uint8 差分矩阵；valid 为同形状 bool 掩码，
        排除黑底填充区（两侧任一像素 ≤ ``BLACK_LEVEL`` 的像素对）。
    """
    if axis == 0:
        jumps = np.abs(gray[1:].astype(np.int16) - gray[:-1]).astype(np.uint8)
        valid = (gray[1:] > BLACK_LEVEL) & (gray[:-1] > BLACK_LEVEL)
    else:
        jumps = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1]).astype(np.uint8)
        valid = (gray[:, 1:] > BLACK_LEVEL) & (gray[:, :-1] > BLACK_LEVEL)
    return jumps, valid


def _axis_seams(profile: np.ndarray, valid: np.ndarray, axis: int) -> list[int]:
    """从差分剖面提取一条轴上的缝位置（一致性投票 + 平均跳变门槛 + 近邻合并）。

    缝行/缝列的特征：沿另一轴方向**持续**跳变（一致性高），且平均跳变高于
    基线；内容边缘（道路/建筑轮廓/噪声）只在小段内跳变，一致性低、平均跳变
    小，被自然滤除。**归一化轴与缝方向一致**：水平缝沿行方向投票（每行的
    有效列占比），垂直缝沿列方向投票（每列的带内有效行占比）——两条轴必须
    分别按 ``axis`` 归一化，否则垂直缝的"全高一致性"会被逐行噪声稀释。

    Args:
        profile: ``(N, M)`` uint8 差分矩阵（行间或列间）。
        valid: ``(N, M)`` bool 掩码（黑底填充区排除）。
        axis: ``0`` = 水平缝（缝沿行方向，profile 为 ``(H-1, W)``）；
            ``1`` = 垂直缝（缝沿列方向，profile 为 ``(H, W-1)``）。

    Returns:
        缝位置列表（升序）；无候选返回空列表。位置语义：axis=0 时为行号
        （第 y 行与第 y+1 行之间），axis=1 时为列号（第 x 列与第 x+1 列之间）。
    """
    sum_axis = 1 - axis  # 沿"缝方向"求和：水平缝按行（axis=0 → sum over columns）
    valid_count = valid.sum(axis=sum_axis)
    if valid_count.max() == 0:
        return []
    median_global = float(np.median(profile[valid]))
    if median_global <= 0:
        return []
    thr = median_global * JUMP_THR_MULT
    hits = (profile > thr) & valid
    consist = hits.sum(axis=sum_axis) / np.maximum(valid_count, 1)
    # 平均跳变只统计有效像素（无效行的差分会污染求和）
    mean_jump = (profile * valid).sum(axis=sum_axis) / np.maximum(valid_count, 1)
    candidates = np.where(
        (consist >= CONSISTENCY_THR) & (mean_jump >= median_global * MEAN_JUMP_THR_MULT)
    )[0]
    if len(candidates) == 0:
        return []
    # 近邻合并：间距 < SEAM_MIN_GAP 的候选归为一组，组内取一致性最高者
    groups: list[list[int]] = []
    for c in candidates.tolist():
        if groups and c - groups[-1][-1] < SEAM_MIN_GAP:
            groups[-1].append(c)
        else:
            groups.append([c])
    merged = [max(g, key=lambda c: (float(consist[c]), float(mean_jump[c]))) for g in groups]
    # 缝最小间距过滤：按平均跳变降序贪心，任一候选与已接受候选间距
    # < SEAM_MIN_SPACING 则剔除（周期性纹理假缝簇只保留最强一条）
    seams: list[int] = []
    for c in sorted(merged, key=lambda c: -float(mean_jump[c])):
        if all(abs(c - s) >= SEAM_MIN_SPACING for s in seams):
            seams.append(c)
    return sorted(seams)


def detect_seams(gray: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """检测大图的拼接缝（水平缝 + 每条带内的垂直缝）。

    先检测全宽水平缝（行边界）得到行条带；每条带内独立检测垂直缝
    （列边界）——每行的拼接边界位置互不相同，垂直缝必须按条带检测。

    Args:
        gray: ``(H, W)`` uint8 灰度图（全分辨率，避免缩放引入对齐误差）。

    Returns:
        ``(seam_ys, xs_per_band)``：
        - ``seam_ys``：水平缝行位置（升序），相邻缝之间为一条行条带；
        - ``xs_per_band``：与条带一一对应（条带数 = len(seam_ys) + 1），
          每条带内的垂直缝列位置（升序）。
    """
    height = gray.shape[0]
    prof_h, valid_h = _jump_profile(gray, axis=0)
    seam_ys = _axis_seams(prof_h, valid_h, axis=0)
    bounds = [0, *seam_ys, height]
    xs_per_band: list[list[int]] = []
    for y0, y1 in zip(bounds[:-1], bounds[1:]):
        if y1 - y0 <= 0:
            continue
        prof_v, valid_v = _jump_profile(gray[y0:y1], axis=1)
        xs_per_band.append(_axis_seams(prof_v, valid_v, axis=1))
    return seam_ys, xs_per_band


def seam_tiles(
    image_size: tuple[int, int],
    seam_ys: list[int],
    xs_per_band: list[list[int]],
    resolution: int,
    overlap: int,
    max_tile_side: int = MAX_TILE_SIDE,
) -> list[tuple[int, int, int, int, int]]:
    """组合缝切分图块矩形列表（条带 × 列区间），超限图块展开为滑窗网格。

    Args:
        image_size: 图像 (宽, 高)。
        seam_ys: ``detect_seams`` 返回的水平缝位置。
        xs_per_band: ``detect_seams`` 返回的每条带垂直缝位置。
        resolution: 推理分辨率（= tile 边长，兜底网格用）。
        overlap: 滑窗重叠像素数（仅超限图块的兜底网格使用；缝图块自身无重叠）。
        max_tile_side: 图块任一边超过该值 → 内部展开滑窗网格兜底
            （缝检测漏缝时图块可能跨两个源图，直接缩放会损失小目标）。

    Returns:
        ``(x0, y0, x1, y1, tile_overlap)`` 列表，按 (y, x) 升序排列：
        缝图块 ``tile_overlap=0``（整块直连，保留真实边界），
        兜底网格块 ``tile_overlap=overlap``。
    """
    width, height = image_size
    if len(xs_per_band) != len(seam_ys) + 1:
        raise ValueError("xs_per_band 长度必须等于条带数（len(seam_ys) + 1）")
    bounds = [0, *seam_ys, height]
    out: list[tuple[int, int, int, int, int]] = []
    for band_idx, (y0, y1) in enumerate(zip(bounds[:-1], bounds[1:])):
        xs = [0, *xs_per_band[band_idx], width]
        for x0, x1 in zip(xs[:-1], xs[1:]):
            tw, th = x1 - x0, y1 - y0
            if tw <= 0 or th <= 0:
                continue
            if tw <= max_tile_side and th <= max_tile_side:
                out.append((x0, y0, x1, y1, 0))
            else:
                # 超限兜底：图块内部滑窗网格（惰性 import 避免循环依赖）
                from scripts.tiling import tile_origins

                for ox, oy in tile_origins((tw, th), resolution, overlap):
                    gx0 = x0 + ox
                    gy0 = y0 + oy
                    gx1 = min(gx0 + resolution, x1)
                    gy1 = min(gy0 + resolution, y1)
                    out.append((gx0, gy0, gx1, gy1, overlap))
    return out


def build_seam_origins_map(
    image_paths: list,
    image_size_map: dict,
    resolution: int,
    overlap: int,
    max_tile_side: int = MAX_TILE_SIDE,
) -> tuple[dict[str, list[tuple[int, int, int, int, int]]], dict[str, tuple[int, int]]]:
    """为一批大图构建缝切分原点映射（主进程顺序执行缝检测）。

    Args:
        image_paths: 大图路径列表。
        image_size_map: ``{stem: (width, height)}`` 尺寸映射。
        resolution: 推理分辨率（= tile 边长）。
        overlap: 滑窗重叠像素数（兜底网格用）。
        max_tile_side: 超限图块阈值（见 ``seam_tiles``）。

    Returns:
        ``(origins_map, seam_stats)``：
        - ``origins_map``：``{stem: [(x0, y0, x1, y1, tile_overlap), ...]}``；
        - ``seam_stats``：``{stem: (len(seam_ys), sum(len(xs) for xs in xs_per_band))}``
          （水平缝数, 垂直缝总数），供日志输出。
    """
    origins_map: dict[str, list[tuple[int, int, int, int, int]]] = {}
    seam_stats: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        stem = image_path.stem
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seam_ys, xs_per_band = detect_seams(gray)
        origins_map[stem] = seam_tiles(
            image_size_map[stem],
            seam_ys,
            xs_per_band,
            resolution,
            overlap,
            max_tile_side=max_tile_side,
        )
        seam_stats[stem] = (len(seam_ys), sum(len(xs) for xs in xs_per_band))
    return origins_map, seam_stats
