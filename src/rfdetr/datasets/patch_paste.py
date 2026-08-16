# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""正/负样本补丁粘贴增强（copy-paste augmentation）。

``PatchPasteDataset`` 包装训练数据集（仿 ``MosaicDataset``），把离线构建的
补丁池（``build_fsc_patch_pool.py`` 产出）随机粘贴到训练图上：

- **正样本补丁**（如发射车 FSC）：粘贴后把其标注框追加进 ``Detections``，
  扩增稀有类的正样本数量；
- **负样本补丁**（如长得像发射车的卡车）：粘贴后**不**追加标注，让模型在
  "有目标的场景图"里直接学到"这类物体不是目标"。

设计约束：
1. **宿主约束**：只有含 ``target_classes``（默认 [24]=FSC）中任一类的图才
   允许粘贴——补丁不贴进无该类目标的背景图，保持场景分布一致；
2. **零交叠**：粘贴位置与宿主全部 GT 框及已贴补丁零像素交叠（拒绝采样）；
3. 粘贴在**原始分辨率**上进行（resize/crop 之前），下游
   ``RandomSizedBBoxSafeCrop``/resize 自然适配；
4. 只作用于训练侧（由 ``RFDETRDataModule.setup("fit")`` 控制包装位置），
   val/test 不受影响；默认关闭（``patch_paste_enabled=False``）。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# D4 群的 8 个元素（PIL transpose 语义：ROTATE_90 为逆时针）
_D4 = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_tb", "transpose", "transverse")
_PIL_D4 = {
    "identity": None,
    "rot90": Image.ROTATE_90,
    "rot180": Image.ROTATE_180,
    "rot270": Image.ROTATE_270,
    "flip_lr": Image.FLIP_LEFT_RIGHT,
    "flip_tb": Image.FLIP_TOP_BOTTOM,
    "transpose": Image.TRANSPOSE,
    "transverse": Image.TRANSVERSE,
}
# 90° 系变换会交换补丁宽高
_SWAP_DIMS = ("rot90", "rot270", "transpose", "transverse")

# 位置拒绝采样参数（不进配置，模块常量）
_MAX_TRIALS = 16
_REJECT_IOU = 0.0  # 0.0 = 与 GT/已贴补丁任何像素交叠都拒绝


@dataclass(frozen=True)
class Patch:
    """单个补丁（加载进内存后的形态）。

    Attributes:
        kind: ``"positive"``（带标注）或 ``"negative"``（无标注）。
        pil: 解码后的 RGB 图片。
        w: 原生像素宽。
        h: 原生像素高。
        box: 正样本的补丁局部 xyxy 像素框；负样本为 ``None``。
        class_id: 正样本的类别索引。
    """

    kind: Literal["positive", "negative"]
    pil: Image.Image
    w: int
    h: int
    box: tuple[float, float, float, float] | None
    class_id: int | None


def load_patch_pool(manifest_path: Path) -> list[Patch]:
    """加载补丁池（读 manifest.json 并全量解码图片进内存）。

    Args:
        manifest_path: ``manifest.json`` 文件路径（其父目录为补丁池根目录，
            含 ``positive/`` 与 ``negative/`` 子目录）。

    Returns:
        补丁列表。manifest 缺失、损坏或图片缺失时抛错（显式开启的功能
        静默失效会污染实验对比）。

    Raises:
        FileNotFoundError: manifest 或补丁图片缺失时抛出。
        ValueError: manifest 结构非法（缺字段/框越界/类别不一致）时抛出。
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"补丁池 manifest 不存在: {manifest_path}（先运行 build_fsc_patch_pool.py）")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_dir = manifest_path.parent
    patches: list[Patch] = []
    for entry in data.get("patches", []):
        kind = entry.get("kind")
        if kind not in ("positive", "negative"):
            raise ValueError(f"manifest 含未知 kind: {kind!r}")
        file_path = pool_dir / entry["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"补丁图片缺失: {file_path}")
        with Image.open(file_path) as img:
            pil = img.convert("RGB")
        w, h = pil.size
        box = None
        class_id = None
        if kind == "positive":
            raw_box = entry.get("box")
            if raw_box is None or len(raw_box) != 4:
                raise ValueError(f"正样本补丁 {entry['id']} 缺少合法 box 字段")
            box = tuple(float(v) for v in raw_box)
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError(f"正样本补丁 {entry['id']} 的 box 越界: {box} (补丁 {w}x{h})")
            class_id = int(entry.get("class_id", -1))
        patches.append(Patch(kind=kind, pil=pil, w=w, h=h, box=box, class_id=class_id))
    return patches


# ------------------------------------------------------------------------
# D4 变换（框的精确映射）
# ------------------------------------------------------------------------
def _d4_point(x: float, y: float, w: int, h: int, op: str) -> tuple[float, float]:
    """把补丁局部点映射到 D4 变换后的坐标系（PIL transpose 语义）。"""
    if op == "identity":
        return (x, y)
    if op == "rot90":  # 逆时针 90°
        return (y, w - 1 - x)
    if op == "rot180":
        return (w - 1 - x, h - 1 - y)
    if op == "rot270":
        return (h - 1 - y, x)
    if op == "flip_lr":
        return (w - 1 - x, y)
    if op == "flip_tb":
        return (x, h - 1 - y)
    if op == "transpose":
        return (y, x)
    if op == "transverse":
        return (h - 1 - y, w - 1 - x)
    raise ValueError(f"未知 D4 变换: {op}")


def _d4_box(box: tuple[float, float, float, float], w: int, h: int, op: str) -> tuple[float, float, float, float]:
    """D4 变换下的轴对齐框精确映射（四角映射 + AABB）。"""
    x1, y1, x2, y2 = box
    xs: list[float] = []
    ys: list[float] = []
    for px, py in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        nx, ny = _d4_point(px, py, w, h, op)
        xs.append(nx)
        ys.append(ny)
    return (min(xs), min(ys), max(xs), max(ys))


# ------------------------------------------------------------------------
# 粘贴核心
# ------------------------------------------------------------------------
def _concat_detections(dets, extra_boxes: list[tuple[float, float, float, float]], class_id: int):
    """把正样本框追加进 supervision Detections（numpy 拼接，不依赖 merge）。"""
    from supervision import Detections

    xyxy = dets.xyxy if len(dets) > 0 else np.zeros((0, 4), dtype=np.float32)
    cid = dets.class_id if dets.class_id is not None else np.zeros(len(xyxy), dtype=int)
    new_xyxy = np.concatenate([xyxy, np.asarray(extra_boxes, dtype=np.float32)], axis=0)
    new_cid = np.concatenate([cid, np.full(len(extra_boxes), class_id, dtype=int)], axis=0)
    return Detections(xyxy=new_xyxy, class_id=new_cid)


def _overlaps(box: tuple[float, float, float, float], other: np.ndarray, reject_iou: float) -> bool:
    """框与一批框的最大 IoU 是否超过阈值（0.0 = 任何像素交叠都拒绝）。"""
    if len(other) == 0:
        return False
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = other[:, 0], other[:, 1], other[:, 2], other[:, 3]
    ix = np.minimum(x2, ox2) - np.maximum(x1, ox1)
    iy = np.minimum(y2, oy2) - np.maximum(y1, oy1)
    inter = np.maximum(ix, 0) * np.maximum(iy, 0)
    area_a = max(x2 - x1, 0) * max(y2 - y1, 0)
    area_b = np.maximum(ox2 - ox1, 0) * np.maximum(oy2 - oy1, 0)
    union = area_a + area_b - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return bool((ious > reject_iou).any())


class PatchPasteDataset(Dataset):
    """正/负样本补丁粘贴包装器（仿 MosaicDataset）。

    Args:
        dataset: 底层训练数据集（YOLO 路径；未知类型告警并退化为直通）。
        manifest_path: 补丁池 manifest.json 路径。
        p: 触发概率（还需宿主图含 ``target_classes`` 才真正粘贴）。
        max_patches: 每图最多粘贴补丁数（实际张数在 [1, max] 均匀采样）。
        target_classes: 宿主约束——仅含这些类中任一类的图才允许粘贴。
        neg_ratio: 负样本占单图补丁的比例（0=全正样本，1=全负样本）。
        scale_range: 补丁相对宿主的缩放范围 ``(min, max)``。
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        manifest_path: Path,
        p: float = 0.5,
        max_patches: int = 2,
        target_classes: list[int] | tuple[int, ...] = (24,),
        neg_ratio: float = 0.5,
        scale_range: tuple[float, float] = (0.8, 1.5),
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p 必须在 [0, 1] 范围内，当前值为 {p}")
        if max_patches < 1:
            raise ValueError(f"max_patches 必须 >= 1，当前值为 {max_patches}")
        if not 0.0 <= neg_ratio <= 1.0:
            raise ValueError(f"neg_ratio 必须在 [0, 1] 范围内，当前值为 {neg_ratio}")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range 非法: {scale_range}")

        self._dataset = dataset
        self.p = p
        self.max_patches = int(max_patches)
        self.target_classes = tuple(sorted(int(c) for c in target_classes))
        self.neg_ratio = float(neg_ratio)
        self.scale_range = scale_range
        self._prepare = getattr(dataset, "prepare", None)
        self._transforms = getattr(dataset, "_transforms", None)
        self._ds_len = len(dataset)  # type: ignore[arg-type]

        # 类型探测：鸭子类型（load_raw/prepare 契约，YoloDetection/CocoDetection 均满足）；
        # 未知类型告警退化直通
        if hasattr(dataset, "load_raw") and hasattr(dataset, "prepare"):
            self._supported = True
        else:
            logger.warning(
                "PatchPasteDataset 接收到不支持的数据集类型 %s（缺 load_raw/prepare），"
                "将回退到直通模式。",
                type(dataset).__name__,
            )
            self._supported = False

        self._pool = load_patch_pool(manifest_path) if self._supported else []
        self._pos = [pt for pt in self._pool if pt.kind == "positive"]
        self._neg = [pt for pt in self._pool if pt.kind == "negative"]
        if not self._pos:
            logger.warning("[PatchPaste] 补丁池无正样本，正样本粘贴跳过")
        if not self._neg:
            logger.warning("[PatchPaste] 补丁池无负样本，负样本粘贴跳过")
        if not self._pos and not self._neg:
            raise ValueError(f"补丁池为空: {manifest_path}")

        # 统计（用于确认增强确实生效）
        self._stats = {"triggered": 0, "placement_failed": 0}
        logger.info(
            "[PatchPaste] 补丁池加载: 正 %d / 负 %d，p=%.2f，宿主类 %s，负比例 %.2f",
            len(self._pos),
            len(self._neg),
            self.p,
            self.target_classes,
            self.neg_ratio,
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._ds_len

    def __getattr__(self, name: str) -> Any:
        """委托给底层数据集（保住 root/coco/classes/ids 等对外接口）。"""
        return getattr(self._dataset, name)

    def __getitem__(self, idx: int):
        """获取训练样本，以概率 ``p`` 触发补丁粘贴（宿主须含目标类）。

        Args:
            idx: 样本索引。

        Returns:
            ``(image, target_dict)``——与底层数据集输出格式一致。
        """
        if not self._supported or random.random() >= self.p:
            return self._dataset[idx]
        path, rgb, dets = self._dataset.load_raw(idx)
        img, dets2, pasted = self._apply_paste(Image.fromarray(rgb), dets)
        if not pasted:
            # 未粘贴（宿主无目标类/位置采样失败）→ 走原路径，保留原行为
            return self._dataset[idx]
        target: dict[str, Any] = {"image_id": idx, "detections": dets2}
        if self._prepare is not None:
            img, target = self._prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def load_raw(self, idx: int) -> tuple[str, np.ndarray, Any]:
        """加载（可能已粘贴的）原始样本，供外层 MosaicDataset 组合。

        Returns:
            ``(path, RGB ndarray HWC, Detections)``——签名与底层数据集一致。
        """
        path, rgb, dets = self._dataset.load_raw(idx)
        if not self._supported or random.random() >= self.p:
            return path, rgb, dets
        img, dets2, pasted = self._apply_paste(Image.fromarray(rgb), dets)
        if not pasted:
            return path, rgb, dets
        return path, np.asarray(img), dets2

    # ------------------------------------------------------------------
    # 粘贴核心
    # ------------------------------------------------------------------
    def _apply_paste(self, img: Image.Image, dets: Any) -> tuple[Image.Image, Any, bool]:
        """执行粘贴：宿主约束 → 采样补丁 → 拒绝采样放置 → 正样本追加框。

        Args:
            img: 宿主 PIL 图。
            dets: 宿主 supervision Detections。

        Returns:
            ``(img, dets, pasted)``：``pasted`` 表示是否实际发生了粘贴
            （负样本-only 粘贴时框不变但图已改，调用方必须据此判断）。
        """
        cls = dets.class_id
        if cls is None or not np.isin(cls, self.target_classes).any():
            return img, dets, False  # 硬约束：宿主必须含目标类
        gt = dets.xyxy if len(dets) > 0 else np.zeros((0, 4), dtype=np.float32)
        pasted_boxes: list[tuple[float, float, float, float]] = []
        any_pasted = False
        n = random.randint(1, self.max_patches)
        for _ in range(n):
            use_negative = random.random() < self.neg_ratio
            pool = self._neg if use_negative else self._pos
            if not pool:
                continue
            patch = random.choice(pool)
            placed = self._sample_placement(patch, img.size, gt, pasted_boxes)
            if placed is None:
                self._stats["placement_failed"] += 1
                continue
            x1, y1, op, s, pw, ph, final_box = placed
            pim = patch.pil.transpose(_PIL_D4[op]) if op != "identity" else patch.pil
            if s != 1.0:
                pim = pim.resize((pw, ph), Image.BILINEAR)
            img.paste(pim, (x1, y1))
            any_pasted = True
            if final_box is not None:
                pasted_boxes.append(final_box)
        if not any_pasted:
            return img, dets, False
        self._stats["triggered"] += 1
        if not pasted_boxes:
            return img, dets, True  # 只贴了负样本：框不变，图已改
        return img, _concat_detections(dets, pasted_boxes, class_id=self.target_classes[0]), True

    def _sample_placement(
        self,
        patch: Patch,
        host_size: tuple[int, int],
        gt_boxes: np.ndarray,
        pasted_boxes: list[tuple[float, float, float, float]],
    ) -> tuple[int, int, str, float, int, int, tuple[float, float, float, float] | None] | None:
        """拒绝采样一个合法的粘贴位置。

        Args:
            patch: 待粘贴补丁。
            host_size: 宿主图尺寸 ``(width, height)``。
            gt_boxes: 宿主全部 GT 框（像素 xyxy）。
            pasted_boxes: 已粘贴正样本框列表。

        Returns:
            ``(x1, y1, op, scale, pw, ph, final_box)``；尝试 ``_MAX_TRIALS``
            次仍失败返回 ``None``（调用方跳过）。
        """
        w, h = host_size
        op = random.choice(_D4)
        op_w, op_h = (patch.h, patch.w) if op in _SWAP_DIMS else (patch.w, patch.h)
        s = random.uniform(*self.scale_range)
        pw = max(1, min(w, round(op_w * s)))
        ph = max(1, min(h, round(op_h * s)))
        box_in_op = _d4_box(patch.box, patch.w, patch.h, op) if patch.box else None
        for _ in range(_MAX_TRIALS):
            x1 = random.randint(0, w - pw)
            y1 = random.randint(0, h - ph)
            cand = (x1, y1, x1 + pw, y1 + ph)
            if _overlaps(cand, gt_boxes, _REJECT_IOU):
                continue
            if pasted_boxes and _overlaps(cand, np.asarray(pasted_boxes, dtype=np.float32), _REJECT_IOU):
                continue
            final_box: tuple[float, float, float, float] | None = None
            if box_in_op is not None:
                fb = (
                    box_in_op[0] * s + x1,
                    box_in_op[1] * s + y1,
                    box_in_op[2] * s + x1,
                    box_in_op[3] * s + y1,
                )
                if fb[2] - fb[0] >= 1 and fb[3] - fb[1] >= 1:
                    final_box = fb
            return x1, y1, op, s, pw, ph, final_box
        return None
