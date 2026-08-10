# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""平方根频率过采样数据集包装器（MMDetection ClassBalancedDataset 数学语义）。

对每个类别 c 计算图像频率 freq(c) = 含类别 c 的图像数 / 总图像数，再按
``repeat_factor(c) = max(1, int(sqrt(t / freq(c))))`` 计算重复倍率；
每张图像取（白名单 ∩ 图内类别）的最大倍率 r(I)，数据集长度 = Σ r(I)。
每图占据连续 ``[starts[i], starts[i+1])`` 段，配合 DataLoader ``shuffle=True``
实现 epoch 内全局均匀混合（无块状分布）。重复样本在每次 ``__getitem__`` 时走
在线增广（Mosaic / 随机变换），因此表现为新的训练样本而非真实重复。

与 MMDetection 原版的两处有意修正（测试锁定）：
1. ``max(1, ·)`` 保证任何图至少 1 份（mmdet 对只含超常见类的图可能算出 r=0 而丢图）；
2. 图像倍率只在白名单 ∩ 图内类别上取 max（白名单外类别不触发过采样）。
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import Any

import torch.utils.data

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class ClassBalancedDataset(torch.utils.data.Dataset[Any]):
    """平方根频率过采样包装数据集。

    数据结构：按每图重复倍率 r(I) 展开（每图连续 r(I) 份），``__getitem__`` 用
    累积和二分定位。无标注图倍率为 1；白名单中无任何匹配图片或所有目标类倍率
    均为 1 时退化为直通模式（与旧 ``RareClassOversampleDataset`` 的宽容哲学一致）。

    Args:
        dataset: 底层数据集，通常为 ``MosaicDataset`` 或裸 ``YoloDetection``。
        threshold: 频率阈值 t（>0）。为 None 时自动推导
            ``t = 4 × max(freq(c) for c in 目标类集)``，保证目标类集内最频繁类
            倍率 ≥ 2（SHWX 上即 HM×3、LQS×2）。
        class_ids: 目标类集白名单（YOLO 标签中的原始 id）。空列表 = 全部类别。
            白名单中无任何图片的类别会被剔除并告警；全部被剔除时退化为直通。
        info_dataset: 用于收集类别频率与属性委托的"信息源"裸数据集。
            传裸数据集（而非 MosaicDataset）可避免触碰 Mosaic 的私有属性，
            并保住 ``coco`` / ``ids`` / ``classes`` / ``root`` 等对外接口。
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
        *,
        threshold: float | None = None,
        class_ids: list[int] | None = None,
        info_dataset: torch.utils.data.Dataset[Any] | None = None,
    ) -> None:
        if threshold is not None:
            threshold = float(threshold)
            # not threshold > 0 同时拒绝 0、负数与 NaN（NaN > 0 为 False）
            if not threshold > 0:
                raise ValueError(f"threshold 必须 > 0，当前值 {threshold!r}")
        self._dataset = dataset
        self._info = info_dataset if info_dataset is not None else dataset
        self._base_len = len(dataset)  # type: ignore[arg-type]
        self._threshold = threshold
        self._whitelist = [int(c) for c in (class_ids or [])]
        self._active = False
        self._length = self._base_len
        self._factors: list[int] = []
        self._starts: list[int] = []

        # ① 单次 O(N) 扫描统计每类图数，并缓存每图类别集合（第二次扫描直接复用）
        per_class_count: dict[int, int] = defaultdict(int)
        image_classes: list[set[int]] = []
        for i in range(self._base_len):
            cats = self._image_class_ids(self._info, i)
            image_classes.append(cats)
            for c in cats:
                per_class_count[c] += 1

        # ② 确定目标类集：白名单剔除无图片的类别；白名单为空时取全部有标注的类
        target_classes = self._resolve_target_classes(per_class_count)
        if not target_classes:
            self._degrade("目标类集为空（白名单类在数据集中无图片，或数据集无任何标注）")
            return

        # ③ 频率、阈值与每类倍率（int 向零截断；max(1,·) 保证不丢图）
        freq = {c: per_class_count[c] / self._base_len for c in target_classes}
        t = threshold if threshold is not None else 4.0 * max(freq.values())
        repeat = {c: max(1, int(math.sqrt(t / freq[c]))) for c in target_classes}
        if all(r == 1 for r in repeat.values()):
            self._degrade(f"阈值 t={t:.4g} 下无任何目标类获得 >1 倍率")
            return

        # ④ 每图倍率与累积和（一次构建；__getitem__ 二分定位）
        target_set = set(target_classes)
        factors = [max((repeat[c] for c in (image_classes[i] & target_set)), default=1) for i in range(self._base_len)]
        starts = [0]
        for f in factors:
            starts.append(starts[-1] + f)

        self._factors = factors
        self._starts = starts
        self._length = starts[-1]
        self._active = True
        logger.info(
            "平方根频率过采样: t=%.4g, 每类倍率 %s, 长度 %d → %d。",
            t,
            {c: repeat[c] for c in target_classes},
            self._base_len,
            self._length,
        )

    def _resolve_target_classes(self, per_class_count: dict[int, int]) -> list[int]:
        """确定目标类集：白名单剔除无图片的类别（并告警）；空白名单取全部有标注类。

        Args:
            per_class_count: 每类图像数统计。

        Returns:
            目标类 id 列表；为空表示应退化为直通。
        """
        if self._whitelist:
            target_classes = [c for c in self._whitelist if per_class_count.get(c, 0) > 0]
            missing = sorted(set(self._whitelist) - set(target_classes))
            if missing:
                logger.warning("平方根频率过采样: 白名单类别 %s 在数据集中无任何图片，已忽略。", missing)
            return target_classes
        return sorted(per_class_count)

    def _image_class_ids(self, info: torch.utils.data.Dataset[Any], index: int) -> set[int]:
        """获取单张图片的类别 id 集合（只读元数据，不解码像素）。

        Args:
            info: 信息源数据集。
            index: 图片索引。

        Returns:
            该图包含的全部类别 id 集合（无标注图为空集）。

        Raises:
            TypeError: 数据集不暴露任何可用的类别查询接口时抛出。
        """
        # YOLO 主路径：sv_dataset.get_image_info(i).class_id 为该图全部实例的类别数组
        sv = getattr(info, "sv_dataset", None)
        if sv is not None and hasattr(sv, "get_image_info"):
            return set(sv.get_image_info(index).class_id.tolist())
        # COCO 回退路径：按 image_id 查标注类别，再经 label2cat 映射到标签 id
        coco = getattr(info, "coco", None)
        ids = getattr(info, "ids", None)
        if coco is not None and ids is not None:
            label2cat = getattr(info, "label2cat", None) or getattr(coco, "label2cat", None)
            cat_ids = {a["category_id"] for a in coco.loadAnns(coco.getAnnIds(ids[index]))}
            if label2cat:
                return {label for label, cat in label2cat.items() if cat in cat_ids}
            return set(cat_ids)
        raise TypeError(f"无法从 {type(info).__name__} 获取图片类别信息")

    def _degrade(self, reason: str) -> None:
        """退化为直通模式：长度恢复为原始长度，不再过采样。

        Args:
            reason: 退化原因（用于日志）。
        """
        self._active = False
        self._length = self._base_len
        logger.warning("平方根频率过采样: %s，退化为直通模式。", reason)

    def __len__(self) -> int:
        """返回展开后的数据集长度。"""
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """按展开索引取样本：累积和二分定位到底层图片索引。"""
        if not self._active:
            return self._dataset[idx]
        if idx < 0 or idx >= self._length:
            raise IndexError(f"索引 {idx} 越界（长度 {self._length}）")
        image_index = bisect.bisect_right(self._starts, idx) - 1
        return self._dataset[image_index]

    def __getattr__(self, name: str) -> Any:
        """其余属性委托给信息源裸数据集（保住 coco/ids/classes/root 等对外接口）。"""
        return getattr(self._info, name)

    @property
    def ids(self) -> list[int]:
        """按展开索引展开的 ids：长度恒等于 len(self)，防止按 sample_index 取 ids 越界。"""
        if not self._active:
            return getattr(self._info, "ids", list(range(self._base_len)))
        base_ids = getattr(self._info, "ids", None) or list(range(self._base_len))
        return [base_ids[i] for i, r in enumerate(self._factors) for _ in range(r)]
