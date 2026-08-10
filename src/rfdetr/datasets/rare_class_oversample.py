# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""少数类重采样数据集包装器。

将训练集长度扩展 ``oversample_factor`` 倍，扩展段循环映射到包含指定少数类 （如 SHWX 数据集的 HM 航母、LQS 两栖舰）的图片，从而提升极端稀有类在训练
中的参与度。开关默认关闭时，本包装器不会被构造，训练行为与基线完全一致。
"""

from __future__ import annotations

from typing import Any

import torch.utils.data

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class RareClassOversampleDataset(torch.utils.data.Dataset[Any]):
    """少数类重采样包装数据集。

    数据结构：首段 ``[0, base_len)`` 保持原始数据集逐位不变（分布不变），
    扩展段 ``[base_len, len)`` 循环映射到包含任一少数类类别的图片索引。
    重复样本在每次 ``__getitem__`` 时走在线增广（Mosaic / 随机变换），
    因此表现为新的训练样本而非真实重复（与 ``GradAccumAlignedDataset``
    的 padding 索引同一哲学）。

    Args:
        dataset: 底层数据集，通常为 ``MosaicDataset`` 或裸 ``YoloDetection``。
        rare_class_ids: 视为少数类的类别 id 列表（YOLO 标签中的原始 id）。
            为空或底层找不到任何含这些类别的图片时退化为直通。
        oversample_factor: 总长度 = 原长度 × oversample_factor，须 >= 1。
        info_dataset: 用于收集少数类索引与属性委托的"信息源"裸数据集。
            传裸数据集（而非 MosaicDataset）可避免触碰 Mosaic 的私有属性，
            并保住 ``coco`` / ``ids`` / ``classes`` / ``root`` 等对外接口。
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
        rare_class_ids: list[int],
        oversample_factor: int = 2,
        *,
        info_dataset: torch.utils.data.Dataset[Any] | None = None,
    ) -> None:
        factor = int(oversample_factor)
        if factor < 1:
            raise ValueError(f"oversample_factor 必须 >= 1，当前值为 {factor}")
        self._dataset = dataset
        self._info = info_dataset if info_dataset is not None else dataset
        self._factor = factor
        self._base_len = len(dataset)  # type: ignore[arg-type]
        rare_ids = set(int(c) for c in (rare_class_ids or []))
        self._rare_indices = self._collect_rare_indices(self._info, rare_ids)
        if not self._rare_indices:
            logger.warning(
                "少数类重采样: 在数据集中未找到包含类别 %s 的图片，退化为直通模式。",
                sorted(rare_ids),
            )
            self._active = False
            self._length = self._base_len
        else:
            self._active = True
            self._length = self._base_len * self._factor
            logger.info(
                "少数类重采样: 类别 %s 命中 %d 张图片，长度 %d → %d。",
                sorted(rare_ids),
                len(self._rare_indices),
                self._base_len,
                self._length,
            )

    def _collect_rare_indices(self, info: torch.utils.data.Dataset[Any], rare_ids: set[int]) -> list[int]:
        """收集底层数据集中包含任一少数类类别的图片索引（只读元数据，不解码像素）。

        Args:
            info: 信息源数据集（裸数据集）。
            rare_ids: 少数类类别 id 集合。

        Returns:
            命中的图片索引列表；找不到任何可用接口时抛出 TypeError。
        """
        # YOLO 主路径：与本库 _build_coco_api_from_samples 同一稳定接口。
        # _LazyYoloSample.class_id 为该图全部实例的类别数组（yolo.py:442）。
        sv = getattr(info, "sv_dataset", None)
        if sv is not None and hasattr(sv, "get_image_info"):
            return [
                i
                for i in range(len(sv))  # type: ignore[arg-type]
                if set(sv.get_image_info(i).class_id.tolist()) & rare_ids
            ]
        # COCO 回退路径：按 image_id 查标注类别，再经 label2cat 映射到标签 id。
        coco = getattr(info, "coco", None)
        ids = getattr(info, "ids", None)
        if coco is not None and ids is not None:
            label2cat = getattr(info, "label2cat", None) or getattr(coco, "label2cat", None)
            out: list[int] = []
            for i, image_id in enumerate(ids):
                cat_ids = {a["category_id"] for a in coco.loadAnns(coco.getAnnIds(image_id))}
                labels = {l for l, c in (label2cat or {}).items() if c in cat_ids} if label2cat else cat_ids
                if labels & rare_ids:
                    out.append(i)
            return out
        raise TypeError(f"无法从 {type(info).__name__} 获取图片类别信息")

    def __len__(self) -> int:
        """返回扩展后的数据集长度。"""
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """按扩展索引取样本：首段走原分布，扩展段循环映射到少数类图片。"""
        if not self._active or idx < self._base_len:
            return self._dataset[idx]  # type: ignore[index]
        return self._dataset[self._rare_indices[(idx - self._base_len) % len(self._rare_indices)]]  # type: ignore[index]

    def __getattr__(self, name: str) -> Any:
        """其余属性委托给信息源裸数据集（保住 coco/ids/classes/root 等对外接口）。"""
        return getattr(self._info, name)

    @property
    def ids(self) -> list[int]:
        """索引映射版 ids：扩展段映射回少数类底层索引，防止按 sample_index 取 ids 越界。"""
        mapped = [self._rare_indices[i % len(self._rare_indices)] for i in range(self._length - self._base_len)]
        return list(range(self._base_len)) + mapped
