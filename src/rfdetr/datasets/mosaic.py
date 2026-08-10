# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Mosaic 数据增强 Dataset 包装器。

将 4 张随机图片拼接为 1 张训练样本，参考 YOLOv5/v8 的实现， 特别适用于小目标检测和遥感数据集。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from supervision import Detections

logger = get_logger()

# Mosaic 画布填充色（灰色，与 YOLO 惯例一致）
_MOSAIC_FILL_COLOR = (114, 114, 114)


def _deep_copy_coco_ann(ann: dict[str, Any]) -> dict[str, Any]:
    """深拷贝单个 COCO 标注，避免共享可变字段。

    Args:
        ann: COCO 标注字典。

    Returns:
        独立拷贝的标注字典。
    """
    bbox = ann.get("bbox", [0, 0, 0, 0])
    return {
        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "category_id": ann.get("category_id", 0),
        "area": float(ann.get("area", bbox[2] * bbox[3])),
        "iscrowd": ann.get("iscrowd", 0),
    }


def _xyxy_to_coco_bbox(xyxy: np.ndarray) -> list[float]:
    """将 [x1, y1, x2, y2] 转换为 COCO [x, y, w, h] 格式。

    Args:
        xyxy: 形状为 (4,) 的绝对像素坐标数组。

    Returns:
        COCO 格式边界框 [x, y, w, h]。
    """
    return [float(xyxy[0]), float(xyxy[1]), float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])]


def _coco_bbox_to_xyxy(bbox: list[float]) -> np.ndarray:
    """将 COCO [x, y, w, h] 转换为 [x1, y1, x2, y2] 格式。

    Args:
        bbox: COCO 格式边界框。

    Returns:
        形状为 (4,) 的 xyxy 数组。
    """
    return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)


class MosaicDataset(torch.utils.data.Dataset[Any]):
    """Mosaic 数据增强包装器。

    以概率 ``p`` 将 4 张随机图片拼接为 1 张训练样本。支持
    :class:`~rfdetr.datasets.coco.CocoDetection` 和
    :class:`~rfdetr.datasets.yolo.YoloDetection` 两种底层数据集。

    实现参考 YOLOv5 的 mosaic 增强：
    1. 创建 2 倍输出尺寸的画布
    2. 随机选择拼接中心点
    3. 将 4 张图片缩放后放置在 4 个象限
    4. 缩放到目标输出尺寸
    5. 合并所有边界框标注

    Args:
        dataset: 底层数据集（CocoDetection 或 YoloDetection）。
        p: Mosaic 增强触发概率，范围 ``[0, 1]``。
        output_size: 输出图片尺寸 ``(height, width)``。

    Examples:
        >>> from rfdetr.datasets.mosaic import MosaicDataset
        >>> # mosaic_dataset = MosaicDataset(train_dataset, p=0.5, output_size=(640, 640))
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
        p: float = 0.5,
        output_size: tuple[int, int] = (640, 640),
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p 必须在 [0, 1] 范围内，当前值为 {p}")

        self._dataset = dataset
        self.p = p
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self._prepare = dataset.prepare
        self._transforms = dataset._transforms

        # 检测数据集类型
        from rfdetr.datasets.coco import CocoDetection
        from rfdetr.datasets.yolo import YoloDetection

        if isinstance(dataset, CocoDetection):
            self._ds_type = "coco"
        elif isinstance(dataset, YoloDetection):
            self._ds_type = "yolo"
        else:
            logger.warning(
                "MosaicDataset 接收到未知数据集类型 %s，将回退到直通模式。",
                type(dataset).__name__,
            )
            self._ds_type = "unknown"

        self._ds_len = len(dataset)  # type: ignore[arg-type]

    def __len__(self) -> int:
        """返回数据集长度（与原始数据集相同）。"""
        return self._ds_len

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """获取训练样本，以概率 ``p`` 触发 mosaic 增强。

        Args:
            idx: 样本索引。

        Returns:
            ``(image_tensor, target_dict)`` 元组。
        """
        # 不触发 mosaic 或数据集类型未知时走原始路径
        if self._ds_type == "unknown" or torch.rand(()).item() >= self.p:
            return self._dataset[idx]

        # 随机选择 4 个索引（包含当前索引以确保至少有一张相关图片）
        indices = [idx]
        for _ in range(3):
            indices.append(int(torch.randint(self._ds_len, ())))

        # 加载 4 张原始图片和标注（COCO 格式，绝对像素坐标）
        images: list[Image.Image] = []
        all_anns: list[list[dict[str, Any]]] = []
        for sample_idx in indices:
            img, anns = self._load_raw(sample_idx)
            images.append(img)
            all_anns.append(anns)

        # 构建 mosaic
        mosaic_img, merged_anns = self._build_mosaic(images, all_anns)

        # 构建 target 并通过正常流程
        target = self._build_target(idx, merged_anns, mosaic_img.size)
        img, target = self._prepare(mosaic_img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    # ------------------------------------------------------------------
    # 原始数据加载
    # ------------------------------------------------------------------

    def _load_raw(self, idx: int) -> tuple[Image.Image, list[dict[str, Any]]]:
        """加载原始图片和 COCO 格式标注。

        Args:
            idx: 样本索引。

        Returns:
            ``(PIL Image, COCO 标注列表)``。标注中 bbox 为绝对像素坐标 ``[x, y, w, h]``。
        """
        if self._ds_type == "coco":
            return self._load_raw_coco(idx)
        return self._load_raw_yolo(idx)

    def _load_raw_coco(self, idx: int) -> tuple[Image.Image, list[dict[str, Any]]]:
        """从 COCO 数据集直接加载原始图片和标注。"""
        ds = self._dataset
        img, target = ds._load_raw(idx)
        raw_anns = target["annotations"]
        return img, [_deep_copy_coco_ann(a) for a in raw_anns]

    def _load_raw_yolo(self, idx: int) -> tuple[Image.Image, list[dict[str, Any]]]:
        """从 YOLO 数据集加载原始图片并转换为 COCO 格式标注。

        supervision Detections.xyxy 为绝对像素坐标。
        """
        ds = self._dataset
        _, rgb_image, detections = ds.load_raw(idx)
        img = Image.fromarray(rgb_image)

        anns: list[dict[str, Any]] = []
        if len(detections) > 0:
            xyxy = detections.xyxy  # 绝对像素坐标 (N, 4)
            class_ids = detections.class_id
            if class_ids is None:
                class_ids = np.zeros(len(xyxy), dtype=int)
            for i in range(len(xyxy)):
                anns.append(
                    {
                        "bbox": _xyxy_to_coco_bbox(xyxy[i]),
                        "category_id": int(class_ids[i]),
                        "area": float((xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])),
                        "iscrowd": 0,
                    }
                )
        return img, anns

    # ------------------------------------------------------------------
    # Target 构建
    # ------------------------------------------------------------------

    def _build_target(self, idx: int, anns: list[dict[str, Any]], img_size: tuple[int, int]) -> dict[str, Any]:
        """根据数据集类型构建原始 target 字典。

        Args:
            idx: 主样本索引。
            anns: COCO 格式标注列表（绝对像素坐标）。
            img_size: Mosaic 图片尺寸 ``(width, height)``。

        Returns:
            符合底层数据集 ``prepare`` 方法期望格式的 target 字典。
        """
        if self._ds_type == "coco":
            return {"image_id": self._dataset.ids[idx], "annotations": anns}
        # YOLO: 转换回 supervision Detections
        return {"image_id": idx, "detections": self._anns_to_detections(anns, img_size)}

    @staticmethod
    def _anns_to_detections(anns: list[dict[str, Any]], img_size: tuple[int, int]) -> Detections:
        """将 COCO 格式标注列表转换为 supervision Detections。

        Args:
            anns: COCO 格式标注列表。
            img_size: 图片尺寸 ``(width, height)``，未使用但保留以兼容未来扩展。

        Returns:
            supervision Detections 对象，xyxy 为绝对像素坐标。
        """
        from supervision import Detections

        if len(anns) == 0:
            return Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                class_id=np.zeros(0, dtype=int),
            )
        xyxy_arr = np.stack([_coco_bbox_to_xyxy(a["bbox"]) for a in anns])
        class_ids = np.array([a.get("category_id", 0) for a in anns], dtype=int)
        return Detections(xyxy=xyxy_arr, class_id=class_ids)

    # ------------------------------------------------------------------
    # Mosaic 构建核心逻辑
    # ------------------------------------------------------------------

    def _build_mosaic(
        self,
        images: list[Image.Image],
        all_anns: list[list[dict[str, Any]]],
    ) -> tuple[Image.Image, list[dict[str, Any]]]:
        """构建 mosaic 图片并合并 4 张图片的标注。

        步骤：
        1. 创建 2 倍输出尺寸的灰色画布
        2. 随机选择拼接中心点
        3. 每张图片缩放到适合其象限的大小并随机放置在象限内
        4. 合并所有标注（调整坐标到画布坐标系）
        5. 缩放到最终输出尺寸

        Args:
            images: 4 张 PIL 图片。
            all_anns: 每张图片的 COCO 格式标注列表（绝对像素坐标）。

        Returns:
            ``(mosaic PIL Image, 合并后的标注列表)``。
        """
        out_h, out_w = self.output_size
        canvas_w = out_w * 2
        canvas_h = out_h * 2

        # 创建灰色画布
        mosaic = Image.new("RGB", (canvas_w, canvas_h), _MOSAIC_FILL_COLOR)

        # 随机选择中心点（在 0.5x ~ 1.5x 输出尺寸范围内抖动）
        cx = int(torch.randint(int(out_w * 0.5), int(out_w * 1.5) + 1, ()))
        cy = int(torch.randint(int(out_h * 0.5), int(out_h * 1.5) + 1, ()))

        # 4 个象限的放置区域
        placements: list[tuple[int, int, int, int]] = [
            (0, 0, cx, cy),  # 左上
            (cx, 0, canvas_w, cy),  # 右上
            (0, cy, cx, canvas_h),  # 左下
            (cx, cy, canvas_w, canvas_h),  # 右下
        ]

        merged_anns: list[dict[str, Any]] = []

        for img, anns, (x1, y1, x2, y2) in zip(images, all_anns, placements):
            region_w = x2 - x1
            region_h = y2 - y1
            if region_w <= 0 or region_h <= 0:
                continue

            orig_w, orig_h = img.size
            if orig_w <= 0 or orig_h <= 0:
                continue

            # 保持宽高比缩放到适合区域
            scale = min(region_w / orig_w, region_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))

            img_resized = img.resize((new_w, new_h), Image.BILINEAR)

            # 在区域内随机放置
            paste_x = x1 + int(torch.randint(0, max(1, region_w - new_w + 1), ()))
            paste_y = y1 + int(torch.randint(0, max(1, region_h - new_h + 1), ()))

            mosaic.paste(img_resized, (paste_x, paste_y))

            # 调整标注坐标：原始 → 缩放后 → 画布坐标系
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            for ann in anns:
                bbox = ann["bbox"]  # COCO: [x, y, w, h]
                merged_anns.append(
                    {
                        "bbox": [
                            bbox[0] * scale_x + paste_x,
                            bbox[1] * scale_y + paste_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y,
                        ],
                        "category_id": ann.get("category_id", 0),
                        "area": bbox[2] * scale_x * bbox[3] * scale_y,
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )

        # 缩放到最终输出尺寸
        mosaic = mosaic.resize((out_w, out_h), Image.BILINEAR)
        final_scale_x = out_w / canvas_w
        final_scale_y = out_h / canvas_h

        # 调整标注坐标并过滤无效框
        valid_anns: list[dict[str, Any]] = []
        for ann in merged_anns:
            x, y, w, h = ann["bbox"]
            x *= final_scale_x
            y *= final_scale_y
            w *= final_scale_x
            h *= final_scale_y

            # 裁剪到图片范围内
            x = max(0.0, x)
            y = max(0.0, y)
            if x >= out_w or y >= out_h:
                continue
            w = min(w, out_w - x)
            h = min(h, out_h - y)
            if w <= 1.0 or h <= 1.0:
                continue

            ann["bbox"] = [x, y, w, h]
            ann["area"] = w * h
            valid_anns.append(ann)

        return mosaic, valid_anns
