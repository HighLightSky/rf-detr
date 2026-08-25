# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""发射车候选框的二阶段视觉复核器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_small, resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

_IMAGE_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_FORMAT = "shwx-fsc-two-stage-verifier-v1"


@dataclass(frozen=True)
class FSCVerifierPolicy:
    """发射车二级复核器的固定推理规则。

    二级网络输出 ``非FSC`` 与 ``FSC`` 两个类别。候选的保留由类别 argmax
    决定，不对验证集或测试集搜索分类概率阈值。
    """

    fsc_class_id: int = 24
    candidate_floor: float = 0.05
    context_scale: float = 2.0
    image_size: int = _IMAGE_SIZE
    architecture: str = "resnet18"

    def validate(self) -> None:
        """校验固定候选和裁剪配置。"""
        if self.fsc_class_id < 0:
            raise ValueError("fsc_class_id 必须为非负整数")
        if not 0.0 < self.candidate_floor < 1.0:
            raise ValueError("candidate_floor 必须位于 (0, 1)")
        if self.context_scale <= 0:
            raise ValueError("context_scale 必须为正数")
        if self.image_size <= 0:
            raise ValueError("image_size 必须为正数")
        if self.architecture not in {"mobilenet_v3_small", "resnet18"}:
            raise ValueError("architecture 必须为 mobilenet_v3_small 或 resnet18")

    def to_dict(self) -> dict[str, Any]:
        """返回可序列化配置。"""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FSCVerifierPolicy":
        """由 checkpoint 中的配置恢复策略。"""
        fields = cls.__dataclass_fields__
        policy = cls() if value is None else cls(**{key: value[key] for key in fields if key in value})
        policy.validate()
        return policy


def crop_fsc_context(
    image: Image.Image | np.ndarray,
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    context_scale: float = 2.0,
    output_size: int = _IMAGE_SIZE,
) -> Image.Image:
    """围绕候选框裁剪正方形上下文并缩放到分类器尺寸。"""
    if context_scale <= 0 or output_size <= 0:
        raise ValueError("context_scale 和 output_size 必须为正数")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
            raise ValueError("图像数组必须为 HxW、HxWx1、HxWx3 或 HxWx4")
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[2] == 4:
            array = array[..., :3]
        source = Image.fromarray(np.ascontiguousarray(array.astype(np.uint8)), mode="RGB")
    else:
        source = image.convert("RGB")

    x0, y0, x1, y1 = (float(value) for value in box)
    width, height = source.size
    side = max(abs(x1 - x0), abs(y1 - y0), 1.0) * context_scale
    center_x, center_y = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    left = max(0, min(width - 1, int(np.floor(center_x - side * 0.5))))
    top = max(0, min(height - 1, int(np.floor(center_y - side * 0.5))))
    right = max(left + 1, min(width, int(np.ceil(center_x + side * 0.5))))
    bottom = max(top + 1, min(height, int(np.ceil(center_y + side * 0.5))))
    return source.crop((left, top, right, bottom)).resize((output_size, output_size), Image.Resampling.BILINEAR)


def iou_xyxy(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """计算两个 ``xyxy`` 框的交并比。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def label_fsc_candidate(
    box: tuple[float, float, float, float],
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    *,
    fsc_class_id: int = 24,
    iou_threshold: float = 0.35,
) -> int:
    """按车辆评测口径把一级 FSC 候选标为 FSC 或非 FSC。"""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须位于 (0, 1]")
    return int(
        any(class_id == fsc_class_id and iou_xyxy(box, gt_box) >= iou_threshold for class_id, gt_box in ground_truth)
    )


def build_fsc_expert(pretrained: bool = True, architecture: str = "resnet18") -> nn.Module:
    """构造 ImageNet 初始化的二分类器。"""
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if architecture != "mobilenet_v3_small":
        raise ValueError("architecture 必须为 mobilenet_v3_small 或 resnet18")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model


def crop_transform(training: bool = False) -> Compose:
    """构造面向遥感方向变化的训练或推理变换。"""
    transforms: list[Any] = []
    if training:
        from torchvision.transforms import ColorJitter, RandomChoice, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

        transforms.extend(
            [
                RandomChoice([RandomRotation((0, 0)), RandomRotation((90, 90)), RandomRotation((180, 180)), RandomRotation((270, 270))]),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            ]
        )
    transforms.extend([ToTensor(), Normalize(_MEAN, _STD)])
    return Compose(transforms)


class FSCVerifier(nn.Module):
    """只复核 detector 已预测为 FSC 的候选框。"""

    def __init__(self, policy: FSCVerifierPolicy | None = None, pretrained: bool = False) -> None:
        """初始化二级分类器。"""
        super().__init__()
        self.policy = policy or FSCVerifierPolicy()
        self.policy.validate()
        self.expert = build_fsc_expert(pretrained=pretrained, architecture=self.policy.architecture)
        self.checkpoint_metadata: dict[str, Any] = {}
        self._transform = crop_transform(training=False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "FSCVerifier":
        """加载已训练的二级复核器。"""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != _FORMAT:
            raise ValueError(f"不是 {_FORMAT} checkpoint")
        raw_policy = dict(payload.get("policy") or {})
        # 旧 checkpoint 没有记录架构，通过权重键恢复，保证训练产物可复现加载。
        if "architecture" not in raw_policy:
            keys = tuple(payload["expert"].keys())
            raw_policy["architecture"] = (
                "mobilenet_v3_small" if any(key.startswith("features.") for key in keys) else "resnet18"
            )
        verifier = cls(policy=FSCVerifierPolicy.from_mapping(raw_policy), pretrained=False)
        verifier.expert.load_state_dict(payload["expert"])
        verifier.checkpoint_metadata = dict(payload.get("metadata", {}))
        return verifier.to(device).eval()

    def checkpoint_payload(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """返回可保存的 checkpoint 内容。"""
        return {
            "format": _FORMAT,
            "policy": self.policy.to_dict(),
            "expert": self.expert.state_dict(),
            "metadata": dict(metadata or {}),
        }

    def predict_probabilities(
        self,
        image: Image.Image | np.ndarray,
        boxes: list[tuple[float, float, float, float] | list[float] | np.ndarray],
    ) -> Tensor:
        """返回顺序为 ``非FSC``、``FSC`` 的候选概率。"""
        if not boxes:
            return torch.zeros((0, 2), device=next(self.parameters()).device)
        crops = [
            self._transform(
                crop_fsc_context(
                    image,
                    box,
                    context_scale=self.policy.context_scale,
                    output_size=self.policy.image_size,
                )
            )
            for box in boxes
        ]
        with torch.inference_mode():
            return self.expert(torch.stack(crops).to(next(self.parameters()).device)).softmax(dim=1)

    def refine_image(
        self,
        image: Image.Image | np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        """以二级类别 argmax 过滤一级 FSC 候选，并保留所有其他类别。"""
        if not (len(boxes) == len(scores) == len(class_ids)):
            raise ValueError("boxes、scores 与 class_ids 长度必须一致")
        indices = [index for index, class_id in enumerate(class_ids) if int(class_id) == self.policy.fsc_class_id]
        probabilities = self.predict_probabilities(image, [boxes[index] for index in indices])
        accepted = {
            index
            for row, index in enumerate(indices)
            if int(probabilities[row].argmax().item()) == 1
        }
        keep = np.asarray(
            [int(class_id) != self.policy.fsc_class_id or index in accepted for index, class_id in enumerate(class_ids)],
            dtype=bool,
        )
        rejected = len(indices) - len(accepted)
        return (
            np.asarray(boxes[keep], dtype=np.float32).reshape(-1, 4),
            np.asarray(scores[keep], dtype=np.float32),
            np.asarray(class_ids[keep], dtype=np.int64),
            {
                "routed_fsc": len(indices),
                "kept": int(keep.sum()),
                "rejected_non_fsc": rejected,
                "unchanged": int(keep.sum()) - len(accepted),
            },
        )


class FSCScoreFusion(nn.Module):
    """融合二级视觉概率与一级候选置信度的学习型二分类头。"""

    _FORMAT = "shwx-fsc-score-fusion-v1"

    def __init__(self) -> None:
        """初始化三维特征到 FSC/非FSC 的线性融合头。"""
        super().__init__()
        self.head = nn.Linear(3, 2)
        self.checkpoint_metadata: dict[str, Any] = {}

    @staticmethod
    def features(probabilities: Tensor, scores: Tensor) -> Tensor:
        """将视觉概率和候选分数转成稳定的融合特征。"""
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("probabilities 必须为 [N, 2]")
        clipped = scores.reshape(-1).clamp(1e-4, 1.0 - 1e-4)
        if probabilities.shape[0] != clipped.shape[0]:
            raise ValueError("probabilities 与 scores 的行数必须一致")
        return torch.cat((probabilities, torch.logit(clipped).unsqueeze(1)), dim=1)

    def forward(self, probabilities: Tensor, scores: Tensor) -> Tensor:
        """输出顺序为非FSC、FSC 的融合 logits。"""
        return self.head(self.features(probabilities, scores))

    def predict(self, probabilities: Tensor, scores: Tensor) -> Tensor:
        """按固定 argmax 输出非FSC/FSC 融合类别。"""
        with torch.inference_mode():
            return self(probabilities, scores).argmax(dim=1)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "FSCScoreFusion":
        """加载已训练的学习型融合头。"""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != cls._FORMAT:
            raise ValueError(f"不是 {cls._FORMAT} checkpoint")
        module = cls()
        module.load_state_dict(payload["state_dict"])
        module.checkpoint_metadata = dict(payload.get("metadata", {}))
        return module.to(device).eval()

    def checkpoint_payload(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """返回可保存的融合 checkpoint 内容。"""
        return {"format": self._FORMAT, "state_dict": self.state_dict(), "metadata": dict(metadata or {})}


class FSCDinoHead(nn.Module):
    """对冻结 RF-DETR DINOv2 多尺度特征执行 FSC/非FSC 分类。"""

    _FORMAT = "shwx-fsc-dino-head-v1"

    def __init__(self, feature_dim: int = 1536) -> None:
        """初始化池化特征上的小型分类头。"""
        super().__init__()
        self.feature_dim = feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 2),
        )
        self.checkpoint_metadata: dict[str, Any] = {}

    def forward(self, features: Tensor) -> Tensor:
        """输出顺序为非FSC、FSC 的 logits。"""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features 必须为 [N, {self.feature_dim}]")
        return self.head(features)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "FSCDinoHead":
        """加载 DINOv2 分类头。"""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != cls._FORMAT:
            raise ValueError(f"不是 {cls._FORMAT} checkpoint")
        module = cls(int(payload.get("feature_dim", 1536)))
        module.load_state_dict(payload["state_dict"])
        module.checkpoint_metadata = dict(payload.get("metadata", {}))
        return module.to(device).eval()

    def checkpoint_payload(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """返回 DINOv2 分类头 checkpoint。"""
        return {
            "format": self._FORMAT,
            "feature_dim": self.feature_dim,
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }


def pool_dino_features(outputs: list[Tensor] | tuple[Tensor, ...], pooling: str = "avg") -> Tensor:
    """将 DINOv2 多尺度特征池化为二级分类头输入。"""
    if pooling not in {"avg", "avgmax"}:
        raise ValueError("pooling 必须为 avg 或 avgmax")
    chunks: list[Tensor] = []
    for output in outputs:
        if output.ndim != 4:
            raise ValueError("DINOv2 特征必须为 [N, C, H, W]")
        chunks.append(output.mean(dim=(2, 3)))
        if pooling == "avgmax":
            chunks.append(output.amax(dim=(2, 3)))
    return torch.cat(chunks, dim=1)
