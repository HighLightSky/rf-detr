# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""通用二阶段候选框复核插件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from rfdetr.refinement.fsc_two_stage import FSCVerifier, crop_fsc_context, crop_transform, iou_xyxy
from val.competition_metrics import BoxRecord


def _resolve_checkpoint_path(value: str | Path | None) -> str | Path | None:
    """将配置中的相对权重路径解析到项目根目录。"""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


@dataclass(frozen=True)
class TwoStageConfig:
    """二阶段复核插件配置。"""

    enabled: bool = False
    backend: str = "dinov3"
    checkpoint: str | Path | None = None
    backbone_checkpoint: str | Path | None = None
    class_ids: tuple[int, ...] = (24,)
    candidate_floor: float = 0.05
    candidate_nms_iou: float = 0.5
    context_scale: float = 2.0
    image_size: int = 224
    batch_size: int = 64
    positive_threshold: float = 0.5
    bypass_score: float | None = None
    detector_score_weight: float = 0.0

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "TwoStageConfig | None":
        """从 YAML 配置解析二阶段插件。"""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("two_stage 必须是字典配置")
        if not bool(value.get("enabled", False)):
            return None
        class_ids = value.get("class_ids", [24])
        if not isinstance(class_ids, (list, tuple)):
            raise ValueError("two_stage.class_ids 必须是整数列表")
        try:
            parsed_ids = tuple(int(item) for item in class_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("two_stage.class_ids 必须只包含整数") from exc
        config = cls(
            enabled=True,
            backend=str(value.get("backend", "dinov3")),
            checkpoint=_resolve_checkpoint_path(value.get("checkpoint")),
            backbone_checkpoint=_resolve_checkpoint_path(value.get("backbone_checkpoint")),
            class_ids=parsed_ids,
            candidate_floor=float(value.get("candidate_floor", 0.05)),
            candidate_nms_iou=float(value.get("candidate_nms_iou", 0.5)),
            context_scale=float(value.get("context_scale", 2.0)),
            image_size=int(value.get("image_size", 224)),
            batch_size=int(value.get("batch_size", 64)),
            positive_threshold=float(value.get("positive_threshold", 0.5)),
            bypass_score=(float(value["bypass_score"]) if value.get("bypass_score") is not None else None),
            detector_score_weight=float(value.get("detector_score_weight", 0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验配置范围和必填项。"""
        if self.backend not in {"dinov3", "resnet18", "mobilenet_v3_small"}:
            raise ValueError("two_stage.backend 必须为 dinov3、resnet18 或 mobilenet_v3_small")
        if not self.checkpoint:
            raise ValueError("two_stage.enabled=true 时必须设置 checkpoint")
        if not self.class_ids:
            raise ValueError("two_stage.class_ids 不能为空")
        if any(class_id < 0 for class_id in self.class_ids):
            raise ValueError("two_stage.class_ids 必须为非负整数")
        if not 0.0 <= self.candidate_floor <= 1.0:
            raise ValueError("two_stage.candidate_floor 必须位于 [0, 1]")
        if not 0.0 < self.candidate_nms_iou <= 1.0:
            raise ValueError("two_stage.candidate_nms_iou 必须位于 (0, 1]")
        if not 0.0 <= self.positive_threshold <= 1.0:
            raise ValueError("two_stage.positive_threshold 必须位于 [0, 1]")
        if self.bypass_score is not None and not 0.0 <= self.bypass_score <= 1.0:
            raise ValueError("two_stage.bypass_score 必须位于 [0, 1] 或为 null")
        if not 0.0 <= self.detector_score_weight <= 1.0:
            raise ValueError("two_stage.detector_score_weight 必须位于 [0, 1]")
        if self.context_scale <= 0 or self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("two_stage.context_scale、image_size 和 batch_size 必须为正数")


@dataclass
class TwoStageStats:
    """二阶段过滤统计。"""

    routed: int = 0
    candidate_nms_suppressed: int = 0
    kept: int = 0
    rejected: int = 0
    images: int = 0
    elapsed_seconds: float = 0.0
    per_image_candidates: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 JSON 的统计字典。"""
        return {
            "routed": self.routed,
            "candidate_nms_suppressed": self.candidate_nms_suppressed,
            "kept": self.kept,
            "rejected": self.rejected,
            "images": self.images,
            "elapsed_seconds": self.elapsed_seconds,
            "average_candidates_per_image": self.routed / self.images if self.images else 0.0,
            "per_image_candidates": dict(self.per_image_candidates),
        }


class _DinoV3Head(nn.Module):
    """兼容现有 DINOv3 FSC 头 checkpoint 的小型分类头。"""

    def __init__(self, feature_dim: int) -> None:
        """初始化分类头。"""
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        """输出非 FSC、FSC 两类 logits。"""
        return self.head(features)


def _nms_indices(records: Sequence[BoxRecord], threshold: float) -> list[BoxRecord]:
    """对同一类别候选执行置信度优先 NMS。"""
    kept: list[BoxRecord] = []
    for record in sorted(records, key=lambda item: float(item.score or 0.0), reverse=True):
        if all(iou_xyxy(record.xyxy, chosen.xyxy) <= threshold for chosen in kept):
            kept.append(record)
    return kept


def _load_backbone_state(path: str | Path) -> Mapping[str, Tensor]:
    """读取 PyTorch 或 safetensors 格式的 backbone 权重。"""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("backbone checkpoint 必须是 state dict")
        return payload
    except (RuntimeError, ValueError, pickle.UnpicklingError):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ValueError("无法读取 backbone checkpoint；请安装 safetensors") from exc
        return load_file(str(path), device="cpu")


class TwoStagePlugin:
    """对一级检测候选执行二分类复核并返回过滤后的框。"""

    def __init__(self, config: TwoStageConfig, device: str | torch.device = "cpu") -> None:
        """加载配置指定的复核 backend。"""
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self._transform = crop_transform(training=False)
        self._verifier: FSCVerifier | None = None
        self._backbone: nn.Module | None = None
        self._head: nn.Module | None = None
        if config.backend in {"resnet18", "mobilenet_v3_small"}:
            self._verifier = FSCVerifier.from_checkpoint(config.checkpoint, device=self.device)
            if self._verifier.policy.architecture != config.backend:
                raise ValueError(
                    "二阶段 checkpoint 架构与配置不一致: "
                    f"{self._verifier.policy.architecture} != {config.backend}"
                )
        else:
            self._load_dinov3()

    def _load_dinov3(self) -> None:
        """加载 DINOv3 backbone 和现有二分类头。"""
        payload = torch.load(str(self.config.checkpoint), map_location="cpu", weights_only=False)
        if payload.get("format") != "shwx-fsc-dinov3-head-v1":
            raise ValueError("不是 shwx-fsc-dinov3-head-v1 checkpoint")
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImportError("dinov3 backend 需要安装 timm") from exc
        metadata = dict(payload.get("metadata") or {})
        model_name = metadata.get("model_name", "vit_base_patch16_dinov3.lvd1689m")
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        backbone_state = payload.get("backbone_state_dict")
        if backbone_state is None and self.config.backbone_checkpoint:
            backbone_state = _load_backbone_state(self.config.backbone_checkpoint)
        if backbone_state is None:
            raise ValueError("dinov3 checkpoint 不含 backbone 权重，必须设置 two_stage.backbone_checkpoint")
        backbone.load_state_dict(backbone_state, strict=True)
        head = _DinoV3Head(int(payload["feature_dim"]))
        head.load_state_dict(payload["state_dict"], strict=True)
        self._backbone = backbone.to(self.device).eval()
        self._head = head.to(self.device).eval()
        # 必须与 train_fsc_dinov3_head.py 的训练预处理完全一致。
        self._transform = create_transform(**resolve_model_data_config(backbone), is_training=False)

    def _predict(self, images: list[Image.Image], boxes: list[BoxRecord]) -> np.ndarray:
        """批量预测候选是否为目标类别。"""
        if self._verifier is not None:
            crops = [
                self._transform(crop_fsc_context(image, record.xyxy, self.config.context_scale, self.config.image_size))
                for image, record in zip(images, boxes, strict=True)
            ]
            output: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(crops), self.config.batch_size):
                    batch = torch.stack(crops[start : start + self.config.batch_size]).to(self.device)
                    logits = self._verifier.expert(batch)
                    output.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            return np.concatenate(output) if output else np.zeros((0,), dtype=bool)
        assert self._backbone is not None and self._head is not None
        crops = [self._transform(crop_fsc_context(image, record.xyxy, self.config.context_scale, self.config.image_size)) for image, record in zip(images, boxes, strict=True)]
        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.config.batch_size):
                batch = torch.stack(crops[start : start + self.config.batch_size]).to(self.device)
                features = self._backbone(batch)
                logits = self._head(features)
                output.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(output) if output else np.zeros((0,), dtype=bool)

    def refine_records(
        self,
        records: Sequence[BoxRecord],
        image_sources: Mapping[str, str | Path | Image.Image | np.ndarray],
    ) -> tuple[list[BoxRecord], TwoStageStats]:
        """按图像批量复核目标类别候选。"""
        started = time.perf_counter()
        grouped: dict[str, list[BoxRecord]] = {}
        for record in records:
            if record.class_id in self.config.class_ids and float(record.score or 0.0) >= self.config.candidate_floor:
                grouped.setdefault(record.image_id, []).append(record)

        stats = TwoStageStats(images=len(grouped))
        selected: list[tuple[str, list[BoxRecord]]] = []
        for image_id, group in grouped.items():
            kept: list[BoxRecord] = []
            for class_id in self.config.class_ids:
                class_group = [record for record in group if record.class_id == class_id]
                kept.extend(_nms_indices(class_group, self.config.candidate_nms_iou))
            kept.sort(key=lambda item: float(item.score or 0.0), reverse=True)
            stats.per_image_candidates[image_id] = len(group)
            stats.routed += len(kept)
            stats.candidate_nms_suppressed += len(group) - len(kept)
            selected.append((image_id, kept))

        image_cache: dict[str, Image.Image] = {}
        classify_images: list[Image.Image] = []
        classify_records: list[BoxRecord] = []
        for image_id, group in selected:
            source = image_sources.get(image_id)
            if source is None:
                raise FileNotFoundError(f"二阶段找不到图像源: {image_id}")
            if isinstance(source, Image.Image):
                image = source.convert("RGB")
            elif isinstance(source, np.ndarray):
                image = Image.fromarray(np.ascontiguousarray(source.astype(np.uint8))).convert("RGB")
            else:
                image = Image.open(source).convert("RGB")
            image_cache[image_id] = image
            classify_images.extend([image] * len(group))
            classify_records.extend(group)

        decisions = self._predict(classify_images, classify_records)
        accepted_scores: dict[int, float] = {}
        for record, probability in zip(classify_records, decisions, strict=True):
            detector_score = float(record.score or 0.0)
            fused_probability = (
                (1.0 - self.config.detector_score_weight) * float(probability)
                + self.config.detector_score_weight * detector_score
            )
            if fused_probability >= self.config.positive_threshold or (
                self.config.bypass_score is not None and detector_score >= self.config.bypass_score
            ):
                # 二阶段已完成 FSC 最终分类，必须以其置信度取代一级候选分数。
                # 否则低分但被二阶段正确识别的候选会被后续统一阈值再次错误过滤。
                accepted_scores[id(record)] = fused_probability
        output: list[BoxRecord] = []
        for record in records:
            if record.class_id not in self.config.class_ids:
                output.append(record)
            elif id(record) in accepted_scores:
                output.append(
                    BoxRecord(
                        image_id=record.image_id,
                        class_id=record.class_id,
                        xyxy=record.xyxy,
                        score=accepted_scores[id(record)],
                    )
                )
        stats.kept = len(accepted_scores)
        stats.rejected = stats.routed - stats.kept
        stats.elapsed_seconds = time.perf_counter() - started
        for image in image_cache.values():
            image.close()
        return output, stats


class TwoStagePluginLoader:
    """按配置创建二阶段插件。"""

    @staticmethod
    def load(config: TwoStageConfig, device: str | torch.device = "cpu") -> TwoStagePlugin:
        """校验 checkpoint 并加载插件。"""
        if config.checkpoint is None or not Path(config.checkpoint).is_file():
            raise FileNotFoundError(f"two_stage checkpoint 不存在: {config.checkpoint}")
        if config.backbone_checkpoint is not None and not Path(config.backbone_checkpoint).is_file():
            raise FileNotFoundError(f"two_stage backbone checkpoint 不存在: {config.backbone_checkpoint}")
        return TwoStagePlugin(config, device=device)
