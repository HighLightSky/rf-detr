"""比赛交付 YAML 的严格配置解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("/app/competition/configs/submission.yaml")
DEFAULT_MODEL_DIR = Path("/app/models")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """校验一个 YAML 节点为映射。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是字典")
    return value


def _strict_keys(value: Mapping[str, Any], name: str, allowed: set[str]) -> None:
    """拒绝拼写错误或未支持的配置项。"""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} 存在未知配置项: {', '.join(sorted(unknown))}")


def _model_path(model_dir: Path, name: Any, label: str) -> Path:
    """将逻辑模型名安全解析到固定模型目录。"""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} 必须是非空模型文件名")
    path = Path(name)
    if path.name != name:
        raise ValueError(f"{label} 只能是 models 目录内的文件名")
    resolved = model_dir / path
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} 不存在: {resolved}")
    return resolved


@dataclass(frozen=True)
class RoleConfig:
    """主检测或边界检测模型的运行配置。"""

    backend: str
    model_path: Path
    resolution: int
    proto_guidance_artifact: Path | None = None


@dataclass(frozen=True)
class DetectorConfig:
    """多角色检测器配置。"""

    name: str
    device: str
    batch_size: int
    roles: dict[str, RoleConfig]


@dataclass(frozen=True)
class PreprocessConfig:
    """普通图和大图预处理配置。"""

    name: str
    large_image_min_side: int
    boundary_confidence: float
    padding: int
    boundary_nms_iou: float
    proxy_max_side: int


@dataclass(frozen=True)
class MsNmsConfig:
    """民船 NMS 配置。"""

    enabled: bool
    ms_class_id: int
    ship_class_ids: tuple[int, ...]
    same_class_iou: float
    same_class_containment: float
    same_class_center_ratio: float
    cross_class_iou: float
    cross_class_containment: float
    cross_class_center_ratio: float
    cross_class_score_margin: float
    keep_ambiguous_cross_class: bool


@dataclass(frozen=True)
class FscNmsConfig:
    """发射车 NMS 配置。"""

    enabled: bool
    containment_enabled: bool
    iou_threshold: float
    containment_threshold: float
    center_ratio_threshold: float


@dataclass(frozen=True)
class PostprocessConfig:
    """比赛输出后处理配置。"""

    name: str
    confidence_threshold: float
    class_names_path: Path
    ms_nms: MsNmsConfig
    fsc_containment_nms: FscNmsConfig


@dataclass(frozen=True)
class SubmissionConfig:
    """完整交付运行时配置。"""

    preprocess: PreprocessConfig
    detector: DetectorConfig
    postprocess: PostprocessConfig


def _parse_role(value: Any, label: str, model_dir: Path) -> RoleConfig:
    """解析一个模型角色配置。"""
    mapping = _mapping(value, label)
    _strict_keys(mapping, label, {"backend", "model", "resolution", "proto_guidance_artifact"})
    backend = str(mapping.get("backend", ""))
    if backend not in {"onnx", "pytorch"}:
        raise ValueError(f"{label}.backend 必须为 onnx 或 pytorch")
    resolution = int(mapping.get("resolution", 0))
    if resolution <= 0:
        raise ValueError(f"{label}.resolution 必须为正整数")
    artifact: Path | None = None
    if mapping.get("proto_guidance_artifact") is not None:
        artifact = _model_path(model_dir, mapping["proto_guidance_artifact"], f"{label}.proto_guidance_artifact")
    if backend == "pytorch" and artifact is None:
        raise ValueError(f"{label} 使用 pytorch 时必须配置 proto_guidance_artifact")
    return RoleConfig(
        backend=backend,
        model_path=_model_path(model_dir, mapping.get("model"), f"{label}.model"),
        resolution=resolution,
        proto_guidance_artifact=artifact,
    )


def _parse_ms_nms(value: Any) -> MsNmsConfig:
    """解析民船 NMS 配置。"""
    mapping = _mapping(value, "postprocess.ms_nms")
    allowed = {
        "enabled", "ms_class_id", "ship_class_ids", "same_class_iou", "same_class_containment",
        "same_class_center_ratio", "cross_class_iou", "cross_class_containment", "cross_class_center_ratio",
        "cross_class_score_margin", "keep_ambiguous_cross_class",
    }
    _strict_keys(mapping, "postprocess.ms_nms", allowed)
    ids = mapping.get("ship_class_ids", [])
    if not isinstance(ids, list) or not ids:
        raise ValueError("postprocess.ms_nms.ship_class_ids 必须是非空列表")
    result = MsNmsConfig(
        enabled=bool(mapping.get("enabled", False)),
        ms_class_id=int(mapping.get("ms_class_id", 3)),
        ship_class_ids=tuple(int(item) for item in ids),
        same_class_iou=float(mapping.get("same_class_iou", 0.8)),
        same_class_containment=float(mapping.get("same_class_containment", 0.9)),
        same_class_center_ratio=float(mapping.get("same_class_center_ratio", 0.35)),
        cross_class_iou=float(mapping.get("cross_class_iou", 0.9)),
        cross_class_containment=float(mapping.get("cross_class_containment", 0.95)),
        cross_class_center_ratio=float(mapping.get("cross_class_center_ratio", 0.25)),
        cross_class_score_margin=float(mapping.get("cross_class_score_margin", 0.05)),
        keep_ambiguous_cross_class=bool(mapping.get("keep_ambiguous_cross_class", True)),
    )
    if result.ms_class_id not in result.ship_class_ids:
        raise ValueError("postprocess.ms_nms.ms_class_id 必须属于 ship_class_ids")
    return result


def _parse_fsc_nms(value: Any) -> FscNmsConfig:
    """解析发射车 NMS 配置。"""
    mapping = _mapping(value, "postprocess.fsc_containment_nms")
    _strict_keys(mapping, "postprocess.fsc_containment_nms", {
        "enabled", "containment_enabled", "iou_threshold", "containment_threshold", "center_ratio_threshold",
    })
    result = FscNmsConfig(
        enabled=bool(mapping.get("enabled", False)),
        containment_enabled=bool(mapping.get("containment_enabled", False)),
        iou_threshold=float(mapping.get("iou_threshold", 0.5)),
        containment_threshold=float(mapping.get("containment_threshold", 0.95)),
        center_ratio_threshold=float(mapping.get("center_ratio_threshold", 0.35)),
    )
    if not 0.0 < result.iou_threshold <= 1.0:
        raise ValueError("postprocess.fsc_containment_nms.iou_threshold 必须位于 (0, 1]")
    if not 0.0 <= result.containment_threshold <= 1.0:
        raise ValueError("postprocess.fsc_containment_nms.containment_threshold 必须位于 [0, 1]")
    if result.center_ratio_threshold < 0.0:
        raise ValueError("postprocess.fsc_containment_nms.center_ratio_threshold 不能为负数")
    return result


def load_submission_config(config_path: Path, model_dir: Path) -> SubmissionConfig:
    """加载并严格校验内置提交配置。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"提交配置不存在: {config_path}")
    root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "submission")
    _strict_keys(root, "submission", {"pipeline"})
    pipeline = _mapping(root.get("pipeline"), "pipeline")
    _strict_keys(pipeline, "pipeline", {"preprocess", "detector", "postprocess"})
    pre = _mapping(pipeline.get("preprocess"), "preprocess")
    _strict_keys(pre, "preprocess", {
        "name", "large_image_min_side", "boundary_confidence", "padding", "boundary_nms_iou", "proxy_max_side",
    })
    preprocess = PreprocessConfig(
        name=str(pre.get("name", "")),
        large_image_min_side=int(pre.get("large_image_min_side", 0)),
        boundary_confidence=float(pre.get("boundary_confidence", 0.25)),
        padding=int(pre.get("padding", 0)),
        boundary_nms_iou=float(pre.get("boundary_nms_iou", 0.5)),
        proxy_max_side=int(pre.get("proxy_max_side", 704)),
    )
    if preprocess.name not in {"direct", "shwx_large_image"}:
        raise ValueError("preprocess.name 必须为 direct 或 shwx_large_image")
    if preprocess.large_image_min_side <= 0 or preprocess.proxy_max_side <= 0 or preprocess.padding < 0:
        raise ValueError("preprocess 的尺寸参数不合法")
    detector_value = _mapping(pipeline.get("detector"), "detector")
    _strict_keys(detector_value, "detector", {"name", "device", "batch_size", "roles"})
    roles_value = _mapping(detector_value.get("roles"), "detector.roles")
    _strict_keys(roles_value, "detector.roles", {"main", "boundary"})
    roles = {name: _parse_role(value, f"detector.roles.{name}", model_dir) for name, value in roles_value.items()}
    if "main" not in roles or (preprocess.name == "shwx_large_image" and "boundary" not in roles):
        raise ValueError("当前预处理配置必须提供所需的 main 和 boundary 模型角色")
    detector = DetectorConfig(
        name=str(detector_value.get("name", "")),
        device=str(detector_value.get("device", "cuda:0")),
        batch_size=int(detector_value.get("batch_size", 1)),
        roles=roles,
    )
    if detector.name != "rfdetr_multi_backend" or detector.batch_size <= 0:
        raise ValueError("detector.name 必须为 rfdetr_multi_backend，且 batch_size 必须为正整数")
    post = _mapping(pipeline.get("postprocess"), "postprocess")
    _strict_keys(post, "postprocess", {"name", "confidence_threshold", "class_names", "ms_nms", "fsc_containment_nms"})
    postprocess = PostprocessConfig(
        name=str(post.get("name", "")),
        confidence_threshold=float(post.get("confidence_threshold", 0.25)),
        class_names_path=Path(str(post.get("class_names", "resources/shwx_class_names.yaml"))),
        ms_nms=_parse_ms_nms(post.get("ms_nms", {})),
        fsc_containment_nms=_parse_fsc_nms(post.get("fsc_containment_nms", {})),
    )
    if postprocess.name != "shwx_competition_v1" or not 0.0 <= postprocess.confidence_threshold <= 1.0:
        raise ValueError("postprocess 配置不合法")
    return SubmissionConfig(preprocess=preprocess, detector=detector, postprocess=postprocess)
