"""比赛交付运行时的无模型单元测试。"""

from __future__ import annotations

import sys
import stat
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

DELIVERY_APP = Path(__file__).resolve().parents[1] / "app"
DELIVERY_DIR = DELIVERY_APP.parent
sys.path.insert(0, str(DELIVERY_APP))
sys.path.insert(0, str(DELIVERY_DIR))

from competition.config import (  # noqa: E402
    DetectorConfig,
    FscNmsConfig,
    MsNmsConfig,
    PostprocessConfig,
    PreprocessConfig,
    RoleConfig,
    load_submission_config,
)
from competition.contracts import CoordinateTransform, InferenceTask, RawDetection  # noqa: E402
from competition.detector.decoding import decode_rfdetr_outputs  # noqa: E402
from competition.pipeline import CompetitionPipeline  # noqa: E402
from competition.detector.multi_backend import MultiBackendDetector  # noqa: E402
from competition.postprocess.shwx_competition import ShwxCompetitionPostprocessor  # noqa: E402
from competition.preprocess.shwx_large_image import ShwxLargeImagePreprocessor  # noqa: E402
from prepare_delivery_assets import _ensure_readable  # noqa: E402


def _postprocessor(
    ms_enabled: bool = True,
    fsc_enabled: bool = True,
    ms_min_box_area: float = 0.0,
) -> ShwxCompetitionPostprocessor:
    """构造使用交付资源类别名称的后处理器。"""
    return ShwxCompetitionPostprocessor(
        PostprocessConfig(
            name="shwx_competition_v1",
            confidence_threshold=0.25,
            class_names_path=Path("resources/shwx_class_names.yaml"),
            ms_min_box_area=ms_min_box_area,
            ms_nms=MsNmsConfig(
                enabled=ms_enabled,
                ms_class_id=3,
                ship_class_ids=(0, 1, 2, 3),
                same_class_iou=0.8,
                same_class_containment=0.9,
                same_class_center_ratio=0.35,
                cross_class_iou=0.9,
                cross_class_containment=0.95,
                cross_class_center_ratio=0.25,
                cross_class_score_margin=0.05,
                keep_ambiguous_cross_class=True,
            ),
            fsc_containment_nms=FscNmsConfig(
                enabled=fsc_enabled,
                containment_enabled=True,
                iou_threshold=0.5,
                containment_threshold=0.95,
                center_ratio_threshold=0.35,
            ),
        )
    )


def test_config_accepts_onnx_and_pytorch_roles(tmp_path: Path) -> None:
    """YAML 可以为不同角色独立选择两种受支持后端。"""
    for file_name in ("main.onnx", "boundary.onnx", "main.pth", "proto.pt"):
        (tmp_path / file_name).touch()
    config_file = tmp_path / "submission.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "preprocess": {
                        "name": "shwx_large_image",
                        "large_image_min_side": 2000,
                        "boundary_confidence": 0.25,
                        "padding": 0,
                        "boundary_nms_iou": 0.5,
                        "proxy_max_side": 704,
                    },
                    "detector": {
                        "name": "rfdetr_multi_backend",
                        "device": "cuda:0",
                        "batch_size": 1,
                        "roles": {
                            "main": {
                                "backend": "pytorch",
                                "model": "main.pth",
                                "proto_guidance_artifact": "proto.pt",
                                "resolution": 1024,
                            },
                            "boundary": {"backend": "onnx", "model": "boundary.onnx", "resolution": 704},
                        },
                    },
                    "postprocess": {
                        "name": "shwx_competition_v1",
                        "confidence_threshold": 0.25,
                        "class_names": "resources/shwx_class_names.yaml",
                        "ms_nms": {
                            "enabled": True,
                            "ms_class_id": 3,
                            "ship_class_ids": [0, 1, 2, 3],
                        },
                        "fsc_containment_nms": {"enabled": True, "containment_enabled": True},
                    },
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    loaded = load_submission_config(config_file, tmp_path)
    assert loaded.detector.roles["main"].backend == "pytorch"
    assert loaded.detector.roles["boundary"].backend == "onnx"


def test_config_accepts_pytorch_boundary_without_proto_guidance(tmp_path: Path) -> None:
    """普通 PyTorch 边界模型不应被强制要求 ProtoGuidance 工件。"""
    for file_name in ("main.pth", "boundary.pth", "proto.pt"):
        (tmp_path / file_name).touch()
    config_file = tmp_path / "submission.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "preprocess": {
                        "name": "shwx_large_image",
                        "large_image_min_side": 2000,
                        "boundary_confidence": 0.25,
                        "padding": 0,
                        "boundary_nms_iou": 0.5,
                        "proxy_max_side": 704,
                    },
                    "detector": {
                        "name": "rfdetr_multi_backend",
                        "device": "cuda:0",
                        "batch_size": 1,
                        "roles": {
                            "main": {
                                "backend": "pytorch",
                                "model": "main.pth",
                                "proto_guidance_artifact": "proto.pt",
                                "resolution": 1024,
                            },
                            "boundary": {"backend": "pytorch", "model": "boundary.pth", "resolution": 704},
                        },
                    },
                    "postprocess": {
                        "name": "shwx_competition_v1",
                        "confidence_threshold": 0.25,
                        "class_names": "resources/shwx_class_names.yaml",
                        "ms_nms": {"enabled": True, "ms_class_id": 3, "ship_class_ids": [0, 1, 2, 3]},
                        "fsc_containment_nms": {"enabled": True, "containment_enabled": True},
                    },
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    loaded = load_submission_config(config_file, tmp_path)
    assert loaded.detector.batch_size == 1
    assert loaded.detector.roles["main"].backend == "pytorch"
    assert loaded.detector.roles["boundary"].backend == "pytorch"
    assert loaded.detector.roles["boundary"].proto_guidance_artifact is None


def test_detector_initialization_error_includes_role_context(tmp_path: Path) -> None:
    """检测角色初始化失败时应保留角色和模型文件上下文。"""
    model_path = tmp_path / "main.pth"
    model_path.touch()
    config = DetectorConfig(
        name="rfdetr_multi_backend",
        device="cuda:0",
        batch_size=1,
        roles={
            "main": RoleConfig(
                backend="pytorch",
                model_path=model_path,
                resolution=1024,
            )
        },
    )
    with mock.patch(
        "competition.detector.multi_backend.PytorchRfdetrDetector",
        side_effect=RuntimeError("cuda 初始化失败"),
    ):
        with pytest.raises(RuntimeError, match="初始化检测角色 main 失败.*main\\.pth") as error:
            MultiBackendDetector(config)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_delivery_asset_permission_is_readable_by_non_root(tmp_path: Path) -> None:
    """复制进入交付目录的模型应允许非 root 容器用户读取。"""
    model_path = tmp_path / "main.pth"
    model_path.touch()
    model_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _ensure_readable(model_path)
    assert model_path.stat().st_mode & stat.S_IROTH


def test_dockerfile_makes_model_directory_readable() -> None:
    """镜像构建必须覆盖源模型文件的过严权限。"""
    dockerfile = (DELIVERY_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN chmod -R a+rX /app/models" in dockerfile


def test_decoder_preserves_task_identity_and_box_coordinates() -> None:
    """解码后的检测框保留裁切来源并处于模型输入坐标系。"""
    task = InferenceTask(
        image_id="image",
        role="main",
        pixels=np.zeros((100, 200, 3), dtype=np.uint8),
        transform=CoordinateTransform(scale_x=1.0, scale_y=1.0),
        task_index=7,
    )
    logits = np.array([[[0.0, 4.0], [-2.0, -3.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.5, 0.4], [0.1, 0.1, 0.2, 0.2]]], dtype=np.float32)
    result = decode_rfdetr_outputs(logits, boxes, [task], max_detections=1)
    assert result[0].task_index == 7
    assert result[0].class_id == 1
    assert result[0].xyxy == pytest.approx((50.0, 30.0, 150.0, 70.0))


def test_large_image_preprocessor_restores_each_crop_task() -> None:
    """大图的不同边界候选生成可区分且可恢复的主检测裁切。"""
    preprocessor = ShwxLargeImagePreprocessor(
        PreprocessConfig(
            name="shwx_large_image",
            large_image_min_side=100,
            boundary_confidence=0.25,
            padding=0,
            boundary_nms_iou=0.5,
            proxy_max_side=64,
        )
    )
    plan = preprocessor.prepare("image", np.zeros((200, 300, 3), dtype=np.uint8), 100, 64)
    tasks = preprocessor.expand(
        plan,
        [
            RawDetection("image", 0, 0.9, (10.0, 20.0, 110.0, 120.0)),
            RawDetection("image", 0, 0.8, (150.0, 50.0, 250.0, 150.0)),
        ],
        100,
    )
    restored = CompetitionPipeline._restore_many(
        tasks,
        [
            RawDetection("image", 1, 0.9, (0.0, 0.0, 100.0, 100.0), task_index=0),
            RawDetection("image", 2, 0.9, (0.0, 0.0, 100.0, 100.0), task_index=1),
        ],
    )
    assert [item.xyxy for item in restored] == pytest.approx(
        [(10.0, 20.0, 110.0, 120.0), (150.0, 50.0, 250.0, 150.0)]
    )


def test_postprocessor_applies_fsc_then_ms_nms() -> None:
    """同图的重叠发射车和民船候选均按启用配置去重。"""
    result = _postprocessor().process(
        [
            RawDetection("image", 24, 0.9, (10.0, 10.0, 30.0, 30.0)),
            RawDetection("image", 24, 0.8, (11.0, 11.0, 31.0, 31.0)),
            RawDetection("image", 3, 0.9, (40.0, 40.0, 60.0, 60.0)),
            RawDetection("image", 3, 0.8, (41.0, 41.0, 61.0, 61.0)),
        ],
        (100, 100),
    )
    assert [(item["category_id"], item["score"]) for item in result] == [(24, 0.9), (3, 0.9)]


@pytest.mark.parametrize(
    ("class_id", "xyxy", "expected_count"),
    [
        pytest.param(3, (10.0, 10.0, 29.0, 30.0), 0, id="discard_small_ms"),
        pytest.param(3, (10.0, 10.0, 35.0, 30.0), 1, id="keep_boundary_area_ms"),
        pytest.param(0, (10.0, 10.0, 29.0, 30.0), 1, id="keep_small_other_class"),
    ],
)
def test_postprocessor_filters_only_ms_boxes_below_configured_area(
    class_id: int,
    xyxy: tuple[float, float, float, float],
    expected_count: int,
) -> None:
    """仅过滤面积低于阈值的民船预测框。"""
    result = _postprocessor(ms_min_box_area=500.0).process(
        [RawDetection("image", class_id, 0.9, xyxy)],
        (100, 100),
    )
    assert len(result) == expected_count
