# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""统一批量推理运行时的回归测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from rfdetr.models.postprocess import PostProcess
from scripts.eval_lib import (
    InferenceCfg,
    _InferenceRuntime,
    predict_batched_to_records,
    predict_large_images_per_image,
    predict_mixed_to_records,
    read_test_image_paths,
)


class _TinyDetector(torch.nn.Module):
    """输出固定候选框的最小 detector。"""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """为每张输入生成一个高置信度候选框。"""
        batch_size = images.shape[0]
        logits = torch.full((batch_size, 2, 2), -8.0, device=images.device)
        logits[:, 0, 0] = 8.0
        boxes = torch.zeros((batch_size, 2, 4), device=images.device)
        boxes[:, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5], device=images.device)
        return {"pred_logits": logits, "pred_boxes": boxes}


def _fake_model() -> SimpleNamespace:
    """构造满足批量推理接口的最小模型包装。"""
    context = SimpleNamespace(
        model=_TinyDetector(),
        resolution=32,
        postprocess=PostProcess(num_select=2),
    )
    return SimpleNamespace(
        model=context,
        means=[0.0, 0.0, 0.0],
        stds=[1.0, 1.0, 1.0],
    )


def _write_image(path: Path, value: int) -> None:
    """写入固定尺寸测试图像。"""
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_read_test_image_paths_includes_mixed_supported_formats(tmp_path: Path) -> None:
    """测试集目录中的 JPG、JPEG 和 PNG 应同时进入评测。"""
    for name in ("b.PNG", "a.jpg", "c.jpeg"):
        (tmp_path / name).touch()
    (tmp_path / "ignored.txt").touch()

    assert [path.name for path in read_test_image_paths(tmp_path)] == ["a.jpg", "b.PNG", "c.jpeg"]


def test_inference_cfg_uses_new_runtime_fields() -> None:
    """新配置不再暴露旧的 use_fp16 字段。"""
    cfg = InferenceCfg()
    assert cfg.precision == "auto"
    assert cfg.copy_prefetch is True
    assert not hasattr(cfg, "use_fp16")


def test_batched_runtime_keeps_warmup_results_and_tail_batch(tmp_path: Path) -> None:
    """真实 warmup 不应丢结果，尾 batch 也必须被处理。"""
    image_paths = [tmp_path / f"image_{index}.jpg" for index in range(3)]
    for index, image_path in enumerate(image_paths):
        _write_image(image_path, 30 + index)

    records, throughput, gpu_util, timed_images = predict_batched_to_records(
        _fake_model(),
        image_paths,
        device="cpu",
        conf_threshold=0.5,
        batch_size=2,
        num_workers=0,
        num_classes=1,
        precision="fp32",
        compile_model=False,
        copy_prefetch=False,
        warmup_batches=1,
        progress_interval_s=0.0,
        gpu_monitor_enabled=False,
    )

    assert len(records) == 3
    assert {record.image_id for record in records} == {path.stem for path in image_paths}
    assert throughput >= 0.0
    assert gpu_util is None
    assert timed_images == 1


def test_mixed_runtime_maps_crop_boxes_and_keeps_small_images(tmp_path: Path) -> None:
    """小图和 crop 共用 runtime，crop 坐标应映射回原图。"""
    small_path = tmp_path / "small.jpg"
    _write_image(small_path, 64)
    source_image = np.full((64, 80, 3), 96, dtype=np.uint8)
    crop_sources = [("large", source_image, (10, 20, 50, 60))]

    records, throughput, gpu_util, timed_images, elapsed = predict_mixed_to_records(
        _fake_model(),
        [small_path],
        crop_sources,
        device="cpu",
        conf_threshold=0.5,
        crop_conf_threshold=0.5,
        batch_size=2,
        num_workers=0,
        precision="fp32",
        compile_model=False,
        copy_prefetch=False,
        warmup_batches=0,
        progress_interval_s=0.0,
        gpu_monitor_enabled=False,
    )

    assert {record.image_id for record in records} == {"small", "large"}
    crop_record = next(record for record in records if record.image_id == "large")
    # crop 先缩放到 detector 分辨率，回映射仍必须使用原始 ROI 的 40x40 尺寸。
    assert crop_record.xyxy == (20.0, 30.0, 40.0, 50.0)
    assert throughput >= 0.0
    assert gpu_util is None
    assert timed_images == 2
    assert elapsed >= 0.0


def test_per_image_runtime_keeps_each_large_image_timing_separate() -> None:
    """逐张模式的耗时明细应只包含对应大图的 crop 推理。"""
    class _Tiler:
        """返回固定 crop 的最小切分器。"""

        def prepare_one(
            self, image_path: Path
        ) -> tuple[
            list[tuple[str, np.ndarray, tuple[int, int, int, int]]],
            list[tuple[float, float, float, float]],
            dict[str, float],
        ]:
            """为每张输入返回一个内存 crop。"""
            image = np.full((32, 32, 3), 120, dtype=np.uint8)
            return [(image_path.stem, image, (0, 0, 32, 32))], [], {
                "boundary_seconds": 0.01,
                "crop_prepare_seconds": 0.02,
            }

    model = _fake_model()
    runtime = _InferenceRuntime(
        model,
        device="cpu",
        resolution=32,
        batch_size=2,
        precision="fp32",
        compile_model=False,
        copy_prefetch=False,
    )
    records, timings, stage_stats = predict_large_images_per_image(
        model,
        [Path("large_a.jpg"), Path("large_b.jpg")],
        _Tiler(),
        runtime,
        detector_conf=0.5,
        two_stage_plugin=None,
        two_stage_cfg=None,
        progress_interval_s=0.0,
    )

    assert stage_stats is None
    assert [item["image_id"] for item in timings] == ["large_a", "large_b"]
    assert all(item["crop_count"] == 1 for item in timings)
    assert all(item["boundary_seconds"] == 0.01 for item in timings)
    assert len(records) == 2
