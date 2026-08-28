"""部署结果 JSON 评测脚本测试。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.eval_deploy_result import evaluate_result_file, visualize_result_file


def test_evaluate_result_file_calculates_metrics_and_runtime(tmp_path: Path) -> None:
    """结果 JSON 应按 SHWX 三大类规则统计并计算最长耗时。"""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "first.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n4 0.25 0.25 0.2 0.2\n24 0.75 0.75 0.1 0.1\n",
        encoding="utf-8",
    )
    (labels / "second.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "status": "success",
                "images": [
                    {
                        "image_id": "first",
                        "width": 100,
                        "height": 100,
                        "run_end_timestamp": 1000,
                        "objects": [
                            {"category_id": 0, "score": 0.9, "bbox": [25, 25, 75, 75]},
                            {"category_id": 24, "score": 0.8, "bbox": [70, 70, 80, 80]},
                            {"category_id": 1, "score": 0.1, "bbox": [0, 0, 10, 10]},
                        ],
                    },
                    {
                        "image_id": "second",
                        "width": 100,
                        "height": 100,
                        "run_end_timestamp": 1250,
                        "objects": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_result_file(result_file, labels)

    assert report.all_result.tp == 2
    assert report.all_result.fp == 1
    assert report.all_result.fn == 2
    assert report.group_macro["ship"]["recall"] == pytest.approx(0.125)
    assert report.total_macro["recall"] == pytest.approx((0.125 + 0.0 + 1.0) / 3)
    assert report.longest_runtime_ms == pytest.approx(250.0)


def test_evaluate_result_file_uses_explicit_runtime(tmp_path: Path) -> None:
    """存在单图耗时字段时应优先使用显式值。"""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "image.txt").write_text("", encoding="utf-8")
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_id": "image",
                        "width": 10,
                        "height": 10,
                        "run_end_timestamp": 100,
                        "runtime_ms": 12.5,
                        "objects": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_result_file(result_file, labels)

    assert report.longest_runtime_ms == pytest.approx(12.5)


def test_visualize_result_file_draws_ship_and_vehicle_ignores_others(tmp_path: Path) -> None:
    """可视化应叠加船舶与发射车 GT/预测框，忽略其余类别的框。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "first.png"), np.zeros((100, 100, 3), dtype=np.uint8))

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "first.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n24 0.75 0.75 0.1 0.1\n4 0.25 0.25 0.2 0.2\n",
        encoding="utf-8",
    )
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_id": "first",
                        "width": 100,
                        "height": 100,
                        "objects": [
                            {"category_id": 0, "score": 0.9, "bbox": [20, 20, 60, 60]},
                            {"category_id": 24, "score": 0.8, "bbox": [70, 70, 80, 80]},
                            {"category_id": 4, "score": 0.7, "bbox": [10, 10, 20, 20]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    written = visualize_result_file(result_file, labels, image_dir, tmp_path / "viz")

    assert [path.name for path in written] == ["first.png"]
    output = tmp_path / "viz" / "first.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_visualize_result_file_respects_custom_class_set(tmp_path: Path) -> None:
    """传入自定义类别集合时应只绘制该类别的框。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "first.png"), np.zeros((100, 100, 3), dtype=np.uint8))

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "first.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_id": "first",
                        "width": 100,
                        "height": 100,
                        "objects": [{"category_id": 0, "score": 0.9, "bbox": [20, 20, 60, 60]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # 只绘制发射车，因此既有 GT 又无预测时当前图像不会落盘。
    written = visualize_result_file(
        result_file,
        labels,
        image_dir,
        tmp_path / "viz",
        class_ids_to_draw=frozenset({24}),
    )

    assert written == []
