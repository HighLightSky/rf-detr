# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""stat_class_counts.py 的单元测试：类别实例统计与 JSON 输出正确性。"""

import json

import pytest

from scripts.data_prep.stat_class_counts import count_class_instances, write_counts_json


class TestCountClassInstances:
    """测试 YOLO 标签解析与逐类计数。"""

    def test_counts_instances_across_files(self, tmp_path):
        """多个标签文件的实例计数正确累加。"""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")  # 空文件应跳过

        counts = count_class_instances(tmp_path, num_classes=4)

        assert counts == [2, 1, 0, 1]

    def test_skips_blank_and_comment_lines(self, tmp_path):
        """空行与注释行不影响计数。"""
        (tmp_path / "a.txt").write_text("\n# 注释\n0 0.5 0.5 0.1 0.1\n  \n", encoding="utf-8")

        counts = count_class_instances(tmp_path, num_classes=2)

        assert counts == [1, 0]

    def test_raises_on_out_of_range_class(self, tmp_path):
        """越界 class_id 应报错（数据完整性保护）。"""
        (tmp_path / "a.txt").write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="越界类别"):
            count_class_instances(tmp_path, num_classes=3)

    def test_missing_dir_raises(self, tmp_path):
        """标签目录不存在应报 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            count_class_instances(tmp_path / "nope", num_classes=2)


class TestWriteCountsJson:
    """测试 JSON 输出内容（counts/n_ref/class_names）与落盘格式。"""

    def test_writes_payload_with_geometric_mean_ref(self, tmp_path):
        """n_ref 为 n_max/n_min 的几何平均；JSON 可被 criterion 构建端直接读取。"""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        out = tmp_path / "sub" / "class_counts.json"

        payload = write_counts_json(tmp_path, out, num_classes=2, class_names=["HM", "LQS"])

        assert payload["counts"] == [1, 2]
        assert payload["n_max"] == 2.0
        assert payload["n_min"] == 1.0
        assert payload["n_zero"] == 0
        assert payload["n_ref"] == pytest.approx(2**0.5)
        assert payload["class_names"] == ["HM", "LQS"]
        # 落盘格式：counts 必须可用 "counts" 键直接读取（lwdetr 构建端依赖）
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded["counts"] == [1, 2]

    def test_writes_ref_from_positive_min_when_some_classes_are_zero(self, tmp_path):
        """存在零样本类别时，n_ref 使用正样本类别最小值，避免退化为 0。"""
        (tmp_path / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        payload = write_counts_json(tmp_path, tmp_path / "out.json", num_classes=3)

        assert payload["counts"] == [0, 2, 0]
        assert payload["n_min"] == 2.0
        assert payload["n_zero"] == 2
        assert payload["n_ref"] == pytest.approx(2.0)

    def test_rejects_all_zero_counts(self, tmp_path):
        """全空标签目录应报错，避免生成无法用于类均衡的统计文件。"""
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="所有类别计数均为 0"):
            write_counts_json(tmp_path, tmp_path / "out.json", num_classes=3)

    def test_rejects_wrong_length_class_names(self, tmp_path):
        """class_names 长度与 num_classes 不一致时报错。"""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="不一致"):
            write_counts_json(tmp_path, tmp_path / "out.json", num_classes=3, class_names=["a", "b"])
