# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""测试配置 ``dataset_dir`` / ``resolution`` 测试：数据集根目录与推理分辨率可被 yaml 配置覆盖。

改动前测试数据集目录固定在 ``DATASET_CONFIGS`` 内置的 ``data_dir``，重新标注后的
数据集（如 ``SHWX-dataset-dict-redo``）无法直接评估。改动后 ``test.py`` 从 yaml 的
``test.dataset_dir`` 读取目录覆盖数据路径（与训练侧 ``dataset_dir`` 同一配置模式），
``test.resolution`` 显式指定推理输入分辨率（nano 704 训练时须与训练一致）。
"""

from pathlib import Path

from scripts import eval_lib, expcfg


class TestBuildDatasetCfgDataDir:
    """``build_dataset_cfg`` 的 ``data_dir`` 参数覆盖内置数据集根目录。"""

    def test_default_uses_builtin_data_dir(self, tmp_path):
        """不传 ``data_dir`` 时使用 ``DATASET_CONFIGS`` 内置数据目录。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path)
        assert dataset.data_dir == Path(eval_lib.DATASET_CONFIGS["shwx"]["data_dir"])

    def test_data_dir_override_moves_image_and_label_paths(self, tmp_path):
        """``data_dir`` 覆盖后，测试图像/标签路径跟随新根目录，类别语义保持不变。"""
        new_root = tmp_path / "SHWX-dataset-dict-redo"
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, data_dir=new_root)
        assert dataset.data_dir == new_root
        assert dataset.test_image_dir == new_root / "images" / "test"
        assert dataset.label_dir == new_root / "labels" / "test"
        # 类别语义（类名/大类分组/IoU 阈值/类别数）仍来自内置配置
        assert dataset.num_classes == eval_lib.DATASET_CONFIGS["shwx"]["num_classes"]
        assert dataset.class_to_group == eval_lib.DATASET_CONFIGS["shwx"]["class_to_group"]
        assert dataset.group_iou_thresholds == eval_lib.DATASET_CONFIGS["shwx"]["group_iou_thresholds"]
        assert dataset.vehicle_class_ids == frozenset(eval_lib.DATASET_CONFIGS["shwx"]["vehicle_class_ids"])

    def test_absolute_data_dir_kept_as_is(self, tmp_path):
        """绝对路径 ``data_dir`` 原样保留，不被 root 二次拼接。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, data_dir="/data/SHWX-redo")
        assert dataset.data_dir == Path("/data/SHWX-redo")
        assert dataset.test_image_dir == Path("/data/SHWX-redo") / "images" / "test"

    def test_output_dir_override_does_not_affect_data_dir(self, tmp_path):
        """``output_dir`` 与 ``data_dir`` 独立覆盖：输出目录不受数据目录覆盖影响。"""
        dataset = eval_lib.build_dataset_cfg(
            "shwx",
            root=tmp_path,
            output_dir="output/abc-eval",
            data_dir=tmp_path / "SHWX-redo",
        )
        assert dataset.exp_output_dir == tmp_path / "output" / "abc-eval"
        assert dataset.test_image_dir == tmp_path / "SHWX-redo" / "images" / "test"


class TestTestYamlDatasetDirParsing:
    """yaml ``test.dataset_dir`` / ``test.resolution`` 经 ``expcfg`` 解析后原样保留。"""

    def test_dataset_dir_resolved_to_absolute(self, tmp_path):
        """``dataset_dir: /绝对路径`` 在 ``build_test_kwargs`` 阶段原样保留（不二次拼接）。"""
        cfg = {"test": {"dataset": "shwx", "dataset_dir": "/data/SHWX-dataset-dict-redo"}}
        kwargs = expcfg.build_test_kwargs(cfg)
        assert kwargs["dataset_dir"] == "/data/SHWX-dataset-dict-redo"

    def test_dataset_defaults_to_shwx_when_omitted(self, tmp_path):
        """省略 ``dataset`` 键时回退内置 ``shwx``（测试.py 消费逻辑）。"""
        cfg = {"test": {"dataset_dir": "/data/SHWX-dataset-dict-redo"}}
        kwargs = expcfg.build_test_kwargs(cfg)
        assert "dataset" not in kwargs
        assert kwargs["dataset_dir"] == "/data/SHWX-dataset-dict-redo"

    def test_resolution_kept_as_int(self, tmp_path):
        """``resolution: 704`` 解析后为整数。"""
        cfg = {"test": {"resolution": 704}}
        kwargs = expcfg.build_test_kwargs(cfg)
        assert kwargs["resolution"] == 704
        assert isinstance(kwargs["resolution"], int)
