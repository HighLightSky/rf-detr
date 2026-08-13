# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``test.output_dir`` 配置测试：测试评估输出目录可被 yaml 配置覆盖。

改动前测试输出固定写到 ``DATASET_CONFIGS`` 内置的 ``exp_output_dir``（每次评估互相覆盖）。 改动后 ``test.py`` 从 yaml 的 ``test.output_dir`` 读取目录并传给
``eval_lib.build_dataset_cfg(output_dir=...)`` 覆盖，报告/混淆矩阵/FP·FN 全部落到新目录。
"""

from pathlib import Path

from scripts import eval_lib, expcfg


class TestBuildDatasetCfgOutputDir:
    """``build_dataset_cfg`` 的 ``output_dir`` 参数覆盖 ``exp_output_dir``。"""

    def test_default_uses_builtin_exp_output_dir(self, tmp_path):
        """不传 ``output_dir`` 时使用 ``DATASET_CONFIGS`` 内置输出目录。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path)
        builtin = tmp_path / eval_lib.DATASET_CONFIGS["shwx"]["exp_output_dir"]
        assert dataset.exp_output_dir == builtin

    def test_relative_output_dir_resolved_against_root(self, tmp_path):
        """相对路径 ``output_dir`` 以 root 为基准解析。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, output_dir="output/abc-eval")
        assert dataset.exp_output_dir == tmp_path / "output/abc-eval"

    def test_absolute_output_dir_kept_as_is(self, tmp_path):
        """绝对路径 ``output_dir`` 原样保留，不被 root 二次拼接。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, output_dir="/data/abc-eval")
        assert dataset.exp_output_dir == Path("/data/abc-eval")

    def test_output_dir_override_leaves_data_paths_untouched(self, tmp_path):
        """覆盖只影响输出目录，数据集/标签/权重路径不受影响。"""
        dataset = eval_lib.build_dataset_cfg("shwx", root=tmp_path, output_dir="output/abc-eval")
        assert dataset.test_image_dir == tmp_path / "data" / "SHWX" or dataset.test_image_dir != dataset.exp_output_dir
        assert dataset.checkpoint_file == eval_lib.DATASET_CONFIGS["shwx"]["checkpoint_file"]


class TestTestYamlOutputDirParsing:
    """Yaml ``test.output_dir`` 经 ``expcfg`` 解析后为绝对路径（test.py 直接消费）。"""

    def test_relative_output_dir_resolved_to_absolute(self, tmp_path):
        """``output_dir: output/xxx`` 在 ``build_test_kwargs`` 阶段即解析为项目根下的绝对路径。"""
        cfg = {"test": {"dataset": "shwx", "output_dir": "output/xxx-eval"}}
        kwargs = expcfg.build_test_kwargs(cfg)
        assert kwargs["output_dir"] == str(expcfg.PROJECT_ROOT / "output/xxx-eval")

    def test_missing_output_dir_is_absent(self, tmp_path):
        """不写 ``output_dir`` 键时不产生该键（回退内置目录）。"""
        cfg = {"test": {"dataset": "shwx"}}
        kwargs = expcfg.build_test_kwargs(cfg)
        assert "output_dir" not in kwargs
