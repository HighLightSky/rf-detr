# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切数据集划分脚本测试：比例切分、符号链接建立与 data.yaml 更新。

覆盖 ``split_large_cut_dataset.py`` 的核心函数：比例解析、纯函数切分、链接幂等性与 yaml 更新。
"""

from pathlib import Path

import numpy as np
import yaml

from scripts.split_large_cut_dataset import (
    create_split_links,
    parse_ratios,
    plan_split,
    write_data_yaml,
)


def _make_fake_dataset(root: Path, n: int = 8) -> Path:
    """构造假数据集：n 张 jpg + 同名 txt + data.yaml。"""
    dataset_dir = root / "large-cut"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "labels").mkdir(parents=True)
    for i in range(1, n + 1):
        stem = f"lc-{i:04d}-100_100"
        (dataset_dir / "images" / f"{stem}.jpg").write_bytes(b"fake-jpg")
        (dataset_dir / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (dataset_dir / "data.yaml").write_text(
        "path: /home/liu/wzt/datasets/large-cut\ntrain: images\nval: images\nnc: 1\nnames:\n  0: sample\n",
        encoding="utf-8",
    )
    return dataset_dir


class TestParseRatios:
    """``parse_ratios`` 比例字符串解析。"""

    def test_valid_ratios(self):
        """``8:1:1`` 解析为三个浮点数。"""
        assert parse_ratios("8:1:1") == (8.0, 1.0, 1.0)

    def test_float_ratios(self):
        """小数比例同样可解析。"""
        assert parse_ratios("0.8:0.1:0.1") == (0.8, 0.1, 0.1)

    def test_invalid_format(self):
        """少于三段的比例字符串报错。"""
        import pytest

        with pytest.raises(ValueError):
            parse_ratios("8:1")

    def test_non_numeric(self):
        """非数值比例报错。"""
        import pytest

        with pytest.raises(ValueError):
            parse_ratios("a:b:c")

    def test_non_positive(self):
        """非正比例报错。"""
        import pytest

        with pytest.raises(ValueError):
            parse_ratios("0:1:1")


class TestPlanSplit:
    """``plan_split`` 纯函数切分。"""

    def test_ratio_counts(self):
        """80 个样本按 8:1:1 得到 64/8/8。"""
        names = [f"img-{i:03d}" for i in range(80)]
        rng = np.random.default_rng(42)
        result = plan_split(names, (8.0, 1.0, 1.0), rng)
        assert len(result["train"]) == 64
        assert len(result["val"]) == 8
        assert len(result["test"]) == 8

    def test_no_overlap_no_missing(self):
        """划分结果无重复、无遗漏（并集等于全集）。"""
        names = [f"img-{i:03d}" for i in range(80)]
        rng = np.random.default_rng(42)
        result = plan_split(names, (8.0, 1.0, 1.0), rng)
        assert set(result["train"]) | set(result["val"]) | set(result["test"]) == set(names)
        assert len(result["train"]) + len(result["val"]) + len(result["test"]) == len(names)

    def test_reproducible_with_seed(self):
        """相同种子两次切分结果一致。"""
        names = [f"img-{i:03d}" for i in range(80)]
        r1 = plan_split(names, (8.0, 1.0, 1.0), np.random.default_rng(42))
        r2 = plan_split(names, (8.0, 1.0, 1.0), np.random.default_rng(42))
        assert r1 == r2

    def test_different_seed_shuffles(self):
        """不同种子切分结果不同（打乱生效）。"""
        names = [f"img-{i:03d}" for i in range(80)]
        r1 = plan_split(names, (8.0, 1.0, 1.0), np.random.default_rng(1))
        r2 = plan_split(names, (8.0, 1.0, 1.0), np.random.default_rng(2))
        assert r1 != r2


class TestCreateSplitLinks:
    """``create_split_links`` 符号链接建立。"""

    def test_structure_and_targets(self, tmp_path):
        """8 个样本按 8:1:1（6/1/1）后目录结构与链接目标正确。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=8)
        rng = np.random.default_rng(42)
        plan = plan_split([f"lc-{i:04d}-100_100" for i in range(1, 9)], (6.0, 1.0, 1.0), rng)
        create_split_links(dataset_dir, plan)

        for split in ("train", "val", "test"):
            image_dir = dataset_dir / "images" / split
            label_dir = dataset_dir / "labels" / split
            assert image_dir.is_dir(), f"缺少 {split} 图像目录"
            assert label_dir.is_dir(), f"缺少 {split} 标签目录"
            assert len(list(image_dir.iterdir())) == len(plan[split])
            assert len(list(label_dir.iterdir())) == len(plan[split])
            for stem in plan[split]:
                link = image_dir / f"{stem}.jpg"
                assert link.is_symlink()
                assert link.resolve() == (dataset_dir / "images" / f"{stem}.jpg").resolve()
                assert link.is_file(), "符号链接应跟随到真实文件"

    def test_idempotent(self, tmp_path):
        """重复执行不重复建链（数量不变，无异常）。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=8)
        plan = {
            "train": ["lc-0001-100_100", "lc-0002-100_100"],
            "val": ["lc-0003-100_100"],
            "test": ["lc-0004-100_100"],
        }
        first = create_split_links(dataset_dir, plan)
        second = create_split_links(dataset_dir, plan)
        assert second == 0  # 全部已存在且指向正确，跳过
        assert first == 8  # 4 样本 × 2（image+label）
        for split in ("train", "val", "test"):
            image_dir = dataset_dir / "images" / split
            assert len(list(image_dir.iterdir())) == len(plan[split])

    def test_force_rebuild(self, tmp_path):
        """``force=True`` 清空重建：新链接总数等于样本 × 2。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=8)
        plan_a = {
            "train": ["lc-0001-100_100"],
            "val": ["lc-0002-100_100"],
            "test": ["lc-0003-100_100"],
        }
        create_split_links(dataset_dir, plan_a)
        # 换一批样本重建
        plan_b = {
            "train": ["lc-0004-100_100"],
            "val": ["lc-0005-100_100"],
            "test": ["lc-0006-100_100"],
        }
        create_split_links(dataset_dir, plan_b, force=True)
        assert (dataset_dir / "images" / "train" / "lc-0001-100_100.jpg").is_symlink() is False
        assert (dataset_dir / "images" / "train" / "lc-0004-100_100.jpg").is_file()

    def test_missing_label_skipped(self, tmp_path):
        """缺少同名标签的图像被跳过并告警。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=2)
        # 删除一个标签
        (dataset_dir / "labels" / "lc-0002-100_100.txt").unlink()
        plan = {"train": ["lc-0001-100_100", "lc-0002-100_100"], "val": [], "test": []}
        created = create_split_links(dataset_dir, plan)
        assert created == 2  # 只有 lc-0001 的 image+label


class TestWriteDataYaml:
    """``write_data_yaml`` 备份与写入。"""

    def test_backup_and_new_content(self, tmp_path):
        """原 yaml 备份为 .bak，新 yaml 指向 images/{train,val,test}。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=2)
        write_data_yaml(dataset_dir)

        assert (dataset_dir / "data.yaml.bak").is_file()
        cfg = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
        assert cfg["train"] == "images/train"
        assert cfg["val"] == "images/val"
        assert cfg["test"] == "images/test"
        assert cfg["nc"] == 1
        assert cfg["names"] == {0: "sample"}
        assert cfg["path"] == str(dataset_dir.resolve())

    def test_idempotent_no_rebackup(self, tmp_path):
        """二次执行不重复备份（已划分过的 yaml 视为最新）。"""
        dataset_dir = _make_fake_dataset(tmp_path, n=2)
        write_data_yaml(dataset_dir)
        bak_content = (dataset_dir / "data.yaml.bak").read_text(encoding="utf-8")
        write_data_yaml(dataset_dir)
        assert (dataset_dir / "data.yaml.bak").read_text(encoding="utf-8") == bak_content
