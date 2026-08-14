# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ROT90 增广预设测试：随机直角翻转（90° 旋转 + 水平/垂直翻转）。

覆盖 ``AUG_ROT90`` 预设的注册、几何性识别、bbox 变换正确性，以及 ``expcfg.AUG_PRESETS`` 的预设名映射。
"""

import pytest
import torch
from PIL import Image

from rfdetr.datasets.aug_configs import AUG_ROT90
from rfdetr.datasets.transforms import GEOMETRIC_TRANSFORMS, AlbumentationsWrapper

#: AUG_ROT90 预设应包含的变换键（顺序即应用顺序）
_EXPECTED_KEYS = ("RandomRotate90", "HorizontalFlip", "VerticalFlip")


class TestAUGROT90Config:
    """``AUG_ROT90`` 预设定义。"""

    def test_has_three_transform_keys(self):
        """预设恰含 RandomRotate90/HorizontalFlip/VerticalFlip 三个键。"""
        assert tuple(AUG_ROT90.keys()) == _EXPECTED_KEYS

    def test_probabilities_default_05(self):
        """每个变换的概率为 0.5（与翻转采样一致）。"""
        for params in AUG_ROT90.values():
            assert params["p"] == 0.5

    def test_all_keys_geometric(self):
        """三个变换都应被识别为几何变换（bbox 自动变换）。"""
        for name in AUG_ROT90:
            assert name in GEOMETRIC_TRANSFORMS, f"{name} 不在 GEOMETRIC_TRANSFORMS 中"

    def test_from_config_builds_three_geometric_wrappers(self):
        """``from_config`` 按预设构建 3 个 wrapper 且全部为几何变换。"""
        wrappers = AlbumentationsWrapper.from_config(AUG_ROT90)
        assert len(wrappers) == 3
        assert all(w._is_geometric for w in wrappers)


class TestAUGROT90BBoxTransform:
    """ROT90 各变换对 bbox 的确定性变换。"""

    @pytest.mark.parametrize(
        "name,params,box_in,box_out",
        [
            # 100x100 图上水平翻转：x 镜像（w - x1, x 保持）
            ("HorizontalFlip", {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [70.0, 20.0, 90.0, 40.0]),
            # 垂直翻转：y 镜像
            ("VerticalFlip", {"p": 1.0}, [10.0, 20.0, 30.0, 40.0], [10.0, 60.0, 30.0, 80.0]),
        ],
    )
    def test_flip_boxes_deterministic(self, name, params, box_in, box_out):
        """水平/垂直翻转的 bbox 坐标变化确定且正确。"""
        from rfdetr.datasets.transforms import _build_albu_transform

        transform = _build_albu_transform(name, params)
        wrapper = AlbumentationsWrapper(transform)
        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([box_in]), "labels": torch.tensor([1])}

        _, aug_target = wrapper(image, target)

        assert torch.allclose(aug_target["boxes"], torch.tensor([box_out]), atol=1.0)

    def test_rot90_boxes_stay_in_bounds(self):
        """随机 90° 旋转后 bbox 仍为合法框（不越界不退化）。"""
        from rfdetr.datasets.transforms import _build_albu_transform

        transform = _build_albu_transform("RandomRotate90", {"p": 1.0})
        wrapper = AlbumentationsWrapper(transform)
        image = Image.new("RGB", (100, 60))
        target = {"boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]), "labels": torch.tensor([1])}

        for _ in range(8):
            _, aug_target = wrapper(image, target)
            x0, y0, x1, y1 = aug_target["boxes"][0].tolist()
            assert x1 > x0 and y1 > y0, f"旋转后框退化: {aug_target['boxes']}"
            assert 0 <= x0 <= x1 and 0 <= y0 <= y1


class TestAUGPresetRegistration:
    """``expcfg.AUG_PRESETS`` 注册。"""

    def test_rot90_registered(self):
        """``ROT90`` 预设名映射到 ``AUG_ROT90`` 本体。"""
        from scripts.expcfg import AUG_PRESETS

        assert AUG_PRESETS["ROT90"] is AUG_ROT90
