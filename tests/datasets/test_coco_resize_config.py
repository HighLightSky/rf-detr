# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests for _build_train_resize_config."""

import pytest

from rfdetr.datasets.coco import _build_train_resize_config


class TestBuildTrainResizeConfigStructure:
    """Top-level structure is always a single-element list wrapping a OneOf."""

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
            pytest.param([640], False, id="nonsquare-single"),
            pytest.param([480, 640], False, id="nonsquare-multi"),
        ],
    )
    def test_returns_single_element_list(self, scales, square):
        result = _build_train_resize_config(scales, square=square)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
            pytest.param([640], False, id="nonsquare-single"),
            pytest.param([480, 640], False, id="nonsquare-multi"),
        ],
    )
    def test_top_level_is_oneof_with_two_branches(self, scales, square):
        result = _build_train_resize_config(scales, square=square)
        entry = result[0]
        assert "OneOf" in entry
        oneof = entry["OneOf"]
        assert len(oneof["transforms"]) == 2


class TestBuildTrainResizeConfigSquareSingleScale:
    """Square=True, single scale — OneOf[Resize] + Sequential[..., OneOf[RandomSizedBBoxSafeCrop]]."""

    def test_option_a_is_oneof_wrapping_single_resize(self):
        result = _build_train_resize_config([640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [{"Resize": {"height": 640, "width": 640}}],
            }
        }

    def test_option_b_is_sequential_with_oneof_crop(self):
        result = _build_train_resize_config([640], square=True)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 640,
                                        "width": 640,
                                        "erosion_rate": 0.0,
                                    }
                                },
                            ],
                        }
                    },
                ]
            }
        }

    def test_uses_correct_scale_value(self):
        result = _build_train_resize_config([480], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [{"Resize": {"height": 480, "width": 480}}],
            }
        }


class TestBuildTrainResizeConfigSquareMultiScale:
    """Square=True, multiple scales — OneOf[Resize] + Sequential[..., OneOf[RandomSizedBBoxSafeCrop]]."""

    def test_option_a_is_oneof_of_resizes(self):
        result = _build_train_resize_config([480, 640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "OneOf": {
                "transforms": [
                    {"Resize": {"height": 480, "width": 480}},
                    {"Resize": {"height": 640, "width": 640}},
                ],
            }
        }

    def test_option_b_is_sequential_with_oneof_crop(self):
        result = _build_train_resize_config([480, 640], square=True)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 480,
                                        "width": 480,
                                        "erosion_rate": 0.0,
                                    }
                                },
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 640,
                                        "width": 640,
                                        "erosion_rate": 0.0,
                                    }
                                },
                            ],
                        }
                    },
                ]
            }
        }

    def test_three_scales_produce_three_resize_options(self):
        result = _build_train_resize_config([384, 512, 640], square=True)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert len(option_a["OneOf"]["transforms"]) == 3


class TestBuildTrainResizeConfigNonSquareSingleScale:
    """Square=False, single scale — SmallestMaxSize uses scalar, default cap 1333."""

    def test_option_a_uses_scalar_size(self):
        result = _build_train_resize_config([640], square=False)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": 640}},
                    {"CappedLongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_option_b_uses_scalar_size(self):
        result = _build_train_resize_config([640], square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 640,
                                        "width": 640,
                                        "erosion_rate": 0.0,
                                    }
                                },
                            ]
                        }
                    },
                ]
            }
        }

    def test_custom_max_size(self):
        result = _build_train_resize_config([640], square=False, max_size=800)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a["Sequential"]["transforms"][1] == {"CappedLongestMaxSize": {"max_size": 800}}


class TestBuildTrainResizeConfigNonSquareMultiScale:
    """Square=False, multiple scales — SmallestMaxSize uses list directly."""

    def test_option_a_uses_list_size(self):
        result = _build_train_resize_config([480, 640], square=False)
        option_a = result[0]["OneOf"]["transforms"][0]
        assert option_a == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [480, 640]}},
                    {"CappedLongestMaxSize": {"max_size": 1333}},
                ]
            }
        }

    def test_option_b_uses_list_size(self):
        result = _build_train_resize_config([480, 640], square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        assert option_b == {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 480,
                                        "width": 480,
                                        "erosion_rate": 0.0,
                                    }
                                },
                                {
                                    "RandomSizedBBoxSafeCrop": {
                                        "height": 640,
                                        "width": 640,
                                        "erosion_rate": 0.0,
                                    }
                                },
                            ]
                        }
                    },
                ]
            }
        }

    def test_custom_max_size_applies_to_option_a_only(self):
        """max_size caps option_a's long side; option_b now resizes the crop directly to the target (no cap step)."""
        result = _build_train_resize_config([480, 640], square=False, max_size=1000)
        option_a = result[0]["OneOf"]["transforms"][0]
        option_b_steps = result[0]["OneOf"]["transforms"][1]["Sequential"]["transforms"]
        assert option_a["Sequential"]["transforms"][1] == {"CappedLongestMaxSize": {"max_size": 1000}}
        assert not any("LongestMaxSize" in step for step in option_b_steps)


class TestBuildTrainResizeConfigNonSquareCropSafety:
    """非 square 路径 Option B 的裁剪必须使用 ``RandomSizedBBoxSafeCrop`` 且 ``erosion_rate=0.0``。

    历史回归背景（https://github.com/roboflow/rf-detr/issues/1018，PR #752）：早期曾把带尺度抖动的
    ``RandomSizeCrop(384, 600)`` 换成固定 ``RandomCrop(384, 384)``，静默移除了训练尺度多样性。本次进一步把会随机
    切断目标框的 ``RandomSizedCrop`` 换成 ``RandomSizedBBoxSafeCrop``：裁窗始终包含所有框的并集，任何随机种子下
    都不切框；裁剪尺度多样性改由 ``SmallestMaxSize`` 400/500/600 的 zoom-out 与框并集驱动的裁窗留白提供。
    """

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
    )
    def test_option_b_crop_step_uses_bbox_safe_crop(self, scales):
        """非 square 的 Option B 裁剪必须是 RandomSizedBBoxSafeCrop，且 erosion_rate 恒为 0.0。"""
        result = _build_train_resize_config(scales, square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        crop_step = option_b["Sequential"]["transforms"][1]
        crop_variants = crop_step["OneOf"]["transforms"]
        assert crop_variants
        for entry in crop_variants:
            assert list(entry) == ["RandomSizedBBoxSafeCrop"]
            assert entry["RandomSizedBBoxSafeCrop"]["erosion_rate"] == 0.0

    @pytest.mark.parametrize(
        "scales",
        [
            pytest.param([640], id="nonsquare-single"),
            pytest.param([480, 640], id="nonsquare-multi"),
        ],
    )
    def test_option_b_crop_resizes_to_each_target_scale(self, scales):
        """每个 scale 都有对应的裁剪分支，裁剪输出统一缩放到 (scale, scale)。"""
        result = _build_train_resize_config(scales, square=False)
        option_b = result[0]["OneOf"]["transforms"][1]
        crop_variants = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        assert {entry["RandomSizedBBoxSafeCrop"]["height"] for entry in crop_variants} == set(scales)
        assert {entry["RandomSizedBBoxSafeCrop"]["width"] for entry in crop_variants} == set(scales)

    @pytest.mark.parametrize(
        "scales,square",
        [
            pytest.param([640], True, id="square-single"),
            pytest.param([480, 640], True, id="square-multi"),
        ],
    )
    def test_square_option_b_uses_bbox_safe_crop(self, scales, square):
        """Square 路径同样使用按 scale 参数化的 RandomSizedBBoxSafeCrop。"""
        result = _build_train_resize_config(scales, square=square)
        option_b = result[0]["OneOf"]["transforms"][1]
        inner_transforms = option_b["Sequential"]["transforms"][1]["OneOf"]["transforms"]
        assert len(inner_transforms) == len(scales)
        for entry in inner_transforms:
            assert list(entry) == ["RandomSizedBBoxSafeCrop"]
            assert entry["RandomSizedBBoxSafeCrop"]["erosion_rate"] == 0.0


class TestBuildTrainResizeConfigScaleJitter:
    """scale_jitter=False drops Option B so only the direct-resize branch (Option A) runs."""

    @pytest.mark.parametrize(
        "square",
        [pytest.param(True, id="square"), pytest.param(False, id="nonsquare")],
    )
    def test_scale_jitter_false_drops_crop_branch(self, square):
        """No crop transform anywhere in result when scale jitter is disabled."""
        import json

        result = _build_train_resize_config([480, 640], square=square, scale_jitter=False)
        assert len(result) == 1
        dump = json.dumps(result)
        assert "RandomSizedCrop" not in dump and "BBoxSafe" not in dump

    @pytest.mark.parametrize(
        "square",
        [pytest.param(True, id="square"), pytest.param(False, id="nonsquare")],
    )
    def test_scale_jitter_true_produces_one_of_with_crop(self, square):
        """Default (scale_jitter=True) outer entry is a two-branch OneOf wrapping option_b with crop."""
        import json

        result = _build_train_resize_config([480, 640], square=square, scale_jitter=True)
        assert len(result) == 1
        assert "OneOf" in result[0]
        assert "RandomSizedBBoxSafeCrop" in json.dumps(result)


class TestCappedLongestMaxSizeRuntimeBehavior:
    """CappedLongestMaxSize only shrinks -- never upscales -- unlike plain LongestMaxSize.

    Regression tests: chaining SmallestMaxSize(resolution) -> LongestMaxSize(cap) previously forced every
    non-square image's longest side up to `cap` (e.g. 1333) regardless of the requested `resolution`, since
    Albumentations' LongestMaxSize always resizes to exactly `max_size`, up or down. CappedLongestMaxSize clamps
    the resolved scale to <= 1.0 so it behaves as a true cap, matching torchvision RandomResize's conditional
    max_size semantics.
    """

    @staticmethod
    def _resize_output_size(image_hw, resolution, max_size, scale_jitter=False):
        import numpy as np
        import torch
        from PIL import Image

        from rfdetr.datasets.coco import _build_train_resize_config
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        config = _build_train_resize_config([resolution], square=False, max_size=max_size, scale_jitter=scale_jitter)
        wrapper = AlbumentationsWrapper.from_config(config)[0]
        image = Image.fromarray(np.zeros((*image_hw, 3), dtype=np.uint8))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        out_image, _ = wrapper(image, target)
        return out_image.size  # (width, height)

    def test_does_not_upscale_when_already_within_cap(self):
        """A resize that already fits under max_size is left alone, not forced up to max_size."""
        width, height = self._resize_output_size((90, 120), resolution=640, max_size=1333)
        assert max(width, height) < 1333, "longest side must not be inflated to the cap"
        assert (width, height) == (853, 640), "SmallestMaxSize(640) result must pass through unchanged"

    def test_still_caps_when_resize_would_exceed_max_size(self):
        """An extreme aspect ratio that would exceed max_size after SmallestMaxSize is still capped."""
        width, height = self._resize_output_size((100, 3000), resolution=640, max_size=1000)
        assert max(width, height) == 1000, "longest side must be capped at max_size when it would be exceeded"


class TestOptionBCropNeverCutsBoxes:
    """Option B 的裁剪在任意随机种子下都不会切断任何框（回归测试）。

    把 ``RandomSizedCrop`` 换成 ``RandomSizedBBoxSafeCrop(erosion_rate=0.0)`` 后，裁窗始终包含所有框的并集
    （裁剪区域 = 并集 + 四边独立随机留白，是构造性算法而非拒绝采样）。逐样本断言两条不变量：
    1) 框数量不变——没有任何框被整框裁掉；
    2) 所有框经历同一缩放（面积比例一致）——若某框被切掉一部分，其面积比例会明显小于其他框。

    覆盖两类框分布：大框主导（占画面 85% 以上，即航母型目标）与小框居中聚拢（裁剪多样性应保留的场景）。
    """

    @staticmethod
    def _build_option_b_wrapper():
        """返回只包含 Option B 的 Albumentations 包装器（强制走裁剪分支，跳过 Option A）。"""
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        config = _build_train_resize_config([640], square=True)
        option_b = config[0]["OneOf"]["transforms"][1]
        return AlbumentationsWrapper.from_config([option_b], strict=True)[0]

    @pytest.mark.parametrize(
        "boxes",
        [
            pytest.param(
                [[60, 40, 940, 660], [820, 620, 850, 645], [100, 320, 112, 332]],
                id="large-box-dominant",
            ),
            pytest.param(
                [[440, 300, 470, 330], [500, 330, 530, 360], [460, 370, 490, 400]],
                id="small-boxes-clustered",
            ),
        ],
    )
    def test_no_box_is_cut_across_seeds(self, boxes):
        """多次随机采样后：框数不变、输出为 (640, 640)、所有框面积缩放比例一致（无框被切）。"""
        import numpy as np
        import torch
        from PIL import Image

        wrapper = self._build_option_b_wrapper()
        # 1000x700 输入：SmallestMaxSize 先缩到短边 400/500/600，再走安全裁剪
        image = Image.fromarray((np.random.default_rng(0).random((700, 1000, 3)) * 255).astype(np.uint8))
        in_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        in_area = (in_boxes[:, 2] - in_boxes[:, 0]) * (in_boxes[:, 3] - in_boxes[:, 1])
        target = {"boxes": in_boxes, "labels": torch.zeros(len(boxes), dtype=torch.long)}

        for _ in range(100):
            out_image, out_target = wrapper(image, target)
            assert out_image.size == (640, 640), "裁剪分支必须输出 (640, 640) 正方形"
            out_boxes = out_target["boxes"]
            assert len(out_boxes) == len(boxes), "裁剪后框数量发生变化，说明有框被切掉"
            out_area = (out_boxes[:, 2] - out_boxes[:, 0]) * (out_boxes[:, 3] - out_boxes[:, 1])
            ratios = out_area / in_area
            assert ratios.max() / ratios.min() < 1.01, f"存在被部分裁掉的框: ratios={ratios}"

    def test_no_boxes_falls_back_to_plain_crop(self):
        """无框图走随机裁剪分支，不崩溃且输出尺寸正确。"""
        import numpy as np
        import torch
        from PIL import Image

        wrapper = self._build_option_b_wrapper()
        image = Image.fromarray((np.random.default_rng(0).random((700, 1000, 3)) * 255).astype(np.uint8))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        out_image, out_target = wrapper(image, target)
        assert out_image.size == (640, 640)
        assert out_target["boxes"].shape == (0, 4)
