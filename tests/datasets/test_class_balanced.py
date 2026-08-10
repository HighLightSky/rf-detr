# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ClassBalancedDataset（平方根频率过采样）单元测试。"""

from __future__ import annotations

import contextlib
import io
import logging
import types
from typing import Any

import numpy as np
import pytest
import torch.utils.data

from rfdetr.datasets.class_balanced import ClassBalancedDataset

# ------------------------------------------------------------------------
# 测试数据构造
# N=100：图0-4 含 {0}（5 张）、图5-14 含 {1}（10 张）、图15-49 含 {0,1}（35 张）、
# 图50-99 空标注（50 张）。freq(0)=0.40、freq(1)=0.45。
# ------------------------------------------------------------------------


def _make_per_image() -> list[list[int]]:
    """返回测试用每图类别列表：图0-4 含{0}、5-14 含{1}、15-49 含{0,1}、50-99 空。"""
    per_image: list[list[int]] = []
    per_image += [[0] for _ in range(5)]
    per_image += [[1] for _ in range(10)]
    per_image += [[0, 1] for _ in range(35)]
    per_image += [[] for _ in range(50)]
    return per_image


class _FakeSV:
    """模拟 YoloDetection.sv_dataset：get_image_info(i).class_id 为类别数组。"""

    def __init__(self, per_image: list[list[int]]) -> None:
        self._per_image = per_image

    def __len__(self) -> int:
        return len(self._per_image)

    def get_image_info(self, i: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(class_id=np.array(self._per_image[i], dtype=np.int64))


class _FakeInfo(torch.utils.data.Dataset[Any]):
    """带 sv_dataset 接口的裸数据集；__getitem__ 返回自身索引便于透传断言。"""

    def __init__(
        self,
        per_image: list[list[int]],
        *,
        with_sv: bool = True,
        coco: Any = None,
        label2cat: dict[int, int] | None = None,
    ) -> None:
        self.sv_dataset = _FakeSV(per_image) if with_sv else None
        self.coco = coco
        self.label2cat = label2cat
        self.classes = ["c0", "c1"]
        self.ids = list(range(len(per_image)))
        self._per_image = per_image

    def __len__(self) -> int:
        return len(self._per_image)

    def __getitem__(self, i: int) -> int:
        return i


class TestClassBalancedDataset:
    """ClassBalancedDataset 单元测试（对齐 TestGradAccumAlignedDataset 组织方式）。"""

    def _make_wrapped(
        self, threshold: float | None, class_ids: list[int] | None = None
    ) -> tuple[ClassBalancedDataset, _FakeInfo]:
        """构造包装器与底层数据集。"""
        info = _FakeInfo(_make_per_image())
        wrapped = ClassBalancedDataset(info, threshold=threshold, class_ids=class_ids, info_dataset=info)
        return wrapped, info

    def test_repeat_factor_truncation(self):
        """Int 截断语义：threshold=1.6 时 r(0)=int(sqrt(4.0))=2、r(1)=int(sqrt(3.55))=1。"""
        wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0, 1])
        assert len(wrapped) == 140
        # threshold=1.7：r(0)=int(sqrt(4.25))=int(2.06)=2（截断不四舍五入）
        wrapped2, _ = self._make_wrapped(threshold=1.7, class_ids=[0, 1])
        assert len(wrapped2) == 140

    def test_len_is_sum_of_factors(self):
        """长度 = Σ r(I)：5×2 + 10×1 + 35×2 + 50×1 = 140。"""
        wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0, 1])
        assert len(wrapped) == 140

    def test_getitem_cumsum_mapping(self):
        """累积和定位正确，且同一图的 r 份连续占段。"""
        wrapped, info = self._make_wrapped(threshold=1.6, class_ids=[0, 1])
        # 图0-4（r=2）各占连续 2 段；图5-14（r=1）各占 1 段；图15-49（r=2）；图50-99（r=1）
        expected_r = [2] * 5 + [1] * 10 + [2] * 35 + [1] * 50
        starts: list[int] = [0]
        for r in expected_r:
            starts.append(starts[-1] + r)
        assert len(starts) - 1 == len(info)
        for i, r in enumerate(expected_r):
            for idx in range(starts[i], starts[i + 1]):
                assert wrapped[idx] == i, f"idx={idx} 应映射到图 {i}"
        # 越界检查
        assert len(wrapped) == starts[-1] == 140

    def test_max_one_floor_never_drops(self):
        """Max(1,·) 保证不丢图：t < freq 时全 r=1 并退化为直通。"""
        wrapped, info = self._make_wrapped(threshold=0.1, class_ids=[0, 1])
        assert len(wrapped) == 100
        for i in range(100):
            assert wrapped[i] == i

    def test_threshold_none_auto_derive(self):
        """threshold=None 自动推导 t = 4×max(freq(targets))：t=1.8 → r(0)=r(1)=2，len=150。"""
        wrapped, _ = self._make_wrapped(threshold=None, class_ids=[0, 1])
        assert len(wrapped) == 150
        # 白名单为空（全部类别）同样推导
        wrapped2, _ = self._make_wrapped(threshold=None, class_ids=[])
        assert len(wrapped2) == 150

    def test_whitelist_scope_max(self):
        """白名单范围：白名单=[0] 时含 {0,1} 的图只取 r(0)，不含 r(1)。"""
        wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0])
        assert len(wrapped) == 140

    def test_ids_expanded_mapping(self):
        """Ids 展开映射：长度与 len 一致，且逐 idx 指向正确的底层 image_id。"""
        wrapped, info = self._make_wrapped(threshold=1.6, class_ids=[0, 1])
        assert len(wrapped.ids) == len(wrapped)
        expected_r = [2] * 5 + [1] * 10 + [2] * 35 + [1] * 50
        for i, r in enumerate(expected_r):
            for _ in range(r):
                pass
        # 逐段抽查：图 i 的 r 份 ids 都等于 base_ids[i]
        pos = 0
        for i, r in enumerate(expected_r):
            for _ in range(r):
                assert wrapped.ids[pos] == info.ids[i]
                pos += 1

    def test_attribute_delegation(self):
        """属性委托：classes/coco 等来自 info 数据集。"""
        info = _FakeInfo(_make_per_image(), coco="fake_coco")
        wrapped = ClassBalancedDataset(info, threshold=1.6, class_ids=[0, 1], info_dataset=info)
        assert wrapped.classes == ["c0", "c1"]
        assert wrapped.coco == "fake_coco"

    @staticmethod
    @contextlib.contextmanager
    def _capture_logs() -> "io.StringIO":
        """给项目 logger 挂 StringIO handler 捕获日志（logger propagate=False，caplog/capsys 均无效）。"""
        from rfdetr.utilities.logger import get_logger

        logger = get_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            yield stream
        finally:
            logger.removeHandler(handler)

    def test_pass_through_no_matching_images(self):
        """白名单类不存在 → 告警并退化为直通。"""
        with self._capture_logs() as stream:
            wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[99])
        assert len(wrapped) == 100
        assert wrapped[7] == 7
        assert "退化为直通" in stream.getvalue()

    def test_pass_through_partially_missing_whitelist(self):
        """部分白名单缺失（[0,99]）：99 被丢弃告警，{0} 正常过采样。"""
        with self._capture_logs() as stream:
            wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0, 99])
        assert len(wrapped) == 140
        assert "99" in stream.getvalue()

    def test_pass_through_empty_annotations(self):
        """全空标注：不抛 ZeroDivisionError，退化为直通。"""
        info = _FakeInfo([[] for _ in range(50)])
        wrapped = ClassBalancedDataset(info, threshold=1.6, class_ids=[0, 1], info_dataset=info)
        assert len(wrapped) == 50
        assert wrapped[3] == 3

    def test_pass_through_all_factor_one(self):
        """所有目标类 r=1 → 退化为直通。"""
        wrapped, info = self._make_wrapped(threshold=1.6, class_ids=[1])
        # 白名单=[1]：r(1)=int(sqrt(1.6/0.45))=1，全部 r=1
        assert len(wrapped) == 100
        assert wrapped[42] == 42

    @pytest.mark.parametrize(
        "threshold",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param(float("nan"), id="nan"),
        ],
    )
    def test_invalid_threshold_raises(self, threshold: float):
        """非法 threshold（0/负/NaN）抛 ValueError，且不扫描即抛。"""
        info = _FakeInfo(_make_per_image())
        with pytest.raises(ValueError, match="threshold"):
            ClassBalancedDataset(info, threshold=threshold, class_ids=[0, 1], info_dataset=info)

    def test_invalid_index_raises(self):
        """越界索引抛 IndexError。"""
        wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0, 1])
        with pytest.raises(IndexError):
            wrapped[-1]
        with pytest.raises(IndexError):
            wrapped[len(wrapped)]

    def test_coco_fallback_path(self):
        """无 sv_dataset 时走 coco+ids+label2cat 回退路径。"""
        per_image = [[0]] * 5 + [[1]] * 10 + [[]] * 85

        class _FakeCoco:
            def __init__(self) -> None:
                self._anns: dict[int, list[dict[str, int]]] = {}
                for i, cats in enumerate(per_image):
                    self._anns[i] = [{"category_id": c} for c in cats]

            def getAnnIds(self, image_id: int) -> list[int]:  # noqa: N802
                return [image_id] if self._anns[image_id] else []

            def loadAnns(self, ids: list[int]) -> list[dict[str, int]]:  # noqa: N802
                return self._anns[ids[0]] if ids else []

        info = _FakeInfo(per_image, with_sv=False, coco=_FakeCoco(), label2cat={0: 0, 1: 1})
        wrapped = ClassBalancedDataset(info, threshold=1.6, class_ids=[0, 1], info_dataset=info)
        # freq(0)=0.05、freq(1)=0.10：r(0)=int(sqrt(1.6/0.05))=int(5.66)=5、r(1)=int(sqrt(16))=4
        assert len(wrapped) == 5 * 5 + 10 * 4 + 85 * 1 == 150

    def test_class_ids_out_of_range_not_raise(self):
        """越界 class_ids 只告警不抛，白名单中存在的类正常过采样。"""
        wrapped, _ = self._make_wrapped(threshold=1.6, class_ids=[0, 500])
        assert len(wrapped) == 140


class TestClassBalancedDatasetDegradedPassthrough:
    """退化直通模式下 __getitem__ 与 ids 的行为。"""

    def test_passthrough_ids_delegates_to_info(self):
        """Inactive 时 ids 委托真实 info.ids 而非 range。"""
        info = _FakeInfo(_make_per_image())
        wrapped = ClassBalancedDataset(info, threshold=0.1, class_ids=[0, 1], info_dataset=info)
        assert len(wrapped) == 100
        assert wrapped.ids == info.ids
