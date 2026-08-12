# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本 SSCL 回调的单元测试。

不依赖 GPU 与网络，验证 RFDETRModelModule 中难例负样本的接线：
- 训练态启用时：难例被选择进分母、监控被喂入、原型库只收 matched 特征。
- 禁用/验证态：难例不选择、监控不喂入、行为与基线一致。
- 监控节流（global_step % interval）。
- TrainConfig 校验（难例依赖原型模式、top-k >= 1）。
- 实例模式忽略难例（无原型库、监控不喂入）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rfdetr.config import TrainConfig
from rfdetr.sscl import HardNegMonitor, SSCLLoss
from rfdetr.training.module_model import RFDETRModelModule

# 5 类测试用语义相似度矩阵（与 test_sscl_prototype_callback.py 一致）
_SEMANTIC_MATRIX = torch.tensor(
    [
        [1.0, 0.7, 0.5, 0.3, 0.1],
        [0.7, 1.0, 0.5, 0.3, 0.1],
        [0.5, 0.5, 1.0, 0.3, 0.1],
        [0.3, 0.3, 0.3, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ]
)

# 单图单 GT：query 0 匹配到 GT（类别 2），query 3 为带内难例候选（IoU 0.25）
_GT_BOXES = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
_QUERY_BOXES = torch.tensor(
    [
        [0.5, 0.5, 0.2, 0.2],  # q0: matched（IoU 1.0）
        [0.1, 0.1, 0.05, 0.05],  # q1: 背景（IoU 0）
        [0.9, 0.9, 0.05, 0.05],  # q2: 背景（IoU 0）
        [0.5, 0.5, 0.1, 0.1],  # q3: 难例带内（IoU 0.25），最高前景分
        [0.1, 0.9, 0.05, 0.05],  # q4: 背景（IoU 0）
        [0.9, 0.1, 0.05, 0.05],  # q5: 背景（IoU 0）
    ]
)


class _FakeCriterion:
    """镜像 RF-DETR criterion._get_src_permutation_idx 的最小实现。"""

    @staticmethod
    def _get_src_permutation_idx(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (batch_idx, src_idx)，与 criterion.py 语义一致。"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def _make_config(**overrides: object) -> SimpleNamespace:
    """构造训练配置的最小替身（仅含难例相关字段）。"""
    kwargs: dict[str, object] = {
        "sscl_hard_neg_topk": 3,
        "sscl_hard_neg_score_thresh": 0.0,
        "sscl_hard_neg_log_interval": 100,
    }
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _build_module(
    training: bool,
    prototype_mode: bool = True,
    hard_neg_enabled: bool = False,
    global_step: int = 0,
) -> RFDETRModelModule:
    """用 __new__ 绕过重型 __init__，构造最小可测模块。

    与 test_sscl_prototype_callback.py 的 harness 一致；难例启用时额外挂监控与配置。
    """
    module = RFDETRModelModule.__new__(RFDETRModelModule)
    module.criterion = _FakeCriterion()
    object.__setattr__(
        module,
        "sscl_loss",
        SSCLLoss(
            semantic_matrix=_SEMANTIC_MATRIX,
            prototype_mode=prototype_mode,
            hidden_dim=16,
            projection_dim=8,
        ),
    )
    module.training = training
    # LightningModule.global_step 是只读属性：getter 依次访问 _trainer/_fabric/
    # _jit_is_scripting（__new__ 绕过 __init__ 后全部缺失，须补齐）
    object.__setattr__(module, "_trainer", SimpleNamespace(global_step=global_step))
    object.__setattr__(module, "_fabric", None)
    object.__setattr__(module, "_jit_is_scripting", False)
    if hard_neg_enabled:
        module.train_config = _make_config()
        object.__setattr__(module, "_hard_neg_monitor", HardNegMonitor())
    return module


def _make_outputs() -> dict[str, torch.Tensor]:
    """构造模型输出：q3 为带内难例候选（IoU 0.25）且前景分最高。"""
    generator = torch.Generator().manual_seed(0)
    hs = torch.randn(1, 6, 16, generator=generator)
    pred_logits = torch.full((1, 6, 3), 0.1)
    pred_logits[0, 3, 0] = 5.0  # q3 前景分最高
    pred_logits[:, :, -1] = 9.0  # background 列高分（必须被忽略）
    return {
        "hs": hs,
        "pred_logits": pred_logits,
        "pred_boxes": _QUERY_BOXES.unsqueeze(0),
    }


def _warm_bank_for_matched(module: RFDETRModelModule) -> None:
    """为 matched 类别 2 建立原型（否则首个训练步损失为 0，断言无意义）。"""
    generator = torch.Generator().manual_seed(1)
    module.sscl_loss.update_prototypes(torch.randn(2, 16, generator=generator), torch.tensor([2, 2]))


class TestSSCLHardNegCallback:
    """难例负样本回调的接线行为。"""

    def test_callback_selects_and_appends(self) -> None:
        """训练态 + 启用：难例被选择进分母、监控被喂入、原型库只收 matched。"""
        module = _build_module(training=True, hard_neg_enabled=True, global_step=0)
        _warm_bank_for_matched(module)
        outputs = _make_outputs()
        targets = [{"labels": torch.tensor([2]), "boxes": _GT_BOXES}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]  # q0 ↔ GT0
        before = module.sscl_loss.prototype_bank.num_updates.clone()

        result = module._sscl_loss_callback(outputs, targets, indices)
        # 返回字典只含 loss_sscl（不污染 loss_dict）
        assert set(result) == {"loss_sscl"}
        assert torch.isfinite(result["loss_sscl"])
        # 监控被喂入（global_step=0 命中 interval=100），含难例统计键
        monitor = module._hard_neg_monitor
        assert monitor._step_count == 1
        assert "hn_count" in monitor._acc
        assert "hn_proto_cos" in monitor._acc
        # 原型库只收 matched 类别：仅类 2 更新一次，其他类别（含难例）绝无更新
        num_updates = module.sscl_loss.prototype_bank.num_updates
        assert int(num_updates[2].item()) == int(before[2].item()) + 1
        others = num_updates.clone()
        others[2] = 0
        assert int(others.sum().item()) == 0

    def test_callback_disabled_no_change(self) -> None:
        """禁用难例时损失与不启用完全一致（基线行为不变）。"""
        module = _build_module(training=True, hard_neg_enabled=False)
        _warm_bank_for_matched(module)
        outputs = _make_outputs()
        targets = [{"labels": torch.tensor([2]), "boxes": _GT_BOXES}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        # 基线 = 直接调损失（无难例参数）
        features, labels = module._extract_matched_query_features(outputs["hs"], indices, targets)
        baseline = module.sscl_loss(features, labels)

        result = module._sscl_loss_callback(outputs, targets, indices)
        assert set(result) == {"loss_sscl"}
        assert torch.allclose(result["loss_sscl"], baseline)
        assert not hasattr(module, "_hard_neg_monitor")

    def test_callback_validation_mode_no_selection(self) -> None:
        """验证态：不选难例、监控不喂入、原型库不更新。"""
        module = _build_module(training=False, hard_neg_enabled=True, global_step=0)
        _warm_bank_for_matched(module)
        outputs = _make_outputs()
        targets = [{"labels": torch.tensor([2]), "boxes": _GT_BOXES}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        before = module.sscl_loss.prototype_bank.num_updates.clone()

        result = module._sscl_loss_callback(outputs, targets, indices)
        assert set(result) == {"loss_sscl"}
        assert module._hard_neg_monitor._step_count == 0
        assert torch.equal(module.sscl_loss.prototype_bank.num_updates, before)

    def test_callback_monitor_throttle(self) -> None:
        """监控按 global_step % interval 节流：步 50 不喂、步 100 喂。"""
        module = _build_module(training=True, hard_neg_enabled=True, global_step=50)
        _warm_bank_for_matched(module)
        outputs = _make_outputs()
        targets = [{"labels": torch.tensor([2]), "boxes": _GT_BOXES}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        module._sscl_loss_callback(outputs, targets, indices)
        assert module._hard_neg_monitor._step_count == 0  # 50 % 100 != 0

        module._trainer.global_step = 100
        module._sscl_loss_callback(outputs, targets, indices)
        assert module._hard_neg_monitor._step_count == 1  # 100 % 100 == 0

    def test_callback_instance_mode_no_bank_no_hn(self) -> None:
        """实例模式 + 启用难例：不崩溃、无原型库、监控不喂入（hardness_stats 为空）。"""
        module = _build_module(training=True, prototype_mode=False, hard_neg_enabled=True, global_step=0)
        outputs = _make_outputs()
        targets = [{"labels": torch.tensor([2]), "boxes": _GT_BOXES}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        result = module._sscl_loss_callback(outputs, targets, indices)
        assert set(result) == {"loss_sscl"}
        assert torch.isfinite(result["loss_sscl"])
        assert not hasattr(module.sscl_loss, "prototype_bank")
        assert module._hard_neg_monitor._step_count == 0


class TestHardNegConfigValidation:
    """TrainConfig 难例字段的校验规则。"""

    def test_config_requires_prototype_mode(self) -> None:
        """未启用原型模式时启用难例应报错。"""
        with pytest.raises(ValueError, match="sscl_prototype_enabled"):
            TrainConfig(
                dataset_dir="/tmp/dummy",
                sscl_hard_neg_enabled=True,
                sscl_prototype_enabled=False,
            )

    def test_config_topk_must_be_positive(self) -> None:
        """Topk < 1 应报错。"""
        with pytest.raises(ValueError, match="sscl_hard_neg_topk"):
            TrainConfig(
                dataset_dir="/tmp/dummy",
                sscl_hard_neg_enabled=True,
                sscl_prototype_enabled=True,
                sscl_hard_neg_topk=0,
            )

    def test_config_valid_combo(self) -> None:
        """合法组合正常构造。"""
        cfg = TrainConfig(
            dataset_dir="/tmp/dummy",
            sscl_hard_neg_enabled=True,
            sscl_prototype_enabled=True,
            sscl_hard_neg_topk=5,
            sscl_hard_neg_score_thresh=0.5,
        )
        assert cfg.sscl_hard_neg_topk == 5
        assert cfg.sscl_hard_neg_score_thresh == 0.5


class TestMultiPrototypeConfigValidation:
    """TrainConfig 多 slot 原型字段的校验规则。"""

    def test_config_accepts_multislot_groups(self) -> None:
        """合法的多 slot + sibling group 配置应通过校验。"""
        cfg = TrainConfig(
            dataset_dir="/tmp/dummy",
            sscl_prototype_enabled=True,
            sscl_prototype_max_slots=2,
            sscl_prototype_multi_slot_classes=[0, 1, 2, 3],
            sscl_prototype_group_pairs=[[0, 1], [2, 3]],
            sscl_prototype_group_weight=1.5,
        )

        assert cfg.sscl_prototype_max_slots == 2
        assert cfg.sscl_prototype_group_pairs == [[0, 1], [2, 3]]

    def test_config_rejects_bad_slot_count(self) -> None:
        """slot 数必须至少为 1。"""
        with pytest.raises(ValueError, match="sscl_prototype_max_slots"):
            TrainConfig(
                dataset_dir="/tmp/dummy",
                sscl_prototype_enabled=True,
                sscl_prototype_max_slots=0,
            )

    def test_config_rejects_duplicate_group_class(self) -> None:
        """同一类别不能重复出现在多个易混组。"""
        with pytest.raises(ValueError, match="重复"):
            TrainConfig(
                dataset_dir="/tmp/dummy",
                sscl_prototype_enabled=True,
                sscl_prototype_group_pairs=[[0, 1], [1, 2]],
            )
