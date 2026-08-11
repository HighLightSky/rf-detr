# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for SetCriterion edge paths: _output_device and num_boxes_for_targets."""

import pytest
import torch
from torch.nn.functional import logsigmoid

from rfdetr.models.criterion import SetCriterion


class _MatcherStub:
    """Minimal matcher that returns identity indices for every target in the batch."""

    def __call__(self, outputs, targets, group_detr=1):
        return [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]


def _bare_criterion() -> SetCriterion:
    """Return a SetCriterion with no losses so forward() is a no-op."""
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.training = True
    criterion.group_detr = 1
    criterion.sum_group_losses = False
    criterion.losses = []
    criterion.weight_dict = {}
    criterion.matcher = _MatcherStub()
    criterion.num_keypoints_per_class = []
    return criterion


class TestOutputDevice:
    """Tests for SetCriterion._output_device — probes top-level tensor values only."""

    def test_returns_device_of_first_tensor(self):
        """Device inferred from the first tensor value in outputs."""
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")

    def test_raises_when_no_tensor_present(self):
        """ValueError raised when no top-level value is a tensor."""
        outputs = {"meta": "string_value", "count": 42}

        with pytest.raises(ValueError, match="at least one tensor"):
            SetCriterion._output_device(outputs)

    def test_skips_non_tensor_values(self):
        """Non-tensor entries at the top level are skipped; first tensor wins."""
        outputs = {"meta": "ignored", "pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")


class TestNumBoxesForTargets:
    """Tests for SetCriterion.num_boxes_for_targets — clamp and empty-target edge cases."""

    def test_returns_tensor_gte_one(self):
        """Result must be clamped to >= 1.0 to prevent division by zero."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.tensor([0, 1])}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() >= 1.0

    def test_clamps_zero_box_count_to_one(self):
        """Empty targets (no labels) must clamp to 1.0 to avoid zero denominator."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.zeros(0, dtype=torch.int64)}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_clamps_empty_target_list(self):
        """Empty target list (batch_size=0 edge case) must also clamp to 1.0."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = []

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_counts_labels_correctly(self):
        """Box count equals total number of labels across all targets in the batch."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [
            {"labels": torch.tensor([0, 1])},
            {"labels": torch.tensor([0])},
        ]

        result = criterion.num_boxes_for_targets(outputs, targets)

        # 2 + 1 = 3 boxes; single-process so no all-reduce
        assert result.item() == pytest.approx(3.0)


class TestClassBalanceBuffers:
    """测试 P0/P1 预计算 buffer（类别权重与 logit bias）的数值正确性。"""

    @staticmethod
    def _build_criterion(**kwargs) -> SetCriterion:
        """构造一个 ia_bce 模式、仅含 labels 损失的 SetCriterion。"""
        return SetCriterion(
            num_classes=3,
            matcher=_MatcherStub(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["labels"],
            ia_bce_loss=True,
            **kwargs,
        )

    @staticmethod
    def _manual_ia_bce(logits: torch.Tensor, pos_w_mult: float = 1.0) -> torch.Tensor:
        """手算 IA-BCE 损失：query 0 匹配 GT（IoU=1），正样本 slot (0,0,0)。

        与 criterion.py loss_labels 的公式逐项一致：
        t = clamp(p^alpha * iou^(1-alpha), 0.01)；pos slot 权重 t（P0 时再乘 w）；
        其余元素负权重 p^gamma。
        """
        p = logits.sigmoid()
        t = torch.clamp(p[0, 0, 0].pow(0.25) * 1.0, 0.01)
        pw = t * pos_w_mult
        nw = 1 - t
        # 正样本 slot 的损失项
        total = nw * logits[0, 0, 0] - logsigmoid(logits[0, 0, 0]) * (pw + nw)
        # 其余 5 个元素（背景/负样本），负权重 p^gamma，gamma=2
        for b in range(logits.shape[0]):
            for q in range(logits.shape[1]):
                for c in range(logits.shape[2]):
                    if (b, q, c) == (0, 0, 0):
                        continue
                    nw_e = p[b, q, c] ** 2
                    total = total + nw_e * logits[b, q, c] - logsigmoid(logits[b, q, c]) * nw_e
        return total

    @staticmethod
    def _make_inputs() -> tuple[dict, list[dict]]:
        """构造单图单 GT 的最小输入：query 0 匹配 GT，query 1 为 unmatched。"""
        logits = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]])
        boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]]])
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5]])}]
        return outputs, targets

    def test_disabled_by_default_is_regression(self):
        """默认（不传均衡参数）时 loss 与基线完全一致（回归保护）。"""
        logits = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]])
        boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]]])
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5]])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        plain = self._build_criterion()
        balanced = self._build_criterion(
            class_balance_enabled=True,
            class_balance_counts=torch.tensor([10.0, 10.0, 10.0]),
            class_balance_ref_count=10.0,
            class_balance_min_count=1,
        )
        # counts 全相等 → 权重全 1，等价未启用
        loss_plain = plain.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        loss_balanced = balanced.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]

        assert loss_balanced == pytest.approx(loss_plain.item(), abs=1e-6)

    def test_p0_scales_only_positive_slot(self):
        """P0：w_c=2 时正样本 slot 的损失贡献精确 ×2 结构（负样本项不变），与手算一致。"""
        outputs, targets = self._make_inputs()
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        criterion = self._build_criterion(
            class_balance_enabled=True,
            class_balance_counts=torch.tensor([10.0, 40.0, 40.0]),
            class_balance_ref_count=40.0,  # w_0 = (40/max(10,10))^0.5 = 2.0
            class_balance_beta=0.5,
            class_balance_min_count=10,
            class_balance_target_classes=[0],
        )
        loss = criterion.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        manual = self._manual_ia_bce(outputs["pred_logits"], pos_w_mult=2.0)

        assert loss.item() == pytest.approx(manual.item(), abs=1e-6)

    def test_p0_target_classes_restricts_weights(self):
        """P0：非 target_classes 的类别权重固定 1.0（多数类不被降权）。"""
        counts = torch.tensor([6.0, 15.0, 443.0])
        weights, _ = SetCriterion._build_class_balance_buffers(
            counts=counts,
            beta=0.25,
            max_weight=3.0,
            min_count=10,
            ref_count=None,  # sqrt(443*6) ≈ 51.6
            target_classes=[0, 1],
            tau=0.1,
            bias_clip=1.0,
        )

        assert weights[0].item() == pytest.approx((51.56 / 10.0) ** 0.25, rel=1e-3)
        assert weights[1].item() == pytest.approx((51.56 / 15.0) ** 0.25, rel=1e-3)
        assert weights[2].item() == pytest.approx(1.0)

    def test_zero_count_classes_do_not_zero_ref_count(self):
        """P0：零样本类别不应让 N_ref 退化为 0，非目标类仍固定 1.0。"""
        weights, _ = SetCriterion._build_class_balance_buffers(
            counts=torch.tensor([0.0, 5.0, 20.0]),
            beta=0.5,
            max_weight=5.0,
            min_count=1,
            ref_count=None,  # sqrt(20*5) = 10，忽略零样本类别
            target_classes=[1],
            tau=0.1,
            bias_clip=1.0,
        )

        assert weights[0].item() == pytest.approx(1.0)
        assert weights[1].item() == pytest.approx((10.0 / 5.0) ** 0.5)
        assert weights[2].item() == pytest.approx(1.0)

    def test_all_zero_counts_raise(self):
        """P0/P1：全零类别统计应显式报错，避免静默生成无效权重。"""
        with pytest.raises(ValueError, match="至少需要一个正样本类别"):
            SetCriterion._build_class_balance_buffers(
                counts=torch.tensor([0.0, 0.0, 0.0]),
                beta=0.25,
                max_weight=3.0,
                min_count=10,
                ref_count=None,
                target_classes=None,
                tau=0.1,
                bias_clip=1.0,
            )

    def test_p1_bias_shifts_logits_exactly(self):
        """P1：warmup=1 时 loss 等于把 logits 平移 bias 后算原损失；warmup=0 时等于原损失。"""
        outputs, targets = self._make_inputs()
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        counts = torch.tensor([10.0, 40.0, 50.0])  # 不均衡 → bias 非零
        la_kwargs = dict(
            class_balance_counts=counts,
            logit_adjustment_enabled=True,
            logit_adjustment_tau=0.5,
            logit_adjustment_bias_clip=1.0,
        )
        # 取 bias 值的参照：直接由静态方法算出
        _, bias = SetCriterion._build_class_balance_buffers(
            counts=counts,
            beta=0.25,
            max_weight=3.0,
            min_count=10,
            ref_count=None,
            target_classes=None,
            tau=0.5,
            bias_clip=1.0,
        )
        plain = self._build_criterion()
        la = self._build_criterion(**la_kwargs)

        # warmup=0 → 与基线一致
        la.set_la_warmup_factor(0.0)
        loss_warmup0 = la.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        loss_plain = plain.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        assert loss_warmup0.item() == pytest.approx(loss_plain.item(), abs=1e-6)

        # warmup=1 → 等价于 logits+bias 上的原损失（且推理输出未被污染）
        la.set_la_warmup_factor(1.0)
        loss_full = la.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        shifted = outputs["pred_logits"] + bias
        manual = self._manual_ia_bce(shifted)
        assert loss_full.item() == pytest.approx(manual.item(), abs=1e-6)
        # 污染保护：loss_labels 不得原地修改 outputs["pred_logits"]
        assert torch.equal(outputs["pred_logits"], torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]]))

    def test_p0_p1_combined_matches_manual(self):
        """P0+P1 叠加：正样本 slot 权重 w 与 logit bias 同时生效。"""
        outputs, targets = self._make_inputs()
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        counts = torch.tensor([10.0, 40.0, 50.0])
        criterion = self._build_criterion(
            class_balance_enabled=True,
            class_balance_counts=counts,
            class_balance_ref_count=40.0,
            class_balance_beta=0.5,
            class_balance_min_count=10,
            class_balance_target_classes=[0],
            logit_adjustment_enabled=True,
            logit_adjustment_tau=0.5,
            logit_adjustment_bias_clip=1.0,
        )
        _, bias = SetCriterion._build_class_balance_buffers(
            counts=counts,
            beta=0.5,
            max_weight=3.0,
            min_count=10,
            ref_count=40.0,
            target_classes=[0],
            tau=0.5,
            bias_clip=1.0,
        )
        criterion.set_la_warmup_factor(1.0)
        loss = criterion.loss_labels(outputs, targets, indices, num_boxes=1.0)["loss_ce"]
        # 手动：先平移 logits（P1），正样本 slot 权重再乘 w_0=2（P0）
        manual = self._manual_ia_bce(outputs["pred_logits"] + bias, pos_w_mult=2.0)

        assert loss.item() == pytest.approx(manual.item(), abs=1e-6)

    @pytest.mark.gpu
    def test_buffers_move_with_device(self):
        """Buffer 随 .to(device) 迁移，且与 num_classes 兼容。"""
        criterion = self._build_criterion(
            class_balance_enabled=True,
            class_balance_counts=torch.tensor([6.0, 15.0, 443.0]),
        ).to("cuda")

        assert criterion.class_balance_weights.is_cuda
        assert criterion.logit_bias.is_cuda
        assert criterion.class_balance_weights.numel() == 3


class TestLossMasksEmptyMatch:
    """Tests for the dict-path zero-GT branch of SetCriterion.loss_masks."""

    def test_dict_path_zero_gt_stays_connected_to_graph(self):
        """Zero-match dict path returns a loss that back-propagates to every segmentation-head output."""
        criterion = _bare_criterion()
        spatial_features = torch.randn(1, 4, 8, 8, requires_grad=True)
        query_features = torch.randn(1, 5, 4, requires_grad=True)
        bias = torch.randn(1, requires_grad=True)
        outputs = {
            "pred_masks": {
                "spatial_features": spatial_features,
                "query_features": query_features,
                "bias": bias,
            }
        }
        empty = torch.empty(0, dtype=torch.long)
        indices = [(empty, empty)]

        losses = criterion.loss_masks(outputs, targets=[{}], indices=indices, num_boxes=1)

        assert losses["loss_mask_ce"].requires_grad
        (losses["loss_mask_ce"] + losses["loss_mask_dice"]).backward()
        assert spatial_features.grad is not None
        assert query_features.grad is not None
        assert bias.grad is not None
