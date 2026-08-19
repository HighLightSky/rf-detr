# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""loss_proto_labels（原型分类辅助损失）单元测试。

验证：
- 无 ``pred_proto_logits`` 键时返回空字典（aux/主层自动跳过）。
- 输出形状/键名正确、损失有限。
- 梯度回传到 ``pred_proto_logits``（余弦打分分支的接收端）。
"""

from __future__ import annotations

import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher


def _make_criterion(num_classes: int = 5) -> SetCriterion:
    """构造最小 SetCriterion（与 builder 的参数约定一致）。"""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
    )


def _make_inputs(
    num_classes: int = 5, q_len: int = 8, bs: int = 2
) -> tuple[dict, list[dict], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """构造 outputs/targets/indices/num_boxes 四元组。

    Returns:
        ``(outputs, targets, indices, num_boxes)``：outputs 含
        ``pred_proto_logits`` ``[bs, q_len, C]``。
    """
    generator = torch.manual_seed(0)
    num_boxes = 0
    targets: list[dict] = []
    indices: list[tuple[torch.Tensor, torch.Tensor]] = []
    for img_idx in range(bs):
        n_gt = 2
        labels = torch.randint(0, num_classes, (n_gt,))
        boxes = torch.rand(n_gt, 4)
        targets.append({"labels": labels, "boxes": boxes})
        num_boxes += n_gt
        indices.append(
            (
                torch.arange(2 * img_idx, 2 * img_idx + n_gt),  # src 索引（互不重叠）
                torch.arange(n_gt),
            )
        )
    proto_logits = torch.randn(bs, q_len, num_classes, generator=generator) * 0.5
    outputs = {"pred_proto_logits": proto_logits}
    return outputs, targets, indices, torch.tensor(float(num_boxes))


class TestLossProtoLabels:
    def test_missing_key_returns_empty(self) -> None:
        """无 pred_proto_logits 键时返回空字典（aux/主层安全跳过）。"""
        criterion = _make_criterion()
        outputs, targets, indices, num_boxes = _make_inputs()
        outputs.pop("pred_proto_logits")
        result = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)
        assert result == {}

    def test_output_key_and_finite(self) -> None:
        """输出键名正确且损失有限。"""
        criterion = _make_criterion()
        outputs, targets, indices, num_boxes = _make_inputs()
        result = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)
        assert "loss_proto_labels" in result
        assert torch.isfinite(result["loss_proto_labels"])

    def test_gradient_flows_to_proto_logits(self) -> None:
        """损失回传到 pred_proto_logits（余弦打分分支的接收端）。"""
        criterion = _make_criterion()
        outputs, targets, indices, num_boxes = _make_inputs()
        outputs["pred_proto_logits"].requires_grad_(True)
        result = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)
        result["loss_proto_labels"].backward()
        grad = outputs["pred_proto_logits"].grad
        assert grad is not None
        assert float(grad.abs().sum()) > 0.0

    def test_loss_goes_down_with_fit(self) -> None:
        """匹配位置的 logits 向 GT 类增大时损失下降（监督方向正确）。"""
        criterion = _make_criterion()
        outputs, targets, indices, num_boxes = _make_inputs()
        q_len = outputs["pred_proto_logits"].shape[1]
        # 放大匹配位置对应类的 logits，其余压低
        with torch.no_grad():
            for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
                for src, tgt in zip(src_idx, tgt_idx, strict=False):
                    class_id = int(targets[batch_idx]["labels"][tgt])
                    outputs["pred_proto_logits"][batch_idx, src, class_id] += 5.0
                    outputs["pred_proto_logits"][batch_idx, src, :] -= 2.0
                    outputs["pred_proto_logits"][batch_idx, src, class_id] += 2.0
        loss_high = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)[
            "loss_proto_labels"
        ]
        # 全部 logits 拉平后损失应更大（判别性下降）
        outputs["pred_proto_logits"].data.normal_()
        loss_flat = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)[
            "loss_proto_labels"
        ]
        assert float(loss_high.item()) < float(loss_flat.item())
        assert q_len > 0  # 防止未使用变量告警

    def test_unmatched_background_logits_do_not_affect_loss(self) -> None:
        """辅助分类只监督 matched foreground，未匹配 query 不应主导优化。"""
        criterion = _make_criterion()
        outputs, targets, indices, num_boxes = _make_inputs()
        baseline = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)[
            "loss_proto_labels"
        ]
        matched = {
            (batch_idx, int(query_idx))
            for batch_idx, (src_idx, _) in enumerate(indices)
            for query_idx in src_idx
        }
        with torch.no_grad():
            for batch_idx in range(outputs["pred_proto_logits"].shape[0]):
                for query_idx in range(outputs["pred_proto_logits"].shape[1]):
                    if (batch_idx, query_idx) not in matched:
                        outputs["pred_proto_logits"][batch_idx, query_idx].fill_(100.0)

        changed = criterion.loss_proto_labels(outputs, targets, indices, num_boxes)[
            "loss_proto_labels"
        ]

        assert torch.allclose(changed, baseline)

    def test_loss_is_class_balanced(self) -> None:
        """重复某一类别的 matched 样本不应改变各类别等权的辅助损失。"""
        criterion = _make_criterion(num_classes=2)
        logits = torch.tensor([[[2.0, -1.0], [-0.5, 1.0], [2.0, -1.0]]])
        outputs = {"pred_proto_logits": logits}
        targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4)}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        base = criterion.loss_proto_labels(outputs, targets, indices, torch.tensor(2.0))[
            "loss_proto_labels"
        ]

        repeated_targets = [{"labels": torch.tensor([0, 1, 0]), "boxes": torch.rand(3, 4)}]
        repeated_indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]
        repeated = criterion.loss_proto_labels(
            outputs,
            repeated_targets,
            repeated_indices,
            torch.tensor(3.0),
        )["loss_proto_labels"]

        assert torch.allclose(repeated, base)


class TestLossProtoDense:
    """验证全量 encoder token 原型对齐损失。"""

    def test_missing_dense_keys_returns_empty(self) -> None:
        """未输出 dense token 时不影响现有训练路径。"""
        criterion = _make_criterion()
        result = criterion.loss_proto_dense({}, [], [], torch.tensor(1.0))
        assert result == {}

    def test_iou_positive_has_finite_loss_and_gradient(self) -> None:
        """IoU 正样本应产生有限损失并回传到 dense logits。"""
        criterion = _make_criterion(num_classes=3)
        logits = torch.tensor([[[0.0, 3.0, -1.0], [0.0, -1.0, 3.0]]], requires_grad=True)
        outputs = {
            "pred_proto_logits_dense": logits,
            "pred_proto_boxes_dense": torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]]),
            "pred_proto_scores_dense": torch.tensor([[0.9, 0.1]]),
        }
        targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]

        result = criterion.loss_proto_dense(outputs, targets, [], torch.tensor(1.0))

        assert torch.isfinite(result["loss_proto_dense"])
        assert torch.allclose(result["proto_dense_positive_count"], torch.tensor(1.0))
        result["loss_proto_dense"].backward()
        assert logits.grad is not None
        assert float(logits.grad.abs().sum()) > 0.0

    def test_center_fallback_supervises_small_object(self) -> None:
        """低 IoU 小目标应由 proposal 中心落框规则补充正样本。"""
        criterion = _make_criterion(num_classes=2)
        logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
        outputs = {
            "pred_proto_logits_dense": logits,
            "pred_proto_boxes_dense": torch.tensor([[[0.5, 0.5, 0.02, 0.02], [0.1, 0.1, 0.02, 0.02]]]),
            "pred_proto_scores_dense": torch.tensor([[0.9, 0.1]]),
        }
        targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]

        result = criterion.loss_proto_dense(outputs, targets, [], torch.tensor(1.0))

        assert torch.allclose(result["proto_dense_positive_count"], torch.tensor(1.0))
        assert float(result["proto_dense_accuracy"]) == 1.0

    def test_loss_is_class_balanced(self) -> None:
        """重复同类 dense 正样本不应改变各类别等权的损失。"""
        criterion = _make_criterion(num_classes=2)
        logits = torch.tensor([[[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]]])
        targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])}]
        base_outputs = {
            "pred_proto_logits_dense": logits[:, :2],
            "pred_proto_boxes_dense": torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]]]),
            "pred_proto_scores_dense": torch.tensor([[0.8, 0.8]]),
        }
        repeated_outputs = {
            "pred_proto_logits_dense": logits,
            "pred_proto_boxes_dense": torch.tensor(
                [[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]]
            ),
            "pred_proto_scores_dense": torch.tensor([[0.8, 0.8, 0.7]]),
        }

        base = criterion.loss_proto_dense(base_outputs, targets, [], torch.tensor(2.0))["loss_proto_dense"]
        repeated = criterion.loss_proto_dense(repeated_outputs, targets, [], torch.tensor(2.0))["loss_proto_dense"]

        assert torch.allclose(base, repeated)

    def test_foreground_loss_suppresses_hard_background(self) -> None:
        """独立前景监督应同时接收前景正梯度和 hard negative 负梯度。"""
        criterion = SetCriterion(
            num_classes=3,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict={"loss_proto_dense": 1.0},
            focal_alpha=0.25,
            losses=["proto_dense"],
            proto_dense_foreground_enabled=True,
            proto_dense_background_ratio=1.0,
        )
        logits = torch.tensor([[[0.0, 3.0, -1.0], [3.0, 0.0, -1.0]]])
        fg_logits = torch.tensor([[2.0, 2.0]], requires_grad=True)
        outputs = {
            "pred_proto_logits_dense": logits,
            "pred_proto_boxes_dense": torch.tensor(
                [[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.05, 0.05]]]
            ),
            "pred_proto_scores_dense": torch.tensor([[0.9, 0.8]]),
            "pred_proto_fg_logits_dense": fg_logits,
        }
        targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
        result = criterion.loss_proto_dense(outputs, targets, [], torch.tensor(1.0))
        assert result["proto_dense_fg_positive_count"] == 1
        assert result["proto_dense_fg_background_count"] == 1
        result["loss_proto_dense_fg"].backward()
        assert fg_logits.grad is not None
        assert fg_logits.grad[0, 0] < 0
        assert fg_logits.grad[0, 1] > 0

    def test_foreground_loss_handles_empty_image(self) -> None:
        """无 GT 图像仍应把采样背景用于 foreground head 的负监督。"""
        criterion = SetCriterion(
            num_classes=2,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict={"loss_proto_dense": 1.0},
            focal_alpha=0.25,
            losses=["proto_dense"],
            proto_dense_foreground_enabled=True,
        )
        fg_logits = torch.tensor([[1.0, 1.0]], requires_grad=True)
        outputs = {
            "pred_proto_logits_dense": torch.zeros(1, 2, 2),
            "pred_proto_boxes_dense": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
            "pred_proto_scores_dense": torch.tensor([[0.9, 0.8]]),
            "pred_proto_fg_logits_dense": fg_logits,
        }
        result = criterion.loss_proto_dense(
            outputs,
            [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}],
            [],
            torch.tensor(1.0),
        )
        assert result["proto_dense_fg_background_count"] == 1
        result["loss_proto_dense_fg"].backward()
        assert fg_logits.grad is not None
        assert fg_logits.grad[0, 0] > 0
