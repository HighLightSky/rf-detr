# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Extracted from lwdetr.py (Phase 10)
# Original copyrights: LW-DETR (Baidu), Conditional DETR (Microsoft),
# DETR (Facebook), Deformable DETR (SenseTime)
# ------------------------------------------------------------------------
"""Loss functions and criterion for RF-DETR training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.models.heads.keypoints import compute_l1_keypoint_loss
from rfdetr.models.heads.segmentation import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.math import accuracy
from rfdetr.utilities import box_ops
from rfdetr.utilities.distributed import get_world_size, is_dist_avail_and_initialized

_LossFunction = Callable[..., dict[str, Tensor]]


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    result: Tensor = loss.mean(1).sum() / num_boxes
    return result


def sigmoid_varifocal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Tensor:
    prob = inputs.sigmoid()
    focal_weight = (
        targets * (targets > 0.0).float() + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float()
    )
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    result: Tensor = loss.sum() / num_masks
    return result


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptFunction[Any, Any]


def sigmoid_ce_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptFunction[Any, Any]


class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.

    The process happens in two steps:
    1) we compute Hungarian assignment between ground truth boxes and the outputs of the model.
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box).
    """

    # Signals that forward() accepts an explicit num_boxes denominator and exposes
    # num_boxes_for_targets() for cross-microbatch accumulation.  Subclasses that
    # override forward() with the legacy 2-arg signature should set this to False so
    # RFDETRModelModule._compute_train_losses() can skip the kwarg.
    supports_loss_normalizer_override: bool = True

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        focal_alpha: float,
        losses: list[str],
        group_detr: int = 1,
        sum_group_losses: bool = False,
        use_varifocal_loss: bool = False,
        use_position_supervised_loss: bool = False,
        ia_bce_loss: bool = False,
        mask_point_sample_ratio: int = 16,
        num_keypoints_per_class: list[int] | None = None,
        # [分类损失均衡化] P0 正样本类均衡 + P1 居中截断 Logit Adjustment（默认全关）
        # 见 docs/改进方案-SSCL/RF-DETR分类损失均衡化改进方案.md
        class_balance_enabled: bool = False,
        class_balance_counts: Tensor | None = None,
        class_balance_beta: float = 0.25,
        class_balance_max_weight: float = 3.0,
        class_balance_min_count: int = 10,
        class_balance_ref_count: float | None = None,
        class_balance_target_classes: list[int] | None = None,
        logit_adjustment_enabled: bool = False,
        logit_adjustment_tau: float = 0.1,
        logit_adjustment_bias_clip: float = 1.0,
        proto_dense_iou_pos: float = 0.3,
        proto_dense_iou_ignore: float = 0.1,
        proto_dense_center_fallback_topk: int = 4,
    ) -> None:
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
            class_balance_enabled: 是否启用 P0 正样本类均衡（只乘正样本 slot 权重，
                不降低负样本惩罚，优先保护 FDR）。
            class_balance_counts: 训练集每类实例数（长度须不小于 num_classes），
                用于计算 P0 权重与 P1 bias。为 None 时 P0/P1 均不生效。
            class_balance_beta: 幂律权重指数 β：w_c = (N_ref / max(n_c, n_min)) ** beta。
            class_balance_max_weight: 权重上限 w_max。
            class_balance_min_count: 分母下限 n_min，防极端小样本类产生极端权重。
            class_balance_ref_count: 参考样本数 N_ref，None 时自动取 sqrt(N_max * N_min)。
            class_balance_target_classes: 生效类别索引列表，其余类别权重固定 1.0。
            logit_adjustment_enabled: 是否启用 P1 居中截断 Logit Adjustment（训练侧）。
            logit_adjustment_tau: LA 强度 τ。
            logit_adjustment_bias_clip: 居中 bias 截断上限。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.num_keypoints_per_class = num_keypoints_per_class or []
        # [SSCL] 可选的 SSCL 损失回调。默认不启用，通过 set_sscl_loss_fn 注入，
        # 在 forward() 复用已计算的 Hungarian matching indices 计算 SSCL 损失。
        self._sscl_loss_fn: Callable | None = None
        # [分类损失均衡化] 预计算 P0 类别权重与 P1 logit bias 为 buffer（persistent=False
        # 不进 state_dict；criterion 每次构建时由训练配置重建，device 随 .to(device) 迁移）。
        self.class_balance_enabled = class_balance_enabled
        self.logit_adjustment_enabled = logit_adjustment_enabled
        self.proto_dense_iou_pos = float(proto_dense_iou_pos)
        self.proto_dense_iou_ignore = float(proto_dense_iou_ignore)
        self.proto_dense_center_fallback_topk = int(proto_dense_center_fallback_topk)
        if not 0.0 <= self.proto_dense_iou_ignore <= self.proto_dense_iou_pos <= 1.0:
            raise ValueError("dense 原型 IoU 阈值必须满足 0 <= ignore <= pos <= 1。")
        if self.proto_dense_center_fallback_topk < 1:
            raise ValueError("dense 原型中心 fallback 的 topk 必须 >= 1。")
        self._la_warmup_factor = 1.0
        weights, bias = self._build_class_balance_buffers(
            counts=class_balance_counts,
            beta=class_balance_beta,
            max_weight=class_balance_max_weight,
            min_count=class_balance_min_count,
            ref_count=class_balance_ref_count,
            target_classes=class_balance_target_classes,
            tau=logit_adjustment_tau,
            bias_clip=logit_adjustment_bias_clip,
        )
        # 类别维补齐到 num_classes（= args.num_classes + 1，含背景槽位）：
        # counts 只覆盖真实类别（如 SHWX 25 类），而 src_logits 最后一维是
        # num_classes（背景槽位索引为 num_classes-1，训练时无正样本）。
        # 权重补 1（不改变该槽位）、bias 补 0（背景槽位不做先验调整）。
        if weights.numel() < self.num_classes:
            pad = self.num_classes - weights.numel()
            weights = torch.cat([weights, torch.ones(pad, dtype=weights.dtype)])
            bias = torch.cat([bias, torch.zeros(pad, dtype=bias.dtype)])
        self.register_buffer("class_balance_weights", weights, persistent=False)
        self.register_buffer("logit_bias", bias, persistent=False)

    @staticmethod
    def _build_class_balance_buffers(
        counts: Tensor | None,
        beta: float,
        max_weight: float,
        min_count: int,
        ref_count: float | None,
        target_classes: list[int] | None,
        tau: float,
        bias_clip: float,
    ) -> tuple[Tensor, Tensor]:
        """按类别统计预计算 P0 权重与 P1 bias（均在 CPU 上，随 buffer 迁移 device）。

        P0 公式：w_c = clamp((N_ref / max(n_c, n_min)) ** beta, 1.0, w_max)，
        非 target_classes 的类别权重固定 1.0。
        P1 公式：bias = tau * clamp(log(pi) - mean(log(pi)), -clip, clip)，pi 为类别频率。

        Args:
            counts: 每类实例数，None 时返回全 1 权重与全 0 bias。
            beta: 幂律指数。
            max_weight: 权重上限。
            min_count: 分母下限。
            ref_count: 参考样本数 N_ref，None 自动取 sqrt(N_max * N_min)。
            target_classes: 生效类别，None 为全部。
            tau: LA 强度。
            bias_clip: bias 截断上限。

        Returns:
            (weights, bias) 两个长度等于 counts 的 CPU 张量。
        """
        num_real = counts.numel() if counts is not None else 0
        weights = torch.ones(num_real)
        bias = torch.zeros(num_real)
        if counts is None:
            return weights, bias
        n = counts.float()
        if n.numel() == 0:
            raise ValueError("class_balance_counts 不能为空")
        if torch.any(n < 0):
            raise ValueError("class_balance_counts 不能包含负数")
        positive = n[n > 0]
        if positive.numel() == 0:
            raise ValueError("class_balance_counts 至少需要一个正样本类别")
        n_eff = torch.maximum(n, torch.full_like(n, float(min_count)))
        if ref_count is None:
            # 几何平均 sqrt(N_max * N_min_positive)，避免零样本类别让 N_ref 退化为 0。
            n_ref = torch.sqrt(n.max() * positive.min())
        else:
            n_ref = torch.tensor(ref_count, dtype=n.dtype)
        w = ((n_ref / n_eff) ** beta).clamp(min=1.0, max=max_weight)
        if target_classes is not None:
            mask = torch.zeros(num_real, dtype=torch.bool)
            mask[torch.as_tensor(target_classes, dtype=torch.long)] = True
            w = torch.where(mask, w, torch.ones_like(w))
        weights = w
        pi = n / n.sum()
        raw = torch.log(pi.clamp_min(1e-6))
        centered = raw - raw.mean()
        bias = (tau * centered).clamp(min=-bias_clip, max=bias_clip)
        return weights, bias

    def set_la_warmup_factor(self, factor: float) -> None:
        """设置 P1 Logit Adjustment 的 warmup 缩放因子（0~1），由训练模块每步调用。

        Args:
            factor: 当前 warmup 进度，0 表示完全关闭 bias，1 表示全量生效。
        """
        self._la_warmup_factor = float(factor)

    @staticmethod
    def _output_device(outputs: dict[str, Any]) -> torch.device:
        """Return the device used by tensor outputs.

        Args:
            outputs: Model output dictionary. Top-level values are probed for tensors;
                nested structures (lists, nested dicts) are not traversed.

        Returns:
            Device of the first tensor value found in ``outputs``.

        Raises:
            ValueError: If no tensor output is present.
        """
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.device
        raise ValueError("SetCriterion requires at least one tensor output to infer the loss device.")

    def num_boxes_for_targets(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """Compute the distributed target-box denominator for a target batch.

        The denominator is the total number of ground-truth boxes in the batch, multiplied by the active number of
        DETR groups (unless ``sum_group_losses`` collapses them), reduced across all distributed ranks, divided by the
        world size, and finally clamped to be at least ``1.0`` so divide-by-zero never occurs on empty batches.

        Args:
            outputs: Model output dictionary; used only to infer the device for the
                returned scalar tensor.
            targets: Per-image target dictionaries for the current batch. Each must
                contain a ``"labels"`` tensor whose length equals the number of
                ground-truth boxes for that image.

        Returns:
            Scalar tensor on the same device as the model outputs, holding the
            average box-count denominator used to normalize criterion losses.

        Note:
            When ``torch.distributed`` is initialized this method performs an
            in-place ``all_reduce`` collective on the returned tensor. Every rank
            must reach this call together or the program will deadlock.

        Note:
            ``group_detr`` is multiplied in only when ``self.training`` is ``True``.
            During evaluation (``self.training`` is ``False``) the denominator
            collapses to a single group, so train-time and eval-time normalizers
            cannot be compared directly.

        Examples:
            >>> import torch
            >>> from rfdetr.models.criterion import SetCriterion
            >>> criterion = SetCriterion.__new__(SetCriterion)
            >>> criterion.training = False
            >>> criterion.group_detr = 1
            >>> criterion.sum_group_losses = False
            >>> outputs = {"pred_logits": torch.zeros(1, 1, 1)}
            >>> targets = [{"labels": torch.tensor([0, 1, 2])}]
            >>> criterion.num_boxes_for_targets(outputs, targets).item()
            3.0
        """
        group_detr = self.group_detr if self.training else 1
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes_tensor = torch.as_tensor(num_boxes, dtype=torch.float, device=self._output_device(outputs))
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_tensor)
        return torch.clamp(num_boxes_tensor / get_world_size(), min=1.0)

    def loss_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
        log: bool = True,
    ) -> dict[str, Tensor]:
        """Classification loss (Binary focal loss) targets dicts must contain the key "labels" containing a tensor of
        dim [nb_target_boxes]"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            # [P1] 居中截断 Logit Adjustment：只调整用于计算损失的局部 logits 副本，
            # 绝不能原地修改 src_logits（与 outputs["pred_logits"] 是同一 tensor，
            # 会污染后续 postprocess 的推理输出）。bias 按 warmup 因子缩放。
            adjusted_logits = src_logits
            if self.logit_adjustment_enabled and self._la_warmup_factor > 0:
                adjusted_logits = src_logits + self._la_warmup_factor * self.logit_bias.to(src_logits.dtype)

            prob = adjusted_logits.sigmoid()
            # init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob**gamma

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)

            t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
            neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
            # [P0] 正样本类均衡：只乘正样本 slot 的类别权重，neg_weights 不动
            # （不降低稀有类负样本惩罚，优先保护 FDR）。
            if self.class_balance_enabled:
                pos_weights[tuple(pos_ind)] *= self.class_balance_weights[target_classes_o].to(pos_weights.dtype)
            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * adjusted_logits - F.logsigmoid(adjusted_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            pos_ious_func = pos_ious_func.to(cls_iou_func_targets.dtype)
            cls_iou_func_targets[tuple(pos_ind)] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets / (
                cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8
            )
            loss_ce = (
                position_supervised_loss(
                    src_logits,
                    norm_cls_iou_func_targets,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )

        elif self.use_varifocal_loss:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            cls_iou_targets[tuple(pos_ind)] = pos_ious
            loss_ce = (
                sigmoid_varifocal_loss(
                    src_logits,
                    cls_iou_targets,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )
        else:
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_proto_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """[ProtoGuidance] 对 matched foreground 的原型 logits 做类别均衡 CE。

        仅在 ``outputs`` 含 ``pred_proto_logits`` 时生效（仅 enc_outputs 分支携带该键，
        主层与 aux 层返回空字典自动跳过，由 forward 的 losses 循环安全遍历）。
        复用 enc 分支的 Hungarian ``indices``——token/box 与 ``pred_logits`` 同序，
        只对 matched foreground 监督，避免把大量未匹配 query 当成 25 个独立
        background 负类。各类别先独立求均值再平均，避免 HM 等高频类吞掉梯度。

        Args:
            outputs: 模型输出字典（含 ``pred_proto_logits`` ``[bs, Q, C_fg]``）。
            targets: 每图 GT 列表。
            indices: enc 分支 Hungarian 匹配结果。
            num_boxes: 归一化分母（GT 框总数）。

        Returns:
            ``{"loss_proto_labels": cross entropy}``；无 ``pred_proto_logits`` 键时返回空字典。
        """
        if "pred_proto_logits" not in outputs:
            return {}
        src_logits = outputs["pred_proto_logits"]
        _, _, num_classes_fg = src_logits.shape
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        target_classes_o = target_classes_o.clamp(0, num_classes_fg - 1)
        if target_classes_o.numel() == 0:
            return {"loss_proto_labels": src_logits.sum() * 0.0}

        matched_logits = src_logits[idx]
        per_sample = F.cross_entropy(matched_logits, target_classes_o, reduction="none")
        class_losses = [per_sample[target_classes_o == class_id].mean() for class_id in target_classes_o.unique()]
        return {"loss_proto_labels": torch.stack(class_losses).mean()}

    def loss_proto_dense(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """对全部 encoder token 的前景原型 logits 施加类别均衡 CE。"""
        required = ("pred_proto_logits_dense", "pred_proto_boxes_dense", "pred_proto_scores_dense")
        if any(key not in outputs for key in required):
            return {}
        logits = outputs["pred_proto_logits_dense"]
        boxes = outputs["pred_proto_boxes_dense"]
        scores = outputs["pred_proto_scores_dense"]
        if logits.ndim != 3 or boxes.shape != logits.shape[:2] + (4,) or scores.shape != logits.shape[:2]:
            raise ValueError("dense 原型输出形状必须为 logits[B,N,C]、boxes[B,N,4]、scores[B,N]。")

        labels_per_image = [
            self._assign_dense_proto_labels(boxes[batch_idx], scores[batch_idx], target)
            for batch_idx, target in enumerate(targets)
        ]
        dense_labels = torch.stack(labels_per_image)
        positive = dense_labels >= 0
        positive_count = positive.sum()
        if not bool(positive.any()):
            zero = logits.sum() * 0.0
            return {
                "loss_proto_dense": zero,
                "proto_dense_positive_count": zero.detach(),
                "proto_dense_accuracy": zero.detach(),
                "proto_dense_margin": zero.detach(),
            }

        positive_logits = logits[positive]
        positive_labels = dense_labels[positive]
        per_sample = F.cross_entropy(positive_logits, positive_labels, reduction="none")
        class_losses = [per_sample[positive_labels == class_id].mean() for class_id in positive_labels.unique()]
        sorted_logits = positive_logits.topk(min(2, positive_logits.shape[-1]), dim=-1).values
        margin = sorted_logits[:, 0]
        if sorted_logits.shape[-1] > 1:
            margin = margin - sorted_logits[:, 1]
        return {
            "loss_proto_dense": torch.stack(class_losses).mean(),
            "proto_dense_positive_count": positive_count.to(dtype=logits.dtype).detach(),
            "proto_dense_accuracy": (positive_logits.argmax(dim=-1) == positive_labels).float().mean().detach(),
            "proto_dense_margin": margin.mean().detach(),
        }

    def _assign_dense_proto_labels(
        self,
        dense_boxes: Tensor,
        dense_scores: Tensor,
        target: dict[str, Tensor],
    ) -> Tensor:
        """按 IoU 与中心 fallback 为单张图的 dense token 分配前景类别。"""
        labels = torch.full((dense_boxes.shape[0],), -1, dtype=torch.long, device=dense_boxes.device)
        target_boxes = target.get("boxes")
        target_labels = target.get("labels")
        if target_boxes is None or target_labels is None or target_boxes.numel() == 0:
            return labels
        target_boxes = target_boxes.to(device=dense_boxes.device, dtype=dense_boxes.dtype)
        target_labels = target_labels.to(device=dense_boxes.device, dtype=torch.long)

        dense_xyxy = box_ops.box_cxcywh_to_xyxy(dense_boxes)
        target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        ious, _ = box_ops.box_iou(dense_xyxy, target_xyxy)
        best_iou, best_target = ious.max(dim=1)
        positive = best_iou >= self.proto_dense_iou_pos
        labels[positive] = target_labels[best_target[positive]]

        centers = dense_boxes[:, :2]
        for target_idx in range(target_boxes.shape[0]):
            if bool((positive & (best_target == target_idx)).any()):
                continue
            x0, y0, x1, y1 = target_xyxy[target_idx]
            inside = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] <= x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] <= y1)
                & (best_target == target_idx)
            )
            candidate_indices = inside.nonzero(as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            count = min(self.proto_dense_center_fallback_topk, int(candidate_indices.numel()))
            selected = candidate_indices[dense_scores[candidate_indices].topk(count).indices]
            labels[selected] = target_labels[target_idx]
        return labels

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes This is not
        really a loss, it is intended for logging purposes only.

        It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Sigmoid/focal heads have no background class; count predictions whose top score is confident
        card_pred = (pred_logits.sigmoid().max(-1).values > 0.5).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss targets dicts must
        contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4] The target boxes are expected in format
        (center_x, center_y, w, h), normalized by the image size."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.

        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'.
        """
        assert "pred_masks" in outputs, "pred_masks missing in model outputs"
        idx = self._get_src_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]

        if isinstance(pred_masks, Tensor):
            # gather matched prediction masks
            # handle no matches
            src_masks = pred_masks[idx]  # [N, H, W]
        else:
            spatial_features = outputs["pred_masks"]["spatial_features"]
            query_features = outputs["pred_masks"]["query_features"]
            bias = outputs["pred_masks"]["bias"]
            # No matches: return a zero loss that still flows through the segmentation-head
            # outputs, so every parameter stays connected in the autograd graph (required for
            # DDP, which errors on parameters that receive no gradient).
            if idx[0].numel() == 0:
                zero = (spatial_features.sum() + query_features.sum() + bias.sum()) * 0.0
                return {"loss_mask_ce": zero, "loss_mask_dice": zero}
            else:
                batched_selected_masks = []
                per_batch_counts = idx[0].unique(return_counts=True)[1]  # type: ignore[no-untyped-call]
                batch_indices = torch.cat((torch.zeros_like(per_batch_counts[:1]), per_batch_counts), dim=0).cumsum(0)

                for i in range(per_batch_counts.shape[0]):
                    batch_indicator = idx[0][batch_indices[i] : batch_indices[i + 1]]
                    box_indicator = idx[1][batch_indices[i] : batch_indices[i + 1]]

                    this_batch_queries = query_features[(batch_indicator, box_indicator)]
                    this_batch_spatial_features = spatial_features[idx[0][batch_indices[i + 1] - 1]]

                    this_batch_masks = (
                        torch.einsum(
                            "chw,nc->nhw",
                            this_batch_spatial_features,
                            this_batch_queries,
                        )
                        + bias
                    )

                    batched_selected_masks.append(this_batch_masks)

                src_masks = torch.cat(batched_selected_masks)

        if src_masks.numel() == 0:
            return {
                "loss_mask_ce": src_masks.sum(),
                "loss_mask_dice": src_masks.sum(),
            }
        # gather matched target masks
        target_masks = torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            src_masks.shape[-2],
            src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio,
        )

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        with torch.no_grad():
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        # ``sigmoid_ce_loss_jit`` and ``dice_loss_jit`` are TorchScripted with
        # ``num_masks: float`` in their signatures, so they reject Tensor inputs at
        # runtime with a "expected float, got Tensor" error.  ``SetCriterion.forward``
        # now hands the criterion a Tensor denominator (so it can be all-reduced across
        # ranks and accumulated across grad-accum microbatches), so it must be unwrapped
        # to a Python scalar exactly here before the JIT call boundary.  Using
        # ``float(...)`` instead of ``.item()`` keeps the conversion safe whether
        # ``num_boxes`` arrives as a Tensor, a Python int/float, or a numpy scalar.
        num_boxes_scalar = float(num_boxes)
        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes_scalar),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes_scalar),
        }

        del src_masks
        del target_masks
        return losses

    def loss_keypoints(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """Compute keypoint losses on matched prediction/target pairs."""
        assert "pred_keypoints" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs["pred_keypoints"][idx]
        target_keypoints = torch.cat([target["keypoints"][j] for target, (_, j) in zip(targets, indices)], dim=0)
        target_classes = torch.cat([target["labels"][j] for target, (_, j) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([target["boxes"][j] for target, (_, j) in zip(targets, indices)], dim=0)
        target_areas = target_boxes[:, 2] * target_boxes[:, 3]

        loss_l1, loss_findable, loss_visible, loss_nll = compute_l1_keypoint_loss(
            all_pred_keypoints=src_keypoints,
            target_keypoints=target_keypoints.to(src_keypoints.device),
            target_classes=target_classes.to(src_keypoints.device),
            target_areas=target_areas.to(src_keypoints.device),
            num_keypoints_per_class=self.num_keypoints_per_class,
        )

        return {
            "loss_keypoints_l1": loss_l1.sum() / num_boxes,
            "loss_keypoints_findable": loss_findable.sum() / num_boxes,
            "loss_keypoints_visible": loss_visible.sum() / num_boxes,
            "loss_keypoints_nll": loss_nll.sum() / num_boxes,
        }

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def set_sscl_loss_fn(self, loss_fn: Callable) -> None:
        """注册 SSCL 损失回调。

        回调在 forward() 中复用已计算的 Hungarian matching ``indices``，
        签名约定：``(outputs: dict, targets: list, indices: list) -> dict[str, Tensor]``，
        返回的字典键值将合并进最终 losses（键为 ``loss_sscl``，值为 SSCL 损失标量）。

        Args:
            loss_fn: SSCL 损失计算回调。
        """
        self._sscl_loss_fn = loss_fn

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        loss_map: dict[str, _LossFunction] = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "keypoints": self.loss_keypoints,
            # [ProtoGuidance] 原型分类辅助损失（仅 enc_outputs 携带 pred_proto_logits 时产出）
            "proto_labels": self.loss_proto_labels,
            "proto_dense": self.loss_proto_dense,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Tensor]],
        num_boxes: Tensor | float | None = None,
    ) -> dict[str, Tensor]:
        """Compute every configured loss for one (outputs, targets) pair.

        The Hungarian matcher is invoked on the last layer's outputs and reused for the auxiliary intermediate layers
        and the optional encoder outputs; each loss is then evaluated on the matched indices and normalized by
        ``num_boxes``.

        Args:
            outputs: Model output dictionary. Must contain the tensors required by
                every loss in ``self.losses`` (for example ``"pred_logits"``,
                ``"pred_boxes"``, ``"pred_masks"``, ``"pred_keypoints"``). May also
                contain ``"aux_outputs"`` (list of layer-wise outputs) and
                ``"enc_outputs"`` (encoder outputs); both are processed identically
                to the last layer and contribute prefixed keys to the returned dict.
            targets: Per-image target dictionaries; ``len(targets) == batch_size``.
                The expected keys depend on the losses being applied — see each
                ``loss_*`` method for its target requirements.
            num_boxes: Optional explicit box-count denominator.

                - ``None`` (default): call :meth:`num_boxes_for_targets` to derive
                  the distributed-reduced normalizer for the current batch.
                - ``float`` / ``int``: cast to a tensor on the model output device
                  and used verbatim. Passing ``1.0`` yields *unnormalized* loss
                  numerators (used by the manual-optimization path so the caller
                  can apply its own accumulated denominator).
                - ``Tensor``: moved to the model output device and used
                  verbatim. The caller is responsible for any cross-rank reduction;
                  no extra all-reduce is performed in this branch.

        Returns:
            Dictionary of named loss tensors. Last-layer losses keep their base
            names (``"loss_ce"``, ``"loss_bbox"``, ``"loss_giou"``,
            ``"loss_mask_ce"``, ``"loss_mask_dice"``, ``"loss_keypoints_*"``).
            Auxiliary-layer losses get a ``"_<i>"`` suffix; encoder-layer losses
            get an ``"_enc"`` suffix.

        Examples:
            >>> import torch
            >>> from unittest.mock import MagicMock
            >>> from rfdetr.models.criterion import SetCriterion
            >>> criterion = SetCriterion.__new__(SetCriterion)
            >>> criterion.training = False
            >>> criterion.group_detr = 1
            >>> criterion.sum_group_losses = False
            >>> criterion.losses = []
            >>> criterion.matcher = MagicMock(return_value=[])
            >>> outputs = {"pred_logits": torch.zeros(1, 1, 1)}
            >>> targets = [{"labels": torch.tensor([0])}]
            >>> criterion.forward(outputs, targets, num_boxes=1.0)
            {}
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)
        # [SSCL] 保存最后一层的匹配结果，供 SSCL 回调使用。后续 aux/enc 循环
        # 会覆盖 indices，而 SSCL 只作用在 decoder 最后一层的 matched query 上。
        last_layer_indices = indices

        if num_boxes is None:
            num_boxes = self.num_boxes_for_targets(outputs, targets)
        elif not torch.is_tensor(num_boxes):
            num_boxes = torch.as_tensor(num_boxes, dtype=torch.float, device=self._output_device(outputs))
        else:
            num_boxes = num_boxes.to(device=self._output_device(outputs), dtype=torch.float)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # [SSCL] 若注册了 SSCL 损失回调，使用最后一层的匹配结果计算 SSCL 损失，
        # 结果合并进 losses 字典（键为 "loss_sscl"，由训练循环按 sscl_lambda 加权）。
        if self._sscl_loss_fn is not None:
            sscl_losses = self._sscl_loss_fn(outputs, targets, last_layer_indices)
            losses.update(sscl_losses)

        return losses
