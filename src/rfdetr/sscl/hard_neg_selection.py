# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本（hard negative）选择器。

从单张图的 decoder 输出中选择"像目标但不是目标"的难例 query：

1. 排除 Hungarian matching 匹配到的 query（补集 = 未匹配 query）；
2. 未匹配 query 的预测框与全部 GT 计算 IoU，过滤 ``max_iou in [0.0, 0.3]``
   带（下界 0.0 纳入纯背景/低 IoU 区域——对准"纯背景虚警"靶点；上界 0.3
   剔除真实目标的重复检测——DETR 的 1-to-1 匹配会让部分真实目标成为
   未匹配 query，绝不能把它们当负样本）；
3. 分数 = 目标前景类别集合内最大 logit，过滤 ``>= score_thresh``；
4. 按分数降序（stable）取 top-k，返回 query 索引。

选择器只返回索引，不截断梯度；训练侧用这些索引直接监督高置信未匹配
query 降低前景分数，从而对虚警本身反传。

纯函数、无 module/训练框架依赖，训练回调与离线诊断脚本共用同一实现。
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou

# 默认难例 IoU 带：对准纯背景/低 IoU 虚警，上界 0.3 保护重复检测不被误标负样本。
IOU_BAND_LOW = 0.0
IOU_BAND_HIGH = 0.3


def select_hard_negatives_for_image(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    gt_boxes: Tensor,
    matched_src: Tensor,
    top_k: int = 3,
    score_thresh: float = 0.0,
    class_score_thresholds: dict[int, float] | None = None,
    target_classes: list[int] | Tensor | None = None,
    iou_low: float = IOU_BAND_LOW,
    iou_high: float = IOU_BAND_HIGH,
    selection_mode: Literal["score", "class_balanced"] = "score",
    class_quota: int = 1,
    require_class_absent: bool = False,
    present_classes: list[int] | Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """单图难例负样本选择（纯函数，无梯度副作用）。

    Args:
        pred_logits: 该图全部 query 的分类 logits ``[Q, C+1]``（末位为
            background）。
        pred_boxes: 该图全部 query 的预测框 ``[Q, 4]``（cxcywh 归一化）。
        gt_boxes: 该图 GT 框 ``[M, 4]``（cxcywh 归一化，可能为空）。
        matched_src: 该图 Hungarian matching 匹配到的 query 索引 ``[N_m]``
            （被排除，绝不能成为难例）。
        top_k: 每图最多选取的难例数量（按最大前景 logit 降序，stable）。
        score_thresh: 最大前景 logit 下限，低于该值不选。
        class_score_thresholds: 按预测类别覆盖 score_thresh 的阈值。
        target_classes: 参与挖掘的前景类别索引。为 ``None`` 时使用全部前景类。
        iou_low: 难例 IoU 带下界。
        iou_high: 难例 IoU 带上界。
        selection_mode: ``"score"`` 按全局分数取 top-k；``"class_balanced"``
            先为每个预测类别保留最多 ``class_quota`` 个难例，再按分数填满
            剩余名额，用于避免宏平均指标被单一类别的难例梯度占满。
        class_quota: 类均衡模式下每个预测类别的首轮配额。
        require_class_absent: 是否只保留预测类别不在当前图 GT 中的候选。开启后
            同图同类的重复框和定位偏差真目标不会被 hard-negative 抑制。
        present_classes: 当前图 GT 中出现的类别索引；仅
            ``require_class_absent=True`` 时使用。

    Returns:
        ``(hn_indices, stats)`` 元组：
        - hn_indices: ``[k]`` 难例 query 索引（k 可能 < top_k 甚至为 0）。
        - stats: ``{"n_selected", "n_band", "n_unmatched", "score_mean",
          "iou_mean"}`` CPU 标量。
    """
    device = pred_logits.device
    if selection_mode not in ("score", "class_balanced"):
        raise ValueError(f"selection_mode 必须是 score 或 class_balanced，收到 {selection_mode}")
    if class_quota < 1:
        raise ValueError(f"class_quota 必须 >= 1，收到 {class_quota}")
    num_q = pred_logits.shape[0]
    num_foreground = max(0, pred_logits.shape[-1] - 1)

    # 1. 排除 matched query（补集 = 未匹配 query）
    unmatched = torch.ones(num_q, dtype=torch.bool, device=device)
    if matched_src.numel() > 0:
        unmatched[matched_src] = False
    stats = {
        "n_selected": 0.0,
        "n_band": 0.0,
        "n_unmatched": float(unmatched.sum().item()),
        "score_mean": 0.0,
        "iou_mean": 0.0,
    }

    # 2. IoU 带过滤（空 GT 时退化：无候选，max_iou 全 0）
    if gt_boxes.shape[0] > 0 and unmatched.any():
        iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))  # [Q, M]
        max_iou = iou.max(dim=1).values  # [Q]
    else:
        max_iou = torch.zeros(num_q, device=device)
    in_band = (max_iou >= iou_low) & (max_iou <= iou_high) & unmatched
    stats["n_band"] = float(in_band.sum().item())

    # 3+4. 分数过滤 + stable 降序 top-k（并列时取索引更小的 query）
    if target_classes is None:
        class_idx = torch.arange(num_foreground, device=device)
    else:
        class_idx = torch.as_tensor(target_classes, dtype=torch.long, device=device)
        class_idx = class_idx[(class_idx >= 0) & (class_idx < num_foreground)]
    if class_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), stats
    target_logits = pred_logits[:, class_idx]
    scores, predicted_idx = target_logits.max(dim=-1)  # [Q]
    predicted_class = class_idx[predicted_idx]
    thresholds = torch.full_like(scores, float(score_thresh))
    for class_id, class_threshold in (class_score_thresholds or {}).items():
        thresholds = torch.where(
            predicted_class == int(class_id),
            torch.as_tensor(float(class_threshold), device=device, dtype=scores.dtype),
            thresholds,
        )
    candidates = in_band & (scores >= thresholds)
    if require_class_absent and present_classes is not None:
        present = torch.as_tensor(present_classes, dtype=torch.long, device=device)
        if present.numel() > 0:
            candidates = candidates & ~torch.isin(predicted_class, present)
    selected = torch.empty(0, dtype=torch.long, device=device)
    if candidates.any():
        cand_idx = candidates.nonzero(as_tuple=False).flatten()
        if selection_mode == "score":
            order = torch.argsort(scores[cand_idx], descending=True, stable=True)[:top_k]
            selected = cand_idx[order]
        else:
            # 首轮按预测类别配额取样，二轮再按全局分数填充，保持确定性。
            first_pass: list[Tensor] = []
            for class_id in class_idx.tolist():
                class_candidates = cand_idx[predicted_class[cand_idx] == class_id]
                if class_candidates.numel() == 0:
                    continue
                class_order = torch.argsort(scores[class_candidates], descending=True, stable=True)
                first_pass.append(class_candidates[class_order[:class_quota]])
            if first_pass:
                selected = torch.cat(first_pass)
                first_order = torch.argsort(scores[selected], descending=True, stable=True)
                selected = selected[first_order[:top_k]]
            if selected.numel() < top_k:
                selected_mask = torch.zeros(num_q, dtype=torch.bool, device=device)
                selected_mask[selected] = True
                remaining = cand_idx[~selected_mask[cand_idx]]
                if remaining.numel() > 0:
                    remaining_order = torch.argsort(scores[remaining], descending=True, stable=True)
                    fill = remaining[remaining_order[: top_k - selected.numel()]]
                    selected = torch.cat([selected, fill])
    stats["n_selected"] = float(selected.shape[0])
    if selected.numel() > 0:
        stats["score_mean"] = float(scores[selected].mean().detach().item())
        stats["iou_mean"] = float(max_iou[selected].mean().detach().item())
    return selected, stats
