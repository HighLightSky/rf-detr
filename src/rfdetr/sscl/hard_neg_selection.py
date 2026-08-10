# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例负样本（hard negative）选择器。

从单张图的 decoder 输出中选择"像目标但不是目标"的难例 query：

1. 排除 Hungarian matching 匹配到的 query（补集 = 未匹配 query）；
2. 未匹配 query 的预测框与全部 GT 计算 IoU，过滤 ``max_iou in [0.1, 0.5]``
   带（下界剔除纯背景、上界剔除真实目标的重复检测——DETR 的 1-to-1 匹配
   会让部分真实目标成为未匹配 query，绝不能把它们当负样本）；
3. 分数 = 最大前景 logit ``pred_logits[:, :-1].max(-1)``，过滤
   ``>= score_thresh``；
4. 按分数降序（stable）取 top-k，索引映射回 decoder hidden states。

选择出的难例特征已 ``detach``，不参与反向传播；逐 batch 动态生成、用完即弃，
不 EMA、不入类别原型库。同时返回等量的**随机未匹配**特征作为硬度对照基线，
供诊断脚本与训练监控评估"难例是否真的比随机未匹配更贴近类别原型"。

纯函数、无 module/训练框架依赖，训练回调与离线诊断脚本共用同一实现。
"""

from __future__ import annotations

import torch
from torch import Tensor

from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou

# 难例 IoU 带：与任一 GT 的最大 IoU 落在此区间内才视为"部分目标/相似背景"
IOU_BAND_LOW = 0.1
IOU_BAND_HIGH = 0.5


def select_hard_negatives_for_image(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    hs: Tensor,
    gt_boxes: Tensor,
    matched_src: Tensor,
    top_k: int = 3,
    score_thresh: float = 0.0,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """单图难例负样本选择（纯函数，无梯度副作用）。

    Args:
        pred_logits: 该图全部 query 的分类 logits ``[Q, C+1]``（末位为
            background）。
        pred_boxes: 该图全部 query 的预测框 ``[Q, 4]``（cxcywh 归一化）。
        hs: 该图全部 query 的 decoder 最后一层 hidden states ``[Q, D]``。
        gt_boxes: 该图 GT 框 ``[M, 4]``（cxcywh 归一化，可能为空）。
        matched_src: 该图 Hungarian matching 匹配到的 query 索引 ``[N_m]``
            （被排除，绝不能成为难例）。
        top_k: 每图最多选取的难例数量（按最大前景 logit 降序，stable）。
        score_thresh: 最大前景 logit 下限，低于该值不选。
        seed: 随机对照特征采样的随机种子（仅测试传参，训练时传 ``None``）。

    Returns:
        ``(hn_features, random_features, stats)`` 元组：
        - hn_features: ``[k, D]`` 已 detach 的难例特征（k 可能 < top_k 甚至为 0）。
        - random_features: ``[k', D]`` 已 detach 的随机未匹配 query 特征
          （硬度对照基线，无 IoU/分数约束；k' = min(top_k, 未匹配数)）。
        - stats: ``{"n_selected", "n_band", "n_unmatched"}`` CPU 标量。
    """
    device = pred_logits.device
    num_q = pred_logits.shape[0]

    # 1. 排除 matched query（补集 = 未匹配 query）
    unmatched = torch.ones(num_q, dtype=torch.bool, device=device)
    if matched_src.numel() > 0:
        unmatched[matched_src] = False
    stats = {
        "n_selected": 0.0,
        "n_band": 0.0,
        "n_unmatched": float(unmatched.sum().item()),
    }

    # 2. IoU 带过滤（空 GT 时退化：无候选，max_iou 全 0）
    if gt_boxes.shape[0] > 0 and unmatched.any():
        iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))  # [Q, M]
        max_iou = iou.max(dim=1).values  # [Q]
    else:
        max_iou = torch.zeros(num_q, device=device)
    in_band = (max_iou >= IOU_BAND_LOW) & (max_iou <= IOU_BAND_HIGH) & unmatched
    stats["n_band"] = float(in_band.sum().item())

    # 3+4. 分数过滤 + stable 降序 top-k（并列时取索引更小的 query）
    scores = pred_logits[:, :-1].max(dim=-1).values  # [Q]
    candidates = in_band & (scores >= score_thresh)
    selected = torch.empty(0, dtype=torch.long, device=device)
    if candidates.any():
        cand_idx = candidates.nonzero(as_tuple=False).flatten()
        order = torch.argsort(scores[cand_idx], descending=True, stable=True)[:top_k]
        selected = cand_idx[order]
    hn_features = hs[selected].detach()
    stats["n_selected"] = float(selected.shape[0])

    # 随机未匹配对照特征（与难例等量，便于硬度对比；训练时可确定性传 seed）。
    # 注意：必须在未匹配 query 的"索引集合"内采样——若先对全量 query 置乱再
    # 用布尔掩码取位置，会混入 matched query（掩码选的是排列的位置而非值）。
    rand_idx = torch.empty(0, dtype=torch.long, device=device)
    if unmatched.any():
        unmatched_idx = unmatched.nonzero(as_tuple=False).flatten()  # [U]
        k = min(top_k, unmatched_idx.shape[0])
        generator = torch.Generator(device=device) if seed is not None else None
        if generator is not None:
            generator.manual_seed(seed)
        perm = torch.randperm(unmatched_idx.shape[0], generator=generator, device=device)
        rand_idx = unmatched_idx[perm[:k]]
    random_features = hs[rand_idx].detach()

    return hn_features, random_features, stats
