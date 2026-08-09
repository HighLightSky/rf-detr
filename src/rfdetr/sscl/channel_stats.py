# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""类别通道掩码的 TF-IDF 离线统计。

对每个 base 类的 matched query 特征，做通道级激活统计：把特征与语义方向逐通道
相乘后做精确 2-means 聚类得到激活簇，据此计算 TF-IDF 打分并得到通道排名。
在线阶段（SemanticResidual）用可学习阈值 θ_c 在排名上做 soft 掩码：

    M_{c,i}(θ_c) = sigmoid((θ_c − r_{c,i}) / τ_mask)

TF 大 = 该类频繁激活，IDF 大 = 他类少激活，Score=TF·IDF 高的通道是类判别通道。
本模块只负责离线统计与掩码构造的确定性实现，全部操作可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# TF-IDF 计算中的平滑项，避免 IDF 除零
_EPS = 1e-8


@dataclass
class ChannelStats:
    """通道 TF-IDF 统计结果。

    Attributes:
        tf: 每类通道激活频率 ``[C, d]``，``TF_c(i)``。
        idf: 通道逆文档频率 ``[d]``，``IDF(i)``。
        score: 每类通道打分 ``[C, d]``，``Score_c(i) = TF_c(i)·IDF(i)``。
        rank: 每类通道排名 ``[C, d]``（int64），``r_{c,i} ∈ [1, d]``，1 表示判别性最强。
        counts: 每类参与统计的实例数 ``{class_id: int}``。
        meta: 来源信息（checkpoint、数据集等）。
    """

    tf: Tensor
    idf: Tensor
    score: Tensor
    rank: Tensor
    counts: dict[int, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _batch_2means_activation_mask(a: Tensor) -> Tensor:
    """批量精确 2-means：对 ``[N, d]`` 的每一行求激活簇掩码（全向量化）。

    升序排序后对每个切点 k（1..d-1）用前缀和一次性批量计算簇内 SSE，取 argmin。
    均值较高的簇（升序排列的右侧）标记为 True（激活簇）。结果确定可复现，
    比逐实例 Python 循环快两个数量级（train 模式收集后特征数可达 20 万+）。

    Args:
        a: 每行一个实例的通道激活度量 ``[N, d]``（n≥2）。

    Returns:
        布尔掩码 ``[N, d]``，True 表示该通道属于该行的激活簇。
    """
    n, d = a.shape
    if d < 2:
        return torch.ones_like(a, dtype=torch.bool)
    order = torch.argsort(a, dim=-1, stable=True)  # [N, d] 每行升序索引
    sorted_a = torch.gather(a, 1, order)  # [N, d]
    # 前缀和（含 0 头）：pre[:, i] = 每行前 i 个元素之和 [N, d+1]
    pre = torch.cat([torch.zeros(n, 1, dtype=a.dtype, device=a.device), torch.cumsum(sorted_a, dim=-1)], dim=-1)
    pre_sq = torch.cat([torch.zeros(n, 1, dtype=a.dtype, device=a.device), torch.cumsum(sorted_a.square(), dim=-1)], dim=-1)

    # 对每个切点 k（左簇 [0,k)，右簇 [k,d)）批量计算 SSE
    left_cnt = torch.arange(1, d, dtype=a.dtype, device=a.device)  # [d-1]
    right_cnt = (d - left_cnt)
    left_sum = pre[:, 1:d]  # [N, d-1]
    left_sq = pre_sq[:, 1:d]
    right_sum = pre[:, d : d + 1] - left_sum
    right_sq = pre_sq[:, d : d + 1] - left_sq
    left_ssr = left_sq - left_sum.square() / left_cnt
    right_ssr = right_sq - right_sum.square() / right_cnt
    ssr = left_ssr + right_ssr  # [N, d-1]
    best_k = ssr.argmin(dim=-1) + 1  # [N] 每行最优切点（1-based）
    # 激活簇 = 升序排列的右侧 [k, d)
    mask_sorted = torch.arange(d, device=a.device) >= best_k.unsqueeze(-1)  # [N, d]
    mask = torch.zeros(n, d, dtype=torch.bool, device=a.device)
    mask.scatter_(1, order, mask_sorted)
    return mask


def compute_channel_tfidf(features_by_class: dict[int, Tensor], s_matrix: Tensor) -> ChannelStats:
    """计算 base 类 matched 特征的通道 TF-IDF 统计。

    对每个实例 v：``a = v/‖v‖ ⊙ s_c``（s_c 已归一化），对 a 的 d 个通道做精确
    2-means 得到激活掩码 g_{c,n}(i)；汇总为 ``TF_c(i)=count_c(i)/N_c``，
    ``DF(i)=mean_c TF_c(i)``，``IDF(i)=log(1/(DF+ε))``，``Score=TF·IDF``，
    每类按 Score 降序给出通道排名。

    Args:
        features_by_class: ``{class_id: matched 特征 [N_c, d]}`` 映射（仅 base 类）。
        s_matrix: 语义方向矩阵 ``[C, d]``，行 = 各类别语义方向。

    Returns:
        ``ChannelStats``，其中 rank 为 ``[C, d]`` 的 1-based 排名。
    """
    if not features_by_class:
        raise ValueError("features_by_class 不能为空。")
    d = s_matrix.shape[1]
    class_ids = sorted(features_by_class)
    s_norm = F.normalize(s_matrix, dim=-1)

    counts: dict[int, int] = {}
    tf_list: list[Tensor] = []
    for c in class_ids:
        feats = features_by_class[c]
        counts[c] = int(feats.shape[0])
        if feats.shape[0] == 0:
            tf_list.append(torch.zeros(d, dtype=torch.float32))
            continue
        v_norm = F.normalize(feats, dim=-1)  # [N, d]
        a = v_norm * s_norm[c].unsqueeze(0)  # [N, d]
        # 批量精确 2-means（全向量化，避免逐实例 Python 循环）
        masks = _batch_2means_activation_mask(a)  # [N, d]
        tf_list.append(masks.float().mean(dim=0))
    tf = torch.stack(tf_list, dim=0)  # [C, d]

    df = tf.mean(dim=0)  # [d]
    idf = torch.log(1.0 / (df + _EPS))  # [d]
    score = tf * idf  # [C, d]
    # 每类按 score 降序排名：-score 升序 argsort 再取 argsort 得 0-based 位置，+1 转 1-based
    rank = (torch.argsort(torch.argsort(-score, dim=-1, stable=True), dim=-1, stable=True) + 1).long()

    return ChannelStats(tf=tf, idf=idf, score=score, rank=rank, counts=counts)


def build_mask_from_rank(rank: Tensor, theta: Tensor, mask_tau: float = 1.0) -> Tensor:
    """由通道排名与可学习阈值构造 soft 掩码。

    ``M_{c,i}(θ_c) = sigmoid((θ_c − r_{c,i}) / τ_mask)``，θ 越大保留通道越多。

    Args:
        rank: 通道排名 ``[C, d]``（1-based，来自 TF-IDF 统计）。
        theta: 每类掩码阈值 ``[C]``。
        mask_tau: 掩码软度 τ_mask。

    Returns:
        掩码 ``[C, d]``，值域 (0, 1)。
    """
    return torch.sigmoid((theta.unsqueeze(-1) - rank) / mask_tau)


def save_channel_stats(path: str | Path, stats: ChannelStats) -> None:
    """保存通道 TF-IDF 统计。

    Args:
        path: 输出文件路径（``.pt`` 后缀）。
        stats: ``ChannelStats`` 实例。
    """
    torch.save(
        {
            "tf": stats.tf.cpu(),
            "idf": stats.idf.cpu(),
            "score": stats.score.cpu(),
            "rank": stats.rank.cpu(),
            "counts": stats.counts,
            "meta": stats.meta,
        },
        path,
    )
    logger.info(f"通道 TF-IDF 统计已保存到: {path}（rank 形状: {tuple(stats.rank.shape)}）")


def load_channel_stats(path: str | Path) -> ChannelStats:
    """加载通道 TF-IDF 统计。

    Args:
        path: ``save_channel_stats`` 保存的文件路径。

    Returns:
        ``ChannelStats`` 实例。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        KeyError: 当文件中缺少 ``rank`` 键时抛出。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"通道统计文件不存在: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "rank" not in data:
        raise KeyError(f"文件 {path} 中缺少 'rank' 键。")
    return ChannelStats(
        tf=data["tf"],
        idf=data["idf"],
        score=data["score"],
        rank=data["rank"],
        counts=data.get("counts", {}),
        meta=data.get("meta", {}),
    )
