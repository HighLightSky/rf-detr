# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""语义分类头 SemanticResidual：在 decoder 分类 logits 上叠加语义残差增量。

分类决策由 ``class_embed = Linear(hidden_dim, num_classes)`` 完成，少样本类的
W_c 行从极少数匹配样本中硬估、方向噪声大。本模块为每类提供一个**不依赖样本量**
的语义先验，以残差增量形式叠加到原版 logits：

    outputs_class = class_embed(hs) + mask_delta + sem_delta
    mask_delta    = hs @ (W ⊙ (M − 1))ᵀ       通道掩码增量（M=全1 时恒为 0）
    sem_delta     = α ⊙ (hs @ Sᵀ)             语义方向增量（α=0 时恒为 0）

等价于把每类的总分类权重改写为 ``W̃_c = W_c ⊙ M_c + α_c·s_c``。其中：

- ``S``：冻结的语义方向矩阵（离线 f_sem 从 CLIP 文本向量投影，L2 归一化）；
- ``M``：通道掩码（离线 TF-IDF 统计排名 + 可学习阈值 θ_c），负责微观子类分离；
- ``α_c``：每类可学习混合系数（clamp [0, α_max]），语义注入强度；
- ``W``：``class_embed.weight`` 的引用，由前向传入（兼容 compile/DDP/类别数变化）。

设计原则：``enc_out_class_embed``（encoder proposal 分类）完全不经过本模块，
proposal 分类器保持原版线性头，query selection 不受语义化影响。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.sscl.channel_stats import load_channel_stats
from rfdetr.sscl.fsem import load_fsem_artifacts
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.config import TrainConfig

logger = get_logger()


class SemanticResidual(nn.Module):
    """语义残差模块（在线运行，离线产物装配）。

    Args:
        num_classes: 类别数 C。
        hidden_dim: decoder hidden dim d。
        mask_enabled: 是否启用通道掩码增量（``mask_delta``）。
        alpha_enabled: 是否启用语义方向增量（``sem_delta``）。
        alpha_max: α 上限（clamp）。
        mask_tau: 掩码软度 τ_mask。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        mask_enabled: bool = True,
        alpha_enabled: bool = True,
        alpha_max: float = 2.0,
        mask_tau: float = 1.0,
    ) -> None:
        super().__init__()
        # 每类混合系数 α 与掩码阈值 θ 为可学习参数（requires_grad 由装配时按配置设置）
        self.alpha = nn.Parameter(torch.zeros(num_classes))
        self.theta = nn.Parameter(torch.zeros(num_classes))
        # 冻结类 θ 掩码：True 表示该类 θ 冻结于初始值（梯度在 backward 时被置零）。
        # 用 buffer 保存（随 checkpoint 无损），配合 theta.register_hook 生效。
        self.register_buffer("_frozen_theta_mask", torch.zeros(num_classes, dtype=torch.bool))

        def _zero_frozen_theta_grad(grad: Tensor) -> Tensor:
            """把冻结类 θ 的梯度置零（θ 是整张量参数，无法按索引设 requires_grad）。"""
            mask = self._frozen_theta_mask
            if mask.any():
                grad = grad.clone()
                grad[mask] = 0.0
            return grad

        self.theta.register_hook(_zero_frozen_theta_grad)
        # 离线产物以 buffer 注册：S/rank 冻结，随 checkpoint 无损保存
        self.register_buffer("S", torch.zeros(num_classes, hidden_dim))
        self.register_buffer("rank", torch.ones(num_classes, hidden_dim, dtype=torch.float32))
        self.register_buffer("score", torch.zeros(num_classes, hidden_dim))
        self.register_buffer("idf", torch.zeros(hidden_dim))
        # 掩码/语义开关为普通 bool 属性（关闭时跳过对应子路径）
        self.mask_enabled = mask_enabled
        self.alpha_enabled = alpha_enabled
        self.alpha_max = alpha_max
        self.mask_tau = mask_tau
        # 最近一次前向的统计信息（detach，供监控读取，不驻留计算图）
        self.last_stats: dict[str, Any] = {}

    def forward(self, hs: Tensor, w: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """计算语义残差增量。

        Args:
            hs: decoder hidden states ``[..., Q, d]``（可为全层栈）。
            w: ``class_embed.weight`` ``[C, d]``（引用，前向传入）。

        Returns:
            ``(delta, stats)`` 二元组：
            - delta: 残差增量 ``[..., Q, C]``，与原版 logits 同形状相加。
            - stats: 监控统计字典（全部 detach）。
        """
        # 语义方向增量（α 在 forward 内 clamp：越界处梯度为 0，天然停学）
        if self.alpha_enabled:
            alpha = self.alpha.clamp(min=0.0, max=self.alpha_max)
            sem_delta = alpha * (hs @ self.S.T)  # [..., Q, C]
        else:
            alpha = self.alpha.detach().clamp(min=0.0, max=self.alpha_max)
            sem_delta = torch.zeros((*hs.shape[:-1], self.S.shape[0]), dtype=hs.dtype, device=hs.device)

        # 通道掩码增量（mask_delta = hs @ (W⊙(M−1))ᵀ，M=1 时恒为 0）
        if self.mask_enabled:
            # 掩码构造与 channel_stats.build_mask_from_rank 保持一致（单点定义在 forward 内）
            m = torch.sigmoid((self.theta.unsqueeze(-1) - self.rank) / self.mask_tau)  # [C, d]
            # w 可能含 background 末行（DETR 约定 C+1），掩码只作用于前景类行
            w_fg = w[: self.S.shape[0]]
            mask_delta = hs @ (w_fg * (m - 1.0)).T  # 原地不修改 w
        else:
            m = torch.ones_like(self.rank)
            mask_delta = torch.zeros_like(sem_delta)

        delta = mask_delta + sem_delta
        self.last_stats = {
            "mask_delta": mask_delta.detach(),
            "sem_delta": sem_delta.detach(),
            "M": m.detach(),
            "alpha": alpha.detach(),
            "theta": self.theta.detach(),
        }
        return delta, self.last_stats

    @classmethod
    def build(cls, cfg: "TrainConfig", num_classes: int, hidden_dim: int) -> "SemanticResidual":
        """从离线产物与训练配置装配语义残差模块。

        读取 ``fsem_shwx.pt``（S 矩阵）与 ``channel_stats_shwx.pt``（通道排名），
        校验形状后初始化 α/θ，并按配置设置可学习性（novel 类 θ 冻结）。

        Args:
            cfg: 训练配置（``semantic_*`` 字段）。
            num_classes: 类别数（须与离线产物一致）。
            hidden_dim: decoder hidden dim（须与离线产物一致）。

        Returns:
            装配完成的 ``SemanticResidual`` 实例。

        Raises:
            FileNotFoundError: 当 fsem/通道统计产物缺失时抛出。
            ValueError: 当产物形状与模型类别数/维度不一致时抛出。
        """
        if not cfg.semantic_fsem_path or not cfg.semantic_channel_stats_path:
            raise ValueError("启用语义头时必须指定 semantic_fsem_path 与 semantic_channel_stats_path。")
        fsem = load_fsem_artifacts(cfg.semantic_fsem_path)
        stats = load_channel_stats(cfg.semantic_channel_stats_path)

        s_matrix = fsem["S"].float()
        rank = stats.rank.float()
        if s_matrix.shape != (num_classes, hidden_dim):
            raise ValueError(
                f"语义方向矩阵形状 {tuple(s_matrix.shape)} 与模型类别数/维度 ({num_classes}, {hidden_dim}) 不一致，"
                "请用当前 checkpoint 重新收集并训练 f_sem。"
            )
        if rank.shape[1] != hidden_dim or rank.shape[0] > num_classes:
            raise ValueError(
                f"通道排名形状 {tuple(rank.shape)} 与模型类别数/维度不一致（应 ≤{num_classes} 行、{hidden_dim} 列）。"
            )

        # [掩码修复] 有效 τ = max(配置值, d/16)：rank 范围 [1, d] 很大而 τ=1 时
        # sigmoid 软带过窄（M≈1 梯度消失，θ 学不动）。d/16 保证 soft 带覆盖约 1/4
        # 通道，θ 梯度存活（E1a 观测到掩码从不收窄即此根因）。
        effective_tau = max(float(cfg.semantic_mask_tau), hidden_dim / 16.0)
        head = cls(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            mask_enabled=bool(cfg.semantic_mask_enabled),
            alpha_enabled=bool(cfg.semantic_alpha_enabled),
            alpha_max=float(cfg.semantic_alpha_max),
            mask_tau=effective_tau,
        )
        # 载入冻结的 S 矩阵与通道统计（rank/score/idf）
        head.S.copy_(F.normalize(s_matrix, dim=-1))
        head.idf.copy_(stats.idf.float())
        # 通道统计只含 base 类（行数 < 类别数）：novel 类用 base 类的平均通道画像
        # （按通道取 mean 后重新排名为 1..d 的置换），保证掩码对全部类别有定义。
        # base 类 id 从 stats.meta["class_ids"] 读取，缺失时退化为按序 0..R-1。
        stats_class_ids = list(stats.meta.get("class_ids", list(range(rank.shape[0]))))
        rank_full = torch.zeros(num_classes, hidden_dim, dtype=torch.float32)
        score_full = torch.zeros(num_classes, hidden_dim, dtype=torch.float32)
        for c in range(num_classes):
            if c in stats_class_ids:
                row = stats_class_ids.index(c)
                rank_full[c] = rank[row]
                score_full[c] = stats.score.float()[row]
            else:
                # novel 类：base 类逐通道均值排名画像（再排名保证是 1..d 的置换）
                base_mean_rank = rank.mean(dim=0)
                rank_full[c] = torch.argsort(torch.argsort(base_mean_rank, stable=True)).float() + 1.0
                score_full[c] = stats.score.float().mean(dim=0)
        head.rank.copy_(rank_full)
        head.score.copy_(score_full)

        # 初始化 α：base 类用 semantic_alpha_init，novel 类用 semantic_novel_alpha_init
        alpha_init = float(cfg.semantic_alpha_init)
        novel_classes = set(cfg.semantic_novel_classes or [])
        with torch.no_grad():
            head.alpha.fill_(alpha_init)
            for c in novel_classes:
                if 0 <= c < num_classes:
                    head.alpha[c] = float(cfg.semantic_novel_alpha_init)

        # 初始化 θ = d + semantic_theta_init·τ（默认 d，M 对最差通道≈0.5、对多数通道≈1，
        # 掩码梯度存活且初始接近"全保留"，随训练可收窄）。semantic_theta_init 为额外偏移。
        theta_init = float(cfg.semantic_theta_init) * effective_tau
        with torch.no_grad():
            head.theta.fill_(hidden_dim + theta_init)

        # 设置可学习性：α 由 alpha_enabled 与 alpha_learnable 共同决定（整张量）；
        # θ 在 frozen 类上冻结于初始值（梯度置零 hook），掩码关闭时整张量不可学习。
        frozen_theta = set(cfg.semantic_frozen_threshold_classes or [])
        head.alpha.requires_grad = bool(cfg.semantic_alpha_enabled and cfg.semantic_alpha_learnable)
        head.theta.requires_grad = bool(cfg.semantic_mask_enabled)
        head.set_frozen_theta_classes(sorted(frozen_theta))

        logger.info(
            f"[SemHead] 语义残差装配完成: C={num_classes}, d={hidden_dim}, "
            f"mask={head.mask_enabled}, alpha={head.alpha_enabled}, "
            f"α_init=({alpha_init}, novel={cfg.semantic_novel_alpha_init}), "
            f"θ_init={float(head.theta[0].item()):.1f}, "
            f"θ 冻结类={sorted(frozen_theta)}"
        )
        return head

    def set_frozen_theta_classes(self, classes: list[int]) -> None:
        """把指定类别的 θ 标记为冻结（梯度在 backward 时置零，保持初始值）。

        Args:
            classes: 冻结的类别索引列表。
        """
        with torch.no_grad():
            mask = torch.zeros(self.theta.numel(), dtype=torch.bool)
            for c in classes:
                if 0 <= c < self.theta.numel():
                    mask[c] = True
            self._frozen_theta_mask.copy_(mask)

    def describe_freeze(self) -> str:
        """返回冻结矩阵的文本摘要（α/θ 每类可训练状态），供日志输出。"""
        alpha_status = "可学习" if self.alpha.requires_grad else "冻结"
        frozen = [int(i) for i in range(self.theta.numel()) if bool(self._frozen_theta_mask[i])]
        learnable = [int(i) for i in range(self.theta.numel()) if not bool(self._frozen_theta_mask[i])]
        return f"α={alpha_status}, θ可学习={learnable}, θ冻结={frozen}"


def attach_from_checkpoint(model: nn.Module, state_dict: dict[str, Tensor]) -> bool:
    """从模型 state_dict 重建语义残差模块并挂载（评估/推理专用）。

    ``from_checkpoint`` 路径不经过 module_model 的装配逻辑，而 ``.pth`` 中
    语义头的 α/θ/S/rank 均以 ``semantic_residual.*`` 键保存。本函数检测到这些键
    时，用缓冲值（S/rank 形状可从 alpha/S 推断）构造占位模块并加载子状态，保证
    离线推理与训练前向一致。

    Args:
        model: LWDETR 模型（``model.model.model``）。
        state_dict: 从 checkpoint 加载的完整模型 state_dict。

    Returns:
        是否成功挂载（False 表示 checkpoint 不含语义头，无需处理）。
    """
    if not any(k.startswith("semantic_residual.") for k in state_dict):
        return False
    head = SemanticResidual.__new__(SemanticResidual)
    nn.Module.__init__(head)
    # 先按占位形状注册参数与 buffer，再加载子状态覆盖数值
    num_classes = int(state_dict["semantic_residual.alpha"].numel())
    hidden_dim = int(state_dict["semantic_residual.S"].shape[1])
    head.alpha = nn.Parameter(torch.zeros(num_classes))
    head.theta = nn.Parameter(torch.zeros(num_classes))
    head.register_buffer("_frozen_theta_mask", torch.zeros(num_classes, dtype=torch.bool))
    head.register_buffer("S", torch.zeros(num_classes, hidden_dim))
    head.register_buffer("rank", torch.ones(num_classes, hidden_dim))
    head.register_buffer("score", torch.zeros(num_classes, hidden_dim))
    head.register_buffer("idf", torch.zeros(hidden_dim))
    head.mask_enabled = True
    head.alpha_enabled = True
    head.alpha_max = 2.0
    head.mask_tau = 1.0
    head.last_stats = {}
    # 提取语义头子状态并加载（strict=True：缺少任何键即报错，保证无损重建）
    sub_state = {k[len("semantic_residual.") :]: v for k, v in state_dict.items() if k.startswith("semantic_residual.")}
    head.load_state_dict(sub_state, strict=True)
    model.semantic_residual = head
    logger.info("[SemHead] 已从 checkpoint 重建语义残差模块（评估路径）。")
    return True
