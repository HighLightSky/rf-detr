# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""多模态原型引导模块（位置查询选择 + 内容查询增强）。

挂在 ``Transformer`` 上（``transformer.proto_guidance``），作用在两个环节：
1. 位置引导（``position_score``）：计算每个 encoder token 与多模态原型的
   余弦相似度分数，供 two-stage top-k 选择以 ``linear + lambda * proto``
   的 residual 形式合并，改变哪些 token 被选为 position query；
2. 内容引导（``enhance_content``）：按选中 token 关联的类别，用该类别的
   多槽位原型做交叉注意力（gate 残差）增强 decoder 的 content query，
   注入类别先验。

防 no-op 设计：
- 所有注入均为 residual + 近恒等初始化（lambda/gamma 初始 0.05，
  gate bias 初始 logit(0.05)），关闭开关时数学恒等于原版；
- lambda/gamma 按 ``current_epoch`` 线性 warmup（由 module_model 注入）；
- 原型分类辅助损失（criterion 的 ``loss_proto_labels``）把梯度直接送达
  余弦打分分支，绕开 top-k 离散选择。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）
from torch import Tensor, nn

from rfdetr.sscl.proto_guidance.artifacts import (
    load_proto_artifacts,
    validate_proto_artifacts,
)
from rfdetr.sscl.proto_guidance.fusion import (
    DEFAULT_TEXT_DIM,
    _content_context,
    build_fusion,
)
from rfdetr.sscl.prototype_bank import SlotPrototypeBank
from rfdetr.sscl.prototype_diagnostics import prototype_geometry
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# 采样节流区间默认值（与 semantic_monitor_log_interval 同量级）
DEFAULT_MONITOR_INTERVAL = 100

# margin 标准差相对线性 objectness 的目标比例；给 top-k 留出稳定主干。
POSITION_SCORE_STD_RATIO = 0.1


def _lerp(start: float, end: float, t: float) -> float:
    """线性插值（t 自动 clamp 到 [0, 1]）。"""
    return start + (end - start) * min(max(t, 0.0), 1.0)


class ProtoGuidance(nn.Module):
    """多模态原型引导模块。

    Args:
        num_classes: 前景类别数（不含 background）。
        hidden_dim: 特征维度（= decoder hidden dim）。
        text_dim: CLIP 文本向量维度。
        fusion_mode: 融合模式，v1 仅支持 ``"simple"``。
        num_slots: 每类视觉子原型槽位数 M。
        target_classes: 位置打分 max 集合；空列表 = 全部类别
            （``selected_class`` 始终在全部类别上 argmax）。
        tau_p: 原型相似度温度（``cosine / tau_p``）。
        lambda_pos_init: 位置 residual 权重初始值。
        lambda_pos_max: 位置 residual 权重上限（warmup 目标）。
        gamma_content_init: 内容注入强度初始值。
        gamma_content_max: 内容注入强度上限（warmup 目标）。
        gate_bias_init: 内容 gate 的 bias 初始值（logit 空间，
            默认 logit(0.05)≈-2.944 使 gate 初始 ≈0.05 保留梯度）。
        w_v_init: 融合时视觉原型权重初始值。
        w_t_init: 融合时文本原型权重初始值。
        warmup_epochs: lambda/gamma 线性 warmup 的 epoch 数（0 直接到上限）。
        position_enabled: 位置引导开关（关闭时 top-k 恒等于原版）。
        content_enabled: 内容引导开关（关闭时 tgt 原样通过）。
        aux_loss_enabled: 原型分类辅助损失开关（配合 criterion
            ``loss_proto_labels`` 使用，仅影响监控/配置语义）。
        visual_ema_update: 训练期是否用 GT 框特征 EMA 更新视觉原型（默认关）。
        visual_ema_momentum: EMA 更新动量。

    Attributes:
        current_epoch: 当前训练 epoch（float，由 module_model 注入，
            驱动 lambda/gamma warmup；不进 state_dict）。
        last_stats: 最近一次前向的监控统计字典（全 detach）。
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        text_dim: int = DEFAULT_TEXT_DIM,
        fusion_mode: Literal["simple", "gated"] = "simple",
        num_slots: int = 10,
        target_classes: list[int] | None = None,
        tau_p: float = 0.1,
        lambda_pos_init: float = 0.05,
        lambda_pos_max: float = 1.0,
        gamma_content_init: float = 0.05,
        gamma_content_max: float = 1.0,
        gate_bias_init: float = -2.944,
        w_v_init: float = 0.3,
        w_t_init: float = 0.7,
        warmup_epochs: float = 2.0,
        position_enabled: bool = True,
        content_enabled: bool = False,
        aux_loss_enabled: bool = False,
        visual_ema_update: bool = False,
        visual_ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}。")
        if num_slots < 1:
            raise ValueError(f"num_slots 必须 >= 1，收到 {num_slots}。")
        if tau_p <= 0.0:
            raise ValueError(f"tau_p 必须 > 0，收到 {tau_p}。")
        if warmup_epochs < 0.0:
            raise ValueError(f"warmup_epochs 必须 >= 0，收到 {warmup_epochs}。")

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.tau_p = float(tau_p)
        self.lambda_pos_init = float(lambda_pos_init)
        self.lambda_pos_max = float(lambda_pos_max)
        self.gamma_content_init = float(gamma_content_init)
        self.gamma_content_max = float(gamma_content_max)
        self.warmup_epochs = float(warmup_epochs)
        self.position_enabled = bool(position_enabled)
        self.content_enabled = bool(content_enabled)
        self.aux_loss_enabled = bool(aux_loss_enabled)
        self.visual_ema_update = bool(visual_ema_update)

        self.target_classes = sorted(
            {int(c) for c in (target_classes or []) if 0 <= int(c) < num_classes}
        )

        # 视觉原型库：复用 SlotPrototypeBank 作 buffer 容器（离线产物 copy_ 进来）
        self.visual_bank = SlotPrototypeBank(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            max_slots=num_slots,
            multi_slot_classes=list(range(num_classes)),
        )
        # CLIP 文本原型（离线构建，加载后不更新）
        self.register_buffer("P_t_clip", torch.zeros(num_classes, text_dim))
        # 融合模块（含 proj_v/proj_t/proj_token 与融合权重）
        self.fusion = build_fusion(
            fusion_mode=fusion_mode,
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            w_v_init=w_v_init,
            w_t_init=w_t_init,
        )
        # 内容增强 gate：MLP([tgt, ctx]) -> 标量，bias 初始 logit(gate_bias_init)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(float(gate_bias_init))
        # 训练期 EMA 更新使用的动量（未启用时不参与计算）
        self.visual_ema_momentum = float(visual_ema_momentum)

        # warmup 调度状态（python 属性，不进 state_dict）
        self.current_epoch = 0.0
        # 最近一次前向的监控统计（collect_stats 组装，全 detach）
        self.last_stats: dict[str, Tensor] | None = None
        # enhance_content 记录的最后 gate 值（[bs, Q] detach），供 collect_stats 读取
        self._last_gate: Tensor | None = None

    @classmethod
    def build(
        cls,
        num_classes: int,
        hidden_dim: int,
        artifacts_path: str | Path,
        text_dim: int = DEFAULT_TEXT_DIM,
        fusion_mode: Literal["simple", "gated"] = "simple",
        num_slots: int = 10,
        target_classes: list[int] | None = None,
        tau_p: float = 0.1,
        lambda_pos_init: float = 0.05,
        lambda_pos_max: float = 1.0,
        gamma_content_init: float = 0.05,
        gamma_content_max: float = 1.0,
        gate_bias_init: float = -2.944,
        w_v_init: float = 0.3,
        w_t_init: float = 0.7,
        warmup_epochs: float = 2.0,
        position_enabled: bool = True,
        content_enabled: bool = False,
        aux_loss_enabled: bool = False,
        visual_ema_update: bool = False,
        visual_ema_momentum: float = 0.99,
    ) -> ProtoGuidance | None:
        """构建模块并加载离线原型产物。

        产物缺失或形状不匹配时降级：记录警告并返回 ``None``（模块不挂载，
        transformer 侧恒等短路，评估/训练不受影响）。

        Args:
            参数含义与 ``__init__`` 一致；``artifacts_path`` 为离线产物路径
            （``stage0_build_proto_guidance.py`` 产出）。

        Returns:
            ``ProtoGuidance`` 实例，或产物不可用时返回 ``None``。
        """
        try:
            data = load_proto_artifacts(artifacts_path)
            data = validate_proto_artifacts(
                data,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                text_dim=text_dim,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning(
                f"[ProtoGuidance] 离线产物不可用（{exc}），模块不挂载，保持原版行为。"
            )
            return None

        module = cls(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            fusion_mode=fusion_mode,
            num_slots=num_slots,
            target_classes=target_classes,
            tau_p=tau_p,
            lambda_pos_init=lambda_pos_init,
            lambda_pos_max=lambda_pos_max,
            gamma_content_init=gamma_content_init,
            gamma_content_max=gamma_content_max,
            gate_bias_init=gate_bias_init,
            w_v_init=w_v_init,
            w_t_init=w_t_init,
            warmup_epochs=warmup_epochs,
            position_enabled=position_enabled,
            content_enabled=content_enabled,
            aux_loss_enabled=aux_loss_enabled,
            visual_ema_update=visual_ema_update,
            visual_ema_momentum=visual_ema_momentum,
        )
        module.visual_bank.prototypes.copy_(data["visual_prototypes"])
        module.visual_bank.slot_valid_mask.copy_(data["valid_slots"])
        # 有效槽位计数置 1（标识已初始化，供 get_normalized_* 等接口判断）
        module.visual_bank.slot_num_updates.masked_fill_(data["valid_slots"], 1)
        module.visual_bank.num_updates.fill_(1)
        module.P_t_clip.copy_(data["text_prototypes"])
        logger.info(
            f"[ProtoGuidance] 多模态原型已加载: {artifacts_path}（"
            f"类别数 {num_classes}，槽位 {num_slots}，数据集 {data['meta'].get('dataset', '?')}）"
        )
        return module

    # ------------------------------------------------------------------
    # 原型融合与打分
    # ------------------------------------------------------------------
    def fused_prototypes(self) -> tuple[Tensor, Tensor]:
        """融合多模态原型。

        Returns:
            ``(P_mm, valid_slots)``：``P_mm`` 形状 ``[C, M, d]``（逐槽位
            L2 归一化），``valid_slots`` 形状 ``[C, M]``（bool）。
        """
        return (
            self.fusion(
                self.visual_bank.prototypes,
                self.P_t_clip,
                self.visual_bank.slot_valid_mask,
            ),
            self.visual_bank.slot_valid_mask,
        )

    def lambda_pos_effective(self) -> float:
        """当前 epoch 下位置 residual 的有效权重（warmup 插值）。"""
        t = 0.0 if self.warmup_epochs <= 0 else self.current_epoch / self.warmup_epochs
        return _lerp(self.lambda_pos_init, self.lambda_pos_max, t)

    def gamma_content_effective(self) -> float:
        """当前 epoch 下内容注入的有效强度（warmup 插值）。"""
        t = 0.0 if self.warmup_epochs <= 0 else self.current_epoch / self.warmup_epochs
        return _lerp(self.gamma_content_init, self.gamma_content_max, t)

    def position_score(self, mem_gidx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """计算 token 与多模态原型的相似度分数。

        Args:
            mem_gidx: 单组 encoder 输出 token 特征 ``[bs, N, d]``。

        Returns:
            ``(proto_logits, proto_score, selected_class)``：
            - ``proto_logits``: ``[bs, N, C]``（``cosine / tau_p``，多槽位取 max）；
            - ``proto_score``: ``[bs, N]``（目标类别与最强竞争类别的 cosine margin）；
            - ``selected_class``: ``[bs, N]``（全部类别上 argmax，供内容增强）。
        """
        p_mm, valid = self.fused_prototypes()  # [C, M, d], [C, M]
        token_h = F.normalize(self.fusion.projectors.proj_token(mem_gidx), dim=-1)
        # 逐槽位余弦 [bs, N, C, M] -> 无效槽位置 -inf -> max over M
        sim = torch.einsum("bnd,cmd->bncm", token_h, p_mm)
        sim = sim.masked_fill(~valid.unsqueeze(0).unsqueeze(0), torch.finfo(sim.dtype).min)
        class_sim = sim.max(dim=-1).values  # [bs, N, C]
        class_valid = valid.any(dim=-1)
        class_sim = class_sim.masked_fill(
            ~class_valid.view(1, 1, -1),
            -1.0,
        )
        proto_logits = class_sim / self.tau_p
        selected_class = class_sim.argmax(dim=-1)  # [bs, N]

        if self.target_classes:
            target_ids = torch.as_tensor(self.target_classes, device=mem_gidx.device)
            target_top = class_sim.index_select(-1, target_ids).max(dim=-1).values
            competitor_mask = torch.ones(
                self.num_classes, dtype=torch.bool, device=mem_gidx.device
            )
            competitor_mask[target_ids] = False
            if bool(competitor_mask.any()):
                competitor_top = class_sim[..., competitor_mask].max(dim=-1).values
                proto_score = target_top - competitor_top
            else:
                proto_score = target_top
        else:
            top_values = class_sim.topk(min(2, self.num_classes), dim=-1).values
            proto_score = top_values[..., 0]
            if self.num_classes > 1:
                proto_score = proto_score - top_values[..., 1]
        return proto_logits, proto_score, selected_class

    def class_confidence(self, proto_logits: Tensor) -> Tensor:
        """把原型分类概率转换为去除类别数影响的置信度。"""
        probabilities = F.softmax(proto_logits, dim=-1)
        max_probability = probabilities.max(dim=-1).values
        uniform = 1.0 / max(proto_logits.shape[-1], 1)
        return ((max_probability - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)

    def calibrate_position_score(self, proto_score: Tensor, linear_score: Tensor) -> Tensor:
        """将 margin 居中并缩放到线性 objectness 的标准差。"""
        if proto_score.shape != linear_score.shape:
            raise ValueError("proto_score 与 linear_score 的形状必须一致。")
        proto_centered = proto_score - proto_score.mean(dim=-1, keepdim=True)
        proto_std = proto_centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        linear_std = linear_score.detach().std(dim=-1, keepdim=True, unbiased=False)
        return proto_centered / proto_std * linear_std * POSITION_SCORE_STD_RATIO

    def enhance_content(
        self,
        tgt_q: Tensor,
        selected_class: Tensor,
        confidence: Tensor | None = None,
    ) -> Tensor:
        """用关联类别的多槽位原型增强内容查询（gate 残差）。

        Args:
            tgt_q: 内容查询 ``[bs, Q, d]``（two-stage 选中的部分）。
            selected_class: 每个查询关联的类别 ``[bs, Q]``。
            confidence: 原型分类置信度 ``[bs, Q]``；为空时兼容旧行为。

        Returns:
            增强后的内容查询（形状不变）。``content_enabled=False`` 时原样返回。
        """
        if not self.content_enabled:
            return tgt_q
        p_mm, valid = self.fused_prototypes()  # [C, M, d], [C, M]
        selected_class = selected_class.clamp(0, self.num_classes - 1)
        p_q = p_mm[selected_class]  # [bs, Q, M, d]
        valid_q = valid[selected_class]  # [bs, Q, M]
        ctx = _content_context(tgt_q, p_q, valid_q)  # [bs, Q, d]
        gate_in = torch.cat([tgt_q, ctx], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))  # [bs, Q, 1]
        gamma = self.gamma_content_effective()
        if confidence is not None:
            confidence = confidence.to(dtype=gate.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            gate = gate * confidence
        self._last_gate = gate.squeeze(-1).detach()
        return tgt_q + gamma * gate * ctx

    def update_visual_ema(self, features: list[Tensor], boxes: list[Tensor], labels: list[Tensor]) -> None:
        """用 GT 框在 backbone/projector 特征上的 masked 平均更新视觉原型（EMA）。

        v1 默认关闭（``visual_ema_update=False``），作为扩展项。实现为
        P4 单尺度特征（与默认 ``projector_scale=["P4"]`` 对齐）按归一化
        box 裁剪区域做平均池化，再喂给 ``visual_bank.update``。

        Args:
            features: backbone 输出特征列表（每项 ``[bs, d, H, W]``，取第一级）。
            boxes: 每图 GT 框列表（每项 ``[N_c, 4]``，归一化 xywh）。
            labels: 每图 GT 类别列表（每项 ``[N_c]``）。
        """
        if not self.visual_ema_update or not features:
            return
        feature = features[0]
        bs, d, h, w = feature.shape
        instance_features: list[Tensor] = []
        instance_labels: list[Tensor] = []
        for img_idx in range(bs):
            if img_idx >= len(boxes) or boxes[img_idx].numel() == 0:
                continue
            img_feat = feature[img_idx]  # [d, H, W]
            for box, label in zip(boxes[img_idx], labels[img_idx], strict=False):
                cx, cy, bw, bh = box.tolist()
                x0 = max(int((cx - bw / 2) * w), 0)
                y0 = max(int((cy - bh / 2) * h), 0)
                x1 = min(int((cx + bw / 2) * w) + 1, w)
                y1 = min(int((cy + bh / 2) * h) + 1, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                region = img_feat[:, y0:y1, x0:x1]
                instance_features.append(region.mean(dim=(1, 2)))
                instance_labels.append(label.unsqueeze(0))
        if not instance_features:
            return
        feats = torch.stack(instance_features)
        labs = torch.cat(instance_labels)
        self.visual_bank.momentum = self.visual_ema_momentum
        self.visual_bank.update(feats, labs)

    # ------------------------------------------------------------------
    # 监控统计
    # ------------------------------------------------------------------
    def collect_stats(
        self,
        proto_logits_ts: Tensor,
        linear_logits_ts: Tensor,
        topk_indices: Tensor,
        linear_topk_indices: Tensor,
        selected_class: Tensor,
        proto_score_ts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """汇总一次前向的监控统计（全 detach，供 monitor 消费）。

        Args:
            proto_logits_ts: 选中 query 的原型 logits ``[bs, Q, C]``。
            linear_logits_ts: 选中 query 的线性 logits ``[bs, Q, C+1]``。
            topk_indices: 合并分数选出的 token 索引 ``[bs, Q]``。
            linear_topk_indices: 纯线性分数选出的 token 索引 ``[bs, Q]``。
            selected_class: 选中 query 的关联类别 ``[bs, Q]``。

        Returns:
            统计字典（键：``topk_overlap``/``proto_selected_ratio``/
            ``lambda_effective``/``proto_logits_pmax_mean``/``proto_logits_entropy_mean``/
            ``selected_class_hist``/``gate_mean``，均为 ``[bs]`` 或标量张量）。
        """
        stats: dict[str, Tensor] = {}
        bs = proto_logits_ts.shape[0]

        # top-k 重叠率：合并分数与纯线性分数选中的 token 交集比例（逐图）
        overlap = (
            topk_indices.unsqueeze(-1) == linear_topk_indices.unsqueeze(-2)
        ).any(dim=-1).float().mean(dim=-1)  # [bs]
        stats["topk_overlap"] = overlap
        stats["proto_selected_ratio"] = 1.0 - overlap

        # 有效 lambda：按实际注入后的标准差计算，包含 margin 校准比例。
        proto_score = (
            proto_score_ts if proto_score_ts is not None else proto_logits_ts.max(dim=-1).values
        )
        linear_score = linear_logits_ts.max(dim=-1).values  # [bs, Q]
        calibrated = self.calibrate_position_score(proto_score, linear_score)
        lam_eff = (
            self.lambda_pos_effective()
            * calibrated.std(dim=-1, unbiased=False)
            / (linear_score.std(dim=-1, unbiased=False) + 1e-6)
        )
        stats["lambda_effective"] = lam_eff

        # 判别性：选中 query 原型 logits 的 max、熵和 cosine margin。
        stats["proto_logits_pmax_mean"] = proto_logits_ts.max(dim=-1).values.mean(dim=-1)
        softmax_p = F.softmax(proto_logits_ts, dim=-1)
        entropy = -(softmax_p * softmax_p.clamp_min(1e-12).log()).sum(dim=-1)
        stats["proto_logits_entropy_mean"] = entropy.mean(dim=-1)
        stats["proto_margin_mean"] = proto_score.mean(dim=-1)

        with torch.no_grad():
            fused_prototypes, valid = self.fused_prototypes()
            geometry = prototype_geometry(fused_prototypes, valid)
        for name, value in geometry.items():
            stats[f"prototype_{name}"] = value.to(dtype=torch.float32)

        # 类别分布：所有选中 query 的关联类别直方图 [C]
        hist = torch.bincount(selected_class.clamp(0, self.num_classes - 1).flatten(), minlength=self.num_classes)
        stats["selected_class_hist"] = hist.float()

        # 内容 gate 均值（content_enabled=False 时无值）
        if self._last_gate is not None:
            stats["gate_mean"] = self._last_gate.float().mean(dim=-1)
        self._last_gate = None

        self.last_stats = {k: v.detach() for k, v in stats.items()}
        return self.last_stats

    def describe_freeze(self) -> dict[str, int]:
        """返回模块可训练参数统计（供装配日志/冻结校验）。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}
