# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""阶段 0 第二步：训练语义方向投影 f_sem + 计算 TF-IDF 通道统计 + 对齐校验门控。

输入：``data/fsem_collect_0805.pt``（stage0_collect_features.py 收集的 base 类
matched query 特征，绝不含少样本舰船类）。

流程：
1. 用 CLIP 文本编码器为全部 25 类编码文本向量（``encode_class_text_embeddings``）。
2. 训练 ``FSemProjection``（两层仿射 + tanh，768→256）：对称 InfoNCE，把
   f_sem(t_c) 与类别 c 的 matched query 特征方向对齐（只更新 f_sem 参数）。
3. 阶段 0 对齐校验（val 集，held-out）：同类 ``cos(mean(h_c), s_c)`` 显著高于
   跨类。**通过标准：mean_align ≥ 0.3 且 mean_gap ≥ 0.2**，不通过则以非零退出码
   结束（人工门控，不进入 Stage-2 训练）。
4. 计算 TF-IDF 通道统计（``compute_channel_tfidf``，全部 base 实例，与 f_sem 同源）。
5. 保存 ``data/fsem_shwx.pt``、``data/channel_stats_shwx.pt`` 与 stage0 报告。

用法：
    python src/scripts/ret-sscl/stage0_train_fsem.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）

from rfdetr.sscl.channel_stats import compute_channel_tfidf, save_channel_stats
from rfdetr.sscl.fsem import FSemProjection, evaluate_alignment, save_fsem_artifacts
from rfdetr.sscl.prompts import SHWX_CLASS_PROMPTS
from rfdetr.sscl.semantic_matrix import DEFAULT_CLIP_MODEL_NAME, encode_class_text_embeddings

# ============================================================================
# 配置 —— 在这里修改
# ============================================================================

# stage0_collect_features.py 的输出（train 模式 + 增广版本，与 Stage-2 训练分布一致）
FEATURES_PATH = str(Path("data/fsem_collect_0805_train_aug.pt").resolve())

# 输出产物（供 stage2_train.py 的 semantic_fsem_path / semantic_channel_stats_path 引用）
OUTPUT_FSEM = str(Path("data/fsem_shwx.pt").resolve())
OUTPUT_STATS = str(Path("data/channel_stats_shwx.pt").resolve())
REPORT_PATH = str(Path("output/0809-SemHead-stage0/stage0_report.json").resolve())

# novel 类 = 舰船 0-3（只影响最终 S 矩阵的行，不参与 f_sem 训练）
NOVEL_CLASSES = [0, 1, 2, 3]
BASE_CLASSES = list(range(4, 25))
NUM_CLASSES = 25
CLIP_MODEL = DEFAULT_CLIP_MODEL_NAME

# f_sem 训练超参
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 512
TAU = 0.07  # InfoNCE 温度
VAL_RATIO = 0.2  # 类别级 held-out 校验比例
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _symmetric_infonce(
    proj: FSemProjection,
    features: torch.Tensor,
    labels: torch.Tensor,
    text_base: torch.Tensor,
    base_ids: list[int],
    tau: float,
) -> torch.Tensor:
    """对称 InfoNCE 损失（图像→文本 + 文本→图像两个方向）。

    - 图像→文本：以每个实例为锚，正样本 = 其类别的文本向量，负样本 = 全部 base
      类文本向量。
    - 文本→图像：以每个 base 类文本为锚，正样本 = 该类的全部实例，负样本 = batch
      内全部实例（同类的其他实例也计入分母，但分子把正样本全部求和，即
      logsumexp 形式的多正样本 softmax）。

    Args:
        proj: f_sem 投影模块。
        features: batch 特征 ``[N, d]``。
        labels: 对应类别标签 ``[N]``（值域在 base 类内）。
        text_base: base 类文本向量 ``[C_base, 768]``。
        base_ids: base 类 id 列表（text_base 的行索引到类别 id 的映射）。
        tau: 温度。

    Returns:
        标量损失。
    """
    v = F.normalize(features, dim=-1)  # [N, d]
    f_t = F.normalize(proj(text_base), dim=-1)  # [C_base, d]

    # 图像→文本：logits [N, C_base]，每行正样本列 = 该实例类别的列位置
    logits_it = v @ f_t.T / tau
    pos_idx = torch.tensor([base_ids.index(int(c)) for c in labels], device=labels.device)
    loss_it = F.cross_entropy(logits_it, pos_idx)

    # 文本→图像：logits [C_base, N]，每行（类锚）正样本 = 该类的实例
    logits_ti = f_t @ v.T / tau  # [C_base, N]
    log_denom = torch.logsumexp(logits_ti, dim=-1)  # 全实例分母
    loss_ti = torch.zeros(logits_ti.shape[0], device=logits_ti.device)
    for row, c in enumerate(base_ids):
        pos_mask = labels == c
        if pos_mask.any():
            log_num = torch.logsumexp(logits_ti[row][pos_mask], dim=-1)
            loss_ti[row] = log_denom[row] - log_num
    loss_ti = loss_ti.mean()

    return 0.5 * (loss_it + loss_ti)


def main() -> None:
    """训练 f_sem + 计算通道统计 + 阶段 0 对齐校验。"""
    torch.manual_seed(SEED)
    random.seed(SEED)

    # --- 1. 加载收集的特征，按类别分组 ---
    data = torch.load(FEATURES_PATH, map_location="cpu", weights_only=True)
    features_all = data["features"]
    labels_all = data["labels"]
    class_names: list[str] = data["class_names"]
    print(f"加载 {features_all.shape[0]} 个 base 类实例（特征维度 {features_all.shape[1]}）")

    # 类别级 8:2 划分（val 不参与训练，作为 held-out 对齐校验）
    train_idx: list[int] = []
    val_idx: list[int] = []
    features_by_class_all: dict[int, torch.Tensor] = {}
    for c in BASE_CLASSES:
        idx_c = (labels_all == c).nonzero().squeeze(-1).tolist()
        random.shuffle(idx_c)
        split = int(len(idx_c) * (1 - VAL_RATIO))
        train_idx += idx_c[:split]
        val_idx += idx_c[split:]
        features_by_class_all[c] = features_all[labels_all == c]
    val_labels = labels_all[val_idx].clone()
    print(f"训练实例 {len(train_idx)} / 校验实例 {len(val_idx)}")

    # --- 2. 编码全部 25 类文本向量（f_sem 训练只取 base 类；S 矩阵需要全类别行） ---
    print(f"用 CLIP 编码类别文本向量（模型: {CLIP_MODEL}）...")
    text_all = encode_class_text_embeddings(SHWX_CLASS_PROMPTS, model_name=CLIP_MODEL, device=DEVICE)
    text_all = text_all.float().cpu()
    text_base = text_all[BASE_CLASSES].to(DEVICE)

    # --- 3. 训练 f_sem ---
    proj = FSemProjection(text_dim=text_all.shape[1], hidden_dim=512, out_dim=features_all.shape[1])
    proj.to(DEVICE).train()
    optimizer = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=1e-4)
    num_steps = (len(train_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"训练 f_sem: {EPOCHS} epochs × {num_steps} steps/batch={BATCH_SIZE} τ={TAU}")

    for epoch in range(EPOCHS):
        order = train_idx.copy()
        random.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx_batch = order[start : start + BATCH_SIZE]
            feats = features_all[idx_batch].to(DEVICE)
            # idx_batch 是 labels_all 的原始索引，直接用 labels_all 取值
            labels_b = labels_all[idx_batch].to(DEVICE)
            optimizer.zero_grad()
            loss = _symmetric_infonce(proj, feats, labels_b, text_base, BASE_CLASSES, TAU)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx_batch)
        print(f"  epoch {epoch + 1}/{EPOCHS}: loss={total_loss / len(order):.4f}")
    proj.eval()

    # --- 4. 阶段 0 对齐校验（val 集，held-out） ---
    # 先计算全 25 类语义方向（evaluate_alignment 按类别 id 直接索引 S 行）
    with torch.no_grad():
        s_all = F.normalize(proj(text_all.to(DEVICE)), dim=-1).cpu()  # [25, d]
    # val 特征 = features_all[val_idx]，与 val_labels（压缩后的标签）形状对齐
    val_features = features_all[val_idx]
    val_by_class = {c: val_features[val_labels == c] for c in BASE_CLASSES}
    val_filtered = {c: f for c, f in val_by_class.items() if f.numel() > 0}
    align_report = evaluate_alignment(val_filtered, s_all)

    print("\n===== 阶段 0 对齐校验（held-out val）=====")
    for c, v in sorted(align_report["per_class"].items()):
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  类 {c} ({name}): 同类cos={v['align']:.4f} 跨类均值cos={v['cross_mean']:.4f} gap={v['gap']:.4f}")
    print(f"  均值: mean_align={align_report['mean_align']:.4f} mean_gap={align_report['mean_gap']:.4f}")

    # 通过标准：mean_align ≥ 0.3 且 mean_gap ≥ 0.2（方案文档 §5 阶段 0）
    gate_pass = align_report["mean_align"] >= 0.3 and align_report["mean_gap"] >= 0.2
    if not gate_pass:
        print("\n[门控] 阶段 0 对齐校验未通过！不会产出可用的 f_sem 产物。")
        print("建议排查：CLIP prompt 质量、f_sem 结构/超参、收集特征质量。")
        raise SystemExit(1)
    print("[门控] 阶段 0 对齐校验通过。")

    # --- 5. 保存 f_sem 产物（s_all 已在步骤 4 计算） ---
    meta = {
        "class_names": class_names,
        "epochs": EPOCHS,
        "align_report": align_report,
        "checkpoint": data["checkpoint"],
        "tau": TAU,
    }
    save_fsem_artifacts(OUTPUT_FSEM, s_all, proj.state_dict(), meta)
    print(f"语义方向矩阵已保存: {OUTPUT_FSEM}（形状 {tuple(s_all.shape)}）")

    # --- 6. 计算 TF-IDF 通道统计（全部 base 实例，与 f_sem 同源） ---
    stats = compute_channel_tfidf(features_by_class_all, s_all)
    # class_ids 记录每行 rank 对应的类别 id（build 时需把 novel 类扩成 base 聚合画像）
    stats.meta = {"class_names": class_names, "checkpoint": data["checkpoint"], "class_ids": BASE_CLASSES}
    save_channel_stats(OUTPUT_STATS, stats)
    # 掩码预览：θ 初始化为 d、有效 τ = d/16 时 M 应处于"接近全保留但梯度存活"区间
    d = features_all.shape[1]
    tau_eff = d / 16.0
    m_preview = torch.sigmoid((float(d) - stats.rank.float()) / tau_eff)
    print(
        f"通道统计已保存: {OUTPUT_STATS}；θ=d, τ=d/16 时 M 均值={m_preview.mean().item():.4f}"
        f"（预期 0.85~0.95，最差通道≈0.5 保证梯度存活）"
    )

    # --- 7. 保存阶段 0 报告 ---
    report = {
        "num_base_instances": int(features_all.shape[0]),
        "per_class_instances": data["num_instances_per_class"],
        "align_report": align_report,
        "gate_pass": gate_pass,
        "fsem_path": OUTPUT_FSEM,
        "channel_stats_path": OUTPUT_STATS,
    }
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"阶段 0 报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
