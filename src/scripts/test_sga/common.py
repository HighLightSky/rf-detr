# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGA 变体短训实验的公共配置与辅助函数。

供 ``run.py`` 调用，集中管理：
    - 实验变体注册表（baseline / spm_only / fixed_sga_lb / fixed_sga_res）
    - 共享的短训超参（25 epoch、固定 seed、EMA 等）
    - 模型构建 / 短训 / 测试评估 / 注意力分析 / 实验报告拼装

背景：实验报告（output/0805-SHWX-SGA-rfdetr/实验报告.md §六）发现当前 SGA 的
SGM 门控在目标处把 SPM 纹理压到 ≈0（框内注意力均值 0.0148），导致小目标召回下降。
P0 修复思路是给门控加保底（lower_bound/residual）并把融合改成残差（保留 projector
语义基线），所有变体参数形状一致，可与既有 checkpoint 兼容。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── 路径：把项目根与 src 目录加入 sys.path（照 test.py 的方式）───────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../src/scripts/test_sga/common.py → 项目根
SRC_DIR = PROJECT_ROOT / "src"
for _p in (SRC_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rfdetr.datasets.aug_configs import AUG_AERIAL  # noqa: E402
from rfdetr.variants import RFDETRMedium  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
#  实验变体注册表
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Variant:
    """单个实验变体的模型配置描述。

    Attributes:
        name: 变体名（用作输出目录后缀）。
        use_sga: 是否启用 SGM 混合编码器分支。
        gate_mode: SGM 门控模式（product/lower_bound/residual/ones）。
        fusion_residual: 融合是否残差保底（fused = feats[i] + gamma*delta）。
        residual_gamma: 残差融合系数。
        attn_bias: SGM 注意力 logits 初值偏置（>0 使初始注意力≈全通；默认 0 = 从 0.5 起步）。
        description: 中文描述（写入实验报告）。
        projector_scale: 参与 decoder 的金字塔等级（默认 P4 单级）。
        fusion_mode: 融合方式（concat = 现有 concat+conv；semantic_film = 语义条件残差调制）。
        residual_alpha_init: semantic_film 各 P 级可学习残差系数 α_s 的初值。
        analyze_attn: 是否对该变体运行 SGM 注意力分析（ones/无 SGM 门控的变体设为 False）。
    """

    name: str
    use_sga: bool
    gate_mode: str
    fusion_residual: bool
    residual_gamma: float
    description: str
    attn_bias: float = 0.0
    projector_scale: list[str] = field(default_factory=lambda: ["P4"])
    fusion_mode: str = "concat"
    residual_alpha_init: float = 1e-3
    analyze_attn: bool = True


VARIANTS: dict[str, Variant] = {
    # 对照：完全关闭 SGA，行为等同原模型
    "baseline": Variant("baseline", False, "product", False, 0.1, "关闭 SGA（对照）"),
    # 消融 A（报告 §五 修订）：M 固定全 1 + 残差融合，与 fixed_sga_lb 唯一差别就是门控（恒 1 vs [0.5,1]）
    "spm_only": Variant("spm_only", True, "ones", True, 0.1, "SPM-only 消融：M 全 1 + 残差融合（隔离门控轴）"),
    # P0 首选：下界门控 [0.5,1] + 残差融合，即使注意力在目标处≈0 也保留一半 SPM
    "fixed_sga_lb": Variant("fixed_sga_lb", True, "lower_bound", True, 0.1, "下界门控 [0.5,1] + 残差融合（P0 首选）"),
    # P0 备选：残差门控 det+det*M（保底更强，SPM 幅值放大至 1~2 倍）+ 残差融合
    "fixed_sga_res": Variant("fixed_sga_res", True, "residual", True, 0.1, "残差门控 det+det*M + 残差融合"),
    # 治本第一版：product 直接门控 + 注意力初值偏置 +2（初始 M≈0.88 全通）+ 残差融合
    "attn_bias": Variant(
        "attn_bias",
        True,
        "product",
        True,
        0.1,
        "注意力初值偏置 +2，product 直接门控 + 残差融合（治本第一版）",
        attn_bias=2.0,
    ),
    # ── 多尺度 P3/P4 验证（docs/改进方案-SGM-encoder/SGA多尺度语义-细节融合优化：10ep短训验证方案.md）──
    # E0：基线 / 纯 DINO 多级 / 真实 CNN 细节（H1 判据）
    "baseline_p4": Variant(
        "baseline_p4",
        False,
        "product",
        False,
        0.1,
        "当前基线：单级 P4 decoder，无 SGA（多尺度实验对照）",
        projector_scale=["P4"],
    ),
    "vit_p3p4": Variant(
        "vit_p3p4",
        False,
        "product",
        False,
        0.1,
        "纯 DINO projector 多级 P3/P4 decoder（无 CNN 细节，验证多级 decoder 本身）",
        projector_scale=["P3", "P4"],
    ),
    "spm_p3p4": Variant(
        "spm_p3p4",
        True,
        "ones",
        True,
        0.1,
        "真实 C2/P3 细节 + 残差 concat 融合，SGM 门控固定全 1（H1 判据）",
        projector_scale=["P3", "P4"],
        analyze_attn=False,
    ),
    # E1：语义条件残差调制（H2 判据，§3.2）
    "semantic_film_p3p4": Variant(
        "semantic_film_p3p4",
        True,
        "ones",
        False,
        0.1,
        "语义条件残差调制（GN + 通道-空间调制 + 可学习残差 α_s），无单通道 SGM（H2 判据）",
        projector_scale=["P3", "P4"],
        fusion_mode="semantic_film",
        residual_alpha_init=1e-3,
        analyze_attn=False,
    ),
}
DEFAULT_VARIANT = "fixed_sga_lb"  # 默认跑 P0 首选变体


# ══════════════════════════════════════════════════════════════════════
#  数据 / 模型 / 训练共享常量
# ══════════════════════════════════════════════════════════════════════
DATASET_DIR = "/home/liu/datasets/SHWX-dataset-dict"  # SHWX 数据集（yolo 格式）
DATASET_FILE = "yolo"
NUM_CLASSES = 25
MODEL_RESOLUTION = 640  # RFDETRMedium 默认分辨率
PROJECTOR_SCALE = ["P4"]  # 单级 fused P4（Phase 1 设置，build_model_for_variant 按 v.projector_scale 覆盖）
USE_CFE = False  # 先不开 CFE，避免变量混合

# 多尺度验证（0807）的输出根目录：output/0807test_sga/<variant>/
EXPERIMENT_ROOT = PROJECT_ROOT / "output/0807test_sga"

# 短训超参（小规模方向验证：COCO 预训练起步、固定 seed、25 epoch）
SEED = 0
EPOCHS = 10
# OOM 调整（2026-08-07）：单级 P4 batch16 已占 ~21GB/24GB，两级 decoder 必然超限。
# 降 batch 到 8 并翻倍累积步数，有效 batch 仍 = 64，超参口径不漂移（文档 §6 风险对策）。
BATCH_SIZE = 8
NUM_WORKERS = 12
LR = 1e-4  # 基础学习率
LR_ENCODER = 1.5e-4  # 编码器学习率
WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 8  # 有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 64
CLIP_MAX_NORM = 0.1
LR_DROP = 15  # ≈原 100ep/60 的比例（60%）
WARMUP_EPOCHS = 2.0
MOSAIC_P = 0.8
EMA_DECAY = 0.993  # 与历史实验一致，EMA 用于 checkpoint_best_total 选点
EVAL_INTERVAL = 1  # 短训期间每轮验证，便于观察动态


# ══════════════════════════════════════════════════════════════════════
#  模型构建 / 训练
# ══════════════════════════════════════════════════════════════════════
def build_model_for_variant(v: Variant) -> RFDETRMedium:
    """按变体配置构建 RFDETRMedium 模型。

    Args:
        v: 实验变体。

    Returns:
        构建好的模型（SGA 分支按变体门控/融合参数初始化，其余从 COCO 预训练权重加载）。
    """
    return RFDETRMedium(
        num_classes=NUM_CLASSES,
        resolution=MODEL_RESOLUTION,
        gradient_checkpointing=True,
        use_sga=v.use_sga,
        use_cfe=USE_CFE,
        projector_scale=v.projector_scale,
        sga_gate_mode=v.gate_mode if v.use_sga else "product",
        sga_fusion_residual=v.fusion_residual,
        sga_residual_gamma=v.residual_gamma,
        sga_attn_bias=v.attn_bias,
        sga_fusion_mode=v.fusion_mode,
        sga_residual_alpha_init=v.residual_alpha_init,
    )


def output_dir_for(v: Variant, date: str) -> Path:
    """返回该变体的输出目录（output/0807test_sga/<variant>）。

    Args:
        v: 实验变体。
        date: 日期字符串（MMDD，仅用于日志；目录结构按 EXPERIMENT_ROOT 固定）。

    Returns:
        输出目录绝对路径。
    """
    return EXPERIMENT_ROOT / v.name


def short_train(v: Variant, out_dir: Path, seed: int, epochs: int, resume: str = "") -> None:
    """执行小规模短训。

    训练栈会自动把 training_config.json / metrics.csv / 各 checkpoint 写入 out_dir，
    无需手工归档。所有变体用相同超参与 seed，保证 head-to-head 可比。

    Args:
        v: 实验变体。
        out_dir: 输出目录（自动创建）。
        seed: 随机种子（需在调用本函数前已通过 seed_all 固定）。
        epochs: 训练轮数。
        resume: 可选恢复 checkpoint（默认空串 = 从预训练权重起步）。
    """
    model = build_model_for_variant(v)
    print(
        f"\n[短训] 变体={v.name} | gate_mode={v.gate_mode} | fusion_residual={v.fusion_residual} "
        f"| seed={seed} | epochs={epochs} | 输出={out_dir}"
    )
    model.train(
        dataset_dir=str(Path(DATASET_DIR).resolve()),
        dataset_file=DATASET_FILE,
        output_dir=str(out_dir),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        lr_encoder=LR_ENCODER,
        weight_decay=WEIGHT_DECAY,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        clip_max_norm=CLIP_MAX_NORM,
        lr_drop=LR_DROP,
        warmup_epochs=WARMUP_EPOCHS,
        tensorboard=True,
        wandb=False,
        device="cuda",
        devices=1,
        num_nodes=1,
        resume=resume,
        eval_interval=EVAL_INTERVAL,
        use_ema=True,
        ema_decay=EMA_DECAY,
        seed=seed,
        compute_val_loss=False,
        aug_config=AUG_AERIAL,
        mosaic_p=MOSAIC_P,
    )


# ══════════════════════════════════════════════════════════════════════
#  评估 / 注意力分析（subprocess 复用 test.py 与 analyze_sga.py）
# ══════════════════════════════════════════════════════════════════════
def run_eval(out_dir: Path) -> None:
    """对指定实验目录运行测试集比赛评估（写 test_result.txt / 混淆矩阵 / FP / FN）。

    Args:
        out_dir: 实验输出目录（须含 checkpoint_best_total.pth 与 training_config.json）。
    """
    subprocess.run(
        [sys.executable, str(SRC_DIR / "scripts" / "test.py"), "--exp-dir", str(out_dir)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def run_attention(v: Variant, out_dir: Path) -> None:
    """对指定 checkpoint 运行 SGM 注意力统计（写 attention_stats.txt / attention_vis/）。

    仅 SGA 变体（含 SemanticGuidingModule）有意义；baseline 不调用。

    Args:
        v: 实验变体（决定有效门控的映射）。
        out_dir: 实验输出目录。
    """
    ckpt = out_dir / "checkpoint_best_total.pth"
    if not ckpt.exists():
        print(f"[w] 缺少 checkpoint，跳过注意力分析: {ckpt}")
        return
    subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "scripts" / "analyze_sga.py"),
            "--checkpoint",
            str(ckpt),
            "--gate-mode",
            v.gate_mode,
            "--residual-gamma",
            str(v.residual_gamma),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  实验报告拼装
# ══════════════════════════════════════════════════════════════════════
_GROUP_LINE = re.compile(
    r"^(?P<name>\w+)\s+TP=(?P<tp>\d+)\s+FP=(?P<fp>\d+)\s+FN=(?P<fn>\d+)\s+"
    r"Recall=(?P<recall>[\d.]+)\s+FDR=(?P<fdr>[\d.]+)\s+Precision=(?P<prec>[\d.]+)"
)
_GROUP_NAMES = ("all", "ship", "aircraft", "vehicle")


def _parse_test_result(test_txt: Path) -> dict[str, dict[str, float]]:
    """解析 test_result.txt 中的大类指标（all/ship/aircraft/vehicle）。

    Args:
        test_txt: test_result.txt 路径（不存在则返回空 dict）。

    Returns:
        {大类名: {tp/fp/fn/recall/fdr/prec}}。
    """
    if not test_txt.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for line in test_txt.read_text(encoding="utf-8").splitlines():
        m = _GROUP_LINE.match(line)
        if m and m.group("name") in _GROUP_NAMES:
            d = m.groupdict()
            out[d["name"]] = {
                "tp": float(d["tp"]),
                "fp": float(d["fp"]),
                "fn": float(d["fn"]),
                "recall": float(d["recall"]),
                "fdr": float(d["fdr"]),
                "prec": float(d["prec"]),
            }
    return out


def _read_best_val(metrics_csv: Path) -> dict[str, Any]:
    """从 metrics.csv 读取最佳 val/mAP_50_95 与最佳 val/ema_mAP_50_95（含 epoch）。

    Args:
        metrics_csv: metrics.csv 路径（不存在则返回空结构）。

    Returns:
        dict: best_reg / best_ema（各含 value/epoch）、last_recall / last_precision /
        last_f1（末行验证集指标，可能为 None）。
    """
    empty: dict[str, Any] = {
        "best_reg": {"value": None, "epoch": None},
        "best_ema": {"value": None, "epoch": None},
        "last_recall": None,
        "last_precision": None,
        "last_f1": None,
    }
    if not metrics_csv.exists():
        return empty
    best_reg = {"value": None, "epoch": None}
    best_ema = {"value": None, "epoch": None}
    last_row: dict[str, str] = {}
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            epoch = float(row["epoch"])
            reg = row.get("val/mAP_50_95")
            if reg not in (None, ""):
                val = float(reg)
                if best_reg["value"] is None or val > best_reg["value"]:
                    best_reg = {"value": val, "epoch": int(epoch)}
            ema = row.get("val/ema_mAP_50_95")
            if ema not in (None, ""):
                val = float(ema)
                if best_ema["value"] is None or val > best_ema["value"]:
                    best_ema = {"value": val, "epoch": int(epoch)}
            last_row = row
    return {
        "best_reg": best_reg,
        "best_ema": best_ema,
        "last_recall": last_row.get("val/recall"),
        "last_precision": last_row.get("val/precision"),
        "last_f1": last_row.get("val/F1"),
    }


def _parse_attention_stats(attn_txt: Path) -> list[str]:
    """读取注意力统计文本中的关键行（框内/背景/有效门控/全局均值）。

    Args:
        attn_txt: attention_stats.txt 路径。

    Returns:
        关键统计行列表。
    """
    if not attn_txt.exists():
        return []
    return [
        ln
        for ln in attn_txt.read_text(encoding="utf-8").splitlines()
        if ("全局 mean" in ln or "目标框内" in ln or "有效门控" in ln or "框内-背景" in ln)
    ]


def _baseline_test_result(out_dir: Path, date: str) -> dict[str, dict[str, float]]:
    """确定对照 baseline 的 test_result。

    优先取本 runner 的 baseline_p4（同 seed/协议，受控对比），回退到历史
    output/0805-SHWX-data-expand-rfdetr-baseline 的结果。

    Args:
        out_dir: 当前变体的输出目录。
        date: 日期字符串（MMDD）。

    Returns:
        解析出的 baseline 大类指标。
    """
    own = EXPERIMENT_ROOT / "baseline_p4" / "test_result.txt"
    if own.exists():
        return _parse_test_result(own)
    legacy = PROJECT_ROOT / "output/0805-SHWX-data-expand-rfdetr-baseline" / "test_result.txt"
    return _parse_test_result(legacy)


def _fmt_metric_row(name: str, data: dict[str, float] | None) -> str:
    """把单个大类的指标格式化为一行。"""
    if not data:
        return f"{name:<10s} (无数据)"
    return (
        f"{name:<10s} TP={data['tp']:<6.0f} FP={data['fp']:<6.0f} FN={data['fn']:<6.0f} "
        f"Recall={data['recall']:.4f} FDR={data['fdr']:.4f} Prec={data['prec']:.4f}"
    )


def write_report(v: Variant, out_dir: Path, date: str, seed: int, epochs: int) -> None:
    """拼装并写入 实验报告.md（实验设置 + val 动态 + 测试集对比 + 注意力机制）。

    Args:
        v: 实验变体。
        out_dir: 输出目录。
        date: 日期字符串（MMDD）。
        seed: 随机种子。
        epochs: 训练轮数。
    """
    lines: list[str] = []
    sep = "=" * 80
    lines += [
        f"# 实验报告：SGA 变体短训验证 — {v.name}",
        "",
        f"> 生成时间：{date}  |  用途：小规模方向验证（P0 修复），确认后再全量微调",
        "",
        sep,
        "## 一、实验设置",
        sep,
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 变体 | `{v.name}` |",
        f"| 描述 | {v.description} |",
        f"| 模型 | RFDETRMedium @ {MODEL_RESOLUTION}（dinov2_windowed_small 编码器） |",
        f"| use_sga / use_cfe | {v.use_sga} / {USE_CFE} |",
        f"| sga_gate_mode | `{v.gate_mode}` |",
        f"| sga_fusion_residual | {v.fusion_residual} |",
        f"| sga_residual_gamma | {v.residual_gamma} |",
        f"| sga_fusion_mode | `{v.fusion_mode}` |",
        f"| sga_residual_alpha_init | {v.residual_alpha_init} |",
        f"| projector_scale | {v.projector_scale} |",
        "| 数据 | SHWX（yolo） |",
        "| 起步 | COCO 预训练权重（SGA 分支随机初始化） |",
        f"| 训练轮数 / lr / lr_encoder / lr_drop | {epochs} / {LR} / {LR_ENCODER} / {LR_DROP} |",
        f"| batch × grad_accum / seed | {BATCH_SIZE} × {GRAD_ACCUM_STEPS} / {seed} |",
        f"| warmup / mosaic / EMA | {WARMUP_EPOCHS} / {MOSAIC_P} / {EMA_DECAY} |",
        "",
    ]

    # ── 二、验证集动态 ───────────────────────────────────────────────
    val = _read_best_val(out_dir / "metrics.csv")
    lines += [sep, "## 二、验证集训练动态（metrics.csv）", sep, ""]
    lines += [
        f"- 最佳 val/mAP_50_95 : {val['best_reg']['value']:.4f}（epoch {val['best_reg']['epoch']}）"
        if val["best_reg"]["value"] is not None
        else "- 最佳 val/mAP_50_95 : （无数据）",
        f"- 最佳 val/ema_mAP_50_95 : {val['best_ema']['value']:.4f}（epoch {val['best_ema']['epoch']}）"
        if val["best_ema"]["value"] is not None
        else "- 最佳 val/ema_mAP_50_95 : （无数据）",
    ]
    if val["last_recall"]:
        lines.append(
            f"- 末轮 val/recall : {float(val['last_recall']):.4f} | "
            f"val/precision : {float(val['last_precision']):.4f} | "
            f"val/F1 : {float(val['last_f1']):.4f}"
        )
    lines.append("")

    # ── 二·五、融合特征统计（冒烟脚本 feature_stats.txt，若存在）─────────
    feat_txt = out_dir / "feature_stats.txt"
    if feat_txt.exists():
        lines += [sep, "## 二·五、融合特征统计（feature_stats.txt）", sep, ""]
        lines += feat_txt.read_text(encoding="utf-8").splitlines()
        lines.append("")

    # ── 三、测试集结果（固定 conf=0.25，比赛口径）────────────────────
    own = _parse_test_result(out_dir / "test_result.txt")
    base = _baseline_test_result(out_dir, date)
    lines += [sep, "## 三、测试集结果对比（conf=0.25，比赛指标）", sep, ""]
    if own:
        lines.append(
            f"{'类别':<10s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
            f"{'Recall':>8s} {'FDR':>7s} {'Prec':>7s} {'ΔRecall':>9s} {'ΔFP':>7s}"
        )
        lines.append("-" * 70)
        for gname in _GROUP_NAMES:
            data = own.get(gname)
            if not data:
                continue
            b = base.get(gname)
            d_recall = f"{data['recall'] - b['recall']:+.4f}" if b else "—"
            d_fp = f"{data['fp'] - b['fp']:+.0f}" if b else "—"
            lines.append(
                f"{gname:<10s} {data['tp']:>6.0f} {data['fp']:>6.0f} {data['fn']:>6.0f} "
                f"{data['recall']:>8.4f} {data['fdr']:>7.4f} {data['prec']:>7.4f} "
                f"{d_recall:>9s} {d_fp:>7s}"
            )
        lines.append("")
        lines.append(f"对照 baseline: {base and base.get('all', {}).get('recall') and '本 runner baseline_p4'}")
    else:
        lines.append("（未生成 test_result.txt，可能 --no-eval 跳过或评估失败）")
    lines.append("")

    # ── 四、注意力机制（SGA 变体）────────────────────────────────────
    if v.use_sga:
        attn_lines = _parse_attention_stats(out_dir / "attention_stats.txt")
        lines += [sep, "## 四、SGM 注意力机制检查（attention_stats.txt）", sep, ""]
        if attn_lines:
            lines += [f"- {ln}" for ln in attn_lines]
        else:
            lines.append("（未生成注意力统计，可能 --no-attn 跳过或 checkpoint 无 SGM）")
        lines.append("")

    # ── 五、方向判断提示 ─────────────────────────────────────────────
    lines += [
        sep,
        "## 五、方向判断（对照报告 §五 标准）",
        sep,
        "",
        "1. all Recall 是否回到 baseline 附近（ΔRecall ≈ 0 或为正）；",
        "2. ship Recall 是否不低于 baseline；",
        "3. vehicle/FSC 是否不再出现 5~10pp 级别的召回缺口；",
        "4. FP 是否仍低于或接近 baseline；",
        "5. 注意力机制：下界门控下框内『有效门控』应 ≥0.5（不再 ≈0，目标处 SPM 未被关掉）。",
        "",
        "若以上均满足，则该变体方向正确，可进入全量微调（src/scripts/train.py 改常量）。",
        "",
    ]
    lines.append("")

    report_path = out_dir / "实验报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] 实验报告已保存: {report_path}")


def read_variant_config(out_dir: Path) -> dict[str, Any]:
    """读取已有输出目录的 training_config.json（供 --no-train 重跑时核对）。"""
    cfg_path = out_dir / "training_config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))
