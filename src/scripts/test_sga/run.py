# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGA 变体短训实验入口。

用法：
    # 跑 P0 首选变体（fixed_sga_lb）
    python src/scripts/test_sga/run.py

    # 跑指定变体
    python src/scripts/test_sga/run.py --variant baseline
    python src/scripts/test_sga/run.py --variant spm_only
    python src/scripts/test_sga/run.py --variant fixed_sga_lb
    python src/scripts/test_sga/run.py --variant fixed_sga_res

    # 复用已有输出目录，只重跑 评估 + 注意力 + 报告（不训练）
    python src/scripts/test_sga/run.py --variant fixed_sga_lb --no-train

    # 构建冒烟（seed→构建→参数形状校验→退出，不训练）
    python src/scripts/test_sga/run.py --variant fixed_sga_lb --build-only

每个变体跑完短训后依次执行：测试集比赛评估（test.py）→ SGM 注意力统计
（analyze_sga.py）→ 生成 实验报告.md；全部产物归档到
output/{MMDD}-SHWX-test_sga-<variant>/。
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# 本脚本目录加入 sys.path，保证 flat import（python src/scripts/test_sga/run.py 直接运行）
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_VARIANT,
    EPOCHS,
    PROJECT_ROOT,
    SEED,
    VARIANTS,
    Variant,
    build_model_for_variant,
    output_dir_for,
    read_variant_config,
    run_attention,
    run_eval,
    short_train,
    write_report,
)


def _build_only_smoke(v: Variant) -> None:
    """构建冒烟：验证变体模型可构建、SGA 分支参数量不随变体改变（resume 兼容）。

    Args:
        v: 实验变体。
    """
    from rfdetr.models.backbone.sga import SGAEncoder

    # 先构建完整模型，验证 config→namespace→build_model→backbone 链路
    model = build_model_for_variant(v)
    print(f"[build-only] 变体 {v.name} 模型构建成功，总参数: "
          f"{sum(p.numel() for p in model.model.model.parameters()):,}")

    # 校验 SGA 分支参数量与参考（product 默认模式）完全一致 → 变体不改变参数形状
    if v.use_sga:
        enc = SGAEncoder(projector_scale=["P4"], hidden_dim=256, sem_channels=384, gate_mode=v.gate_mode)
        ref = SGAEncoder(projector_scale=["P4"], hidden_dim=256, sem_channels=384)
        n_enc = sum(p.numel() for p in enc.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        assert n_enc == n_ref, f"SGA 分支参数量不一致: {n_enc} vs {n_ref}"
        print(f"[build-only] SGA 分支参数量校验通过: {n_enc:,}（各变体一致，resume 兼容）")
    else:
        print("[build-only] 变体未启用 SGA，跳过 SGA 参数量校验")
    print("[build-only] 冒烟通过 ✓")


def main() -> None:
    """解析参数并编排一次短训实验（训练 → 评估 → 注意力 → 报告）。"""
    ap = argparse.ArgumentParser(description="SGA 变体小规模短训实验（P0 方向验证）")
    ap.add_argument("--variant", type=str, default=DEFAULT_VARIANT, help=f"变体名，可选: {sorted(VARIANTS)}")
    ap.add_argument("--date", type=str, default=datetime.date.today().strftime("%m%d"), help="输出目录日期（MMDD）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（各变体统一固定）")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="短训轮数")
    ap.add_argument("--resume", type=str, default="", help="可选的恢复 checkpoint（默认从 COCO 预训练起步）")
    ap.add_argument("--no-train", action="store_true", help="跳过训练，复用已有输出目录（只重跑评估/注意力/报告）")
    ap.add_argument("--no-eval", action="store_true", help="跳过测试集比赛评估")
    ap.add_argument("--no-attn", action="store_true", help="跳过 SGM 注意力分析")
    ap.add_argument("--build-only", action="store_true", help="仅构建模型并校验参数形状（冒烟），不训练")
    args = ap.parse_args()

    if args.variant not in VARIANTS:
        raise ValueError(f"未知变体: {args.variant}，可选: {sorted(VARIANTS)}")
    v = VARIANTS[args.variant]
    out_dir = output_dir_for(v, args.date)

    # ── 固定 seed：必须在任何模型/数据加载器构建之前生效 ────────────────
    # 模型随机初始化发生在 RFDETRMedium(...)，早于 TrainConfig.seed 的 fit-start 阶段，
    # 故此处必须先 seed_all 一次。
    from rfdetr.utilities.reproducibility import seed_all

    seed_all(args.seed)

    if args.build_only:
        _build_only_smoke(v)
        return

    # ── 训练 ──────────────────────────────────────────────────────────
    if args.no_train:
        if not out_dir.exists():
            raise FileNotFoundError(
                f"--no-train 但输出目录不存在: {out_dir}\n请先不带 --no-train 跑一次，或改用其它 --date。"
            )
        print(f"[i] 跳过训练，复用已有输出目录: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        short_train(v, out_dir, seed=args.seed, epochs=args.epochs, resume=args.resume)

    # ── 测试集评估（比赛指标）─────────────────────────────────────────
    if not args.no_eval:
        print(f"\n[评估] 运行测试集比赛指标评估 -> {out_dir}")
        run_eval(out_dir)

    # ── SGM 注意力分析（SGA 变体）─────────────────────────────────────
    if not args.no_attn and v.use_sga:
        print(f"\n[注意力] 运行 SGM 注意力统计 -> {out_dir}")
        run_attention(v, out_dir)
    elif not args.no_attn:
        print("\n[注意力] baseline 无 SGM，跳过注意力分析")

    # ── 实验报告 ──────────────────────────────────────────────────────
    write_report(v, out_dir, args.date, seed=args.seed, epochs=args.epochs)

    # 打印归档摘要
    cfg = read_variant_config(out_dir)
    print("\n" + "=" * 60)
    print(f"实验归档完成: {out_dir}")
    print(f"  - training_config.json / metrics.csv / checkpoints（训练栈自动生成）")
    print(f"  - test_result.txt / confusion_matrix.png / FP / FN（评估生成）")
    if v.use_sga:
        print(f"  - attention_stats.txt / attention_vis/（注意力分析生成）")
    print(f"  - 实验报告.md（本次生成）")
    if cfg:
        print(f"  - 变体配置: {cfg.get('model_config', {}).get('sga_gate_mode', '?')} / "
              f"residual={cfg.get('model_config', {}).get('sga_fusion_residual', '?')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
