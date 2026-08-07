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

    # 多尺度 P3/P4 验证（0807 批次，输出 output/0807test_sga/<variant>/）
    python src/scripts/test_sga/run.py --variant baseline_p4
    python src/scripts/test_sga/run.py --variant vit_p3p4
    python src/scripts/test_sga/run.py --variant spm_p3p4
    python src/scripts/test_sga/run.py --variant semantic_film_p3p4

    # 复用已有输出目录，只重跑 评估 + 注意力 + 报告（不训练）
    python src/scripts/test_sga/run.py --variant fixed_sga_lb --no-train

    # 构建冒烟（seed→构建→warm-start→前向/反向→特征统计→退出，不训练）
    python src/scripts/test_sga/run.py --variant fixed_sga_lb --build-only

每个变体跑完短训后依次执行：测试集比赛评估（test.py）→ SGM 注意力统计
（analyze_sga.py，仅 analyze_attn=True 的变体）→ 生成 实验报告.md；全部产物归档到
output/0807test_sga/<variant>/。
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

# 本脚本目录加入 sys.path，保证 flat import（python src/scripts/test_sga/run.py 直接运行）
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_VARIANT,
    EPOCHS,
    MODEL_RESOLUTION,
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


def _check_multiscale_warm_start(nn_model: Any) -> None:
    """多级 P3/P4 warm-start 校验（文档 §6.3.2）。

    与单级 P4 参考模型对比：
        - projector 的 P4 stage 权重应逐元素一致（共享权重已加载），P3 stage 随机初始化；
        - decoder 各 MSDeformAttn 级数应翻倍，且 level-0 权重与单级一致（warm-start 平铺）。

    Args:
        nn_model: 已构建的 LWDETR 底层模块（``model.model.model``）。
    """
    import torch

    from rfdetr.models.ops.modules import MSDeformAttn

    # 单级 P4 参考模型（默认 use_sga=False，projector/decoder 结构与多级版本共享槽位）
    ref = build_model_for_variant(Variant("_ref", False, "product", False, 0.1, "warm-start 参考"))
    ref_nn = ref.model.model  # 与主模型 model.model.model 等价：RFDETR→ModelContext→LWDETR
    v_sd, ref_sd = nn_model.state_dict(), ref_nn.state_dict()

    # (a) projector P4 stage 权重相等（按 C2f 首卷积输入通道指纹匹配 P4 槽位）
    def _stage_channels(sd: dict[str, Any]) -> dict[int, int]:
        return {
            int(k.split(".projector.stages.")[1].split(".")[0]): int(v.shape[1])
            for k, v in sd.items()
            if ".projector.stages." in k and k.endswith(".0.cv1.conv.weight") and ".stages_sampling." not in k
        }

    ref_ch = _stage_channels(ref_sd)
    v_ch = _stage_channels(v_sd)
    assert len(ref_ch) == 1, f"参考单级模型 projector stages 异常: {ref_ch}"
    ref_idx, ref_cin = next(iter(ref_ch.items())), next(iter(ref_ch.values()))
    p4_idx = [i for i, c in v_ch.items() if c == ref_cin]
    assert p4_idx, f"多级模型未找到 P4 projector stage（通道指纹 {ref_cin}）: {v_ch}"
    p4_idx = p4_idx[0]
    for key in [k for k in ref_sd if f".projector.stages.{ref_idx}." in k]:
        v_key = key.replace(f".projector.stages.{ref_idx}.", f".projector.stages.{p4_idx}.")
        if v_key in v_sd:
            assert torch.allclose(v_sd[v_key].float(), ref_sd[key].float()), f"P4 projector stage 权重未对齐: {key}"
    assert len(v_ch) == 2, f"P3P4 模型应比单级多一个 projector stage: {v_ch}"
    print(f"[build-only] warm-start 校验: P4 projector stage（槽位 {p4_idx}）与单级预训练一致，P3 stage 随机初始化 ✓")

    # (b) decoder MSDeformAttn 级数 warm-start：级数翻倍且 level-0 权重与单级一致
    v_attn = {n: m for n, m in nn_model.named_modules() if isinstance(m, MSDeformAttn)}
    ref_attn = {n: m for n, m in ref_nn.named_modules() if isinstance(m, MSDeformAttn)}
    checked = 0
    target_levels: int | None = None
    for name, m in v_attn.items():
        rm = ref_attn.get(name)
        if rm is None or rm.n_levels != 1:
            continue  # 只对比单级参考中存在的模块
        assert m.n_levels >= 2, f"MSDeformAttn {name} 级数未扩展: {m.n_levels}"
        v_s0 = m.sampling_offsets.weight.view(m.n_heads, m.n_levels, m.n_points, 2, -1)[:, 0].reshape(-1)
        r_s0 = rm.sampling_offsets.weight.view(rm.n_heads, rm.n_levels, rm.n_points, 2, -1)[:, 0].reshape(-1)
        assert torch.allclose(v_s0.float(), r_s0.float()), f"MSDeformAttn {name} level-0 权重未 warm-start"
        target_levels = m.n_levels
        checked += 1
    assert checked >= 1, "未找到可校验的 MSDeformAttn 模块"
    assert target_levels is not None
    print(f"[build-only] MSDeformAttn warm-start 校验通过（{checked} 个模块，级数 1→{target_levels}）")
    del ref, ref_nn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _build_only_smoke(v: Variant) -> None:
    """构建冒烟：验证变体模型可构建、多级 warm-start 正确、前向/反向正常、新分支有非零梯度。

    覆盖文档 §6.3 冒烟检查项 1~5：
        1. config→namespace→build_model→backbone 构建链路 + COCO 预训练加载；
        2. P3P4 变体的 P4 共享权重已加载、P3 随机初始化、MSDeformAttn 级数 warm-start；
        3. backbone.forward_export 输出 P3/P4 特征 shape 正确（无 padding 导出路径）；
        4. 反向后 SPM / 融合层 / film α_s 均有非零有限梯度；
        5. 打印各 P 级融合特征 mean/std/L2 与 film α_s 初值 → 写入 feature_stats.txt（§4.1.5）。

    Args:
        v: 实验变体。
    """
    import torch

    from rfdetr.models.backbone.sga import SGAEncoder

    # 1) 构建完整模型（触发 COCO 预训练加载 + 多级 warm-start）
    model = build_model_for_variant(v)
    nn_model = model.model.model
    print(f"[build-only] 变体 {v.name} 模型构建成功，总参数: {sum(p.numel() for p in nn_model.parameters()):,}")

    # 2) SGA 分支参数量一致性（仅同 fusion_mode 之间可比，resume 不跨 fusion_mode）
    if v.use_sga:
        enc = SGAEncoder(
            projector_scale=v.projector_scale,
            hidden_dim=256,
            sem_channels=384,
            gate_mode=v.gate_mode,
            fusion_mode=v.fusion_mode,
            residual_alpha_init=v.residual_alpha_init,
        )
        ref = SGAEncoder(
            projector_scale=v.projector_scale,
            hidden_dim=256,
            sem_channels=384,
            gate_mode=v.gate_mode,
            fusion_mode=v.fusion_mode,
        )
        n_enc = sum(p.numel() for p in enc.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        assert n_enc == n_ref, f"SGA 分支参数量不一致: {n_enc} vs {n_ref}"
        print(f"[build-only] SGA(fusion={v.fusion_mode}) 分支参数量校验通过: {n_enc:,}")
    else:
        print("[build-only] 变体未启用 SGA，跳过 SGA 参数量校验")

    # 3) 多级 warm-start 校验（仅 P3P4 变体）
    if len(v.projector_scale) > 1:
        _check_multiscale_warm_start(nn_model)

    # 4) 前向 + 反向（backbone.forward_export 无 padding 的导出路径）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_model = nn_model.to(device)
    backbone = nn_model.backbone[0]
    x = torch.randn(2, 3, MODEL_RESOLUTION, MODEL_RESOLUTION, device=device)
    feats, _masks, _cross = backbone.forward_export(x)
    level_stride = {"P3": 8, "P4": 16, "P5": 32}
    for i, lvl in enumerate(v.projector_scale):
        exp = MODEL_RESOLUTION // level_stride[lvl]
        assert feats[i].shape == (2, 256, exp, exp), (
            f"{lvl} 特征 shape 异常: {tuple(feats[i].shape)}，期望 {(2, 256, exp, exp)}"
        )
    print(f"[build-only] forward_export 特征 shape 校验通过: {[tuple(f.shape) for f in feats]}")

    # 各 P 级特征都要进入反向，确保每级新分支都有梯度
    sum(f.sum() for f in feats).backward()

    # 5) 特征统计 + 梯度断言 → feature_stats.txt
    stats_lines: list[str] = []
    for i, lvl in enumerate(v.projector_scale):
        f = feats[i].detach().float()
        stats_lines.append(
            f"[P{lvl[1:]}] mean={f.mean():.4f} std={f.std():.4f} L2={f.norm(2).item():.4f} shape={tuple(f.shape)}"
        )
    if v.use_sga:
        sga = backbone.sga
        named = {n: p for n, p in sga.named_parameters()}
        grad_ok = [
            n
            for n, p in named.items()
            if p.grad is not None and p.grad.abs().sum().item() > 0 and bool(torch.isfinite(p.grad).all().item())
        ]
        # 关键分支必须有有效梯度（文档 §6.3.4：反向后 spm/P3 融合/α_s 均有非零有限梯度）
        assert "spm.conv2.conv.weight" in grad_ok, "SPM 分支（spm.conv2.conv.weight）无有效梯度"
        if v.fusion_mode == "semantic_film":
            assert any(n.startswith("film_blocks.") and n.endswith(".alpha") and n in grad_ok for n in named), (
                "semantic_film 的 α_s 无有效梯度"
            )
            for i, lvl in enumerate(v.projector_scale):
                alpha = named[f"film_blocks.{i}.alpha"]
                ag = alpha.grad.item() if alpha.grad is not None else None
                stats_lines.append(f"[film {lvl}] α_s={alpha.item():.6f} grad={ag}")
        else:
            assert any(n.startswith("fusion_layers.") and n in grad_ok for n in named), "concat 融合层无有效梯度"
        print(f"[build-only] 反向梯度检查通过，SGA 分支有梯度参数数: {len(grad_ok)}/{len(named)}")
    else:
        print("[build-only] 无 SGA 分支，跳过梯度断言")

    # 写入 feature_stats.txt（冒烟初值统计，供实验报告引用）
    out_dir = output_dir_for(v, datetime.date.today().strftime("%m%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print("[build-only] 特征统计已写入 feature_stats.txt:\n" + "\n".join(stats_lines))
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

    # ── SGM 注意力分析（仅门控真正使用 M 的 SGA 变体）────────────────────
    # analyze_attn=False：ones 门控或无 SGM 的变体（spm_p3p4 / semantic_film_p3p4），
    # 其注意力图不参与损失、无梯度流，统计无意义，跳过。
    if not args.no_attn and v.use_sga and v.analyze_attn:
        print(f"\n[注意力] 运行 SGM 注意力统计 -> {out_dir}")
        run_attention(v, out_dir)
    elif not args.no_attn:
        print("\n[注意力] 该变体无有效 SGM 门控（baseline / ones / semantic_film），跳过注意力分析")

    # ── 实验报告 ──────────────────────────────────────────────────────
    write_report(v, out_dir, args.date, seed=args.seed, epochs=args.epochs)

    # 打印归档摘要
    cfg = read_variant_config(out_dir)
    print("\n" + "=" * 60)
    print(f"实验归档完成: {out_dir}")
    print("  - training_config.json / metrics.csv / checkpoints（训练栈自动生成）")
    print("  - test_result.txt / confusion_matrix.png / FP / FN（评估生成）")
    if v.use_sga:
        print("  - attention_stats.txt / attention_vis/（注意力分析生成）")
    print("  - 实验报告.md（本次生成）")
    if cfg:
        print(
            f"  - 变体配置: {cfg.get('model_config', {}).get('sga_gate_mode', '?')} / "
            f"residual={cfg.get('model_config', {}).get('sga_fusion_residual', '?')}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
