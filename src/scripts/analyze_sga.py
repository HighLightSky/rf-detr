# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SGA 猜想验证脚本（零训练，纯推理）。

用途：验证实验报告中的两个核心猜想，全程不需要重新训练：

  1. **阈值扫描**：对 SGA 与 Baseline 两个 checkpoint 在测试集上各推理一次
     （conf=0.05 收集全部框并保留分数），再对 conf ∈ [0.05, 0.50] 逐档重算
     比赛指标。若 SGA 的召回随阈值降低而追回，说明其掉点源于置信度分布偏移，
     可通过阈值/校准解决，无需重训。

  2. **SGM 注意力统计**：对 SGA 模型注册 SemanticGuidingModule 前向 hook，
     统计 sigmoid 注意力图的全局分布（均值/标准差/直方图）以及“目标框内 vs 背景”
     的注意力均值，并保存若干可视化热力图。若注意力集中在 0.5 附近，则证实
     “语义门控未学起来、SGA 退化为随机卷积分支”的猜想。

用法：
    # 只输出某个实验 checkpoint_best_total.pth 的 SGM 注意力热力图
    python src/scripts/analyze_sga.py --exp-dir output/0805-SHWX-SGA-rfdetr
    python src/scripts/analyze_sga.py --checkpoint output/xxx/checkpoint_best_total.pth

    # 不指定实验：默认做 SGA vs Baseline 阈值扫描 + SGA 注意力统计
    python src/scripts/analyze_sga.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as F  # noqa: N812
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════════
#  路径与配置
# ══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))

# 以独立模块名加载 test.py，复用其推理/评估基础设施（避免与标准库 test 冲突）
_spec = importlib.util.spec_from_file_location("rfdetr_test_script", SCRIPTS_DIR / "test.py")
test_mod = importlib.util.module_from_spec(_spec)
sys.modules["rfdetr_test_script"] = test_mod
_spec.loader.exec_module(test_mod)

SGA_CKPT = PROJECT_ROOT / "output/0805-SHWX-SGA-rfdetr/checkpoint_best_total.pth"
BASE_CKPT = PROJECT_ROOT / "output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth"
LOW_CONF = 0.05  # 单次推理的收集阈值（保留分数，供后续扫描）
CONF_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
BATCH_SIZE = 32
NUM_WORKERS = 12
ATTN_VIS_NUM = 8  # 保存热力图的图像数量

RESOLUTION = 640  # 模型推理分辨率（与训练 config.resolution 一致）
REPORT_PATH = PROJECT_ROOT / "output/0805-SHWX-SGA-rfdetr" / "conf_sweep_report.txt"
ATTN_VIS_DIR = PROJECT_ROOT / "output/0805-SHWX-SGA-rfdetr" / "attention_vis"


def _f1(result) -> float:
    """由 Recall/Precision 计算调和平均 F1。"""
    if result is None:
        return float("nan")
    denom = result.recall + result.precision
    return 2 * result.recall * result.precision / denom if denom > 0 else 0.0


class _AttnHook:
    """注册在 SemanticGuidingModule 上的前向 hook，收集 sigmoid 注意力图。

    每次前向调用产生 ``len(projector_scale)`` 张注意力图，每张为 ``[B,1,Hi,Wi]``，
    存储于 ``maps`` 列表，供推理主循环按 batch 弹出。
    """

    def __init__(self) -> None:
        self.maps: list[list[torch.Tensor]] = []

    def __call__(self, module: object, args: tuple, output: list[torch.Tensor]) -> None:
        # 在 inference_mode 下 hook 仍会触发，tensor 先 detach 再转 CPU 保存
        self.maps.append([m.detach().float().cpu() for m in output])


def run_inference(
    model,
    image_paths: list[Path],
    device: str,
    low_conf: float,
    collect_attn: bool = False,
) -> tuple[list[test_mod.BoxRecord], dict[str, torch.Tensor]]:
    """对测试集执行一次批量前向推理，返回 (预测框, 每图注意力图)。

    预处理与 ``model.predict`` 逐像素一致（uint8→float、antialias=False 缩放、
    归一化）。注意力收集仅在 SGA 模型（含 SemanticGuidingModule）上生效。

    Args:
        model: 已加载的 RFDETRMedium 实例。
        image_paths: 测试图像路径列表。
        device: 推理设备（如 ``"cuda:0"``）。
        low_conf: 收集全部框的置信度下界（通常设很小，供后续扫阈值）。
        collect_attn: 是否注册 hook 收集 SGM 注意力图。

    Returns:
        ``(pred_records, attn_by_image)``：预测框记录列表与 {image_id: 注意力图}。
    """
    resolution = int(model.model.resolution)
    dataset = test_mod._InferenceDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=test_mod.PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_mod._inference_collate,
        worker_init_fn=test_mod._worker_init_fn,
        persistent_workers=NUM_WORKERS > 0,
    )
    model.model.model = model.model.model.to(device)
    model.model.model.eval()
    model_dtype = next(model.model.model.parameters()).dtype
    means, stds = model.means, model.stds

    hook: _AttnHook | None = None
    if collect_attn:
        from rfdetr.models.backbone.sga import SemanticGuidingModule

        for m in model.model.model.modules():
            if isinstance(m, SemanticGuidingModule):
                hook = _AttnHook()
                m.register_forward_hook(hook)
                break
        if hook is None:
            print("[w] 模型中未找到 SemanticGuidingModule，跳过注意力收集", flush=True)

    pred_records: list[test_mod.BoxRecord] = []
    attn_by_image: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for stems, rgb_tensors, orig_sizes in loader:
            gpu_images = [
                F.resize(
                    t.to(device, non_blocking=True).to(model_dtype).div_(255.0),
                    (resolution, resolution),
                    antialias=False,
                )
                for t in rgb_tensors
            ]
            batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)
            predictions = model.model.model(batch_tensor)
            target_sizes = torch.tensor(orig_sizes, device=device)
            results = model.model.postprocess(predictions, target_sizes=target_sizes)

            for stem, result in zip(stems, results):
                keep = result["scores"] > low_conf
                for xyxy, cid, sc in zip(
                    result["boxes"][keep].cpu().numpy(),
                    result["labels"][keep].cpu().numpy(),
                    result["scores"][keep].cpu().numpy(),
                ):
                    pred_records.append(
                        test_mod.BoxRecord(
                            image_id=stem,
                            class_id=int(cid),
                            xyxy=tuple(float(v) for v in xyxy),
                            score=float(sc),
                        )
                    )

            if hook is not None and hook.maps:
                # 每次模型前向恰好触发一次 sgm，弹出该 batch 的注意力图
                batch_maps = hook.maps.pop()
                for stem, att in zip(stems, batch_maps[0]):
                    attn_by_image[stem] = att

    return pred_records, attn_by_image


def sweep_metrics(gt_records: list[test_mod.BoxRecord], pred_records: list[test_mod.BoxRecord]) -> list[dict]:
    """对多个置信度阈值重算比赛指标，返回逐阈值评估结果列表。"""
    config = test_mod.EvalConfig(
        class_to_group=test_mod.CLASS_TO_GROUP,
        group_iou_thresholds=test_mod.GROUP_IOU_THRESHOLDS,
        default_iou_threshold=0.50,
        class_aware=True,
    )
    rows: list[dict] = []
    for th in CONF_SWEEP:
        filtered = [r for r in pred_records if r.score > th]
        ev = test_mod.evaluate_competition_metrics(gt_records, filtered, config)
        groups = ev["groups"]
        rows.append(
            {
                "conf": th,
                "all": ev["all"],
                "ship": groups.get("ship"),
                "aircraft": groups.get("aircraft"),
                "vehicle": groups.get("vehicle"),
            }
        )
    return rows


def _effective_gate(m: torch.Tensor, gate_mode: str, residual_gamma: float = 0.1) -> torch.Tensor:
    """把原始 sigmoid 注意力 M 映射为"有效门控"系数（真实作用于 SPM 特征的乘数）。

    Args:
        m: sigmoid 注意力图（0~1）。
        gate_mode: 门控模式（product/lower_bound/residual/ones）。
        residual_gamma: 残差门控的保留系数（当前 SGA 实现未用 gamma 缩放门控，保留参数仅为接口一致性）。

    Returns:
        与 m 同形状的有效门控系数张量。
    """
    if gate_mode == "product":
        return m  # 原版：目标处可被压到 ≈0
    if gate_mode == "lower_bound":
        return 0.5 + 0.5 * m  # 下界门控：范围 [0.5,1]
    if gate_mode == "residual":
        return 1.0 + m  # 残差门控：范围 [1,2]
    if gate_mode == "ones":
        return torch.ones_like(m)  # SPM-only 消融：恒为 1
    raise ValueError(f"不支持的门控模式: {gate_mode}，可选: product/lower_bound/residual/ones")


def analyze_attention(
    attn_by_image: dict[str, torch.Tensor],
    gt_records: list[test_mod.BoxRecord],
    image_size_map: dict[str, tuple[int, int]],
    vis_dir: Path,
    attn_num: int = 8,
    gate_mode: str = "product",
    residual_gamma: float = 0.1,
) -> list[str]:
    """统计 SGM 注意力图分布 + 目标框内/背景均值 + 保存可视化热力图。

    除原始 sigmoid M 的统计外，还会按门控模式计算"有效门控"（真实作用于 SPM 的系数）
    的框内/背景均值，用于验证 P0 修复（下界门控下框内有效门控应 ≥0.5，不再 ≈0）。

    Args:
        attn_by_image: {image_id: [1,Hi,Wi] sigmoid 注意力图}。
        gt_records: 测试集真实框记录。
        image_size_map: {image_id: (width, height)} 原始尺寸映射。
        vis_dir: 热力图保存目录。
        attn_num: 保存热力图的图像数量上限。
        gate_mode: 门控模式（影响有效门控的映射）。
        residual_gamma: 残差融合系数（暂仅用于报告展示，不参与门控映射）。

    Returns:
        统计文本行列表（供打印/写文件）。
    """
    all_vals: list[torch.Tensor] = []
    fg_sum, fg_cnt = 0.0, 0.0
    bg_sum, bg_cnt = 0.0, 0.0
    eff_fg_sum, eff_fg_cnt = 0.0, 0.0  # 有效门控的框内累计
    eff_bg_sum, eff_bg_cnt = 0.0, 0.0  # 有效门控的背景累计
    gt_by_image: dict[str, list[test_mod.BoxRecord]] = defaultdict(list)
    for g in gt_records:
        gt_by_image[g.image_id].append(g)

    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_done = 0

    for stem, att in attn_by_image.items():
        att_up = tnf.interpolate(
            att[None], size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False
        )[0, 0]  # [H,W]，与模型分辨率对齐，便于按 GT 框取样
        w_orig, h_orig = image_size_map.get(stem, (RESOLUTION, RESOLUTION))

        mask = torch.zeros((RESOLUTION, RESOLUTION))
        for g in gt_by_image.get(stem, []):
            x1, y1, x2, y2 = g.xyxy
            x1s = x1 * RESOLUTION / w_orig
            x2s = x2 * RESOLUTION / w_orig
            y1s = y1 * RESOLUTION / h_orig
            y2s = y2 * RESOLUTION / h_orig
            x1i, x2i = max(0, int(x1s)), min(RESOLUTION, int(x2s) + 1)
            y1i, y2i = max(0, int(y1s)), min(RESOLUTION, int(y2s) + 1)
            if x2i > x1i and y2i > y1i:
                mask[y1i:y2i, x1i:x2i] = 1.0

        fg_sum += (att_up * mask).sum().item()
        fg_cnt += mask.sum().item()
        bg_sum += (att_up * (1 - mask)).sum().item()
        bg_cnt += (1 - mask).sum().item()
        # 有效门控 = 真实作用于 SPM 特征的系数（P0 修复变体下应保证目标处不低于下界）
        att_eff = _effective_gate(att_up, gate_mode, residual_gamma)
        eff_fg_sum += (att_eff * mask).sum().item()
        eff_fg_cnt += mask.sum().item()
        eff_bg_sum += (att_eff * (1 - mask)).sum().item()
        eff_bg_cnt += (1 - mask).sum().item()
        all_vals.append(att.reshape(-1))

        # 保存前若干张图像的热力图（叠加在原图上）
        if vis_done < attn_num:
            img_path = test_mod.TEST_IMAGE_DIR / f"{stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                img_path = test_mod.TEST_IMAGE_DIR / f"{stem}.png"
                img = cv2.imread(str(img_path))
            if img is not None:
                ih, iw = img.shape[:2]
                att_orig = (
                    tnf.interpolate(att[None], size=(ih, iw), mode="bilinear", align_corners=False)[0, 0]
                    .numpy()
                    .clip(0, 1)
                )
                heat = (att_orig * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                blend = cv2.addWeighted(img, 0.6, heat_color, 0.4, 0)
                cv2.imwrite(str(vis_dir / f"{stem}_heat.jpg"), heat_color)
                cv2.imwrite(str(vis_dir / f"{stem}_blend.jpg"), blend)
            vis_done += 1

    vals = torch.cat(all_vals).numpy()
    mean = float(vals.mean())
    std = float(vals.std())
    p05, p50, p95 = np.percentile(vals, [5, 50, 95])
    frac_mid = float(((vals >= 0.4) & (vals <= 0.6)).mean())
    frac_lo = float((vals < 0.3).mean())
    frac_hi = float((vals > 0.7).mean())
    fg_mean = fg_sum / fg_cnt if fg_cnt > 0 else float("nan")
    bg_mean = bg_sum / bg_cnt if bg_cnt > 0 else float("nan")
    # 有效门控的框内/背景均值（关键验证指标：下界门控下框内应 ≥0.5，不再压掉目标处 SPM）
    eff_fg_mean = eff_fg_sum / eff_fg_cnt if eff_fg_cnt > 0 else float("nan")
    eff_bg_mean = eff_bg_sum / eff_bg_cnt if eff_bg_cnt > 0 else float("nan")

    lines = [
        "=" * 78,
        "SGM 注意力图统计（sigmoid，0~1；若“没学起来”应集中在 0.5 附近）",
        "=" * 78,
        f"样本数(图) : {len(attn_by_image)}",
        f"全局 mean  : {mean:.4f}    std: {std:.4f}",
        f"percentile : p05={p05:.4f}  p50={p50:.4f}  p95={p95:.4f}",
        f"取值分布   : [0.4,0.6] 占比 {frac_mid*100:.1f}% | <0.3 占比 {frac_lo*100:.1f}% | >0.7 占比 {frac_hi*100:.1f}%",
        f"目标框内   : {fg_mean:.4f}   (fg_cnt={fg_cnt:.1f} px @640)",
        f"背景       : {bg_mean:.4f}   (bg_cnt={bg_cnt:.1f} px @640)",
        f"框内-背景  : {fg_mean - bg_mean:+.4f}   （>0 说明注意力能区分目标，≈0 说明门控失效）",
        f"有效门控(模式={gate_mode}) 框内均值: {eff_fg_mean:.4f}   背景均值: {eff_bg_mean:.4f}",
        f"  └ 下界门控应框内≥0.5；残差门控应框内≥1.0；若框内有效门控仍≈0 说明修复未生效",
        f"热力图已存 : {vis_dir}",
    ]
    print("\n".join(lines), flush=True)
    return lines


def load_exp_config(ckpt_path: Path) -> tuple[bool, bool, list[str], str, bool, float, float]:
    """从 checkpoint 同目录的 training_config.json 读取模型结构配置。

    Args:
        ckpt_path: checkpoint 文件路径。

    Returns:
        ``(use_sga, use_cfe, projector_scale, sga_gate_mode, sga_fusion_residual,
        sga_residual_gamma, sga_attn_bias)``；配置文件缺失时回退到 SGA 原版
        ``(True, False, ["P4"], "product", False, 0.1, 0.0)``。
    """
    cfg_path = ckpt_path.parent / "training_config.json"
    if cfg_path.exists():
        import json

        model_config = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_config", {})
        return (
            bool(model_config.get("use_sga", True)),
            bool(model_config.get("use_cfe", False)),
            list(model_config.get("projector_scale", ["P4"])),
            model_config.get("sga_gate_mode", "product"),
            bool(model_config.get("sga_fusion_residual", False)),
            float(model_config.get("sga_residual_gamma", 0.1)),
            float(model_config.get("sga_attn_bias", 0.0)),
        )
    return (True, False, ["P4"], "product", False, 0.1, 0.0)


def main() -> None:
    """主流程：默认 SGA vs Baseline 阈值扫描 + 注意力统计；指定实验则只出热力图。"""
    ap = argparse.ArgumentParser(description="SGA 分析：阈值扫描 + SGM 注意力热力图")
    ap.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="实验输出目录（含 checkpoint_best_total.pth 与 training_config.json），指定后仅生成该实验的注意力热力图",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="直接指定 .pth checkpoint 路径（优先于 --exp-dir）",
    )
    ap.add_argument("--attn-num", type=int, default=8, help="保存的热力图数量")
    ap.add_argument("--no-attn", action="store_true", help="默认模式跳过注意力统计")
    ap.add_argument("--sga", action="store_true", help="默认模式仅跑 SGA（跳过 Baseline）")
    ap.add_argument(
        "--gate-mode",
        type=str,
        default=None,
        help="门控模式（product/lower_bound/residual/ones），默认从 training_config.json 读取",
    )
    ap.add_argument(
        "--residual-gamma",
        type=float,
        default=None,
        help="残差融合系数，默认从 training_config.json 读取",
    )
    args = ap.parse_args()

    device = test_mod.resolve_device("cuda")
    test_image_paths = test_mod.read_test_image_paths(test_mod.TEST_IMAGE_DIR)
    print(f"[i] 测试图像: {len(test_image_paths)} 张", flush=True)
    image_size_map = test_mod.build_image_size_map(test_image_paths)
    gt_records = test_mod.load_yolo_labels(test_mod.LABEL_DIR, image_size_map)
    print(f"[i] 真实框: {len(gt_records)} 个", flush=True)

    # ── 指定实验：只生成该 checkpoint 的注意力热力图 ─────────────────────
    if args.checkpoint or args.exp_dir:
        ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.exp_dir) / "checkpoint_best_total.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
        use_sga, use_cfe, proj_scale, gate_mode, fusion_residual, residual_gamma, attn_bias = load_exp_config(ckpt)
        # 命令行显式指定的门控模式优先，便于对旧 checkpoint 套用不同门控做"零训练"推演
        if args.gate_mode is not None:
            gate_mode = args.gate_mode
        if args.residual_gamma is not None:
            residual_gamma = args.residual_gamma
        print(
            f"[i] 加载 {ckpt} (use_sga={use_sga}, use_cfe={use_cfe}, projector_scale={proj_scale}, "
            f"gate_mode={gate_mode}, fusion_residual={fusion_residual}, residual_gamma={residual_gamma}, "
            f"attn_bias={attn_bias})",
            flush=True,
        )
        model = test_mod.RFDETRMedium.from_checkpoint(
            str(ckpt),
            use_sga=use_sga,
            use_cfe=use_cfe,
            projector_scale=proj_scale,
            sga_gate_mode=gate_mode,
            sga_fusion_residual=fusion_residual,
            sga_residual_gamma=residual_gamma,
            sga_attn_bias=attn_bias,
        )
        _, attn = run_inference(model, test_image_paths, device, LOW_CONF, collect_attn=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if not attn:
            print(
                "[!] 该 checkpoint 不含 SemanticGuidingModule（可能未启用 SGA），无法生成注意力热力图",
                flush=True,
            )
            return
        attn_lines = analyze_attention(
            attn,
            gt_records,
            image_size_map,
            ckpt.parent / "attention_vis",
            args.attn_num,
            gate_mode=gate_mode,
            residual_gamma=residual_gamma,
        )
        # 把注意力统计落盘，供实验运行器拼装实验报告
        (ckpt.parent / "attention_stats.txt").write_text("\n".join(attn_lines) + "\n", encoding="utf-8")
        print(f"[i] 注意力统计已保存: {ckpt.parent / 'attention_stats.txt'}", flush=True)
        return

    # ── 默认模式：SGA vs Baseline 阈值扫描 + SGA 注意力统计 ──────────────
    print(f"[i] 加载 SGA checkpoint: {SGA_CKPT}", flush=True)
    sga_model = test_mod.RFDETRMedium.from_checkpoint(
        str(SGA_CKPT), use_sga=True, use_cfe=False, projector_scale=["P4"]
    )
    sga_preds, attn_by_image = run_inference(
        sga_model, test_image_paths, device, LOW_CONF, collect_attn=not args.no_attn
    )
    del sga_model
    gc.collect()
    torch.cuda.empty_cache()

    base_preds: list[test_mod.BoxRecord] = []
    if not args.sga:
        print(f"[i] 加载 Baseline checkpoint: {BASE_CKPT}", flush=True)
        base_model = test_mod.RFDETRMedium.from_checkpoint(
            str(BASE_CKPT), use_sga=False, use_cfe=False, projector_scale=["P4"]
        )
        base_preds, _ = run_inference(base_model, test_image_paths, device, LOW_CONF, collect_attn=False)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    sga_rows = sweep_metrics(gt_records, sga_preds)
    base_rows = sweep_metrics(gt_records, base_preds) if base_preds else None

    # ── 阈值扫描对比 ────────────────────────────────────────────────────
    lines = ["=" * 78, "阈值扫描对比（测试集比赛指标，SGA vs Baseline）", "=" * 78]
    header = (
        f"{'conf':>5} | "
        f"{'all_Recall':>10} {'Δ':>6} | "
        f"{'all_Prec':>9} {'Δ':>6} | "
        f"{'all_F1':>7} {'Δ':>6} | "
        f"{'ship_R':>7} {'Δ':>6} | "
        f"{'veh_R':>6} {'Δ':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, srow in enumerate(sga_rows):
        brow = base_rows[i] if base_rows else None
        d_r = srow["all"].recall - brow["all"].recall if brow else float("nan")
        d_p = srow["all"].precision - brow["all"].precision if brow else float("nan")
        d_f = _f1(srow["all"]) - _f1(brow["all"]) if brow else float("nan")
        d_s = srow["ship"].recall - brow["ship"].recall if brow and brow["ship"] else float("nan")
        d_v = srow["vehicle"].recall - brow["vehicle"].recall if brow and brow["vehicle"] else float("nan")
        lines.append(
            f"{srow['conf']:>5.2f} | "
            f"{srow['all'].recall:>10.4f} {d_r:+.4f} | "
            f"{srow['all'].precision:>9.4f} {d_p:+.4f} | "
            f"{_f1(srow['all']):>7.4f} {d_f:+.4f} | "
            f"{(srow['ship'].recall if srow['ship'] else 0):>7.4f} {d_s:+.4f} | "
            f"{(srow['vehicle'].recall if srow['vehicle'] else 0):>6.4f} {d_v:+.4f}"
        )
    lines.append("")
    lines.append("说明：Δ = SGA − Baseline（正值表示 SGA 更好）")

    # 关键阈值处的 TP/FP/FN 明细
    for th in (0.20, 0.25, 0.30):
        lines.append("")
        lines.append(f"── conf={th} TP/FP/FN 明细 ──")
        for name, rows in (("SGA", sga_rows), ("Baseline", base_rows)):
            if rows is None:
                continue
            row = next(r for r in rows if abs(r["conf"] - th) < 1e-9)
            a = row["all"]
            lines.append(
                f"{name:>8} all: TP={a.tp:<5} FP={a.fp:<5} FN={a.fn:<5} "
                f"R={a.recall:.4f} P={a.precision:.4f}"
            )

    print("\n".join(lines), flush=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[i] 阈值扫描报告已保存: {REPORT_PATH}", flush=True)

    # ── 注意力统计 ──────────────────────────────────────────────────────
    if not args.no_attn and attn_by_image:
        # 默认模式针对旧 SGA checkpoint，门控模式从其训练配置读取（旧实验无 sga_* 时回退 product）
        _use_sga, _use_cfe, _proj, _gate, _fusion, _gamma, _attn_bias = load_exp_config(SGA_CKPT)
        attn_lines = analyze_attention(
            attn_by_image,
            gt_records,
            image_size_map,
            ATTN_VIS_DIR,
            args.attn_num,
            gate_mode=_gate,
            residual_gamma=_gamma,
        )
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n" + "\n".join(attn_lines) + "\n")


if __name__ == "__main__":
    main()
