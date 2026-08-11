# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""探针头实验：把"均衡线性头"作为舰船小类分类器，端到端验证。

线性探针实验证明：在 matched query 特征上，均衡训练的线性头能把 LQS
分类准确率从模型自身的 0.571 拉到 0.80-0.86（跨域口径）。本脚本把这一
结论落地为可部署形态：

1. **训练探针头**：从训练集提取 matched query 特征（与 linear_probe 相同
   协议），按类均衡采样训练 ``Linear(hidden_dim, 4)`` 船类分类头——
   只用训练集，不偷看测试；
2. **测试端到端重分类**：推理时 hook class_embed 记录每张测试图的 query
   特征，对置信度 ≥0.25 且预测为船类的每个框，用探针头重新分类；
   飞机/车辆预测保持不变；
3. **比赛口径评估**：重分类后的预测走 test.py 的完整评估管线，对比
   "原始模型" vs "探针头重分类" 的总 recall/FDR 与逐类指标。

注意：探针头只修"框对类错"（船类小类混淆块），不修"完全漏检"；
评估时探针头用跨域协议（训练集训、测试集测），无测试集泄漏。

用法：
    python src/scripts/probe_head.py <last.ckpt> <输出名>

输出：
    output/probe_head/<输出名>/report.json（含原始 vs 重分类对比）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

from rfdetr import RFDETR  # noqa: E402
from scripts import eval_lib  # noqa: E402
from scripts.analysis import linear_probe  # noqa: E402
from val.competition_metrics import (  # noqa: E402
    BoxRecord,
    EvalConfig,
    evaluate_competition_metrics,
    load_yolo_labels,
)

# SHWX 数据集配置与推理参数（与 test 模板一致）
_DS = eval_lib.build_dataset_cfg("shwx")
_INF = eval_lib.InferenceCfg()

OUTPUT_ROOT = PROJECT_ROOT / "output" / "probe_head"

# 探针头训练超参
HEAD_EPOCHS = 300
HEAD_LR = 1e-3
BALANCE_K = 30  # 严格按类均衡：每类至多 K 个训练样本（少样本类用全部）
CONF_THRESHOLD = 0.25
RECLASS_SCORE_THRESHOLD = 0.5  # 只对 score≥该值的 ship 预测重分类（门控变体）


def _train_probe_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_ids: list[int],
    device: str,
    seed: int = 0,
) -> tuple[nn.Linear, np.ndarray, np.ndarray]:
    """按类均衡采样训练船类线性头（训练集特征，跨域协议）。

    Args:
        x_train: 训练集 matched query 特征 ``[N, D]``。
        y_train: 训练集标签 ``[N]``（类别 id 0-24）。
        class_ids: 船类类别 id 列表（按位置压缩为 0..3）。
        seed: 随机种子。

    Returns:
        ``(model, mean, std)``：训练好的 ``Linear(D, len(class_ids))`` 及
        训练集特征标准化统计（重分类时用同一统计，保证分布一致）。
    """
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    y_comp = np.array([class_to_idx[int(v)] for v in y_train])
    idx_by_class = [np.where(y_comp == c)[0] for c in range(len(class_ids))]
    # 严格按类均衡：每类至多 BALANCE_K 个（不均衡采样会被多数类主导，
    # 复现"MS 500 vs LQS 15 → LQS 测试 0.286"的现象，见诊断记录）
    gen = np.random.default_rng(seed)
    sel = np.concatenate([idxs[gen.permutation(len(idxs))[:BALANCE_K]] for idxs in idx_by_class if len(idxs) > 0])
    x_sel = torch.from_numpy(x_train[sel]).float().to(device)
    y_sel = torch.from_numpy(y_comp[sel]).long().to(device)
    mean = x_sel.mean(dim=0, keepdim=True).cpu().numpy()
    std = x_sel.std(dim=0, keepdim=True).cpu().numpy() + 1e-8
    x_norm = (x_sel - torch.from_numpy(mean).to(device)) / torch.from_numpy(std).to(device)
    model = nn.Linear(x_sel.shape[1], len(class_ids)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HEAD_LR)
    torch.manual_seed(seed)
    for _ in range(HEAD_EPOCHS):
        perm = torch.randperm(x_norm.shape[0])
        model.zero_grad()
        loss = F.cross_entropy(model(x_norm[perm]), y_sel[perm])
        loss.backward()
        opt.step()
    return model.cpu(), mean, std


def _reclassify_ships(
    model_lw: nn.Module,
    image_paths: list[Path],
    resolution: int,
    device: str,
    probe_head: nn.Linear,
    head_mean: np.ndarray,
    head_std: np.ndarray,
    means: list[float],
    stds: list[float],
    image_size_map: dict[str, tuple[int, int]],
    batch_size: int = 32,
) -> list[BoxRecord]:
    """端到端推理 + 探针头重分类船类预测。

    hook class_embed 记录 query 特征；对每个预测框（score>阈值、预测类为
    船类）用探针头重分类（特征按训练时同一统计标准化）；其余类预测原样保留。

    Args:
        model_lw: LWDETR 模型（eval 模式）。
        image_paths: 测试图像路径列表。
        resolution: 模型分辨率。
        device: 推理设备。
        probe_head: 已训练的船类探针头。
        head_mean/head_std: 探针头训练时的特征标准化统计。
        means/std: 图像归一化参数。
        batch_size: 推理批量。

    Returns:
        重分类后的预测框记录列表（与原模型同结构）。
    """
    ship_ids = set(linear_probe.SHIP_CLASS_IDS)
    captured: list[torch.Tensor] = []
    orig_forward = model_lw.class_embed.forward

    def _hook(x: torch.Tensor) -> torch.Tensor:
        captured.append(x.detach())
        return orig_forward(x)

    model_lw.class_embed.forward = _hook  # type: ignore[method-assign,assignment]
    model_lw.eval()
    mean_t = torch.from_numpy(head_mean).float()
    std_t = torch.from_numpy(head_std).float()
    records: list[BoxRecord] = []
    try:
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            tensors = [linear_probe._preprocess(cv2.imread(str(p)), resolution, means, stds) for p in chunk]
            batch_t = torch.stack(tensors).to(device)
            with torch.inference_mode():
                outputs = model_lw(batch_t)
            hs = torch.cat([c[-1] for c in captured], dim=0)  # (B, Q, D)
            captured.clear()
            logits = outputs["pred_logits"]  # (B, Q, C+1)，末位背景
            boxes = outputs["pred_boxes"]  # (B, Q, 4) cxcywh 归一化
            num_cls = logits.shape[-1]
            # 复刻 postprocess 的选择逻辑：sigmoid 概率展平后在 (query×类) 上
            # 取全局 top-k（与 test.py 完全一致的预测空间），并保留 query 索引
            prob = logits.sigmoid()
            flat = prob.view(logits.shape[0], -1)  # (B, Q*(C+1))
            num_select = min(300, flat.shape[1])
            topk_values, topk_indexes = torch.topk(flat, num_select, dim=1)
            scores = topk_values  # (B, K)
            topk_boxes = topk_indexes // num_cls  # (B, K) query 索引
            labels = topk_indexes % num_cls  # (B, K)
            keep = scores > CONF_THRESHOLD
            # 框按每图原图尺寸缩放（与 postprocess 的 target_sizes 一致）
            wh = torch.tensor(
                [image_size_map[path.stem] for path in chunk], dtype=boxes.dtype, device=boxes.device
            )  # (B, 2) (W, H)
            scale = wh[:, None, :].repeat(1, 1, 2)  # (B, 1, 4) (W,H,W,H)
            xyxy_all = (
                torch.stack(
                    [
                        (boxes[:, :, 0] - boxes[:, :, 2] / 2),
                        (boxes[:, :, 1] - boxes[:, :, 3] / 2),
                        (boxes[:, :, 0] + boxes[:, :, 2] / 2),
                        (boxes[:, :, 1] + boxes[:, :, 3] / 2),
                    ],
                    dim=-1,
                )
                * scale
            )  # (B, Q, 4) 原图像素
            # 探针头重分类：ship 类预测批量前向（向量化）
            ship_ids_t = torch.tensor(sorted(ship_ids), device=logits.device)
            ship_q = keep & torch.isin(labels, ship_ids_t) & (scores > RECLASS_SCORE_THRESHOLD)
            new_labels = labels.clone()
            if ship_q.any():
                q_idx = topk_boxes[ship_q]  # [K]
                b_idx = ship_q.nonzero(as_tuple=False)[:, 0]  # [K]
                feats = (hs[b_idx, q_idx].cpu() - mean_t) / std_t  # [K, D]
                with torch.inference_mode():
                    logit = probe_head(feats)  # [K, 4]
                    new_idx = logit.argmax(dim=-1)  # [K]
                for k in range(new_idx.shape[0]):
                    new_labels[ship_q.nonzero(as_tuple=False)[k, 0], ship_q.nonzero(as_tuple=False)[k, 1]] = (
                        linear_probe.SHIP_CLASS_IDS[int(new_idx[k].item())]
                    )
            for b, path in enumerate(chunk):
                for k in range(scores.shape[1]):
                    if not keep[b, k]:
                        continue
                    q = int(topk_boxes[b, k].item())
                    records.append(
                        BoxRecord(
                            image_id=path.stem,
                            class_id=int(new_labels[b, k].item()),
                            xyxy=tuple(float(v) for v in xyxy_all[b, q].tolist()),
                            score=float(scores[b, k].item()),
                        )
                    )
    finally:
        model_lw.class_embed.forward = orig_forward  # type: ignore[method-assign,assignment]
    return records


def _extract_train_matched(
    model_lw: nn.Module,
    resolution: int,
    device: str,
    means: list[float],
    stds: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """提取训练集 matched query 特征（船类）。"""
    image_paths, gt_records = linear_probe._load_train_images_and_gt()
    size_map = eval_lib.build_image_size_map(image_paths)
    scaled = linear_probe._scale_boxes(gt_records, resolution, size_map)
    gt_map = linear_probe._build_gt_map(gt_records, scaled, set(linear_probe.SHIP_CLASS_IDS))
    img_paths = [p for p in image_paths if p.stem in gt_map]
    x, y, _ = linear_probe.extract_query_features(model_lw, img_paths, gt_map, resolution, means, stds, device)
    return x, y


def main() -> None:
    """训练探针头并在测试集上端到端验证重分类。"""
    if len(sys.argv) < 3:
        print("用法: python src/scripts/probe_head.py <last.ckpt> <输出名>")
        sys.exit(1)
    checkpoint = Path(sys.argv[1]).resolve()
    exp_name = sys.argv[2]
    out_dir = OUTPUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = eval_lib.resolve_device(_INF.device)
    means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # 1) 加载检测权重（best_total）
    best_total = checkpoint.parent / "checkpoint_best_total.pth"
    model = RFDETR.from_checkpoint(str(best_total))
    model_lw = model.model.model.to(device)
    resolution = int(model.model.resolution)
    print(f"[i] 模型: {type(model).__name__} | 分辨率 {resolution}")

    # 2) 训练集 matched 特征 → 训练探针头（跨域协议，不偷看测试）
    print("[i] 提取训练集 matched query 特征 ...")
    x_tr, y_tr = _extract_train_matched(model_lw, resolution, device, means, stds)
    print(f"[i] 训练集 matched 特征: {x_tr.shape}")
    ship_mask = np.isin(y_tr, linear_probe.SHIP_CLASS_IDS)
    from collections import Counter

    print(f"[i] 训练集船类样本数: {dict(Counter(y_tr[ship_mask].tolist()))}")
    probe_head, head_mean, head_std = _train_probe_head(
        x_tr[ship_mask], y_tr[ship_mask], linear_probe.SHIP_CLASS_IDS, device
    )
    print("[i] 探针头训练完成（均衡采样 cap=500）", flush=True)

    # 3) 测试集：原始预测（复用 yolo_preds）vs 探针头重分类
    test_image_paths = eval_lib.read_test_image_paths(_DS.test_image_dir)
    test_size_map = eval_lib.build_image_size_map(test_image_paths)
    gt = load_yolo_labels(_DS.label_dir, test_size_map)
    cfg = EvalConfig(
        class_to_group=_DS.per_class_to_group,
        group_iou_thresholds=_DS.per_class_iou_thresholds,
        default_iou_threshold=0.50,
        class_aware=True,
    )

    def _eval(records: list[BoxRecord], tag: str) -> dict[str, Any]:
        per_class = evaluate_competition_metrics(gt, records, cfg)
        group_macro = eval_lib.compute_group_macro_averages(per_class["groups"], _DS.class_to_group, _DS.class_names)
        total = eval_lib.compute_total_metrics(group_macro)
        print(f"  {tag:<12s} 预测框={len(records)} 总 R={total['recall']:.4f} FDR={total['fdr']:.4f}", flush=True)
        for g in ["ship", "vehicle"]:
            print(f"    {g:<8s} R={group_macro[g]['recall']:.4f} FDR={group_macro[g]['fdr']:.4f}", flush=True)
        return {"total": total, "groups": group_macro}

    # 3a) 原始模型：复用已保存的 yolo_preds（conf 0.25，与 test.py 同口径）
    from val.competition_metrics import load_yolo_predictions

    yolo_dir = checkpoint.parent / "yolo_preds"
    if not yolo_dir.exists():
        raise FileNotFoundError(f"yolo_preds 目录不存在（先跑 test.py 生成）: {yolo_dir}")
    orig = load_yolo_predictions(yolo_dir, test_size_map)
    report: dict[str, Any] = {"original": _eval(orig, "原始模型")}

    # 3b) 探针头重分类（走 LWDETR 前向，与训练集同协议）
    reclassified = _reclassify_ships(
        model_lw,
        test_image_paths,
        resolution,
        device,
        probe_head,
        head_mean,
        head_std,
        means,
        stds,
        test_size_map,
    )
    report["probe_head"] = _eval(reclassified, "探针头重分类")

    # 4) 逐类对比
    print("\n===== 逐类对比（Recall / FDR）=====")
    per_orig = evaluate_competition_metrics(gt, orig, cfg)["groups"]
    per_new = evaluate_competition_metrics(gt, reclassified, cfg)["groups"]
    for cid in sorted(_DS.class_names):
        name = _DS.class_names[cid]
        if cid not in (0, 1, 2, 3):
            continue
        a, b = per_orig[name], per_new[name]
        print(f"  {name:<6s} 原始 R={a.recall:.3f} FDR={a.fdr:.3f} → 探针头 R={b.recall:.3f} FDR={b.fdr:.3f}")

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
