# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""线性探针（linear probe）实验：骨干特征与解码器 query 特征能否分离舰船小类。

回答两个问题：

1. **骨干特征是否包含区分船类小类（HM/LQS/QHS/MS）的信息**：从 SHWX 训练集
   每张图提取骨干 P4 特征（stride 16），对每个舰船 GT 框做 ROI 平均池化得到
   特征向量，训练线性分类器并测量逐类准确率。若准确率高 → 骨干特征信息充足，
   瓶颈在分类头训练/数据量；若低 → 骨干特征本身分不开小类。

2. **解码器 query 特征（分类头真正消费的特征）是否可分**：hook class_embed
   记录最后一层 decoder 输出，与 GT 按比赛口径匹配后取 matched query 特征，
   对比"线性探针准确率"与"模型自身 class_embed 准确率"。若探针 >> 模型 →
   特征可分但分类头没训好；若两者都低 → 训练过程把特征压扁了。

同时输出：
- 少样本消融：每类 K∈{2,5,10,15,30,60,120} 个样本训练探针的逐类准确率曲线，
  直接测量"小样本下分类头能学到什么程度"；
- 飞机类对照：4 个数据充足的飞机小类跑同一协议，验证方法学本身可靠
  （数据充足时应接近 100%）。

用法：
    python src/scripts/linear_probe.py <checkpoint.pth> <输出名>

输出：
    output/linear_probe/<输出名>/report.json 与 features.npz（特征缓存）。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812
from torch import Tensor, nn
from torchvision.ops import roi_align

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

import test  # noqa: E402

from rfdetr import RFDETR  # noqa: E402
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list  # noqa: E402
from val.competition_metrics import BoxRecord, compute_iou, load_yolo_labels  # noqa: E402

# ============================================================================
# 实验配置
# ============================================================================

# 目标小类（舰船）：0=HM 航母, 1=LQS 两栖舰, 2=QHS 驱护舰, 3=MS 民船
SHIP_CLASS_IDS = [0, 1, 2, 3]
# 飞机对照小类（数据充足，验证方法学）：4=A1_SU-35, 5=A2_C-130, 6=A3_C-17, 7=A4_C-5
CONTROL_CLASS_IDS = [4, 5, 6, 7]
# 少样本消融的每类样本数（'all' 表示用全部样本）
FEWSHOT_KS = [2, 5, 10, 15, 30, 60, 120, "all"]
# 交叉验证折数与随机种子数
CV_FOLDS = 5
CV_SEEDS = 3
FEWSHOT_SEEDS = 5
# 探针训练超参数（线性层 + 交叉熵，Adam）
PROBE_EPOCHS = 300
PROBE_LR = 1e-3
# 推理流水线参数
BATCH_SIZE = 32
NUM_WORKERS = 12
# 匹配 IoU（舰船/飞机均为 0.50；车辆 0.35 不在本实验范围）
MATCH_IOU = 0.50

OUTPUT_ROOT = PROJECT_ROOT / "output" / "linear_probe"


# ============================================================================
# 特征提取
# ============================================================================


def _load_train_images_and_gt() -> tuple[list[Path], list[BoxRecord]]:
    """读取 SHWX 训练集图像路径与真实框（YOLO 格式）。

    Returns:
        ``(image_paths, gt_records)``：图像路径列表与真实框记录列表。
    """
    image_dir = test.DATA_DIR / "images" / "train"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(image_dir.glob("*.png"))
    image_size_map = test.build_image_size_map(image_paths)
    gt_records = load_yolo_labels(test.DATA_DIR / "labels" / "train", image_size_map)
    return image_paths, gt_records


def _preprocess(img_bgr: np.ndarray, resolution: int, means: list[float], stds: list[float]) -> Tensor:
    """单张 BGR 图 → 归一化 RGB 张量 ``(3, res, res)``（与 test.py 推理预处理一致）。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    tensor = TF.resize(tensor, (resolution, resolution), antialias=False)
    return TF.normalize(tensor, means, stds)  # torchvision 逐通道 (x-mean)/std，与 test.py 一致


def _scale_boxes(
    records: list[BoxRecord], resolution: int, size_map: dict[str, tuple[int, int]]
) -> dict[str, list[list[float]]]:
    """把 GT 框缩放到模型分辨率坐标，返回按图聚合的 ``[[x1,y1,x2,y2], ...]``。

    Args:
        records: 真实框记录列表。
        resolution: 模型分辨率。
        size_map: ``{image_id: (width, height)}`` 原始图像尺寸映射。

    Returns:
        ``{image_id: [[x1,y1,x2,y2], ...]}`` 字典，坐标为缩放后像素。
    """
    scaled: dict[str, list[list[float]]] = defaultdict(list)
    for r in records:
        w, h = size_map[r.image_id]
        x0, y0, x1, y1 = r.xyxy
        scaled[r.image_id].append([x0 * resolution / w, y0 * resolution / h, x1 * resolution / w, y1 * resolution / h])
    return scaled


def extract_backbone_features(
    model_lw: nn.Module,
    image_paths: list[Path],
    gt_by_image: dict[str, list[tuple[int, list[float]]]],
    resolution: int,
    means: list[float],
    stds: list[float],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取骨干 P4 特征的舰船/对照类 GT 框 ROI 平均池化向量。

    对每张图：骨干前向得到 P4 特征图（stride 16），对每个 GT 框用
    ``roi_align``（output_size=1）取框内特征均值，得到固定维特征向量。

    Args:
        model_lw: LWDETR 模型（``model.model.model``），需已 ``eval()``。
        image_paths: 图像路径列表（与 ``gt_by_image`` 键一致）。
        gt_by_image: ``{image_id: [(class_id, [x1,y1,x2,y2]), ...]}``
            （坐标为缩放后像素）。
        resolution: 模型分辨率。
        means/std: 归一化均值/标准差。
        device: 推理设备。

    Returns:
        ``(x, y, sample_ids)``：特征矩阵 ``[N, D]``、标签 ``[N]``、
        ``[N, 3]`` 的 (图索引, GT 框索引, 类别) 记录。
    """
    backbone = model_lw.backbone
    model_lw.eval()
    feat_list: list[np.ndarray] = []
    y_list: list[int] = []
    sample_ids: list[list[int]] = []
    feat_w = None
    for start in range(0, len(image_paths), BATCH_SIZE):
        chunk = image_paths[start : start + BATCH_SIZE]
        tensors = [_preprocess(cv2.imread(str(p)), resolution, means, stds) for p in chunk]
        batch_t = torch.stack(tensors).to(device)
        with torch.inference_mode():
            feats, _, _ = backbone(nested_tensor_from_tensor_list(batch_t))
        feat = feats[0].tensors  # (B, C, Hf, Wf)，P4 单尺度
        if feat_w is None:
            feat_w = feat.shape[-1]
        for b, path in enumerate(chunk):
            img_id = path.stem
            for gt_idx, (cls_id, box) in enumerate(gt_by_image.get(img_id, [])):
                rois = torch.tensor([[0, box[0], box[1], box[2], box[3]]], dtype=torch.float32, device=device)
                pooled = roi_align(
                    feat[b : b + 1], rois, output_size=1, spatial_scale=feat_w / resolution, aligned=True
                )
                feat_list.append(pooled.flatten().cpu().numpy())
                y_list.append(cls_id)
                sample_ids.append([start + b, gt_idx, cls_id])
    return np.stack(feat_list), np.array(y_list, dtype=np.int64), np.array(sample_ids, dtype=np.int64)


def extract_query_features(
    model_lw: nn.Module,
    image_paths: list[Path],
    gt_by_image: dict[str, list[tuple[int, list[float]]]],
    resolution: int,
    means: list[float],
    stds: list[float],
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
    """提取与舰船 GT 框匹配的解码器 query 特征（分类头实际消费的特征）。

    用 hook 记录 ``class_embed`` 的输入（最后一层 decoder 输出），对每张图
    按**框 IoU≥0.5、置信度降序、一对一**（不看预测类别）把预测 query 与 GT
    匹配——即"模型找到了目标"的集合，其中既含分类正确的也含分类错误的。
    取 matched query 特征，标签为 **GT 类别**；同时统计模型自身 class_embed
    在每个 GT 类别上的分类正确率（"框对了，类对不对"的对照基线）。

    Args:
        同 ``extract_backbone_features``（仅舰船类）。

    Returns:
        ``(x, y, model_acc)``：matched query 特征矩阵 ``[N, D]``、
        GT 类别标签 ``[N]``、``{GT类别id: {"n": 匹配数, "acc": 模型自身
        分类正确率}}``。
    """
    captured: list[Tensor] = []
    orig_forward = model_lw.class_embed.forward

    def _hook(x: Tensor) -> Tensor:
        captured.append(x.detach())
        return orig_forward(x)

    model_lw.class_embed.forward = _hook  # type: ignore[method-assign,assignment]
    model_lw.eval()
    feat_list: list[np.ndarray] = []
    y_list: list[int] = []
    model_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "acc": 0.0})
    try:
        for start in range(0, len(image_paths), BATCH_SIZE):
            chunk = image_paths[start : start + BATCH_SIZE]
            tensors = [_preprocess(cv2.imread(str(p)), resolution, means, stds) for p in chunk]
            batch_t = torch.stack(tensors).to(device)
            with torch.inference_mode():
                outputs = model_lw(batch_t)
            # class_embed 一次消费全部 decoder 层输出 hs[L, B, Q, D]，
            # 取最后一层（[-1]）作为"分类头实际消费的特征"；多段捕获按 batch 拼接
            hs = torch.cat([c[-1] for c in captured], dim=0)  # (B, Q, D)
            captured.clear()
            logits = outputs["pred_logits"]  # (B, Q, C+1)，末位为背景
            for b, path in enumerate(chunk):
                img_id = path.stem
                gts = gt_by_image.get(img_id, [])
                if not gts:
                    continue
                scores = logits[b, :, :-1].max(dim=-1).values  # 最大前景得分
                pred_cls = logits[b, :, :-1].argmax(dim=-1)
                boxes = outputs["pred_boxes"][b]  # (Q, 4) cxcywh 归一化
                xyxy = torch.stack(
                    [
                        (boxes[:, 0] - boxes[:, 2] / 2) * resolution,
                        (boxes[:, 1] - boxes[:, 3] / 2) * resolution,
                        (boxes[:, 0] + boxes[:, 2] / 2) * resolution,
                        (boxes[:, 1] + boxes[:, 3] / 2) * resolution,
                    ],
                    dim=-1,
                )
                # 框匹配（不看类别）：置信度降序、一对一、IoU≥0.5——
                # 得到"模型找到了目标"的集合，分类正确与否都计入
                order = scores.argsort(descending=True)
                matched_gt = [False] * len(gts)
                for q in order.tolist():
                    best_gt = -1
                    best_iou = MATCH_IOU
                    for gi, (_, gbox) in enumerate(gts):
                        if matched_gt[gi]:
                            continue
                        iou = compute_iou(tuple(xyxy[q].tolist()), tuple(gbox))
                        if iou >= best_iou:
                            best_iou = iou
                            best_gt = gi
                    if best_gt >= 0:
                        matched_gt[best_gt] = True
                        gt_cls = gts[best_gt][0]
                        feat_list.append(hs[b, q].cpu().numpy())
                        y_list.append(gt_cls)
                        model_stats[gt_cls]["n"] += 1
                        model_stats[gt_cls]["acc"] += int(pred_cls[q].item()) == gt_cls
    finally:
        model_lw.class_embed.forward = orig_forward  # type: ignore[method-assign,assignment]
    model_acc: dict[int, dict[str, float]] = {}
    for c, st in model_stats.items():
        model_acc[c] = {"n": st["n"], "acc": st["acc"] / st["n"] if st["n"] else 0.0}
    return np.stack(feat_list), np.array(y_list, dtype=np.int64), model_acc


# ============================================================================
# 线性探针
# ============================================================================


def train_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    num_classes: int,
    seed: int = 0,
) -> dict[int, float]:
    """训练单层线性分类器并返回逐类验证准确率。

    特征按训练集均值/标准差标准化；单层 Linear + 交叉熵，Adam 优化。
    这是线性探针的标准协议（不调架构、不加正则花活）。

    Args:
        x_tr/y_tr: 训练特征与标签。
        x_va/y_va: 验证特征与标签。
        num_classes: 类别数。
        seed: 随机种子（训练样本打乱与初始化）。

    Returns:
        ``{类别id: 验证准确率}``（无验证样本的类不在字典中）。
    """
    torch.manual_seed(seed)
    dim = x_tr.shape[1]
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-8
    x_tr = (x_tr - mean) / std
    x_va = (x_va - mean) / std
    xt = torch.from_numpy(x_tr).float()
    yt = torch.from_numpy(y_tr).long()
    xv = torch.from_numpy(x_va).float()
    yv = torch.from_numpy(y_va).long()
    model = nn.Linear(dim, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(xt.shape[0])
        model.zero_grad()
        loss = F.cross_entropy(model(xt[perm]), yt[perm])
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = model(xv).argmax(dim=-1)
    acc: dict[int, float] = {}
    for c in range(num_classes):
        mask = yv == c
        if mask.sum() > 0:
            acc[c] = float((preds[mask] == c).float().mean().item())
    return acc


def run_cv_probe(x: np.ndarray, y: np.ndarray, class_ids: list[int], name: str) -> dict[str, Any]:
    """重复 K 折交叉验证的逐类准确率。

    Args:
        X/y: 特征与标签。
        class_ids: 参与实验的类别 id 列表（标签空间按其在 class_ids 中的位置压缩）。
        name: 实验名（日志用）。

    Returns:
        ``{"per_class_acc": {类别名: mean}, "per_class_std": {...}, "n": {...}}``。
    """
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    y_comp = np.array([class_to_idx[int(v)] for v in y])
    per_acc: dict[int, list[float]] = defaultdict(list)
    n_counts: dict[int, int] = defaultdict(int)
    for seed in range(CV_SEEDS):
        gen = np.random.default_rng(seed)
        fold_assign = gen.permutation(len(y_comp)) % CV_FOLDS
        for fold in range(CV_FOLDS):
            tr_mask = fold_assign != fold
            va_mask = fold_assign == fold
            acc = train_probe(x[tr_mask], y_comp[tr_mask], x[va_mask], y_comp[va_mask], len(class_ids), seed)
            for c, a in acc.items():
                per_acc[c].append(a)
    for c in range(len(class_ids)):
        n_counts[c] = int((y_comp == c).sum())
    return {
        "per_class_acc": {test.CLASS_NAMES[class_ids[c]]: float(np.mean(v)) for c, v in per_acc.items()},
        "per_class_std": {test.CLASS_NAMES[class_ids[c]]: float(np.std(v)) for c, v in per_acc.items()},
        "n": {test.CLASS_NAMES[class_ids[c]]: n_counts[c] for c in range(len(class_ids))},
    }


def run_cross_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    class_ids: list[int],
    name: str,
) -> dict[str, Any]:
    """跨域探针：在训练集特征上训练、在测试集特征上评估（模拟分类头泛化上限）。

    用训练集特征按比赛小类压缩标签空间后训练线性分类器（可选每类 K 样本
    模拟少样本），再在测试集特征上评估逐类准确率。这是"分类头在测试分布上
    能达到什么水平"的直接测量——若训练特征可分而测试特征不可分，说明特征
    本身不泛化（小样本记忆化）；若测试也高分，说明头在测试上的失败另有原因。

    Args:
        x_tr/y_tr: 训练特征与标签。
        x_te/y_te: 测试特征与标签。
        class_ids: 类别 id 列表（按其在列表中的位置压缩标签空间）。
        name: 实验名（日志用）。

    Returns:
        ``{"full": {类别名: acc}, "fewshot_k10": {...}, "fewshot_k15": {...}}``。
    """
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    y_tr_c = np.array([class_to_idx[int(v)] for v in y_tr])
    y_te_c = np.array([class_to_idx[int(v)] for v in y_te])

    def _eval(x_tr_k: np.ndarray, y_tr_k: np.ndarray) -> dict[str, float]:
        # 标准化统计只用训练集，防止测试分布泄漏进探针
        mean = x_tr_k.mean(axis=0, keepdims=True)
        std = x_tr_k.std(axis=0, keepdims=True) + 1e-8
        xt = torch.from_numpy((x_tr_k - mean) / std).float()
        yt = torch.from_numpy(y_tr_k).long()
        xv = torch.from_numpy((x_te - mean) / std).float()
        torch.manual_seed(0)
        model = nn.Linear(x_tr_k.shape[1], len(class_ids))
        opt = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
        for _ in range(PROBE_EPOCHS):
            perm = torch.randperm(xt.shape[0])
            model.zero_grad()
            loss = F.cross_entropy(model(xt[perm]), yt[perm])
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = model(xv).argmax(dim=-1)
        out: dict[str, float] = {}
        for c in range(len(class_ids)):
            mask = y_te_c == c
            if mask.sum() > 0:
                out[test.CLASS_NAMES[class_ids[c]]] = float((preds[mask] == c).float().mean().item())
        return out

    results: dict[str, dict[str, float]] = {"full": _eval(x_tr, y_tr_c)}

    # 少样本变体：每类 K 样本训练（与 run_fewshot_probe 相同的采样协议）
    idx_by_class = [np.where(y_tr_c == c)[0] for c in range(len(class_ids))]
    for k in (10, 15):
        accs: dict[str, list[float]] = defaultdict(list)
        for seed in range(FEWSHOT_SEEDS):
            gen = np.random.default_rng(seed)
            train_idx: list[int] = []
            for c, idxs in enumerate(idx_by_class):
                n = len(idxs)
                take = min(k, n)
                train_idx.extend(idxs[gen.permutation(n)[:take]])
            if not train_idx:
                continue
            acc = _eval(x_tr[np.array(train_idx)], y_tr_c[np.array(train_idx)])
            for cls_name, a in acc.items():
                accs[cls_name].append(a)
        results[f"fewshot_k{k}"] = {n: float(np.mean(v)) for n, v in accs.items()}
    return results


def run_fewshot_probe(x: np.ndarray, y: np.ndarray, class_ids: list[int], name: str) -> dict[str, Any]:
    """少样本消融：每类 K 个样本训练探针，测量逐类准确率。

    Args:
        X/y: 特征与标签。
        class_ids: 类别 id 列表。
        name: 实验名（日志用）。

    Returns:
        ``{"K": {K: {类别名: 平均准确率}}}``。
    """
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    y_comp = np.array([class_to_idx[int(v)] for v in y])
    idx_by_class = [np.where(y_comp == c)[0] for c in range(len(class_ids))]
    out: dict[str, dict[str, float]] = {}
    for k in FEWSHOT_KS:
        acc_by_class: dict[int, list[float]] = defaultdict(list)
        for seed in range(FEWSHOT_SEEDS):
            gen = np.random.default_rng(seed)
            train_idx: list[int] = []
            for c, idxs in enumerate(idx_by_class):
                n = len(idxs)
                take = n if k == "all" else min(k, n)
                train_idx.extend(idxs[gen.permutation(n)[:take]])
            va_idx = [i for i in range(len(y_comp)) if i not in set(train_idx)]
            if not va_idx:
                continue
            train_idx = np.array(train_idx)
            va_idx = np.array(va_idx)
            acc = train_probe(x[train_idx], y_comp[train_idx], x[va_idx], y_comp[va_idx], len(class_ids), seed)
            for c, a in acc.items():
                acc_by_class[c].append(a)
        out[str(k)] = {test.CLASS_NAMES[class_ids[c]]: float(np.mean(v)) for c, v in acc_by_class.items()}
    return {"K": out}


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    """运行线性探针实验并输出报告。"""
    if len(sys.argv) < 3:
        print("用法: python src/scripts/linear_probe.py <checkpoint.pth> <输出名>")
        sys.exit(1)
    checkpoint = Path(sys.argv[1]).resolve()
    exp_name = sys.argv[2]
    out_dir = OUTPUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = test.resolve_device(test.DEVICE)
    print(f"[i] 加载模型 {checkpoint} ...")
    model = RFDETR.from_checkpoint(str(checkpoint))
    model_lw = model.model.model.to(device)
    resolution = int(model.model.resolution)
    means, stds = model.means, model.stds
    print(f"[i] 模型: {type(model).__name__} | 分辨率 {resolution}")

    image_paths, gt_records = _load_train_images_and_gt()
    size_map = test.build_image_size_map(image_paths)
    scaled = _scale_boxes(gt_records, resolution, size_map)
    # {image_id: [(cls_id, [x1,y1,x2,y2]), ...]}，只保留实验目标类
    target_ids = set(SHIP_CLASS_IDS + CONTROL_CLASS_IDS)
    ship_target_ids = set(SHIP_CLASS_IDS)
    gt_map_all = _build_gt_map(gt_records, scaled, target_ids)
    gt_map_ship = _build_gt_map(gt_records, scaled, ship_target_ids)
    # 只对含目标 GT 的图跑前向（加速：约一半图不含船/对照类）
    img_paths_all = [p for p in image_paths if p.stem in gt_map_all]
    img_paths_ship = [p for p in image_paths if p.stem in gt_map_ship]

    from collections import Counter

    ship_count = Counter(r.class_id for r in gt_records if r.class_id in SHIP_CLASS_IDS)
    ctrl_count = Counter(r.class_id for r in gt_records if r.class_id in CONTROL_CLASS_IDS)
    print(f"[i] 舰船类样本数: {dict(ship_count)}")
    print(f"[i] 飞机对照类样本数: {dict(ctrl_count)}")
    print(f"[i] 含目标 GT 的图像数: {len(img_paths_all)}（共 {len(image_paths)}）")

    x, y, sample_ids = extract_backbone_features(
        model_lw,
        img_paths_all,
        gt_map_all,
        resolution,
        means,
        stds,
        device,
    )
    print(f"[i] 骨干特征已提取: {x.shape}")

    ship_mask = np.isin(y, SHIP_CLASS_IDS)
    ctrl_mask = np.isin(y, CONTROL_CLASS_IDS)

    report: dict[str, Any] = {"model": checkpoint.name, "resolution": resolution}

    # 1) 骨干特征全量探针（舰船 4 类 + 飞机对照 4 类）
    print("\n===== 骨干特征全量探针（5 折 × 3 种子）=====")
    for mask, cls_ids, label in [
        (ship_mask, SHIP_CLASS_IDS, "舰船小类"),
        (ctrl_mask, CONTROL_CLASS_IDS, "飞机对照"),
    ]:
        cv_res = run_cv_probe(x[mask], y[mask], cls_ids, label)
        report[f"backbone_cv_{label}"] = cv_res
        for cls_name in cv_res["per_class_acc"]:
            acc = cv_res["per_class_acc"][cls_name]
            std = cv_res["per_class_std"][cls_name]
            n = cv_res["n"][cls_name]
            print(f"  {cls_name:<6s} acc={acc:.3f} ±{std:.3f} (n={n})")

    # 2) 骨干特征少样本消融（舰船 4 类）
    print("\n===== 骨干特征少样本消融（每类 K 样本，5 种子）=====")
    fs_res = run_fewshot_probe(x[ship_mask], y[ship_mask], SHIP_CLASS_IDS, "骨干-舰船")
    report["backbone_fewshot"] = fs_res
    for k, accs in fs_res["K"].items():
        print(f"  K={k:<5s} " + " ".join(f"{n}={a:.3f}" for n, a in accs.items()))

    # 3) 解码器 query 特征探针 + 模型自身分类准确率
    print("\n===== 解码器 query 特征探针（matched query）=====")
    xq, yq, model_acc = extract_query_features(
        model_lw,
        img_paths_ship,
        gt_map_ship,
        resolution,
        means,
        stds,
        device,
    )
    print(f"[i] matched query 特征: {xq.shape}（框匹配，不看类别）")
    for cid, st in model_acc.items():
        print(f"[i]   模型自身分类准确率 {test.CLASS_NAMES[cid]}: {st['acc']:.3f} (n={int(st['n'])})")
    q_res = run_cv_probe(xq, yq, SHIP_CLASS_IDS, "query-舰船")
    q_fs = run_fewshot_probe(xq, yq, SHIP_CLASS_IDS, "query-舰船")
    report["query_cv"] = q_res
    report["query_fewshot"] = q_fs
    report["model_own_class_acc"] = model_acc
    for cls_name in q_res["per_class_acc"]:
        acc = q_res["per_class_acc"][cls_name]
        std = q_res["per_class_std"][cls_name]
        n = q_res["n"][cls_name]
        print(f"  {cls_name:<6s} 探针acc={acc:.3f} ±{std:.3f} (n={n})")

    # 4) 跨域探针：训练集特征训 → 测试集特征测（分类头泛化上限）
    print("\n===== 跨域探针（训练特征 → 测试特征，舰船 4 类）=====")
    test_image_paths = test.read_test_image_paths(test.TEST_IMAGE_DIR)
    test_size_map = test.build_image_size_map(test_image_paths)
    test_gt = load_yolo_labels(test.LABEL_DIR, test_size_map)
    test_scaled = _scale_boxes(test_gt, resolution, test_size_map)
    gt_map_test = _build_gt_map(test_gt, test_scaled, ship_target_ids)
    img_paths_test = [p for p in test_image_paths if p.stem in gt_map_test]
    x_te, y_te, _ = extract_backbone_features(
        model_lw,
        img_paths_test,
        gt_map_test,
        resolution,
        means,
        stds,
        device,
    )
    cross = run_cross_probe(x[ship_mask], y[ship_mask], x_te, y_te, SHIP_CLASS_IDS, "跨域-舰船")
    report["cross_domain"] = cross
    print(f"[i] 测试集舰船特征: {x_te.shape}")
    for mode, accs in cross.items():
        print(f"  {mode:<10s} " + " ".join(f"{n}={a:.3f}" for n, a in accs.items()))

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 报告已保存: {report_path}")


def _build_gt_map(
    gt_records: list[BoxRecord],
    scaled: dict[str, list[list[float]]],
    target_ids: set[int],
) -> dict[str, list[tuple[int, list[float]]]]:
    """按记录流构建 ``{image_id: [(cls_id, [x1,y1,x2,y2]), ...]}``。

    Args:
        gt_records: 真实框记录列表。
        scaled: ``{image_id: 缩放后框列表}``。
        target_ids: 参与实验的类别 id 集合。

    Returns:
        ``{image_id: [(类别, 框), ...]}``（按记录顺序，索引与 scaled 一致）。
    """
    counter: dict[str, int] = defaultdict(int)
    gt_map: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    for r in gt_records:
        if r.class_id not in target_ids:
            continue
        idx = counter[r.image_id]
        counter[r.image_id] += 1
        gt_map[r.image_id].append((r.class_id, scaled[r.image_id][idx]))
    return dict(gt_map)


if __name__ == "__main__":
    main()
