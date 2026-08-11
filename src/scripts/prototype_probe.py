# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""原型即分类器离线验证：SSCL EMA 视觉原型能否直接作为舰船小类分类器。

背景：SSCL 原型模式下，训练过程维护每类的 EMA 视觉原型（投影空间 128 维），
并配有两层 MLP 投影头。本脚本用 **PL 原始 checkpoint（last.ckpt，唯一保存
SSCL 模块权重的文件）** 恢复原型库与投影头，对测试集 matched query 特征
（模型检测到且框匹配的舰船）做四路分类对比：

1. **原型分类**：特征 → 投影头 → 与类别原型余弦 → 最近原型 argmax
   （这就是"原型即分类器"的推理形态，零训练）；
2. **线性探针（投影空间）**：同一投影特征上 5 折交叉验证训线性头，
   测投影空间的可分性上限（如果探针 ≫ 原型分类 → 原型质量是短板；
   如果接近 → 原型本身够用）；
3. **线性探针（hidden 空间）**：不经投影头的原始 query 特征，作为对照
   （线性探针实验已证明 hidden 空间可分）；
4. **模型自身 class_embed**：当前检测头的分类准确率（matched 集）。

同时输出原型结构诊断：船类原型间余弦、matched 特征与自身原型对齐余弦、
原型分类的混淆去向——回答"该推开的类对是否真的被推开了"。

用法：
    python src/scripts/prototype_probe.py <last.ckpt> <输出名>

注意：必须传 ``last.ckpt``（PL 原始 checkpoint）；``checkpoint_best_*.pth``
只含检测权重，不含原型/投影头。

输出：
    output/prototype_probe/<输出名>/report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

import linear_probe  # noqa: E402
import test  # noqa: E402

from rfdetr import RFDETR  # noqa: E402
from rfdetr.sscl.sscl_loss import SSCLLoss  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "output" / "prototype_probe"


def load_lwdetr_and_sscl(checkpoint: Path, device: str) -> tuple[nn.Module, SSCLLoss, int]:
    """恢复检测权重与 SSCL 模块（原型库 + 投影头）。

    检测权重通过 ``RFDETR.from_checkpoint`` 加载同级 ``checkpoint_best_total.pth``
    （已验证的加载路径，自动推断模型尺寸与分辨率）；SSCL 模块（原型库/投影头/
    语义矩阵）只存在于 PL 原始 checkpoint（last.ckpt）的 ``sscl_loss.*`` 键中，
    从该文件单独恢复。

    Args:
        checkpoint: PL 原始 checkpoint 路径（last.ckpt）。
        device: 目标设备。

    Returns:
        ``(model_lw, sscl_loss, resolution)``：LWDETR 模型（已加载权重并
        eval）、SSCLLoss 模块（含原型库与投影头）、模型分辨率。
    """
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = sd["state_dict"]
    sscl_keys = {k[len("sscl_loss.") :]: v for k, v in state.items() if k.startswith("sscl_loss.")}

    # 检测权重：同级 best_total（strip 产物）经 from_checkpoint 加载
    best_total = checkpoint.parent / "checkpoint_best_total.pth"
    if not best_total.exists():
        raise FileNotFoundError(f"同级 checkpoint_best_total.pth 不存在: {best_total}")
    model = RFDETR.from_checkpoint(str(best_total))
    model_lw = model.model.model
    resolution = int(model.model.resolution)
    model_lw.to(device).eval()

    # 构造 SSCLLoss 并加载原型/投影头/语义矩阵
    semantic_matrix = sscl_keys["semantic_matrix"]
    sscl_loss = SSCLLoss(
        semantic_matrix=semantic_matrix,
        tau=0.1,
        rho=0.3,
        omega_max=2.0,
        anchor_classes=[0, 1, 2, 3],
        confusing_classes=[0, 1, 2, 3],
        prototype_mode=True,
        prototype_momentum=0.99,
        prototype_min_samples=1,
        hidden_dim=256,
        projection_dim=128,
        prototype_instance_pos=True,
    )
    sscl_loss.load_state_dict(sscl_keys)
    sscl_loss.to(device).eval()
    return model_lw, sscl_loss, resolution


def evaluate_prototype_classifier(
    proj_features: np.ndarray,
    labels: np.ndarray,
    prototypes: torch.Tensor,
    valid: torch.Tensor,
    class_ids: list[int],
) -> dict[str, Any]:
    """原型分类（最近原型 argmax）的逐类准确率与混淆统计。

    Args:
        proj_features: 投影空间特征 ``[N, D]``（CPU numpy）。
        labels: 舰船小类标签 ``[N]``。
        prototypes: 全部类别原型 ``[C, D]``（CPU tensor，未归一化）。
        valid: 有效原型掩码 ``[C]``。
        class_ids: 参与 argmax 的类别 id 列表（按列表位置压缩标签空间）。

    Returns:
        ``{"per_class_acc": {类名: acc}, "confusion": [[int]]}``。
    """
    proto = F.normalize(prototypes[class_ids], dim=-1)  # [4, D]
    u = F.normalize(torch.from_numpy(proj_features).float(), dim=-1)
    sim = u @ proto.T  # [N, 4]
    pred = sim.argmax(dim=-1).numpy()
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    y_comp = np.array([class_to_idx[int(v)] for v in labels])
    acc: dict[str, float] = {}
    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=int)
    for i in range(len(class_ids)):
        mask = y_comp == i
        if mask.sum() > 0:
            acc[test.CLASS_NAMES[class_ids[i]]] = float((pred[mask] == i).mean())
        for j in range(len(class_ids)):
            confusion[i, j] = int(((y_comp == i) & (pred == j)).sum())
    return {"per_class_acc": acc, "confusion": confusion.tolist(), "valid_prototypes": valid[class_ids].tolist()}


def main() -> None:
    """运行原型即分类器离线验证。"""
    if len(sys.argv) < 3:
        print("用法: python src/scripts/prototype_probe.py <last.ckpt> <输出名>")
        sys.exit(1)
    checkpoint = Path(sys.argv[1]).resolve()
    exp_name = sys.argv[2]
    out_dir = OUTPUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = test.resolve_device(test.DEVICE)
    print(f"[i] 加载 {checkpoint} ...")
    model_lw, sscl_loss, resolution = load_lwdetr_and_sscl(checkpoint, device)
    print("[i] LWDETR 已恢复 | SSCL 模块已恢复（原型/投影头/语义矩阵）")

    # 1) 提取测试集 matched query 特征（hidden 空间，框匹配 IoU≥0.5 不看类）
    test_image_paths = test.read_test_image_paths(test.TEST_IMAGE_DIR)
    test_size_map = test.build_image_size_map(test_image_paths)
    from val.competition_metrics import load_yolo_labels

    test_gt = load_yolo_labels(test.LABEL_DIR, test_size_map)
    test_scaled = linear_probe._scale_boxes(test_gt, resolution, test_size_map)
    gt_map = linear_probe._build_gt_map(test_gt, test_scaled, set(linear_probe.SHIP_CLASS_IDS))
    img_paths = [p for p in test_image_paths if p.stem in gt_map]
    x_hidden, y, model_acc = linear_probe.extract_query_features(
        model_lw,
        img_paths,
        gt_map,
        resolution,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        device,
    )
    print(f"[i] matched query 特征: {x_hidden.shape}（hidden 空间）")

    # 2) 投影到对比空间
    with torch.inference_mode():
        x_proj = sscl_loss.projection_head(torch.from_numpy(x_hidden).float().to(device)).cpu().numpy()
    print(f"[i] 投影空间特征: {x_proj.shape}")

    # 3) 原型结构诊断
    proto_norm, valid = sscl_loss.prototype_bank.get_normalized_prototypes()
    proto_norm = proto_norm.cpu()
    valid = valid.cpu()
    ship_ids = linear_probe.SHIP_CLASS_IDS
    ship_sim = [[round(v, 3) for v in row] for row in (proto_norm[ship_ids] @ proto_norm[ship_ids].T).tolist()]
    names4 = [test.CLASS_NAMES[c] for c in ship_ids]
    print("\n===== 船类原型间余弦（投影空间）=====")
    print("     " + " ".join(f"{n:>6s}" for n in names4))
    for i, n in enumerate(names4):
        print(f"  {n:>4s} " + " ".join(f"{ship_sim[i][j]:6.3f}" for j in range(4)))

    # 4) 四路分类对比（matched 集上逐类准确率）
    report: dict[str, Any] = {"model": checkpoint.name, "resolution": resolution}

    # a) 原型分类（船类内 argmax）
    proto_res = evaluate_prototype_classifier(x_proj, y, proto_norm, valid, ship_ids)
    report["prototype_classifier"] = proto_res
    print("\n===== 原型分类（船类 4 类内最近原型）=====")
    for n, a in proto_res["per_class_acc"].items():
        print(f"  {n:<6s} acc={a:.3f}")

    # b) 线性探针（投影空间）
    proj_cv = linear_probe.run_cv_probe(x_proj, y, ship_ids, "投影空间")
    report["linear_probe_proj"] = proj_cv
    print("\n===== 线性探针（投影空间，5 折 × 3 种子）=====")
    for n in proj_cv["per_class_acc"]:
        print(
            f"  {n:<6s} acc={proj_cv['per_class_acc'][n]:.3f} ±{proj_cv['per_class_std'][n]:.3f} (n={proj_cv['n'][n]})"
        )

    # c) 线性探针（hidden 空间，对照）
    hidden_cv = linear_probe.run_cv_probe(x_hidden, y, ship_ids, "hidden 空间")
    report["linear_probe_hidden"] = hidden_cv
    print("\n===== 线性探针（hidden 空间，5 折 × 3 种子）=====")
    for n in hidden_cv["per_class_acc"]:
        print(
            f"  {n:<6s} acc={hidden_cv['per_class_acc'][n]:.3f} ±{hidden_cv['per_class_std'][n]:.3f} "
            f"(n={hidden_cv['n'][n]})"
        )

    # d) 模型自身 class_embed（hidden 空间 argmax）
    report["model_own_class_acc"] = model_acc
    print("\n===== 模型自身 class_embed（matched 集）=====")
    for cid, st in model_acc.items():
        print(f"  {test.CLASS_NAMES[cid]:<6s} acc={st['acc']:.3f} (n={int(st['n'])})")

    # 5) 对齐诊断：matched 特征与自身类原型的余弦
    with torch.inference_mode():
        u = F.normalize(torch.from_numpy(x_proj).float(), dim=-1)
        self_cos = (u * proto_norm[y].unsqueeze(0).squeeze(0)).sum(dim=-1).numpy()
    align: dict[str, float] = {}
    for cid in ship_ids:
        mask = y == cid
        align[test.CLASS_NAMES[cid]] = float(self_cos[mask].mean()) if mask.sum() else 0.0
    report["self_proto_align_cos"] = align
    print("\n===== matched 特征与自身原型对齐余弦（投影空间）=====")
    for n, v in align.items():
        print(f"  {n:<6s} cos={v:.3f}")

    # 6) 原型分类混淆去向
    confusion = proto_res["confusion"]
    print("\n===== 原型分类混淆矩阵（行=真类，列=预测）=====")
    print("     " + " ".join(f"{n:>6s}" for n in names4))
    for i, n in enumerate(names4):
        print(f"  {n:>4s} " + " ".join(f"{confusion[i][j]:6d}" for j in range(4)))

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
