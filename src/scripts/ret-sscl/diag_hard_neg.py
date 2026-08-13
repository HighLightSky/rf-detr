# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""难例硬度前置验证脚本（训练前运行，验证"难例是否真的硬"）。

假设：高分 unmatched query（与任一 GT 的 IoU∈[0.0,0.3]）的特征比随机未匹配
更贴近类别原型——即它们确实代表"像目标但不是目标的区域"，作为 SSCL 分母的
额外负样本列才有意义。本脚本在 0807 checkpoint 上验证该假设，**无需训练**：

1. 加载检测权重（checkpoint_best_total.pth，仅含 model.*）与投影头 + 原型库
   （last.ckpt 中的 sscl_loss.* 键，best checkpoint 不含这两者）。
2. 在 SHWX 测试集逐图推理（eval 模式单组 query），直接取 decoder 输出
   （hs/pred_logits/pred_boxes），执行与训练完全一致的难例选择
   （select_hard_negatives_for_image，匹配近似用 IoU>=0.5 的比赛惯例）。
3. 输出三组余弦对照：难例 vs 随机未匹配 vs matched（各与类别原型的平均余弦），
   加上 IoU 带填充率，并给出假设成立的判据结论。

判据：
- ``hn_vs_random_gap > 0``：难例比随机未匹配更贴类别原型（"硬"的来源）；
- ``hn_vs_matched_gap < 0``：难例比真实目标（matched）离类中心更远
  （若难例比 matched 还贴类中心，说明误报区与目标几乎不可分，负样本无益）；
- 填充率（IoU 带内未匹配候选占比）过低（< 5%）说明该采样规则在数据上
  产出不足，需要放宽。

用法：
    python src/scripts/ret-sscl/diag_hard_neg.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── 路径设置（与 eval_ablation.py 一致：src 与项目根加入 sys.path）──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
for _p in (str(SRC_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402, N812 -- 与 test.py 一致（resize/normalize）
from torch.utils.data import DataLoader  # noqa: E402

from rfdetr import RFDETRMedium  # noqa: E402
from rfdetr.sscl import SSCLLoss, load_semantic_matrix, normalize_semantic_matrix  # noqa: E402
from rfdetr.sscl.hard_neg_selection import select_hard_negatives_for_image  # noqa: E402
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402
from scripts import eval_lib as _lib  # noqa: E402  (复用 SHWX 配置与推理预处理)
from val.competition_metrics import load_yolo_labels  # noqa: E402

# ── 0807 基线实验输出目录（双源 checkpoint 加载）──
RUN_DIR = PROJECT_ROOT / "output/0807-SHWX-SSCL-Proj-原型+实例正样本"
DET_CHECKPOINT = RUN_DIR / "checkpoint_best_total.pth"
LIGHTNING_CKPT = RUN_DIR / "last.ckpt"

# 测试集（与 test 模板一致，来自 eval_lib 的 SHWX 配置）
_DS = _lib.build_dataset_cfg("shwx")
_INF = _lib.InferenceCfg()
TEST_IMAGE_DIR = _DS.test_image_dir
LABEL_DIR = _DS.label_dir
DEVICE = _INF.device
BATCH_SIZE = 8  # 诊断脚本从简：主进程解码，batch 小一点避免 CPU 瓶颈
RESOLUTION = 640

# 匹配近似阈值：与比赛惯例一致（舰船/飞机 IoU=0.5），用于推理时估计"matched"
MATCH_IOU_THRESHOLD = 0.5

# 难例前景分下限（原始 logit）：可经环境变量覆盖用于参数对比。
# 默认 -2.0（≈ p>0.12）：0.0 在原始 logit 空间下过严，会把强模型的带内
# 候选几乎全部滤掉（弱模型误报才 >0），机制饿死；-2.0 保留"弱前景"
# 的次优框（LMP 论文的难例是纯几何采样，无分数门槛）。
SCORE_THRESH = float(os.environ.get("HN_SCORE_THRESH", "-2.0"))


def _resolve_matrix_path(matrix_cfg: str) -> Path:
    """解析训练配置里的语义矩阵路径，兼容镜像机/旧机器的绝对路径。

    训练配置可能存的是镜像机路径（如 /home/liu/wzt/Ruiyingshizong/rf-detr/
    data/semantic_matrix_shwx.pt），本机不存在。解析优先级：
    1. 相对路径 → 挂 PROJECT_ROOT 下（本项目训练配方的写法）；
    2. 绝对路径且存在 → 直接用；
    3. 绝对路径不存在 → 取路径中 ``rf-detr`` 目录段之后的相对部分挂
       PROJECT_ROOT 下（镜像与本体目录结构一致）；仍不存在则报错。

    Args:
        matrix_cfg: 训练配置里的 sscl_semantic_matrix_path 原值。

    Returns:
        解析后的本机语义矩阵路径（确保存在）。

    Raises:
        FileNotFoundError: 所有解析候选均不存在时抛出。
    """
    cfg = Path(matrix_cfg)
    candidates: list[Path] = []
    if cfg.is_absolute():
        candidates.append(cfg)
        # 镜像映射：/…/rf-detr/<rel> → PROJECT_ROOT/<rel>
        parts = cfg.parts
        for i, part in enumerate(parts):
            if part == "rf-detr" and i + 1 < len(parts):
                candidates.append(PROJECT_ROOT.joinpath(*parts[i + 1 :]))
                break
        if not candidates:
            candidates.append(PROJECT_ROOT / cfg.name)
    else:
        candidates.append(PROJECT_ROOT / cfg)
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"语义矩阵路径均不存在: {[str(c) for c in candidates]}（配置存的是 {matrix_cfg}）")


def _build_sscl_loss_from_run() -> SSCLLoss:
    """按 0807 实验的 training_config.json 重建 SSCLLoss 并加载投影头/原型库。

    从 last.ckpt 提取 sscl_loss.* 键（best checkpoint 只含 model 权重），键缺失即抛错。
    """
    cfg_path = RUN_DIR / "training_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        train_cfg = json.load(f)["train_config"]
    semantic_matrix = load_semantic_matrix(str(_resolve_matrix_path(train_cfg["sscl_semantic_matrix_path"])))
    if train_cfg.get("sscl_matrix_normalize", "minmax") != "none":
        semantic_matrix = normalize_semantic_matrix(semantic_matrix, mode=train_cfg["sscl_matrix_normalize"])
    loss_fn = SSCLLoss(
        semantic_matrix=semantic_matrix,
        tau=train_cfg["sscl_tau"],
        rho=train_cfg["sscl_rho"],
        omega_max=train_cfg["sscl_omega_max"],
        prototype_mode=True,
        prototype_momentum=train_cfg["sscl_prototype_momentum"],
        prototype_min_samples=train_cfg["sscl_prototype_min_samples"],
        hidden_dim=256,  # medium 的 decoder hidden dim
        projection_dim=train_cfg["sscl_projection_dim"],
        prototype_instance_pos=train_cfg.get("sscl_prototype_instance_pos", False),
    )
    ckpt = torch.load(str(LIGHTNING_CKPT), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    sscl_keys = {k.removeprefix("sscl_loss."): v for k, v in state_dict.items() if k.startswith("sscl_loss.")}
    if not any(k.startswith("projection_head.") for k in sscl_keys) or not any(
        k.startswith("prototype_bank.") for k in sscl_keys
    ):
        raise FileNotFoundError(
            f"{LIGHTNING_CKPT} 中缺少 sscl_loss.projection_head.* / prototype_bank.* 键，"
            "无法重建投影空间（best checkpoint 不包含 SSCL 附加模块，必须双源加载）。"
        )
    loss_fn.load_state_dict(sscl_keys, strict=False)
    return loss_fn


def main() -> None:
    """运行难例硬度前置验证并输出判据结论。"""
    print(f"难例硬度前置验证 | 检测权重: {DET_CHECKPOINT}")
    print(f"SSCL 投影头/原型库: {LIGHTNING_CKPT}")
    print(f"测试集: {TEST_IMAGE_DIR}（{len(_lib.read_test_image_paths(TEST_IMAGE_DIR))} 张）")

    # 1. 加载检测模型（eval 模式，单组 query；与 test.py 相同的预处理）
    model = RFDETRMedium.from_checkpoint(str(DET_CHECKPOINT))
    raw_model = model.model.model.to(DEVICE)
    raw_model.eval()
    means, stds = model.means, model.stds

    # 2. 重建 SSCL 投影空间（投影头 + 原型库 + 语义矩阵）
    loss_fn = _build_sscl_loss_from_run()
    loss_fn = loss_fn.to(DEVICE)

    # 3. GT 记录 → 按图像分组（像素 xyxy → 归一化 cxcywh）
    image_paths = _lib.read_test_image_paths(TEST_IMAGE_DIR)
    image_size_map = _lib.build_image_size_map(image_paths)
    gt_records = load_yolo_labels(LABEL_DIR, image_size_map)
    gt_by_image: dict[str, list[torch.Tensor]] = {}
    for rec in gt_records:
        w, h = image_size_map[rec.image_id]
        x0, y0, x1, y1 = rec.xyxy
        gt_by_image.setdefault(rec.image_id, []).append(
            torch.tensor([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        )

    # 4. 逐 batch 推理 + 难例选择 + 特征收集
    loader = DataLoader(
        _lib._InferenceDataset(image_paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=_lib._inference_collate,
    )
    hn_parts: list[torch.Tensor] = []
    random_parts: list[torch.Tensor] = []
    matched_parts: list[torch.Tensor] = []
    total_band = total_unmatched = total_selected = 0
    n_images = 0

    with torch.inference_mode():
        for stems, rgb_tensors, orig_sizes in loader:
            gpu_images = [
                F.resize(
                    tensor.to(DEVICE, non_blocking=True).to(torch.float32).div_(255.0),
                    (RESOLUTION, RESOLUTION),
                    antialias=False,
                )
                for tensor in rgb_tensors
            ]
            batch_tensor = F.normalize(torch.stack(gpu_images), means, stds)
            preds = raw_model(batch_tensor)
            pred_logits = preds["pred_logits"]  # [B, Q, C+1]
            pred_boxes = preds["pred_boxes"]  # [B, Q, 4] cxcywh 归一化
            hs = preds["hs"]  # [B, Q, D]

            for b, stem in enumerate(stems):
                if gt_by_image.get(stem):
                    gt_boxes = torch.stack(gt_by_image[stem]).to(DEVICE)
                else:
                    gt_boxes = torch.zeros(0, 4, device=DEVICE)
                # 推理无 Hungarian indices：用比赛惯例 IoU>=0.5 的 query 作为 matched 近似
                matched_src = torch.empty(0, dtype=torch.long, device=DEVICE)
                if gt_boxes.shape[0] > 0:
                    iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes[b]), box_cxcywh_to_xyxy(gt_boxes))
                    matched_src = (iou >= MATCH_IOU_THRESHOLD).any(dim=-1).nonzero(as_tuple=False).flatten()

                hn_idx, stats = select_hard_negatives_for_image(
                    pred_logits=pred_logits[b],
                    pred_boxes=pred_boxes[b],
                    gt_boxes=gt_boxes,
                    matched_src=matched_src,
                    top_k=3,
                    score_thresh=SCORE_THRESH,
                )
                if hn_idx.shape[0] > 0:
                    hn_parts.append(hs[b][hn_idx].detach().cpu())
                unmatched = torch.ones(hs.shape[1], dtype=torch.bool, device=DEVICE)
                if matched_src.shape[0] > 0:
                    unmatched[matched_src] = False
                if unmatched.any():
                    rand_idx = unmatched.nonzero(as_tuple=False).flatten()[: min(3, int(unmatched.sum().item()))]
                    random_parts.append(hs[b][rand_idx].detach().cpu())
                if matched_src.shape[0] > 0:
                    matched_parts.append(hs[b][matched_src].cpu())
                total_band += int(stats["n_band"])
                total_unmatched += int(stats["n_unmatched"])
                total_selected += int(stats["n_selected"])
                n_images += 1
            print(f"已处理 {n_images}/{len(image_paths)} 张", end="\r")
    print()

    del raw_model
    _lib.release_cuda_cache(DEVICE)

    # 5. 硬度统计与判据结论
    if not hn_parts:
        print("【结论】未选出任何难例：IoU 带 [0.0,0.3] 内无候选，需放宽采样规则后重试。")
        return
    hn_all = torch.cat(hn_parts, dim=0)
    random_all = torch.cat(random_parts, dim=0)
    matched_all = torch.cat(matched_parts, dim=0) if matched_parts else torch.zeros(0, 256)
    stats = loss_fn.hardness_stats(matched_all.to(DEVICE), hn_all.to(DEVICE), random_all.to(DEVICE))

    fill_rate = total_band / max(1, total_unmatched)
    print("\n================ 难例硬度验证结果 ================")
    print(f"图像数: {n_images} | 每图平均难例数: {total_selected / max(1, n_images):.2f}")
    print(f"IoU 带填充率（带内未匹配候选 / 未匹配总数）: {fill_rate * 100:.1f}%")
    print(f"难例特征数: {hn_all.shape[0]} | 随机未匹配: {random_all.shape[0]} | matched: {matched_all.shape[0]}")
    print("\n投影空间与类别原型的平均余弦（越大越贴类中心）：")
    print(f"  难例      hn_proto_cos    = {stats.get('hn_proto_cos', float('nan')):.4f}")
    print(f"  随机未匹配 random_proto_cos = {stats.get('random_proto_cos', float('nan')):.4f}")
    print(f"  matched   matched_proto_cos = {stats.get('matched_proto_cos', float('nan')):.4f}")
    print(f"  难例-随机差距 hn_vs_random_gap  = {stats.get('hn_vs_random_gap', float('nan')):+.4f}")
    print(f"  难例-matched差距 hn_vs_matched_gap = {stats.get('hn_vs_matched_gap', float('nan')):+.4f}")

    print("\n================ 判据结论 ================")
    hn_vs_random = stats.get("hn_vs_random_gap", 0.0)
    hn_vs_matched = stats.get("hn_vs_matched_gap", 0.0)
    checks = [
        (hn_vs_random > 0.0, f"难例比随机未匹配更贴类中心（gap = {hn_vs_random:+.4f} > 0）"),
        (hn_vs_matched < 0.0, f"难例比真实目标离类中心更远（gap = {hn_vs_matched:+.4f} < 0）"),
        (fill_rate >= 0.05, f"IoU 带填充率足够（{fill_rate * 100:.1f}% >= 5%）"),
    ]
    for passed, desc in checks:
        print(f"  {'✅' if passed else '❌'} {desc}")
    if all(p for p, _ in checks):
        print(
            "\n【结论】难例假设成立：高分 unmatched query 确实代表'像目标但不是目标'"
            "的区域，适合作为 SSCL 分母的额外负样本列，可以启动训练实验。"
        )
    else:
        print(
            "\n【结论】难例假设未完全成立：请结合上方数值调整采样规则"
            "（IoU 带边界 / score_thresh / top_k）后重跑本脚本，再决定是否训练。"
        )


if __name__ == "__main__":
    main()
