# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""离线构建发射车（FSC，类24）正/负样本补丁池。

供 ``PatchPasteDataset``（补丁粘贴增强）消费。两类补丁：

1. **正样本**：训练集里所有 FSC GT 框区域（带类24标注），粘贴时扩增稀有类；
2. **负样本**：指定 checkpoint 在**训练集**上推理得到的高分未匹配 FSC 预测
   （多为"长得像发射车的卡车"），无标注，粘贴时教模型"卡车 ≠ 发射车"。

设计约束（与用户确认）：
- 正样本只用训练集 GT，负样本只用训练集推理挖掘——**测试集完全不进入训练**；
- 补丁以框中心外扩上下文裁剪，坐标记录为**补丁局部像素**（粘贴期再做
  D4 旋转/缩放/平移）；
- 负样本裁剪区与任何 GT 相交则丢弃（防止把真目标裁进负补丁）。

用法：
    python src/scripts/data_prep/build_fsc_patch_pool.py \
      --data-dir /home/liu/wzt/datasets/SHWX-dataset-dict-redo \
      --ckpt output/0816-SHWX-ProtoGuidance-1024/checkpoint_best_total.pth \
      --out-dir data/fsc_patch_pool

输出：
    data/fsc_patch_pool/manifest.json —— 补丁清单（positive/negative 两类）
    data/fsc_patch_pool/positive/P*.jpg
    data/fsc_patch_pool/negative/N*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

from rfdetr import RFDETR  # noqa: E402
from scripts import eval_lib  # noqa: E402
from val.competition_metrics import compute_iou  # noqa: E402

FSC_CLASS = 24
MANIFEST_VERSION = 1
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


# ------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------
def _find_image(images_dir: Path, stem: str) -> Path | None:
    """按文件名主干找图片（训练集全部为 .jpg，兜底扫描常见后缀）。"""
    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _normalized_to_pixel(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """归一化 cxcywh → 像素 xyxy（clamp 到图像范围）。"""
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def _context_crop(box: tuple[int, int, int, int], img_w: int, img_h: int, context: float) -> tuple[int, int, int, int]:
    """以框中心外扩 context 倍（宽高分别乘），clamp 到图像内。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * context, (y2 - y1) * context
    crop = (
        max(0, int(cx - bw / 2)),
        max(0, int(cy - bh / 2)),
        min(img_w, int(cx + bw / 2)),
        min(img_h, int(cy + bh / 2)),
    )
    return crop


def _intersects(box_a: tuple, box_b: tuple) -> bool:
    """两个像素框是否有正面积交叠。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    return ix > 0 and iy > 0


def _nms(boxes: list[tuple], scores: list[float], iou_thresh: float = 0.5) -> list[bool]:
    """按置信度降序的贪心 NMS，返回保留掩码。"""
    keep = [True] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    for i in range(len(order)):
        if not keep[order[i]]:
            continue
        for j in range(i + 1, len(order)):
            if not keep[order[j]]:
                continue
            if compute_iou(boxes[order[i]], boxes[order[j]]) > iou_thresh:
                keep[order[j]] = False
    return keep


# ------------------------------------------------------------------------
# 正样本：训练集 FSC GT 区域
# ------------------------------------------------------------------------
def _collect_positives(
    data_dir: Path, out_dir: Path, context: float, min_box_px: int, seed: int
) -> list[dict]:
    """从训练集收集 FSC GT 框区域，裁图保存并返回 manifest 条目。"""
    labels_dir = data_dir / "labels" / "train"
    images_dir = data_dir / "images" / "train"
    pos_out = out_dir / "positive"
    pos_out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    skipped = 0
    for txt_path in sorted(labels_dir.glob("*.txt")):
        img_path = _find_image(images_dir, txt_path.stem)
        if img_path is None:
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != FSC_CLASS:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:])
            box = _normalized_to_pixel(cx, cy, w, h, img_w, img_h)
            if box[2] - box[0] < min_box_px or box[3] - box[1] < min_box_px:
                skipped += 1
                continue
            crop = _context_crop(box, img_w, img_h, context)
            if crop[2] - crop[0] < 2 or crop[3] - crop[1] < 2:
                skipped += 1
                continue
            patch_id = f"P{len(manifest):04d}"
            with Image.open(img_path) as img:
                patch_img = img.crop(crop)
            patch_img.convert("RGB").save(pos_out / f"{patch_id}.jpg", quality=95)
            # 框坐标 → 补丁局部像素
            local_box = (box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1])
            manifest.append(
                {
                    "kind": "positive",
                    "id": patch_id,
                    "file": f"positive/{patch_id}.jpg",
                    "width": crop[2] - crop[0],
                    "height": crop[3] - crop[1],
                    "class_id": FSC_CLASS,
                    "box": list(local_box),
                    "source_image": f"{txt_path.stem}.jpg",
                }
            )
    print(f"[正样本] 收集 {len(manifest)} 个（跳过过小框 {skipped} 个）")
    return manifest


# ------------------------------------------------------------------------
# 负样本：checkpoint 在训练集上推理挖掘高分未匹配 FSC 预测
# ------------------------------------------------------------------------
def _collect_negatives(
    data_dir: Path,
    out_dir: Path,
    checkpoints: list[str],
    splits: list[str],
    context: float,
    iou_threshold: float,
    conf_threshold: float,
    max_neg: int,
    device: str,
    batch_size: int,
    num_workers: int,
    nms: bool,
) -> list[dict]:
    """用多个 checkpoint 在指定 split 推理，挖高分且与 GT IoU<阈值的 FSC 框。

    不同 checkpoint 的错误模式不同，合并挖掘能显著扩充负样本池；同时支持
    在多个 split 上挖掘（train 上模型过拟合、虚警少；val 未参与训练、
    虚警分布更接近真实部署场景，且 val 已用于 best checkpoint 选择，
    挖掘其负样本不引入新的信息泄露类别）。
    """
    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"
    neg_out = out_dir / "negative"
    neg_out.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, tuple, float]] = []  # (stem, xyxy, score)
    for checkpoint in checkpoints:
        print(f"[负样本] 加载 checkpoint: {checkpoint}")
        model = RFDETR.from_checkpoint(checkpoint)
        for split in splits:
            image_paths = sorted((images_dir / split).glob("*.jpg"))
            print(f"[负样本] {split} 集推理 {len(image_paths)} 张 ...")
            records, _, _, _ = eval_lib.predict_batched_to_records(
                model,
                image_paths,
                device=device,
                conf_threshold=conf_threshold,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=25,
            )
            for rec in records:
                if rec.class_id != FSC_CLASS:
                    continue
                stem = Path(rec.image_id).stem
                candidates.append((stem, tuple(float(v) for v in rec.xyxy), float(rec.score)))
            print(f"[负样本] checkpoint {checkpoint} @ {split} 贡献候选 {len(candidates)} 条（累计）")
        del model
        import gc

        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    manifest: list[dict] = []
    skipped_gt = 0
    by_image: dict[str, list[tuple, float]] = {}
    for stem, box, score in candidates:
        by_image.setdefault(stem, []).append((box, score))

    for stem, preds in tqdm(by_image.items(), desc="挖掘负样本"):
        # 找到该图所属 split（train/val），读取其 GT 与图片
        split = next((s for s in splits if (labels_dir / s / f"{stem}.txt").exists()), None)
        if split is None:
            continue
        txt_path = labels_dir / split / f"{stem}.txt"
        img_path = _find_image(images_dir / split, stem)
        if img_path is None:
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        # 该图全部 GT（像素）
        gt_all: list[tuple] = []
        for ln in txt_path.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:])
            gt_all.append(_normalized_to_pixel(cx, cy, w, h, img_w, img_h))
        # 过滤：与任一 GT IoU >= iou_threshold 的预测是召回/匹配，不挖
        kept = []
        for box, score in preds:
            if all(compute_iou(box, g) < iou_threshold for g in gt_all):
                kept.append((box, score))
        if nms:
            boxes = [b for b, _ in kept]
            scores = [s for _, s in kept]
            mask = _nms(boxes, scores)
            kept = [k for k, m in zip(kept, mask, strict=False) if m]
        for box, score in kept:
            crop = _context_crop(box, img_w, img_h, context)
            # 硬过滤：裁剪区不得与任何 GT 相交（外扩后可能蹭到真目标边缘）
            if any(_intersects(crop, g) for g in gt_all):
                skipped_gt += 1
                continue
            patch_id = f"N{len(manifest):04d}"
            with Image.open(img_path) as img:
                patch_img = img.crop(crop)
            patch_img.convert("RGB").save(neg_out / f"{patch_id}.jpg", quality=95)
            manifest.append(
                {
                    "kind": "negative",
                    "id": patch_id,
                    "file": f"negative/{patch_id}.jpg",
                    "width": crop[2] - crop[0],
                    "height": crop[3] - crop[1],
                    "source_image": f"{stem}.jpg",
                }
            )
            if len(manifest) >= max_neg:
                break
        if len(manifest) >= max_neg:
            break
    print(f"[负样本] 收集 {len(manifest)} 个（跳过蹭 GT 裁剪 {skipped_gt} 个）")
    return manifest


# ------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------
def main() -> None:
    """解析参数、收集正/负补丁并写 manifest。"""
    parser = argparse.ArgumentParser(description="构建 FSC 正/负样本补丁池")
    parser.add_argument("--data-dir", type=str, required=True, help="SHWX 数据集根目录（含 images/ 与 labels/）")
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        default=None,
        help="负样本挖掘用 checkpoint（可传多个合并挖掘；省略则只收集正样本）",
    )
    parser.add_argument("--out-dir", type=str, default="data/fsc_patch_pool", help="补丁池输出目录")
    parser.add_argument("--pos-context", type=float, default=1.3, help="正样本外扩倍数")
    parser.add_argument("--neg-context", type=float, default=1.8, help="负样本外扩倍数")
    parser.add_argument("--min-box-px", type=int, default=16, help="正样本最小框边长（像素），小于则跳过")
    parser.add_argument("--iou-threshold", type=float, default=0.35, help="负样本与训练 GT 的 IoU 上限（比赛口径）")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="负样本挖掘置信度下限")
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "val"],
        help="负样本挖掘的 split（可传多个；val 未参与训练，虚警分布更接近真实场景）",
    )
    parser.add_argument("--max-neg", type=int, default=300, help="负样本数量上限")
    parser.add_argument("--nms", action="store_true", default=True, help="负样本 NMS 去重（默认开启）")
    parser.add_argument("--no-nms", dest="nms", action="store_false", help="关闭 NMS")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批量大小")
    parser.add_argument("--num-workers", type=int, default=12, help="推理预取 worker 数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--force", action="store_true", help="重建已存在的补丁池")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[跳过] {manifest_path} 已存在（--force 重建）")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    positives = _collect_positives(data_dir, out_dir, args.pos_context, args.min_box_px, args.seed)
    negatives: list[dict] = []
    if args.ckpt:
        negatives = _collect_negatives(
            data_dir,
            out_dir,
            args.ckpt,
            args.split,
            args.neg_context,
            args.iou_threshold,
            args.conf_threshold,
            args.max_neg,
            args.device,
            args.batch_size,
            args.num_workers,
            args.nms,
        )
    else:
        print("[负样本] 未提供 --ckpt，跳过负样本挖掘")

    manifest = {
        "version": MANIFEST_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "patches": positives + negatives,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 补丁池: 正 {len(positives)} / 负 {len(negatives)} -> {manifest_path}")


if __name__ == "__main__":
    main()
