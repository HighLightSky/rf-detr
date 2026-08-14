# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切矩形边界检测器测试集评估：pycocotools AP（AP50/AP75/AP90/AP50:95）。

矩形边界检测的目标是"裁切越准越好"，比赛口径（固定 IoU=0.5 的 TP/FP/FN）
无法反映边界精度差异，因此本脚本用 pycocotools 的 COCOeval 在测试集
（``images/test`` + ``labels/test``，单类 sample）上评估 AP 曲线：
IOU 阈值 0.5 反映"框是否找对位置"，0.75/0.90 反映"边界是否够准"。

推理复用 ``eval_lib.predict_batched_to_records``（多进程预取 + GPU 批量
前向，方形拉伸预处理与训练一致），与 ``model.predict`` 逐像素一致。

用法：
    python src/scripts/eval_large_cut.py \
        --checkpoint output/0814-large-cut-rfdetr-nano-704-rot90/checkpoint_best_total.pth

输出：
    - ``output_dir/eval_result.txt``：AP 报告；
    - ``output_dir/visualization/*.jpg``：抽样 10 张的 GT（绿）/预测（红）叠加图；
    - ``output_dir/yolo_preds/``（--save-yolo-preds 时）：YOLO 格式预测 txt。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.eval_lib import (  # noqa: E402
    BoxRecord,
    build_image_size_map,
    predict_batched_to_records,
    read_test_image_paths,
    resolve_device,
    save_yolo_predictions,
)
from val.competition_metrics import load_yolo_labels  # noqa: E402

#: 可视化抽样张数（固定种子保证可复现）
VIS_SAMPLE_COUNT = 10

#: 可视化抽样随机种子
VIS_SEED = 42


def boxes_to_coco_gt(
    records: list[BoxRecord],
    image_ids: list[str],
    image_sizes: dict[str, tuple[int, int]],
) -> dict:
    """把真实框记录转换为 COCO 格式 gt dict（单类）。

    真值框按 ``image_id`` 组织，bbox 由 xyxy 转 ``[x, y, w, h]`` 像素坐标
    （pycocotools 标准格式），并计算面积与 iscrowd 字段。

    Args:
        records: 真实框记录列表（xyxy 像素坐标）。
        image_ids: 全部测试图像 id（stem）列表。
        image_sizes: ``{image_id: (width, height)}`` 映射。

    Returns:
        COCO 格式 gt dict（含 ``images``/``annotations``/``categories``）。
    """
    images = [
        {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "width": int(image_sizes[image_id][0]),
            "height": int(image_sizes[image_id][1]),
        }
        for image_id in image_ids
    ]
    annotations = []
    for ann_id, record in enumerate(records, start=1):
        x0, y0, x1, y1 = record.xyxy
        width = float(x1 - x0)
        height = float(y1 - y0)
        annotations.append(
            {
                "id": ann_id,
                "image_id": record.image_id,
                "category_id": 0,  # 单类 sample
                "bbox": [float(x0), float(y0), width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "sample"}],
    }


def boxes_to_coco_pred(records: list[BoxRecord], image_ids: list[str]) -> dict:
    """把预测框记录转换为 COCO 格式 pred dict（含 score）。

    Args:
        records: 预测框记录列表（xyxy 像素坐标 + score）。
        image_ids: 全部测试图像 id（stem）列表（保证每张图都有条目）。

    Returns:
        COCO 格式 pred dict。
    """
    images = [{"id": image_id} for image_id in image_ids]
    annotations = []
    for ann_id, record in enumerate(records, start=1):
        x0, y0, x1, y1 = record.xyxy
        width = float(x1 - x0)
        height = float(y1 - y0)
        annotations.append(
            {
                "id": ann_id,
                "image_id": record.image_id,
                "category_id": 0,  # 单类 sample
                "bbox": [float(x0), float(y0), width, height],
                "area": width * height,
                "score": float(record.score or 0.0),
            }
        )
    return {"images": images, "annotations": annotations}


def compute_coco_ap(gt: dict, pred: dict, max_dets: int = 500) -> dict[str, float]:
    """用 pycocotools 计算单类检测的 AP（全 IoU 档）。

    Args:
        gt: COCO 格式真值 dict（``boxes_to_coco_gt`` 输出）。
        pred: COCO 格式预测 dict（``boxes_to_coco_pred`` 输出）。
        max_dets: 每图最大检测数（大图矩形检测每图最多 183 框，默认 500 覆盖）。

    Returns:
        ``{"AP50:95": ..., "AP50": ..., "AP75": ..., "AP90": ...}``：
        AP50:95 为 IoU∈[0.5:0.05:0.95] 全档平均，AP50/AP75/AP90 为单档。
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_pred = COCO()
    coco_pred.dataset = pred
    coco_pred.createIndex()

    evaluator = COCOeval(coco_gt, coco_pred, "bbox")
    # maxDets 前 3 档须为 [1, 10, 100]（summarize 的 AP@[.5:.95] 用写死的
    # maxDets=100 查档），追加第 4 档覆盖每图最多 183 框（precision[..., -1]）
    evaluator.params.maxDets = [1, 10, 100, max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()  # 打印标准 AP 摘要（同时填充 evaluator.stats）
    stats = evaluator.stats  # [AP50:95, AP50, AP75, AP_small, AP_medium, AP_large, ...]

    # AP90：precision 张量 [iouThrs=10, recThrs=101, K=1, areaRng=1, maxDets=M]，
    # IoU 档 0.9 位于索引 8（0.5 + 8×0.05），取最后 maxDets 档
    precision = evaluator.eval["precision"]
    ap90 = float(precision[8, :, 0, 0, -1].mean()) if precision.size else 0.0
    return {
        "AP50:95": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP90": ap90,
    }


def save_vis_samples(
    image_paths: list[Path],
    gt_records: list[BoxRecord],
    pred_records: list[BoxRecord],
    output_dir: Path,
    count: int = VIS_SAMPLE_COUNT,
    seed: int = VIS_SEED,
) -> Path:
    """抽样可视化：原图上叠加 GT（绿）与预测（红）矩形。

    Args:
        image_paths: 测试图像路径列表。
        gt_records: 真实框记录列表。
        pred_records: 预测框记录列表。
        output_dir: 可视化输出目录（自动创建）。
        count: 抽样张数。
        seed: 抽样随机种子（固定保证可复现）。

    Returns:
        可视化输出目录路径。
    """
    rng = np.random.default_rng(seed)
    sampled = rng.choice(image_paths, size=min(count, len(image_paths)), replace=False)
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    gt_by_image: dict[str, list[BoxRecord]] = {}
    pred_by_image: dict[str, list[BoxRecord]] = {}
    for record in gt_records:
        gt_by_image.setdefault(record.image_id, []).append(record)
    for record in pred_records:
        pred_by_image.setdefault(record.image_id, []).append(record)

    for image_path in sampled:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        stem = image_path.stem
        for record in gt_by_image.get(stem, []):
            x0, y0, x1, y1 = (int(v) for v in record.xyxy)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)  # GT 绿框
        for record in pred_by_image.get(stem, []):
            x0, y0, x1, y1 = (int(v) for v in record.xyxy)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # 预测红框
        out_path = vis_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), image)
    print(f"[i] 可视化已保存: {vis_dir}（{len(sampled)} 张）")
    return vis_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="大图裁切矩形边界检测器测试集 AP 评估（pycocotools）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="边界检测器 checkpoint 路径（checkpoint_best_total.pth）",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/liu/wzt/datasets/large-cut",
        help="数据集根目录（含 images/test 与 labels/test），默认 large-cut",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=704,
        help="推理分辨率（须与训练一致，默认 704）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="候选框置信度下限（默认 0.1：AP 评估需保留低分候选交给 COCOeval 排序）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评估输出目录（默认与 checkpoint 同目录）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="GPU 单次前向的图像数（默认 32）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="CPU 预取 worker 数（默认 12）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推理设备（默认 cuda:0，无 CUDA 自动回退 CPU）",
    )
    parser.add_argument(
        "--save-yolo-preds",
        action="store_true",
        help="额外把预测框保存为 YOLO 格式 txt",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：推理 → 计算 AP → 可视化 → 写报告。"""
    from rfdetr import RFDETR

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    image_paths = read_test_image_paths(dataset_dir / "images" / "test")
    image_size_map = build_image_size_map(image_paths)
    gt_records = load_yolo_labels(dataset_dir / "labels" / "test", image_size_map)
    image_ids = [path.stem for path in image_paths]

    model = RFDETR.from_checkpoint(checkpoint_path, resolution=args.resolution)
    print(f"[i] 模型加载完成: {checkpoint_path}（{len(image_paths)} 张测试图）")

    pred_records, throughput, gpu_util, timed_images = predict_batched_to_records(
        model,
        image_paths,
        device=device,
        conf_threshold=args.conf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=1,
    )

    gt_dict = boxes_to_coco_gt(gt_records, image_ids, image_size_map)
    pred_dict = boxes_to_coco_pred(pred_records, image_ids)
    ap_metrics = compute_coco_ap(gt_dict, pred_dict)

    # ── 写评估报告 ──────────────────────────────────────────────────
    lines = [
        "=" * 80,
        "大图裁切矩形边界检测器评估报告（pycocotools AP）",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"权重: {checkpoint_path}",
        f"数据集: {dataset_dir}",
        f"测试图像数: {len(image_paths)} | 真实框数: {len(gt_records)} | 预测框数: {len(pred_records)}",
        f"置信度下限: {args.conf} | 推理吞吐: {throughput:.1f} img/s"
        + (f" | GPU 利用率: {gpu_util:.1f}%" if gpu_util is not None else ""),
        "-" * 80,
        f"AP50:95 = {ap_metrics['AP50:95']:.4f}",
        f"AP50    = {ap_metrics['AP50']:.4f}",
        f"AP75    = {ap_metrics['AP75']:.4f}",
        f"AP90    = {ap_metrics['AP90']:.4f}",
        "-" * 80,
        "说明: AP50 反映矩形位置是否找对；AP75/AP90 反映边界精度（裁切是否够准）。",
        "=" * 80,
    ]
    report_text = "\n".join(lines) + "\n"
    report_path = output_dir / "eval_result.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[完成] 评估报告已保存至: {report_path}")

    save_vis_samples(image_paths, gt_records, pred_records, output_dir)
    if args.save_yolo_preds:
        save_yolo_predictions(pred_records, output_dir / "yolo_preds", image_size_map)


if __name__ == "__main__":
    main()
