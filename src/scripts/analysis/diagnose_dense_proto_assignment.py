# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""诊断 dense ProtoGuidance 的 proposal-to-GT 正样本分配质量。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.training.module_data import RFDETRDataModule  # noqa: E402
from rfdetr.utilities import box_ops  # noqa: E402
from rfdetr.variants import RFDETRMedium  # noqa: E402


def _assign_labels_with_sources(
    dense_boxes: Tensor,
    dense_scores: Tensor,
    target_boxes: Tensor,
    target_labels: Tensor,
    iou_pos: float,
    fallback_topk: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """复现 dense loss 标签分配，并额外返回 IoU 与 fallback 来源。

    Returns:
        ``(labels, iou_positive, fallback_positive, best_iou)``。
    """
    labels = torch.full((dense_boxes.shape[0],), -1, dtype=torch.long, device=dense_boxes.device)
    iou_positive = torch.zeros_like(labels, dtype=torch.bool)
    fallback_positive = torch.zeros_like(labels, dtype=torch.bool)
    if target_boxes.numel() == 0:
        return labels, iou_positive, fallback_positive, dense_boxes.new_zeros((dense_boxes.shape[0],))

    dense_xyxy = box_ops.box_cxcywh_to_xyxy(dense_boxes)
    target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
    ious, _ = box_ops.box_iou(dense_xyxy, target_xyxy)
    best_iou, best_target = ious.max(dim=1)
    iou_positive = best_iou >= iou_pos
    labels[iou_positive] = target_labels[best_target[iou_positive]]

    centers = dense_boxes[:, :2]
    for target_idx in range(target_boxes.shape[0]):
        if bool((iou_positive & (best_target == target_idx)).any()):
            continue
        x0, y0, x1, y1 = target_xyxy[target_idx]
        inside = (
            (centers[:, 0] >= x0)
            & (centers[:, 0] <= x1)
            & (centers[:, 1] >= y0)
            & (centers[:, 1] <= y1)
            & (best_target == target_idx)
        )
        candidate_indices = inside.nonzero(as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        count = min(fallback_topk, int(candidate_indices.numel()))
        selected = candidate_indices[dense_scores[candidate_indices].topk(count).indices]
        labels[selected] = target_labels[target_idx]
        fallback_positive[selected] = True
    return labels, iou_positive, fallback_positive, best_iou


def _parse_args() -> argparse.Namespace:
    """解析诊断参数。"""
    parser = argparse.ArgumentParser(description="诊断 dense ProtoGuidance token 标签分配")
    parser.add_argument("--checkpoint", required=True, help="含 ProtoGuidance 配置的训练 checkpoint")
    parser.add_argument("--dataset-dir", required=True, help="YOLO 格式训练数据集目录")
    parser.add_argument("--output", required=True, help="诊断 JSON 输出路径")
    parser.add_argument("--batch-size", type=int, default=2, help="验证 batch 大小")
    parser.add_argument("--num-workers", type=int, default=2, help="验证 DataLoader worker 数")
    parser.add_argument("--max-images", type=int, default=0, help="最多诊断图像数，0 表示全部")
    return parser.parse_args()


def _class_names(dataset_dir: Path) -> list[str]:
    """从 data.yaml 读取类别名称。"""
    import yaml

    data = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[index]) for index in range(len(names))]
    return [str(name) for name in names]


def _new_class_stats() -> dict[str, int]:
    """创建单类别累计计数器。"""
    return {
        "gt": 0,
        "iou_covered": 0,
        "fallback_covered": 0,
        "uncovered": 0,
        "positive_tokens": 0,
        "linear_topk_covered": 0,
        "proto_topk_covered": 0,
        "linear_topk_class_correct": 0,
        "proto_topk_class_correct": 0,
        "linear_topk_positive_tokens": 0,
        "proto_topk_positive_tokens": 0,
    }


def main() -> None:
    """在验证集运行 dense 分配与原型分类诊断。"""
    args = _parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    model = RFDETRMedium.from_checkpoint(checkpoint)
    train_config = model.get_train_config(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",
        output_dir=str(output.parent),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module = RFDETRDataModule(model.model_config, train_config)
    data_module.setup("validate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = model.model.model.to(device).train()
    proto_guidance = nn_model.transformer.proto_guidance
    if proto_guidance is None or not nn_model.transformer.proto_guidance_dense_loss_enabled:
        raise ValueError("checkpoint 未启用 dense ProtoGuidance 输出，无法诊断。")

    class_names = _class_names(dataset_dir)
    iou_pos = float(model.model_config.proto_guidance_dense_iou_pos)
    fallback_topk = int(model.model_config.proto_guidance_dense_center_fallback_topk)
    aggregate: dict[str, Any] = {
        "images": 0,
        "gt": 0,
        "iou_covered_gt": 0,
        "fallback_covered_gt": 0,
        "uncovered_gt": 0,
        "iou_positive_tokens": 0,
        "fallback_positive_tokens": 0,
        "positive_tokens": 0,
        "correct_positive_tokens": 0,
        "box_values_outside_unit_interval": 0,
        "box_values_total": 0,
        "linear_topk_gt_covered": 0,
        "proto_topk_gt_covered": 0,
        "topk_changed_tokens": 0,
        "topk_total_tokens": 0,
        "linear_topk_positive_tokens": 0,
        "proto_topk_positive_tokens": 0,
        "linear_topk_class_correct": 0,
        "proto_topk_class_correct": 0,
    }
    per_class: dict[str, dict[str, int]] = defaultdict(_new_class_stats)

    with torch.no_grad():
        proto_guidance.current_epoch = proto_guidance.warmup_epochs
        for samples, targets in data_module.val_dataloader():
            samples = samples.to(device)
            output_dict = nn_model(samples, targets)
            dense = output_dict["enc_outputs"]
            logits = dense["pred_proto_logits_dense"]
            boxes = dense["pred_proto_boxes_dense"]
            scores = dense["pred_proto_scores_dense"]
            foreground_logits = dense.get("pred_proto_fg_logits_dense")
            for batch_idx, target in enumerate(targets):
                if args.max_images and aggregate["images"] >= args.max_images:
                    break
                target_boxes = target["boxes"].to(device=device, dtype=boxes.dtype)
                target_labels = target["labels"].to(device=device, dtype=torch.long)
                labels, iou_positive, fallback_positive, best_iou = _assign_labels_with_sources(
                    boxes[batch_idx],
                    scores[batch_idx],
                    target_boxes,
                    target_labels,
                    iou_pos=iou_pos,
                    fallback_topk=fallback_topk,
                )
                del best_iou
                aggregate["images"] += 1
                aggregate["gt"] += int(target_labels.numel())
                aggregate["iou_positive_tokens"] += int(iou_positive.sum())
                aggregate["fallback_positive_tokens"] += int(fallback_positive.sum())
                aggregate["box_values_total"] += int(boxes[batch_idx].numel())
                aggregate["box_values_outside_unit_interval"] += int(
                    ((boxes[batch_idx] < 0.0) | (boxes[batch_idx] > 1.0)).sum()
                )
                positive = labels >= 0
                aggregate["positive_tokens"] += int(positive.sum())
                aggregate["correct_positive_tokens"] += int(
                    (logits[batch_idx][positive].argmax(dim=-1) == labels[positive]).sum()
                )

                dense_xyxy = box_ops.box_cxcywh_to_xyxy(boxes[batch_idx])
                target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
                iou_matrix, _ = box_ops.box_iou(dense_xyxy, target_xyxy)
                _, best_target = iou_matrix.max(dim=1)
                topk_count = min(nn_model.num_queries, scores.shape[1])
                linear_topk = scores[batch_idx].topk(topk_count).indices
                if foreground_logits is not None:
                    foreground_score = foreground_logits[batch_idx]
                    proto_topk = (
                        scores[batch_idx]
                        + proto_guidance.lambda_pos_effective()
                        * proto_guidance.calibrate_position_score(
                            foreground_score.unsqueeze(0), scores[batch_idx].unsqueeze(0)
                        ).squeeze(0)
                    ).topk(topk_count).indices
                else:
                    proto_topk = linear_topk
                proto_classes = logits[batch_idx].argmax(dim=-1)
                aggregate["topk_changed_tokens"] += int(
                    (~torch.isin(proto_topk, linear_topk)).sum()
                )
                aggregate["topk_total_tokens"] += topk_count
                for target_idx, class_id in enumerate(target_labels.tolist()):
                    name = class_names[class_id]
                    stats = per_class[name]
                    stats["gt"] += 1
                    iou_covered = bool((iou_positive & (best_target == target_idx)).any())
                    fallback_covered = bool(fallback_positive[labels == class_id].any()) and not iou_covered
                    if iou_covered:
                        aggregate["iou_covered_gt"] += 1
                        stats["iou_covered"] += 1
                    elif fallback_covered:
                        aggregate["fallback_covered_gt"] += 1
                        stats["fallback_covered"] += 1
                    else:
                        aggregate["uncovered_gt"] += 1
                        stats["uncovered"] += 1
                    linear_covered = bool((iou_matrix[linear_topk, target_idx] >= iou_pos).any())
                    proto_covered = bool((iou_matrix[proto_topk, target_idx] >= iou_pos).any())
                    if linear_covered:
                        aggregate["linear_topk_gt_covered"] += 1
                        stats["linear_topk_covered"] += 1
                    if proto_covered:
                        aggregate["proto_topk_gt_covered"] += 1
                        stats["proto_topk_covered"] += 1
                    linear_matched = linear_topk[iou_matrix[linear_topk, target_idx] >= iou_pos]
                    proto_matched = proto_topk[iou_matrix[proto_topk, target_idx] >= iou_pos]
                    if linear_matched.numel():
                        linear_correct = int((proto_classes[linear_matched] == class_id).sum())
                        aggregate["linear_topk_positive_tokens"] += int(linear_matched.numel())
                        aggregate["linear_topk_class_correct"] += linear_correct
                        stats["linear_topk_positive_tokens"] += int(linear_matched.numel())
                        stats["linear_topk_class_correct"] += linear_correct
                    if proto_matched.numel():
                        proto_correct = int((proto_classes[proto_matched] == class_id).sum())
                        aggregate["proto_topk_positive_tokens"] += int(proto_matched.numel())
                        aggregate["proto_topk_class_correct"] += proto_correct
                        stats["proto_topk_positive_tokens"] += int(proto_matched.numel())
                        stats["proto_topk_class_correct"] += proto_correct
                for class_id, name in enumerate(class_names):
                    per_class[name]["positive_tokens"] += int((labels == class_id).sum())
            if args.max_images and aggregate["images"] >= args.max_images:
                break

    positive_tokens = max(1, aggregate["positive_tokens"])
    gt_count = max(1, aggregate["gt"])
    report = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "iou_pos": iou_pos,
        "fallback_topk": fallback_topk,
        "aggregate": {
            **aggregate,
            "gt_coverage": 1.0 - aggregate["uncovered_gt"] / gt_count,
            "iou_gt_coverage": aggregate["iou_covered_gt"] / gt_count,
            "fallback_gt_coverage": aggregate["fallback_covered_gt"] / gt_count,
            "positive_accuracy": aggregate["correct_positive_tokens"] / positive_tokens,
            "mean_positive_tokens_per_gt": aggregate["positive_tokens"] / gt_count,
            "box_value_outside_rate": aggregate["box_values_outside_unit_interval"]
            / max(1, aggregate["box_values_total"]),
            "linear_topk_gt_coverage": aggregate["linear_topk_gt_covered"] / gt_count,
            "proto_topk_gt_coverage": aggregate["proto_topk_gt_covered"] / gt_count,
            "topk_change_ratio": aggregate["topk_changed_tokens"] / max(1, aggregate["topk_total_tokens"]),
            "linear_topk_class_accuracy": aggregate["linear_topk_class_correct"]
            / max(1, aggregate["linear_topk_positive_tokens"]),
            "proto_topk_class_accuracy": aggregate["proto_topk_class_correct"]
            / max(1, aggregate["proto_topk_positive_tokens"]),
        },
        "per_class": {
            name: {
                **stats,
                "coverage": 1.0 - stats["uncovered"] / max(1, stats["gt"]),
                "iou_coverage": stats["iou_covered"] / max(1, stats["gt"]),
                "fallback_coverage": stats["fallback_covered"] / max(1, stats["gt"]),
                "linear_topk_coverage": stats["linear_topk_covered"] / max(1, stats["gt"]),
                "proto_topk_coverage": stats["proto_topk_covered"] / max(1, stats["gt"]),
                "linear_topk_class_accuracy": stats["linear_topk_class_correct"]
                / max(1, stats["linear_topk_positive_tokens"]),
                "proto_topk_class_accuracy": stats["proto_topk_class_correct"]
                / max(1, stats["proto_topk_positive_tokens"]),
            }
            for name, stats in sorted(per_class.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    for name in ("MS", "FSC"):
        if name in report["per_class"]:
            print(f"{name}: {json.dumps(report['per_class'][name], ensure_ascii=False)}")
    print(f"诊断结果已保存: {output}")


if __name__ == "__main__":
    main()
