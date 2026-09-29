"""用已训练 RF-DETR 检查 DIOR 类别频率与 two-stage query 选择的关系。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rfdetr import RFDETR
from rfdetr.datasets.coco import build_roboflow_from_coco
from rfdetr.models.transformer import gen_encoder_output_proposals
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou
from rfdetr.utilities.tensors import make_collate_fn


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "output/0804重标注之前的实验/0726-DIOR-rfdetr_medium/checkpoint_best_total.pth"
DEFAULT_DATASET = Path("/home/liu/wzt/datasets/DIOR-rfdetr")
DEFAULT_GROUPS = ROOT / "output/0928DIOR数据集分析结果/dior_long_tail_summary.json"
DEFAULT_OUTPUT = ROOT / "output/0928查询选择与类频率的关系"
KS = (50, 100, 200, 300)
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def measure_instances(
    proposal_boxes: Tensor,
    scores: Tensor,
    gt_boxes: Tensor,
    *,
    ks: tuple[int, ...] = KS,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> list[dict[str, Any]]:
    """计算每个 GT 的正 proposal 覆盖和 top-K 命中。"""
    if gt_boxes.numel() == 0:
        return []
    ious = box_iou(proposal_boxes, gt_boxes)[0]
    ranks = torch.empty_like(scores, dtype=torch.long)
    ranks[torch.argsort(scores, descending=True, stable=True)] = torch.arange(1, len(scores) + 1, device=scores.device)
    selected_indices = {k: torch.topk(scores, min(k, len(scores))).indices for k in ks}
    rows: list[dict[str, Any]] = []
    for gt_idx in range(gt_boxes.shape[0]):
        row: dict[str, Any] = {"best_proposal_iou": float(ious[:, gt_idx].max().item())}
        for threshold in iou_thresholds:
            suffix = f"iou{round(threshold * 100)}"
            positive = ious[:, gt_idx] >= threshold
            covered = bool(positive.any().item())
            best_rank = int(ranks[positive].min().item()) if covered else None
            row[f"covered_{suffix}"] = covered
            row[f"positive_proposals_{suffix}"] = int(positive.sum().item())
            row[f"best_positive_rank_{suffix}"] = best_rank
            for k in ks:
                row[f"selected_{suffix}_k{k}"] = bool(positive[selected_indices[k]].any().item())
        rows.append(row)
    return rows


class ProposalCapture:
    """通过只读 hooks 捕获 encoder 的未筛选分类分数和框。"""

    def __init__(self, model: torch.nn.Module) -> None:
        """安装 forward hooks。"""
        transformer = model.transformer
        self.transformer = transformer
        self.proposals: Tensor | None = None
        self.logits: Tensor | None = None
        self.deltas: Tensor | None = None
        self.handles = [
            transformer.register_forward_pre_hook(self._capture_proposals),
            transformer.enc_out_class_embed[0].register_forward_hook(self._capture_logits),
            transformer.enc_out_bbox_embed[0].register_forward_hook(self._capture_deltas),
        ]

    def _capture_proposals(self, module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        """用实际特征图尺寸和 mask 重建初始 proposal 网格。"""
        srcs, masks = inputs[0], inputs[1]
        self.logits = None
        self.deltas = None
        shapes = [tuple(src.shape[-2:]) for src in srcs]
        memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], dim=1)
        mask = torch.cat([part.flatten(1) for part in masks], dim=1) if masks is not None else None
        _, self.proposals = gen_encoder_output_proposals(
            memory, mask, shapes, unsigmoid=not self.transformer.bbox_reparam
        )

    def _capture_logits(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        """保存 group 0 的全量 encoder 类别 logits。"""
        if self.proposals is not None and output.shape[1] == self.proposals.shape[1]:
            self.logits = output

    def _capture_deltas(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        """保存 group 0 的全量 encoder 框回归输出。"""
        if self.proposals is not None and output.shape[1] == self.proposals.shape[1]:
            self.deltas = output

    def get(self) -> tuple[Tensor, Tensor]:
        """还原模型真实使用的 encoder 框和选择分数。"""
        if self.proposals is None or self.logits is None or self.deltas is None:
            raise RuntimeError("encoder proposal hooks 未捕获到完整结果")
        if self.transformer.bbox_reparam:
            boxes = torch.cat(
                [
                    self.deltas[..., :2] * self.proposals[..., 2:] + self.proposals[..., :2],
                    self.deltas[..., 2:].exp() * self.proposals[..., 2:],
                ],
                dim=-1,
            )
        else:
            boxes = (self.deltas + self.proposals).sigmoid()
        return boxes, self.logits.max(-1).values

    def close(self) -> None:
        """移除 hooks。"""
        for handle in self.handles:
            handle.remove()


def _rate(rows: list[dict[str, Any]], field: str, *, conditional: bool = False) -> float:
    """计算覆盖率或 top-K 选择率。"""
    subset = [row for row in rows if row["covered_iou50"]] if conditional else rows
    return sum(bool(row[field]) for row in subset) / len(subset) if subset else float("nan")


def _bootstrap(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    """按图像重采样，计算 head-tail 差异和频率相关的区间。"""
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row)
    image_ids = np.array(list(by_image))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    correlations: list[float] = []
    class_ids = sorted({int(row["category_id"]) for row in rows})
    frequencies = {int(row["category_id"]): int(row["train_instances"]) for row in rows}
    groups = {int(row["category_id"]): row["group"] for row in rows}
    image_counts = np.zeros((len(image_ids), len(class_ids), 2), dtype=np.int32)
    cid_index = {cid: index for index, cid in enumerate(class_ids)}
    for image_index, image_id in enumerate(image_ids):
        for row in by_image[int(image_id)]:
            if row["covered_iou50"]:
                class_index = cid_index[int(row["category_id"])]
                image_counts[image_index, class_index, 0] += 1
                image_counts[image_index, class_index, 1] += int(row["selected_iou50_k300"])
    group_indexes = {
        group: [cid_index[cid] for cid in class_ids if groups[cid] == group]
        for group in ("head", "tail")
    }
    log_freq = np.log([frequencies[cid] for cid in class_ids])
    for _ in range(samples):
        selection = rng.integers(0, len(image_ids), size=len(image_ids))
        counts = image_counts[selection].sum(axis=0)
        rates = np.divide(counts[:, 1], counts[:, 0], out=np.full(len(class_ids), np.nan), where=counts[:, 0] > 0)
        head = counts[group_indexes["head"]].sum(axis=0)
        tail = counts[group_indexes["tail"]].sum(axis=0)
        if head[0] and tail[0]:
            differences.append(float(head[1] / head[0] - tail[1] / tail[0]))
        if np.isfinite(rates).all():
            correlations.append(float(spearmanr(log_freq, rates).statistic))
    return {
        "head_minus_tail_selection_rate_ci95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "spearman_rho_ci95": np.percentile(correlations, [2.5, 97.5]).tolist(),
        "bootstrap_samples": samples,
        "bootstrap_unit": "image",
    }


def _summarize(rows: list[dict[str, Any]], class_info: dict[int, dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    """汇总逐类、分组和相关性统计。"""
    per_class = []
    for category_id, info in sorted(class_info.items(), key=lambda item: -item[1]["train_instances"]):
        subset = [row for row in rows if row["category_id"] == category_id]
        covered = [row for row in subset if row["covered_iou50"]]
        per_class.append(
            {
                **info,
                "validation_gt": len(subset),
                "proposal_coverage_iou50": _rate(subset, "covered_iou50"),
                "selection_rate_iou50_k300_given_coverage": _rate(subset, "selected_iou50_k300", conditional=True),
                "selection_rate_iou50_k300_all_gt": _rate(subset, "selected_iou50_k300"),
                "median_best_positive_rank_iou50": float(np.median([r["best_positive_rank_iou50"] for r in covered])) if covered else None,
                "median_gt_area_fraction": float(np.median([r["gt_area_fraction"] for r in subset])) if subset else None,
            }
        )
    groups = {}
    for group in ("head", "medium", "tail"):
        subset = [row for row in rows if row["group"] == group]
        groups[group] = {
            "validation_gt": len(subset),
            "proposal_coverage_iou50": _rate(subset, "covered_iou50"),
            "selection_rate_iou50_k300_given_coverage": _rate(subset, "selected_iou50_k300", conditional=True),
            "selection_rate_iou50_k300_all_gt": _rate(subset, "selected_iou50_k300"),
            "macro_selection_rate_iou50_k300_given_coverage": float(np.mean([
                row["selection_rate_iou50_k300_given_coverage"] for row in per_class if row["group"] == group
            ])),
        }
        for threshold in IOU_THRESHOLDS:
            suffix = f"iou{round(threshold * 100)}"
            covered = [row for row in subset if row[f"covered_{suffix}"]]
            groups[group][f"proposal_coverage_{suffix}"] = len(covered) / len(subset)
            for k in KS:
                groups[group][f"selection_rate_{suffix}_k{k}_given_coverage"] = (
                    sum(row[f"selected_{suffix}_k{k}"] for row in covered) / len(covered) if covered else None
                )
    log_freq = np.log([row["train_instances"] for row in per_class])
    selection = [row["selection_rate_iou50_k300_given_coverage"] for row in per_class]
    correlation = spearmanr(log_freq, selection)
    correlation_without_top2 = spearmanr(log_freq[2:], selection[2:])
    return {
        "per_class": per_class,
        "groups": groups,
        "spearman_log_train_frequency_vs_conditional_selection": {
            "rho": float(correlation.statistic),
            "p_two_sided": float(correlation.pvalue),
            "n_classes": len(per_class),
        },
        "spearman_without_two_most_frequent_classes": {
            "rho": float(correlation_without_top2.statistic),
            "p_two_sided": float(correlation_without_top2.pvalue),
            "n_classes": len(per_class) - 2,
        },
        "bootstrap": _bootstrap(rows, samples=bootstrap_samples, seed=20260928),
    }


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """保存逐实例数据、统计报告和图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    palette = {"head": "#c23b22", "medium": "#e69f00", "tail": "#3478b8"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ordered = list(reversed(summary["per_class"]))
    axes[0].barh(
        [row["class_name"] for row in ordered],
        [row["selection_rate_iou50_k300_given_coverage"] for row in ordered],
        color=[palette[row["group"]] for row in ordered],
    )
    axes[0].set_xlabel("P(进入 top-300 | 存在 IoU≥0.5 正 proposal)")
    axes[0].set_xlim(0, 1)
    axes[1].scatter(
        [row["train_instances"] for row in summary["per_class"]],
        [row["selection_rate_iou50_k300_given_coverage"] for row in summary["per_class"]],
        c=[palette[row["group"]] for row in summary["per_class"]],
        s=60,
    )
    for row in summary["per_class"]:
        axes[1].annotate(row["class_name"], (row["train_instances"], row["selection_rate_iou50_k300_given_coverage"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("训练集类别实例数（对数坐标）")
    axes[1].set_ylabel("条件选择率")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Spearman ρ={summary['spearman_log_train_frequency_vs_conditional_selection']['rho']:.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "selection_vs_frequency.png", dpi=180)
    plt.close(fig)

    h = summary["groups"]["head"]
    t = summary["groups"]["tail"]
    rho = summary["spearman_log_train_frequency_vs_conditional_selection"]
    ci = summary["bootstrap"]["head_minus_tail_selection_rate_ci95"]
    lines = [
        "# DIOR 查询选择与类频率的关系",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"数据集: `{summary['dataset_root']}/valid`，图像 {summary['images_processed']} 张，GT {len(rows)} 个。",
        f"推理输入: {summary['resolution']}×{summary['resolution']}，query K=300，模型 eval/group 0。",
        "",
        "定义：GT 与未筛选 encoder 回归框 IoU≥0.5 时存在正 proposal；其中至少一个进入原模型 top-300 则算命中。选择率以存在正 proposal 的 GT 为分母。",
        "",
        "| 分组 | GT | 正 proposal 覆盖率 | 条件 top-300 选择率 | 全部 GT 命中率 | 类别宏平均条件选择率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("head", "medium", "tail"):
        g = summary["groups"][group]
        lines.append(f"| {group} | {g['validation_gt']:,} | {g['proposal_coverage_iou50']:.2%} | {g['selection_rate_iou50_k300_given_coverage']:.2%} | {g['selection_rate_iou50_k300_all_gt']:.2%} | {g['macro_selection_rate_iou50_k300_given_coverage']:.2%} |")
    lines.extend([
        "",
        f"head - tail 条件选择率差：{h['selection_rate_iou50_k300_given_coverage'] - t['selection_rate_iou50_k300_given_coverage']:+.2%}；按图像 bootstrap 95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}]。",
        f"20 类 Spearman(训练频次, 条件选择率)：ρ={rho['rho']:.3f}，双侧 p={rho['p_two_sided']:.4g}；去掉最高频两类后 ρ={summary['spearman_without_two_most_frequent_classes']['rho']:.3f}，p={summary['spearman_without_two_most_frequent_classes']['p_two_sided']:.4g}。",
        "",
        "## 逐类结果",
        "",
        "| 类别 | 分组 | 训练实例 | 验证 GT | 正 proposal 覆盖率 | 条件选择率 | 全部 GT 命中率 | 正 proposal 最佳 rank 中位数 | GT 面积中位数占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["per_class"]:
        rank = "-" if row["median_best_positive_rank_iou50"] is None else f"{row['median_best_positive_rank_iou50']:.0f}"
        coverage = "-" if np.isnan(row["proposal_coverage_iou50"]) else f"{row['proposal_coverage_iou50']:.2%}"
        conditional = "-" if np.isnan(row["selection_rate_iou50_k300_given_coverage"]) else f"{row['selection_rate_iou50_k300_given_coverage']:.2%}"
        all_gt = "-" if np.isnan(row["selection_rate_iou50_k300_all_gt"]) else f"{row['selection_rate_iou50_k300_all_gt']:.2%}"
        area = "-" if row["median_gt_area_fraction"] is None else f"{row['median_gt_area_fraction']:.4%}"
        lines.append(f"| {row['class_name']} | {row['group']} | {row['train_instances']:,} | {row['validation_gt']:,} | {coverage} | {conditional} | {all_gt} | {rank} | {area} |")
    lines.extend([
        "",
        "## 解释限制",
        "",
        "该分析是观察性结果。类别频率与目标尺寸、场景密度及类别难度可能相关；相关性本身不能证明频率造成选择偏置。IoU 匹配按 encoder 回归框定义，阈值和 K 敏感性数据保存在 `summary.json`。只有选择率差的图像 bootstrap 区间排除 0 且类别级相关检验显著时，才支持两个命题；否则应如实报告未验证。",
        "",
        "逐实例数据见 `per_instance.csv`，可用于面积分层与图像聚类重采样。",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """运行完整验证集的 encoder query selection 诊断。"""
    parser = argparse.ArgumentParser(description="验证 DIOR 类频率与 query top-K 选择的关系")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--group-file", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前实验需要 CUDA GPU")
    groups = json.loads(args.group_file.read_text(encoding="utf-8"))
    class_info = {
        int(row["category_id"]): {
            "category_id": int(row["category_id"]),
            "class_name": row["class_name"],
            "group": row["group"],
            "train_instances": int(row["train_instances"]),
        }
        for row in groups["class_frequency"]
    }
    training_config_path = args.checkpoint.parent / "training_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    model_config = training_config["model_config"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_names = checkpoint["args"].get("class_names")
    sorted_names = [class_info[cid]["class_name"] for cid in sorted(class_info)]
    if checkpoint_names != sorted_names:
        raise ValueError("checkpoint 类别顺序与 DIOR 标注不一致")
    detector = RFDETR.from_checkpoint(
        args.checkpoint,
        resolution=int(model_config["resolution"]),
        positional_encoding_size=int(model_config["positional_encoding_size"]),
    )
    model = detector.model.model.to("cuda").eval()
    capture = ProposalCapture(model)
    dataset_args = detector.model.args
    dataset_args.dataset_dir = str(args.dataset_root)
    dataset_args.square_resize_div_64 = bool(training_config["train_config"]["square_resize_div_64"])
    dataset = build_roboflow_from_coco("val", dataset_args, int(model_config["resolution"]))
    if args.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(int(model_config["patch_size"]) * int(model_config["num_windows"])),
        pin_memory=True,
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for samples, targets in tqdm(loader, desc="DIOR valid encoder query selection"):
            model(samples.to("cuda"))
            proposal_boxes, scores = capture.get()
            for batch_index, target in enumerate(targets):
                gt_boxes = box_cxcywh_to_xyxy(target["boxes"].to("cuda"))
                per_gt = measure_instances(
                    box_cxcywh_to_xyxy(proposal_boxes[batch_index].float()),
                    scores[batch_index].float(),
                    gt_boxes,
                )
                for gt_index, record in enumerate(per_gt):
                    category_id = int(target["labels"][gt_index].item()) + 1
                    record.update(
                        {
                            "image_id": int(target["image_id"].item()),
                            **class_info[category_id],
                            "gt_area_fraction": float((target["boxes"][gt_index, 2] * target["boxes"][gt_index, 3]).item()),
                        }
                    )
                    rows.append(record)
    capture.close()
    if not rows:
        raise RuntimeError("验证集没有可统计的 GT")
    summary = _summarize(rows, class_info, args.bootstrap_samples)
    summary.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "group_file": str(args.group_file.resolve()),
            "resolution": int(model_config["resolution"]),
            "images_processed": len({row["image_id"] for row in rows}),
            "iou_thresholds": IOU_THRESHOLDS,
            "top_k_values": KS,
        }
    )
    _write_outputs(args.output_dir, rows, summary)
    print(f"结果已保存到 {args.output_dir.resolve()}")
    print(json.dumps({"groups": summary["groups"], "spearman": summary["spearman_log_train_frequency_vs_conditional_selection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
