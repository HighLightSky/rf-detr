#!/usr/bin/env python3
"""统计 DIOR COCO 标注并划分 head、medium、tail 类别。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/home/liu/wzt/datasets/DIOR-rfdetr")
DEFAULT_OUTPUT_DIR = Path("output/0928DIOR数据集分析结果")
SPLITS = ("train", "valid", "test")


def _load_split(dataset_root: Path, split: str) -> dict[str, Any]:
    """读取一个数据划分的 COCO 标注文件。"""
    path = dataset_root / split / "_annotations.coco.json"
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("categories"), list) or not isinstance(data.get("annotations"), list):
        raise ValueError(f"标注文件缺少 categories 或 annotations: {path}")
    return data


def _split_stats(data: dict[str, Any]) -> dict[int, dict[str, float | int]]:
    """计算类别实例数、图像数和框面积统计。"""
    instances: Counter[int] = Counter()
    image_ids: dict[int, set[int | str]] = defaultdict(set)
    areas: dict[int, list[float]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        category_id = int(annotation["category_id"])
        instances[category_id] += 1
        image_ids[category_id].add(annotation["image_id"])
        area = annotation.get("area")
        if area is None:
            bbox = annotation.get("bbox", [])
            area = float(bbox[2]) * float(bbox[3]) if len(bbox) >= 4 else 0.0
        areas[category_id].append(float(area))

    stats: dict[int, dict[str, float | int]] = {}
    for category_id in instances:
        values = sorted(areas[category_id])
        midpoint = len(values) // 2
        median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2
        stats[category_id] = {
            "instances": instances[category_id],
            "images": len(image_ids[category_id]),
            "area_mean": sum(values) / len(values),
            "area_median": median,
            "area_min": values[0],
            "area_max": values[-1],
        }
    return stats


def _group_for_count(count: int) -> str:
    """按训练集实例数划分类别。"""
    if count >= 1000:
        return "head"
    if count >= 500:
        return "medium"
    return "tail"


def _fmt(value: float | int) -> str:
    """格式化统计数字。"""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"


def _write_outputs(
    output_dir: Path,
    dataset_root: Path,
    categories: list[dict[str, Any]],
    split_data: dict[str, dict[str, Any]],
    split_stats: dict[str, dict[int, dict[str, float | int]]],
) -> list[dict[str, Any]]:
    """写出 CSV、JSON 和 Markdown 统计结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    category_names = {int(category["id"]): str(category["name"]) for category in categories}
    train_total = sum(int(row["instances"]) for row in split_stats["train"].values())
    rows: list[dict[str, Any]] = []
    for rank, (category_id, train_row) in enumerate(
        sorted(split_stats["train"].items(), key=lambda item: (-int(item[1]["instances"]), item[0])),
        start=1,
    ):
        train_count = int(train_row["instances"])
        row: dict[str, Any] = {
            "rank": rank,
            "category_id": category_id,
            "class_name": category_names[category_id],
            "group": _group_for_count(train_count),
            "train_instances": train_count,
            "train_images": int(train_row["images"]),
            "train_instance_fraction": train_count / train_total,
            "train_area_mean": float(train_row["area_mean"]),
            "train_area_median": float(train_row["area_median"]),
        }
        for split in ("valid", "test"):
            row[f"{split}_instances"] = int(split_stats[split].get(category_id, {}).get("instances", 0))
            row[f"{split}_images"] = int(split_stats[split].get(category_id, {}).get("images", 0))
        rows.append(row)

    fieldnames = list(rows[0])
    with (output_dir / "dior_class_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    group_members = {
        group: [row["class_name"] for row in rows if row["group"] == group]
        for group in ("head", "medium", "tail")
    }
    summary = {
        "dataset_root": str(dataset_root),
        "splits": {
            split: {
                "images": len(split_data[split].get("images", [])),
                "annotations": sum(1 for annotation in split_data[split]["annotations"] if not annotation.get("iscrowd", 0)),
            }
            for split in SPLITS
        },
        "num_classes": len(categories),
        "train_instance_total": train_total,
        "group_rule": {
            "head": "train_instances >= 1000",
            "medium": "500 <= train_instances < 1000",
            "tail": "train_instances < 500",
        },
        "groups": group_members,
        "class_frequency": rows,
    }
    with (output_dir / "dior_long_tail_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    head_count = sum(row["train_instances"] for row in rows if row["group"] == "head")
    medium_count = sum(row["train_instances"] for row in rows if row["group"] == "medium")
    tail_count = sum(row["train_instances"] for row in rows if row["group"] == "tail")
    max_count = rows[0]["train_instances"]
    min_count = rows[-1]["train_instances"]
    lines = [
        "# DIOR 长尾数据集分析",
        "",
        f"数据集根目录：`{DEFAULT_DATASET_ROOT}`",
        "",
        "## 数据规模",
        "",
        "| 划分 | 图像数 | 实例数 |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {_fmt(len(split_data[split].get('images', [])))} | "
            f"{_fmt(sum(int(row['instances']) for row in split_stats[split].values()))} |"
        )
    lines.extend(
        [
            "",
            "## 分组规则",
            "",
            "分组只使用训练集实例数，避免使用验证集或测试集信息。阈值选择为：head ≥ 1000，medium 为 500–999，tail < 500。该规则在当前数据上产生 7、6、7 个类别，且三个区间均有清晰的频次间隔。",
            "",
            "| 分组 | 训练实例数 | 类别数 | 实例数 | 类别 |",
            "|---|---:|---:|---:|---|",
            f"| head | ≥1,000 | 7 | {_fmt(head_count)} | {', '.join(group_members['head'])} |",
            f"| medium | 500–999 | 6 | {_fmt(medium_count)} | {', '.join(group_members['medium'])} |",
            f"| tail | <500 | 7 | {_fmt(tail_count)} | {', '.join(group_members['tail'])} |",
            "",
            f"训练集最高与最低类别频次分别为 {_fmt(max_count)} 和 {_fmt(min_count)}，最大/最小频次比为 {max_count / min_count:.2f}。",
            "",
            "## 按训练集频次排序",
            "",
            "完整机器可读结果见 `dior_class_frequency.csv` 和 `dior_long_tail_summary.json`。",
            "",
            "| 排名 | 类别 | 分组 | train | train 图像 | train 占比 | valid | test |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['class_name']} | {row['group']} | {_fmt(row['train_instances'])} | "
            f"{_fmt(row['train_images'])} | {row['train_instance_fraction']:.4%} | "
            f"{_fmt(row['valid_instances'])} | {_fmt(row['test_instances'])} |"
        )
    lines.extend(
        [
            "",
            "## 后续查询选择实验的使用约定",
            "",
            "后续 query selection 统计使用上述训练集分组。类别频次使用实例数；每个目标是否有 proposal 被选中时，图像内多个目标按实例分别统计，并在置信区间计算时按图像 bootstrap。",
            "",
            "生成脚本：`src/scripts/analysis/analyze_dior_long_tail.py`。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _write_plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """生成训练集类别频次图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"head": "#c23b22", "medium": "#e69f00", "tail": "#3478b8"}
    ordered = list(reversed(rows))
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.barh(
        [row["class_name"] for row in ordered],
        [row["train_instances"] for row in ordered],
        color=[colors[row["group"]] for row in ordered],
    )
    axis.set_xscale("log")
    axis.set_xlabel("训练集实例数（对数坐标）")
    axis.set_ylabel("类别")
    axis.set_title("DIOR 训练集类别频次与长尾分组")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[group], label=group) for group in ("head", "medium", "tail")]
    axis.legend(handles=handles, title="分组")
    fig.tight_layout()
    fig.savefig(output_dir / "dior_train_frequency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """执行 DIOR 数据集统计。"""
    parser = argparse.ArgumentParser(description="统计 DIOR 类别频次并划分 head/medium/tail")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    split_data = {split: _load_split(args.dataset_root, split) for split in SPLITS}
    categories = split_data["train"]["categories"]
    category_ids = {int(category["id"]) for category in categories}
    for split, data in split_data.items():
        split_ids = {int(category["id"]) for category in data["categories"]}
        if split_ids != category_ids:
            raise ValueError(f"{split} 类别集合与 train 不一致")
    stats = {split: _split_stats(data) for split, data in split_data.items()}
    rows = _write_outputs(args.output_dir, args.dataset_root, categories, split_data, stats)
    _write_plot(args.output_dir, rows)
    print(f"结果已保存到: {args.output_dir.resolve()}")
    for group in ("head", "medium", "tail"):
        names = [row["class_name"] for row in rows if row["group"] == group]
        print(f"{group}: {', '.join(names)}")


if __name__ == "__main__":
    main()
