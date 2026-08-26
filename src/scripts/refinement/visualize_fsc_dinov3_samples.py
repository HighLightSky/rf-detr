# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""将 FSC DINOv3 二级头的候选训练样本导出为可检查图集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.refinement import crop_fsc_context  # noqa: E402

_INTERNAL_HOLDOUT_SALT = "fsc-dinov3-head-v1:"
_TILE_SIZE = 224
_GAP = 12


def _parse_args() -> argparse.Namespace:
    """解析样本图集导出参数。"""
    parser = argparse.ArgumentParser(description="导出 FSC DINOv3 二级头候选训练样本图集")
    parser.add_argument("--cache", required=True, help="二级候选缓存 JSON")
    parser.add_argument("--output-dir", required=True, help="图集输出目录")
    parser.add_argument("--samples-per-group", type=int, default=16, help="每个标签组导出的固定样本数")
    parser.add_argument("--columns", type=int, default=4, help="图集列数")
    return parser.parse_args()


def _inner_split(row: dict[str, Any]) -> str:
    """按二级头训练时使用的规则划分训练和内部留出样本。"""
    split_key = f"{_INTERNAL_HOLDOUT_SALT}{Path(row['image']).name}"
    digest = hashlib.sha256(split_key.encode("utf-8")).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def select_rows(payload: dict[str, Any], split: str, label: int, limit: int) -> list[dict[str, Any]]:
    """确定性地选择指定内部划分和标签的候选样本。"""
    if split not in {"train", "holdout"}:
        raise ValueError("split 必须为 train 或 holdout")
    if label not in {0, 1}:
        raise ValueError("label 必须为 0 或 1")
    if limit <= 0:
        raise ValueError("limit 必须为正数")
    rows = [
        row
        for row in payload["candidates"]
        if row["split"] == "train" and _inner_split(row) == split and int(row["label"]) == label
    ]
    return sorted(rows, key=lambda row: hashlib.sha256(f"sample-v1:{row['image']}:{row['prediction_index']}".encode("utf-8")).hexdigest())[:limit]


def _font() -> ImageFont.ImageFont:
    """获取稳定的默认字体。"""
    return ImageFont.load_default()


def _candidate_overview(image: Image.Image, xyxy: list[float]) -> Image.Image:
    """生成带候选框的较宽局部场景视图。"""
    x0, y0, x1, y1 = xyxy
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    left = max(int(x0 - width), 0)
    top = max(int(y0 - height), 0)
    right = min(int(x1 + width), image.width)
    bottom = min(int(y1 + height), image.height)
    overview = image.crop((left, top, right, bottom)).resize((_TILE_SIZE, _TILE_SIZE))
    draw = ImageDraw.Draw(overview)
    scale_x = _TILE_SIZE / max(right - left, 1)
    scale_y = _TILE_SIZE / max(bottom - top, 1)
    box = ((x0 - left) * scale_x, (y0 - top) * scale_y, (x1 - left) * scale_x, (y1 - top) * scale_y)
    draw.rectangle(box, outline=(255, 64, 64), width=3)
    return overview


def _sample_tile(row: dict[str, Any], label_name: str) -> Image.Image:
    """生成一个含场景、真实输入裁剪和元信息的样本块。"""
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
    overview = _candidate_overview(image, row["xyxy"])
    model_crop = crop_fsc_context(image, row["xyxy"], context_scale=2.0, output_size=_TILE_SIZE)
    tile = Image.new("RGB", (_TILE_SIZE * 2 + _GAP, _TILE_SIZE + 30), "white")
    tile.paste(overview, (0, 0))
    tile.paste(model_crop, (_TILE_SIZE + _GAP, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((2, _TILE_SIZE + 5), f"{label_name}  score={float(row['score']):.3f}", fill="black", font=_font())
    return tile


def _save_grid(rows: list[dict[str, Any]], label_name: str, columns: int, output_path: Path) -> None:
    """保存固定宽度的候选样本图集。"""
    if not rows:
        return
    sample_width = _TILE_SIZE * 2 + _GAP
    sample_height = _TILE_SIZE + 30
    grid_rows = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * sample_width, grid_rows * sample_height), "white")
    for index, row in enumerate(rows):
        x = (index % columns) * sample_width
        y = (index // columns) * sample_height
        canvas.paste(_sample_tile(row, label_name), (x, y))
    canvas.save(output_path)


def _write_manifest(groups: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    """写出每个图集样本的源文件和候选框，便于回查。"""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "label", "score", "image", "xyxy"])
        writer.writeheader()
        for group, rows in groups.items():
            for row in rows:
                writer.writerow({"group": group, "label": row["label"], "score": row["score"], "image": row["image"], "xyxy": json.dumps(row["xyxy"])})


def main() -> None:
    """从训练候选缓存生成正负样本可视化图集。"""
    args = _parse_args()
    if args.samples_per_group <= 0 or args.columns <= 0:
        raise ValueError("samples-per-group 和 columns 必须为正数")
    payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    if payload.get("format") != "shwx-fsc-verifier-cache-v1":
        raise ValueError("候选缓存格式错误")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "训练集_正例": select_rows(payload, "train", 1, args.samples_per_group),
        "训练集_负例": select_rows(payload, "train", 0, args.samples_per_group),
        "内部留出_正例": select_rows(payload, "holdout", 1, args.samples_per_group),
        "内部留出_负例": select_rows(payload, "holdout", 0, args.samples_per_group),
    }
    for group, rows in groups.items():
        _save_grid(rows, "FSC" if rows and int(rows[0]["label"]) else "non-FSC", args.columns, output_dir / f"{group}.jpg")
    _write_manifest(groups, output_dir / "样本索引.csv")
    (output_dir / "README.md").write_text(
        "# 二级头训练样本图集\n\n"
        "每个样本左图是一级 FSC 候选附近的 3 倍局部场景，红框为一级候选；右图是二级 DINOv3 实际输入的 2 倍上下文裁剪。"
        "正例表示候选与 FSC 标注匹配，负例表示未匹配 FSC 标注的一级 FSC 候选。图集仅来自原始训练划分。\n",
        encoding="utf-8",
    )
    print(f"[完成] 可视化目录: {output_dir}")


if __name__ == "__main__":
    main()
