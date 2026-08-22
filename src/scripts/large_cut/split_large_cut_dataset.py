# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""大图裁切数据集划分脚本：按比例随机切分 train/val/test 并建立符号链接。

数据布局（YOLO 格式，与 ``dataset_file: yolo`` 的数据加载要求一致）：

- ``images/*.jpg``：原始图像（全部文件，无划分）；
- ``labels/*.txt``：同名 YOLO 标注（归一化 ``class cx cy w h``）；
- ``data.yaml``：yolo.py 通过 ``path`` + ``train``/``val``/``test`` 键定位
  split 目录，labels 目录由路径中 ``images`` 段替换为 ``labels`` 自动推导
  （见 ``rfdetr/datasets/yolo.py::_parse_yaml_split_dirs``）。

本脚本不复制任何图像（3.7G），而是为每个 split 建立指向源文件的符号链接
（``is_file()`` 跟随链接，数据加载无感知）。原 ``data.yaml`` 先备份为
``data.yaml.bak`` 再写入指向 split 子目录的新配置。

用法：
    python src/scripts/large_cut/split_large_cut_dataset.py                          # 默认 8:1:1
    python src/scripts/large_cut/split_large_cut_dataset.py --ratios 0.8:0.1:0.1     # 比例可配置
    python src/scripts/large_cut/split_large_cut_dataset.py --force                  # 重建已有划分
    python src/scripts/large_cut/split_large_cut_dataset.py --dry-run                # 只打印计划
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#: 划分后的 split 名称顺序（与 yolo.py 的 data.yaml 键一致）
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def parse_ratios(raw: str) -> tuple[float, float, float]:
    """解析 ``8:1:1`` 形式的比例字符串为三个浮点数。

    Args:
        raw: 冒号分隔的比例字符串（如 ``"8:1:1"``、``"0.8:0.1:0.1"``）。

    Returns:
        ``(train, val, test)`` 三个正浮点数。

    Raises:
        ValueError: 格式非法（段数不足 3 或包含非数值）。
    """
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"比例须为 train:val:test 三段，得到: {raw}")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"比例须为数值，得到: {raw}") from exc
    if any(r <= 0 for r in ratios):
        raise ValueError(f"比例必须为正数，得到: {raw}")
    return ratios[0], ratios[1], ratios[2]


def plan_split(
    names: list[str],
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """按比例随机切分文件 stem 列表（纯函数，可单测）。

    固定种子的 ``np.random.Generator`` 保证划分结果可复现。切分方式：
    全部样本先随机打乱，再按 train → val → test 顺序截取（末段取余数，
    避免浮点误差导致样本丢失）。

    Args:
        names: 图像 stem 列表（不含扩展名）。
        ratios: ``(train, val, test)`` 比例（正数，不需要和为 1）。
        rng: 随机数生成器（由外部以固定种子创建，保证可复现）。

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` 的划分结果，
        与 ``names`` 的元素一一对应（无重复无遗漏）。
    """
    total = len(names)
    shuffled = names.copy()
    rng.shuffle(shuffled)

    n_train = int(total * ratios[0] / sum(ratios))
    n_val = int(total * ratios[1] / sum(ratios))
    # 末段取余数，避免浮点误差导致样本丢失

    split_names: dict[str, list[str]] = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return split_names


def _ensure_empty_split_dir(split_dir: Path, force: bool) -> None:
    """确保 split 目录可写入：已存在时按 ``force`` 决定清空或跳过。

    Args:
        split_dir: ``images/{train,val,test}`` 或 ``labels/{train,val,test}``
            目标目录。
        force: ``True`` 时删除已有符号链接重建，``False`` 时保留。
    """
    if split_dir.exists():
        if not force:
            return
        # 只删除符号链接本身（不删除任何真实文件/目录）
        for entry in split_dir.iterdir():
            if entry.is_symlink():
                entry.unlink()
    else:
        split_dir.mkdir(parents=True, exist_ok=True)


def create_split_links(
    dataset_dir: Path,
    split_names: dict[str, list[str]],
    *,
    force: bool = False,
) -> int:
    """为每个 split 建立 images/labels 符号链接（指向源文件，不复制数据）。

    幂等：目标链接已存在且指向同一源文件时跳过；``force=True`` 时先清空
    split 目录再重建（仅删除符号链接，不触碰源文件）。

    Args:
        dataset_dir: 数据集根目录（含 ``images/`` 与 ``labels/``）。
        split_names: ``{"train": [stems], ...}`` 划分结果。
        force: 已存在划分时是否重建。

    Returns:
        实际创建（或已存在跳过）的链接总数。
    """
    total_links = 0
    for split in SPLIT_NAMES:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        _ensure_empty_split_dir(image_dir, force)
        _ensure_empty_split_dir(label_dir, force)
        for stem in split_names[split]:
            src_image = dataset_dir / "images" / f"{stem}.jpg"
            src_label = dataset_dir / "labels" / f"{stem}.txt"
            dst_image = image_dir / f"{stem}.jpg"
            dst_label = label_dir / f"{stem}.txt"

            # 源文件缺失时告警跳过（保证 labels 与 images 一一对应）
            if not src_image.is_file():
                print(f"[w] 缺少图像文件，跳过: {src_image}")
                continue
            if not src_label.is_file():
                print(f"[w] 缺少标签文件，跳过: {src_label}")
                continue

            for src, dst in ((src_image, dst_image), (src_label, dst_label)):
                if dst.is_symlink() and dst.resolve() == src.resolve():
                    continue  # 已存在且指向正确，幂等跳过
                os.symlink(str(src.resolve()), str(dst))
                total_links += 1
    return total_links


def write_data_yaml(dataset_dir: Path) -> None:
    """备份原 ``data.yaml`` 并写入指向 split 子目录的新配置。

    新配置由 ``path`` + ``train: images/train``/``val``/``test`` 组成，
    labels 目录由 yolo.py 自动推导（``images`` 段替换为 ``labels``）。
    类别数 ``nc`` 与 ``names`` 从原 yaml 读取并保留；读取失败（如 yaml
    缺失/损坏）时回退单类 ``sample``。

    Args:
        dataset_dir: 数据集根目录。
    """
    data_yaml = dataset_dir / "data.yaml"
    backup = dataset_dir / "data.yaml.bak"

    # 解析原配置中的类别定义（划分不改变类别语义）
    nc = 1
    names: dict[int, str] = {0: "sample"}
    if data_yaml.exists():
        with open(data_yaml, encoding="utf-8") as f:
            old_cfg = yaml.safe_load(f) or {}
        if isinstance(old_cfg.get("names"), dict):
            names = {int(k): str(v) for k, v in old_cfg["names"].items()}
        nc = int(old_cfg.get("nc", len(names)))
        # 已划分过的 yaml（train 指向 images/train）不需要备份
        if old_cfg.get("train") != "images/train":
            backup.write_text(data_yaml.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[i] 原 data.yaml 已备份为: {backup}")

    # 逐键构造保证顺序稳定（path/train/val/test/nc/names）
    lines = [
        f"path: {str(dataset_dir.resolve())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {nc}",
        "names:",
    ]
    for class_id in sorted(names.keys()):
        lines.append(f"  {class_id}: {names[class_id]}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[i] data.yaml 已更新: {data_yaml}")


def _scan_dataset(dataset_dir: Path) -> tuple[list[str], list[str]]:
    """扫描数据集的图像 stem 与缺失标签的 stem。

    Args:
        dataset_dir: 数据集根目录（含 ``images/`` 与 ``labels/``）。

    Returns:
        ``(all_stems, missing_label_stems)``：全部图像 stem（按文件名排序）
        与标签缺失的 stem 列表。
    """
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"未找到图像目录: {image_dir}")

    all_stems: list[str] = []
    missing: list[str] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        all_stems.append(image_path.stem)
        if not (label_dir / f"{image_path.stem}.txt").is_file():
            missing.append(image_path.stem)
    if not all_stems:
        raise FileNotFoundError(f"图像目录中未找到 .jpg 文件: {image_dir}")
    return all_stems, missing


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="大图裁切数据集 train/val/test 划分（符号链接，不复制数据）")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/liu/wzt/datasets/large-cut",
        help="数据集根目录（含 images/ 与 labels/），默认 large-cut",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="8:1:1",
        help="train:val:test 比例（默认 8:1:1）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42，保证可复现）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已存在划分时先清理符号链接再重建",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印划分计划，不创建任何文件",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：扫描 → 规划划分 → 建立链接 → 更新 data.yaml。"""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    ratios = parse_ratios(args.ratios)

    all_stems, missing_stems = _scan_dataset(dataset_dir)
    if missing_stems:
        print(f"[w] 有 {len(missing_stems)} 张图像缺少同名标签，将跳过: {missing_stems[:5]} ...")
        all_stems = [stem for stem in all_stems if stem not in missing_stems]

    rng = np.random.default_rng(args.seed)
    split_names = plan_split(all_stems, ratios, rng)

    print(f"数据集: {dataset_dir} | 样本数: {len(all_stems)} | 种子: {args.seed}")
    for split in SPLIT_NAMES:
        print(f"  {split}: {len(split_names[split])}（{len(split_names[split]) / len(all_stems):.1%}）")

    if args.dry_run:
        print("[i] dry-run 模式：未创建任何文件")
        return

    total_links = create_split_links(dataset_dir, split_names, force=args.force)
    write_data_yaml(dataset_dir)
    print(f"[完成] 共建立/确认 {total_links} 个符号链接，划分完成")


if __name__ == "__main__":
    main()
