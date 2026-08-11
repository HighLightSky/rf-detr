# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""类别实例数统计脚本（分类损失均衡化的数据准备）。

遍历 YOLO 格式的 labels 目录（每行 ``class_id cx cy w h``），统计每个类别在
训练集中的实例总数，并输出 JSON 供 ``class_balance_counts_path`` 使用：

.. code-block:: json

    {
        "counts": [n0, n1, ...],
        "n_max": 2032,
        "n_min": 6,
        "n_ref": 110.4,
        "class_names": ["HM", "LQS", ...]
    }

其中 ``n_ref = sqrt(n_max * n_min)``（几何平均），作为 P0 正样本类均衡的参考
样本数 N_ref（避免直接用 N_max 使权重过大）。

用法：
    python src/scripts/stat_class_counts.py <labels_dir> <output_json> [num_classes] [class_names_csv]

- labels_dir：YOLO 训练集标签目录（``*.txt``）。
- output_json：输出 JSON 路径，建议写到实验目录（随实验保存，避免手写错）。
- num_classes：类别数，默认 25。
- class_names_csv：可选，逗号分隔的类别名（与 data.yaml 顺序一致），用于审计。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def count_class_instances(labels_dir: str | Path, num_classes: int) -> list[int]:
    """统计 YOLO labels 目录下每个类别的实例总数。

    Args:
        labels_dir: YOLO 标签目录（每行 ``class_id cx cy w h``）。
        num_classes: 类别总数，超出范围的 class_id 视为数据错误。

    Returns:
        长度为 ``num_classes`` 的计数列表。

    Raises:
        ValueError: 标签中的 class_id 超出 ``[0, num_classes)``。
    """
    counts = [0] * num_classes
    label_dir = Path(labels_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    for txt in sorted(label_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            class_id = int(line.split()[0])
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"{txt.name} 含越界类别 {class_id}（num_classes={num_classes}）")
            counts[class_id] += 1
    return counts


def write_counts_json(
    labels_dir: str | Path,
    output_path: str | Path,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """统计类别实例数并写出 JSON。

    Args:
        labels_dir: YOLO 标签目录。
        output_path: 输出 JSON 路径（父目录不存在时自动创建）。
        num_classes: 类别总数。
        class_names: 可选类别名列表（长度须为 num_classes），写入 JSON 供审计。

    Returns:
        写出的完整 JSON 内容（同时便于单测断言）。
    """
    counts = count_class_instances(labels_dir, num_classes)
    n_max = float(max(counts))
    positive_counts = [count for count in counts if count > 0]
    if not positive_counts:
        raise ValueError("所有类别计数均为 0，请检查 labels_dir 是否指向训练集标签目录")
    n_min = float(min(positive_counts))
    payload = {
        "counts": counts,
        "n_max": n_max,
        "n_min": n_min,
        "n_zero": counts.count(0),
        "n_ref": math.sqrt(n_max * n_min),
    }
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(f"class_names 长度 {len(class_names)} 与 num_classes {num_classes} 不一致")
        payload["class_names"] = class_names
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    """命令行入口：解析 sys.argv 并写出统计 JSON。"""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    labels_dir = sys.argv[1]
    output_path = sys.argv[2]
    num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    class_names = [n for n in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    payload = write_counts_json(labels_dir, output_path, num_classes, class_names)
    print(f"已统计 {len(payload['counts'])} 个类别，总实例数 {sum(payload['counts'])}")
    print(f"n_max={payload['n_max']:.0f}  n_min={payload['n_min']:.0f}  n_ref={payload['n_ref']:.2f}")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
