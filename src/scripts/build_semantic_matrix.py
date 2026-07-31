# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""离线构建 SHWX 数据集类别的 CLIP 语义相似度矩阵。

SSCL 方案的第一阶段：使用 CLIP 文本编码器为 25 个类别生成语义相似度
矩阵，并输出验证统计供人工检查。该矩阵是 SSCL 训练的唯一外部先验来源。

用法：
    python src/scripts/build_semantic_matrix.py [--model openai/clip-vit-large-patch14]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 强制 transformers/huggingface_hub 离线模式：
# 本脚本使用本地缓存的 CLIP 模型（网络不可用时的离线方案），
# 避免 from_pretrained 尝试访问网络导致失败。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 将项目 src 目录加入 sys.path，便于直接运行脚本
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rfdetr.sscl import (  # noqa: E402
    build_semantic_similarity_matrix,
    save_semantic_matrix,
    validate_matrix,
)
from rfdetr.sscl.prompts import SHWX_CLASS_NAMES, SHWX_CLASS_PROMPTS  # noqa: E402


def main() -> None:
    """构建并验证语义相似度矩阵。"""
    parser = argparse.ArgumentParser(description="构建 SHWX 类别 CLIP 语义相似度矩阵")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/liu/wzt/Ruiyingshizong/AeroGen/ckpt/clip/clip-vit-large-patch14",
        help=("HuggingFace CLIP 模型名称或本地路径。默认使用本地缓存路径 （网络不可用时的离线方案）"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "semantic_matrix_shwx.pt"),
        help="语义相似度矩阵输出路径（默认: data/semantic_matrix_shwx.pt）",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"类别数: {len(SHWX_CLASS_PROMPTS)}")
    print(f"CLIP 模型: {args.model}")

    # 1. 构建语义相似度矩阵
    matrix = build_semantic_similarity_matrix(
        class_prompts=SHWX_CLASS_PROMPTS,
        model_name=args.model,
    )

    # 2. 保存矩阵
    save_semantic_matrix(matrix, str(output_path))
    print(f"矩阵已保存到: {output_path}")

    # 3. 验证矩阵质量（供人工检查）
    class_names = [SHWX_CLASS_NAMES[i] for i in sorted(SHWX_CLASS_NAMES.keys())]
    stats = validate_matrix(matrix, class_names)

    # 输出最相似的 15 个类别对（排除自身），便于人工确认合理性
    flat = matrix.clone().fill_diagonal_(-2.0)
    num_classes = matrix.shape[0]
    values, indices = flat.flatten().topk(15)
    print("\n最相似的 15 个类别对:")
    for rank, (value, flat_idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        i, j = flat_idx // num_classes, flat_idx % num_classes
        print(f"  {rank:2d}. {class_names[i]:<10s} - {class_names[j]:<10s} : {value:.4f}")

    print("\n矩阵验证统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
