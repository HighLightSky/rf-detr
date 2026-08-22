# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""离线构建数据集的 CLIP 语义相似度矩阵。

使用 CLIP 文本编码器，将数据集中各类别的提示词（来自
``src/rfdetr/sscl/prompts/<dataset>.yaml``）编码为类别文本向量，
两两计算余弦相似度，得到 ``[C, C]`` 的语义相似度矩阵并保存为 ``.pt``。
该矩阵是 SSCL 训练的唯一外部先验来源（CLIP 只参与离线计算，不参与在线训练）。

使用方案：
    # 1. 新增数据集：在 src/rfdetr/sscl/prompts/ 下添加 <dataset>.yaml
    #    （格式参考 shwx.yaml，包含 class_names 与 class_prompts 两部分）。
    #    之后本脚本无需任何改动，只要传入 --dataset 即可复用。

    # 2. 构建矩阵（以 SHWX / DIOR 为例）
    python src/scripts/data_prep/build_semantic_matrix.py --dataset shwx
    python src/scripts/data_prep/build_semantic_matrix.py --dataset dior

    # 3. 可选：指定 CLIP 模型（离线场景传本地缓存路径）与输出路径
    python src/scripts/data_prep/build_semantic_matrix.py --dataset dior \
        --model /path/to/clip-vit-large-patch14 \
        --output data/semantic_matrix_dior.pt

常用参数：
    --dataset  数据集名称（必填），对应 src/rfdetr/sscl/prompts/<dataset>.yaml
    --model    CLIP 模型名称或本地路径（默认取环境变量 CLIP_MODEL，否则用本地缓存路径）
    --output   矩阵输出路径（默认 data/semantic_matrix_<dataset>.pt）

输出文件默认保存在 data/semantic_matrix_<dataset>.pt，
训练脚本通过 --sscl-semantic-matrix-path 指定使用。
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
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rfdetr.sscl import (  # noqa: E402
    build_semantic_similarity_matrix,
    save_semantic_matrix,
    validate_matrix,
)
from rfdetr.sscl.prompts import load_class_prompts  # noqa: E402

# 默认 CLIP 模型：优先读环境变量 CLIP_MODEL，否则使用本机离线缓存路径。
# 均可通过 --model 显式覆盖。
_DEFAULT_CLIP_MODEL = os.environ.get(
    "CLIP_MODEL",
    "data/clip/clip-vit-large-patch14",
)


def main() -> None:
    """解析参数，加载提示词并构建、保存、验证语义相似度矩阵。"""
    parser = argparse.ArgumentParser(description="构建数据集的 CLIP 语义相似度矩阵")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称，对应 src/rfdetr/sscl/prompts/<dataset>.yaml（如 shwx、dior）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_CLIP_MODEL,
        help=f"CLIP 模型名称或本地路径（默认: {_DEFAULT_CLIP_MODEL}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="矩阵输出路径（默认: data/semantic_matrix_<dataset>.pt）",
    )
    args = parser.parse_args()

    # 1. 从 YAML 加载提示词与类别名称
    class_names, class_prompts = load_class_prompts(args.dataset)

    # 默认输出路径随数据集名变化，不写死具体数据集
    output_path = Path(args.output) if args.output else _PROJECT_ROOT / "data" / f"semantic_matrix_{args.dataset}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"数据集: {args.dataset}  类别数: {len(class_prompts)}")
    print(f"CLIP 模型: {args.model}")

    # 2. 构建语义相似度矩阵
    matrix = build_semantic_similarity_matrix(
        class_prompts=class_prompts,
        model_name=args.model,
    )

    # 3. 保存矩阵
    save_semantic_matrix(matrix, str(output_path))
    print(f"矩阵已保存到: {output_path}")

    # 4. 验证矩阵质量（供人工检查）
    names = [class_names[i] for i in sorted(class_names.keys())]
    stats = validate_matrix(matrix, names)

    # 输出最相似的 15 个类别对（排除自身），便于人工确认合理性
    flat = matrix.clone().fill_diagonal_(-2.0)
    num_classes = matrix.shape[0]
    values, indices = flat.flatten().topk(15)
    print("\n最相似的 15 个类别对:")
    for rank, (value, flat_idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        i, j = flat_idx // num_classes, flat_idx % num_classes
        print(f"  {rank:2d}. {names[i]:<22s} - {names[j]:<22s} : {value:.4f}")

    print("\n矩阵验证统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
