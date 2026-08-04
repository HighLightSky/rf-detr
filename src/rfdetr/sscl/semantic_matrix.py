# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLIP 类别语义相似度矩阵的构建、保存、加载与验证。

本模块在训练前离线运行：使用 CLIP 文本编码器对每个类别的多个 prompt
编码取平均，得到类别文本向量，再计算两两余弦相似度，构成类别语义
相似度矩阵。该矩阵后续用于 SSCL 损失中对易混类别对施加更强的分离约束。

CLIP 只参与本模块的离线计算，不参与在线训练或推理。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- 项目约定别名（见 AGENTS.md）

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# 默认 CLIP 模型名称（HuggingFace），首次运行会自动下载权重
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def build_semantic_similarity_matrix(
    class_prompts: dict[int, list[str]],
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """使用 CLIP 文本编码器构建类别语义相似度矩阵。

    对每个类别的所有 prompt 逐条编码后取平均作为该类别的文本向量，
    再计算任意两个类别向量间的余弦相似度。

    Args:
        class_prompts: ``{class_id: [prompt_1, prompt_2, ...]}`` 映射。
        model_name: HuggingFace CLIP 模型名称。
        device: 计算设备，默认自动选择 CUDA/CPU。

    Returns:
        形状 ``[num_classes, num_classes]`` 的余弦相似度矩阵，对角线上为 1。

    Raises:
        ValueError: 当 ``class_prompts`` 为空或存在无 prompt 的类别时抛出。
    """
    if not class_prompts:
        raise ValueError("class_prompts 不能为空。")
    if any(len(prompts) == 0 for prompts in class_prompts.values()):
        raise ValueError("每个类别至少需要 1 个 prompt。")

    # 延迟导入 transformers，避免未安装时拖慢包导入。
    # 注：transformers 5.x 中 CLIPModel 不再内置 tokenizer，需单独加载。
    from transformers import AutoTokenizer, CLIPModel

    logger.info(f"加载 CLIP 模型: {model_name}（设备: {device}）")
    clip = CLIPModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class_vectors: list[torch.Tensor] = []
    for class_id in sorted(class_prompts.keys()):
        prompts = class_prompts[class_id]
        text_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            # 使用 text encoder 输出并执行 mean-pooling + 归一化，
            # 与 CLIP 图像-文本对比训练时的 text feature 提取方式一致。
            last_hidden_state = clip.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            ).last_hidden_state
            # 仅对真实 token（非 pad）求均值，再经投影头映射到对比空间
            attention_mask = text_inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            text_embed = clip.text_projection(pooled)
            text_embed = F.normalize(text_embed, dim=-1)
        class_vector = text_embed.mean(dim=0)  # 多 prompt 取平均
        class_vectors.append(class_vector)
        logger.info(f"  类别 {class_id}: {len(prompts)} 个 prompt 已编码")

    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 堆叠为 [C, D] 并归一化，计算余弦相似度矩阵
    vectors = F.normalize(torch.stack(class_vectors, dim=0), dim=-1)
    matrix = vectors @ vectors.T
    # 数值误差可能导致对角线略偏离 1，强制修正
    matrix = matrix.clamp(min=-1.0, max=1.0)
    matrix.fill_diagonal_(1.0)
    logger.info(f"语义相似度矩阵构建完成，形状: {tuple(matrix.shape)}")
    return matrix


def normalize_semantic_matrix(
    matrix: torch.Tensor,
    mode: str = "minmax",
    temperature: float = 0.1,
    top_k: int | None = None,
) -> torch.Tensor:
    """对原始 CLIP 余弦相似度矩阵做后处理。

    Args:
        matrix: 原始余弦相似度矩阵 ``[C, C]``。
        mode: 归一化方式：
            - ``"minmax"``: 线性映射到 ``[0, 1]`` 区间（默认）。
            - ``"softmax"``: 按行做温度缩放的 softmax，使相似度分布更尖锐。
        temperature: ``softmax`` 模式下的温度系数，越小越尖锐。
        top_k: 若给定，仅保留每行最相似的 top-k 个类别对，其余置 0。

    Returns:
        处理后的相似度矩阵，形状不变。

    Raises:
        ValueError: 当 ``mode`` 不是受支持的值时抛出。
    """
    result = matrix.clone()
    if mode == "minmax":
        # 保留对角线（自相似为 1），仅将非对角线元素线性映射到 [0, 1]
        mask = torch.eye(result.shape[0], device=result.device, dtype=torch.bool)
        off_diag = result[~mask]
        if off_diag.numel() > 0:
            lo, hi = off_diag.min(), off_diag.max()
            if hi - lo > 1e-8:
                result[~mask] = (off_diag - lo) / (hi - lo)
            else:
                result[~mask] = 0.0
        result.fill_diagonal_(1.0)
    elif mode == "softmax":
        result = F.softmax(result / temperature, dim=-1)
        # softmax 后对角线不再是 1，但保持对称性即可
        result = (result + result.T) / 2.0
    else:
        raise ValueError(f"不支持的归一化方式: {mode}，可选: 'minmax', 'softmax'")

    if top_k is not None and top_k > 0:
        # 仅保留每行 top-k 个（含自身），其余置 0，降低无关类别的噪声影响
        values, _ = torch.topk(result, k=min(top_k, result.shape[0]), dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        result = torch.where(result >= threshold, result, torch.zeros_like(result))
    return result


def save_semantic_matrix(matrix: torch.Tensor, path: str) -> None:
    """将语义相似度矩阵保存到文件。

    Args:
        matrix: 语义相似度矩阵 ``[C, C]``。
        path: 输出文件路径（``.pt`` 后缀）。
    """
    torch.save({"semantic_matrix": matrix.cpu()}, path)
    logger.info(f"语义相似度矩阵已保存到: {path}")


def load_semantic_matrix(path: str) -> torch.Tensor:
    """从文件加载语义相似度矩阵。

    Args:
        path: 之前 ``save_semantic_matrix`` 保存的文件路径。

    Returns:
        语义相似度矩阵 ``[C, C]``。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        KeyError: 当文件中缺少 ``semantic_matrix`` 键时抛出。
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"语义相似度矩阵文件不存在: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "semantic_matrix" not in data:
        raise KeyError(f"文件 {path} 中缺少 'semantic_matrix' 键。")
    return data["semantic_matrix"]


def validate_matrix(
    matrix: torch.Tensor,
    class_names: list[str],
    group_a: list[int] | None = None,
    group_b: list[int] | None = None,
) -> dict[str, Any]:
    """验证语义相似度矩阵质量，供人工检查。

    通用统计：
    - 全局最相似的非自反类别对（确认没有异常的跨组相似）。
    - （可选）当给定 ``group_a``/``group_b`` 时，额外计算各组内部及两组之间
      的平均相似度，用于观察易混类别分组是否符合预期。

    Args:
        matrix: 语义相似度矩阵 ``[C, C]``。
        class_names: 按类别索引顺序排列的类别名称列表。
        group_a: 第一组类别索引列表，默认 ``None``（不计算组统计）。
        group_b: 第二组类别索引列表，默认 ``None``（不计算组统计）。

    Returns:
        包含关键统计量的字典：``max_pair``、``max_pair_value``；
        当给定组参数时额外包含 ``group_a_mean``、``group_b_mean``、
        ``cross_ab_mean``。

    Raises:
        ValueError: 当 ``class_names`` 长度与矩阵类别数不一致时抛出。
    """
    num_classes = matrix.shape[0]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names 长度 {len(class_names)} 与矩阵类别数 {num_classes} 不一致")

    def _mean_of_pairs(ids_a: list[int], ids_b: list[int]) -> float:
        """计算两组类别之间（含组内）的平均相似度，越界索引自动忽略。"""
        values = []
        for i in ids_a:
            for j in ids_b:
                if 0 <= i < num_classes and 0 <= j < num_classes:
                    values.append(float(matrix[i, j]))
        return sum(values) / len(values) if values else 0.0

    stats: dict[str, Any] = {}

    # 全局最相似的非自反类别对
    flat = matrix.clone().fill_diagonal_(-2.0)  # 排除对角线
    max_ij = int(flat.argmax().item())
    i, j = max_ij // num_classes, max_ij % num_classes
    stats["max_pair"] = f"{class_names[i]}-{class_names[j]}"
    stats["max_pair_value"] = float(matrix[i, j])

    # 可选：组内/跨组平均相似度
    if group_a:
        stats["group_a_mean"] = _mean_of_pairs(group_a, group_a)
    if group_b:
        stats["group_b_mean"] = _mean_of_pairs(group_b, group_b)
    if group_a and group_b:
        stats["cross_ab_mean"] = _mean_of_pairs(group_a, group_b)

    log_lines = [f"  全局最相似类别对: {stats['max_pair']} = {stats['max_pair_value']:.4f}"]
    if "group_a_mean" in stats:
        log_lines.append(f"  组 A 内部平均相似度: {stats['group_a_mean']:.4f}")
    if "group_b_mean" in stats:
        log_lines.append(f"  组 B 内部平均相似度: {stats['group_b_mean']:.4f}")
    if "cross_ab_mean" in stats:
        log_lines.append(f"  组 A-B 平均相似度: {stats['cross_ab_mean']:.4f}")
    logger.info("语义矩阵验证:\n" + "\n".join(log_lines))
    return stats
