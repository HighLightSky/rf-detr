"""安全批量推理辅助：启动按空闲显存 clamp batch，运行时 OOM 退避。

设计目标：目标服务器显存未知时，既想用批量换取吞吐，又不能因显存不足而崩溃。
两条防线：
1. 启动标定 —— 用两次探测前向量出"每张图约需的推理增量显存"，再按空闲显存把
   ``batch_size`` 下探到安全值（不会主动放大，只会在显存不足时降低）。
2. 运行时 OOM 退避 —— 若标定低估或显存被并发占用（预留过小、碎片化），在逐个 batch
   前向包一层异常处理：捕获显存不足异常后把该批批量减半重试，并把安全值持久化，
   避免后续图像重复触发 OOM。

本模块只提供纯函数与调度逻辑，具体的"如何量峰值显存 / 如何判定 OOM"由各后端注入，
以兼顾 PyTorch（用 torch 缓存分配器峰值）与 ONNX（异常消息启发式）的差异。
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger("rf-detr.runtime")

# 推理峰值显存占卡片空闲显存的比例上限，为基座/碎片/并发预留余量。
BATCH_MEMORY_RATIO = 0.6
# 启动标定使用的两个探测批量（差值除以图数得到每图增量）。
_PROBE_LOW = 1
_PROBE_HIGH = 5


def clamp_batch(requested: int, free_mb: float, per_image_mb: Optional[float]) -> int:
    """按空闲显存把请求的批量下探到安全值。

    Args:
        requested: 配置中请求的 batch_size。
        free_mb: 推理开始时卡片空闲显存（MB，已扣除模型/运行时基座）。
        per_image_mb: 每张图约需的推理增量显存（MB）。为 ``None`` 或非正时视为无法标定，
            保守返回 ``requested``，交由运行时 OOM 退避兜底。

    Returns:
        夹在 ``[1, requested]`` 区间内的安全批量。
    """
    if per_image_mb is None or per_image_mb <= 0:
        return requested
    budget = free_mb * BATCH_MEMORY_RATIO
    max_batch = max(1, int(budget // per_image_mb))
    return min(requested, max_batch)


def safe_batch_forward(
    total: int,
    batch_size: int,
    run_slice: Callable[[int, int], list],
    is_oom: Callable[[Exception], bool],
    on_oom: Optional[Callable[[], None]] = None,
) -> tuple[list, int]:
    """按批量执行 ``run_slice``，并在显存不足时减半重试。

    ``run_slice(start, count)`` 处理 ``[start, start+count)`` 这批任务，返回该批的检测结果
    （按任务顺序）。``is_oom(exc)`` 判定异常是否为显存不足；``on_oom()`` 为可选回调（如
    ``empty_cache``），在重试前释放缓存。

    Args:
        total: 任务总数。
        batch_size: 初始批量（可为被 clamp 后的安全值）。
        run_slice: 执行一批前向并返回检测结果的函数。
        is_oom: 判定一次异常是否为显存不足。
        on_oom: 可选，OOM 后重试前的清理回调。

    Returns:
        (detections, final_effective_batch)：检测结果列表；实际使用的批量（含退避后的值），
        调用方可据此持久化，供后续图像复用。
    """
    eff = max(1, batch_size)
    start = 0
    detections: list = []
    while start < total:
        count = min(eff, total - start)
        try:
            detections.extend(run_slice(start, count))
            start += count
        except Exception as exc:  # noqa: BLE001 - 退出条件由 is_oom 收窄为"显存不足"
            if eff > 1 and is_oom(exc):
                if on_oom is not None:
                    on_oom()
                eff = max(1, eff // 2)
                logger.warning("检测推理遇显存不足(OOM)，批量减半至 %d 后重试", eff)
                # 不推进 start，重试当前这批
            else:
                raise
    return detections, eff
