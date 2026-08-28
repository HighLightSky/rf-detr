"""RF-DETR 原始输出的通用解码。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from competition.contracts import InferenceTask, RawDetection


def _sigmoid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """以数值稳定方式计算 sigmoid。"""
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def decode_rfdetr_outputs(
    pred_logits: NDArray[np.floating],
    pred_boxes: NDArray[np.floating],
    tasks: list[InferenceTask],
    max_detections: int = 300,
) -> list[RawDetection]:
    """将 RF-DETR 的 logits 和归一化 cxcywh 框解码为像素 xyxy 框。"""
    logits = np.asarray(pred_logits, dtype=np.float32)
    boxes = np.asarray(pred_boxes, dtype=np.float32)
    if logits.ndim != 3 or boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("RF-DETR 输出形状必须为 [批次, 查询, 类别] 和 [批次, 查询, 4]")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[0] != len(tasks):
        raise ValueError("RF-DETR 输出批次或查询维度与输入任务不一致")
    decoded: list[RawDetection] = []
    for index, task in enumerate(tasks):
        scores = _sigmoid(logits[index]).reshape(-1)
        count = min(max_detections, scores.size)
        if count == 0:
            continue
        selected = np.argpartition(scores, scores.size - count)[-count:]
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        query_indices = selected // logits.shape[-1]
        class_indices = selected % logits.shape[-1]
        width, height = task.input_size
        task_boxes = boxes[index, query_indices]
        centers_x = task_boxes[:, 0] * width
        centers_y = task_boxes[:, 1] * height
        box_widths = task_boxes[:, 2] * width
        box_heights = task_boxes[:, 3] * height
        for score, class_id, center_x, center_y, box_width, box_height in zip(
            scores[selected], class_indices, centers_x, centers_y, box_widths, box_heights, strict=True
        ):
            left = float(np.clip(center_x - box_width / 2.0, 0.0, width))
            top = float(np.clip(center_y - box_height / 2.0, 0.0, height))
            right = float(np.clip(center_x + box_width / 2.0, 0.0, width))
            bottom = float(np.clip(center_y + box_height / 2.0, 0.0, height))
            decoded.append(
                RawDetection(
                    image_id=task.image_id,
                    class_id=int(class_id),
                    score=float(score),
                    xyxy=(left, top, right, bottom),
                    task_index=task.task_index,
                )
            )
    return decoded
