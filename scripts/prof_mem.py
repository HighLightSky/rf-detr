# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""显存剖析脚本：复现 train.py 的 SGA+CFE 配置，测量多尺度下的峰值显存。

用法：
    python scripts/prof_mem.py

逐个运行 {scale} x {case} 的 forward+backward，打印峰值 CUDA 显存。
"""
from __future__ import annotations

import itertools
import os

import torch

from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.variants import RFDETRMedium
from rfdetr.utilities.tensors import NestedTensor

DEVICE = "cuda"

CASES = {
    # 之前 Phase 1（可正常训练）：SGA on，CFE off，单级 P4
    "Phase1 SGA+P4": dict(use_sga=True, use_cfe=False, projector_scale=["P4"]),
    # 当前失败配置：SGA+CFE，双级 P3/P4
    "Phase2 SGA+CFE P3P4": dict(use_sga=True, use_cfe=True, projector_scale=["P3", "P4"]),
}

SCALES = [480, 640, 768, 800]
B = 16
NUM_OBJS = int(os.environ.get("NUM_OBJS", "100"))  # 每图目标数，可用 NUM_OBJS 环境变量覆盖


def build_model(**kw: object) -> tuple[torch.nn.Module, object]:
    """照 train.py 构建 medium 模型，返回 (nn.Module, args_namespace)。"""
    wrapper = RFDETRMedium(
        num_classes=25,
        resolution=640,
        gradient_checkpointing=True,
        **kw,
    )
    model = wrapper.model.model  # RFDETR -> ModelContext -> LWDETR nn.Module
    model.train()
    model.to(DEVICE)
    criterion, _ = build_criterion_and_postprocessors(wrapper.model.args)
    criterion.to(DEVICE)
    return model, criterion


def make_batch(scale: int, num_objs: int = NUM_OBJS) -> tuple[NestedTensor, list[dict[str, torch.Tensor]]]:
    """构造方形 scale 的合成 batch。"""
    tensors = torch.rand(B, 3, scale, scale, device=DEVICE)
    mask = torch.zeros(B, scale, scale, dtype=torch.bool, device=DEVICE)
    targets: list[dict[str, torch.Tensor]] = []
    for _ in range(B):
        boxes = torch.rand(num_objs, 4, device=DEVICE) * 0.8 + 0.1  # cxcywh
        boxes[:, 2:] = boxes[:, 2:].clamp(min=0.05)  # 保证非退化
        targets.append(
            {
                "labels": torch.randint(0, 25, (num_objs,), device=DEVICE),
                "boxes": boxes,
                "area": torch.rand(num_objs, device=DEVICE) * scale * scale,
                "iscrowd": torch.zeros(num_objs, dtype=torch.bool, device=DEVICE),
            }
        )
    return NestedTensor(tensors, mask), targets


def run(model: torch.nn.Module, criterion: object, scale: int, num_objs: int = NUM_OBJS) -> float:
    """跑一次 forward + criterion(matcher+loss) + backward，返回峰值显存 MB。"""
    torch.cuda.reset_peak_memory_stats()
    samples, targets = make_batch(scale, num_objs)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(samples, targets)
        loss_dict = criterion(out, targets)  # 含匈牙利匹配的 giou/cdist/分类成本矩阵
        loss = sum(loss_dict[k] * 1.0 for k in loss_dict if k.endswith("loss_ce")) + sum(
            loss_dict[k] for k in loss_dict if "bbox" in k or "giou" in k
        )
        loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    del samples, targets, out, loss_dict, loss
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    peak_stats: dict[str, list[float]] = {}

    for case_name, kw in CASES.items():
        model, criterion = build_model(**kw)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n=== {case_name} (参数量 {n_params:.1f}M) ===")
        row: list[float] = []
        for scale in SCALES:
            try:
                peak = run(model, criterion, scale)
            except torch.cuda.OutOfMemoryError as exc:
                import traceback

                traceback.print_exc()
                torch.cuda.empty_cache()
                print(f"  scale={scale:4d}  peak=OOM!")
                row.append(float("inf"))
                break
            row.append(peak)
            print(f"  scale={scale:4d}  peak={peak:8.1f} MB")
        peak_stats[case_name] = row
        del model, criterion
        torch.cuda.empty_cache()

    # 汇总
    print("\n===== 汇总（峰值 MB）=====")
    print(f"{'case':<20}" + "".join(f"{s:>10}" for s in SCALES))
    for name, row in peak_stats.items():
        print(f"{name:<20}" + "".join(f"{v:>10.0f}" for v in row))


if __name__ == "__main__":
    main()
