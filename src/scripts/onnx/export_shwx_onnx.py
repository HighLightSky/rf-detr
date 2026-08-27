# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""导出 SHWX 大图切分流水线使用的两个 ONNX 检测模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析双模型 ONNX 导出参数。"""
    parser = argparse.ArgumentParser(description="导出 SHWX 主检测与大图边界检测 ONNX 模型")
    parser.add_argument("--detector-checkpoint", required=True, help="SHWX 主检测 checkpoint 路径")
    parser.add_argument("--boundary-checkpoint", required=True, help="大图边界检测 checkpoint 路径")
    parser.add_argument("--output-dir", required=True, help="ONNX 输出目录")
    parser.add_argument("--detector-resolution", type=int, default=1024, help="主检测输入分辨率")
    parser.add_argument("--boundary-resolution", type=int, default=704, help="边界检测输入分辨率")
    return parser.parse_args()


def _export_checkpoint(checkpoint: Path, output_dir: Path, resolution: int, name: str) -> Path:
    """从 checkpoint 恢复模型并导出支持动态 batch 的固定空间 ONNX 图。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    model = RFDETR.from_checkpoint(str(checkpoint), resolution=resolution)
    export_path = model.export(
        output_dir=str(output_dir),
        shape=(resolution, resolution),
        batch_size=1,
        dynamic_batch=True,
        opset_version=17,
        notes={"checkpoint": str(checkpoint), "resolution": resolution, "role": name},
    )
    target = output_dir / f"{name}.onnx"
    if export_path != target:
        if target.exists():
            raise FileExistsError(f"ONNX 输出已存在，拒绝覆盖: {target}")
        export_path.replace(target)
    return target


def main() -> None:
    """导出主检测和边界检测 ONNX，并输出可直接填入 YAML 的路径。"""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_path = _export_checkpoint(
        Path(args.detector_checkpoint), output_dir, args.detector_resolution, "shwx_detector_1024"
    )
    boundary_path = _export_checkpoint(
        Path(args.boundary_checkpoint), output_dir, args.boundary_resolution, "large_cut_boundary_704"
    )
    print(f"[完成] 主检测 ONNX: {detector_path}")
    print(f"[完成] 边界检测 ONNX: {boundary_path}")


if __name__ == "__main__":
    main()
