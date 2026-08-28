"""从实验仓库构建可审计的比赛交付资产集合。"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    """一个从仓库复制到交付 models 目录的模型或工件。"""

    source: Path
    destination: str


def _parse_args() -> argparse.Namespace:
    """解析可选的 PyTorch 运行时装配开关。"""
    parser = argparse.ArgumentParser(description="准备比赛 Docker 所需模型和运行时代码")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="RF-DETR 仓库根目录",
    )
    parser.add_argument(
        "--include-pytorch-runtime",
        action="store_true",
        help="同时复制 stage1 checkpoint、ProtoGuidance 工件和 rfdetr 运行时",
    )
    return parser.parse_args()


def _default_assets(source_root: Path, include_pytorch_runtime: bool) -> list[Asset]:
    """返回首版 ONNX 配置及可选 PyTorch 变体的明确资产清单。"""
    experiment_dir = source_root / "output/0827-0825基线方案/2基线+多模态原型+SSCL+难例抑制"
    assets = [
        Asset(experiment_dir / "onnx/shwx_detector_1024.onnx", "shwx_detector_1024.onnx"),
        Asset(experiment_dir / "onnx/large_cut_boundary_704.onnx", "large_cut_boundary_704.onnx"),
    ]
    if include_pytorch_runtime:
        assets.extend(
            [
                Asset(experiment_dir / "stage1.pth", "stage1.pth"),
                Asset(
                    source_root / "data/proto_guidance_shwx_1024_from120ep.pt",
                    "proto_guidance_shwx_1024_from120ep.pt",
                ),
            ]
        )
    return assets


def _copy_asset(asset: Asset, destination_dir: Path) -> None:
    """复制一个预先声明的文件，仅允许复用内容一致的已有资产。"""
    if not asset.source.is_file():
        raise FileNotFoundError(f"源资产不存在: {asset.source}")
    destination = destination_dir / asset.destination
    if destination.exists():
        if filecmp.cmp(asset.source, destination, shallow=False):
            return
        raise FileExistsError(f"目标资产已存在且内容不同，拒绝覆盖: {destination}")
    shutil.copy2(asset.source, destination)


def _copy_pytorch_runtime(source_root: Path, vendor_dir: Path) -> None:
    """复制 PyTorch RF-DETR 的受控运行时代码闭包。"""
    source = source_root / "src/rfdetr"
    destination = vendor_dir / "rfdetr"
    if not source.is_dir():
        raise FileNotFoundError(f"RF-DETR 源码不存在: {source}")
    if destination.exists():
        raise FileExistsError(f"PyTorch 运行时已存在，拒绝混合覆盖: {destination}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )


def main() -> None:
    """按选择的后端变体复制模型和运行时代码。"""
    args = _parse_args()
    source_root = args.source_root.resolve()
    delivery_dir = Path(__file__).resolve().parent
    model_dir = delivery_dir / "models"
    vendor_dir = delivery_dir / "app/vendor"
    for asset in _default_assets(source_root, args.include_pytorch_runtime):
        _copy_asset(asset, model_dir)
    if args.include_pytorch_runtime:
        _copy_pytorch_runtime(source_root, vendor_dir)


if __name__ == "__main__":
    main()
