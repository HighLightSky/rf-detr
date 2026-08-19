# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""使用单个 YAML 顺序训练 SHWX PAN/RGB 两个 RF-DETR medium 模型。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import expcfg  # noqa: E402
from scripts.dual_shwx import PAN_CLASSES, RGB_CLASSES, prepare_dual_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="SHWX PAN/RGB 双模型顺序训练")
    parser.add_argument("-c", "--config", required=True, help="双模型 YAML 配置路径")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖 dual/model/train 配置")
    parser.add_argument("--dump-kwargs", action="store_true", help="只打印两个模型配置，不启动训练")
    return parser.parse_args()


def _resolve_section(section: dict[str, Any], root: Path) -> dict[str, Any]:
    """解析配置段中的相对路径。"""
    return expcfg.resolve_paths(root, dict(section))


def _model_and_train_kwargs(
    cfg: dict[str, Any],
    dual: dict[str, Any],
    modality: str,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """构造单个局部类别模型的 RF-DETR 参数。"""
    model_kwargs = _resolve_section(cfg.get("model", {}), PROJECT_ROOT)
    train_kwargs = expcfg.build_train_kwargs({"train": cfg.get("train", {})})
    variant = str(dual.get("variant", "medium"))
    if variant not in expcfg.MODEL_REGISTRY:
        raise ValueError(f"不支持的模型变体: {variant}")
    classes = PAN_CLASSES if modality == "pan" else RGB_CLASSES
    model_kwargs["num_classes"] = len(classes)
    model_kwargs["resolution"] = int(dual.get("resolution", model_kwargs.get("resolution", 1024)))
    if dual.get("pretrain_weights"):
        model_kwargs["pretrain_weights"] = dual["pretrain_weights"]
    train_kwargs.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset_file": "yolo",
            "output_dir": str(output_dir),
            "epochs": int(dual.get("epochs", 120)),
            "batch_size": int(dual.get("batch_size", 8)),
            "grad_accum_steps": int(dual.get("grad_accum_steps", 8)),
            "dataset_cache_mode": dual.get("dataset_cache_mode", "raw"),
            "dataset_cache_dir": dual.get("dataset_cache_dir"),
            "dataset_cache_rebuild": bool(dual.get("dataset_cache_rebuild", False)),
        }
    )
    class_ids_key = f"{modality}_class_balanced_class_ids"
    if class_ids_key in dual:
        train_kwargs["class_balanced_class_ids"] = [int(class_id) for class_id in dual[class_ids_key]]
    effective_batch = int(train_kwargs["batch_size"]) * int(train_kwargs["grad_accum_steps"])
    expected_batch = int(dual.get("effective_batch_size", effective_batch))
    if effective_batch != expected_batch:
        raise ValueError(
            f"{modality} 有效 batch 不匹配: batch_size({train_kwargs['batch_size']}) * "
            f"grad_accum_steps({train_kwargs['grad_accum_steps']}) = {effective_batch}，期望 {expected_batch}"
        )
    return model_kwargs, train_kwargs, variant


def main() -> None:
    """加载配置、准备缓存并顺序训练两个模型。"""
    args = _parse_args()
    cfg = expcfg.load_config(args.config)
    cfg = expcfg.apply_overrides(cfg, args.set)
    dual = _resolve_section(cfg.get("dual", {}), PROJECT_ROOT)
    dataset_dir = Path(dual.get("dataset_dir", "/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test")).resolve()
    cache_dir = Path(
        dual.get("cache_dir", dataset_dir / ".rfdetr_dual_cache" / "shwx_v1")
    ).resolve()
    output_dir = Path(dual.get("output_dir", "output/dual-shwx-medium-1024")).resolve()
    manifest = prepare_dual_dataset(dataset_dir, cache_dir, rebuild=bool(dual.get("cache_rebuild", False)))
    print(
        f"[i] 双模型缓存: {manifest.cache_dir} | PAN/RGB 路由阈值: {manifest.threshold:.6f} "
        f"| 校准准确率: {manifest.threshold_accuracy:.4f}"
    )

    plans: dict[str, dict[str, Any]] = {}
    for modality in ("pan", "rgb"):
        model_dir = manifest.cache_dir / modality
        model_kwargs, train_kwargs, variant = _model_and_train_kwargs(
            cfg,
            dual,
            modality,
            model_dir,
            output_dir / modality,
        )
        plans[modality] = {"model": model_kwargs, "train": train_kwargs, "variant": variant}
    if args.dump_kwargs:
        print(json.dumps(plans, ensure_ascii=False, indent=2, default=str))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for modality in ("pan", "rgb"):
        plan = plans[modality]
        model_cls, default_resolution = expcfg.MODEL_REGISTRY[plan["variant"]]
        plan["model"].setdefault("resolution", default_resolution)
        print(
            f"[i] 开始训练 {modality.upper()} 模型: variant={plan['variant']} "
            f"classes={plan['model']['num_classes']} epochs={plan['train']['epochs']} "
            f"effective_batch={plan['train']['batch_size'] * plan['train']['grad_accum_steps']}"
        )
        model = model_cls(**plan["model"])
        model.train(**plan["train"])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "dataset_dir": str(dataset_dir),
        "cache_dir": str(cache_dir),
        "cache_fingerprint": manifest.fingerprint,
        "route_threshold": manifest.threshold,
        "route_threshold_accuracy": manifest.threshold_accuracy,
        "pan_global_classes": list(PAN_CLASSES),
        "rgb_global_classes": list(RGB_CLASSES),
        "plans": plans,
    }
    (output_dir / "dual_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[完成] 双模型训练完成: {output_dir}")


if __name__ == "__main__":
    main()
