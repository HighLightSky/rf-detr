# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""统一训练模板：按 yaml 配置启动 RF-DETR 训练（替代全部旧 train_*.py 脚本）。

``train:`` 段 100% 透传为 ``model.train(**kwargs)``（零映射表，新字段自动可用）；
``model:`` 段为变体类构造参数（``variant`` 查 ``expcfg.MODEL_REGISTRY``）。
``TrainConfig`` 的 ``extra="forbid"`` 天然校验 yaml 键拼写错误。

用法：
    python src/scripts/train.py -c configs/experiments/train_sscl_class_balance_E1.yaml
    # --set 覆盖任意配置项（yaml < --set < 专用 CLI 参数）：
    python src/scripts/train.py -c ... --set train.class_balance_enabled=true
    # 只打印展开后的 kwargs 不训练（等价性验证用）：
    python src/scripts/train.py -c ... --dump-kwargs

配置结构（详见 configs/experiments/README.md 与 templates/train_template.yaml）：

.. code-block:: yaml

    _template: {class_counts: auto}   # auto=训练前统计类别数并自动注入
    model: {variant: medium, num_classes: 25, resolution: 640, ...}
    train: {dataset_dir: ..., output_dir: ..., epochs: 6, sscl_enabled: true, ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

# ── 项目路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import expcfg  # noqa: E402
from scripts.data_prep.stat_class_counts import write_counts_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（-c 必填，--set 可多次，--dump-kwargs 只打印）。"""
    parser = argparse.ArgumentParser(description="RF-DETR 统一训练模板（yaml 配置）")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="实验 yaml 配置文件路径（configs/experiments/*.yaml）",
    )
    parser.add_argument(
        "--set",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项（可多次），如 --set train.sscl_lambda=0.2",
    )
    parser.add_argument(
        "--dump-kwargs",
        action="store_true",
        help="只打印展开后的 model/train kwargs 不训练（等价性验证用）",
    )
    return parser.parse_args()


def main() -> None:
    """加载 yaml 配置，展开 kwargs 并启动训练。"""
    args = _parse_args()
    cfg = expcfg.load_config(args.config)
    cfg = expcfg.apply_overrides(cfg, args.set)
    model_kwargs, variant = expcfg.build_model_kwargs(cfg.get("model"))
    train_kwargs = expcfg.build_train_kwargs(cfg)

    model_cls, default_resolution = expcfg.MODEL_REGISTRY[variant]
    model_kwargs.setdefault("resolution", default_resolution)

    # [_template.class_counts=auto] 训练前自动统计类别实例数并写 class_counts.json，
    # 同时注入 class_balance_counts_path（等价旧 train_sscl_class_balance 的前置步骤）。
    if cfg.get("_template", {}).get("class_counts") == "auto":
        labels_dir = Path(train_kwargs["dataset_dir"]) / "labels" / "train"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"未找到训练集标签目录（YOLO 布局）: {labels_dir}")
        counts_json = str(Path(train_kwargs["output_dir"]) / "class_counts.json")
        payload = write_counts_json(labels_dir, counts_json, model_kwargs["num_classes"])
        print(f"类别实例数（自动统计）: {payload['counts']}")
        print(
            f"n_max={payload['n_max']:.0f}  n_min={payload['n_min']:.0f}  "
            f"n_ref={payload['n_ref']:.2f}  -> {counts_json}"
        )
        train_kwargs.setdefault("class_balance_counts_path", counts_json)

    # 打印本次实验配置摘要
    print(f"模型: {variant} | 类别数: {model_kwargs.get('num_classes')} | 分辨率: {model_kwargs.get('resolution')}")
    print(f"数据集: {train_kwargs.get('dataset_dir')} | 输出: {train_kwargs.get('output_dir')}")
    print(f"Epochs: {train_kwargs.get('epochs')} | Batch: {train_kwargs.get('batch_size')}")

    if args.dump_kwargs:
        print(json.dumps({"model": model_kwargs, "train": train_kwargs}, ensure_ascii=False, indent=2))
        return

    # TrainConfig extra="forbid"：yaml 键拼错会抛 ValidationError，这里打印出错字段
    try:
        model = model_cls(**model_kwargs)
        model.train(**train_kwargs)
    except ValidationError as exc:
        print("[错误] 训练配置校验失败（请检查 yaml 键名与取值）：")
        for err in exc.errors():
            print(f"  - 字段 {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
        raise

    print(f"\n训练完成！输出目录: {train_kwargs.get('output_dir')}")


if __name__ == "__main__":
    main()
