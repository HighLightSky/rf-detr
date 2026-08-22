# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Stage-2 语义分类头消融训练入口。

在 0805 Stage-1 checkpoint 上冻结"适配后"骨干，解冻 decoder 末层 + 分类头 + 语义组件（θ/α）+ SSCL，施加语义残差增量 + SSCL + 基类蒸馏。

用法：     python src/scripts/semantic_experiments/stage2_train.py [experiment]

``experiment`` 为 ABLATIONS 中的实验名（默认 e1a），也可用环境变量 ``SEM_HEAD_EXPERIMENT`` 指定。每个实验输出到独立目录 ``output/0810-SHWX-
SemHead-<suffix>``，保证实验结果可归因。

前置条件：已运行 stage0_collect_features.py 与 stage0_train_fsem.py， 产出 data/fsem_shwx.pt 与 data/channel_stats_shwx.pt。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ablation_configs as cfg

from rfdetr.variants import RFDETRMedium

# ============================================================================
# 配置 —— 在这里修改
# ============================================================================

# 输出目录前缀（日期 + 实验名，实验后缀自动拼接）
OUTPUT_PREFIX = "output/0808-SHWX-SemHead/0808-SHWX-SemHead-"


def _resolve_experiment() -> str:
    """从命令行参数或环境变量解析实验名（默认 e1a）。"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("SEM_HEAD_EXPERIMENT", cfg.DEFAULT_EXPERIMENT)


def main() -> None:
    """按实验名构造训练配置并启动 Stage-2 训练。"""
    experiment = _resolve_experiment()
    if experiment not in cfg.ABLATIONS:
        print(f"未知实验名: {experiment}。可选: {sorted(cfg.ABLATIONS)}")
        sys.exit(1)

    spec = cfg.ABLATIONS[experiment]
    overrides = spec["overrides"]
    recipe = {**cfg.BASE_RECIPE, **overrides}

    output_dir = str(cfg.PROJECT_ROOT / f"{OUTPUT_PREFIX}{spec['output_suffix']}")
    recipe["output_dir"] = output_dir

    # 校验离线产物存在（启用语义头时）
    if recipe.get("semantic_head_enabled"):
        for p in (cfg.FSEM_PATH, cfg.CHANNEL_STATS_PATH):
            if not Path(p).exists():
                print(f"缺少语义头离线产物: {p}。请先运行 stage0_collect_features.py 与 stage0_train_fsem.py。")
                sys.exit(1)

    print("=" * 72)
    print(f"实验: {experiment} — {spec['desc']}")
    print(f"输出目录: {output_dir}")
    print("关键 overrides:")
    for k, v in overrides.items():
        print(f"  {k} = {v}")
    print("=" * 72)

    # 构建模型（加载 0805 checkpoint 作为微调起点）
    model = RFDETRMedium(
        num_classes=25,
        resolution=640,
        gradient_checkpointing=True,
        pretrain_weights=str(cfg.BASE_CHECKPOINT),
    )
    # 训练：BASE_RECIPE + overrides + output_dir 全部作为 kwargs 传入
    model.train(**recipe)
    print(f"\n训练完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
