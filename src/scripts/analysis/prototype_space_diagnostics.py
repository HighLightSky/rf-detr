# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""比较 ProtoGuidance 与 SSCL 原型来源及类别关系。

示例：
    uv run python src/scripts/analysis/prototype_space_diagnostics.py \
        output/0814-SHWX-ProtoGuidance-E4/last.ckpt \
        --artifacts data/proto_guidance_shwx.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rfdetr.sscl.proto_guidance.guidance import ProtoGuidance
from rfdetr.sscl.proto_guidance.artifacts import load_proto_artifacts
from rfdetr.sscl.prototype_diagnostics import prototype_geometry, prototype_relation_alignment


def _checkpoint_state(path: Path) -> dict[str, Tensor]:
    """读取 Lightning checkpoint 的 state_dict。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 缺少 state_dict: {path}")
    return state


def _proto_guidance_report(
    state: dict[str, Tensor],
    artifacts_path: Path,
) -> tuple[dict[str, Any], Tensor]:
    """加载 ProtoGuidance 并返回融合原型及其报告。"""
    artifacts = load_proto_artifacts(artifacts_path)
    visual = artifacts["visual_prototypes"]
    text = artifacts["text_prototypes"]
    prefix = "model.transformer.proto_guidance."
    subset = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not subset:
        raise ValueError("checkpoint 中没有 model.transformer.proto_guidance.* 状态。")
    module = ProtoGuidance.build(
        num_classes=int(visual.shape[0]),
        hidden_dim=int(visual.shape[-1]),
        text_dim=int(text.shape[-1]),
        num_slots=int(visual.shape[1]),
        artifacts_path=artifacts_path,
        tau_p=0.1,
    )
    if module is None:
        raise ValueError("ProtoGuidance 离线产物不可加载。")
    module.load_state_dict(subset, strict=False)
    fused, valid = module.fused_prototypes()
    geometry = prototype_geometry(fused.detach(), valid)
    report = {
        "source": "P4 区域池化视觉原型 + CLIP 文本原型，经可学习投影融合",
        "visual_shape": list(visual.shape),
        "text_shape": list(text.shape),
        "geometry": {key: float(value) for key, value in geometry.items()},
    }
    return report, fused.detach()


def _sscl_report(state: dict[str, Tensor]) -> tuple[dict[str, Any], Tensor | None]:
    """从 checkpoint 判断 SSCL 是否真的保存了视觉原型库。"""
    prefix = "sscl_loss.prototype_bank."
    prototypes = state.get(prefix + "prototypes")
    if prototypes is None:
        return {
            "enabled": False,
            "source": (
                "当前 checkpoint 没有 SSCL prototype_bank；SSCL 使用 matched query "
                "instance-to-instance 模式"
            ),
        }, None
    valid = state.get(prefix + "slot_valid_mask")
    if valid is None:
        updates = state.get(prefix + "slot_num_updates")
        valid = updates > 0 if updates is not None else torch.ones(prototypes.shape[:2], dtype=torch.bool)
    geometry = prototype_geometry(prototypes, valid)
    return {
        "enabled": True,
        "source": "decoder 最后一层 Hungarian matched query，经 SSCL 投影头后的 EMA 原型",
        "shape": list(prototypes.shape),
        "geometry": {key: float(value) for key, value in geometry.items()},
    }, prototypes.detach()


def build_report(checkpoint: Path, artifacts: Path) -> dict[str, Any]:
    """构建原型来源、几何和关系一致性报告。"""
    state = _checkpoint_state(checkpoint)
    proto_report, proto = _proto_guidance_report(state, artifacts)
    sscl_report, sscl = _sscl_report(state)
    report: dict[str, Any] = {"proto_guidance": proto_report, "sscl": sscl_report}
    if sscl is not None:
        report["relation_alignment"] = float(prototype_relation_alignment(proto, sscl))
    else:
        report["relation_alignment"] = None
    return report


def main() -> None:
    """解析参数并打印 JSON 诊断结果。"""
    parser = argparse.ArgumentParser(description="诊断 ProtoGuidance 与 SSCL 原型空间")
    parser.add_argument("checkpoint", type=Path, help="Lightning last.ckpt 路径")
    parser.add_argument("--artifacts", type=Path, required=True, help="ProtoGuidance 离线产物路径")
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径")
    args = parser.parse_args()
    report = build_report(args.checkpoint, args.artifacts)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
