# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""运行冻结 RF-DETR 与 FSC 二级复核器的端到端推理。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import FSCScoreFusion, FSCDinoHead, FSCVerifier, FSCVerifierPolicy, crop_fsc_context, crop_transform, pool_dino_features  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析端到端推理参数。"""
    parser = argparse.ArgumentParser(description="运行 FSC 两阶段检测")
    parser.add_argument("--detector", required=True, help="一级 RF-DETR checkpoint")
    parser.add_argument("--verifier", required=True, help="二级 FSC/非FSC checkpoint")
    parser.add_argument("--fusion", default=None, help="可选的学习型视觉/一级分数融合 checkpoint")
    parser.add_argument("--dino-head", default=None, help="可选的冻结 RF-DETR DINOv2 分类头 checkpoint")
    parser.add_argument("--image", required=True, help="单张图像或图像目录")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _images(path: Path) -> list[Path]:
    """获取单图或目录中的常见格式图像。"""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
        if images:
            return images
    raise FileNotFoundError(f"未找到可推理图像: {path}")


def _draw(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, path: Path, color: tuple[int, int, int]) -> None:
    """保存 FSC 候选框可视化。"""
    canvas = image.copy()
    for box, score in zip(boxes, scores, strict=True):
        x0, y0, x1, y1 = (int(value) for value in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"FSC {float(score):.3f}", (x0, max(y0 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    """对指定图像运行一级候选生成和二级视觉拒识。"""
    args = _parse_args()
    verifier = FSCVerifier.from_checkpoint(args.verifier, device="cuda:0" if torch.cuda.is_available() else "cpu")
    fusion = FSCScoreFusion.from_checkpoint(args.fusion, device=next(verifier.parameters()).device) if args.fusion else None
    detector = RFDETR.from_checkpoint(args.detector)
    dino_head = FSCDinoHead.from_checkpoint(args.dino_head, device=next(verifier.parameters()).device) if args.dino_head else None
    dino_policy = FSCVerifierPolicy.from_mapping(dino_head.checkpoint_metadata.get("policy")) if dino_head else verifier.policy
    dino_encoder = None
    dino_external = False
    if dino_head is not None:
        metadata = dino_head.checkpoint_metadata
        if metadata.get("repo") and metadata.get("backbone_checkpoint"):
            sys.path.insert(0, str(Path(metadata["repo"]).resolve()))
            from dinov2.hub.backbones import dinov2_vitl14_reg

            dino_encoder = dinov2_vitl14_reg(pretrained=False)
            dino_encoder.load_state_dict(torch.load(metadata["backbone_checkpoint"], map_location="cpu", weights_only=True))
            dino_external = True
        else:
            dino_encoder = detector.model.model.backbone[0].encoder
        dino_encoder = dino_encoder.to(next(verifier.parameters()).device).eval()
    dino_transform = crop_transform(training=False) if dino_head else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for path in _images(Path(args.image).resolve()):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image_rgb = np.asarray(image)
        detections = detector.predict(image_rgb, threshold=verifier.policy.candidate_floor, include_source_image=False)
        boxes = np.asarray(detections.xyxy, dtype=np.float32)
        scores = np.asarray(detections.confidence, dtype=np.float32)
        class_ids = np.asarray(detections.class_id, dtype=np.int64)
        fsc_mask = class_ids == verifier.policy.fsc_class_id
        raw_boxes, raw_scores = boxes[fsc_mask], scores[fsc_mask]
        if dino_head is not None and dino_encoder is not None and dino_transform is not None:
            fsc_indices = np.flatnonzero(class_ids == verifier.policy.fsc_class_id)
            crops = [
                dino_transform(crop_fsc_context(image, boxes[index], dino_policy.context_scale, dino_policy.image_size))
                for index in fsc_indices
            ]
            with torch.inference_mode():
                crop_batch = torch.stack(crops).to(next(verifier.parameters()).device)
                if dino_external:
                    output = dino_encoder.forward_features(crop_batch)
                    dino_features = torch.cat((output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)), dim=1)
                else:
                    pooling = str(dino_head.checkpoint_metadata.get("pooling", "avg"))
                    dino_features = pool_dino_features(dino_encoder(crop_batch), pooling)
                decisions = dino_head(dino_features).argmax(dim=1).cpu().numpy()
            keep = np.ones(len(class_ids), dtype=bool)
            keep[fsc_indices] = decisions == 1
            out_boxes, out_scores, out_classes = boxes[keep], scores[keep], class_ids[keep]
            audit = {"routed_fsc": int(len(fsc_indices)), "kept": int(keep.sum()), "rejected_non_fsc": int((decisions == 0).sum()), "unchanged": int(keep.sum() - (decisions == 1).sum())}
        elif fusion is None:
            out_boxes, out_scores, out_classes, audit = verifier.refine_image(image, boxes, scores, class_ids)
        else:
            fsc_indices = np.flatnonzero(class_ids == verifier.policy.fsc_class_id)
            probabilities = verifier.predict_probabilities(image, [boxes[index] for index in fsc_indices])
            decisions = fusion.predict(probabilities, torch.from_numpy(scores[fsc_indices]).to(probabilities.device)).cpu().numpy()
            keep = np.ones(len(class_ids), dtype=bool)
            keep[fsc_indices] = decisions == 1
            out_boxes, out_scores, out_classes = boxes[keep], scores[keep], class_ids[keep]
            audit = {
                "routed_fsc": int(len(fsc_indices)),
                "kept": int(keep.sum()),
                "rejected_non_fsc": int((decisions == 0).sum()),
                "unchanged": int(keep.sum() - (decisions == 1).sum()),
            }
        final_mask = out_classes == verifier.policy.fsc_class_id
        final_boxes, final_scores = out_boxes[final_mask], out_scores[final_mask]
        _draw(np.asarray(image)[:, :, ::-1], raw_boxes, raw_scores, output_dir / f"{path.stem}_stage1.jpg", (0, 0, 255))
        _draw(np.asarray(image)[:, :, ::-1], final_boxes, final_scores, output_dir / f"{path.stem}_stage2.jpg", (0, 255, 0))
        results.append(
            {
                "image": str(path),
                "candidate_floor": verifier.policy.candidate_floor,
                "stage1_fsc": int(raw_boxes.shape[0]),
                "stage2_fsc": int(final_boxes.shape[0]),
                "audit": audit,
                "stage2_boxes": final_boxes.tolist(),
                "stage2_scores": [float(score) for score in final_scores],
            }
        )
    (output_dir / "prediction_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
