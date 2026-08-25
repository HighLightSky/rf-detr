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
import torchvision
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402
from rfdetr.refinement import FSCEnsembleHead, FSCFeatureGeometryHead, FSCMultiViewHead, FSCScoreFusion, FSCDinoHead, FSCVerifier, FSCVerifierPolicy, crop_fsc_context, crop_transform, iou_xyxy, pool_dino_features  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析端到端推理参数。"""
    parser = argparse.ArgumentParser(description="运行 FSC 两阶段检测")
    parser.add_argument("--detector", required=True, help="一级 RF-DETR checkpoint")
    parser.add_argument("--verifier", required=True, help="二级 FSC/非FSC checkpoint")
    parser.add_argument("--fusion", default=None, help="可选的学习型视觉/一级分数融合 checkpoint")
    parser.add_argument("--dino-head", default=None, help="可选的冻结 RF-DETR DINOv2 分类头 checkpoint")
    parser.add_argument("--multiview-head", default=None, help="可选的 DINOv2 紧框/上下文双视图头 checkpoint")
    parser.add_argument("--geometry-head", default=None, help="可选的 DINOv2 几何辅助头 checkpoint")
    parser.add_argument("--ensemble-head", default=None, help="可选的单视图/旋转头 stacking checkpoint")
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


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """对同一类别候选执行置信度优先 NMS。"""
    order = np.argsort(-scores)
    kept: list[int] = []
    for index in order.tolist():
        if all(iou_xyxy(tuple(boxes[index]), tuple(boxes[chosen])) <= iou_threshold for chosen in kept):
            kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def main() -> None:
    """对指定图像运行一级候选生成和二级视觉拒识。"""
    args = _parse_args()
    verifier = FSCVerifier.from_checkpoint(args.verifier, device="cuda:0" if torch.cuda.is_available() else "cpu")
    fusion = FSCScoreFusion.from_checkpoint(args.fusion, device=next(verifier.parameters()).device) if args.fusion else None
    detector = RFDETR.from_checkpoint(args.detector)
    dino_head = FSCDinoHead.from_checkpoint(args.dino_head, device=next(verifier.parameters()).device) if args.dino_head else None
    multiview_head = FSCMultiViewHead.from_checkpoint(args.multiview_head, device=next(verifier.parameters()).device) if args.multiview_head else None
    geometry_head = FSCFeatureGeometryHead.from_checkpoint(args.geometry_head, device=next(verifier.parameters()).device) if args.geometry_head else None
    ensemble_head = FSCEnsembleHead.from_checkpoint(args.ensemble_head, device=next(verifier.parameters()).device) if args.ensemble_head else None
    active_head = ensemble_head or geometry_head or multiview_head or dino_head
    dino_policy = FSCVerifierPolicy.from_mapping(active_head.checkpoint_metadata.get("policy")) if active_head else verifier.policy
    dino_encoder = None
    dino_external = False
    ensemble_single = ensemble_rotation = None
    if active_head is not None:
        metadata = active_head.checkpoint_metadata
        if metadata.get("repo") and metadata.get("backbone_checkpoint"):
            sys.path.insert(0, str(Path(metadata["repo"]).resolve()))
            from dinov2.hub.backbones import dinov2_vitl14_reg

            dino_encoder = dinov2_vitl14_reg(pretrained=False)
            dino_encoder.load_state_dict(torch.load(metadata["backbone_checkpoint"], map_location="cpu", weights_only=True))
            if ensemble_head is not None:
                single_payload = torch.load(metadata["single_head"], map_location="cpu", weights_only=False)
                rotation_payload = torch.load(metadata["rotation_head"], map_location="cpu", weights_only=False)
                ensemble_single = FSCDinoHead(int(single_payload["feature_dim"])).to(next(verifier.parameters()).device)
                ensemble_rotation = FSCDinoHead(int(rotation_payload["feature_dim"])).to(next(verifier.parameters()).device)
                ensemble_single.load_state_dict(single_payload["state_dict"]); ensemble_rotation.load_state_dict(rotation_payload["state_dict"])
                ensemble_single.eval(); ensemble_rotation.eval()
            backbone_patch = metadata.get("backbone_patch")
            if backbone_patch:
                patch = torch.load(backbone_patch, map_location="cpu", weights_only=True)
                if patch.get("format") != "shwx-fsc-dino-backbone-patch-v1":
                    raise ValueError("不是 FSC DINO 尾部微调权重")
                dino_encoder.load_state_dict(patch["state_dict"], strict=False)
            fine_tuned_backbone = metadata.get("fine_tuned_backbone_checkpoint")
            if fine_tuned_backbone:
                dino_encoder.load_state_dict(torch.load(fine_tuned_backbone, map_location="cpu", weights_only=True))
            dino_external = True
        else:
            dino_encoder = detector.model.model.backbone[0].encoder
        dino_encoder = dino_encoder.to(next(verifier.parameters()).device).eval()
    dino_transform = crop_transform(training=False) if active_head else None
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
        raw_keep = _nms_indices(raw_boxes, raw_scores)
        fsc_indices = np.flatnonzero(fsc_mask)[raw_keep]
        raw_boxes, raw_scores = boxes[fsc_indices], scores[fsc_indices]
        if active_head is not None and dino_encoder is not None and dino_transform is not None:
            context_scales = tuple(float(value) for value in active_head.checkpoint_metadata.get("context_scales", [dino_policy.context_scale]))
            if multiview_head is not None or ensemble_head is not None:
                context_scales = (dino_policy.context_scale,)
            crops = [
                dino_transform(crop_fsc_context(image, boxes[index], scale, dino_policy.image_size))
                for index in fsc_indices
                for scale in context_scales
            ]
            with torch.inference_mode():
                crop_batch = torch.stack(crops).to(next(verifier.parameters()).device)
                if dino_external:
                    def _features(batch: torch.Tensor) -> torch.Tensor:
                        output = dino_encoder.forward_features(batch)
                        return torch.cat((output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)), dim=1)

                    tta_rotations = tuple(float(value) for value in active_head.checkpoint_metadata.get("tta_rotations", []))
                    if tta_rotations:
                        pooled = torch.stack([_features(torchvision.transforms.functional.rotate(crop_batch, angle)) for angle in tta_rotations]).mean(dim=0)
                    else:
                        pooled = _features(crop_batch)
                    if ensemble_head is not None:
                        rotations = torch.stack([_features(torchvision.transforms.functional.rotate(crop_batch, angle)) for angle in (0, 90, 180, 270)]).mean(0)
                        ensemble_features = torch.stack((ensemble_single(pooled).softmax(1)[:, 1], ensemble_rotation(rotations).softmax(1)[:, 1], torch.from_numpy(scores[fsc_indices]).to(crop_batch.device)), dim=1)
                        decisions = ensemble_head(ensemble_features).argmax(dim=1).cpu().numpy()
                    elif geometry_head is not None:
                        geometry = torch.tensor([
                            [
                                (float(boxes[index][0]) + float(boxes[index][2])) / (2 * image.width),
                                (float(boxes[index][1]) + float(boxes[index][3])) / (2 * image.height),
                                (float(boxes[index][2]) - float(boxes[index][0])) / image.width,
                                (float(boxes[index][3]) - float(boxes[index][1])) / image.height,
                                float(scores[index]),
                            ]
                            for index in fsc_indices
                        ], device=crop_batch.device)
                        decisions = geometry_head(pooled, geometry).argmax(dim=1).cpu().numpy()
                    elif multiview_head is not None:
                        tight_scale = float(multiview_head.checkpoint_metadata.get("tight_scale", 0.575))
                        tight_batch = torch.stack([
                            dino_transform(crop_fsc_context(image, boxes[index], dino_policy.context_scale * tight_scale, dino_policy.image_size))
                            for index in fsc_indices
                        ]).to(next(verifier.parameters()).device)
                        decisions = multiview_head(_features(tight_batch), pooled).argmax(dim=1).cpu().numpy()
                    else:
                        dino_features = pooled.reshape(len(fsc_indices), len(context_scales) * pooled.shape[1])
                else:
                    pooling = str(active_head.checkpoint_metadata.get("pooling", "avg"))
                    dino_features = pool_dino_features(dino_encoder(crop_batch), pooling)
                if multiview_head is None and geometry_head is None and ensemble_head is None:
                    decisions = dino_head(dino_features).argmax(dim=1).cpu().numpy()
            keep = np.ones(len(class_ids), dtype=bool)
            keep[np.flatnonzero(fsc_mask)] = False
            keep[fsc_indices] = decisions == 1
            out_boxes, out_scores, out_classes = boxes[keep], scores[keep], class_ids[keep]
            audit = {"routed_fsc": int(len(fsc_indices)), "kept": int(keep.sum()), "rejected_non_fsc": int((decisions == 0).sum()), "unchanged": int(keep.sum() - (decisions == 1).sum()), "nms_suppressed": int(fsc_mask.sum() - len(fsc_indices))}
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
