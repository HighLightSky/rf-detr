"""批量生成两份模型预测与 YOLO GT 的逐图对比结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402


CLASS_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "A1_SU-35", 5: "A2_C-130",
    6: "A3_C-17", 7: "A4_C-5", 8: "A5_F-16", 9: "A6_TU-160", 10: "A7_E-3",
    11: "A8_B-52", 12: "A9_P-3C", 13: "A10_B-1B", 14: "A11_E-8", 15: "A12_TU-22",
    16: "A13_F-15", 17: "A14_KC-135", 18: "A15_F-22", 19: "A16_FA-18",
    20: "A17_TU-95", 21: "A18_KC-10", 22: "A19_SU-34", 23: "A20_SU-24",
    24: "FSC", 25: "truck",
}


def _color(class_id: int) -> tuple[int, int, int]:
    """根据类别生成稳定的 BGR 颜色。"""
    palette = [(0, 140, 255), (0, 220, 0), (255, 80, 0), (0, 220, 220),
               (220, 0, 220), (220, 220, 0), (80, 80, 255), (160, 0, 220)]
    return palette[class_id % len(palette)]


def _draw(image: np.ndarray, boxes: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    """绘制框、类别名称和置信度。"""
    height, width = image.shape[:2]
    thickness = max(1, round(min(width, height) / 650))
    box_thickness = 3
    font_scale = max(0.35, min(0.65, min(width, height) / 1100))
    for x1, y1, x2, y2, class_id, score in boxes:
        color = _color(class_id)
        p1 = (max(0, int(round(x1))), max(0, int(round(y1))))
        p2 = (min(width - 1, int(round(x2))), min(height - 1, int(round(y2))))
        cv2.rectangle(image, p1, p2, color, box_thickness)
        label = f"{CLASS_NAMES.get(class_id, str(class_id))} {score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ly = max(0, p1[1] - th - base)
        cv2.rectangle(image, (p1[0], ly), (min(width - 1, p1[0] + tw), min(height - 1, ly + th + base)), color, -1)
        cv2.putText(image, label, (p1[0], min(height - 1, ly + th)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _gt_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float, int, float]]:
    """读取归一化 YOLO 标签并转换为像素框。"""
    boxes: list[tuple[float, float, float, float, int, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id, xc, yc, bw, bh = int(float(fields[0])), *(float(v) for v in fields[1:5])
        boxes.append(((xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height, class_id, 1.0))
    return boxes


def _predict(model: RFDETR, image: np.ndarray, threshold: float) -> list[tuple[float, float, float, float, int, float]]:
    """对 RGB 图片执行模型推理并返回可绘制框。"""
    detections = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=threshold, include_source_image=False)
    if detections is None or len(detections) == 0:
        return []
    return [(float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(cls), float(score))
            for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id)]


def main() -> None:
    """加载两个 checkpoint 并生成逐图三联结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.preferred_blas_library("cublas")
    print(f"加载 baseline: {args.baseline}", flush=True)
    baseline = RFDETR.from_checkpoint(str(args.baseline))
    print(f"加载 ours: {args.ours}", flush=True)
    ours = RFDETR.from_checkpoint(str(args.ours))
    # 当前机器的 CUDA/cuBLAS 不可用时显式回退 CPU。
    if not torch.cuda.is_available():
        baseline.model.device = torch.device("cpu")
        ours.model.device = torch.device("cpu")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total = 0
    for split in ("train", "val", "test"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
        for index, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            base_boxes = _predict(baseline, image, args.threshold)
            ours_boxes = _predict(ours, image, args.threshold)
            gt_boxes = _gt_boxes(label_dir / f"{image_path.stem}.txt", width, height)
            target = args.output / split / image_path.stem
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / "01_baseline.jpg"), _draw(image.copy(), base_boxes))
            cv2.imwrite(str(target / "02_ours.jpg"), _draw(image.copy(), ours_boxes))
            cv2.imwrite(str(target / "03_gt.jpg"), _draw(image.copy(), gt_boxes))
            total += 1
            if index == 1 or index % 100 == 0 or index == len(images):
                print(f"{split}: {index}/{len(images)}", flush=True)
    print(f"完成，共生成 {total} 张图片的对比目录: {args.output}", flush=True)


if __name__ == "__main__":
    main()
