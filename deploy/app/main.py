"""比赛 Docker 的命令行入口。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from competition.config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR, load_submission_config
from competition.pipeline import CompetitionPipeline
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_args() -> argparse.Namespace:
    """解析赛事规定的输入输出参数。"""
    parser = argparse.ArgumentParser(description="比赛目标检测 Docker 推理入口")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_images(input_dir: Path) -> list[tuple[Path, Image.Image]]:
    """读取输入目录第一层的全部赛事支持图像。"""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    images: list[tuple[Path, Image.Image]] = []
    for path in paths:
        with Image.open(path) as image:
            images.append((path, image.convert("RGB").copy()))
    return images


def _image_result(path: Path, image: Image.Image, objects: list[dict[str, Any]]) -> dict[str, Any]:
    """在一张图推理完成后构造赛事规定的结果记录。"""
    return {
        "image_id": path.stem,
        "file_name": path.name,
        "width": image.width,
        "height": image.height,
        "run_end_timestamp": time.time_ns() // 1_000_000,
        "objects": objects,
    }


def main() -> None:
    """加载配置和模型，执行推理并写出 result.json。"""
    args = _parse_args()
    config = load_submission_config(DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR)
    pipeline = CompetitionPipeline.from_config(config)
    pipeline.check_gpu()
    images = _read_images(Path(args.input))
    results: list[dict[str, Any]] = []
    for path, image in images:
        objects = pipeline.predict(path.stem, image)
        results.append(_image_result(path, image, objects))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps({"status": "success", "images": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
