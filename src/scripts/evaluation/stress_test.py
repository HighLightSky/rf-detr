"""RF-DETR 高负荷稳定性实验。

该脚本覆盖连续推理、峰值批量和受限显存三类实验，并输出逐次 CSV、汇总 CSV、PNG 图表及 Markdown 报告。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import RFDETR  # noqa: E402


def _rss_mb() -> float:
    """读取当前进程的常驻内存，无法读取时返回 NaN。"""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    except ImportError:
        return float("nan")


def _gpu_stats() -> dict[str, float]:
    """读取当前 CUDA 设备的内存统计。"""
    if not torch.cuda.is_available():
        return {"gpu_allocated_mb": float("nan"), "gpu_reserved_mb": float("nan"), "gpu_peak_mb": float("nan")}
    return {
        "gpu_allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "gpu_reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }


def _load_images(image_dir: Path | None, count: int, size: int) -> list[np.ndarray]:
    """加载固定数量 RGB 图像，不足时使用确定性的合成图补齐。"""
    images: list[np.ndarray] = []
    if image_dir is not None:
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for path in paths[:count]:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
    if not images:
        rng = np.random.default_rng(20260901)
        for index in range(count):
            image = np.zeros((size, size, 3), dtype=np.uint8)
            image[:] = (index * 37 % 255, index * 67 % 255, index * 97 % 255)
            image[:: max(size // 16, 1), :, :] = rng.integers(
                0, 255, (len(image[:: max(size // 16, 1)]), 1, 3), dtype=np.uint8
            )
            images.append(image)
    while len(images) < count:
        images.append(images[len(images) % len(images)].copy())
    return images[:count]


def _prediction_digest(prediction: Any) -> str:
    """将 Supervision 预测规范化为稳定摘要哈希。"""
    fields: list[bytes] = []
    for name in ("xyxy", "confidence", "class_id"):
        value = getattr(prediction, name, None)
        if value is None:
            fields.append(b"none")
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = np.round(array.astype(np.float64), decimals=6)
        else:
            array = array.astype(np.int64, copy=False)
        fields.append(str(array.shape).encode())
        fields.append(array.tobytes())
    return hashlib.sha256(b"|".join(fields)).hexdigest()[:16]


def _prediction_count(prediction: Any) -> int:
    """返回预测框数量。"""
    value = getattr(prediction, "xyxy", None)
    return int(len(value)) if value is not None else 0


def _infer(model: RFDETR, batch: list[np.ndarray], threshold: float) -> tuple[list[Any], float]:
    """执行一次批量推理并返回预测及耗时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.predict(batch, threshold=threshold, include_source_image=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    predictions = result if isinstance(result, list) else [result]
    return predictions, elapsed


def _record(
    experiment: str,
    iteration: int,
    batch_size: int,
    elapsed: float,
    predictions: list[Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一行实验记录。"""
    row: dict[str, Any] = {
        "experiment": experiment,
        "iteration": iteration,
        "batch_size": batch_size,
        "elapsed_s": elapsed,
        "images": len(predictions),
        "detections": sum(_prediction_count(prediction) for prediction in predictions),
        "digests": ",".join(_prediction_digest(prediction) for prediction in predictions),
        "rss_mb": _rss_mb(),
    }
    row.update(_gpu_stats())
    if extra:
        row.update(extra)
    return row


def _long_run(model: RFDETR, images: list[np.ndarray], args: argparse.Namespace) -> list[dict[str, Any]]:
    """执行连续推理实验。"""
    rows: list[dict[str, Any]] = []
    batch = images[: args.long_batch]
    for iteration in range(args.warmup + args.long_iterations):
        if iteration == args.warmup and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        predictions, elapsed = _infer(model, batch, args.threshold)
        if iteration >= args.warmup:
            rows.append(_record("long_run", iteration - args.warmup, len(batch), elapsed, predictions))
    return rows


def _peak_batch(model: RFDETR, images: list[np.ndarray], args: argparse.Namespace) -> list[dict[str, Any]]:
    """逐级测试批量大小，记录成功或 CUDA OOM。"""
    rows: list[dict[str, Any]] = []
    baseline_predictions, _ = _infer(model, [images[0]], args.threshold)
    baseline_digest = _prediction_digest(baseline_predictions[0])
    for batch_size in args.batch_sizes:
        batch = [images[index % len(images)] for index in range(batch_size)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            predictions, elapsed = _infer(model, batch, args.threshold)
            consistent = _prediction_digest(predictions[0]) == baseline_digest
            rows.append(
                _record(
                    "peak_batch",
                    batch_size,
                    batch_size,
                    elapsed,
                    predictions,
                    {"status": "ok", "consistent": consistent},
                )
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(
                _record("peak_batch", batch_size, batch_size, float("nan"), [], {"status": "oom", "consistent": ""})
            )
    return rows


def _constrained_run(model: RFDETR, images: list[np.ndarray], args: argparse.Namespace) -> list[dict[str, Any]]:
    """在不同 CUDA allocator 上限下用自适应批量处理全部图像。"""
    rows: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return [
            _record(
                "constrained",
                0,
                1,
                float("nan"),
                [],
                {"memory_fraction": "cpu", "status": "skipped", "processed": 0, "total": len(images)},
            )
        ]
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for fraction in args.memory_fractions:
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
        except RuntimeError:
            rows.append(
                _record(
                    "constrained",
                    0,
                    0,
                    float("nan"),
                    [],
                    {"memory_fraction": fraction, "status": "cap_error", "processed": 0, "total": len(images)},
                )
            )
            continue
        processed = 0
        batch_size = args.constrained_batch
        iteration = 0
        while processed < len(images):
            current = min(batch_size, len(images) - processed)
            batch = images[processed : processed + current]
            try:
                predictions, elapsed = _infer(model, batch, args.threshold)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                torch.cuda.empty_cache()
                if batch_size == 1:
                    rows.append(
                        _record(
                            "constrained",
                            iteration,
                            1,
                            float("nan"),
                            [],
                            {
                                "memory_fraction": fraction,
                                "status": "oom_batch1",
                                "processed": processed,
                                "total": len(images),
                                "cap_mb": fraction * total_memory / 1024**2,
                            },
                        )
                    )
                    break
                batch_size = max(1, batch_size // 2)
                continue
            rows.append(
                _record(
                    "constrained",
                    iteration,
                    current,
                    elapsed,
                    predictions,
                    {
                        "memory_fraction": fraction,
                        "status": "ok",
                        "processed": processed + current,
                        "total": len(images),
                        "cap_mb": fraction * total_memory / 1024**2,
                    },
                )
            )
            processed += current
            iteration += 1
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写出字典行 CSV。"""
    materialized = list(rows)
    if not materialized:
        return
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """根据逐次记录生成横向双联稳定性图。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_family = "sans-serif"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    for axis in axes:
        axis.set_facecolor("#F8FAFC")
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    long_rows = [row for row in rows if row["experiment"] == "long_run"]
    if long_rows:
        x = [int(row["iteration"]) for row in long_rows]
        memory_axis = axes[0]
        memory_axis.plot(
            x,
            [float(row["gpu_allocated_mb"]) for row in long_rows],
            color="#0F766E",
            linewidth=2.2,
            label="已分配显存",
        )
        memory_axis.plot(
            x,
            [float(row["gpu_reserved_mb"]) for row in long_rows],
            color="#F97316",
            linewidth=2.2,
            label="保留显存",
        )
        memory_axis.set_xlabel("推理次数")
        memory_axis.set_ylabel("GPU 显存（MB）")
        memory_axis.set_ylim(100, 400)
        handles, labels = memory_axis.get_legend_handles_labels()
        memory_axis.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(0.98, 0.60),
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#E2E8F0",
            fontsize=8.5,
        )
    peak_rows = [row for row in rows if row["experiment"] == "peak_batch"]
    if peak_rows:
        axis = axes[1]
        ok = [row for row in peak_rows if row["status"] == "ok"]
        oom = [row for row in peak_rows if row["status"] == "oom"]
        axis.plot(
            [int(row["batch_size"]) for row in ok],
            [float(row["gpu_peak_mb"]) for row in ok],
            color="#2563EB",
            marker="o",
            markersize=6,
            linewidth=2.2,
        )
        if oom:
            axis.scatter(
                [int(row["batch_size"]) for row in oom],
                [0] * len(oom),
                color="#DC2626",
                marker="x",
                s=60,
                label="显存不足",
            )
        axis.set_xlabel("批量大小")
        axis.set_ylabel("峰值已分配显存（MB）")
        axis.set_xticks([int(row["batch_size"]) for row in peak_rows])
    fig.savefig(output_dir / "stability_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """计算实验摘要指标。"""
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "device": str(args.device),
        "image_count": args.image_count,
    }
    long_rows = [row for row in rows if row["experiment"] == "long_run"]
    if len(long_rows) >= 2:
        x = np.asarray([float(row["iteration"]) for row in long_rows])
        gpu = np.asarray([float(row["gpu_allocated_mb"]) for row in long_rows])
        rss = np.asarray([float(row["rss_mb"]) for row in long_rows])
        result["long_gpu_slope_mb_per_iter"] = float(np.polyfit(x, gpu, 1)[0])
        result["long_rss_slope_mb_per_iter"] = float(np.polyfit(x, rss, 1)[0])
        result["long_gpu_peak_mb"] = float(max(float(row["gpu_peak_mb"]) for row in long_rows))
        result["long_digest_unique"] = len({row["digests"] for row in long_rows})
    peak_rows = [row for row in rows if row["experiment"] == "peak_batch"]
    result["peak_max_success_batch"] = max(
        (int(row["batch_size"]) for row in peak_rows if row["status"] == "ok"), default=0
    )
    result["peak_first_oom_batch"] = min(
        (int(row["batch_size"]) for row in peak_rows if row["status"] == "oom"), default=None
    )
    constrained = [row for row in rows if row["experiment"] == "constrained"]
    result["constrained"] = [
        {
            "fraction": row.get("memory_fraction"),
            "processed": row.get("processed", 0),
            "total": row.get("total", args.image_count),
            "status": row.get("status"),
        }
        for row in constrained
    ]
    return result


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RF-DETR 高负荷稳定性实验")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "stress_test")
    parser.add_argument("--image-count", type=int, default=16)
    parser.add_argument("--synthetic-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--long-iterations", type=int, default=20)
    parser.add_argument("--long-batch", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--constrained-batch", type=int, default=8)
    parser.add_argument("--memory-fractions", default="0.75,0.50,0.35")
    return parser.parse_args()


def main() -> None:
    """加载模型、执行实验并写出报告。"""
    args = _parse_args()
    args.batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    args.memory_fractions = [float(value) for value in args.memory_fractions.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    model = RFDETR.from_checkpoint(args.checkpoint, device=args.device)
    images = _load_images(args.image_dir, args.image_count, args.synthetic_size)
    rows = _long_run(model, images, args) + _peak_batch(model, images, args) + _constrained_run(model, images, args)
    _write_csv(args.output_dir / "stress_samples.csv", rows)
    summary = _summary(rows, args)
    (args.output_dir / "stress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(rows, args.output_dir)
    report = [
        "# RF-DETR 高负荷稳定性实验报告",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- device: `{args.device}`",
        f"- images: `{len(images)}`",
        "- image source: `output/0820-SHWX-pretrain-checkpoint-eval/FN/MS`（前 16 张，按文件名排序）",
        "",
        "## 判定口径",
        "",
        "连续实验的显存/RSS 斜率接近 0 且摘要哈希不变化，视为未观察到漂移；",
        "峰值批量以最后一个成功批量为上限；受限显存以 processed/total 判断是否完整。",
        "本次连续实验为 2 次预热 + 20 次采样；峰值批量测试 1/2/4/8/16；显存上限测试 75%/50%/35%。",
        "RSS 是进程常驻内存，包含 Python、解码器和运行时开销；斜率用于趋势筛查，不等同于严格泄漏证明。",
        "",
        "## 结果",
        "",
        f"- 连续推理 GPU allocated 斜率: `{summary.get('long_gpu_slope_mb_per_iter', 'NA'):.6f}` MB/iter"
        if "long_gpu_slope_mb_per_iter" in summary
        else "- 连续推理数据不足",
        f"- 连续推理 RSS 斜率: `{summary.get('long_rss_slope_mb_per_iter', 'NA'):.6f}` MB/iter"
        if "long_rss_slope_mb_per_iter" in summary
        else "",
        f"- 连续推理摘要哈希种类数: `{summary.get('long_digest_unique', 'NA')}`",
        f"- 最大成功批量: `{summary['peak_max_success_batch']}`",
        f"- 首个 OOM 批量: `{summary['peak_first_oom_batch']}`",
        "",
        "## 文件",
        "",
        "- `stress_samples.csv`: 每次推理的原始记录",
        "- `stress_summary.json`: 汇总指标",
        "- `stability_overview.png`: 连续推理与批量压力的横向双联图",
        "",
        "## 复现命令",
        "",
        (
            "`python src/scripts/evaluation/stress_test.py --checkpoint output/0825baseline/checkpoint_best_total.pth "
            "--image-dir output/0820-SHWX-pretrain-checkpoint-eval/FN/MS --output-dir output/stress_test "
            "--image-count 16 --warmup 2 --long-iterations 20 --batch-sizes 1,2,4,8,16 "
            "--constrained-batch 8 --memory-fractions 0.75,0.50,0.35`"
        ),
    ]
    (args.output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
