# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""为历史 SHWX 实验补充统一全量测试，并保存可恢复的执行清单。"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_RESULT_DIR = OUTPUT_ROOT / "experiment_comparison_20260820"
FULL_TEST_DATASET = "/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test"
CHECKPOINT_NAMES = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last_ema.pth")


@dataclass(frozen=True)
class Candidate:
    """一个待补测的独立权重文件。"""

    checkpoint: Path
    method: str
    source_report: str
    selection_reason: str


def parse_args() -> argparse.Namespace:
    """解析补测调度参数。"""
    parser = argparse.ArgumentParser(description="补测历史 SHWX 实验权重")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="最多运行的候选数，0 表示全部")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_sibling_checkpoint(report_path: Path) -> Path | None:
    """在报告目录及其实验祖先目录中寻找可用的主权重。"""
    for parent in (report_path.parent, *report_path.parents):
        if parent == OUTPUT_ROOT.parent:
            break
        for name in CHECKPOINT_NAMES:
            checkpoint = parent / name
            if checkpoint.is_file():
                return checkpoint.resolve()
        if parent == OUTPUT_ROOT:
            break
    return None


def resolve_report_checkpoint(report_path: Path, content: str) -> Path | None:
    """优先使用报告记录的权重路径，失效时回退到本地实验目录。"""
    match = re.search(r"^权重:\s*(.+)$", content, flags=re.MULTILINE)
    if match:
        value = match.group(1).strip()
        if not value.startswith("PAN="):
            checkpoint = Path(value)
            if checkpoint.is_file():
                return checkpoint.resolve()
    return find_sibling_checkpoint(report_path)


def is_shwx_historical_report(report_path: Path, content: str) -> bool:
    """判断报告是否属于需要补测的 SHWX 历史完整测试。"""
    if FULL_TEST_DATASET in content:
        return False
    image_match = re.search(r"^测试图像数:\s*(\d+)$", content, flags=re.MULTILINE)
    image_count = int(image_match.group(1)) if image_match else 0
    lowered = str(report_path).lower()
    return (
        image_count >= 600
        and "数据集: shwx" in content
        and "dior" not in lowered
        and "_score_analysis" not in lowered
        and "测试大图实验" not in str(report_path)
    )


def collect_candidates() -> list[Candidate]:
    """从历史报告和未报告的 SHWX 主权重中收集候选，按路径去重。"""
    candidates: dict[Path, Candidate] = {}
    completed_checkpoints: set[Path] = set()
    for report_path in sorted(OUTPUT_ROOT.rglob("test_result.txt")):
        if DEFAULT_RESULT_DIR in report_path.resolve().parents:
            continue
        content = report_path.read_text(encoding="utf-8", errors="replace")
        checkpoint = resolve_report_checkpoint(report_path, content)
        if FULL_TEST_DATASET in content and checkpoint is not None:
            completed_checkpoints.add(checkpoint)
            continue
        if not is_shwx_historical_report(report_path, content):
            continue
        if checkpoint is None:
            continue
        method = str(report_path.parent.relative_to(OUTPUT_ROOT))
        candidates.setdefault(
            checkpoint,
            Candidate(checkpoint, method, str(report_path), "历史 SHWX 测试报告未使用统一全量测试集"),
        )

    for checkpoint in sorted(OUTPUT_ROOT.rglob("checkpoint_best_total.pth")):
        path_string = str(checkpoint)
        lowered = path_string.lower()
        if (
            "dior" in lowered
            or "cgwx" in lowered
            or "large-cut" in lowered
            or "/dual-" in lowered
            or "/pan/" in lowered
            or DEFAULT_RESULT_DIR.name in lowered
        ):
            continue
        resolved = checkpoint.resolve()
        if resolved in completed_checkpoints:
            continue
        candidates.setdefault(
            resolved,
            Candidate(
                resolved,
                str(checkpoint.parent.relative_to(OUTPUT_ROOT)),
                "",
                "存在主权重但未发现可比较的统一全量测试报告",
            ),
        )
    return sorted(candidates.values(), key=lambda candidate: (candidate.method, str(candidate.checkpoint)))


def write_manifest(path: Path, candidates: list[Candidate], states: dict[Path, str]) -> None:
    """写入可恢复的候选清单和当前执行状态。"""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("method", "checkpoint", "source_report", "selection_reason", "state", "result_dir"),
        )
        writer.writeheader()
        for index, candidate in enumerate(candidates, start=1):
            result_dir = path.parent / "backfill_results" / f"{index:03d}"
            writer.writerow(
                {
                    "method": candidate.method,
                    "checkpoint": candidate.checkpoint,
                    "source_report": candidate.source_report,
                    "selection_reason": candidate.selection_reason,
                    "state": states.get(candidate.checkpoint, "pending"),
                    "result_dir": result_dir,
                }
            )


def write_test_config(path: Path, checkpoint: Path, output_dir: Path) -> None:
    """创建最小统一评测配置，保留权重自身记录的推理分辨率。"""
    path.write_text(
        "\n".join(
            (
                "test:",
                "  dataset: shwx",
                f"  dataset_dir: {FULL_TEST_DATASET}",
                f"  checkpoint: {checkpoint}",
                f"  output_dir: {output_dir}",
                "  conf_threshold: 0.25",
                "  class_conf_thresholds: {}",
                "  device: cuda:0",
                "  batch_size: 32",
                "  num_workers: 12",
                "  prefetch_factor: 3",
                "  precision: auto",
                "  compile_model: false",
                "  copy_prefetch: true",
                "  warmup_batches: 1",
                "  progress_interval_s: 1.0",
                "  gpu_monitor_enabled: false",
                "  save_fp_fn: false",
                "  save_yolo_preds: false",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    """创建清单并顺序执行尚未完成的补测任务。"""
    args = parse_args()
    result_dir = args.result_dir.resolve()
    backfill_dir = result_dir / "backfill_results"
    backfill_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates()
    states: dict[Path, str] = {}
    selected = 0
    for index, candidate in enumerate(candidates, start=1):
        run_dir = backfill_dir / f"{index:03d}"
        result_path = run_dir / "test_result.txt"
        if FULL_TEST_DATASET in result_path.read_text(encoding="utf-8", errors="replace") if result_path.exists() else False:
            states[candidate.checkpoint] = "completed"
            continue
        if args.limit and selected >= args.limit:
            states[candidate.checkpoint] = "pending"
            continue
        selected += 1
        if args.dry_run:
            states[candidate.checkpoint] = "dry_run"
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "test_config.yaml"
        write_test_config(config_path, candidate.checkpoint, run_dir)
        command = [sys.executable, "src/scripts/test.py", "-c", str(config_path)]
        print(f"[i] ({index}/{len(candidates)}) 补测: {candidate.method}", flush=True)
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        states[candidate.checkpoint] = "completed" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"
        write_manifest(result_dir / "backfill_manifest.csv", candidates, states)
    write_manifest(result_dir / "backfill_manifest.csv", candidates, states)
    print(f"[i] 候选权重 {len(candidates)} 个，本轮处理 {selected} 个", flush=True)


if __name__ == "__main__":
    main()
