# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""汇总 SHWX 实验报告，并按比赛七项排名规则生成表格和图表。"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "experiment_comparison_20260820"
FULL_TEST_DATASET = "/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test"
GROUPS = ("ship", "aircraft", "vehicle")


@dataclass(frozen=True)
class Report:
    """一份测试报告中可用于比赛排序的核心信息。"""

    report_path: str
    method: str
    checkpoint: str
    test_images: int | None
    dataset_dir: str
    is_full_test: bool
    throughput_img_s: float | None
    ship_recall: float | None
    ship_fdr: float | None
    aircraft_recall: float | None
    aircraft_fdr: float | None
    vehicle_recall: float | None
    vehicle_fdr: float | None
    total_recall: float | None
    total_fdr: float | None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="汇总 SHWX 实验指标并生成比赛口径排名")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    """把正则捕获的数值转换为浮点数。"""
    return float(value) if value is not None else None


def parse_report(report_path: Path, output_root: Path) -> Report:
    """解析单个 ``test_result.txt`` 文件。

    Args:
        report_path: 待解析的报告路径。
        output_root: ``output`` 根目录，用于生成稳定的方法标识。

    Returns:
        解析后的报告对象；缺失字段以 ``None`` 表示。
    """
    content = report_path.read_text(encoding="utf-8", errors="replace")
    dataset_match = re.search(r"^数据集目录:\s*(.+)$", content, flags=re.MULTILINE)
    image_match = re.search(r"^测试图像数:\s*(\d+)$", content, flags=re.MULTILINE)
    checkpoint_match = re.search(r"^权重:\s*(.+)$", content, flags=re.MULTILINE)
    speed_match = re.search(r"^推理吞吐:\s*([0-9.]+)\s*img/s", content, flags=re.MULTILINE)
    total_match = re.search(
        r"^total\s+avgTP=.*?avgRecall=([0-9.]+)\s+avgFDR=([0-9.]+)",
        content,
        flags=re.MULTILINE,
    )
    metrics: dict[str, tuple[float | None, float | None]] = {}
    for group in GROUPS:
        match = re.search(
            rf"^{group}\s+avgTP=.*?avgRecall=([0-9.]+)\s+avgFDR=([0-9.]+)",
            content,
            flags=re.MULTILINE,
        )
        metrics[group] = (parse_float(match.group(1)) if match else None, parse_float(match.group(2)) if match else None)

    dataset_dir = dataset_match.group(1).strip() if dataset_match else ""
    try:
        method = str(report_path.parent.relative_to(output_root))
    except ValueError:
        method = str(report_path.parent)
    return Report(
        report_path=str(report_path),
        method=method,
        checkpoint=checkpoint_match.group(1).strip() if checkpoint_match else "",
        test_images=int(image_match.group(1)) if image_match else None,
        dataset_dir=dataset_dir,
        is_full_test=FULL_TEST_DATASET in dataset_dir,
        throughput_img_s=parse_float(speed_match.group(1)) if speed_match else None,
        ship_recall=metrics["ship"][0],
        ship_fdr=metrics["ship"][1],
        aircraft_recall=metrics["aircraft"][0],
        aircraft_fdr=metrics["aircraft"][1],
        vehicle_recall=metrics["vehicle"][0],
        vehicle_fdr=metrics["vehicle"][1],
        total_recall=parse_float(total_match.group(1)) if total_match else None,
        total_fdr=parse_float(total_match.group(2)) if total_match else None,
    )


def competition_ranks(reports: list[Report], field: str, higher_is_better: bool) -> dict[str, int | None]:
    """按比赛名次规则计算单项排名，相同数值并列并跳过后续名次。"""
    valid = [report for report in reports if getattr(report, field) is not None]
    valid.sort(key=lambda report: getattr(report, field), reverse=higher_is_better)
    ranks: dict[str, int | None] = {report.report_path: None for report in reports}
    previous_value: float | None = None
    previous_rank = 0
    for index, report in enumerate(valid, start=1):
        value = getattr(report, field)
        if previous_value is None or value != previous_value:
            previous_rank = index
            previous_value = value
        ranks[report.report_path] = previous_rank
    return ranks


def ranked_rows(reports: list[Report]) -> list[dict[str, object]]:
    """生成七项排名及排名和，并附加短板诊断字段。"""
    rank_specs = (
        ("ship_recall", True),
        ("ship_fdr", False),
        ("aircraft_recall", True),
        ("aircraft_fdr", False),
        ("vehicle_recall", True),
        ("vehicle_fdr", False),
        ("throughput_img_s", True),
    )
    rank_maps = {field: competition_ranks(reports, field, higher) for field, higher in rank_specs}
    rows: list[dict[str, object]] = []
    for report in reports:
        row = asdict(report)
        metric_ranks = [rank_maps[field][report.report_path] for field, _ in rank_specs]
        row.update({f"rank_{field}": rank for (field, _), rank in zip(rank_specs, metric_ranks)})
        present_ranks = [rank for rank in metric_ranks if rank is not None]
        row["rank_sum"] = sum(present_ranks) if len(present_ranks) == len(rank_specs) else None
        row["worst_rank"] = max(present_ranks) if len(present_ranks) == len(rank_specs) else None
        recalls = [value for value in (report.ship_recall, report.aircraft_recall, report.vehicle_recall) if value is not None]
        fdrs = [value for value in (report.ship_fdr, report.aircraft_fdr, report.vehicle_fdr) if value is not None]
        row["min_group_recall"] = min(recalls) if len(recalls) == 3 else None
        row["max_group_fdr"] = max(fdrs) if len(fdrs) == 3 else None
        row["balance_warning"] = bool(
            row["min_group_recall"] is not None
            and (row["min_group_recall"] < 0.75 or row["max_group_fdr"] > 0.25)
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["rank_sum"] is None,
            row["rank_sum"] if row["rank_sum"] is not None else float("inf"),
            row["worst_rank"] if row["worst_rank"] is not None else float("inf"),
            row["method"],
        ),
    )


def balanced_rows(reports: list[Report]) -> list[dict[str, object]]:
    """按六项检测指标筛选内部最佳方案，不把吞吐量纳入方案优劣。

    先要求最低大类召回率达到 0.78、最高大类虚警率不超过 0.245，
    再按平均召回率降序、平均虚警率升序、最低召回率降序、最高虚警率升序排序。
    这只是内部选型的均衡性筛选，不代表正式比赛的跨队伍名次。
    """
    rows: list[dict[str, object]] = []
    for report in reports:
        row = asdict(report)
        recalls = [report.ship_recall, report.aircraft_recall, report.vehicle_recall]
        fdrs = [report.ship_fdr, report.aircraft_fdr, report.vehicle_fdr]
        if any(value is None for value in (*recalls, *fdrs)):
            continue
        row["mean_group_recall"] = sum(recalls) / 3
        row["mean_group_fdr"] = sum(fdrs) / 3
        row["min_group_recall"] = min(recalls)
        row["max_group_fdr"] = max(fdrs)
        row["recall_range"] = max(recalls) - min(recalls)
        row["fdr_range"] = max(fdrs) - min(fdrs)
        row["passes_balance_gate"] = row["min_group_recall"] >= 0.78 and row["max_group_fdr"] <= 0.245
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not row["passes_balance_gate"],
            -row["mean_group_recall"],
            row["mean_group_fdr"],
            -row["min_group_recall"],
            row["max_group_fdr"],
            row["method"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """将字典行写入 UTF-8 BOM CSV，便于电子表格直接打开。"""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    """把方法路径转换为可跨平台使用的文件名。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")[:180]


def copy_source_reports(reports: list[Report], result_dir: Path) -> None:
    """将全部原始报告复制到汇总目录，并以清单建立来源映射。"""
    source_dir = result_dir / "source_reports"
    source_dir.mkdir(exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for index, report in enumerate(reports, start=1):
        destination = source_dir / f"{index:03d}_{safe_name(report.method)}.txt"
        shutil.copyfile(report.report_path, destination)
        manifest_rows.append(
            {
                "source_report": report.report_path,
                "copied_report": str(destination),
                "method": report.method,
                "checkpoint": report.checkpoint,
                "is_full_test": report.is_full_test,
            }
        )
    write_csv(result_dir / "source_report_manifest.csv", manifest_rows)


def draw_chart(rows: list[dict[str, object]], output_path: Path) -> None:
    """绘制按排名和排序的召回率、虚警率和吞吐量对比曲线。"""
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["rank_sum"] is not None]
    if not selected:
        return
    labels = [str(index) for index in range(1, len(selected) + 1)]
    x_axis = list(range(1, len(selected) + 1))
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    for key, label, color in (
        ("ship_recall", "ship", "#1f77b4"),
        ("aircraft_recall", "aircraft", "#2ca02c"),
        ("vehicle_recall", "vehicle", "#d62728"),
    ):
        axes[0].plot(x_axis, [row[key] for row in selected], marker="o", label=label, color=color)
    axes[0].set_title("Group recall by competition rank-sum order")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=3)
    for key, label, color in (
        ("ship_fdr", "ship", "#1f77b4"),
        ("aircraft_fdr", "aircraft", "#2ca02c"),
        ("vehicle_fdr", "vehicle", "#d62728"),
    ):
        axes[1].plot(x_axis, [row[key] for row in selected], marker="o", label=label, color=color)
    axes[1].set_title("Group false discovery rate by competition rank-sum order")
    axes[1].set_ylabel("FDR (lower is better)")
    axes[1].legend(ncol=3)
    axes[2].plot(x_axis, [row["throughput_img_s"] for row in selected], marker="o", color="#9467bd")
    axes[2].set_title("Inference throughput by competition rank-sum order")
    axes[2].set_ylabel("Images / second")
    axes[2].set_xlabel("Scheme position in rank-sum order")
    axes[2].set_xticks(x_axis, labels)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(path: Path, all_reports: list[Report], ranked: list[dict[str, object]]) -> None:
    """输出可读的汇总说明和当前排名前列方案。"""
    full_rows = [row for row in ranked if row["rank_sum"] is not None]
    historical_count = len(all_reports) - sum(report.is_full_test for report in all_reports)
    backfill_manifest = path.parent / "backfill_manifest.csv"
    backfill_total = backfill_success = backfill_failed = 0
    if backfill_manifest.exists():
        with backfill_manifest.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                backfill_total += 1
                if row.get("state") == "completed":
                    backfill_success += 1
                elif row.get("state", "").startswith("failed"):
                    backfill_failed += 1
    lines = [
        "# SHWX 实验统一评估汇总",
        "",
        f"- 扫描报告数：{len(all_reports)}",
        f"- 明确使用统一测试集的报告数：{sum(report.is_full_test for report in all_reports)}",
        f"- 历史非统一报告数：{historical_count}",
        f"- 历史权重补测：{backfill_total} 个候选，成功 {backfill_success} 个，失败 {backfill_failed} 个（失败项保留日志，不计入排名）",
        f"- 统一测试集：`{FULL_TEST_DATASET}`",
        "- DIOR、CGWX、双流分支和大图边界检测器等非 SHWX 完整检测方案不进入比赛七项排名；它们仍保留在 `all_report_inventory.csv` 和原始报告归档中。",
        "- 排名项：船/飞机/车辆的召回率和虚警率，以及推理吞吐量；召回率和吞吐量越高越好，虚警率越低越好。",
        "- 名次使用并列跳号规则，七项名次相加后升序排列；`worst_rank` 仅作同分时的均衡性比较。",
        "- `balance_warning=true` 表示至少一类召回率低于 0.75，或至少一类虚警率高于 0.25。",
        "",
        "## 本团队内部最佳方案",
        "",
        "正式比赛的七项名次和用于跨团队比较，不能用来决定本团队内部哪个实验最好。内部选型只看船、飞机、车辆三类的召回率和虚警率，吞吐量不参与效果优劣判断。",
        "当前推荐 `proto_guidance_alignment_repair/20_ms_semantic_selection_w025_test`，对应权重 `output/proto_guidance_alignment_repair/20_ms_semantic_selection_w025/checkpoint_best_total.pth`。三类召回率为 `0.8978 / 0.9954 / 0.7844`，三类虚警率为 `0.1701 / 0.0193 / 0.2428`；最低召回率 `0.7844`，最高虚警率 `0.2428`，没有明显短板。完整内部排序见 `balanced_internal_ranked.md` 和 `balanced_internal_ranked.csv`。",
        "`21b_selection_group0_lambda050/last_ema.pth` 是偏保守压低虚警的备选：最高虚警率为 `0.2353`，但车辆召回率为 `0.7784`。",
        "",
        "## 已统一测试的方案排名",
        "",
        "|排名|方法|权重文件|七项名次和|最差单项名次|最低召回率|最高虚警率|吞吐(img/s)|均衡警告|",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(full_rows, start=1):
        lines.append(
            f"|{index}|{row['method']}|`{Path(str(row['checkpoint'])).name}`|{row['rank_sum']}|{row['worst_rank']}|"
            f"{row['min_group_recall']:.4f}|{row['max_group_fdr']:.4f}|"
            f"{row['throughput_img_s']:.1f}|{row['balance_warning']}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_balanced_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """输出不含吞吐量的内部均衡选型说明。"""
    lines = [
        "# 内部方案均衡性排序",
        "",
        "此表用于选择本团队内部最终方案，不模拟正式比赛的跨队伍排名。吞吐量不参与方案优劣，只作为工程参考。",
        "筛选门槛：最低大类召回率 >= 0.7800，最高大类虚警率 <= 0.2450；门槛内按平均召回率降序、平均虚警率升序，再按最差大类指标排序。",
        "",
        "|排名|方法|权重|平均召回率|平均虚警率|最低召回率|最高虚警率|船召回/虚警|飞机召回/虚警|车辆召回/虚警|吞吐(img/s)|通过均衡门槛|",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---:|---|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"|{index}|{row['method']}|`{Path(str(row['checkpoint'])).name}`|"
            f"{row['mean_group_recall']:.4f}|{row['mean_group_fdr']:.4f}|"
            f"{row['min_group_recall']:.4f}|{row['max_group_fdr']:.4f}|"
            f"{row['ship_recall']:.4f}/{row['ship_fdr']:.4f}|"
            f"{row['aircraft_recall']:.4f}/{row['aircraft_fdr']:.4f}|"
            f"{row['vehicle_recall']:.4f}/{row['vehicle_fdr']:.4f}|"
            f"{row['throughput_img_s']:.1f}|{row['passes_balance_gate']}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relabel_backfill_reports(reports: list[Report], result_dir: Path) -> list[Report]:
    """用补测清单中的原实验路径替换编号输出目录。"""
    manifest_path = result_dir / "backfill_manifest.csv"
    if not manifest_path.exists():
        return reports
    labels: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            labels[str(Path(row["result_dir"]).resolve())] = row["method"]
    relabeled: list[Report] = []
    for report in reports:
        label = labels.get(str(Path(report.report_path).parent.resolve()))
        relabeled.append(replace(report, method=label) if label else report)
    return relabeled


def main() -> None:
    """扫描报告、生成库存表、统一测试排名表和可视化图。"""
    args = parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    source_reports_dir = result_dir / "source_reports"
    reports = [
        parse_report(path, args.output_root.resolve())
        for path in sorted(args.output_root.rglob("test_result.txt"))
        if source_reports_dir not in path.resolve().parents
    ]
    reports = relabel_backfill_reports(reports, result_dir)
    all_rows = [asdict(report) for report in reports]
    full_reports = [report for report in reports if report.is_full_test]
    ranked = ranked_rows(full_reports)
    balanced = balanced_rows(full_reports)
    write_csv(result_dir / "all_report_inventory.csv", all_rows)
    write_csv(result_dir / "full_test_ranked.csv", ranked)
    write_csv(result_dir / "balanced_internal_ranked.csv", balanced)
    copy_source_reports(reports, result_dir)
    draw_chart(ranked, result_dir / "full_test_metric_trends.png")
    write_summary(result_dir / "README.md", reports, ranked)
    write_balanced_summary(result_dir / "balanced_internal_ranked.md", balanced)
    print(f"[i] 已扫描 {len(reports)} 份报告，统一测试报告 {len(full_reports)} 份")
    print(f"[i] 结果目录: {result_dir}")


if __name__ == "__main__":
    main()
