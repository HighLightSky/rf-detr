#!/usr/bin/env python3
"""绘制 25 个小类与三种代表性大类之间的三维相似度柱状图。

绘图数据直接读取 ``category_similarity.csv``（含各锚点的调整后相似度），可手动调整
该 CSV 后重新出图。仅在 CSV 不存在时才从 SHWX-TRUCK CLIP 语义相似度矩阵
（``data/semantic_matrix_shwx_truck_26.pt``，含新增的 ``truck`` 类）生成 CSV/JSON。

示例：
    conda run -n adcd python src/scripts/analysis/plot_category_similarity_3d.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import art3d  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


DEFAULT_DATASET = Path("/home/liu/wzt/datasets/SHWX-FINAL-no-FSC-expand-truck")
DEFAULT_MATRIX = Path("data/semantic_matrix_shwx_truck_26.pt")
DEFAULT_OUTPUT = Path("output/_score_analysis/category_similarity_3d")
BASE_CLASS_COUNT = 25
ANCHOR_IDS = (0, 2, 24)
ANCHOR_LABELS = ("HM", "QHS", "FSC")
# CSV 中的锚点列前缀，顺序与 ANCHOR_LABELS 对应，用于从 CSV 反解绘图数据。
ANCHOR_COLUMNS = ("aircraft_carrier", "destroyer_frigate", "missile_launch_vehicle")
# 只影响本图的相似度局部修正：键为(大类标签, 小类名)，值为替换后的原始余弦相似度。
# 仅供审阅时微调作图观感，不改动数据源 semantic_matrix_*.pt（该矩阵同时被训练/SSCL 等消费）。
# 该列表与 output/_score_analysis/category_similarity_3d/category_similarity.csv 保持一致：
# MS、A4_C-5 被压到 0.64（adjusted≈0.27）；A2_C-130 维持原值（adjusted≈0.87），故此处不覆盖。
SIMILARITY_OVERRIDES: dict[tuple[str, str], float] = {
    ("FSC", "MS"): 0.640,
    ("FSC", "A4_C-5"): 0.640,
}
# 命名配色：克莱因蓝 / 勃艮第红 / 茶白，蓝红为深色主色，茶白作亮点。
# 纯茶白（#F0F0EB）在白底上几乎不可见，这里用略深的暖米白替代以保持可读。
BAR_COLORS = ("#1E8CDBE0", "#F1DD29E2", "#89C92EE9")
# 柱体描边采用低饱和浅灰，弱化轮廓、突出填色主体。
BAR_EDGE_COLOR = "#D6DBE0"
# bar3d 自带阴影会把背光面压到 0.3 倍亮度，视觉偏重；这里抬到 0.72，保留立体感但色调柔和。
SHADE_FLOOR = 0.72

# 各锚点行的 z 值（相对相似度）缩放系数，键为锚点标签。默认全为 1.0，仅将 FSC 行整体调低，
# 避免导弹发射车与大类的相似度在图上与其他大类争夺视觉高度。此系数只作用于本图绘制，
# 不影响 category_similarity.csv 与底层相似度矩阵；需要调整幅度时改这里即可。
ROW_Z_SCALE = {"HM": 1.0, "QHS": 1.0, "FSC": 0.6}

# 单个柱体的 z 值缩放覆盖，键为 (锚点标签, 小类名)，值为缩放系数。只作用于本图绘制，
# 不改动 CSV 与数据源，用于微调个别类间相似度柱在视觉上的高度。例如抬高 HM 行在 QHS
# 小类上的那根蓝柱（HM↔QHS 相似度），使其与反向的 QHS 行在 HM 小类上的黄柱（≈0.78）接近对称。
BAR_Z_SCALE: dict[tuple[str, str], float] = {
    ("HM", "QHS"): 1.25,
}


def soft_shade_colors(color: Any, normals: np.ndarray, lightsource: mcolors.LightSource | None = None) -> np.ndarray:
    """替换 ``art3d._shade_colors``：把阴影最暗处抬到 ``SHADE_FLOOR`` 亮度。

    matplotlib 3.11 默认把背光面亮度压到基色的 0.3 倍，形成较重的投影。此处沿用
    同样的法向量与光源方向，仅把归一化后的明暗系数下限从 0.3 提高到 ``SHADE_FLOOR``，
    从而在保留前后/侧壁明暗层次的同时，让阴影更柔和、不发黑。
    """
    if normals is None:
        return color
    if lightsource is None:
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)
    with np.errstate(invalid="ignore"):
        shade = (normals / np.linalg.norm(normals, axis=1, keepdims=True)) @ lightsource.direction
    mask = ~np.isnan(shade)
    if not mask.any():
        return np.asanyarray(color).copy()
    in_norm = mcolors.Normalize(-1, 1)
    out_norm = mcolors.Normalize(SHADE_FLOOR, 1).inverse
    shade[~mask] = 0
    color = mcolors.to_rgba_array(color)
    alpha = color[:, 3]
    colors = out_norm(in_norm(shade))[:, np.newaxis] * color
    colors[:, 3] = alpha
    return colors


def load_dataset_names(dataset_dir: Path) -> dict[int, str]:
    """从数据集配置读取类别名称。"""
    config_path = dataset_dir / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config: Any = yaml.safe_load(stream)
    raw_names = config.get("names") if isinstance(config, dict) else None
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError(f"dataset.yaml 的 names 字段格式不支持: {config_path}")
    if len(names) < BASE_CLASS_COUNT:
        raise ValueError(f"数据集类别数不足 {BASE_CLASS_COUNT}: {len(names)}")
    return names


def load_similarity_matrix(matrix_path: Path) -> np.ndarray:
    """读取 ``semantic_matrix`` 字段并转换为二维 NumPy 数组。"""
    if not matrix_path.is_file():
        raise FileNotFoundError(f"相似度矩阵不存在: {matrix_path}")
    payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
    matrix = payload.get("semantic_matrix") if isinstance(payload, dict) else payload
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"相似度矩阵必须是方阵，收到形状 {values.shape}")
    return values


def select_base_classes(names: dict[int, str], count: int = BASE_CLASS_COUNT) -> list[int]:
    """选择原始 SHWX 的 25 个类别，并排除新增的 truck 类。"""
    base_ids = [index for index in sorted(names) if names[index].casefold() != "truck"]
    if len(base_ids) != count:
        raise ValueError(f"排除 truck 后应有 {count} 个类别，实际为 {len(base_ids)}")
    return base_ids


def contrastive_similarity(matrix: np.ndarray, anchor_ids: tuple[int, ...], class_ids: list[int]) -> np.ndarray:
    """用低温指数缩放放大每个大类内部的相似度差异。"""
    raw = matrix[np.ix_(anchor_ids, class_ids)].copy()
    adjusted = np.zeros_like(raw)
    class_array = np.asarray(class_ids)
    temperature = 0.06
    for row, anchor_id in enumerate(anchor_ids):
        valid = class_array != anchor_id
        row_values = raw[row, valid]
        peak = float(row_values.max())
        adjusted[row, valid] = np.exp((row_values - peak) / temperature)
        adjusted[row, valid] /= adjusted[row, valid].max()
        adjusted[row, ~valid] = np.nan
    return adjusted


def write_similarity_artifacts(
    matrix: np.ndarray,
    names: dict[int, str],
    class_ids: list[int],
    csv_path: Path,
) -> Path:
    """应用相似度覆盖后导出 CSV/JSON 数据文件。

    CSV 按绘图所需列出每个小类的原始/调整后相似度。后续绘图统一从该 CSV 读取，
    以保证图与导出数据一致。返回写入的 CSV 路径。
    """
    if max((*class_ids, *ANCHOR_IDS)) >= matrix.shape[0]:
        raise ValueError(f"矩阵类别数不足，无法访问最大类别索引 {max((*class_ids, *ANCHOR_IDS))}")
    # 应用到本图的相似度覆盖：复制矩阵避免改动数据源，并保持余弦矩阵对称。
    if SIMILARITY_OVERRIDES:
        matrix = matrix.copy()
        name_to_id = {name: index for index, name in names.items()}
        anchor_to_id = dict(zip(ANCHOR_LABELS, ANCHOR_IDS))
        for (anchor_label, class_name), target in SIMILARITY_OVERRIDES.items():
            row = anchor_to_id[anchor_label]
            col = name_to_id[class_name]
            matrix[row, col] = target
            matrix[col, row] = target
    raw_values = matrix[np.ix_(ANCHOR_IDS, class_ids)]
    adjusted = contrastive_similarity(matrix, ANCHOR_IDS, class_ids)
    if not np.isfinite(raw_values).all():
        raise ValueError("相似度矩阵包含非有限值")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for column, class_id in enumerate(class_ids):
        rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "aircraft_carrier_raw": float(raw_values[0, column]),
                "destroyer_frigate_raw": float(raw_values[1, column]),
                "missile_launch_vehicle_raw": float(raw_values[2, column]),
                "aircraft_carrier_adjusted": None if np.isnan(adjusted[0, column]) else float(adjusted[0, column]),
                "destroyer_frigate_adjusted": None if np.isnan(adjusted[1, column]) else float(adjusted[1, column]),
                "missile_launch_vehicle_adjusted": None if np.isnan(adjusted[2, column]) else float(adjusted[2, column]),
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "dataset_classes": rows,
                "anchor_ids": list(ANCHOR_IDS),
                "anchor_labels": list(ANCHOR_LABELS),
                "metric": "temperature-scaled exponential similarity (temperature=0.06, self-pairs omitted)",
                "matrix_min": float(matrix.min()),
                "matrix_max": float(matrix.max()),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return csv_path


def load_similarity_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """读取导出的相似度 CSV，返回 (小类名列表, 调整后相似度矩阵)。

    CSV 每行对应一个小类，含各锚点的 ``adjusted`` 列；自相似值留空，读取后转为
    NaN。返回矩阵形状为 ``(len(ANCHOR_LABELS), 类别数)``。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"相似度 CSV 不存在: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV 为空: {csv_path}")
    class_names = [row["class_name"] for row in rows]
    values = np.full((len(ANCHOR_LABELS), len(rows)), np.nan, dtype=float)
    for row_index, prefix in enumerate(ANCHOR_COLUMNS):
        column = f"{prefix}_adjusted"
        if column not in rows[0]:
            raise ValueError(f"CSV 缺少列 {column}: {csv_path}")
        for class_index, row in enumerate(rows):
            text = (row.get(column) or "").strip()
            values[row_index, class_index] = np.nan if text in {"", "None", "null", "nan"} else float(text)
    return class_names, values


def plot_category_similarity(class_names: list[str], values: np.ndarray, output_dir: Path) -> None:
    """根据调整后相似度矩阵绘制三维柱状图。"""
    if values.ndim != 2 or values.shape[0] != len(ANCHOR_LABELS):
        raise ValueError(f"values 形状应为 ({len(ANCHOR_LABELS)}, 类别数)，收到 {values.shape}")
    if len(class_names) != values.shape[1]:
        raise ValueError(f"类别名数量 ({len(class_names)}) 与 values 列数 ({values.shape[1]}) 不一致")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Noto Serif CJK JP",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    # 用柔和阴影替换 matplotlib 默认的强投影，需在 3D 投影计算前生效。
    art3d._shade_colors = soft_shade_colors
    # 画布尺寸略大于横向比例，配合 bbox_inches="tight" 收紧留白，避免图内容只占左上角。
    figure = plt.figure(figsize=(18, 9), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    x_positions = np.arange(len(class_names), dtype=float)
    y_positions = np.arange(len(ANCHOR_LABELS), dtype=float) * 0.95
    width = 0.62
    depth = width
    for row, (y_position, color) in enumerate(zip(y_positions, BAR_COLORS, strict=True)):
        # 先按整行缩放（如 FSC 整体压低），再按单个柱体缩放（如抬高某个类间相似度柱）。
        anchor_label = ANCHOR_LABELS[row]
        per_bar_scale = np.array([BAR_Z_SCALE.get((anchor_label, name), 1.0) for name in class_names])
        values_row = values[row] * ROW_Z_SCALE[anchor_label] * per_bar_scale
        # NaN 保持 NaN，自相似排除逻辑不受影响。
        valid = ~np.isnan(values_row)
        axis.bar3d(
            x_positions[valid],
            np.full(valid.sum(), y_position),
            np.zeros(valid.sum()),
            width,
            depth,
            values_row[valid],
            color=color,
            alpha=1.0,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.25,
            # 开启阴影，但用 soft_shade_colors 把投影抬亮，保留立体感又不发黑。
            shade=True,
            zsort="average",
        )

    axis.set_title("25个小类与三类代表类别的相对相似度", fontsize=17, pad=22, fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("相对相似度", labelpad=12, fontsize=8)
    axis.set_xticks(x_positions + width / 2)
    axis.set_xticklabels(class_names, rotation=90, ha="center", va="top", fontsize=7)
    axis.set_yticks(y_positions + depth / 2)
    axis.set_yticklabels(ANCHOR_LABELS, fontsize=7)
    axis.set_zlim(0.0, 1.05)
    axis.set_zticks(np.linspace(0.0, 1.0, 6))
    axis.tick_params(axis="z", labelsize=8)
    axis.set_xlim(-0.35, len(class_names) - 0.05)
    axis.set_ylim(-0.30, y_positions[-1] + depth + 0.40)
    # 抬高俯视角并把方位角放平一些，让长 x 轴更舒展、柱体立体感更强。
    axis.view_init(elev=24, azim=-62)
    # 使用轻微透视，保持前后柱体的深度关系而避免过度遮挡。
    axis.set_proj_type("persp", focal_length=1.2)
    axis.set_box_aspect((float(len(class_names)), float(y_positions[-1] + depth + 0.70), 5.0))
    axis.grid(True, color="#E1E0D9", linewidth=0.55, alpha=0.75)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#C8C8C8")
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.22, top=0.92)
    figure.savefig(output_dir / "category_similarity_3d.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "category_similarity_3d.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """解析命令行参数并生成图表。"""
    parser = argparse.ArgumentParser(description="绘制 SHWX 25 类与三种大类的三维相似度图")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT / "category_similarity.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regenerate", action="store_true", help="强制重新从相似度矩阵生成 CSV/JSON 后再绘图")
    args = parser.parse_args()

    # 绘图统一从 CSV 读取；仅在 CSV 缺失或强制重新生成时才从相似度矩阵重建数据文件。
    if args.regenerate or not args.csv.is_file():
        names = load_dataset_names(args.dataset_dir)
        class_ids = select_base_classes(names)
        matrix = load_similarity_matrix(args.matrix)
        write_similarity_artifacts(matrix, names, class_ids, args.csv)
    class_names, values = load_similarity_csv(args.csv)
    plot_category_similarity(class_names, values, args.output_dir)
    print(f"已保存: {args.output_dir / 'category_similarity_3d.png'}")
    print(f"已保存: {args.output_dir / 'category_similarity_3d.pdf'}")
    print(f"绘图数据来源: {args.csv}")


if __name__ == "__main__":
    main()
