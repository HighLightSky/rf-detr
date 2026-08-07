# SGA 变体小规模实验（src/scripts/test_sga/）

用于验证 `output/0805-SHWX-SGA-rfdetr/实验报告.md` §五提出的 P0 修复方向：在**短训规模**
（25 epoch、COCO 预训练起步、固定 seed）下 head-to-head 对比各门控/融合变体，确认方向正确后再
用 `src/scripts/train.py` 做全量微调。

## 背景一句话

原版 SGA 的 SGM 门控在目标位置把 SPM 纹理压到 ≈0（框内注意力均值 0.0148），导致小目标召回下降；
P0 修复 = **门控加保底**（下界/残差）+ **融合改残差**（保留 projector 语义基线）。所有变体参数形状
完全一致，可与既有 checkpoint 兼容（resume / from_checkpoint 免改动）。

## 变体一览

| 变体 | use_sga | 门控 | 融合 | attn_bias | 回答的问题 |
|---|---|---|---|---|---|
| `baseline` | False | — | — | 0 | 小规模对照 |
| `spm_only` | True | `ones`（M 固定全 1） | **残差** `feats+0.1*delta` | 0 | SPM 分支 + 残差融合本身是否有用（**无门控**，隔离"门控"轴） |
| `fixed_sga_lb` | True | `lower_bound` `det*(0.5+0.5M)` | 残差 `feats+0.1*delta` | 0 | 下限保底能否追回召回（P0 首选） |
| `fixed_sga_res` | True | `residual` `det+det*M` | 残差 `feats+0.1*delta` | 0 | 更强保底对照 |
| `attn_bias` | True | `product` `det*M`（注意力**直接**门控） | 残差 `feats+0.1*delta` | **+2.0** | 注意力初值≈全通（M≈0.88），能否让注意力自己学成好门控（治本第一版） |

> 说明：`spm_only` 用残差融合，使其与 `fixed_sga_lb` 的唯一差别就是门控（恒 1 vs [0.5,1]），
> 从而干净隔离「门控轴」。`attn_bias` 用 product 直接门控，最贴近「学出来的注意力直接做门控」的目标；
> `+2.0` 使 sigmoid 初始≈0.88（近似全通但可学习），防早期向「目标处抑制」收敛。

## 用法

```bash
# 跑 P0 首选变体
python src/scripts/test_sga/run.py

# 跑指定变体（每个变体独立一次命令）
python src/scripts/test_sga/run.py --variant baseline
python src/scripts/test_sga/run.py --variant spm_only
python src/scripts/test_sga/run.py --variant fixed_sga_lb
python src/scripts/test_sga/run.py --variant fixed_sga_res

# 复用已有输出目录，只重跑 评估+注意力+报告（不训练）
python src/scripts/test_sga/run.py --variant fixed_sga_lb --no-train

# 构建冒烟（seed→构建→SGA 参数量校验→退出，不训练）
python src/scripts/test_sga/run.py --variant fixed_sga_lb --build-only
```

### 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--variant` | 变体名 | `fixed_sga_lb` |
| `--date` | 输出目录日期（MMDD） | 今天 |
| `--seed` | 随机种子（各变体统一固定） | 0 |
| `--epochs` | 短训轮数 | 25 |
| `--resume` | 可选恢复 checkpoint（默认从 COCO 预训练起步） | 空 |
| `--no-train` | 跳过训练，复用已有输出目录 | 关 |
| `--no-eval` | 跳过测试集比赛评估 | 关 |
| `--no-attn` | 跳过 SGM 注意力分析 | 关 |
| `--build-only` | 仅构建+参数形状校验（冒烟） | 关 |

## 输出归档

每个变体输出到 `output/{MMDD}-SHWX-test_sga-<variant>/`：

- `training_config.json` / `metrics.csv` / `checkpoint_*.pth` / `last.ckpt` — 训练栈自动生成
- `test_result.txt` / `confusion_matrix.png` / `FP/` / `FN/` — 测试集比赛评估（`test.py --exp-dir`）
- `attention_stats.txt` / `attention_vis/` — SGM 注意力统计（`analyze_sga.py --checkpoint --gate-mode`）
- `实验报告.md` — 自动拼装：实验设置 + val 动态 + 测试集对比（含对 baseline 的 Δ）+ 注意力机制

## 判断方向（对照报告 §五）

1. all Recall 是否回到 baseline 附近；
2. ship Recall 是否不低于 baseline；
3. vehicle/FSC 是否不再出现 5~10pp 级别的召回缺口；
4. FP 是否仍低于或接近 baseline；
5. 注意力机制：框内**有效门控**应 ≥0.5（不再 ≈0，目标处 SPM 未被关掉）。

**矩阵判读**：
- `spm_only` vs `fixed_sga_lb`：仅门控不同（恒 1 vs [0.5,1]）→ 若 `spm_only` ≥ lb，门控是纯负担，走「去门控」路线；若 lb > `spm_only`，门控有调制价值。
- `attn_bias` vs `fixed_sga_lb`：好初值直接门控 vs 下限保底 → 若 `attn_bias` 框内有效门控（=M）稳定 ≥0.5 且召回不落，证明「注意力本身能学好」，再考虑是否上更重的 `attn_reg`（GT 正则）；若仍塌向 <0.5，则需上 `attn_reg`。

满足以上 → 该变体方向正确，再进全量微调（`src/scripts/train.py` 改 `SGA_GATE_MODE` /
`SGA_FUSION_RESIDUAL` / `SGA_ATTN_BIAS` 常量）。

## 相关改动

本次为支撑该实验，对 SGA 模块做了配置化改造（新增 4 个 `ModelConfig` 字段）：

- `sga_gate_mode`：`product` | `lower_bound` | `residual` | `ones`（默认 `product` = 原版）
- `sga_fusion_residual`：融合是否残差保底（默认 `False` = 原版）
- `sga_residual_gamma`：残差融合系数（默认 0.1）
- `sga_attn_bias`：SGM 注意力 logits 初值偏置（默认 0.0 = 原版从 0.5 起步；`attn_bias` 变体用 +2.0）

默认值复现原版行为，既有配置 / checkpoint / 测试不受影响。旧 checkpoint 不含 `sga_*` 时，
评估脚本（`test.py` / `analyze_sga.py`）自动回退默认值 = 原版 SGA 行为。
