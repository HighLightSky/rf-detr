# RF-DETR 分类损失均衡化 —— 最终方案（融合专家修订）

> 本文是 `RF-DETR分类损失均衡化改进方案.md`（专家修订版）的落地定稿：明确"到底怎么
> 优化损失函数"、其理由与原理，以及代码实现方案与实验命令。专家修订版保留原样，
> 本文是其可执行的最终结论。

## 1. 最终决策（一句话版）

**首发只做 P0 正样本类均衡 IA-BCE；P0 不足时叠加 P1 居中截断 Logit Adjustment；
matcher 分类成本首轮不动；每轮训练后必须用阈值重标定脚本重搜逐类阈值。**
不采用：完整强 LA、Seesaw/EQL 式负梯度调节、inverse-frequency 全量加权（理由见 §2）。

## 2. 理由与原理

### 2.1 问题定位

0807 基线三大类宏平均 Recall=0.8506 / FDR=0.2420（目标 Recall≥0.85、FDR≤0.20），
瓶颈集中在极小样本类：HM（FDR=0.64）、LQS（FDR=0.50）、MS（FDR=0.23）、FSC（FDR=0.31）。
探针证据链表明表示空间已可分，**瓶颈在分类头的训练目标与分数校准被类别频率牵引**。

### 2.2 当前损失缺什么

IA-BCE（`criterion.py::loss_labels`）已含两个维度：

- **难度维度**：负样本权重 `prob^γ`（γ=2，focal 式难负例抑制）；
- **质量维度**：正样本权重 `t = prob^α · pos_iou^(1-α)`（α=0.25）。

缺失的正是**类别频率维度**：HM/LQS 正样本对分类头参数的累计梯度远小于 MS/QHS。
改进只补这一块，不破坏 IA-BCE 已有的难例/质量机制。

### 2.3 为什么不能用完整 Logit Adjustment（专家修订核心）

RF-DETR 分类头是 **sigmoid IA-BCE + 一对一匹配**，不是 softmax 单标签：

1. softmax 中加同一常数不改变相对概率；sigmoid 中 logit 绝对值直接改变 objectness。
    `τ·log(π_c)` 对多数类为负 → 全体 logit 负漂移。
2. 类别偏置不仅影响正样本，还影响大量 unmatched query 的负样本损失；对 HM/LQS 用
    强负先验偏置会降低其负样本惩罚，推理时再减先验抬高分数 → 小类 FDR 暴涨。
3. 同时改 matcher 成本会让低质量 query 被分给稀有类，召回升、虚警也升。

### 2.4 论文验证依据 → 本方案映射

| 候选（论文验证）                                      | 机制                                                                                 | 本方案角色                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Class-Balanced Loss（Cui et al., CVPR 2019）          | 按有效样本数加权 `(1-β)/(1-β^{n_c})`，或幂律 `(N_ref/N_c)^β`                         | **P0 理论来源**：类权重只乘正样本 slot                                                                       |
| Logit Adjustment（Menon et al., ICLR 2021 Spotlight） | 训练 `logits+τ·log π`，推理 `argmax(f−τ·log π)`；对平衡错误 Fisher 一致              | **P1 理论来源**，仅取"居中截断"安全化形式                                                                    |
| Bag of Tricks 幂律 LA（Zhang et al., AAAI 2022）      | `logits−τ·log(π_c^α)` 软化先验                                                       | P1 的 τ 扫描与先验软化依据                                                                                   |
| Focal Loss（Lin et al., ICCV 2017）                   | `(1-p)^γ`                                                                            | 已在 IA-BCE 中                                                                                               |
| Seesaw Loss（Wang et al., CVPR 2021，LVIS 长尾检测）  | mitigation `(N_j/N_i)^p` 降多数类对稀有类的负梯度 + compensation `(σ_j/σ_i)^q` 防 FP | **不首发**：mitigation 正是"主动降低稀有类负样本惩罚"，与 FDR 保护冲突；P0/P1 无效且有负样本压制诊断时再评估 |
| Equalization Loss v1/v2（Tan et al., CVPR 2020/2021） | 稀有类负梯度抑制/梯度引导重加权                                                      | 同上，延后                                                                                                   |

### 2.5 两个公式（实现即采用此式）

**P0 正样本类均衡（CB 风格幂律权重）**

```python
n_c_eff = max(n_c, n_min)                     # n_min=10：防极端小样本类极端权重
w_c = clamp((N_ref / n_c_eff) ** beta, 1.0, w_max)   # N_ref 默认 sqrt(N_max·N_min)
w_c[非 target_classes] = 1.0                  # 首发只对 HM/LQS
pos_weights[positive_index] *= w_c[target_classes_o]  # 只改正样本，负样本不动
```

**P1 居中截断 Logit Adjustment（LA 安全化）**

```python
raw_bias = log(π_c + eps)                     # π_c = n_c / Σn
centered_bias = raw_bias - raw_bias.mean()    # 居中：不整体偏移 objectness
logit_bias = τ · clamp(centered_bias, -clip, clip)
adjusted_logits = src_logits + warmup_factor · logit_bias   # warmup 1 epoch 线性升
```

推理侧对照：`raw / −0.5·bias / −1·bias` 三种，各配阈值搜索。

### 2.6 预期与止损

温和均衡有机会带来 **1-4pp 宏 FDR 收益**；强均衡方差大。出现任一情况即停止加码：
总 Recall\<0.85、总 FDR 劣于基线、MS/QHS recall 掉 3pp+、HM/LQS FP 增而 TP 不增
（说明不是类别先验问题，转向难例/数据/matcher 诊断）。

## 3. 实现方案

### 3.1 新增配置项（TrainConfig，默认全关）

```python
class_balance_enabled: bool = False
class_balance_counts_path: str | None = None  # stat_class_counts.py 生成的 JSON
class_balance_beta: float = 0.25
class_balance_max_weight: float = 3.0
class_balance_min_count: int = 10
class_balance_ref_count: float | None = None  # None → 自动 sqrt(N_max·N_min)
class_balance_target_classes: list[int] | None = None

logit_adjustment_enabled: bool = False
logit_adjustment_tau: float = 0.1
logit_adjustment_bias_clip: float = 1.0
logit_adjustment_warmup_epochs: float = 1.0
```

### 3.2 代码落点

| 文件                                               | 改动                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/rfdetr/config.py`                             | TrainConfig 新增上述 11 项（自动透传 namespace）                                                                                                                                                                                                                                                                                                                                  |
| `src/rfdetr/models/criterion.py`                   | `__init__` 新参数 + `_build_class_balance_buffers` 预计算权重/bias buffer（`persistent=False`）；`loss_labels` ia_bce 分支：P0 在 `pos_weights[tuple(pos_ind)]` 赋值后乘 `class_balance_weights[target_classes_o]`；P1 用**局部** `adjusted_logits = src_logits + warmup·bias` 计算损失（绝不原地改 `outputs["pred_logits"]`，会污染推理输出）；`set_la_warmup_factor()` 每步更新 |
| `src/rfdetr/models/lwdetr.py`                      | `build_criterion_and_postprocessors` 读配置、加载 counts JSON（每个 rank 同一文件，DDP 一致）                                                                                                                                                                                                                                                                                     |
| `src/rfdetr/training/module_model.py`              | `_compute_train_losses` 按 `global_step / (warmup_epochs × steps_per_epoch)` 设 warmup                                                                                                                                                                                                                                                                                            |
| `src/scripts/stat_class_counts.py`                 | 新增：训练集类别统计 → `class_counts.json`（含 n_ref）                                                                                                                                                                                                                                                                                                                            |
| `src/scripts/train_sscl_class_balance.py`          | 新增：0807 同配方 + 均衡开关，训练前自动统计类别数                                                                                                                                                                                                                                                                                                                                |
| `src/scripts/calibrate_thresholds.py`              | 新增：推理一次 → 离线坐标上升搜逐类阈值（含 LA 推理侧 bias 对照）                                                                                                                                                                                                                                                                                                                 |
| `src/scripts/test.py`                              | 可选 `LOGIT_ADJUSTMENT_BIAS_PATH/K/TAU/CLIP` 常量（默认 None）                                                                                                                                                                                                                                                                                                                    |
| `tests/models/test_criterion.py`、`tests/scripts/` | P0/P1 数值与回归测试、统计脚本测试、选优逻辑测试                                                                                                                                                                                                                                                                                                                                  |

### 3.3 关键实现要点

- 权重/bias 预计算为 buffer，device 随 `.to(device)` 迁移，DDP/checkpoint 行为一致；
- P1 必须用局部 `adjusted_logits`（dtype 同步 `.to(src_logits.dtype)`，防 bf16 提升）；
- aux outputs 走同一 `loss_labels`，P0/P1 自动一致生效；
- 类别统计从训练集 labels 自动生成并随实验目录保存，不手写（数据变动后自动跟随）。

## 4. 实验命令（用户自跑）

```bash
# ① 生成类别统计（训练脚本会自动做，独立跑可审计）
uv run python src/scripts/stat_class_counts.py \
    /home/liu/wzt/datasets/SHWX-dataset-dict/labels/train \
    output/<实验目录>/class_counts.json 25 "HM,LQS,QHS,MS,A1_SU-35,...,FSC"

# ② E1：P0 β=0.25, w_max=3, HM/LQS（脚本默认即为 E1，运行前确认 DATASET_DIR 与 OUTPUT_DIR 常数）
uv run python src/scripts/train_sscl_class_balance.py

# ③ E2：改脚本常数 OUTPUT_DIR/BETA=0.5/MAX_WEIGHT=5.0 后再跑一次
# ④ E3：E1 + LA（LOGIT_ADJUSTMENT_ENABLED=True, TAU=0.1, CLIP=1.0, WARMUP=1.0）

# ⑤ 训练后：逐类阈值重标定（必做，loss 变了分数分布必变）
uv run python src/scripts/calibrate_thresholds.py output/<实验目录>/checkpoint_best_total.pth

# ⑥ 把重标定输出的 CLASS_CONF_THRESHOLDS 贴入 test.py 后跑标准评估
uv run python src/scripts/test.py output/<实验目录>/checkpoint_best_total.pth

# ⑦（E3 附加）LA 推理侧对照：先跑 ⑤ 不带 --bias-json 得阈值，
#    再跑 ⑥ 时在 test.py 设 LOGIT_ADJUSTMENT_BIAS_PATH=<实验目录>/class_counts.json、BIAS_K=0.5/1.0
```

## 5. 首轮实验矩阵与判定

| 编号 | 方案                            | 参数                                      | 判定                                                                         |
| ---- | ------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| E0   | 0807 checkpoint 复评 + 阈值重搜 | 只跑 ⑤                                    | 基线与校准上限（实测：仅阈值重标定**无法**同时达到门槛，佐证训练侧改动必要） |
| E1   | P0                              | β=0.25, w_max=3, targets=[0,1]            | 小类 TP 是否出现、FDR 是否降                                                 |
| E2   | P0 加强                         | β=0.5, w_max=5, targets=[0,1]             | 更强均衡上限、FDR 失控阈值                                                   |
| E3   | E1 + P1                         | τ=0.1, clip=1.0, warmup=1ep + 推理 k 对照 | logit bias 增量收益                                                          |

硬门槛（留 0.5pp 余量）：**总宏 Recall ≥ 0.855 且 FDR ≤ 0.195**；MS/QHS recall 不比
0807+阈值低 3pp；飞机大类 FDR 不显著变差；HM/LQS 至少一类 TP/F1 明显改善。
E1 有效则不加码；E1/E2 只提 TP 但 FDR 高 → 优先调阈值；E1/E2 无效 → 回头诊断
matcher 与框质量。

## 5.5 实验结果记录（0811 E1/E2）

> 基线对照：`output/0805-SHWX-data-expand-rfdetr-baseline`（SSCL off）。
> 同配方：RFDETRMedium 640、6 epoch、从 0805 基线微调、SSCL 原型+投影头+实例正样本全开。

**固定 0.25 阈值（同口径公平对比）：**

| 指标         | 0805 基线 | 0807   | E1 (β=.25,w=3) | E2 (β=.5,w=5) |
| ------------ | --------- | ------ | -------------- | ------------- |
| 总宏 Recall  | 0.8638    | 0.8759 | **0.9073**     | 0.9076        |
| 总宏 FDR     | 0.3505    | 0.3217 | 0.3470         | 0.3656        |
| HM TP/FP/FN  | 3/7/2     | 4/7/1  | **5/30/0**     | 5/45/0        |
| LQS TP/FP/FN | 6/6/1     | 6/6/1  | **7/14/0**     | 7/20/0        |
| MS TP/FP     | 377/187   | -      | 371/153        | 373/155       |
| FSC TP/FP    | 64/69     | -      | 64/50          | 64/56         |

**校准后最优点（calibrate_thresholds.py 字典序选优）：**

| 方案          | Recall | FDR    | 阈值 (HM/LQS/QHS/MS/FSC) |
| ------------- | ------ | ------ | ------------------------ |
| 0807+手调阈值 | 0.8506 | 0.2420 | 0.25/0.25/0.50/0.30/0.35 |
| E1+校准       | 0.7869 | 0.1949 | 0.60/0.45/0.60/0.60/0.30 |
| E2+校准       | 0.7754 | 0.1941 | 0.50/0.60/0.60/0.60/0.40 |

**E1 的 Pareto 前沿关键点（Recall≈0.85 处）：** 0.8535/0.2664（阈值 0.25/0.45/0.60/0.60/0.30）
——**同 Recall 下 FDR 高于 0807 手调阈值（0.2420）**。

**E1 ship FP 成因分解（analyze_fp_decomposition.py）：**
纯背景虚警 155 (56.2%)、船类小类混淆 74 (26.8%)、定位临界 30 (10.9%)、重复检测 17 (6.2%)；
纯背景虚警置信度集中在 \[0.25,0.60)（104+38 个），与 ship TP 的低分尾（66+115 个）重叠。

**结论（2026-08-11，含隔离实验修正）：**

1. **灰度数据增广是主因**（关键修正）：E1/E2 训练前数据集已加入 HM/LQS 灰度副本
    （HM 6→58、LQS 15→85），E1/E2 与 0807 的差异 = 数据增广 + P0 两个变量混杂。
    隔离实验 `无均衡-E0`（同数据集、P0 关）固定 0.25 下为 0.9081/0.3417，**与 E1
    （0.9073/0.3470）几乎相同** → 召回升幅与虚警升幅由数据增广贡献，**P0 独立贡献 ≈ 0**。
2. **灰度增广有效但带泄漏**：HM/LQS 样本量 9×/5.7× 使 FN 归零（Recall 0.8759→0.9081）；
    但 89% 的 HM 图为灰度 → "灰度≈HM/LQS" 泄漏特征 → 测试 RGB 低饱和背景被触发，
    HM FP 7→20（无 P0 时已发生）。修复方向：随机灰度化（50%）或全局灰度增广。
3. **E2 无增益**：E1→E2 只增 HM/LQS FP（30→45、14→20），TP 不涨。
4. **校准后三方重合**：无均衡 0.7719、E1 0.7869、E2 0.7754（均 @FDR≈0.195）；
    P0 未改善分数分离度。宏平均门槛仍未达成，瓶颈 = ship 类（尤其 MS）TP/FP 分数重叠 + 纯背景虚警。
5. **下一步**：优先修复灰度泄漏（随机灰度化）重训；虚警侧叠加难例负样本
    （sscl_hard_neg）或 objectness 门控（QNorm-Obj）；P0/P1 暂缓
    （如需再试，权重应基于未增广 counts 计算）。详见
    `docs/改进方案-SSCL/RF-DETR分类损失均衡化实验报告-0811.md`。

## 6. 参考文献

1. Menon, Jayasumana, Rawat, Jain, Veit, Kumar. "Long-tail learning via logit adjustment." **ICLR 2021**（Spotlight）.
2. Cui et al. "Class-Balanced Loss Based on Effective Number of Samples." CVPR 2019.
3. Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
4. Wang et al. "Seesaw Loss for Long-Tailed Instance Segmentation." CVPR 2021.
5. Tan et al. "Equalization Loss for Long-Tailed Object Recognition." CVPR 2020; "Equalization Loss v2." CVPR 2021.
6. Zhang et al. "Bag of Tricks for Long-Tailed Visual Recognition." AAAI 2022.
7. Chen et al. "Balanced Hierarchical Contrastive Learning with Decoupled Queries..." CVPR 2026（见 `docs/参考论文/`）。
