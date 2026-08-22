# RF-DETR SSCL 难例原型改进方案

## 1. 方案目标

在现有 SSCL 原型模式（原型锚定 + 投影头 + 实例正样本）基础上，引入**难例负样本（hard negatives）**：从每张图中选择 Hungarian matching 之外、与任一 GT 的 IoU 落在 [0.1, 0.5] 区间的 unmatched query，按最大前景 logit 取 top-k，作为**额外负样本列**追加进原型模式损失的分母。

SSCL 现有的负样本全是"**类间负样本**"（其他类别原型 + CLIP 语义权重），解决的是"HM/LQS/QHS/MS 类别边界不清"。但它对"**像目标但其实是背景/局部干扰**"的区域（码头、浪花、阴影、局部船体、密集排列）完全没有建模——这正是跨域小样本检测假阳性的主要来源。难例负样本补齐"**前景-背景边界负样本**"这一维度，直接压制误报区域的特征投影贴向类别中心。

参照论文：《Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection》（CVPR 2026，转 Markdown 版见 `docs/参考论文/Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection.md`）。其硬负原型（GT 框抖动，IoU∈[0.1,0.5]）是 1-shot 增益最大的组件（+8 mAP，N=3 优于 N=5）。本项目以其为参照，但负样本来源从"几何抖动"改为"**模型驱动的 unmatched query**"：与 GT 锚定规则一致（IoU 带），同时按模型当前误报点采样（top-k 前景分），自带课程学习性质，且与 SSCL anchor 同处 decoder hidden space，零结构改动。

## 2. 背景：为什么需要难例负样本

### 2.1 SSCL 负样本的盲区

当前原型模式损失（`src/rfdetr/sscl/sscl_loss.py` 的 `_prototype_forward`）：

```
anchor query  → 本类原型 = 正样本
anchor query  → 其他类别原型 = 负样本（语义权重 w_ic = clamp(1 + ρ·S[y_i, c], 1, ω_max)）
```

负样本全部由**类别原型**构成，权重只编码 **CLIP 类别间语义相似度**。两类虚警它管不到：

1. **外观相似背景**：码头、浪花、阴影与舰船在像素上相似，但没有类别身份——语义矩阵对它们"失明"；
2. **部分目标/局部干扰**：舰船局部船体、飞机翼尖等区域与完整目标特征高度重叠，却在分类上属于背景。

这两类恰恰是 `analyze_fp_decomposition.py`（`src/scripts/analyze_fp_decomposition.py`）5 类舰船 FP 成因中"**定位临界**（IoU∈\[0.3,0.5)）"与"**纯背景虚警**"的直接来源。

### 2.2 论文证据

LMP 论文的消融（5-shot，6 域平均）：文本原型基线 40.8 mAP → +类级视觉原型 42.7 → +硬负原型 44.0（最优）。硬负原型的增益随 shot 减少而增大（1-shot +8.0 mAP），因为少样本时正样本稀缺，显式构造难背景提供几乎免费的判别约束。其硬负原型构造：GT 框随机抖动采样 N 个、保留 IoU∈[0.1,0.5]、RoIAlign+GAP 得到特征——**几何锚定**。

### 2.3 本项目选源：unmatched decoder query（模型驱动）

与论文"几何抖动"不同，本项目用 **Hungarian matching 未被匹配到的 decoder query** 作为难例候选，采样规则为几何锚定 + 模型驱动的混合：

| 判据     | 本项目做法                             | 对应论文                    |
| -------- | -------------------------------------- | --------------------------- |
| 几何锚定 | 与任一 GT 的 IoU ∈ [0.1, 0.5]          | jitter 后同带过滤（Eq.2-3） |
| 模型驱动 | 按最大前景 logit 取 top-k              | 无（纯几何）                |
| 特征空间 | decoder hidden space（与 anchor 同源） | RoIAlign(encoder 特征)      |

选择理由：

1. **同空间零适配**：unmatched query 特征就是 `out["hs"]` 的其余行，与 SSCL anchor 同源同空间，过同一个投影头即可，不需要 RoI 适配器；
2. **直接打击实际 FP**：分数最高的 unmatched query 就是模型"差点判成目标"的区域——比随机几何抖动更贴合误报成因；
3. **自带课程学习**：模型进步 → 误报点变化 → 难例自动换新，无需任何调度；
4. **严格性**：IoU 带 [0.1,0.5] 下界剔除纯背景、**上界剔除真实目标的重复检测**（DETR 的 1-to-1 匹配会让部分真实目标成为 unmatched query，绝不能当负样本——这是召回保护的关键）。

## 3. 难例负样本设计

### 3.1 采样规则（逐图）

```
unmatched query（匹配补集）
    → 排除：max IoU(预测框, 任一 GT) 不在 [0.1, 0.5]
    → 分数 = max 前景 logit（pred_logits[:, :-1].max，忽略 background 列）
    → 过滤 score < score_thresh（默认 -2.0，≈ p>0.12，见 §4.1 诊断发现）
    → stable 降序取 top-k（默认 k=3，对齐论文 N=3 经验）
    → hs 对应行 detach
```

> 阈值说明：score_thresh 作用于**原始 logit**（非概率）。DETR 的 focal loss
> 把未匹配 query 的 logit 压到很负，logit>0 等价于 p>0.5（只有真误报才有），
> 阈值 0.0 会把强模型的带内候选几乎全部滤掉导致机制饿死；-2.0（≈p>0.12）
> 保留"弱前景"的次优框，与 LMP 论文"难例纯几何采样、无分数门槛"对齐。

### 3.2 损失公式

原型模式分母扩展为：

```
denom_i = Σ_c exp(w_ic · sim_ic / τ)      ← 类别原型列（语义加权，原样）
        + Σ_k exp(sim_ik / τ)              ← 难例列（权重 1.0，新增）
        + [Σ_j exp(sim_ij / τ)]            ← 实例正样本列（prototype_instance_pos 开启时）
```

分子不变（本类原型 + 实例正样本）。符号表：

| 符号                                   | 含义                                              |
| -------------------------------------- | ------------------------------------------------- |
| u_i                                    | anchor（matched query）在投影空间的 L2 归一化特征 |
| p_c                                    | 类别 c 的原型（投影空间，EMA）                    |
| h_k                                    | 难例（unmatched query 特征，投影空间，detach）    |
| sim_ic = u_iᵀ p_c                      | 余弦相似度（未除温度）                            |
| w_ic = clamp(1 + ρ·S[y_i,c], 1, ω_max) | 类别原型列的语义权重                              |
| τ, k                                   | 温度（0.1）、每图难例数（3）                      |

### 3.3 分母列布局

```
denom_logits = [ C 个原型列 | K 个难例列 | N 个实例正样本列 ]
```

难例列在**实例正样本分支之前**追加（保证实例正样本也出现在难例路径的分母中）。列序由代码中的 `torch.cat` 顺序保证。

### 3.4 设计约束（不可违反）

1. **只进分母**：难例永不作正样本、永不进分子；
2. **权重恒 1.0**：难例无类别身份，语义矩阵不适用；
3. **detach**：难例方向不产生梯度（难例是扰动采样，对它建图没有合理目标）；anchor 与投影头经 `sim_ik` 列正常获得梯度；
4. **不进原型库**：难例刻画"背景/干扰分布"而非类别稳定中心，EMA 进类原型会污染类中心空间——`update_prototypes` 只收 matched features；
5. **不 EMA、无队列**：逐 batch 动态生成、用完即弃。硬例分布随模型演化，EMA 会混合不同训练时期的失败模式；多样性问题（跨 batch 重复模式）留待 v2 用 FIFO 队列 + 新颖性过滤解决；
6. **数值安全**：非有限（NaN/Inf）难例列置 -inf，等价于不参与分母 logsumexp；
7. **单调性**：分子不变、分母单调增 ⇒ `loss(含难例) ≥ loss(不含难例) ≥ 0` 恒成立——难例只会"加强约束"，不会破坏原有语义加权的相对关系。

### 3.5 与 LMP 机制的等价性

LMP 的机制：难例 token 注入注意力 → query 对难例区域高响应 → **focal loss 压向 background**，等于增大训练中的"非类"质量。本项目没有第二分支和注意力注入，但把同一目的移植到已有的对比损失通道：难例列贡献 `exp(sim_ik/τ)`，真正的难例（高相似度）贡献指数级大 → 损失显著惩罚"anchor 投影贴难例"。分母中"非类"质量增大，正是 LMP background channel 的对比损失等价物。此外分类头（focal）本就在 logits 层面把 unmatched query 压向 background，SSCL 难例在特征层面做同一件事，双通道互补。

## 4. 验证方案：难例是否真正代表"难例"

三层验证，**先验证后实验**：

### 4.1 前置离线验证（训练前，几分钟级，必做）

脚本：`src/scripts/semantic_experiments/diag_hard_neg.py`（无需训练）。

加载 0807 checkpoint 双源：检测权重 `checkpoint_best_total.pth` + 投影头/原型库 `last.ckpt`（best checkpoint 不含 SSCL 附加模块），在 SHWX 测试集逐图推理（eval 模式单组 query，直接取 `hs`/`pred_logits`/`pred_boxes`），执行与训练完全一致的难例选择，输出：

| 指标              | 定义                                          | 预期                                   |
| ----------------- | --------------------------------------------- | -------------------------------------- |
| hn_proto_cos      | 难例与类别原型的平均余弦                      | —                                      |
| random_proto_cos  | 随机未匹配与类别原型的平均余弦                | —                                      |
| matched_proto_cos | 真实目标（IoU≥0.5 query）与类别原型的平均余弦 | —                                      |
| hn_vs_random_gap  | hn_proto_cos − random_proto_cos               | **> 0**（难例比随机更贴类中心 = "硬"） |
| hn_vs_matched_gap | hn_proto_cos − matched_proto_cos              | **< 0**（难例比真实目标离类中心更远）  |
| IoU 带填充率      | 带内未匹配候选 / 未匹配总数                   | ≥ 5%                                   |

判据：三个检查全部通过 → 假设成立，启动训练；任一项失败 → 调整采样规则（IoU 带边界 / score_thresh / top_k）后重跑，再决定是否训练。

**诊断发现（0807 checkpoint，2026-08-10 实测）**：

| 阈值              | 难例数（672 图） | 每图难例 | hn_vs_random_gap | hn_vs_matched_gap |
| ----------------- | ---------------- | -------- | ---------------- | ----------------- |
| score_thresh=0.0  | **8**            | 0.01     | +0.0174          | -0.0043           |
| score_thresh=-2.0 | **147**          | 0.22     | +0.0196          | -0.0005           |

- **0.0 阈值在原始 logit 空间几乎饿死机制**（强模型的带内候选前景 logit 几乎全 \<0，batch 64 时每步难例 \<1 个，分母不变，实验无收益）——这是诊断跑出的关键教训，默认值已改为 -2.0；
- -2.0 下难例数提升 18 倍（batch 64 → 约 14 个/步），三判据全过，假设成立；
- `hn_vs_matched_gap ≈ 0`：难例与真实目标在投影空间几乎同样贴类中心——这正是"难"的定义（模型几乎分不出），也解释了论文少样本场景收益最大的机理；召回保护不依赖难例离类中心远，而依赖"难例永不作正样本 + IoU 带排除重复检测"的结构约束。

### 4.2 训练时监控（每 epoch）

HardNegMonitor 按 `sscl_hard_neg_log_interval`（默认 100 步）节流采样，epoch 末输出到 `train/sscl/*`：

```
train/sscl/hn_count             每图平均难例数
train/sscl/hn_fill_rate         IoU 带填充率
train/sscl/hn_proto_cos         难例 vs 原型余弦
train/sscl/random_proto_cos     随机未匹配 vs 原型余弦
train/sscl/matched_proto_cos    matched vs 原型余弦
train/sscl/hn_vs_random_gap     难例-随机差距（应保持 > 0）
train/sscl/hn_vs_matched_gap    难例-matched 差距（应保持 < 0）
```

### 4.3 端到端验证（训练后）

`src/scripts/analyze_fp_decomposition.py <checkpoint>` 的 5 类舰船 FP 成因对比。预期：**"定位临界"与"纯背景虚警"类 FP 下降**，召回不掉（`eval_hardneg.py` 判定表 + FP 置信度直方图佐证）。

## 5. 实验方案

### 5.1 基线事实（重要）

0807 基线实验（`output/0807-SHWX-SSCL-Proj-原型+实例正样本`）**不是 100 轮训练**：它是**从 100 轮全量微调基线（`output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth`）出发的 6 epoch 微调**。新实验照抄同一配方（同起点、同 6 epoch、同全部超参），保证严格可比——**不重新跑 100 轮**。

### 5.2 对照实验矩阵（双臂）

| 实验   | 输出目录                                     | 起点                 | SSCL 配置                           | 难例 k | 验证目标                       |
| ------ | -------------------------------------------- | -------------------- | ----------------------------------- | ------ | ------------------------------ |
| 基线   | `output/0807-SHWX-SSCL-Proj-原型+实例正样本` | 0805 基线 checkpoint | 原型+投影头(128)+实例正样本，λ=0.02 | 关     | 复用 checkpoint，**不重跑**    |
| 实验一 | `output/0810-SHWX-SSCL-Proj-HardNeg-k3`      | 同上                 | 与基线完全一致                      | **3**  | 难例有效性                     |
| 实验二 | `output/0810-SHWX-SSCL-Proj-HardNeg-k5`      | 同上                 | 与基线完全一致                      | **5**  | k 敏感性（论文：N=3 优于 N=5） |

统一配方（与 0807 完全一致）：EPOCHS=6、BATCH=32×grad_accum 2、LR=1e-5（恒定）、WARMUP=0、Mosaic=0、multi_scale=False、freeze="conservative"（仅 decoder 最后一层 + class_embed 可训练）、use_ema、λ_sscl=0.02、τ=0.1、ρ=0.3、ω_max=2.0、anchor/confusing=[0,1,2,3]、start_epoch=0。难例参数：k 为实验变量，score_thresh=-2.0（原始 logit，见 §4.1 诊断发现）。

### 5.3 训练脚本开关（src/scripts/train_sscl_hardneg.py）

```python
SSCL_PROTOTYPE_ENABLED = True
SSCL_PROJECTION_ENABLED = True
SSCL_PROJECTION_DIM = 128
SSCL_PROTOTYPE_INSTANCE_POS = True
SSCL_HARD_NEG_ENABLED = True  # 本次实验变量
SSCL_HARD_NEG_TOPK = int(os.environ.get("SSCL_HARD_NEG_TOPK", "3"))  # 双运行：3 / 5
SSCL_HARD_NEG_SCORE_THRESH = -2.0  # 原始 logit 下限（≈p>0.12，见 §4.1 诊断发现）
SSCL_HARD_NEG_LOG_INTERVAL = 100
OUTPUT_DIR = f"output/0810-SHWX-SSCL-Proj-HardNeg-k{SSCL_HARD_NEG_TOPK}"
```

```bash
uv run python src/scripts/train_sscl_hardneg.py        # k=3
SSCL_HARD_NEG_TOPK=5 uv run python src/scripts/train_sscl_hardneg.py   # k=5
```

### 5.4 评估指标（不能只看 val mAP）

`src/scripts/semantic_experiments/eval_hardneg.py` 三向对照输出：

- 整体/舰船/飞机/车辆 P/R/F1、**fp_ship**；
- HM/LQS/QHS/MS 逐类 P/R（SSCL 的主战场）、FSC recall；
- 相对基线判定：舰船 FP ≤ 基线 + 5、总体 Recall ≥ 基线 − 0.01（不伤召回）、舰船 F1 > 基线；
- FP 成因分解（`analyze_fp_decomposition.py`）与 `train/sscl/*` 曲线（`metrics.csv`/TensorBoard）。

### 5.5 预期收益

- **主战场是 FP 而非 mAP 大头**：若当前瓶颈在误检（密集场景、域差异大），收益明显；若瓶颈在漏检，收益偏小——先用 `analyze_fp_decomposition.py` 扫一眼当前 FP 占比再判断；
- 稀有类（HM/LQS 等少数类）AP 提升应领先基类（少样本时难例增益最大）；
- 增益随"有效 shot 数"减少而增大的趋势可以检验（与论文 1-shot +8.0 > 5-shot +3.6 > 10-shot +2.1 的趋势对齐，幅度不期待一致）。

## 6. 实现细节

### 6.1 修改文件表

| 文件                                            | 动作 | 内容                                                                                                                                             |
| ----------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/rfdetr/config.py`                          | 改   | 4 个新字段（`sscl_hard_neg_enabled`/`_topk`/`_score_thresh`/`_log_interval`）+ `validate_sscl_hard_neg` 校验器（难例依赖原型模式；topk ≥ 1）     |
| `src/rfdetr/sscl/hard_neg_selection.py`         | 新增 | `select_hard_negatives_for_image` 纯函数选择器（逐图：排除 matched → IoU 带 → top-k → detach；附带随机未匹配对照特征与统计）                     |
| `src/rfdetr/sscl/sscl_loss.py`                  | 改   | `forward`/`_prototype_forward` 增加可选参 `hard_neg_features`；分母追加难例列（NaN/Inf 守卫）；新增 `hardness_stats` 诊断方法                    |
| `src/rfdetr/sscl/hard_neg_monitor.py`           | 新增 | `HardNegMonitor` 训练监控累加器（epoch 末输出 `train/sscl/*`）                                                                                   |
| `src/rfdetr/sscl/__init__.py`                   | 改   | 导出新模块                                                                                                                                       |
| `src/rfdetr/training/module_model.py`           | 改   | `_setup_sscl` 建监控；`_select_hard_negatives` 批量选择；`_sscl_loss_callback` 接线（返回字典仍只含 `loss_sscl`）；`on_train_epoch_end` 冲刷监控 |
| `src/scripts/train_sscl_hardneg.py`             | 新增 | 双臂训练脚本（env 参数化 k）                                                                                                                     |
| `src/scripts/semantic_experiments/diag_hard_neg.py`         | 新增 | 前置硬度验证脚本                                                                                                                                 |
| `src/scripts/semantic_experiments/eval_hardneg.py`          | 新增 | 三向对照评估脚本                                                                                                                                 |
| `tests/models/test_sscl_hard_neg.py`            | 新增 | 损失级单测（9 项）                                                                                                                               |
| `tests/models/test_hard_neg_selection.py`       | 新增 | 选择器单测（7 项）                                                                                                                               |
| `tests/training/test_sscl_hard_neg_callback.py` | 新增 | 回调级单测 + config 校验（9 项）                                                                                                                 |

### 6.2 两个关键工程细节

1. **checkpoint 双源加载**：`checkpoint_best_total.pth` 只含检测模型权重（`best_model.py` 从 unwrapped model 取 state_dict）；**投影头 + 原型库在 `last.ckpt`**（`sscl_loss.*` 键）。诊断脚本必须双源加载，键缺失时直接报错。
2. **难例列在实例正样本分支之前追加**：保证实例正样本也出现在难例路径的分母中，列序 `[C, K, N]`。

### 6.3 冻结策略

不变：`sscl_freeze_strategy="conservative"` 只解冻 decoder 最后一层 + norm + class_embed。难例特征来自 decoder 最后一层输出（冻结层之上），梯度经 `sim_ik` 列回流到**可训练部分**（decoder 最后层 + 投影头），backbone/encoder 不受影响。

## 7. 测试与风险

### 7.1 单测清单（TDD，已实现）

- 损失级：零难例与基线完全一致；难例使 loss ≥ 基线；NaN/Inf 行守卫（≡ 删除该行）；难例 detach（`hn.grad is None`，anchor 与投影头有梯度）；与实例正样本组合；实例模式忽略难例；`hardness_stats` 值域与边界。
- 选择器级：IoU 带过滤（0.25 入选、0.0/1.0 排除、边界 0.1/0.5）；matched 排除；top-k 按最大前景 logit 且忽略 background 列、stable 并列取小索引；score_thresh；空 GT 不崩；随机对照特征排除 matched 且 seed 确定性；全 matched 无候选。
- 回调级：返回字典只含 `loss_sscl`；难例进分母且原型库只收 matched；禁用时与基线一致；验证态不选难例不喂监控；监控节流（50/100）；config 校验（无原型模式报错、topk=0 报错）。

运行：

```bash
uv run --no-sync pytest tests/models/test_sscl_hard_neg.py tests/models/test_hard_neg_selection.py \
    tests/training/test_sscl_hard_neg_callback.py -q -o addopts=""
```

### 7.2 风险评估

| 风险                                               | 缓解                                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| K 波动致 loss 抖动（每图难例数不定，分母大小波动） | top-k 上限 + score 下限；`loss ≥ 基线` 不变量作为哨兵；观察 `train/loss_sscl` 与 `hn_count` 曲线 |
| IoU 0.5 上界 vs 匹配阈值边界                       | 上界低于匹配阈值、matched query 按构造排除；[0.1,0.5] 带过滤是召回保护的承重墙                   |
| 填充率过低 → 难例不常见                            | `hn_fill_rate` 诊断；< 5% 时放宽带边界或降低 score_thresh                                        |
| NaN/Inf 毒化分母                                   | 列级 `isfinite` 守卫（置 -inf）+ 选择侧稳健性 + 单测                                             |
| 诊断脚本 checkpoint 双源加载失败                   | 显式断言键存在，缺失即报错（见 §6.2.1）                                                          |
| 难例与正样本边界（误伤召回）                       | IoU 带排除重复检测 + 端到端 Recall 判定约束（≥ 基线 − 0.01）                                     |

### 7.3 实施顺序

1. config 字段 + 校验器 → 2. 选择器 → 3. 损失改造 → 4. 监控 → 5. 接线 → 6. 单测全绿 → 7. 训练/诊断/评估脚本 → 8. 本文档 → 9. `pre-commit run --all-files` → 10. 前置验证（diag_hard_neg.py）→ 11. 双臂训练 → 12. 评估对照。
