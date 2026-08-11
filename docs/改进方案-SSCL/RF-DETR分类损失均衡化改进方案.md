# RF-DETR 分类损失均衡化改进方案评估与实验设计

## 0. 客观结论

这套 loss 改进方向**值得做**，因为它正好作用在 `output/0807-SHWX-SSCL-Proj-原型+实例正样本` 后仍然薄弱的环节：SSCL/原型/投影头主要改善 query 表示空间，而比赛指标和现有探针结果都指向**分类头决策边界仍被类别频率牵引**。在 HM、LQS 这类极小样本类别上，分类损失均衡化确实有机会带来收益，尤其是减少它们被 QHS/MS 吸走、或在固定阈值下完全出不来的情况。

但原方案里把 **Logit Adjustment** 写成“理论最优、训练时加、推理时减、预期 FDR 必降”的表述过强。RF-DETR 当前分类头是 **多标签 sigmoid + IA-BCE + DETR 一对一匹配**，不是普通 softmax 单标签分类。这个差异很关键：

- softmax 中加同一个常数不改变类别相对概率；sigmoid 中每个类别的 logit 绝对值会直接改变 objectness 分数。
- 类别先验偏置不仅影响正样本，还会影响大量 unmatched query 上的负样本损失。
- 如果对 HM/LQS 使用过强负先验偏置，训练时会降低这些类别负样本的惩罚，推理时再减先验又会抬高它们的分数，容易把 QHS/MS 或背景推成 HM/LQS，导致小类 FDR 暴涨。
- 如果同时改 matcher 成本，少数类可能拿到更多匹配 query，但也可能让框质量较差的 query 被分给稀有类，召回上升、虚警也上升。

因此，推荐结论是：

> **可以期待小样本收益，但不要直接上完整强 LA。第一优先级应是“正样本侧类均衡 IA-BCE”，第二优先级才是“居中、截断、可退火的 logit bias”。最优方案必须通过 0807 checkpoint 同配方消融 + 阈值重标定取得。**

## 1. 针对三个问题的判断

### 1.1 是否可能带来小样本收益

**可能，而且收益点比较明确。** 0807 方案已经把表示学习做到一定程度，继续增强 SSCL 的边际收益可能有限；如果同一特征上的均衡探针明显优于不均衡探针，说明模型不是“看不懂 HM/LQS”，而是最终分类边界和分数校准偏向多数类。

预期最可能改善的现象：

- HM/LQS 在固定阈值下 TP 从 0 或很低变为可检出。
- HM/LQS 被 MS/QHS 吸收的 FN 减少。
- matched query 上的小类分类准确率提升。
- 小类 AP 或 macro-F1 有提升。

但收益不应按“探针准确率 0.80-0.90”线性外推到检测指标。检测最终还受 Hungarian 匹配、box 质量、top-k 竞争、阈值和重复预测影响。合理预期是：**温和均衡有机会带来 1-4pp 的 macro FDR/Recall 组合收益；强均衡可能收益更大，但不稳定。**

### 1.2 会不会对多数类造成大影响

**会，尤其是 MS/QHS。** 它们是 HM/LQS 最直接的易混多数类，也是稀有类分数抬高后最容易被“挤掉”或“误改名”的类别。

主要风险有三类：

| 风险           | 表现                                   | 触发条件                              | 控制方式                                    |
| -------------- | -------------------------------------- | ------------------------------------- | ------------------------------------------- |
| 多数类召回下降 | MS/QHS 的 TP 变少，FN 增多             | 稀有类 bias 太强，边界过度让给 HM/LQS | β/τ 从小到大，bias 截断，监控 MS/QHS recall |
| 稀有类虚警暴涨 | HM/LQS FP 增多，FDR 上升               | 推理侧直接减去完整 `τ log π_c`        | 推理校准单独搜索，逐类阈值联调              |
| top-k 资源挤占 | 低质量稀有类分数进入 top-k，影响其他类 | 所有 query 的稀有类分数被整体抬高     | 使用正样本侧加权优先，logit bias 居中/截断  |

在比赛宏平均口径下，少数类收益可以抵消一部分多数类损失，但前提是多数类不要出现 5-10pp 级别的 precision/recall 回落。对于当前目标（Recall ≥ 0.85、FDR ≤ 0.20），更应把多数类影响作为硬约束，而不是事后解释。

### 1.3 如何取得最优方案

最优方案不应该只搜一个 `τ`。建议按“先低成本判断方向，再训练侧消融，再阈值重标定”的顺序：

1. **固定 0807 基线复评**：用同一权重、同一 test 脚本、同一阈值导出 per-class TP/FP/FN、混淆矩阵、raw logits 或至少 score。
2. **先做推理侧零训练校准**：在不训练的情况下，对 HM/LQS/QHS/MS 做 class bias/threshold 网格搜索，判断小类是否存在“只差分数校准”的收益上限。
3. **训练侧首选正样本类均衡 IA-BCE**：只增强稀有类正样本梯度，不主动降低其负样本惩罚，优先保护 FDR。
4. **再做 capped logit adjustment**：使用居中、截断、warmup 的 logit bias，避免 sigmoid 绝对分数整体漂移。
5. **最后才动 matcher**：只有当诊断显示 HM/LQS 主要失败于“匹配不到 query”时，再把轻量 bias 加入 matcher 分类成本。
6. **每个训练方案都重新搜逐类阈值**：loss 改完后分数分布一定会变，沿用 0807 阈值可能不是公平比较。

## 2. 背景与问题定位

### 2.1 比赛口径

SHWX 25 类会汇总为舰船、飞机、车辆三大类。若按大类内小类宏平均再汇总，每个小类对最终指标的权重远高于它在训练样本中的频率权重。HM 只有个位数样本、LQS 只有十余样本，而 MS 是千级样本，这会让“按样本数投票”的分类损失天然偏向多数类。

### 2.2 0807 对比基础

本方案建议基于以下实验继续做严格对比：

```text
output/0807-SHWX-SSCL-Proj-原型+实例正样本
```

该实验配置要点：

- 模型：`RFDETRMedium`
- 输入分辨率：640
- 训练轮数：6 epoch
- 预训练权重：`output/0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth`
- SSCL：开启
- 原型库：开启
- 投影头：开启，`projection_dim=128`
- 实例正样本：开启
- 关注类：`HM/LQS/QHS/MS`

已有记录中，0807 验证日志显示 HM/LQS/MS/QHS 的 AP 仍明显低于飞机多数类别；方案原记录还给出了固定阈值口径：

| 对照               | Recall |    FDR | 备注                           |
| ------------------ | -----: | -----: | ------------------------------ |
| 0807 全类阈值 0.25 | 0.8759 | 0.3217 | 召回达标，FDR 不达标           |
| 0807 + 逐类阈值    | 0.8506 | 0.2420 | 接近召回下限，FDR 仍差约 4.2pp |
| 0809 全量配方      | 0.8533 | 0.2262 | 仍未过 0.20                    |

这些数字说明：当前不是单纯“检测不到”，而是 **召回和虚警之间的校准空间非常窄**。loss 均衡如果能把 HM/LQS 的分类边界拉正，同时不制造大量新 FP，就有可能补上最后缺口。

### 2.3 探针证据链

已有探针实验给出的核心判断是：表示空间已经有可分性，瓶颈更靠近分类目标和分数校准。

| 结构部件           | 状态     | 证据                                           |
| ------------------ | -------- | ---------------------------------------------- |
| 骨干 ROI 特征      | 可分     | QHS/MS 等多数船类探针表现较高                  |
| decoder query 特征 | 可分     | 训练集内 query 特征探针表现高                  |
| SSCL 投影空间      | 可分     | 投影空间探针与 hidden 空间接近                 |
| 分类头训练目标     | 主要瓶颈 | 同一特征上，均衡训练头明显优于不均衡训练头     |
| EMA 原型估计       | 有风险   | 原型分类弱于同空间探针，说明原型质量不是唯一解 |

结论可以保留，但措辞应更准确：

> 当前证据支持“分类边界存在类别频率偏置”，但还不能证明“完整 Logit Adjustment 一定优于其他均衡损失”。在 RF-DETR 的 sigmoid IA-BCE 场景，正样本加权通常是更稳的第一实验。

## 3. 当前分类损失剖析

RF-DETR 默认分类损失为 `ia_bce_loss=True`，核心实现位于 `src/rfdetr/models/criterion.py` 的 `SetCriterion.loss_labels`。它不是 softmax CE，而是对 `(B, Q, C)` 的每个 query/class 独立做 sigmoid 型损失：

```python
prob = src_logits.sigmoid()
pos_weights = torch.zeros_like(src_logits)
neg_weights = prob**gamma

t = prob[positive_index].pow(alpha) * pos_ious.pow(1 - alpha)
t = torch.clamp(t, 0.01).detach()

pos_weights[positive_index] = t
neg_weights[positive_index] = 1 - t
loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
```

它已经包含两个维度：

- **难度维度**：负样本权重 `prob^gamma`，类似 focal 的 hard negative 抑制。
- **质量维度**：正样本权重同时看分类置信度和 IoU。

缺失的是：

- **类别频率维度**：不同类别正样本出现次数相差极大，HM/LQS 对分类头参数的累计梯度远小于 MS/QHS。
- **类间混淆维度**：HM/LQS/QHS/MS 的错误很多是舰船内部错分，不只是前景/背景错分。

因此，loss 改进应尽量只补“类别频率”这一块，不要破坏 IA-BCE 已经处理好的难例和质量机制。

## 4. 方法对比与推荐优先级

| 方法                                | 适配 RF-DETR IA-BCE | 小样本收益潜力 | 多数类风险 |        建议优先级 |
| ----------------------------------- | ------------------- | -------------: | ---------: | ----------------: |
| 正样本类均衡 IA-BCE                 | 高                  |          中-高 |      低-中 |                P0 |
| 居中截断 Logit Adjustment           | 中                  |          中-高 |      中-高 |                P1 |
| 推理侧 class bias/阈值校准          | 高                  |             中 |       可控 | P0 诊断 + P2 联调 |
| matcher 分类成本均衡                | 中                  |             中 |         高 |                P2 |
| 完整 inverse-frequency class weight | 中                  |             高 |         高 |        不建议首发 |
| 继续加大 SSCL 强度                  | 高                  |          低-中 |         中 |            非主线 |

### 4.1 P0：正样本类均衡 IA-BCE（主推荐）

目标：让 HM/LQS 的**正样本监督**在每个 epoch 中更有存在感，但不显著放松这些类在背景/unmatched query 上的负样本约束。

推荐形式：

```python
w_c = clamp((N_ref / max(N_c, n_min)) ** beta, 1.0, w_max)
pos_weights[positive_index] *= w_c[target_classes_o]
```

建议初始值：

| 参数     | 建议                                                           |
| -------- | -------------------------------------------------------------- |
| `N_ref`  | 使用中位类别样本数或 `sqrt(N_max * N_min)`，不要直接用 `N_max` |
| `n_min`  | 10 或 20，避免 HM 6 样本产生极端权重                           |
| `beta`   | `{0.25, 0.5}`                                                  |
| `w_max`  | `{3, 5, 8}`，首发用 3 或 5                                     |
| 生效类别 | 首发只对 `HM/LQS`，第二轮再扩到 `HM/LQS/QHS`                   |

优点：

- 与 IA-BCE 结构贴合，改动小。
- 主要提升少数类正样本梯度，不主动制造稀有类背景 FP。
- 对多数类影响可控，适合作为 0807 的第一组训练侧改动。

风险：

- 如果 HM/LQS 的问题主要来自 matcher 分配或 box 质量，单纯正样本加权可能收益有限。
- 过大 `w_max` 会让小样本类别过拟合，出现训练集指标好、测试 FDR 高。

### 4.2 P1：居中截断 Logit Adjustment（谨慎增强）

原方案的 LA 思路可以保留，但要改成适合 sigmoid 检测头的形式。

不建议直接使用：

```python
logit_bias = tau * torch.log(class_prior)
adjusted_logits = logits + logit_bias
```

原因是 `class_prior < 1` 时所有 bias 都为负，会改变 sigmoid 绝对分数尺度；少数类 bias 尤其负，会降低大量负样本惩罚，训练后再推理侧减先验可能进一步抬高稀有类 FP。

建议使用**居中 + 截断 + warmup**：

```python
raw_bias = torch.log(class_prior.clamp_min(eps))
centered_bias = raw_bias - raw_bias.mean()
logit_bias = tau * centered_bias.clamp(min=-bias_clip, max=bias_clip)
adjusted_logits = logits + warmup_factor * logit_bias
```

建议初始值：

| 参数        | 建议                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| `tau`       | `{0.1, 0.25, 0.5}`，不要首发 1.0/2.0                                       |
| `bias_clip` | `{1.0, 2.0}`                                                               |
| `warmup`    | 前 20%-30% step 从 0 线性升到目标值                                        |
| 推理侧      | 先比较 `raw logits`、`logits - bias`、`logits - k*bias`，`k ∈ {0, 0.5, 1}` |

这里的 LA 不再声称严格理论最优，而是作为**可控的先验去偏置 margin**。它是否优于 P0，必须由实验决定。

### 4.3 P2：matcher 成本是否同步调整

当前 matcher 位于 `src/rfdetr/models/matcher.py`，分类成本同样基于 sigmoid focal cost。是否把 class prior/bias 加入 matcher，不应默认打开。

建议判断标准：

- 如果 HM/LQS 的 GT 周围已有高 IoU query，但分类分数被 MS/QHS 压住：优先只改 loss，不改 matcher。
- 如果 HM/LQS 的 matched query 框质量明显差，或经常被错误类 query 抢走：尝试 matcher 分类成本轻量均衡。
- matcher 调整只用小强度，并建议从训练中后期打开，避免早期分配噪声放大。

首发建议：**matcher 不动**。这是保护多数类和定位质量的关键。

## 5. 推荐实验矩阵

所有实验都以 `output/0807-SHWX-SSCL-Proj-原型+实例正样本/checkpoint_best_total.pth` 或其训练配方为对照，保持数据、epoch、增强、SSCL、分辨率、eval 脚本一致。

### 5.1 阶段 A：零训练校准诊断

目的：判断小样本问题是否主要是分数校准。

| 实验 | 内容                                  | 判断                                        |
| ---- | ------------------------------------- | ------------------------------------------- |
| A0   | 0807 原阈值复评                       | 固定基线                                    |
| A1   | 只搜 HM/LQS 阈值                      | 若 TP 明显上升且 FDR 可控，说明后处理空间大 |
| A2   | 对 HM/LQS 加推理 bias，再搜阈值       | 若优于 A1，说明 logit prior 有用            |
| A3   | 对 HM/LQS/QHS/MS 联合 bias + 阈值网格 | 观察舰船内部混淆是否改善                    |

输出必须包含：

- 总 Recall/FDR。
- 三大类 macro Recall/FDR。
- 25 类 TP/FP/FN/Precision/Recall/FDR。
- HM/LQS/QHS/MS 混淆矩阵。
- 每类 score 分布直方图或分位数。

### 5.2 阶段 B：训练侧 P0 消融

| 实验 | 改动                                      | 预期                         |
| ---- | ----------------------------------------- | ---------------------------- |
| B0   | 0807 配方重跑                             | 排除随机性                   |
| B1   | 正样本均衡，`beta=0.25,w_max=3,HM/LQS`    | 最稳，优先看小类 TP 是否出现 |
| B2   | 正样本均衡，`beta=0.5,w_max=5,HM/LQS`     | 更强，观察 FDR 是否开始失控  |
| B3   | 正样本均衡，`beta=0.5,w_max=5,HM/LQS/QHS` | 若 QHS 也受 MS 压制，可尝试  |

优先选择标准：

```text
Recall >= 0.85
FDR <= 0.20
MS/QHS recall 不低于 0807+阈值 3pp 以上
飞机大类 FDR 不显著变差
HM/LQS 至少一个类别 TP/F1 明显改善
```

### 5.3 阶段 C：训练侧 P1 消融

只在 B1/B2 有收益但仍未达标时做。

| 实验 | 改动                                     | 预期           |
| ---- | ---------------------------------------- | -------------- |
| C1   | P0 + centered LA，`tau=0.1,bias_clip=1`  | 温和校准       |
| C2   | P0 + centered LA，`tau=0.25,bias_clip=1` | 中等强度       |
| C3   | P0 + centered LA，`tau=0.5,bias_clip=2`  | 高风险上限测试 |

每组都必须同时评估：

- 训练后 raw logits 推理。
- 训练后 `logits - 0.5*bias` 推理。
- 训练后 `logits - bias` 推理。
- 每种推理再搜逐类阈值。

### 5.4 阶段 D：matcher 调整上限测试

只在以下诊断成立时做：

```text
HM/LQS 的 GT 附近存在高 IoU query，但被 matcher 分给 MS/QHS；
或 HM/LQS 的 matched query 数量/质量显著低于其他船类。
```

实验：

| 实验 | 改动                                                               |
| ---- | ------------------------------------------------------------------ |
| D1   | matcher 分类成本对 HM/LQS 使用小强度 centered bias，训练后半程打开 |
| D2   | matcher + P0 正样本均衡                                            |

若 D1/D2 造成 MS/QHS recall 或全局 FDR 明显变差，直接放弃 matcher 路线。

## 6. 指标与判定

### 6.1 主指标

优先级从高到低：

1. 比赛口径总 Recall/FDR 是否同时过线。
2. 三大类 macro Recall/FDR 是否均衡，不能只靠飞机类拉高。
3. HM/LQS/QHS/MS 的 per-class TP/FP/FN 是否改善。
4. MS/QHS 多数类是否被过度牺牲。
5. fixed threshold 和 tuned threshold 的差距是否变小。

### 6.2 早停判断

出现以下任一情况，应停止加大均衡强度：

- 总 Recall 低于 0.85。
- 总 FDR 高于 0807+阈值基线。
- MS 或 QHS recall 较基线下降超过 3pp。
- HM/LQS TP 没增加但 FP 增加。
- 飞机类 FDR 明显上升，说明全局 objectness 被扰动。

### 6.3 最优方案选择

最优不是单点最高 Recall，也不是单点最低 FDR，而是在硬约束内最大化稳健 margin：

```text
score = macro_recall - lambda_fdr * max(0, fdr - 0.20)
```

实际选择时建议使用硬门槛：

```text
Recall >= 0.855
FDR <= 0.195
```

留出 0.5pp 左右安全余量，避免测试集波动或脚本细节差异导致临界不过线。

## 7. 实现建议

### 7.1 配置项

建议新增配置项，而不是写死 SHWX 类别：

```python
class_balance_enabled: bool = False
class_balance_counts_path: str | None = None
class_balance_beta: float = 0.25
class_balance_max_weight: float = 3.0
class_balance_min_count: int = 10
class_balance_target_classes: list[int] | None = None

logit_adjustment_enabled: bool = False
logit_adjustment_tau: float = 0.1
logit_adjustment_bias_clip: float = 1.0
logit_adjustment_warmup_epochs: float = 1.0
logit_adjustment_in_matcher: bool = False
```

### 7.2 代码落点

| 模块                                              | 改动                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/rfdetr/config.py`                            | 增加配置项                                                                                  |
| `src/rfdetr/models/criterion.py`                  | 在 `loss_labels` 中对正样本 `pos_weights` 乘类别权重；可选对 logits 加 centered/capped bias |
| `src/rfdetr/models/matcher.py`                    | 可选加入 matcher bias，默认关闭                                                             |
| `src/rfdetr/models/postprocess.py` 或外部测试脚本 | 推理侧 bias/阈值校准，建议实验脚本先实现                                                    |
| `src/scripts/`                                    | 增加统计 class counts、导出 raw logits/score 分布、网格搜索阈值的脚本                       |

### 7.3 实现原则

- 类别统计从训练集 labels 自动生成，并随实验目录保存，避免手工写错。
- 所有权重和 bias 注册为 buffer，保证 device、DDP、checkpoint 行为一致。
- 首发只改最终 decoder 层和 aux 层共用的 criterion 输入；不要先改 backbone/decoder 结构。
- 对 aux outputs 使用同一类权重，保持训练信号一致。
- 所有新增注释使用中文，符合本仓库 agent 约束。

## 8. 推荐首轮实验

首轮不要做 3 个 τ 的完整 LA。建议只跑下面 4 组，成本低、归因清楚：

| 编号 | 方案                                                  | 目的                           |
| ---- | ----------------------------------------------------- | ------------------------------ |
| E0   | 0807 checkpoint 复评 + 阈值重搜                       | 固定基线和校准上限             |
| E1   | 正样本均衡 `beta=0.25,w_max=3,HM/LQS`                 | 低风险验证小样本正梯度是否有效 |
| E2   | 正样本均衡 `beta=0.5,w_max=5,HM/LQS`                  | 验证更强均衡上限               |
| E3   | E1 + centered LA `tau=0.1,bias_clip=1,warmup=1 epoch` | 验证 logit bias 是否有额外收益 |

若 E1 已能明显降低 FDR 且 Recall 过线，就不要急着加强。若 E1/E2 只提高 HM/LQS TP 但 FDR 高，则优先调阈值，不要继续增大 loss 权重。若 E1/E2 对 HM/LQS 完全无效，再回头诊断 matcher 和 box 质量。

## 9. 预期收益与风险修订

更客观的预期如下：

| 指标          |     温和正样本均衡 |                   强 LA/强权重 |
| ------------- | -----------------: | -----------------------------: |
| HM/LQS Recall |           可能上升 |                   可能明显上升 |
| HM/LQS FDR    | 不确定，需阈值保护 |                     高风险上升 |
| MS/QHS Recall |     小幅下降或持平 |                   可能明显下降 |
| 总 Recall     |     目标持平或微升 | 可能升，也可能因多数类下降而降 |
| 总 FDR        |   有机会下降 1-4pp |               方差大，可能恶化 |

最值得追求的形态不是“稀有类分数越高越好”，而是：

```text
HM/LQS 从完全不可检或低检出变为少量稳定 TP；
同时 MS/QHS 的 FP/FN 不被大幅交换到 HM/LQS。
```

如果最终只有 HM/LQS FP 增加、TP 不增加，说明问题不是类别先验，而是候选框质量、跨域背景干扰或小类视觉特征不足，应该转向难例负样本、数据切片或 matcher 诊断。

## 10. 与 SSCL 系列方案的关系

SSCL 仍然有价值，但它解决的是表示空间中的类间拉开；本方案解决的是分类损失对不同类别正样本的梯度预算。二者可以叠加，但不要在同一轮同时加大 SSCL 强度和分类均衡强度，否则很难归因。

推荐组合顺序：

1. 固定 0807 SSCL 配方。
2. 加 P0 正样本类均衡。
3. 重新搜阈值。
4. 仍不足时加 P1 centered LA。
5. 最后才考虑 matcher 或 SSCL 负类均衡。

## 11. 参考文献

1. Menon, A., Jayasumana, S., Rawat, A., Jain, H., Veit, A., Kumar, S. "Long-tail learning via logit adjustment." ICML 2021.
2. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. "Focal loss for dense object detection." ICCV 2017.
3. Carion, N., et al. "End-to-End Object Detection with Transformers." ECCV 2020.
4. Chen et al. "Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection." CVPR 2026. 见 `docs/参考论文/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.md`
