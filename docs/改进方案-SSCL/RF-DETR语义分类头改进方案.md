# RF-DETR 语义分类头改进方案

> 基于论文《Rethinking Prototype-based Similarity Learning for Few-Shot Object Detection》(ReSet)
> 的文本锚定思想，将 CLIP 语义先验注入 RF-DETR 的 decoder 分类头，缓解 SHWX 舰船细粒度
> 类别（HM/LQS/QHS/MS）在少样本下的类间混淆。本文档是实验方案，供后续编写代码并逐步验证。

---

## 0. 背景与动机

### 0.1 已确认的实验事实

| 实验                          | 结论                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| 全量微调 baseline（0805）     | 少样本类能被检测到，但 HM/LQS/QHS/MS 之间类间混淆、FP 多    |
| 完全冻结 raw DINOv2（0808）   | **少样本类（HM/LQS）完全崩溃** → 骨干必须适配 SHWX 后才能用 |
| SSCL 二阶段（0807，保守冻结） | 在适配后 checkpoint 上冻结骨干修混淆，可行                  |

**关键推论**：冻结"raw 骨干"（未适配）崩溃，冻结"适配后的骨干"可行。语义头必须建立在
**已适配**的特征空间上，不能试图替代特征适配。

### 0.2 问题定位

- SSCL 修的是**特征层**（拉大类间特征距离）；
- 但真正的分类决策由 `class_embed = nn.Linear(hidden_dim, num_classes)` 完成，少样本类的
    W_c 行从极少数匹配样本中硬估，方向噪声大 → 边界漂移；
- 本文档的语义分类头修的是**决策层**：给少样本类的分类方向一个**不依赖样本量**的语义先验。

### 0.3 设计原则

1. **语义进分类器，不进骨干**：骨干自由适配，分类器调和"语义先验 + 任务特征"；
2. **冻结"适配后"而非"raw"骨干**：唯一能让语义方向站得住的前提；
3. **语义是软补充，不是硬替换**：原版 `logits_learned` 始终主导，语义项只是给少样本类决策
    边界一个稳定锚；
4. **少样本类的语义参数不与少样本数据接触**：f_sem、S 完全冻结，只有 α_c 可学习。

---

## 1. 语义头结构设计

### 1.1 总体结构（残差增量形式，最小侵入）

在 decoder 前向中，`outputs_class` 由原版线性头给出，语义头作为**残差增量**叠加：

```
outputs_class = class_embed(hs)                         # 原版路径（不动）
outputs_class = outputs_class + SemanticResidual(hs)    # 语义增量（新增）
```

残差增量内部两条子路径（可独立开关，天然可消融）：

```
SemanticResidual(hs) =  hs @ (W ⊙ (M − 1))ᵀ    通道掩码增量（M=全1 时恒为 0）
                       + α ⊙ (hs @ Sᵀ)         语义方向增量（α=0 时恒为 0）
```

其中 W 是 `class_embed.weight` 的引用。残差形式等价于把总分类权重改写为：

```
W̃_c = W_c ⊙ M_c + α_c·s_c        （每类一个"语义锚定 + 通道选择"的超平面方向）
logit_c = W̃_cᵀ·h + b_c
```

采用"原版 + 残差"而不是直接替换 W 的理由：

- `enc_out_class_embed`（[lwdetr.py:220](src/rfdetr/models/lwdetr.py#L220)）是 `class_embed` 的深拷贝，
    用于 encoder proposal 分类 / top-k query 选择。若直接替换 class_embed，proposal 分类器也会
    被语义化，改变 query selection、影响 recall。残差形式让 **proposal 路径完全不受影响**；
- 原版 W 及其偏置初始化（[lwdetr.py:207](src/rfdetr/models/lwdetr.py#L207) `bias = bias_value`）、
    `_resize_linear`（num_classes 变化）逻辑全部保持不动。

### 1.2 语义方向投影 f_sem（离线训练，冻结）

- 输入：类别 CLIP 文本向量 t_c ∈ ℝ^768（`semantic_matrix.py` 已产出同源文本向量）；
- 输出：类别语义方向 s_c = f_sem(t_c) ∈ ℝ^d（d = hidden_dim，medium 为 256）；
- 结构：两层仿射 + tanh（Talk2DINO 的 warping，见论文 Eq.1）：

```
f_sem(t) = W_bᵀ(tanh(W_aᵀ t + b_a)) + b_b
```

- **训练数据**：Stage-1（全量微调）模型在 **base 类**上提取的 matched query 特征
    （一次离线前向，详见 §4.1 收集脚本）。**绝不含少样本类**，避免对齐被噪声带歪；
- **训练目标**：对称 InfoNCE，使 f_sem(t_c) 与类别 c 的 matched query 特征对齐（详见 §3.2）；
- **对齐空间**：decoder 特征空间（d 维），因为分类发生在该空间。注意：论文对齐 DINOv2 patch
    特征，但 RF-DETR 分类在 decoder 空间，目标空间必须一致，否则语义方向落不准；
- **使用**：训练完成后冻结，S 矩阵（每行 s_c）以 buffer 存入模型。

### 1.3 通道掩码 M_c（离线统计 + 可学习阈值）

- 来自每类 matched query 特征的通道级激活统计（TF-IDF，详见 §3.3），**离线**从 base 类
    特征计算（与 f_sem 同源，一次收集脚本产出两份统计）；
- 每个通道一个软掩码权重：M\_{c,i} = σ((θ_c − r\_{c,i})/τ_mask)；
- **θ_c**：每类一个可学习标量，控制保留多少通道。少样本类训练前初始化为 base 类 θ 均值
    （论文技巧），训练中冻结或极慢更新；
- 作用：让分类器"只听该类判别性强的通道"，抑制风格/共享通道。这是论文 TSMa 从"掩码原型"
    搬到"掩码分类器行"，**负责微观子类（HM vs LQS）的分离**，因为纯文本方向对它们几乎相同。

### 1.4 混合系数 α_c

- 每类一个可学习标量，控制语义方向注入强度；
- 初始化：base 类小扫描（见 §5.1 实验 3）确定安全起点；少样本类可略大，基类保持小；
- 约束：clamp 到 [0, α_max]（默认 2.0），防止语义项把舰船 logit 整体抬高引发 FP；
- 基类保护：α_c 小 + 现有蒸馏（受保护类 = 飞机类 + FSC）。

### 1.5 与原版 class_embed 的完整关系

```
                     ┌── class_embed.weight (W) ⊙ M_c  ──→  通道选择性学习方向
每个类一个分类方向 ──┤
                     └── α_c · s_c = α_c · f_sem(t_c) ──→  冻结的语义锚定方向

   logit_c(h) = (W_c ⊙ M_c + α_c·s_c)ᵀ · h + b_c
```

- W、b：数据学习（原版）；
- M_c：统计（无梯度）+ θ_c（可学习）；
- s_c：完全冻结（来自离线 f_sem）；
- α_c：可学习（唯一接触梯度的语义参数）。

---

## 2. 训练策略设计

### 2.1 三阶段总览

```
Stage 1（已有，不动）
  全量微调 baseline（backbone 适配 SHWX）
  └─ 产出: 0805 checkpoint（同时也是蒸馏 teacher）

离线准备（Stage 1 之后）
  collect_base_features.py：Stage-1 模型在 base 类上收集 matched query 特征
  train_fsem.py：训练 f_sem + 计算 TF-IDF 通道统计
  └─ 产出: fsem_shwx.pt, channel_stats_shwx.pt（供 Stage 2 加载）

Stage 2（本文档新增）
  在 0805 checkpoint 上冻结"适配后"骨干，解冻 decoder 末层 + 分类头 + 语义组件 + SSCL
  加语义头 + SSCL + 蒸馏
  └─ 产出: 语义头实验 checkpoint
```

### 2.2 Stage 2 冻结/可训练矩阵

| 模块                        | 状态                                   | 说明                          |
| --------------------------- | -------------------------------------- | ----------------------------- |
| DINOv2 骨干                 | 冻结                                   | 适配后的 checkpoint，不是 raw |
| encoder 主体                | 冻结                                   |                               |
| bbox_embed                  | 冻结                                   | 保框回归稳定                  |
| decoder 前几层              | 冻结                                   |                               |
| **decoder 最后一层 + norm** | **可训练**                             | SSCL 需重塑 query 特征        |
| **class_embed (W, b)**      | **可训练**                             | 分类决策层                    |
| **θ_c（掩码阈值）**         | **可训练**（少样本类冻结于 base 均值） | 小参数量                      |
| **α_c（混合系数）**         | **可训练**                             | 语义注入强度                  |
| f_sem / S / M 统计          | 冻结 buffer                            | 不与少样本数据接触            |
| SSCL 投影头                 | 可训练                                 | 沿用现有逻辑                  |
| 蒸馏 teacher                | 冻结                                   | 0805 checkpoint               |

### 2.3 参数组（优化器）

| 参数组             | 学习率             | 说明                                              |
| ------------------ | ------------------ | ------------------------------------------------- |
| decoder 末层       | 1e-5               | 沿用现有                                          |
| class_embed (W, b) | 1e-5               | 沿用现有                                          |
| **θ_c, α_c**       | **1e-4**（独立组） | 标量参数需足够 LR，否则学不出来（投影头同款教训） |
| SSCL 投影头        | 独立组（现有逻辑） |                                                   |

### 2.4 损失组成

```
L = L_det + λ_sscl·L_SSCL + λ_distill·L_distill
```

- L_det：原版 focal（分类）+ L1/GIoU（回归），作用于叠加后的 logits；
- L_SSCL：现有语义加权对比损失（作用在 matched query features，详见 §3.5）；
- L_distill：基类 logits 蒸馏（受保护 = 飞机类 + FSC，舰船类不蒸馏）；
- 语义头**不新增独立损失**——语义以结构方式（残差 logit）参与，靠 L_det 端到端学习 α/θ。

---

## 3. 关键公式与变量含义

### 3.1 符号总表

| 符号                                                     | 维度         | 含义                                             |
| -------------------------------------------------------- | ------------ | ------------------------------------------------ |
| d                                                        | —            | decoder hidden dim（medium = 256）               |
| C                                                        | —            | 类别数（SHWX = 25）                              |
| B, Q                                                     | —            | batch size, query 数                             |
| h                                                        | ℝ^{B,Q,d}    | decoder 最后一层 query 特征（matched 或全部）    |
| W, b                                                     | ℝ^{C,d}, ℝ^C | class_embed 权重与偏置                           |
| t_c                                                      | ℝ^768        | 类别 c 的 CLIP 文本向量                          |
| f_sem                                                    | 768→d        | 文本→视觉空间语义方向投影（冻结）                |
| s_c                                                      | ℝ^d          | s_c = f_sem(t_c)，类别 c 的语义方向（L2 归一化） |
| S                                                        | ℝ^{C,d}      | 语义方向矩阵，行 = s_c                           |
| M                                                        | [0,1]^{C,d}  | 每类通道掩码                                     |
| θ_c                                                      | ℝ            | 每类掩码阈值（可学习）                           |
| τ_mask                                                   | ℝ            | 掩码软度                                         |
| α                                                        | ℝ^C          | 每类语义注入强度（可学习，clamp [0, α_max]）     |
| g\_{c,n}(i)                                              | {0,1}        | 类 c 第 n 个实例的通道 i 是否属高激活簇          |
| count_c(i), TF_c(i), DF(i), IDF(i), Score_c(i), r\_{c,i} | —            | TF-IDF 通道打分与排名                            |
| τ                                                        | ℝ            | SSCL 温度                                        |

### 3.2 语义方向投影 f_sem 的训练（离线，InfoNCE）

设收集到的 base 类样本对为 {(v_i, t\_{c_i})}，v_i ∈ ℝ^d 是实例 i 的 matched query 特征，
t\_{c_i} 是其类别文本。定义归一化相似度：

$$
\mathrm{sim}(v_i, \tilde t_j) = \frac{v_i}{\|v_i\|_2} \cdot \frac{f_{\mathrm{sem}}(t_j)}{\|f_{\mathrm{sem}}(t_j)\|_2}
$$

对称 InfoNCE（对齐论文补充 Eq.S5）：

$$
\mathcal{L}_{\mathrm{align}} = -\frac{1}{2B}\sum_i \log\frac{e^{\mathrm{sim}(v_i,\tilde t_i)/\tau}}{\sum_j e^{\mathrm{sim}(v_i,\tilde t_j)/\tau}}
- \frac{1}{2B}\sum_i \log\frac{e^{\mathrm{sim}(v_i,\tilde t_i)/\tau}}{\sum_j e^{\mathrm{sim}(v_j,\tilde t_i)/\tau}}
$$

只更新 f_sem 的参数。训练集 = base 类，不含少样本类。

### 3.3 通道掩码 M 的计算（离线统计 + 在线阈值）

**离线统计**（在 base 类特征上计算，与 f_sem 同源）：

$$
a_{c,n} = \frac{v_{c,n}}{\|v_{c,n}\|_2} \odot \frac{s_c}{\|s_c\|_2} \in \mathbb{R}^d
$$

对 a\_{c,n} 的 d 个通道做 k-means(k=2)，均值更高的簇记为激活簇，得到 g\_{c,n}(i) ∈ {0,1}。然后：

$$
\mathrm{TF}_c(i) = \frac{\mathrm{count}_c(i)}{N_c}, \qquad
\mathrm{DF}(i) = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\mathrm{TF}_c(i),
\qquad
\mathrm{IDF}(i) = \log\frac{1}{\mathrm{DF}(i)+\varepsilon}
$$

$$
\mathrm{Score}_c(i) = \mathrm{TF}_c(i)\cdot\mathrm{IDF}(i)
$$

TF 大 = 该类频繁激活，IDF 大 = 他类少激活 → Score 高的通道是**类判别通道**。
记 r\_{c,i} 为通道 i 按 Score_c 降序的排名。

**在线掩码**（θ_c 可学习）：

$$
M_{c,i}(\theta_c) = \sigma\!\Big(\frac{\theta_c - r_{c,i}}{\tau_{\mathrm{mask}}}\Big)
$$

- θ_c 初始化为 d（最低排名，所有通道 M≈1，防止模型坍塌，随训练逐步收窄）；
- 少样本类 θ_c 初始化为 base 类 θ 均值，且冻结（样本太少，学不准）。

### 3.4 语义头前向（在线）

```
outputs_class = class_embed(hs)                      # [B,Q,C]  原版
mask_delta    = hs @ (W ⊙ (M − 1))ᵀ                 # [B,Q,C]  通道掩码增量
sem_delta     = (α ⊙ (hs @ Sᵀ))                      # [B,Q,C]  语义方向增量
outputs_class = outputs_class + mask_delta + sem_delta
```

等价总分类权重：W̃_c = W_c ⊙ M_c + α_c·s_c。

### 3.5 训练损失（沿用现有 SSCL，作用点不变）

SSCL 仍作用在 decoder 最后一层 matched foreground query features（投影后）：

$$
L_{\mathrm{SSCL}} = -\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}}
\log\frac{\sum_{j\in P(i)} e^{s_{ij}/\tau}}
{\sum_{j\in P(i)} e^{s_{ij}/\tau} + \sum_{c\in N(i)} e^{\omega_{ic}\,s_{ic}/\tau}}
$$

- P(i) = 本类原型 ∪（可选）同类实例；N(i) = 全部其他类原型；
- ω_ic = clamp(1 + ρ·Φ[y_i,c], 1, ω_max)，Φ 为 CLIP 语义矩阵；
- **语义头不改变 SSCL 的公式**，只改变分类 logits（L_det 的输入）。

---

## 4. 对原版 RF-DETR 的修改

### 4.1 新增文件

| 文件                                   | 内容                                                                                                |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `src/rfdetr/sscl/semantic_head.py`     | `SemanticResidual(nn.Module)`：加载 S/M/α/θ，前向返回残差增量                                       |
| `src/scripts/collect_base_features.py` | 用 Stage-1 模型在 base 类上收集 matched query 特征 + 标签，保存 `.pt`                               |
| `src/scripts/train_fsem.py`            | 读收集特征 → 训练 f_sem（InfoNCE）→ 计算 TF-IDF 统计 → 保存 `fsem_shwx.pt`、`channel_stats_shwx.pt` |

### 4.2 修改文件

| 文件                                  | 改动                                                                                                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/rfdetr/config.py`                | 新增语义头配置字段（见 4.4）                                                                                                                                                         |
| `src/rfdetr/models/lwdetr.py`         | 新增 `self.semantic_residual` 属性（默认 None）；在 [530 行](src/rfdetr/models/lwdetr.py#L530) `outputs_class = self.class_embed(hs)` 之后叠加残差。**enc_out_class_embed 完全不动** |
| `src/rfdetr/training/module_model.py` | `_setup_sscl` 中加载 f_sem/M 统计构建 semantic_residual；冻结策略把 θ/α 纳入可训练                                                                                                   |
| `src/rfdetr/training/param_groups.py` | 新增 `get_semantic_head_param_dict()`（收集 θ、α 为独立参数组）                                                                                                                      |
| `src/scripts/train_sscl.py`           | 常量开关 `SEMANTIC_HEAD_ENABLED` 等 + `OUTPUT_DIR`                                                                                                                                   |
| `src/scripts/test.py`                 | 若启用，checkpoint 中已含语义头权重，确认加载无异常                                                                                                                                  |

### 4.3 明确不改的部分

- `enc_out_class_embed`（proposal 分类 / top-k 选择）：**保持原版线性头**，避免 query selection 语义化后 recall 波动；
- `class_embed` 本身及其 bias 初始化、`_resize_linear`：原样保留（语义头是残差叠加，不替换）；
- backbone / encoder / bbox_embed 的冻结逻辑：沿用现有 conservative 策略。

### 4.4 新增配置字段（`TrainConfig`）

```python
semantic_head_enabled: bool = False  # 语义头主开关
semantic_fsem_path: str | None = None  # fsem_shwx.pt 路径（S 矩阵）
semantic_channel_stats_path: str | None = None  # channel_stats_shwx.pt（M 统计）
semantic_mask_enabled: bool = True  # 通道掩码增量开关（消融用）
semantic_alpha_init: float = 0.0  # α 初始值（base 扫描结果）
semantic_alpha_max: float = 2.0  # α 上限
semantic_alpha_learnable: bool = True
semantic_mask_tau: float = 1.0  # 掩码软度
semantic_lr: float = 1e-4  # θ/α 独立参数组学习率
semantic_novel_classes: list[int] | None = None  # 少样本类集合（HM/LQS/...），α 更大、θ 冻结
semantic_frozen_threshold_classes: list[int] | None = None  # θ 冻结于 base 均值的类
```

---

## 5. 分阶段实验方案

> 原则：**每次只变一个因素**，输出目录必须分开，无法归因就不算数。
> 评估口径见 §5.3，**不能只看 val mAP**。

### 阶段 0：前置验证（不通过则停，节省算力）

| 步骤 | 内容                                             | 通过标准                                      |
| ---- | ------------------------------------------------ | --------------------------------------------- |
| 0a   | 收集 base 类 matched query 特征，训练 f_sem      | —                                             |
| 0b   | base 类 held-out 对齐校验：`cos(s_c, mean(h_c))` | 同类 cos 显著高于跨类（≥0.3 且 > 跨类 + 0.2） |
| 0c   | 语义矩阵复查（复用现有 validate_matrix）         | 舰船/飞机/车辆组间无异常相似                  |

不通过 → 排查 prompt 或 f_sem 结构，不进入训练。

### 阶段 1：结构选择（平行 vs 残差 vs 初始化）

三选一，确定语义项注入方式：

| 实验 | 结构                             | 说明                                                                    |
| ---- | -------------------------------- | ----------------------------------------------------------------------- |
| 1a   | **残差增量**（推荐，本文档默认） | `outputs_class += (W⊙(M−1) + α·S)ᵀ·h`                                   |
| 1b   | 平行 logit                       | `outputs_class = class_embed(hs) + α ⊙ (hs@Sᵀ)`（无通道掩码，作为对照） |
| 1c   | 语义初始化                       | 少样本类 W_c 行初始化为 α·s_c，再正常微调                               |

**判断**：ship 大类 precision/F1、HM/LQS 逐类 AP、aircraft/FSC 不掉。
**推荐**：若 1a 与 1b 差异小，保留 1a（结构统一、天然可消融）。

### 阶段 2：组件消融（M_c 通道掩码的贡献）

| 实验 | 配置                                                       | 验证                                     |
| ---- | ---------------------------------------------------------- | ---------------------------------------- |
| 2a   | 语义头完整（M_c + α·S）                                    | 基准                                     |
| 2b   | `semantic_mask_enabled=False`（M=1）                       | **M_c 对 HM/LQS 微观分离的独立贡献**     |
| 2c   | `semantic_alpha_learnable=False, alpha_init=0`（无语义项） | 回退 = 纯原版 + SSCL，确认语义头整体增益 |

**判断**：重点看 HM/LQS 与 QHS/MS 的 per-class precision/recall 和 ship FP。若 2b 与 2a 差距大
→ M_c 是关键；若 2a 与 2c 差距小 → 语义头收益有限，考虑只保留 M_c 或放弃语义项。

### 阶段 3：α 调度（注入强度）

| 实验 | 配置                                             |
| ---- | ------------------------------------------------ |
| 3a   | α 全可学习，init=base 扫描安全值，clamp [0, 2.0] |
| 3b   | α 冻结在 base 扫描值                             |
| 3c   | 少样本类 α 初始更大（×1.5），基类 α 冻结为小值   |

先做 **base 类 α 扫描**：α 固定 ∈ {0, 0.1, 0.3, 0.5}，在 base 类验证集上找"基类不掉"的最大 α，
作为 3a/3b/3c 的起点。**判断**：少样本类 precision 提升的同时 aircraft/FSC 不掉。

### 阶段 4：与 SSCL 的交互（最终组合）

| 实验 | 语义头 | SSCL | ω                        |
| ---- | ------ | ---- | ------------------------ |
| 4a   | ✓      | ✓    | 语义加权（现有）         |
| 4b   | ✓      | ✓    | ω=1（退化为普通 SupCon） |
| 4c   | ✓      | ✗    | —                        |

**判断**：若 4a ≈ 4b → 语义头已承担语义先验，SSCL 的 CLIP 权重冗余，可简化；若 4a > 4b →
两者语义来源互补，都保留。4c 对比确认 SSCL 在语义头存在时仍有独立增益。

### 阶段 5（可选）：骨干适配策略

| 实验 | Stage-1 骨干                   |
| ---- | ------------------------------ |
| 5a   | 全量微调（现有）               |
| 5b   | LoRA-DINOv2（`train_LoRA.py`） |

若少样本类仍不稳定，再考虑。LoRA 在保住适配的同时降低过拟合，语义头结构不变，直接复用。

### 5.1 实验输出目录约定

```
output/08xx-SHWX-SemHead-<结构>-<组件>-<α策略>-<SSCL开关>
示例：
  output/0810-SHWX-SemHead-residual-MaskOn-alphaLearn-sscl
  output/0810-SHWX-SemHead-residual-MaskOff-alphaLearn-sscl
  output/0810-SHWX-SemHead-residual-MaskOn-alphaFix-sscl
```

### 5.2 统一超参（除变量外保持一致）

- 起点：`0805-SHWX-data-expand-rfdetr-baseline/checkpoint_best_total.pth`
- EPOCHS=6，BATCH=32×4，LR_decoder=1e-5，LR_semantic=1e-4，WARMUP=0，Mosaic=0
- λ_sscl=0.01~0.05，τ=0.1，ρ=0.3，ω_max=2.0
- 蒸馏：teacher=0805，受保护=[4..24]（飞机+FSC），λ_distill=0.5

### 5.3 评估指标（约束式选模型）

- overall / ship / aircraft / vehicle 的 TP、FP、FN、precision、recall、F1；
- HM、LQS、QHS、MS 逐类 AP / precision / recall；
- ship GIoU / AP75（盯语义头对框回归的影响）；
- HM/LQS/QHS/MS 的 FP 可视化；
- `train/loss_sscl` 曲线健康度。

**硬约束**：aircraft F1 ≥ 基线；FSC recall ≥ 基线；FP_ship ≤ 基线 + δ(10~20)；HM/LQS precision 提升。

### 5.4 决策树

```
阶段0 通过？
 ├─ 否 → 停，修 prompt / f_sem
 └─ 是 → 阶段1 定结构
          ├─ 1a 胜 → 阶段2（M_c 消融）
          │           ├─ M_c 显著 → 保留，进阶段3
          │           └─ M_c 不显著 → 去掉 M_c，只留 α·S，进阶段3
          └─ 1b/1c 胜 → 按胜者调整，重跑阶段2
          → 阶段3 定 α（少样本收益 vs 基类保持）
          → 阶段4 定 SSCL 是否保留/简化
          → （可选）阶段5 换骨干策略
```

---

## 6. 风险与对策

| 风险                                                       | 对策                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| **FP 重演**（舰船 logit 被语义项整体抬高，MS/QHS FP 爆增） | α clamp [0, α_max]、base α 扫描起步、per-class threshold 重新校准 |
| f_sem 对齐不准（base 与 novel 分布漂移）                   | 阶段0 对齐校验前置；不通过直接停                                  |
| 文本饱和（HM/LQS 的 s_c 几乎相同，语义项分不开它们）       | 微观分离交给 M_c 通道掩码 + SSCL，语义项只管宏观                  |
| 细粒度类 matched 特征少 → M_c 统计噪声大                   | M 统计只用 base 类；少样本 θ 冻结于 base 均值                     |
| 加残差后 logit 尺度变化影响 focal/matching                 | 训练后统一重校准 per-class threshold                              |
| 基类回退                                                   | α 基类小值 + 蒸馏（受保护类）                                     |

---

## 7. 实施顺序（给编码的 checklist）

1. 写 `collect_base_features.py`，跑 base 类特征收集；
2. 写 `train_fsem.py`，训练 f_sem + TF-IDF 统计，跑阶段 0 对齐校验；
3. 写 `semantic_head.py`（SemanticResidual）；
4. 改 `config.py` + `lwdetr.py`（叠加残差，enc_out 不动）+ `param_groups.py` + `module_model.py`；
5. 改 `train_sscl.py` 加开关，先跑 2c（语义头整体增益确认）；
6. 按阶段 1→2→3→4 逐步消融，输出目录分离；
7. 每阶段出结果后回填决策树，再决定下一步。
