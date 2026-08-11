# RF-DETR 引入 SSCL 语义相似度引导方案

## 1. 方案目标

本方案目标是在 RF-DETR 已有检测能力基础上，引入语义相似度引导的监督对比学习损失，即 SSCL（Semantic Correlation-Driven Supervised Contrastive Learning），缓解 SHWX 遥感数据集中舰船细粒度类别之间的分类混淆问题，尤其关注 HM、LQS、QHS、MS 等类别的误检和类别边界不清晰问题。

与“用 CLIP 初始化分类头行”的方案不同，本方案不直接把 CLIP 文本向量写入 RF-DETR 分类头，也不把 CLIP 当作在线分类器。CLIP 只用于构造类别之间的语义相似度矩阵，告诉模型哪些类别更容易混淆，从而在训练时对这些类别之间的 query 特征施加更强的分离约束。

核心思想是：

- RF-DETR 仍然负责图像特征提取、query 解码、分类和框回归。
- CLIP 只离线生成类别语义相似度先验矩阵。
- SSCL 作用在 decoder 输出的 matched foreground query features 上。
- 对语义相似、容易混淆的不同类 query 特征施加更强的分离约束。
- 通过小权重辅助损失、轻量解冻和基类蒸馏，尽量避免损害原有基类指标。

## 2. 为什么不用 CLIP 直接初始化分类头

前期实验表明，直接使用 CLIP 文本特征初始化 HM、LQS、QHS、MS 分类头行存在明显风险。主要原因是 CLIP 文本空间与 RF-DETR 分类头空间并未天然对齐。如果直接将 CLIP 特征截断、归一化后写入 RF-DETR 分类头，会把小样本类分类方向推到错误位置，导致舰船类 FP 增加。

因此，本方案不再让 CLIP 输出直接参与分类头权重初始化，而是只使用 CLIP 的相对语义关系。例如：

- HM 与 MS 可能比 HM 与 F-16 更相似。
- LQS 与 QHS 可能比 LQS 与 FSC 更相似。
- 飞机类之间也可能存在一定语义相似性，但在当前任务中它们原始指标已经较稳定，应避免过强扰动。

这种“类别之间谁更像谁”的关系对跨空间对齐要求较低，因为它只使用 CLIP 文本空间内部的相对相似度，而不是要求 CLIP 文本向量和 RF-DETR 视觉向量逐维对齐。

## 3. RF-DETR 中 query selection 的基本流程

RF-DETR Medium 当前使用 two-stage 结构。其 query 生成与选择流程可以概括为：

1. 图像经过 DINOv2 backbone 和 encoder 得到多尺度视觉 memory。
2. encoder memory 上的每个空间位置生成候选 proposal。
3. encoder proposal 分类头输出每个位置的类别 logits。
4. 对每个 proposal 取最大类别 logit 作为 objectness-like 分数。
5. 通过 top-k 选择分数最高的 proposal，作为 decoder 初始 query。
6. decoder 对这些 query 进行多层 refine，输出最终 query features、分类 logits 和 bbox。
7. Hungarian matcher 将最终预测 query 与 GT 框进行一对一匹配。

需要注意：top-k query selection 本身只是选择候选 proposal，不代表这些 query 都是前景目标。top-k 中仍然会包含大量最终被判为 background 或 no-object 的 query。因此，SSCL 不应直接作用在所有 top-k query 上，而应作用在 Hungarian matching 后确认的 foreground matched queries 上。

推荐训练路径为：

$$
\text{encoder top-k queries}
\rightarrow
\text{decoder refinement}
\rightarrow
\text{Hungarian matching}
\rightarrow
\text{matched foreground query features}
\rightarrow
L_{\mathrm{SSCL}}
$$

## 4. SSCL 应该作用的位置

本方案建议将 SSCL 作用在 RF-DETR decoder 最后一层输出的 foreground query features 上。

设 decoder 最后一层输出为：

$$
H = \{h_i\}_{i=1}^{N_q}, \quad h_i \in \mathbb{R}^{d}
$$

其中 $N_q$ 是 query 数量，$d$ 是 hidden dim。RF-DETR 分类头根据 $h_i$ 输出分类 logits：

$$
z_i = W h_i + b
$$

Hungarian matcher 会根据分类代价、bbox L1 代价和 GIoU 代价，为 GT 目标选择匹配 query。设匹配到 GT 的 foreground query 集合为：

$$
\mathcal{F} = \{(h_i, y_i)\}_{i=1}^{N_f}
$$

其中 $y_i$ 是该 query 匹配到的 GT 类别。

SSCL 应作用在 $\mathcal{F}$ 中的 $h_i$ 上，而不是作用在：

- 所有 top-k query 上。
- background query 上。
- `class_embed.weight` 分类头权重上。
- 最终经过 sigmoid 后的分类 score 上。

这样做的意义是直接改善类别特征空间的几何结构，让容易混淆的类别在 decoder query feature 空间中分得更开。

## 5. 语义相似度矩阵构建

CLIP 用于构建类别语义相似度矩阵 $\Phi$。

对于每个类别 $c$，准备文本 prompt $p_c$，通过 CLIP text encoder 得到文本向量：

$$
t_c = \mathrm{CLIPTextEncoder}(p_c)
$$

任意两个类别 $c_a$ 和 $c_b$ 的语义相似度为：

$$
\Phi_{a,b}
=
\frac{t_{c_a} \cdot t_{c_b}}
{\|t_{c_a}\|_2 \|t_{c_b}\|_2}
$$

得到：

$$
\Phi \in \mathbb{R}^{C \times C}
$$

其中 $C$ 是前景类别数。矩阵 $\Phi$ 是对称矩阵，对角线为 $1$。

为了更适合训练，建议对原始 CLIP cosine 相似度做后处理：

$$
\Phi'_{a,b} =
\mathrm{NormalizeToRange}(\Phi_{a,b}, 0, 1)
$$

或使用温度缩放：

$$
\Phi'_{a,b} =
\mathrm{Softmax}\left(\frac{\Phi_{a,b}}{\tau_{\mathrm{text}}}\right)
$$

实际训练中不建议让所有类别对都使用强约束。可以只保留最容易混淆的 top-k 类别对，或只对舰船内部类别对启用较大权重。

## 6. Prompt 设计建议

不建议只使用 `HM`、`LQS`、`QHS`、`MS` 等缩写作为 prompt。CLIP 对缩写的理解不稳定，容易产生无意义或错误的语义相似度。

每个类别建议准备多个 prompt，然后平均文本向量：

$$
t_c =
\frac{1}{M}
\sum_{m=1}^{M}
\mathrm{CLIPTextEncoder}(p_{c,m})
$$

prompt 应包含：

- 类别完整名称。
- 类别所属大类，例如 ship、aircraft、vehicle。
- 遥感视角描述，例如 satellite image、aerial image、overhead view。
- 可见外观描述，例如 small vessel、large vessel、military aircraft、launcher vehicle。

对于无法准确解释的代号类，建议人工补充类别含义。如果没有可靠语义描述，则不应让 CLIP 矩阵对该类别产生过强约束。

## 7. SSCL 损失形式

这一节先把公式里每个字母是什么讲清楚，再一步一步拆开看它在做什么。普通监督对比学习的目标是：**让同一个类别的 query 特征互相靠近，让不同类别的 query 特征互相远离**。SSCL 只在"远离"这一步加了语义先验：**两个类别在语义上越像（比如 HM 和 LQS 都是军舰），越要用力把它们推开**。

### 7.1 符号含义对照表

| 符号                      | 含义                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $h_i$                     | decoder 最后一层输出的第 $i$ 个 matched foreground query 的特征向量，$h_i \in \mathbb{R}^d$。它就是该 query 在特征空间里的"坐标"。                                       |
| $u_i$                     | 把 $h_i$ 做 L2 归一化后的单位向量：$u_i = h_i / \|h_i\|_2$。只保留"方向"、去掉"长度"，因为对比学习只关心两个向量的夹角。                                                 |
| $u_i^\top u_j$            | 两个归一化向量 $u_i$、$u_j$ 的点积，**也就是它们的余弦相似度**（取值 $[-1,1]$，越大越像）。$^\top$ 表示转置，$u_i^\top u_j$ 即"把 $u_i$ 转过来和 $u_j$ 逐维相乘再求和"。 |
| $y_i$                     | 第 $i$ 个 query 匹配到的 GT 类别编号（SHWX 中是 0~24，比如 0=HM）。                                                                                                      |
| $P(i)$                    | anchor $i$ 的**正样本集合**：batch 中与 $i$ 同类别、且不是 $i$ 自己的所有 query。                                                                                        |
| $N(i)$                    | anchor $i$ 的**负样本集合**：batch 中与 $i$ 不同类别的所有 query。                                                                                                       |
| $\mathcal{A}$             | **有效 anchor 集合**：真正参与计算损失的 query 子集（即 matched foreground query 中"至少有一个同类正样本"的那些，详见第 8 节）。                                         |
| $\lvert\mathcal{A}\rvert$ | 有效 anchor 的个数，用于对每个 anchor 的损失取平均。                                                                                                                     |
| $\tau$                    | **温度系数**。放在指数里做除法：$\tau$ 越小，$\exp(u_i^\top u_j/\tau)$ 对相似度越敏感，正负样本拉得越狠（一般取 0.1）。                                                  |
| $\omega_{i,j}$            | **语义权重**：anchor $i$ 对负样本 $j$ 的"推开力度"，见 7.4。                                                                                                             |
| $\Phi'_{y_i,y_j}$         | CLIP 语义相似度矩阵中，类别 $y_i$ 与类别 $y_j$ 的相似度值（后处理归一化到 $[0,1]$，越大越像）。                                                                          |
| $\rho$                    | 语义放大系数，控制"语义像"对推开力度的放大程度。                                                                                                                         |
| $\omega_{\max}$           | 语义权重上限，防止 $\omega$ 过大导致训练不稳定。                                                                                                                         |
| $\exp(\cdot)$             | 指数函数 $e^x$。把相似度分数映射成正数，作为"该样本在分母里占的权重"。                                                                                                   |

### 7.2 公式分块拆解

把式子抄下来，一行对应一个作用：

$$
L_{\mathrm{SSCL}}
=
- \frac{1}{|\mathcal{A}|}
\sum_{i \in \mathcal{A}}
\log
\frac{
\underbrace{\sum_{j \in P(i)} \exp(u_i^\top u_j / \tau)}_{\text{① 同类吸引力总和}}
}{
\underbrace{\sum_{j \in P(i)} \exp(u_i^\top u_j / \tau)}_{\text{① 同类吸引力总和}}
+
\underbrace{\sum_{j \in N(i)} \exp(\omega_{i,j} \, u_i^\top u_j / \tau)}_{\text{② 异类排斥力总和（带语义权重）}}
}
$$

- **① 分子（= 分母第一项）：同类吸引力总和。** 对 anchor $i$，把和它**同类**的所有 query 的相似度 $u_i^\top u_j$ 经过指数放大后加起来。值越大，说明 $i$ 离同类越近。损失希望它尽量大（同类抱团）。
- **② 分母第二项：异类排斥力总和。** 把和 $i$ **不同类**的所有 query 的相似度乘上语义权重 $\omega_{i,j}$ 后经过指数放大再加起来。值越大，说明 $i$ 和异类纠缠越严重。损失希望它尽量小（异类分离）。
- **③ 整体结构：一个"二分类交叉熵"。** 比值
    $$
    \frac{\text{同类吸引力}}{\text{同类吸引力} + \text{异类排斥力}}
    $$
    可以理解为"在所有候选样本里，$i$ 认为自己是同类的概率"。用 $-\\log(\\cdot)$ 取负对数后，**这个比值越接近 1，损失越小**。所以最小化损失 = 让同类样本占绝对主导 = 同类拉近、异类推远。
- **④ 外层平均。** $-\frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}}$ 对全部有效 anchor 的损失取平均，得到 batch 级别的标量损失。

### 7.3 语义权重 $\omega_{i,j}$ 的作用机制

$$
\omega_{i,j} = 1 + \rho \cdot \Phi'_{y_i,y_j}
$$

- 若两个类别**语义很不像**（$\Phi'_{y_i,y_j}\approx 0$），则 $\omega_{i,j}\approx 1$，退化为普通对比学习——按正常力度推开。
- 若两个类别**语义很像**（如 HM 与 LQS，$\Phi'$ 接近 1），则 $\omega_{i,j}>1$。指数里变成 $\omega_{i,j} \cdot u_i^\top u_j$，数值更大 → $\exp(\cdot)$ 更大 → **这个负样本在分母里占的分量被放大** → 比值变小 → 损失变大 → 梯度更强地推动 $u_i$ 与 $u_j$ 分离。

一句话：**越容易混淆的类别对，在分母里占的比重越大，模型被迫把它们拉得越开**，这正是"语义相似度引导分离"的机制。

为避免 $\omega$ 失控，限制在保守范围：

$$
\omega_{i,j} \in [1, \omega_{\max}], \quad \omega_{\max}=1.5 \text{ 或 } 2.0
$$

### 7.4 数值小例子（便于对照）

假设 batch 里只有 3 个 matched query：$q_1$（HM）、$q_2$（LQS）、$q_3$（HM）。取 anchor $i=q_1$：

- 正样本：$P(q_1)=\{q_3\}$（另一个 HM）
- 负样本：$N(q_1)=\{q_2\}$（LQS）
- 假设余弦相似度 $u_1^\top u_3=0.7$（同类，较近）、$u_1^\top u_2=0.5$（异类，也较近，因为 HM/LQS 本来就像），$\tau=0.1$，$\rho=0.3$，$\Phi'_{HM,LQS}=0.8$。

则 $\omega_{1,2}=1+0.3\times 0.8=1.24$，指数项分别为 $\exp(0.7/0.1)\approx 1096$、$\exp(1.24\times 0.5/0.1)\approx 487$，损失为：

$$
L_{q_1}=-\log\frac{1096}{1096+487}\approx -\log(0.69)\approx 0.37
$$

若**没有**语义权重（$\omega=1$），LQS 项是 $\exp(0.5/0.1)\approx 148$，损失为 $-\log\frac{1096}{1096+148}\approx 0.13$——明显更小。可见**加了语义权重后，HM 与 LQS 这对易混类被要求分得更开**。

> 注：上式是"实例对实例"版本（负样本是 batch 里的其他 query）。若引入类别原型作锚点，只需把 $P(i)$/$N(i)$ 替换为"各类别原型向量"，公式结构完全不变。

## 8. Anchor 与样本选择策略

SSCL 不应对所有 query 使用，只应对 matched foreground query 使用。推荐规则如下：

- 只使用 Hungarian matching 后匹配到 GT 的 query。
- 忽略 background/no-object query。
- 如果一个 batch 中某个类别只有一个 foreground query，则该 query 没有同类正样本，可跳过或使用跨 batch memory bank。
- 优先使用最后一层 decoder query feature。
- 可选地对中间 decoder layer 加辅助 SSCL，但第一版不建议这样做。

为了让 HM、LQS 等小样本类有正样本对，建议使用以下方式之一：

- 增大 batch size 或有效 batch size。
- 使用 class-balanced sampler，让每个 batch 尽量包含多个同类实例。
- 使用 feature memory bank 存储历史 foreground query features。
- 使用多尺度或数据增强生成同类正样本。

第一版建议不引入复杂 memory bank，先通过采样策略保证 batch 内存在足够正样本。

## 9. 对 RF-DETR 训练参数的影响

如果 SSCL 作用在 decoder query feature 上，那么它会对产生这些 query feature 的共享模块产生梯度。因此，这个方案不再是“只训练 HM/LQS/QHS/MS 分类头行”的方案。

为了让 SSCL 真正改变特征空间，需要至少解冻以下模块之一：

- decoder 最后一层。
- decoder 后两层。
- decoder 最后一层加分类头。
- 可选的 encoder proposal 分类头。

不建议第一版解冻 backbone。DINOv2 backbone 参数量大，直接解冻容易过拟合，也更容易影响已有基类指标。

推荐第一版解冻范围：

- 冻结 backbone。
- 冻结 encoder 主体。
- 冻结 bbox head。
- 解冻 decoder 最后一层。
- 解冻 `class_embed`。
- 可选解冻 `transformer.enc_out_class_embed`。

如果希望更强保护基类，可以冻结 base class rows，仅允许 novel rows 更新；但此时 decoder 解冻后仍可能改变基类 query feature，因此仍需基类蒸馏。

## 10. 是否需要所有类别一起微调

建议使用 SHWX 全类别一起微调，而不是只使用 HM/LQS/QHS/MS。

原因如下：

- SSCL 需要不同类别之间的对比关系。
- 如果只看小样本类，模型不知道它们应该和哪些基类或相似类分开。
- 基类样本可以作为稳定的负样本参照，防止 novel 类特征漂移。
- 全类别微调有助于保持原有类别结构。

但全类别微调不意味着所有类别同等强度地更新。可以通过损失权重和采样策略控制：

- 检测损失对所有类别正常计算。
- SSCL anchor 可以偏向 HM/LQS/QHS/MS。
- SSCL negative pairs 可以重点覆盖舰船内部和易混类对。
- 基类通过蒸馏保护，避免指标下降。

## 11. 基类保护策略

一旦解冻 decoder，基类指标就不再天然保持不变。为了避免 aircraft、FSC 等基类下降，建议加入 teacher-student 蒸馏。

teacher 使用原始 RF-DETR checkpoint，完全冻结。student 是加入 SSCL 后训练的模型。

对同一 batch，teacher 和 student 都输出分类 logits。对基类 logits 做蒸馏：

$$
L_{\mathrm{distill}}
=
\mathrm{KL}
\left(
\sigma(z_{\mathrm{teacher}}^{\mathrm{base}} / T),
\sigma(z_{\mathrm{student}}^{\mathrm{base}} / T)
\right)
$$

其中 $T$ 是蒸馏温度，$\sigma$ 可以按 RF-DETR 当前分类损失形式选择 sigmoid 或 softmax。由于 RF-DETR 当前检测分类更接近多标签 sigmoid/focal loss 形式，第一版建议使用 sigmoid logit distillation 或 MSE logit distillation。

也可以对 query feature 做蒸馏：

$$
L_{\mathrm{feat}}
=
\|h_{\mathrm{student}} - h_{\mathrm{teacher}}\|_2^2
$$

但 feature 蒸馏需要保证 teacher/student query 对齐。第一版建议先使用 logits 蒸馏，工程风险较低。

总损失建议为：

$$
L_{\mathrm{total}}
=
L_{\mathrm{det}}
+
\lambda_{\mathrm{sscl}} L_{\mathrm{SSCL}}
+
\lambda_{\mathrm{distill}} L_{\mathrm{distill}}
$$

推荐初始权重：

- $\lambda_{\mathrm{sscl}} = 0.01$ 到 $0.05$。
- $\lambda_{\mathrm{distill}} = 0.5$ 到 $1.0$。
- SSCL 温度 $\tau = 0.1$。
- 语义负样本放大系数 $\rho = 0.2$ 到 $0.5$。

## 12. 对 HM/LQS/QHS/MS 的特殊处理

结合前期实验结果，不建议一开始强力优化所有舰船类。MS 和 QHS 当前不是纯小样本新类，而且它们是 FP 增加的主要来源。推荐第一版采用保守策略：

- HM 和 LQS 作为重点 anchor 类。
- QHS 和 MS 作为相似负样本类参与对比。
- 不对 MS/QHS 施加强 novel 类召回导向。
- 对 HM-MS、HM-QHS、LQS-MS、LQS-QHS 等类对施加更强分离。

这样可以避免再次出现 MS logits 被整体抬高，导致 MS FP 大量增加的问题。

可定义重点类集合：

$$
C_{\mathrm{focus}} = \{\mathrm{HM}, \mathrm{LQS}\}
$$

舰船易混集合：

$$
C_{\mathrm{ship}} = \{\mathrm{HM}, \mathrm{LQS}, \mathrm{QHS}, \mathrm{MS}\}
$$

第一版 SSCL anchor 集合可设为：

$$
\mathcal{A}
=
\{i \mid y_i \in C_{\mathrm{focus}}\}
$$

负样本优先使用：

$$
j \in N(i), \quad y_j \in C_{\mathrm{ship}}
$$

如果 batch 内舰船负样本不足，再使用其他类别作为普通负样本。

## 13. 与 query selection 的关系

SSCL 不直接改变 top-k query selection 的排序规则。它通过训练 decoder query feature，使被选中的 foreground query 在类别空间中更可分。

作用路径是间接的：

1. 当前 top-k selection 选择潜在目标 query。
2. decoder 输出 query feature。
3. Hungarian matching 找到正确 foreground query。
4. SSCL 拉开易混类别的 foreground query feature。
5. 后续训练中 decoder 和分类头逐渐形成更清晰的类别边界。

不建议第一版修改 top-k selection 逻辑。原因如下：

- top-k 基于 encoder proposal 最大分类 logit，是离散选择。
- 修改 selection 可能影响 recall。
- 当前主要问题是舰船 FP 和类间混淆，不一定是 proposal 没选中。
- 直接改 query selection 会扩大改动面，难以定位收益来源。

如果后续确认 proposal selection 对 HM/LQS 漏检有影响，再考虑 encoder proposal 层面的辅助损失。

## 14. 实施阶段建议

建议按以下阶段实施，避免一次性改动过大。

### 14.1 阶段一：离线构建语义相似度矩阵

为 25 个类别构建 prompt，使用 CLIP text encoder 得到 $\Phi$。人工检查矩阵中最相似的类别对，重点确认：

- HM/LQS/QHS/MS 之间的相似度是否合理。
- 舰船类和飞机类是否被明显区分。
- FSC 是否不会与舰船类过度相似。
- 缩写 prompt 是否导致异常相似度。

如果矩阵不合理，先改 prompt，不进入训练。

### 14.2 阶段二：只加入 SSCL，极小权重训练

使用原始 RF-DETR checkpoint 初始化。训练设置：

- 冻结 backbone。
- 冻结 encoder 主体。
- 冻结 bbox head。
- 解冻 decoder 最后一层和分类头。
- 使用 SHWX 全类别训练数据。
- SSCL 只作用于 matched foreground query。
- $\lambda_{\mathrm{sscl}}$ 从 $0.01$ 开始。
- 训练 1 到 3 epoch。

该阶段目标是验证 SSCL 是否能减少 HM/LQS 与其他舰船类混淆，而不是追求训练集或验证集 mAP 最大。

### 14.3 阶段三：加入基类蒸馏

如果阶段二出现 aircraft 或 FSC 指标下降，则加入 teacher logits distillation。teacher 使用原始 RF-DETR checkpoint。

该阶段目标是保护基类指标，避免 decoder 微调破坏已有能力。

### 14.4 阶段四：扩展 SSCL 范围

如果 HM/LQS 改善但幅度不足，可以逐步尝试：

- 将 anchor 从 HM/LQS 扩展到 HM/LQS/QHS。
- 对 decoder 后两层加入 SSCL。
- 引入 feature memory bank 增加正负样本数量。
- 对 encoder proposal features 加轻量 SSCL。

每次只改一个因素，避免无法判断收益来源。

## 15. 实验评估指标

评估时不能只看 val mAP。前期实验已经说明 val mAP 可能虚高，测试集 precision 可能恶化。建议记录：

- overall TP、FP、FN、precision、recall、F1。
- ship 大类 TP、FP、FN、precision、recall、F1。
- aircraft 大类 TP、FP、FN、precision、recall、F1。
- vehicle 大类 TP、FP、FN、precision、recall、F1。
- HM、LQS、QHS、MS 逐类 TP、FP、FN、precision、recall、AP。
- HM/LQS/QHS/MS 的 FP 可视化。
- 每图推理耗时。

模型选择建议使用约束式指标：

- aircraft F1 不低于基线。
- FSC recall 不低于基线。
- ship FP 不高于基线过多。
- HM/LQS precision 提升。
- overall F1 不下降。

可以设置硬约束：

$$
\mathrm{FP}_{\mathrm{ship}} \leq \mathrm{FP}_{\mathrm{ship}}^{\mathrm{baseline}} + \delta
$$

其中 $\delta$ 可先设为 $10$ 到 $20$。

## 16. 预期收益

本方案预期能改善：

- HM/LQS 与 QHS/MS 的类间混淆。
- 舰船内部细粒度类别边界。
- HM/LQS precision。
- HM/LQS AP。
- ship 大类 precision。

它对 recall 的提升应保持谨慎。由于第一版建议冻结 backbone 和 bbox head，目标发现能力不会发生根本变化。收益主要来自更清晰的分类特征，而不是更多 proposal 被选中。

## 17. 风险与不足

本方案存在以下风险：

- 如果 CLIP 语义矩阵本身不可靠，SSCL 会强化错误的类间关系。
- 如果 decoder 解冻过多，基类指标可能下降。
- 如果 $\lambda_{\mathrm{sscl}}$ 太大，模型可能过度追求特征分离，损害检测分类校准。
- 如果 batch 内同类正样本太少，SSCL 会不稳定。
- 如果 HM/LQS 在视觉空间中确实不可分，仅靠 SSCL 收益有限。
- 如果仍以 val mAP 选模型，可能再次出现验证集提升但测试集恶化。

因此，本方案必须配合：

- 语义矩阵人工检查。
- 小权重 SSCL。
- 保守解冻范围。
- 基类蒸馏。
- per-class threshold 重新校准。

## 18. 总结

SSCL 在 RF-DETR 中最合理的落点是 decoder 最后一层的 matched foreground query features。CLIP 不直接初始化分类头，也不参与在线推理，而是构造类别语义相似度矩阵，用来指导对比学习中负样本的分离强度。

该方案的核心价值是让模型把训练重点放在真正容易混淆的类别对上，例如 HM、LQS、QHS、MS 内部的舰船细粒度混淆。由于它作用在 query feature 空间，而不是只改分类头行，因此需要在 SHWX 全类别数据上进行轻量微调，并通过冻结 backbone、限制 decoder 解冻范围和基类蒸馏来保护原有基类指标。

推荐第一版实现保持保守：只解冻 decoder 最后一层和分类头，SSCL 只对 HM/LQS anchor 生效，QHS/MS 作为易混负样本参与，训练 1 到 3 epoch，并以 fixed-threshold F1、ship precision 和基类保持情况作为主要判断标准。
