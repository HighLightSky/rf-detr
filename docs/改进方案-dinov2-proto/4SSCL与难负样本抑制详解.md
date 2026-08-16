# SSCL 与难负样本抑制：原理与实现详解

> 配套文档：
> - [1视觉原型构建详解.md](1视觉原型构建详解.md) / [2文本原型与视觉原型对齐详解.md](2文本原型与视觉原型对齐详解.md) /
>   [3多模态原型库作用机制详解.md](3多模态原型库作用机制详解.md)（多模态原型，负责**召回入口**）
> - [RF-DETR-DINOv2多模态原型引导方案.md](RF-DETR-DINOv2多模态原型引导方案.md)（整体设计）
>
> 本文档讲解训练侧的另外两个判别机制：**SSCL**（decoder 层类别分离）与
> **难负样本抑制**（decoder 层背景虚警压制），实现主体在
> [sscl_loss.py](../../src/rfdetr/sscl/sscl_loss.py)、
> [hard_neg_selection.py](../../src/rfdetr/sscl/hard_neg_selection.py) 与
> [module_model.py](../../src/rfdetr/training/module_model.py)。

## 0. 核心结论

**SSCL**：对 decoder 最后一层 **Hungarian matched foreground query 特征**做**语义加权监督对比学习**——
与普通对比学习的区别在于，对 CLIP 语义上相似（易混淆）的异类负样本赋予更大分离权重，
把容易混淆的类别在 query 特征空间里拉得更开。**它管"进了 decoder 之后别认错"（判别）**。

**难负样本抑制**：从**未匹配且预测框与 GT 只有低 IoU、但前景分数很高**的 query 中选 top-k，
直接惩罚它们的高前景 logit——**它管"别把背景当目标"（虚警）**。

三者与 ProtoGuidance 的分工：

```text
ProtoGuidance（encoder 侧）→ 让目标进 decoder（召回入口）
SSCL（decoder 侧）       → 进来的 query 别和易混类混淆（类间判别）
难负样本抑制（decoder 侧）→ 进来的 query 别把背景当目标（前景-背景边界）
```

---

## 1. 全景：两者在训练循环中的位置

两者都挂在 criterion 的 forward 内（复用 Hungarian matching 的 indices，不重复匹配），
产出后进 loss 聚合：

```text
criterion.forward()
 ├─ Hungarian matching（enc 分支 + decoder 最后一层）
 ├─ [SSCL 回调] _sscl_loss_callback（module_model.py:692-739）
 │    ├─ 提取 matched query 特征 → sscl_loss(features, labels) → "loss_sscl"
 │    └─ （训练时）难负样本：选择 → 抑制损失 → "loss_sscl_hard_neg"
 └─ 主检测损失（box + class）
        │
        ▼
 loss = Σ loss_k × weight_k
（loss_sscl 权重 = sscl_lambda；loss_sscl_hard_neg 权重 = sscl_hard_neg_loss_lambda；
  两者各自受 start_epoch 门控，见 §5）
```

---

## 2. SSCL：语义加权监督对比学习

### 2.1 特征来源（监督什么）

[module_model.py:741-762](../../src/rfdetr/training/module_model.py#L741-L762)：
decoder 最后一层 hidden states `hs [B, Q, D]`，按 Hungarian matching 索引 gather
**matched foreground query** 的特征与 GT 标签 → `features [N_fg, D]`、`labels [N_fg]`。
只用 matched query：它们是最接近 GT 的"好样本"，类别语义可信，且数量少（每图目标数），
对比矩阵规模可控。

### 2.2 损失公式（实例对实例模式，当前 E 系列与 joint 实验所用）

对归一化特征 `u_i = h_i / ||h_i||`（[sscl_loss.py:214-284](../../src/rfdetr/sscl/sscl_loss.py#L214-L284)）：

```text
sim_ij = u_i^T u_j / τ                              # 带温度的余弦相似度
P(i)   = { j ≠ i : y_j = y_i }                      # 同类样本 = 正样本
N(i)   = { j ≠ i : y_j ≠ y_i }                      # 异类样本 = 负样本

L_SSCL = -1/|A| · Σ_{i∈A}  log(
              Σ_{j∈P(i)} exp(sim_ij)
              / ( Σ_{j∈P(i)} exp(sim_ij) + Σ_{j∈N(i)} exp(w_ij · sim_ij) )
          )
```

与标准 InfoNCE 的唯一区别在 **负样本语义权重**：

```text
w_ij = clamp(1 + ρ · S[y_i, y_j], 1, ω_max)

S = CLIP 类别语义相似度矩阵 [C, C]（minmax 归一化到 [0,1]，对角线为 1）
ρ = 放大系数（0.3）   ω_max = 权重上限（2.0）
```

**语义直觉**：如果 anchor 是 HM（航母），其负样本里 LQS（两栖舰）的 `S[HM, LQS]` 很高
→ `w > 1` → 负样本相似度被放大 → 分母变大 → 该负样本被推得更远。而飞机类负样本
`S[HM, A10]` 低 → `w ≈ 1`，只按普通负样本处理。**对比压力按语义相似度重新分配**。

两个过滤开关：

- `anchor_classes`（如 [0,1,2,3] 舰船类）：只对指定类别的样本做 anchor——对比损失
  的梯度集中投向关注类；
- `confusing_classes`（同 [0,1,2,3]）：只对"anchor 是这些类且负样本不同类"的配对
  施加语义放大，其余负样本权重保持 1.0（[sscl_loss.py:228-241](../../src/rfdetr/sscl/sscl_loss.py#L228-L241)）。

数值实现要点：logsumexp 稳定形式（避免 exp 溢出）；`loss = log(denom) − log(分子) ≥ 0`
保证非负；batch 内同类样本不足（无有效 anchor）或 `N_fg < 2` 时返回
`features.sum() * 0.0` 零损失——**保持计算图连接**（DDP 下各参数仍收到梯度），
且不能用 `loss.sum()*0`（-inf×0 = NaN）。

### 2.3 原型锚定模式（可选，当前关闭）

`sscl_prototype_enabled=True` 时启用 `_prototype_forward`（[sscl_loss.py:286-419](../../src/rfdetr/sscl/sscl_loss.py#L286-L419)）：
正样本 = 本类 EMA 原型 slot，负样本 = 全部类别原型（同样语义加权），每个 anchor 恒有正负锚点，
摆脱 batch 内同类样本不足的零损失问题；投影头（`sscl_projection_enabled`）把特征映射到
低维对比空间，原型库同步住在投影空间，缓解对比压力对共享特征（同时喂 class_embed 与
bbox_embed）的直接冲击。

**当前 E 系列与 joint120/20 实验均使用 instance-to-instance 模式**（`sscl_prototype_enabled: false`），
即 §2.2 的公式——不要把它误称为"SSCL 视觉原型实验"。

### 2.4 冻结策略

- `sscl_freeze_strategy: conservative`：冻结 backbone/encoder/bbox 头/decoder 前几层，
  只解冻 decoder 末尾 N 层 + norm + class_embed + 附加模块（含 ProtoGuidance）——保证
  **SSCL 只通过 decoder 最后一层重塑 query 特征空间，不扰动主干与定位能力**（用于已收敛
  checkpoint 微调，E 系列与 joint20 均为此）；
- `sscl_freeze_strategy: none`：全量微调，不冻结任何参数（联合训练 joint120 必须用它，
  否则 proto_guidance 开启时 conservative 会连带冻结 backbone）。

---

## 3. 难负样本抑制

### 3.1 选择器（谁算难例）

[hard_neg_selection.py:36-112](../../src/rfdetr/sscl/hard_neg_selection.py#L36-L112)，
逐图四个过滤条件，全部条件满足才算难例：

```text
① 排除 Hungarian 匹配到的 query            ← 真阳绝不碰
② 预测框与全部 GT 的 max_iou ∈ [0.0, 0.3]  ← 低 IoU 带
③ 目标类集合内最大前景 logit ≥ score_thresh（默认 -0.7）
④ 按分数 stable 降序取 top-k（默认 5）
```

- **① 的必要性**：DETR 是 1-to-1 匹配，一个 GT 只配一个 query——真实目标旁边的
  重复检测 query 天然未匹配，但它们**不是负样本**；
- **② 的双层含义**：上界 0.3 保护"真实目标的重复检测"；下界 0.0 纳入纯背景/低 IoU 区域——
  对准"纯背景虚警"靶点（港口、码头、道路、阴影纹理）；
- **③④**：只挑"高置信但没撞上目标"的 query，每图限量，避免背景 query 洪流淹没梯度。

### 3.2 为什么抑制 unmatched query 有用（原理）

一个常见的疑问：训练时被抑制的都是 unmatched query，"这些 query 反正不输出，压它有什么用？"
这个疑问基于一个误解——**"unmatched"只是训练时 Hungarian 分配的结果，不代表推理时
这个 query 不会输出**。拆开看：

**① 推理时没有匹配过程，所有 query 都会发射。** DETR 推理（[postprocess.py](../../src/rfdetr/models/postprocess.py)）
让全部 query 输出 logits 与框，按置信度阈值取检测——一个 query 训练时被标为 unmatched，
只是因为这一轮没被分配给任何 GT；只要它 logit 高，**推理时它就是一个检测框**。
测试结果就是直接证据：E4-hard-neg-noMS 里 MS 有 207 个 FP——这些框在训练时的某个
step 里就扮演着"unmatched 但前景分数高"的角色，被抑制的正是它们。

**② unmatched 里混着两类东西，抑制只瞄准其中一类：**

| 类型 | 例子 | 是否被抑制 |
| --- | --- | --- |
| 真实目标的重复检测 | 同一个 GT 被两个 query 都打中（IoU 0.6），但 1-to-1 匹配只许一个 | **排除**——IoU 带上界 0.3 就是为它设计的 |
| **像目标但不是目标** | 港口、码头、道路纹理、阴影，logit 高但和任何 GT 都不重叠 | **靶点**——这正是推理时的 FP 群体 |

"unmatched = 没用"的直觉错在：真正没用的是重复检测，而它已被 IoU 上界排除；
剩下被选中的是"会在推理时变成 FP"的 query。

**③ 降低"某个"query 的本质是更新共享参数，效果是泛化的。** logits 由**共享的
class_embed 作用在共享的 decoder 特征**上产生——惩罚 query k 的超额 logit，梯度更新
的是共享头与特征表示，影响所有 query、所有图像。语义是"这种'外观像目标但没撞上
GT'的模式，class_embed 不要再给它高分"，而这类模式（港口、甲板纹理、船影）在新
图像里会不断出现。这正是经典检测 **hard negative mining** 的思路：不平均地罚所有
背景，而是把最危险的几个挑出来重点罚。

**④ 与标准分类损失的区别。** 标准 DETR 分类损失确实把所有 unmatched query 当背景罚
（focal loss），hard-neg 是对它的显式补充：

- **难例挖掘**：标准损失对每图 200+ 个背景 query 的梯度分散；hard-neg 显式挑出 top-k
  个最危险的（高分 + IoU 带内），梯度聚焦；
- **margin 语义**：`softplus((logit − margin)/T)` 只在 logit 超过 margin 时才罚，是
  "封顶"——标准 focal 从 0 就开始压；
- **GT 几何信息**：IoU 带过滤让机制知道"这个 query 确定没撞上任何 GT"（纯背景虚警），
  这是标准逐 query 分类损失用不到的 GT 相对信息。

**⑤ 实验的双向验证。** E4-hard-neg 把 MS recall 从 0.7568 打到 0.5340，恰恰证明被抑制
的"unmatched"里混着真阳——MS 目标密集，很多真目标的 query 匹配时 IoU 只有 0.1~0.15，
被当成难例压掉了；noMS 排除 MS 后 recall 恢复（§6）。这说明训练时 unmatched ≠ 没用，
有些是"没匹配上但确实是目标"——选择器的 IoU 上界、分数阈值、目标类集合三个旋钮
直接决定它是"精准压虚警"还是"误伤真阳"。

### 3.3 抑制损失（怎么罚）

[module_model.py:838-904](../../src/rfdetr/training/module_model.py#L838-L904)：

```text
L_hn = softplus((max_logit − margin) / T) · T
     + λ_proto · softplus((max_cos − proto_margin) / T_p) · T_p   # 仅原型模式下可选
```

- **logit 抑制项**：难例的最大前景 logit 超过 margin（默认 -1.5）就惩罚，softplus 保证
  惩罚平滑且恒正——直接对"高置信背景"反传，把它的前景分数压回 margin 以下；
- **原型排斥项**：仅在 `sscl_prototype_enabled=True` 时生效——让难例特征远离前景原型
  （注意 [sscl_loss.py:469-484](../../src/rfdetr/sscl/sscl_loss.py#L469-L484) 的硬性约束：
  **原型库只喂 matched features，绝不可 EMA 进难例特征**——难例刻画的是背景分布，
  混入会污染类中心）。

**当前 joint20 配置的 v2 参数**（吸取 E4-hard-neg 教训，见 §6）：`λ=0.15`（减半）、
`margin=-2.0`（放宽）、`target_classes=[0,1,2]`（排除 MS）、`start_epoch=10`（晚启动）。

### 3.4 监控

[hard_neg_monitor.py](../../src/rfdetr/sscl/hard_neg_monitor.py) 每 `sscl_hard_neg_log_interval`
步采样，epoch 末输出 `train/sscl/*`：`hn_count`（每图平均难例数）、`hn_fill_rate`（IoU 带
填充率）、`hn_score_mean`/`hn_iou_mean`（选中难例的分数/IoU 均值）——用于判断难例
选择是否在工作、是否误伤真阳（score 过高 + iou 接近 0.3 上界时要警惕）。

---

## 4. 两者与 ProtoGuidance 的分工与一致性

| | ProtoGuidance | SSCL | 难负样本抑制 |
| --- | --- | --- | --- |
| 作用位置 | encoder 侧（top-k 前 + content 入口） | decoder 最后一层 | decoder 最后一层 |
| 特征来源 | encoder token + 离线多模态原型 | matched query 特征 | 未匹配高分 query |
| 监督信号 | 原型相似度（余弦/tau_p）+ aux CE | 语义加权对比损失 | logit softplus 抑制 |
| 分工 | **召回入口**（让目标进 decoder） | **类间判别**（易混类拉远） | **背景压制**（虚警降分） |
| 类别语义 | 离线原型（冻结） | CLIP 语义矩阵 S（冻结） | 目标类集合（配置） |

一致性要求（AGENTS.md）：三者必须统一**类别索引、数据集语义、prompt 版本**；向量空间
不同（encoder 原型空间 vs decoder query 空间），**不直接比较或拼接**，跨空间一致性
只能比较类别关系矩阵（`prototype_relation_alignment`）。

---

## 5. 权重装配与调度门控

- 权重：`weight_dict["loss_sscl"] = sscl_lambda`、`weight_dict["loss_sscl_hard_neg"] = sscl_hard_neg_loss_lambda`
  （[module_model.py:586-588](../../src/rfdetr/training/module_model.py#L586-L588)），恒定权重、无 ramp；
- 门控：`loss_sscl` 在 `sscl_start_epoch` 前置 0；`loss_sscl_hard_neg` 在
  `sscl_hard_neg_start_epoch` 前置 0（**两个门控已解耦**，见
  [module_model.py:1302-1321](../../src/rfdetr/training/module_model.py#L1302-L1321)）；
- 验证阶段：SSCL 回调只在训练前向执行（`self.training` 门控），推理零开销。

---

## 6. 实验教训（为什么 v2 参数这么定）

| 现象 | 数据 | 结论 |
| --- | --- | --- |
| SSCL 增益在 precision | E2-fixed vs E2-fixed-SSCL：FDR 0.2927→0.2656，recall 0.7782→0.7645 | SSCL 压混淆类 FP，代价是少量 recall——判别机制符合设计定位 |
| 难负样本全强度 + 早期 + 含 MS 是灾难 | E4-hard-neg：MS recall 0.7568→**0.5340**（-22pt），FDR 0.3029→0.0924 | logit 抑制把 MS 真阳当背景罚了；`margin=-1.5` 太激进、λ=0.3 太重、从 epoch 0 就施加、目标类含高频易误伤类 |
| 排除 MS 后恢复 | E4-hard-neg-noMS：MS recall 回到 0.7242 | 目标类选择是关键——高频类参与抑制 = 误伤面 |
| joint20 的应对 | λ=0.15、margin=-2.0、targets=[0,1,2]、start_epoch=10 | 减量 + 放宽 + 排除 + 晚启动，四重缓冲 |

---

## 7. 代码索引

| 环节 | 位置 |
| --- | --- |
| SSCL 损失（实例模式公式） | [sscl_loss.py:180-284](../../src/rfdetr/sscl/sscl_loss.py#L180-L284) |
| SSCL 损失（原型锚定模式） | [sscl_loss.py:286-419](../../src/rfdetr/sscl/sscl_loss.py#L286-L419) |
| 特征提取（matched query） | [module_model.py:741-762](../../src/rfdetr/training/module_model.py#L741-L762) |
| SSCL 回调装配与调用 | [module_model.py:692-739](../../src/rfdetr/training/module_model.py#L692-L739) |
| 难例选择器 | [hard_neg_selection.py](../../src/rfdetr/sscl/hard_neg_selection.py) |
| 抑制损失 | [module_model.py:838-904](../../src/rfdetr/training/module_model.py#L838-L904) |
| 语义矩阵构建/归一化 | [semantic_matrix.py](../../src/rfdetr/sscl/semantic_matrix.py) |
| epoch 门控（已解耦） | [module_model.py:1302-1321](../../src/rfdetr/training/module_model.py#L1302-L1321) |
| 难例监控 | [hard_neg_monitor.py](../../src/rfdetr/sscl/hard_neg_monitor.py) |
