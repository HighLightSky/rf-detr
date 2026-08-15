# RF-DETR DINOv2 多模态原型引导改进方案

> 面向遥感小样本目标检测，参考《Multi-modal Prototype Guided Few-shot Object Detection》
> 的多模态原型引导思想，在 RF-DETR 的 DINOv2 backbone/projector 与 transformer decoder
> 之间加入原型引导的 query selection 与 content query enhancement，并保留现有 SSCL
> 作为易混类别判别约束。本文档是结构设计与训练方案，重点服务发射车等微小目标召回、
> 航母/两栖船等少样本舰船类召回，以及舰船细粒度类别混淆和虚警控制。

---

## 0. 总体结论

建议加入三个核心部分：

```text
DINOv2 backbone + projector
    ↓
多尺度 encoder/projector tokens
    ↓
位置 query 选择增强：原始 encoder 辅助头分数 + 多模态原型相似度分数
    ↓
top-k query selection
    ↓
内容 query 增强：选中 query 与对应类别原型做 cross-attention，并用 gate/residual 注入
    ↓
RF-DETR decoder
    ↓
原检测损失 + 现有 SSCL / 原型对比约束
```

其中：

- **位置增强**不直接修改坐标，而是改变 top-k 选中哪些 encoder token/proposal；
- **内容增强**在 top-k 后把选中 query 与其关联类别原型交互，给 decoder content query 注入类别先验；
- **SSCL**不负责召回入口，而负责 decoder 后的特征判别，拉开航母、两栖船、普通舰船、港口结构、车辆背景等易混对象。

---

## 1. 加入哪些模块，结构是什么样的

### 1.1 离线/在线原型库

新增 `DINOv2ProtoBank` 或同等模块，保存每个目标类的多模态原型：

```text
visual prototypes:
  P_v[c, m, d_v]     # 类 c 的第 m 个视觉子原型

text prototype:
  P_t[c, d_t]        # 类 c 的 CLIP 文本原型

fused prototypes:
  P_mm[c, m, d] 或 P_mm[c, d]
```

视觉原型建议从**已适配 RF-DETR 的 DINOv2/projector 特征空间**中提取，而不是直接使用 raw DINOv2
特征。原因是 RF-DETR 的 top-k selection 和 decoder cross-attention 实际消费的是 projector 后的
`hidden_dim` 特征，原型必须和这个空间对齐。

视觉原型生成流程：

```text
support images + boxes
    ↓
RF-DETR DINOv2 backbone/projector
    ↓
按 GT box 在多尺度 feature map 上做 masked average / RoI pooling
    ↓
每类 M 个子原型，EMA 或 Sinkhorn/soft assignment 更新
```

文本原型生成流程：

```text
class name + 遥感 prompt templates
    ↓
CLIP text encoder
    ↓
多 prompt 平均
    ↓
text projection 到 hidden_dim
```

多模态融合：

```text
H_v = proj_v(P_v)
H_t = proj_t(P_t)
T_v = CrossAttn(query=H_t, key=H_v, value=H_v)
g   = sigmoid(MLP([T_v, H_t, T_v - H_t, T_v * H_t]))
P_mm = g * T_v + (1 - g) * H_t
```

如果第一版希望更稳，可以先用简化版：

```text
P_mm = normalize(w_v * proj_v(P_v_mean) + w_t * proj_t(P_t))
```

后续再替换为论文式 gated fusion。

### 1.2 位置 query 选择增强模块

插入位置：`src/rfdetr/models/transformer.py` 中 two-stage query selection 的 top-k 之前。

原 RF-DETR 的选择逻辑是：

```text
output_memory_gidx = enc_output_norm(enc_output(output_memory))
linear_logits      = enc_out_class_embed(output_memory_gidx)
linear_score       = max(linear_logits, dim=-1)
topk(linear_score)
```

建议改为 residual scoring：

```text
proto_logits = tau_p * cosine(norm_proto_query(output_memory_gidx), norm(P_mm))
proto_score  = max(proto_logits over target classes)

select_score = linear_score + lambda_pos * proto_score
topk(select_score)
```

同时记录每个被选中 token 的原型关联类别：

```text
selected_class = argmax_c proto_logits[token, c]
```

这里的“位置增强”本质是 **prototype-guided position query selection**。原型分数本身不包含显式坐标，
但它改变 top-k 选中的 token index，而 RF-DETR 随后会从这些 token index gather 对应的
proposal box/refpoint 作为 decoder position query。因此它增强的是 position query 的**来源质量**。

### 1.3 内容 query 增强模块

插入位置：top-k 得到 `memory_ts/refpoint_embed_ts` 后，decoder 调用之前。

推荐结构：

```text
selected query content: tgt_q
selected class: c_q
class prototype slots: P_mm[c_q, 1:M]

proto_context = CrossAttn(query=tgt_q, key=P_mm[c_q], value=P_mm[c_q])
gate          = sigmoid(MLP([tgt_q, proto_context]))
tgt_q         = tgt_q + gamma_content * gate * proto_context
```

如果每类只有一个原型 slot，cross-attention 退化为近似 `proj(P_c)`，可以先实现轻量版：

```text
proto_context = proj(P_mm[c_q])
tgt_q = tgt_q + gamma_content * gate * proto_context
```

但针对航母、两栖船、多尺度舰船和不同视角发射车，更推荐保留多 slot 原型和 cross-attention：

- query 可以根据自身局部外观选择最相关子原型；
- 类内多形态不被简单平均抹平；
- 对航母/两栖船这种细粒度小样本类更友好。

gate 不是 cross-attention 的替代，而是 cross-attention 的稳定器。推荐最终形式是：

```text
tgt = tgt + gate * CrossAttn(tgt, prototype_slots, prototype_slots)
```

### 1.4 SSCL 原型升级

保留现有 SSCL 主体，但将其原型锚点升级为 DINOv2/多模态原型参与的稳定类别锚：

```text
matched decoder query feature
    ↓
projection head
    ↓
与类别原型做对比：正样本 = 本类原型，负样本 = 易混类原型/其他类原型
```

第一版有两种稳妥选择：

| 方式 | 说明 | 优点 | 风险 |
| --- | --- | --- | --- |
| A. DINOv2 原型初始化 SSCL EMA bank | 用 DINOv2/多模态原型初始化现有原型库，训练中继续 EMA 更新 | 适配现有代码最小 | 原型可能逐步漂向少样本噪声 |
| B. 冻结多模态原型 + 可学习投影到 SSCL 空间 | 原型作为冻结 anchor，SSCL projection 学会对齐 | anchor 稳定，少样本不污染 | 新投影需要额外训练信号 |

建议首版采用 A+B 的折中：

```text
P_sscl = normalize(beta * P_ema + (1 - beta) * proj_mm_to_sscl(P_mm))
```

`beta` 可 warmup，从更信任冻结原型逐步过渡到任务 EMA 原型。

---

## 2. 为什么这么设计，与原论文有什么差别

### 2.1 原论文的机制

MP-DETR 的核心链路是：

```text
多模态原型生成
    ↓
用原型 cross-attention 增强 encoder image features
    ↓
基于 token-prototype similarity 做 position query selection
    ↓
根据选中位置关联类别增强 content query
    ↓
prototype-based classifier + contrastive loss
```

论文中的位置增强包含两个动作：

```text
image tokens 与 prototypes 交互
token-prototype similarity 改变 top-k 位置选择
```

论文中的内容增强是：

```text
top-k 选中位置后，将该位置关联到最相似类别
用该类 prototype 增强 content query
```

### 2.2 RF-DETR 适配差别一：不直接替换原始 encoder 辅助头

论文更接近“用原型相关性主导 query selection”。RF-DETR 适配版首选 residual scoring：

```text
select_score = linear_score + lambda_pos * proto_score
```

原因：

- RF-DETR 已有预训练好的 `enc_out_class_embed`，它不只表达类别，也隐含 objectness/proposal 质量；
- 遥感图像背景纹理复杂，直接替换为相似度头容易把港口、道路、甲板局部纹理抬进 top-k；
- residual 形式初始可近似原模型，降低新增随机层破坏 query selection 的风险；
- `lambda_pos` 可以 warmup，便于观察召回和虚警的 trade-off。

因此，RF-DETR 适配版的第一阶段不是：

```text
topk(proto_score)
```

而是：

```text
topk(linear_score + lambda_pos * proto_score)
```

### 2.3 RF-DETR 适配差别二：先不强行修改整段 encoder memory

原论文在 top-k 前用 cross-attention 增强 image features。RF-DETR 首版可先把增强压缩成
“相似度 residual 分数”，不直接把 prototype context 写回所有 `output_memory`。

原因：

- 修改全部 memory 会影响 encoder proposal 分类、box proposal、decoder cross-attention 的共享输入；
- 新增 cross-attention 随机初始化，早期可能把 DINOv2/projector 的已适配特征扰乱；
- 分数 residual 的行为更可控，便于做消融：只改变 query 来源，不改变 feature 本身。

第二版可以更贴近论文：

```text
proto_context = CrossAttn(query=output_memory_gidx, key=P_mm, value=P_mm)
output_memory_proto = output_memory_gidx + gate_mem * proto_context
proto_score = similarity(output_memory_proto, P_mm)
```

但建议等 residual scoring 证明有效后再启用。

### 2.4 RF-DETR 适配差别三：保留线性分类头与现有 SSCL

论文最终分类器使用 prototype-based classifier。RF-DETR 当前分支已经有：

- decoder 后语义残差分类头；
- QNorm-Obj；
- SSCL；
- prototype logit calibrator；
- hard negative 抑制。

为了减少一次性改动的风险，首版不建议完全替换 `class_embed`。分类阶段仍保留 RF-DETR 的线性头，
用 SSCL 和可选 prototype logit residual 来增强判别。这样：

- query selection 负责提升少样本/微小目标进入 decoder 的概率；
- decoder 线性头保持预训练稳定性；
- SSCL 负责拉开易混类；
- 后续再比较是否需要把最终分类也换成 prototype similarity classifier。

---

## 3. 各模块作用和原理

### 3.1 多模态原型：提供“类是什么”的稳定锚

视觉原型负责细节：

- 发射车的细长车体、发射筒、阴影形态；
- 航母的长甲板、舰岛、甲板纹理；
- 两栖船与普通舰船的外形差异。

文本原型负责泛化：

- 少样本视觉原型容易被个别样本视角、尺度、成像条件带偏；
- CLIP 文本原型提供类别语义，尤其在 1-shot/5-shot 下可缓解视觉原型不代表的问题。

融合原型的作用：

```text
P_mm = visual detail + text semantic prior
```

它不是替代 detector，而是给 query selection、content query、SSCL 提供一致的类别先验。

### 3.2 位置 query 选择增强：让目标有机会进入 decoder

DETR 系列的 decoder 只能处理固定数量 query。如果微小目标或少样本类在 top-k 阶段没被选中，
后续分类头和 SSCL 再强也很难补救。

原始选择：

```text
score_i = max_c Linear(h_i)_c
```

原型引导选择：

```text
proto_score_i = max_c cosine(proj(h_i), P_mm_c)
score_i = score_i + lambda_pos * proto_score_i
```

效果：

- 提升与发射车、航母、两栖船原型相似的低置信 token 的排名；
- 保留原始线性分数，避免纯相似度把背景纹理误选为目标；
- top-k index 改变后，gather 出来的 proposal/refpoint 改变，所以 position query 的空间来源被增强。

这就是“位置增强”的 RF-DETR 体现。

### 3.3 内容 query 增强：让 query 带上“可能是什么类”的先验

top-k 只决定 decoder 看哪里，content query 决定 decoder 用什么语义方向去解释该位置。

在选出 token 后，通过 `selected_class = argmax proto_logits` 绑定每个 query 的候选类别，然后做：

```text
proto_context_q = CrossAttn(tgt_q, P_mm[selected_class_q], P_mm[selected_class_q])
tgt_q = tgt_q + gamma_content * gate_q * proto_context_q
```

作用：

- query 不只是“一个空间位置”，还携带“像发射车/航母/两栖船”的语义偏置；
- 多 slot cross-attention 可以处理同类多形态；
- gate/residual 保证新增模块初始不破坏预训练 query 分布。

这就是“内容增强”的 RF-DETR 体现。

### 3.4 SSCL：进去 decoder 后别认错

位置增强和内容增强主要提高召回，但它们也可能把更多难背景和相似类候选带进 decoder。
因此需要 SSCL 继续控制类间边界。

SSCL 的任务：

```text
matched query feature 靠近本类原型
matched query feature 远离易混类别原型
```

对遥感目标尤其重要：

- 航母 vs 两栖船；
- 两栖船 vs 普通大型舰船；
- 发射车 vs 车辆/道路/机场设施；
- 船舶目标 vs 港口结构、码头、阴影。

如果后续加 hard negative，则 SSCL 还可以补充前景-背景边界：

```text
matched query 远离高分 unmatched query / 难背景原型
```

---

## 4. 训练策略：避免新增层训练成 no-op 或扰乱预训练

### 4.1 风险来源

新增层包括：

- `proj_v/proj_t/proj_token`；
- prototype fusion gate；
- position prototype score 的 temperature/scale；
- content cross-attention；
- content gate；
- SSCL 原型投影或融合层。

主要风险有两类。

第一类是 **no-op 风险**：

- top-k 是离散选择，原型分数如果只影响排序，梯度可能弱；
- gate 如果初始化太小且没有辅助损失，可能长期接近 0；
- 原始线性分数很强时，`lambda_pos * proto_score` 可能完全改变不了 top-k。

第二类是 **扰乱预训练风险**：

- 新 cross-attention 随机初始化，直接写入 `output_memory` 或 `tgt` 可能破坏预训练 query 分布；
- 原型相似度过强会把背景纹理、港口结构、甲板局部拉进 top-k，引发 FP；
- 少样本视觉原型噪声大，训练早期如果权重大，容易把模型带偏。

### 4.2 初始化策略

所有新增路径都采用 residual + near-identity 初始化：

```text
select_score = linear_score + lambda_pos * proto_score
tgt = tgt + gamma_content * gate * proto_context
```

推荐初始值：

| 参数 | 初始 | 说明 |
| --- | --- | --- |
| `lambda_pos` | 0.0 或 0.05 | 从原模型起步，warmup 后逐渐使用原型分数 |
| `gamma_content` | 0.0 或 0.05 | 内容增强初始弱注入 |
| content gate bias | logit(0.05) 到 logit(0.1) | 不完全关闭，保留梯度 |
| fusion gate | 偏向文本或均衡 | 少样本早期视觉原型噪声大，文本可作为稳定锚 |
| prototype temperature | 可学习但 clamp | 防止相似度 logits 过大 |

不要把 gate 初始化成极端负值导致 sigmoid 接近 0，否则容易训练成 no-op。

### 4.3 给位置分支增加可导辅助监督

仅靠 top-k 排序间接影响最终检测损失，原型位置分支训练信号可能不足。建议给 `proto_logits`
增加 encoder/token 级辅助损失。

可选方案一：复用 encoder proposal 的监督。

```text
enc_outputs_proto_logits = proto_logits
与 enc_outputs 一样参与 Hungarian matching / focal loss
```

可选方案二：用 GT box 给 tokens/proposals 打软标签。

```text
proposal 与 GT IoU >= t_pos → 对应 GT 类为正
proposal 与 GT IoU < t_neg  → background/ignore
proto_logits 做 focal loss 或 BCE
```

首版建议用方案一，因为它最贴近现有 DETR 训练框架。

### 4.4 让原型分数真正改变 top-k

训练时需要监控：

```text
topk_overlap = overlap(topk(linear_score), topk(linear_score + lambda_pos * proto_score))
proto_selected_ratio = 被原型 residual 新带入 top-k 的 query 比例
target_recall_enc = GT 周围 proposal 进入 top-k 的比例
lambda_pos_effective = lambda_pos * std(proto_score) / std(linear_score)
```

如果 `topk_overlap` 长期接近 1.0，说明原型分支几乎没有改变 query selection，可能已经 no-op。
处理方式：

- 提高 `lambda_pos` 上限；
- 对目标少样本类做 class-balanced top-k；
- 对微小目标额外保留一部分 prototype top-k query；
- 对 `proto_logits` 加强 encoder 辅助损失。

### 4.5 分阶段训练

建议采用三阶段，避免新增层和预训练主干互相拖拽。

阶段 A：原型对齐与冷启动。

```text
冻结 DINOv2 backbone、projector、decoder、bbox head、class head
只训练：prototype projection、fusion gate、position scale、content cross-attn、content gate
使用 encoder proto auxiliary loss + 少量检测损失
lambda_pos/gamma_content 从 0 warmup 到小值
```

目标：新增原型分支先学会在 RF-DETR hidden space 中产生有意义分数，不追求最终 AP。

阶段 B：query selection/content query 联合微调。

```text
冻结 backbone 和 bbox head
解冻 decoder 后 1-2 层、decoder norm、class_embed、新增原型模块
启用完整检测损失
lambda_pos/gamma_content warmup 到目标值
```

目标：让 decoder 适应新增 query 分布，同时保持定位稳定。

阶段 C：加入 SSCL 做判别收敛。

```text
保留阶段 B 的可训练模块
启用 SSCL / prototype contrastive / hard negative
SSCL lambda 小值起步，必要时延迟若干 epoch
```

目标：减少航母/两栖船/普通舰船混淆，控制前景-背景虚警。

### 4.6 防止虚警放大的约束

位置增强会提高召回，但可能带来 FP。建议同步加入以下约束：

| 约束 | 作用 |
| --- | --- |
| residual scoring 而非替换 scoring | 保留原始 objectness |
| `lambda_pos` warmup + upper bound | 防止原型分数过强 |
| class-balanced prototype top-k 限额 | 防止某类原型独占 query |
| SSCL 易混类负样本加权 | 减少舰船细粒度混淆 |
| hard negative unmatched query | 压制港口/道路/局部纹理虚警 |
| base class distillation | 防止已有类能力回退 |

---

## 5. 推荐消融顺序

不要一次性打开所有模块。推荐按以下顺序验证：

| 实验 | 模块 | 目的 |
| --- | --- | --- |
| E0 | 原 RF-DETR baseline | 确认召回/虚警基线 |
| E1 | 只加 position residual score | 验证微小目标/少样本类是否更多进入 top-k |
| E2 | E1 + encoder proto auxiliary loss | 验证位置分支是否摆脱 no-op |
| E3 | E2 + content query gated cross-attn | 验证 decoder 是否能利用类别先验 |
| E4 | E3 + SSCL 多模态原型 | 验证易混类混淆是否下降 |
| E5 | E4 + hard negative | 验证背景虚警是否下降 |

每个实验至少记录：

```text
整体 mAP / AP50 / recall
发射车 AP / recall / FP
航母、两栖船、普通舰船混淆矩阵
top-k 原型带入比例
GT proposal 进入 top-k 的比例
SSCL matched feature 类间余弦
FP decomposition
```

---

## 6. 首版实现建议

首版最小可行实现：

```text
1. 离线构建每类 DINOv2/projector 视觉原型和 CLIP 文本原型
2. 加 PrototypeAdapter，将原型投影到 transformer hidden_dim
3. 在 transformer two-stage top-k 前加入 proto_score residual
4. 记录 selected_class，top-k 后对 tgt 做 gated prototype residual
5. 用多模态原型初始化或融合现有 SSCL prototype bank
6. 加监控，确认原型分支真的改变 top-k 且未显著放大 FP
```

首版不建议直接做：

- 完全替换 `enc_out_class_embed`；
- 完全替换 decoder `class_embed` 为 prototype classifier；
- 直接用随机初始化 cross-attention 改写全部 encoder memory；
- 一开始全量解冻 DINOv2 backbone。

这些都可以作为第二阶段实验，但不适合作为第一版验证。

---

## 7. 与原文位置增强/内容增强的对应关系

| 原文组件 | 原文含义 | 本方案对应实现 |
| --- | --- | --- |
| Position Query Selection | 原型增强/评分 encoder image tokens，选出更相关位置作为 position query | top-k 前 `linear_score + lambda_pos * proto_score`，改变被 gather 的 proposal/refpoint |
| Content Query Enhancement | 根据选中位置关联类别，用该类原型增强 content query | top-k 后 `tgt = tgt + gate * CrossAttn(tgt, P_mm[c], P_mm[c])` |
| Prototype-based Contrastive Classifier | decoder query 与原型相似度分类，并加对比损失 | 首版保留线性分类头，使用现有 SSCL + 多模态原型锚点；后续可消融替换最终分类头 |

一句话概括：

```text
位置增强负责“让目标进 decoder”；
内容增强负责“让 query 带着类别先验去解码”；
SSCL 负责“进来之后别和易混类/难背景混淆”。
```
