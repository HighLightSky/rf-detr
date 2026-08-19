# RF-DETR DINOv2 多模态原型对齐与检测改进总结

## 1. 结论摘要

本轮实验回答了三个核心问题：

1. 原来的多模态原型并非完全无效，真正的问题是监督只作用在已经经过 top-k 选择的少量 query 上，encoder token 没有得到足够的类别对齐信号。
2. 使用全 encoder token 的 dense 原型监督后，原型空间获得了明确的类别判别能力；独立前景头又解决了“背景 token 也可能拥有很高语义相似度”的问题。
3. 原型分支参与 query selection 后，MS 的召回可以提升，但全局 residual 会改变 FSC 的候选分布。因此当前最好的平衡配方不是把原型权重无限增大，而是使用类别定向、低强度的选择信号，并让 decoder 最后一层适应新的候选分布。

当前建议保留两种配方：

| 用途 | 配方 | 测试集 total macro recall | MS recall | FSC recall | 结论 |
|---|---|---:|---:|---:|---|
| 默认平衡配方 | 24：decoder 适应 + MS 语义选择，`lambda_pos_max=0.25` | **0.9044** | 0.8137 | **0.8204** | 当前最适合作为默认候选 |
| MS 优先研究配方 | 23：decoder 适应 + MS 语义选择，`lambda_pos_max=0.50` | 0.8994 | **0.8219** | 0.8024 | MS 提升明显，但 FSC 有代价 |

24 的结果相对于同样进行 decoder 适应、但关闭 selection 的 22：

- total recall：`0.9005 -> 0.9044`，提升 0.39 个百分点；
- MS recall：`0.8144 -> 0.8137`，基本持平；
- FSC recall：`0.8084 -> 0.8204`，提升 1.20 个百分点。

23 则证明原型确实能够帮助小目标船舶：相对于 22，MS recall 提升 0.75 个百分点，但 FSC recall 下降 0.60 个百分点。这说明下一步应研究类别条件化选择，而不是继续做全局权重扫描。

## 2. 原版本的问题

### 2.1 原型辅助损失的监督范围过窄

原版本的 `loss_proto_labels_enc` 只对 selection 分支已经匹配为前景的少量 memory/query 计算类别均衡 CE。top-k 是离散排序，未被选中的 encoder token 没有直接梯度，因此投影层只能从一个很窄的样本子集学习。

其语义是“被选中的特征必须分类正确”，而不是“所有可能成为候选的 encoder token 都应被映射到正确的原型空间”。当原型分支尚未可靠时，selection 本身又会把错误样本送入损失，形成监督闭环中的弱启动。

### 2.2 原版本没有独立的前景/背景判别

纯类别原型 logits 只能说明 token 更像哪个类别，不能说明 token 是否是目标。背景纹理可能在某一类别上有较高相似度，进而被 selection residual 抬高。原版本只对正样本做 CE，这有助于避免大量背景导致类别塌缩，但也意味着背景原型分数没有被校准。

### 2.3 原型对 query selection 的作用路径很弱

原始 two-stage selection 仍以 encoder linear objectness 为主。即使原型 logits 改变，top-k 的离散排序也可能完全不变；而且 SHWX 验证集上 linear top-k 对 GT proposal 的覆盖率已经接近 100%，所以原型无法通过“找回完全没有 proposal 的目标”获得大幅收益，只能调整候选密度和候选质量。

### 2.4 训练/评估状态和 checkpoint 选择曾经干扰判断

dense 对齐只更新 `proj_token` 时，检测分支没有使用该参数，因此检测结果保持不变是预期行为。与此同时，按检测 mAP 保存的 `checkpoint_best_total.pth` 不一定是 dense 对齐最好的 epoch。现在实验均显式保存完整配置，并在独立测试时显式传入 checkpoint 路径。

## 3. 当前最终方案

### 3.1 原型库

当前实验使用：

```text
data/proto_guidance_shwx_1024_current.pt
```

它包含 25 类 SHWX 原型、每类 10 个视觉 slot 以及 CLIP 文本原型。随机初始选择与 K-means++ 初始选择在阶段一的检测结果逐位相同，说明当前主要瓶颈不是这两个初始化策略之间的差异。因此后续统一使用 `current`，避免把原型初始化和训练结构混在一起。

### 3.2 Dense encoder token 对齐

Transformer 输出 group 0 的所有 encoder token：

- `pred_proto_logits_dense: [B, N, C]`：每个 token 的类别原型 logits；
- `pred_proto_boxes_dense: [B, N, 4]`：与 token 对应的归一化预测框，统一为 `cxcywh`；
- `pred_proto_fg_logits_dense: [B, N]`：独立前景 logit。

正样本分配规则：

- proposal 与 GT IoU `>= 0.30`：分配给最大 IoU GT，施加类别 CE；
- IoU `< 0.10`：背景候选，进入前景 BCE 的 hard-negative 采样；
- 中间区间：忽略，不参与类别 CE；
- 小目标没有满足 IoU 阈值的 proposal 时，使用 GT 中心落入 proposal 的 fallback，最多选择 4 个 objectness 最高 token；
- 类别 CE 只对前景 token 计算，并按类别先独立求均值，再做类别间平均。

前景 BCE 每个前景 token 最多采样一个高 objectness 背景 token，防止背景数量压倒正样本。这样类别对齐和前景判别被解耦：类别 CE 学“是哪一类”，前景 BCE 学“是不是目标”。

### 3.3 多槽位融合

slot 聚合从硬 `max` 扩展为温度化 `logsumexp`：

```yaml
proto_guidance_slot_reduction: lse
proto_guidance_slot_reduction_tau: 0.1
```

硬 `max` 只有一个 slot 得到主要梯度；LSE 让多个有效视觉子原型共同参与训练，减少单 slot 独占和原型库局部塌缩风险。推理仍可以通过配置选择 `max` 或 `lse`。

### 3.4 Selection 分数

当前支持三种位置分数：

- `margin`：类别 top-1 与 top-2 原型间隔；
- `foreground`：独立前景头 logit；
- `foreground_semantic`：前景 logit 加指定目标类别的语义相对间隔。

目前对 MS 最有效的是：

```yaml
proto_guidance_target_classes: [3]
proto_guidance_position_score_mode: foreground_semantic
proto_guidance_position_semantic_weight: 0.25
```

selection 仍采用 residual 形式：

```text
select_score = linear_objectness + lambda_pos * calibrated_proto_score
```

原型分数先居中，并按 linear objectness 的标准差校准，避免原型 logits 的绝对尺度直接破坏原始 objectness 排序。

### 3.5 Decoder 适应

只有原型模块训练时，selection 改变了候选但 decoder 不一定能利用新候选。22/23/24 使用保守解冻：

- decoder 最后一层；
- decoder 最终 LayerNorm；
- `class_embed`；
- ProtoGuidance 模块。

backbone、encoder 主体、bbox 头和 decoder 前层保持冻结，保证收益主要来自“原型改变候选 + 检测器适应候选”这条路径。

## 4. 实验过程与证据

### 4.1 阶段一：原有 selected-query CE

`00/01/02` 的检测结果基本一致，`current` 与 K-means++ 原型库也没有可区分差异。结论是：原有 selected-query 监督没有为检测分支提供可见收益，不能继续单纯调大 `loss_proto_labels_enc`。

### 4.2 阶段二：dense CE

`10b` 只训练 `proj_token`，dense 原型准确率从约 27.5% 提升到最终约 71.5%，但 detection mAP/recall 不变。这不是加载失败，而是因为 position/content 关闭时 `proj_token` 不参与最终检测决策。该实验证明表示空间学好了，但没有使用路径。

### 4.3 阶段三：前景头和 selection

`13` 加入独立前景头，验证集前景准确率约 86%，正样本前景 logit 高于背景。`16` 使用 foreground selection 后：

- total recall `0.8899 -> 0.8924`；
- MS recall `0.8130 -> 0.8150`；
- FSC recall `0.7784 -> 0.7844`。

将强度增大到 `17` 后 FSC 回落，证明 selection residual 存在有效但有限的安全区间。

### 4.4 阶段四：MS 语义选择

`20` 的 MS 语义权重为 0.25、冻结 decoder，测试结果为 total `0.8925`、MS `0.8164`、FSC `0.7844`。`21` 将语义权重提高到 0.50 后回退到 total `0.8903`、MS `0.8137`、FSC `0.7784`。因此 0.50 不是更好的通用方案。

### 4.5 阶段五：decoder 适应

| 实验 | decoder 适应 | selection | total recall | MS recall | FSC recall |
|---|---|---|---:|---:|---:|
| 22 | 是 | 关闭 | 0.9005 | 0.8144 | 0.8084 |
| 23 | 是 | MS 语义，`lambda=0.50` | 0.8994 | **0.8219** | 0.8024 |
| 24 | 是 | MS 语义，`lambda=0.25` | **0.9044** | 0.8137 | **0.8204** |

这组实验是当前最重要的因果证据：原型 selection 能带来 MS 增益，但全局选择分数会和 FSC 发生竞争；降低 residual 能显著保护 FSC，并提高总体指标。

`25` 在本总结前被停止，未作为结论依据。

## 5. 相对原版本的代码改动

### 模型与损失

- `src/rfdetr/models/transformer.py`：输出 group 0 的 dense token logits、boxes、foreground logits；确保训练 dense 监督与 eval 推理路径一致。
- `src/rfdetr/models/criterion.py`：新增 IoU/中心 fallback 分配、类别均衡 dense CE、前景 BCE、背景 hard-negative 采样和诊断统计。
- `src/rfdetr/sscl/proto_guidance/guidance.py`：新增前景头、selection score 模式、LSE slot 聚合、位置分数标准差校准、显式原型 reload。
- `src/rfdetr/models/lwdetr.py`、`src/rfdetr/config.py`、`src/rfdetr/_namespace.py`：贯通新增配置字段和 loss 权重。
- `src/rfdetr/training/module_model.py`：支持 `token`/`token_fg` 可训练范围；加载 checkpoint 后重新覆盖实验指定的原型 buffer 和 content gate 初值；支持 decoder 适应。

### Checkpoint 与诊断

- `src/rfdetr/utilities/state_dict.py`：精简 checkpoint 保留 `model_config`，避免恢复后丢失 1024 分辨率和 ProtoGuidance 配置。
- `src/scripts/analysis/diagnose_dense_proto_assignment.py`：统计 proposal 覆盖、top-k 变化、正样本密度和按类选择准确率，区分“原型找不到目标”和“原型改变候选排序”两类问题。

### 测试与实验配置

- 新增 dense、前景头、空 GT、checkpoint 配置恢复和 group 0 一致性单测。
- `configs/experiments/proto_guidance_alignment_repair/` 集中保存 00--25 阶段实验，README 记录运行顺序和门控判据。

## 6. 下一步优化方向

### 6.1 优先实现可微 selection ranking loss

当前 dense CE 保证 token 类别分类正确，但没有直接优化候选排序。建议对同一图像构造 token 对：

- 正 token：IoU `>=0.30` 或中心 fallback token；
- 负 token：IoU `<0.10` 且 foreground/objectness 较高的 token；
- 约束：正 token 的 selection score 至少比负 token 高一个 margin。

可使用 logistic pairwise loss 或 soft top-k loss。该损失应只作用在 selection score，不直接改变 decoder 分类头，避免再次把类别 CE 与 objectness 混在一起。诊断重点是：

- GT top-k positive token 数量是否提升；
- top-k 的正样本密度是否提升；
- top-k GT coverage 已饱和时，是否减少重复/背景 token；
- overall、MS、FSC 是否同时不下降。

### 6.2 按类别或尺度使用 selection 强度

23/24 证明全局 lambda 不能同时优化 MS 和 FSC。下一步应实现按类别/尺度的门控：

- MS 使用小幅 semantic residual；
- FSC 默认只使用 foregroundness 或关闭语义 residual；
- 飞机类别保持线性 objectness，避免已高召回类别被扰动；
- 小目标可依据 box area 或 feature level 使用更高的 selection 系数。

这比继续扫描一个全局 `lambda_pos_max` 更有解释力。

### 6.3 优化原型库本身

现有 `current` 与 K-means++ 的差异很小，说明初始化不是主要瓶颈。后续应从数据质量和类别结构入手：

1. 按类别、目标尺寸、feature level 分层收集 DINOv2-P4 特征，避免大目标/易样本支配原型。
2. 对 MS、FSC、truck 与易混淆飞机对分别建立子原型，而不是只增加每类 slot 数。
3. 计算类内方差、类间 cosine margin、原型与验证 GT token 的 nearest-prototype accuracy，删除低质量或跨类污染 slot。
4. 使用已训练 `proj_token` 将离线特征重新投影后重建原型，保证原型库和训练后的 token 空间在同一坐标系。
5. 对每类保留视觉原型与文本原型的关系矩阵，检查 CLIP 文本先验是否把相近飞机类别错误拉近。

### 6.4 利用现有 SSCL

现有 SSCL 不应与 ProtoGuidance 直接拼接向量，因为二者来自不同阶段和不同特征空间。推荐通过共享类别关系和诊断指标协同：

- SSCL projection head 继续只约束 decoder matched foreground query，保护 bbox/分类共享特征；
- 使用 ProtoGuidance 的 dense token 类别关系作为 SSCL 的软正样本/类间温度先验；
- 对 MS/FSC 等目标类别降低过强的全局类间拉远，采用类别对条件的 margin；
- 对 SSCL 原型和 ProtoGuidance 原型比较类别关系矩阵，而不是比较向量绝对坐标；
- 只有在 pairwise selection ranking loss 稳定后，才把 selection 产生的高质量 token 作为 SSCL 的额外正样本。

### 6.5 利用现有 SSCL hard negative

SSCL hard negative 解决的是“像目标但其实是背景/局部干扰”的特征误报，ProtoGuidance foreground head 解决的是 encoder token 的前景性。二者可以形成互补闭环：

- dense foreground head 先筛选高 objectness、低 IoU 背景 token；
- SSCL hard-negative 模块从 matched query 之外选择 IoU 区间内的高相似 query；
- hard negative 只进入 SSCL 分母，不进入类别原型 EMA，防止污染类中心；
- 监控 `hn_count`、`hn_vs_random_gap`、`hn_vs_matched_gap` 与 ProtoGuidance 的背景前景间隔；
- 若 MS recall 下降，先降低 hard-negative 权重或扩大 IoU 忽略区间，不要同时提高 selection lambda。

建议的联合顺序：

1. 先完成 selection ranking loss 单独实验；
2. 在通过 recall 门槛后，加入 SSCL hard negative，保持 ProtoGuidance 强度不变；
3. 最后才评估 SSCL prototype/instance positive 与多模态原型的协同；
4. 每一步都保留 no-selection 与 no-hard-negative 对照。

## 7. 当前验收标准

后续方案只有同时满足以下条件，才应称为“多模态原型对检测产生稳定正作用”：

- dense encoder 原型 top-1 和前景/背景间隔稳定，不依赖单个 slot；
- query selection 的正样本密度提升，而不是只增加候选扰动；
- overall recall 不低于 decoder-adapt baseline；
- MS recall 至少提升 1 个百分点，或在不牺牲 FSC 的情况下达到稳定增益；
- FSC、飞机类别没有明显回退；
- current 与 K-means++ 原型库趋势一致；
- SSCL hard negative 的 hardness 指标通过预设阈值，且没有引起召回下降。

当前 24 已达到“整体平衡改进”的候选标准，但 MS 专项标准尚未达到 1 个百分点。因此应把 24 作为默认基线，把 23 作为 MS 研究对照，下一步重点实现 ranking loss、类别条件化 selection 和 SSCL/hard-negative 协同。
