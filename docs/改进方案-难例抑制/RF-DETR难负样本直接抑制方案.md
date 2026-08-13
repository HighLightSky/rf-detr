# RF-DETR 难负样本直接抑制方案（Hard-Neg Suppression）

> 对应实验配置：`configs/experiments/train_sscl_multproto_hardneg_suppress_v1.yaml`
> 相关代码：`src/rfdetr/sscl/hard_neg_selection.py`（难例选择）、`src/rfdetr/training/module_model.py`（抑制损失）、`src/rfdetr/sscl/sscl_loss.py`（原型库与投影头）
> 状态：v1/v2 三组实验已完成，机制被证实有效但存在结构性缺陷（见 §6），文档随实验结论更新。

## 1. 总体方案设计

### 1.1 要解决的问题

SHWX 测试集上，多 slot 原型（MultiProto）基线在 conf 0.28 下 ship 虚警严重：ship FP=193、total avgFDR=0.2903。`analyze_fp_decomposition.py` 的分解显示 ship FP 中 **56% 是纯背景虚警**（与任一 GT 的 IoU=0），且"像目标但其实是背景/局部干扰"的区域（码头、浪花、阴影、局部船体）无法被任何现有损失直接打击：

- **背景交叉熵**（DETR 对未匹配 query 的分类损失）压的是*相对概率*（让 background 概率更高），对"目标类 logit 与 background logit 同时偏高"的 query 约束不足；
- **SSCL 原型损失**压的是*特征空间*的类间边界，对无类别身份的背景区域"失明"；
- 两者都没有对"高分未匹配 query"（推理时虚警的直接来源）施加直接梯度。

### 1.2 核心思路

**用模型自己的"高分未匹配 query"当作虚警靶点，直接对它反传**：

```
每个 batch（训练态）：
  逐图选择难例（纯函数，读 pred_logits / pred_boxes / GT / matching 结果）
      └─ 选出 [batch_idx, query_idx]（保留梯度，不再是 detach 特征）
  两项损失（权重进 criterion.weight_dict，经现有 loss 聚合流程反传）：
      ├─ logit 抑制（主项）：压选中 query 的目标类 logits 到 margin 之下
      └─ 原型排斥（辅项）：把选中 query 的投影特征推离前景原型
  监控（每 log_interval 步采样）→ train/sscl/hn_* 曲线
```

### 1.3 与旧机制（v1 早期版本）的本质区别

| | 旧机制（SSCL 分母追加） | 新机制（直接抑制） |
|---|---|---|
| 难例用途 | detach 特征追加为 SSCL 损失分母列（权重 1.0） | 返回索引，对选中 query 直接加损失 |
| 梯度流向 | **无**（特征已 detach，只推离 anchor 特征） | **有**（梯度直达分类 logits 与 decoder 特征） |
| 对虚警的作用 | 间接（改变对比学习分布） | 直接（虚警由 logits 产生，就直接压 logits） |
| 依赖 | 必须开启原型模式 | 独立于原型模式（logit 抑制不依赖原型库） |

旧机制失效的根因：推理时的虚警由**分类 logits** 直接决定（conf = sigmoid(logits)），而旧机制对 logits 没有任何梯度路径。新机制把损失作用到产生虚警的 logits 本身。

## 2. 三个机制分别作用在 RF-DETR 的哪个结构上

RF-DETR 的相关结构（decoder 侧）：

```
                 RF-DETR 模型（decoder 部分）
  ┌────────────────────────────────────────────────┐
  │  backbone → encoder → decoder(多层)            │
  │                                │                │
  │                    outputs["hs"]（最后一层）     │
  │                       [B, Q, D]                │
  │                    ↓        ↓                  │
  │               分类头       框头                │
  │        pred_logits [B,Q,C+1]  pred_boxes [B,Q,4]│
  └────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
     ①难例选择        ②logit抑制       ③原型排斥
    读 logits/boxes   压 logits         投影特征 vs 原型
    （只读不写）      （写梯度）        （写梯度，弱）
```

| 机制 | 作用的结构 | 数据流 | 是否改网络结构 |
|---|---|---|---|
| **① 难例选择** | **分类头输出 `pred_logits` + 框头输出 `pred_boxes` + Hungarian matching 结果**。只读取这三个量 + GT，决定"选谁" | 逐图纯函数 `select_hard_negatives_for_image`，返回 batch/query 索引 | 否（纯数据流，训练回调内完成） |
| **② logit 抑制** | **分类头的输出 logits**（选中 query 的 `pred_logits[batch_idx, query_idx][:, target_classes]`）。梯度经分类头反传，同时影响 decoder 与 backbone（共享参数） | `_hard_negative_suppression_loss` 内直接构造损失项，经 weight_dict 聚合 | 否（损失层，无新参数） |
| **③ 原型排斥** | **SSCL 投影头之后的对比空间**。难例的 `outputs["hs"]`（decoder 最后一层）经 `SSCLLoss._project`（`ProjectionHead(hidden_dim, projection_dim)`）投影后，与原型库（EMA buffer）比较相似度。梯度只流向难例侧特征，**原型库本身 detach** | `sscl_loss.prototype_bank.get_normalized_slot_prototypes()` 返回无梯度归一化原型 | 否（复用现有投影头与原型库） |

要点：

- **① 和 ② 是同一组索引的两次使用**：① 选出索引，② 用索引 gather logits 计算损失。难例选择本身不产生梯度（纯函数），抑制损失才产生梯度。
- **③ 与 SSCL 主损失共享投影头与原型库**：`_project` 与 SSCL anchor 用的是同一个 `ProjectionHead`；原型库只由 **matched foreground** 特征更新，难例只参与排斥损失、绝不进库。
- **三个机制都不新增网络参数**，全部通过现有 loss 聚合流程（weight_dict）注入。

## 3. 三者的原理

### 3.1 难例选择（`hard_neg_selection.py`）

**目标**：选出"训练时高分、但没匹配到 GT"的 query——这些正是推理时越过 conf 阈值成为虚警的直接来源。

**规则**（逐图）：

```
未匹配 query（Hungarian 匹配补集）
  → IoU 带过滤：max IoU(预测框, 任一 GT) ∈ [iou_low, iou_high]
      下界 0.0：纳入纯背景（对准 56% 纯背景虚警靶点）
      上界 0.3：排除真实目标的重复检测（DETR 1-to-1 匹配下部分真目标
               会成为未匹配 query，绝不能当负样本——召回保护）
  → 分数 = pred_logits[:, target_classes].max()   # target 类内最大原始 logit
      只算 target_classes（当前 [0,1,2,3] = 4 个少样本船类），
      忽略 background 列与其他前景类——"只打击想打击的类"
  → 过滤 score >= score_thresh（-1.0，对应 sigmoid ≈ 0.27）
  → stable 降序取 top-k（每图最多 5 个）
```

**为什么选"分数"而不是别的**：score_thresh=-1.0 对应的 sigmoid 概率 0.269 恰好低于测试 conf 阈值 0.28（对应 logit≈-0.944）——**选择窗口与虚警窗口对齐**：训练时被选的 query 几乎正好是推理时会变成虚警的那批。被压之后 logit 低于 -1.0 就不再被选，机制自带"自我衰减"（收敛性）。

**注意**：难例分数是 **raw logit**（不做 sigmoid），与推理 conf（全类 sigmoid 最大，含 background 列）是两个不同的量，对应关系需透过 sigmoid 换算。

### 3.2 logit 抑制（`_hard_negative_suppression_loss` 主项）

**目标**：把选中 query 的目标类 logits 压到 margin 之下，使它们在推理时 sigmoid 概率低于 conf 阈值，不再输出。

**公式**（每个被选 query）：

```
hardest = max over target_classes of logits[b, q]      # 该 query 的目标类最高 logit
loss = softplus((hardest - logit_margin) / logit_temperature) * logit_temperature
```

- 以 **softplus** 而非 hinge：logit 高于 margin 时惩罚接近线性（hardest − margin），低于 margin 后梯度按指数衰减（e^(x−margin)），**软边界**——不产生阈值处的不连续，收敛平稳；
- `logit_margin = -1.0`：目标 logit 的软上界（sigmoid 0.269，压在测试阈值之下留出余量）；
- `logit_temperature = 1.0`：温度越大曲线越平滑；趋近 0 时退化为硬 hinge；
- 整项权重 `sscl_hard_neg_loss_lambda = 0.3` 注入 `weight_dict["loss_sscl_hard_neg"]`，与检测损失直接竞争。

**为什么有效**：推理 conf = sigmoid(最大 logit)。CE 背景损失压的是 softmax 的*相对*分布（惩罚的是 background 概率不够高），对"bg logit=5、ship logit=4"这种绝对水平双双偏高的 query 惩罚不足（softmax 下 ship 概率仍有 0.27）；logit 抑制直接对目标类 logit 的**绝对水平**施加梯度，这是 CE 给不了的。

### 3.3 原型排斥（`_hard_negative_suppression_loss` 辅项）

**目标**：在特征空间把难例推开前景原型——虚警的视觉根因是"背景区域的特征贴向类别中心"，这项直接打击特征层面。

**公式**：

```
hn_features = ProjectionHead(hs[batch_idx, query_idx])        # 投影到对比空间，有梯度
sim = cos(hn_norm, 所有有效 slot 原型)                         # 与最近原型的相似度
proto_loss = softplus((sim.max() - proto_margin) / proto_temperature) * proto_temperature * proto_lambda
```

- 原型来自 `prototype_bank`（EMA buffer），**无梯度**——排斥只推难例，不拉原型，原型库仍只由 matched foreground 更新；
- `proto_margin = 0.1`、`proto_temperature = 0.1`：比 logit 项低一个量级的温度，刻意做成**近似硬 hinge**（相似度低于 0.1 几乎无梯度，超过才惩罚）；
- `proto_lambda = 0.05`：辅项权重，远小于 logit 项。

## 4. 参数说明（当前实验配置）

| 参数 | 当前值 | 作用阶段 | 语义 |
|---|---|---|---|
| `sscl_hard_neg_enabled` | true | 总开关 | 关闭后 weight_dict 不注册 `loss_sscl_hard_neg`，行为与纯 MultiProto 一致 |
| `sscl_hard_neg_target_classes` | [0, 1, 2, 3] | 选择+抑制 | 参与计分与抑制的前景类；None = 全部 |
| `sscl_hard_neg_iou_low / iou_high` | 0.0 / 0.3 | 选择 | 候选与任一 GT 的最大 IoU 区间 |
| `sscl_hard_neg_score_thresh` | -1.0 | 选择 | 目标类最大 logit 下限（选择窗口下沿） |
| `sscl_hard_neg_topk` | 5 | 选择 | 每图最多选取数（stable 降序） |
| `sscl_hard_neg_loss_lambda` | 0.3 | 抑制 | logit 抑制 + 原型排斥总权重（weight_dict） |
| `sscl_hard_neg_logit_margin` | -1.0 | 抑制 | 目标 logit 软上界 |
| `sscl_hard_neg_logit_temperature` | 1.0 | 抑制 | softplus 温度（1 = 软，→0 = 硬 hinge） |
| `sscl_hard_neg_proto_lambda` | 0.05 | 排斥 | 原型排斥项权重（0 = 关闭） |
| `sscl_hard_neg_proto_margin` | 0.1 | 排斥 | 与最近原型的余弦相似度软上界 |
| `sscl_hard_neg_proto_temperature` | 0.1 | 排斥 | 排斥 softplus 温度（硬） |
| `sscl_hard_neg_log_interval` | 100 | 工程 | 监控采样步间隔 |

与 SSCL 主配置的边界：`loss_sscl` 只由 `sscl_lambda` 控制；`loss_sscl_hard_neg` 只由 `sscl_hard_neg_loss_lambda` 控制；两者同步受 `sscl_start_epoch` 门控。

## 5. 实验记录（v1 / v2 三轮）

全部在 conf 0.25 下测试（注意：基线为 conf 0.28，**对比存在阈值不一致**；v2-ship_only 另设逐类 FSC=0.30）。

| 指标 | 基线(0.28) | v1：ship+FSC，m=-1.5 | v2：去 FSC，m=-1.5 | v2：去 FSC，m=-1 |
|---|---|---|---|---|
| ship TP / FP / FN | 474/193/141 | 374/**56**/241 | 374/63/241 | 374/58/241 |
| ship recall | 0.7707 | 0.6081 | 0.6081 | 0.6081 |
| FSC(发射车) recall | 0.8077 | 0.5513 | 0.8205 | 0.8205 |
| FSC FP | 44 | 8 | 49 | 59 |
| total avgRecall | 0.8580 | 0.7326 | 0.8228 | 0.8069 |
| total avgFDR | 0.2903 | 0.1668 | 0.2819 | 0.2882 |

**结论**：

1. **机制生效**：`loss_sscl_hard_neg` 全程非零、`hn_score_mean` 后期稳定在 -0.4~-0.7（被选难例目标 logit 被压到低位）；FP 分层下降精确符合 `target_classes`（飞机类 FP 三轮纹丝不动）；
2. **靶向性验证**：发射车（类 24）从 target 移除后 FSC 立即恢复基线水平；
3. **margin 调参无效**：-1.5 → -1 结果逐字节相同（ship TP/FN/recall 全同），因被选难例 logit 集中在 -1~0，softplus 梯度在两种 margin 下几乎一样强；
4. **弱类伤 recall 是结构性的**：ship（少样本弱类，候选稀少）被压 query 往往是该 GT 的"孤点覆盖"，压掉 = 永久漏检（TP -100）；FSC（base 类，多框覆盖）压掉次优框只减重复 FP（FP -36 vs TP -20，净赚）。

## 6. 方案缺陷与已知问题

### 6.1 结构性缺陷（实验实证，调参救不回）

1. **弱类匹配池被直接缩减**：训练时"高分未匹配"的 query 是 ship 弱类最珍贵的匹配候选。压 logit → 匹配 cost 上升 → 更难被匹配 → 压掉 100 个就永久丢失 100 个检测能力。三轮实验 ship recall 稳定在 0.6081（验收线 0.75 未达标）。
2. **无法区分"纯背景虚警"与"GT 次优框"**：IoU 带 [0, 0.3] 同时包含两类候选。压 IoU≈0 的纯背景（不覆盖 GT，压掉永不产生 FN）是安全的；压 IoU 0.1–0.3 的次优框（真实目标的偏移覆盖，测试分布偏移时可能是唯一高分覆盖者）会产生 FN。**修正方向：收窄上界 `iou_high` 0.3 → 0.05~0.1**，只压准纯背景（注意：不是抬下界，抬下界效果相反）。此修正尚未实验验证。
3. **训练-测试语义错位**：训练时的"未匹配"由 Hungarian 匹配决定，测试时没有匹配、只有 conf 阈值竞争。被压 query 在训练时是冗余框，在测试时可能是唯一输出——机制对"训练时冗余"的惩罚会在测试时落到"唯一覆盖者"头上。

### 6.2 参数层面的缺陷

4. **margin 位置不是有效旋钮**（§5 结论 3）：被选难例 logit 分布的区间内，margin 移动几乎不改变梯度强度。要保 recall 必须改选择策略（收窄 IoU 带 / 提高 score_thresh / 降低 topk），不是改 margin。
5. **覆盖稀疏**：实测 `hn_count` 仅 0.08–0.16/图（每 batch 32 图只选 2–5 个），top-k 上限基本无压力；收窄 IoU 带后会更稀疏，机制覆盖总量下降。
6. **量纲混淆风险**：难例分数与 score_thresh 是 raw logit，推理 conf 是 sigmoid 概率——两者对齐依赖 sigmoid 换算，若推理后处理改为 softmax，margin 语义会完全失效（当前依赖"推理用 sigmoid"这一前提）。
7. **原型排斥项贡献极小**（实测 loss 0.014–0.031，相对 logit 项低一个量级），是否值得保留存疑，待 ablation（λ=0）验证。
8. **与背景 CE 部分重复**：对"目标类与背景 logit 双双偏高"的 query，CE 与 logit 抑制方向一致、职责重叠；对"bg logit 正常、目标类偏高"的 query 才是新机制的独有贡献——虚警分布中后者的占比决定机制上限。

### 6.3 流程/验证缺陷

9. **对比阈值不一致**：基线 0.28 vs 实验 0.25，且 v2-ship_only 对 FSC 设了 0.30 逐类阈值——严格对比需统一 0.28 重测。
10. **验收未达成**：v2-ship_only 距验收线最近（total avgRecall 0.8228 vs 0.84、total avgFDR 0.2819 vs 0.27、ship avgRecall 0.6546 vs 0.75），四条线中 ship recall 是唯一硬伤。

### 6.4 下一步优先级

1. `iou_high` 0.3 → 0.05~0.1（只压准纯背景，理论保 recall）——唯一未经实验但有明确机制依据的方向；
2. 弱类降 topk / 提 score_thresh，缩小匹配池损耗面；
3. 发射车保留在 target（净收益为正）；
4. proto 排斥 λ=0 ablation；
5. 统一 conf 0.28 重测定版。
