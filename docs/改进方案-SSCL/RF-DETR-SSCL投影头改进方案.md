# RF-DETR SSCL 投影头改进方案

## 1. 方案目标

在现有 SSCL（语义相似度引导的监督对比学习）基础上，引入**投影头（ProjectionHead）**：把 decoder 输出的 matched foreground query features 先映射到低维对比空间，再在该空间内施加对比损失。目标是缓解对比压力对共享特征（同时喂给 class_embed 与 bbox_embed）的直接冲击，避免对比学习迫使特征按对比几何剧烈变形而干扰分类与框回归分支。同时引入**实例正样本**（对齐论文 Eq.9），解决投影头随机初始化导致的类别原型冷启动不准问题。

参照论文：《Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection》（CVPR 2026）。转 Markdown 版见 `docs/参考论文/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.md`。

## 2. 背景：为什么需要投影头

SSCL 对比损失是个很强的约束——它要求同类完全聚拢、异类完全分开。当前实现直接作用在 `outputs["hs"]`（decoder 最后一层 hidden states）上，而这份特征同时喂给 `class_embed`（分类）和 `bbox_embed`（回归）。直接施加对比损失会：

1. **压缩同类内部方差**：对比损失想让同类聚成一个点，但检测任务恰恰需要同类内部保留大小/位置/长宽比差异，否则框回归退化。
2. **几何对抗**：对比损失在 L2 归一化球面 + 极低温度（τ=0.1）上运行的强约束，与 focal loss 想要的几何不匹配。
3. **同类语义聚合 vs 空间分离冲突**：把同父类/同类目标在共享特征上拉近，会导致框也聚在一起。

投影头提供可学习的缓冲层：对比压力先被投影头消化在低维空间，共享特征只被软约束。

## 3. 投影头设计

### 3.1 结构

两层 MLP：

```
Linear(in_dim, proj_dim) → ReLU → Linear(proj_dim, proj_dim)
```

- `in_dim = model_config.hidden_dim`（medium = 256）
- `proj_dim` 默认 128（低于 hidden dim）
- 不引入 LayerNorm/BatchNorm（query 特征逐样本独立、batch 内样本少，BN 统计不稳定）

代码：`src/rfdetr/sscl/projection.py`

### 3.2 作用机制

```
解码器最后一层 hs ──┬──> class_embed / bbox_embed（读原始 hs，不变）
                    └──> ProjectionHead(降维) → L2 归一化 → SSCL 对比损失（在投影空间算）
```

- 分类/回归分支仍读原始 hs，只有对比损失在投影空间计算。
- 梯度仍会经投影头回流到 hs，只是被投影头这层非线性缓冲掉大部分对比压力。

### 3.3 与原型库的关系：原型住在投影空间

加了投影头后，损失在投影空间算 → anchor 在投影空间。那么"类的中心"（原型）也必须住在投影空间，否则 `sim(anchor投影, 原型原始)` 是两个坐标系的东西点积，无意义。

因此 `PrototypeBank` 的维度改为投影维度：

```python
hidden_dim = projection_dim if projection_dim is not None else hidden_dim
```

三者（正样本、负样本、原型）同处一个几何空间，这是机制成立的前提。

### 3.4 实例正样本（对齐论文 Eq.9）

论文 Eq.9 的正样本是 `P'(i) = 同类别实例 ∪ 本类原型`。当前原型模式原本正样本只用本类原型（自参照），冷启动时投影头随机、原型不准，自参照信号弱。实例正样本把 batch 内**真实同类实例**也作为正样本：

```
正样本 = { sim(anchor, 本类原型) } ∪ { sim(anchor, 同类实例1), sim(anchor, 同类实例2), ... }
```

- 真实同类实例提供 ground-truth 级引力，锚定投影头冷启动。
- 负样本仍为全部有效原型（语义加权），保持不变。
- 所有正项都在分母中（权重 1）→ 数学上 `loss >= 0` 恒成立。

## 4. 训练策略（冷启动与收敛性）

**结论：EMA 原型 + 对比损失协同适应是标准且收敛的机制（MoCo、本论文均验证），配三重稳定器 + 实例正样本锚定。**

1. **原型是无梯度 EMA 统计量**：随投影头演化而非固定真值，momentum 0.99 慢速变化，是稳定锚点；未就绪原型被 `valid_proto` 掩码排除，返回零损失，无 NaN 死锁。
2. **小 λ + 低 LR + 保守冻结**：只有 decoder 最后一层 + class_embed + 投影头可训练，λ=0.01~0.05、LR=1e-5，协同适应窗口的扰动被双保险吸收。
3. **实例正样本锚定**：真实同类实例从第一步提供正向引力，投影头快速学到有意义投影 → EMA 原型随之变准 → 良性循环。
4. **实验配置**：
   - 微调（train_sscl.py）：`SSCL_START_EPOCH=0`，投影头 + 原型 + 实例正样本从第 0 epoch 协同训练，原型约 1 个 epoch 内稳定。
   - 从头（train_sscl_all.py）：保持 `SSCL_START_EPOCH=30`，基类先收敛，到 30 后同时启动。

## 5. 副作用与风险

| 副作用 | 说明 | 检测/缓解 |
|---|---|---|
| 同类视觉差异大的实例被强行拉近 | 点对点引力是"硬目标"，可能压缩同类框回归差异 | 盯 GIoU / AP75 |
| 损失系统性变小 | 正样本同时进分子分母，`log(denom)-log(num)` 单调不增，梯度被稀释 | 必要时调大 λ |
| batch 组成方差回潮 | 实例正样本数量随 batch 变化，部分违背原型模式"batch 无关"初衷 | 观察 loss_sscl step 波动 |
| 冷启动收益是统计上的 | 随机投影下同类相似度仍带噪声，只是平均优于异类 | 投影头 LR 需足够，否则学不出来 |
| 损失数值不因投影头变小 | 低维空间随机向量余弦更分散，损失量级相当或略大 | 无 |

**关键提示**：真正的瓶颈更可能是**投影头 LR 太低**（当前与 decoder 共用 `tc.lr`=1e-5）。投影头从随机初始化学整个对比空间，可能需要比 decoder 高 10~100 倍的学习率（如 1e-4）。若 `train/loss_sscl` 曲线平着不动或升高，优先调投影头 LR，而不是 λ。

## 6. 实验方案

### 6.1 对照实验矩阵

| 实验 | 投影头 | 原型 | 实例正样本 | 验证目标 |
|---|---|---|---|---|
| 基线 | 关 | 开 | 关 | 现有 SSCL 原型模式的基准 |
| 实验 1（本次） | 开 | 开 | 开 | 完整方案：投影头 + 原型 + 实例正样本 |
| 实验 2（消融） | 开 | 开 | 关 | 实例正样本的独立贡献（对比实验 1） |
| 实验 3（消融） | 关 | 开 | 开 | 投影头的独立贡献（对比实验 1） |

每组**输出目录必须分开**（如 `output/0807-...-纯原型`、`...-Proj-原型+实例正样本`），否则无法归因收益来源。

### 6.2 训练脚本开关（src/scripts/train_sscl.py）

```python
SSCL_PROTOTYPE_ENABLED = True       # 原型锚定（主开关）
SSCL_PROJECTION_ENABLED = True      # 投影头（本次实验变量）
SSCL_PROJECTION_DIM = 128           # 投影空间维度
SSCL_PROTOTYPE_INSTANCE_POS = True  # 实例正样本（本次实验变量）
SSCL_START_EPOCH = 0                # 协同训练从第 0 epoch
SSCL_FREEZE_STRATEGY = "conservative"
```

### 6.3 评估指标（不能只看 val mAP）

- overall / ship / aircraft / vehicle 大类的 TP、FP、FN、precision、recall、F1。
- HM、LQS、QHS、MS 逐类 AP / precision / recall。
- **ship GIoU / AP75**（重点盯投影头与实例正样本对框回归的影响）。
- `train/loss_sscl` 曲线（健康应随训练下降；平着不动 → 投影头没学到）。
- HM/LQS/QHS/MS 的 FP 可视化。

### 6.4 预期收益

- HM/LQS 与 QHS/MS 的类间混淆改善（投影空间分离更干净）。
- 共享特征受对比压力冲击更小，框回归保持稳定。
- 实例正样本为常见类（QHS/MS）提供更强同类凝聚力。

## 7. 实现细节

### 7.1 新增/改动文件

| 文件 | 改动 |
|---|---|
| `src/rfdetr/sscl/projection.py`（新增） | `ProjectionHead` 两层 MLP |
| `src/rfdetr/sscl/sscl_loss.py` | 构造参数 `projection_dim` / `prototype_instance_pos`；`_project()` 辅助；原型库维度随投影；`_prototype_forward` 实例正样本；零损失图连接修复 |
| `src/rfdetr/config.py` | `sscl_projection_enabled` / `sscl_projection_dim` / `sscl_prototype_instance_pos` |
| `src/rfdetr/training/param_groups.py` | `get_projection_head_param_dict()` 收集投影头参数为独立参数组 |
| `src/rfdetr/training/module_model.py` | `_setup_sscl` 传参；`configure_optimizers` 在 requires_grad 过滤后追加投影头参数组 |
| `src/scripts/train_sscl.py` / `train_sscl_all.py` | 常量开关 `SSCL_PROJECTION_ENABLED` 等 |

### 7.2 两个关键工程细节

1. **投影头参数组必须手动追加进优化器**：投影头挂在 `sscl_loss` 上而非 LWDETR 内部，`get_param_dict` 只扫 model 的 `named_parameters()`，收集不到。须在 `configure_optimizers` 的 `requires_grad` 过滤**之后**追加（该过滤假定每组 params 是单个张量，而投影头参数组是列表，提前追加会抛 `AttributeError`）。
2. **零损失路径保持投影头在计算图中**：所有零损失返回从 `features.sum()*0.0` 改为投影后 `features.sum()*0.0`，否则投影头参数被踢出计算图，DDP 下触发"梯度未归约"报错。

### 7.3 冻结策略

`_apply_sscl_freeze` 只冻结 `self.model` 参数，投影头挂 `sscl_loss` 上天然可训练（conservative / none 两种策略下都如此），无需改冻结代码。旧 checkpoint resume 因 `strict_loading=False` 不报错，投影头从头初始化（预期行为）。

## 8. 测试

- `tests/models/test_sscl_projection.py`（新增，13 项）：投影头形状、投影+实例/原型模式的损失有限非负、原型库维度 == projection_dim、`projection_dim=None` 回归安全、`hidden_dim=None` 报错、零损失路径图连接、state_dict 往返。
- `tests/training/test_sscl_prototype_callback.py`（扩展）：投影回调下原型维度 == projection_dim、实例正样本回调。
- `tests/training/test_module_model.py`（扩展）：`get_projection_head_param_dict` 直测 + `configure_optimizers` 集成（投影头参数进入优化器）。

验证命令：

```bash
uv run --no-sync pytest tests/models/test_sscl.py tests/models/test_sscl_prototype.py \
  tests/models/test_sscl_projection.py tests/training/test_sscl_prototype_callback.py \
  tests/training/test_module_model.py -q -o addopts=""
```
