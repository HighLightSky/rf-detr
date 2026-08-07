# SGM 注意力门控反向优化：实验复盘与 attn_reg 弱监督方案

> 撰写日期：2026-08-07
> 范围：本文是 `RF-DETR引入SGM混合编码器改进方案.md` 的**实验验证篇**，聚焦"SGM 语义门控"这一根因的复盘与修复方案。结构设计（SPM/SGM/融合）见上一篇方案文档，本文件不重复。
> 相关代码：
> - SGA 实现：[src/rfdetr/models/backbone/sga.py](src/rfdetr/models/backbone/sga.py)
> - 变体短训实验：[src/scripts/test_sga/](src/scripts/test_sga/)
> - 注意力分析：[src/scripts/analyze_sga.py](src/scripts/analyze_sga.py)
> - 损失回调机制（SSCL 先例）：[src/rfdetr/training/module_model.py](src/rfdetr/training/module_model.py)、[src/rfdetr/models/criterion.py](src/rfdetr/models/criterion.py)

---

## 0. 方案摘要（TL;DR）

SGM 语义门控在 SHWX 数据集上**学反了方向**：注意力图在目标框内均值 **0.10**、背景均值 **0.73**（目标处比背景低 7 倍），把对舰船召回最有用的 SPM 高频纹理在目标处关掉，导致小目标召回下降。P0 的"保底门控 + 残差融合"修复只是**机械上**保证目标处 SPM 不再归零，但**没有修正注意力的极性**，因此门控始终是净负担（SPM-only 就能拿到几乎相同的 ship 召回增益）。

本文提出的修复方案 **attn_reg**：在检测 loss 之上增加一个**用 GT 框监督 SGM 注意力图**的辅助损失，直接给"目标处注意力过低"施加惩罚，把门控从"目标抑制器"掰回"目标增强器"。这是与本仓库现有 SSCL 损失完全相同的"回调注入 criterion"模式，改动面小、可开关、可短训快速验证。

**一句话定位**：这不是加结构，是给注意力补上"正确的监督信号"——检测 loss 只奖励最终检测精度，它天然会奖励"目标处关 SPM"（因为能少报虚警）；attn_reg 用 GT 框显式告诉注意力"目标处必须增强"，从而对抗这个坏的最优解。

---

## 1. 背景：SGM 注意力门控是用来干什么的

SGA 分支（Semantic Guiding Adapter）的结构是：**SPM 空间先验分支 + SGM 语义引导模块 + 融合层**。

- **SPM**：对原始图像做轻量 CNN 下采样，提供 DINOv2（patch16）拿不到的**原生 stride-8/16/32 高频纹理**，是小目标（船、发射车）最依赖的细节来源。
- **SGM**：取 DINOv2 **最深语义特征** `raw_feats[-1]`（[sga.py:261](src/rfdetr/models/backbone/sga.py#L261)）作为唯一引导源，蒸馏出多尺度 sigmoid 注意力图 `M_att`，再对 SPM 特征做逐元素门控：

```python
# sga.py _apply_gate：product 门控（原版）
gated_det = det * M_att            # 目标处 M→0 ⇒ 目标处 SPM 被关掉
```

- **融合层**：`cat([语义 feats[i], gated_det]) → 1×1+BN+GELU+3×3+BN → delta`，可选残差 `feats[i] + γ·delta`。

设计意图：让语义特征"告诉"SPM 纹理哪里该保留（目标处）、哪里该滤除（背景），以抑制遥感背景杂波、突出目标。**前提是注意力图的方向必须正确：目标处高、背景低。**

---

## 2. 已完成实验回顾

三批实验，时间顺序：**全量 SGA（100ep）→ 零训练猜想验证 → P0 修复五变体短训（10ep）**。

### 2.1 实验一：SGA 全量实验（`output/0805-SHWX-SGA-rfdetr`，100ep）

对比 Baseline（`use_sga=False`）与 SGA（`use_sga=True`，product 门控、非残差融合、`projector_scale=['P4']`），测试集比赛指标（conf=0.25）：

| 类别 | Baseline | SGA | Δ |
|---|---|---|---|
| all Recall | 0.9566 | 0.9493 | **−0.73pp** |
| all FDR | 0.1246 | 0.1195 | −0.51pp（FP 减少） |
| all F1 | 0.9141 | 0.9136 | −0.05pp（基本持平） |
| ship Recall | 0.7971 | 0.7805 | **−1.66pp** |
| vehicle Recall | 0.8571 | 0.7308 | **−12.63pp** |

**核心现象**：验证集上 SGA 几乎全面领先（val/ema_mAP_50_95 0.7545 vs 0.7495），但**测试集召回不升反降**。方向是典型的"Precision 略升、Recall 略降"（FP 减少 +24、FN 增加 +25），损失集中在舰船、车辆这些小目标。

### 2.2 实验二：零训练猜想验证（`analyze_sga.py`，纯推理不重训）

对两个 checkpoint 做阈值扫描 + SGM 注意力统计，把"为什么召回下降"从猜想收敛成实证：

1. **阈值扫描（否定"降阈值可追回"）**：SGA 在 conf∈[0.05,0.50] **所有**阈值下 all-Recall 都低于 Baseline（Δ恒为 −0.3~−1.0pp）。召回缺口是结构性的，不是置信度偏移。
2. **注意力统计（发现根因）**：

| 统计量 | 数值 |
|---|---|
| 目标框内注意力均值 | **0.0148**（几乎为 0） |
| 背景注意力均值 | 0.5457 |
| 框内 − 背景 | **−0.5308** |

**结论**：SGM 注意力**学起来了但方向反了**——它学会了在目标位置把门控压到 ≈0、背景保持 ≈0.55。这直接解释了机制：`det * M_att` 在目标处把 SPM 高频纹理几乎完全关掉 → 依赖纹理的小目标漏检（FN↑、Recall↓），同时目标处纹理噪声被抑制 → 虚警减少（FP↓、Precision↑），与测试结果方向完全吻合。

### 2.3 实验三：P0 修复五变体短训（`src/scripts/test_sga/`，10ep，seed=0，受控对比）

针对"目标处关 SPM"根因，按报告 §五做 P0 修复：**门控加保底**（下界/残差）+ **融合改残差**（保留 projector 语义基线）。五个变体：

| 变体 | 门控 | 融合 | 说明 |
|---|---|---|---|
| baseline | —（无 SGA） | — | 受控对照 |
| spm_only | `ones`（M≡1，无门控） | 残差 | 隔离"门控轴"：SPM+融合本身是否有用 |
| fixed_sga_lb | `lower_bound` `det·(0.5+0.5M)` | 残差 | P0 首选：下界保底 [0.5,1] |
| fixed_sga_res | `residual` `det+det·M` | 残差 | 更强保底 [1,2] |
| attn_bias | `product` `det·M` + logits 偏置 +2 | 残差 | 治本第一版：注意力初值≈全通，让它自己学成好门控 |

测试集结果（conf=0.25，比赛指标）：

| 变体 | all TP/FP/FN | all Recall | ΔRecall | FDR | ship R | veh R |
|---|---|---|---|---|---|---|
| baseline | 3170/930/221 | 0.9348 | — | 0.2268 | 0.7252 | 0.7051 |
| spm_only | 3180/948/211 | 0.9378 | +0.30 | 0.2297 | 0.7642 | 0.6667 |
| fixed_sga_lb | 3194/990/197 | 0.9419 | **+0.71** | 0.2366 | 0.7675 | 0.6923 |
| fixed_sga_res | 3168/975/223 | 0.9342 | −0.06 | 0.2353 | 0.7545 | 0.7179 |
| attn_bias | 3182/948/209 | 0.9384 | +0.36 | 0.2295 | 0.7805 | 0.6410 |

SGM 注意力统计（训练后的最终 checkpoint，672 测试图）：

| 变体 | 框内 M | 背景 M | 框内−背景 | 框内有效门控 |
|---|---|---|---|---|
| spm_only | 0.4398 | 0.4843 | −0.0445 | 1.000（M≡1） |
| fixed_sga_lb | **0.0991** | 0.7279 | −0.6289 | 0.5495 |
| fixed_sga_res | **0.1092** | 0.7278 | −0.6186 | 1.1092 |
| attn_bias | **0.1984** | 0.8906 | −0.6922 | 0.1984 |

> 注：ship 有 615 个 GT（召回差 1pp ≈ 6 框，可信）；vehicle 只有 78 个 GT（±6pp ≈ ±5 框，属噪声，不作为结论依据）。所有 SGA 变体 ship 召回均显著高于 baseline（+2.9~5.5pp），该信号跨变体一致。

### 2.4 三批实验的三个核心结论

1. **P0 保底机制修复成功（机械层面）**：`fixed_sga_lb` / `fixed_sga_res` 的框内有效门控分别 ≥0.55 / ≥1.11，不再出现原版"目标处有效门控≈0.015"的灭灯现象。目标处 SPM 至少保留一半。
2. **但 ship 召回的大头不是门控给的，是 SPM + 残差融合给的**：`spm_only`（门控完全固定 M≡1、注意力无梯度）ship 0.7642，与 `fixed_sga_lb` 的 0.7675 只差 0.3pp（噪声内）；四个 SGA 变体 10ep 验证曲线几乎重合。**门控模式不是杠杆，门控有无才是。**
3. **注意力极性依然反向，attn_bias 没治好它**：三个"读头有梯度"的变体框内 M（0.099~0.198）都比背景（0.73~0.89）低 7 倍以上；偏置 +2 只抬高了全局均值（0.668→0.825），框内−背景反而更反（−0.63→−0.69）。**"好初值"不足以阻止注意力收敛回反向门控。**

---

## 3. 原因分析

### 3.1 直接机制：SGM 读头学到了"目标处抑制"

SGM 的注意力生成器是自由训练、无任何约束的：`M = σ(Conv1×1(GELU(BN(Conv3×3(F_sem^L)))))`。它只受检测 loss 的间接梯度约束。观测事实：

- 读头**能**收到梯度的三个变体（lb / res / attn_bias），注意力**全部**反转；
- 读头**收不到**梯度的变体（spm_only，门控 M≡1 使 M 从计算图中脱离），注意力停在 0.48（不学、无空间结构）。

这构成一个天然的判别实验：**只要读头能学，它就学成反转**——反极化的推力主要来自读头自身的梯度，而不是别的因素。

### 3.2 为什么检测 loss 会"奖励"反转（SHWX 的特殊性）

对检测 loss 来说，SGM 注意力只是一个"门控函数"，它只关心最终检测精度。在 SHWX 上优化器发现：

- 目标（船、飞机、发射车）通常位于**高纹理背景**上：船在波浪海面、飞机在停机坪标线/跑道。这些位置叠加 SPM 高频纹理，会引入**目标周边的杂波虚警**（FP↑、loss↑）。
- 于是"在目标处把 SPM 关掉"能直接降低 loss —— 这是一个**以召回换精度的局部最优**：牺牲依赖纹理的弱小目标召回，换取整体精度。

这个解释与**注意力图与图像梯度的负相关**一致（抽样 8 图，`corr(M, grad)` ≈ −0.05 ~ −0.41）：注意力在平滑背景高、在高纹理区（恰是目标常处的位置）低。

### 3.3 数据证据：反转来自"读头"而非"编码器协同漂移"

一种常见的归因是"DINOv2 与门控联合微调、深层特征被带坏"。但 §2.3 的数据不支撑这是主因：

- 若主因是编码器协同漂移，那么在"读头能学"与"读头不能学"之间，编码器受到的梯度几乎一样（差异只在门控那条边），三个能学的变体应当与 spm_only 有可比性——但事实是**一学就反转、不学就中性**，干净地指向**读头自身的 loss 激励**。
- 因此，"detach `raw_feats[-1]` 切断编码器→门控梯度边"或"冻结 DINOv2"这类**源侧**修复，大概率无法翻正极性（读头从任意源上都能学到"目标处输出低"）。它们只能作为廉价的证伪实验，不能作为主方案。

### 3.4 与 SKYDET 的差异：论文不靠 loss，靠"冻结先验"

SKYDET 论文（`docs/参考论文/SKYDET.md`）**没有给注意力加任何损失函数**。它保证门控方向正确的机制是：

- **冻结整个 DINOv3 backbone**（§IV-D "the backbone parameters are kept frozen"），SGM 只从冻结的最深特征读取注意力；
- 论文论证 DINOv3 因自蒸馏预训练具有 **emergent object-centric localization** 先验（§III-B），即**引导源在训练前就指向目标**，SGM 只是这个冻结先验的轻量读出器，优化器无法把它拐坏；
- §G 的消融（5 seeds）显示冻结比全量微调更高且方差小 5 倍（0.726±0.001 vs 0.719±0.005），并把全量微调的坏解归因于"highly nonconvex optimized terrain"。

**为什么这套在本项目不直接成立**：① RF-DETR 的 DINOv2 是**COCO 检测微调**出来的（且 windowed 配置下位置编码与原生 DINOv2 不同，加载原生权重有 warning），没有 DINOv3 那种"自蒸馏目标中心先验"可冻结；② 论文在 DOTA/AI-TOD 上"目标处加纹理"并不明显伤精度，SHWX 恰恰相反；③ RF-DETR 依赖编码器微调，全冻结可能伤 baseline。**结论：我们不能靠"冻结先验"拿到正确极性，需要显式监督。**

### 3.5 已排除 / 已证伪的方向

| 假设 | 实验 | 结论 |
|---|---|---|
| "召回下降是置信度偏移，降阈值可追回" | 阈值扫描 conf∈[0.05,0.50] | **证伪**：所有阈值 recall 均低于 baseline |
| "SGM 没学起来，退化成随机分支" | 注意力统计 | **修正**：学起来了，但方向反了 |
| "下界门控能追回召回" | fixed_sga_lb | **部分成立**：all recall 最佳，但极性仍反、FP 最高 |
| "注意力初值全通能让它自己学好" | attn_bias（+2） | **证伪**：极性依旧反向，只是整体抬高 |
| "门控模式（lb/res/product）是主要杠杆" | 四变体 10ep | **证伪**：验证曲线几乎重合、ship 差异在噪声内 |
| "SPM 细节分支本身有用" | spm_only | **成立**：ship +3.9pp，最干净的正收益 |

---

## 4. attn_reg 弱监督方案

### 4.1 设计目标

给 SGM 注意力补上**检测 loss 缺失的监督信号**：显式告诉它"目标框内注意力必须高"。目标不是微调某个超参，而是**改变注意力的收敛方向**——从"目标抑制"翻成"目标增强"，让 SPM 纹理在目标处真正得到保留和增强。

### 4.2 主方案：框内下界 hinge 损失（推荐先做）

在现有检测 loss 之上增加一项：

```
L_attn = mean_pixels( max(0, floor − M) ⊙ box_mask )        # 逐像素 mean，仅在框内
total  = L_det + λ_attn · L_attn
```

- `M`：SGM 输出的 sigmoid 注意力图（P4 级，40×40 @640 输入）；
- `box_mask`：GT 框在注意力图分辨率下的二值掩膜（框内 1，其余 0），按 (cx,cy,w,h) 归一化框坐标换算；
- `floor`：框内注意力目标下界，建议 **0.7**（使有效门控在 product 模式下也能保留 70% SPM）；
- `λ_attn`：权重，建议从 **0.1~0.5** 起步（与 `loss_giou` 权重 2 同量级偏小，先不压过主 loss）。

**性质**：非对称 hinge——只在 `M < floor` 且框内处惩罚，背景不管。可微，梯度 = `−λ_attn·M(1−M)·box_mask`（M<floor 处）。空图（无 GT）时 `box_mask` 全零、loss 为 0，需按框内像素数归一化并防除零。

### 4.3 可选扩展（按需逐步加）

| 变体 | 公式 | 作用 | 建议 |
|---|---|---|---|
| **背景上限** | `+ λ_bg · mean( max(0, M − bg_ceil) ⊙ (1−box_mask) )` | 压低背景注意力，形成"框内高、背景低"对比 | 主方案跑通后再加；`bg_ceil` 建议 0.5 |
| **对比 margin** | `max(0, margin − (mean_fg(M) − mean_bg(M)))` | 直接对齐"框内−背景"指标 | 可作为第二种候选 |
| **logits 上的 BCE** | `BCEWithLogits(score_logits, box_mask)` | 最强约束（框内→1、框外→0） | 风险最高，易与检测任务打架，最后再试 |
| **多尺度** | 对 P3/P4 各算一份 | 缓解 40×40 对小目标的粗监督 | 配合 projector_scale 上 P3 时启用 |

### 4.4 为什么它能引导注意力学习（梯度机制）

当前注意力的收敛方向由两个力决定：

1. **检测 loss 的隐式梯度**：目标处抬高 M（多保留 SPM 纹理）→ 虚警增多 → loss 升高 → 梯度**压低**目标处 M；
2. **attn_reg 的显式梯度**：目标处 M < floor → 惩罚 → 梯度**抬高**目标处 M。

加入 attn_reg 后，读头的平衡点被强制抬高：只要目标处 M 低于 floor，就持续受罚，**再也塌不到 0.1 那种"几乎关门"的解**。当 `floor` 较高（0.7）时，门控从"目标衰减器"变成"目标增强器"，SPM 细节在目标处完整保留——这正是 SPM-only 已经证明有价值的路径，加上门控后有望在其之上产生真正的增量。

**与备选方案的关系**（源侧 vs 读头侧）：

| 方案 | 干预位置 | 能否对抗读头的反转激励 | 代价 |
|---|---|---|---|
| detach `raw_feats[-1]` | 源侧（切编码器→门控梯度边） | **不能**（读头照样能学反转） | 极低，一行 |
| 冻结 DINOv2 | 源侧（引导源恒定） | **不能保证**（读头能反转任意源） | 高，可能伤 baseline |
| **attn_reg（本文）** | **读头侧（直接对抗反转激励）** | **能，直接** | 低，加一项 loss |
| SPM-only（去门控） | 结构侧（删掉门控） | 不适用（无门控可纠） | 低，放弃语义引导 |

### 4.5 实现细节（复用仓库现成的 SSCL 回调模式）

本仓库已有**一模一样的机制**可照抄——SSCL 损失就是通过"回调注入 criterion"实现的：

1. **注册回调**：[criterion.py:646](src/rfdetr/models/criterion.py#L646) `set_sscl_loss_fn()`；在 [criterion.py:781-782](src/rfdetr/models/criterion.py#L781-L782) 的 `forward` 里（Hungarian 匹配之后）调用 `self._sscl_loss_fn(outputs, targets, last_layer_indices)`，返回的 `{"loss_xxx": 标量}` 自动并入 loss dict。
2. **注入权重**：[module_model.py:507-509](src/rfdetr/training/module_model.py#L507-L509) 把 `weight_dict["loss_sscl"] = cfg.sscl_lambda` 注入，训练循环自动按权重聚合。
3. **注意力图的获取**：attn_reg 比 SSCL 多一步——SGM 在 backbone 深处，注意力图不在 decoder 的 `outputs` 里，需要给 `SemanticGuidingModule` 注册一个 **forward hook** 存下每步的 sigmoid 注意力图（`analyze_sga.py` 的 `_AttnHook` 已是现成模板）。模型前向与 criterion.forward 每个微步一一对应，hook 存、回调取天然同步（含 grad_accum=4 的每个微步）。
4. **GT 框来源**：criterion 的 `targets` 里 `boxes` 是 `(cx, cy, w, h)` 归一化坐标（[criterion.py loss_boxes 注释](src/rfdetr/models/criterion.py)写明），换算到 40×40 生成 `box_mask`。

配置项（仿照已有 `sga_*` 字段，[config.py:532-535](src/rfdetr/config.py#L532-L535)）：

```python
sga_attn_reg_lambda: float = 0.0    # 0 = 关闭 attn_reg；>0 启用并加权
sga_attn_floor: float = 0.7          # 框内注意力下界（hinge 阈值）
sga_attn_bg_ceil: float = 0.0        # 0 = 不启用背景上限
```

### 4.6 实验设计与验证判据

建议在 `src/scripts/test_sga/` 加一个 `attn_reg` 变体（其余配置照 `fixed_sga_lb`：下界门控 + 残差融合），10ep seed=0 短训先行：

| 判据 | 目标值 | 判定 |
|---|---|---|
| 框内 M（训练后统计） | ≥0.5（显著高于现在的 0.099） | 极性翻正 |
| 框内−背景 | ≥0（目标处高于背景） | 门控方向正确 |
| ship Recall | ≥ spm_only（0.7642） | 不倒退 |
| all Recall | ≥ fixed_sga_lb（0.9419） | 或至少不低 |
| FP/FDR | 不显著高于 fixed_sga_lb | 不引入更多虚警 |

若成立 → 门控在正确极性下有增量，进全量微调 + 多 seed；若不成立 → 说明门控即使方向正确也难有增量，回落 SPM-only 路线，并保留 attn_reg 作为可选正则。

### 4.7 风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| 弱监督与检测任务打架 | 背景上限/BCE 形式可能过度约束 | 先用非对称框内下界；λ 从小起 |
| 小目标监督信号粗 | 40×40 图上 20px 目标只占 1~2 cell | 上 P3/P4 多级后再加细粒度约束 |
| 误分类框噪声 | GT 框含大小类边界误差 | 下界只约束"≥floor"，不追求精确边界 |
| 注意力只抬均值不学结构 | 可能整体抬高而失去目标选择性 | 配合背景上限/对比 margin 变体 |

---

## 5. 分阶段推进计划

| 阶段 | 内容 | 验证 |
|---|---|---|
| **S0（诊断，1 短训）** | 加 `attn_detach` 变体：detach `raw_feats[-1]` 喂给 SGM，证伪"编码器协同漂移为主因" | 极性是否翻正（预期不翻正，确认读头是主因） |
| **S1（主方案，1 短训）** | 加 `attn_reg` 变体：框内下界 hinge，λ=0.3、floor=0.7 | 框内 M ≥0.5、ship ≥ spm_only |
| **S2（调参）** | λ/floor 网格 + 可选背景上限 | 找精确率/召回最佳点 |
| **S3（确认）** | 全量训练（100ep）+ ≥2~3 seed，baseline/SPM-only/attn_reg 同 recipe | 测试集 ship/all recall 显著为正 |
| **S4（叠加）** | 确认后再上 P3+P4 多级 + CFE，attn_reg 同时多尺度监督 | AP_S / AP75 真提升 |

> 纪律：所有对比用相同 recipe 与 seed；盯测试集固定阈值指标（比赛口径），不能只看 val mAP（实验一已证明"val 涨、test 不涨"的坑）。

---

## 6. 实现文件清单

**修改**
- `src/rfdetr/models/backbone/sga.py`：无结构改动（可选加一行 `raw_feats[-1].detach()` 的 detach 变体路径）
- `src/rfdetr/models/criterion.py`：新增 `set_attn_reg_loss_fn` 回调（照 `set_sscl_loss_fn`），`forward` 内调用
- `src/rfdetr/training/module_model.py`：构建 `AttnRegLoss`、给 `SemanticGuidingModule` 注册 hook、注入 `weight_dict["loss_attn_reg"]`（照 SSCL 全套）
- `src/rfdetr/config.py` / `_namespace.py`：新增 `sga_attn_reg_lambda` / `sga_attn_floor` / `sga_attn_bg_ceil`
- `src/scripts/test_sga/common.py`：注册表加 `attn_reg` / `attn_detach` 变体

**新增**
- `src/rfdetr/training/attn_reg.py`：`AttnRegLoss`（hinge 框内下界 + 可选背景上限 + `box_mask` 生成）

---

## 7. 独立评估与落地建议（重要参考）

### 7.1 方案是否有价值

`attn_reg` 有明确的诊断价值和一定的工程价值，但目前还不能把它视为已经验证的最终改进。

已有实验已经证明：SGM 不是没有学习，而是稳定地学成了“目标处低、背景处高”的反向门控；并且这种反转只在注意力读头能收到梯度的变体中出现（见 §2.3、§3.1）。因此，给读头增加目标区域的显式梯度，比只改初始化、detach 引导特征或冻结 backbone 更直接。

但是，`M_fg > M_bg` 只是注意力统计上的成功，不等于检测指标一定提升。融合层仍可能抵消门控效果，或者目标纹理增强带来更多虚警。最终价值必须以固定测试集上的 `ship/all Recall`、FDR、AP_S 等指标相对于 `SPM-only` 的增量判断，而不能只看 `M_fg` 是否升高。

### 7.2 可能的副作用

1. **召回率与精确率冲突**：强行提高框内注意力可能重新引入纹理虚警，导致 FP、FDR 上升。
2. **注意力退化为全图高响应**：当前主方案只约束框内下界，不约束背景；模型可能通过整体抬高注意力来满足损失，失去空间选择性。背景上限或对比 margin 应作为后续实验，而不是第一版就加入。
3. **小目标监督过粗**：P4 的 40×40 注意力图上，小目标可能只有 1～2 个 cell，二值掩膜的量化误差会很大。必要时应改为逐框池化（或 ROIAlign），再考虑 P3+P4 多尺度监督。
4. **框内并不全是目标**：GT 框包含目标周围背景，重叠框和标注误差也会把不应增强的区域纳入监督；因此不建议第一版直接使用框内/框外 BCE。
5. **训练与部署分布差异**：训练时注意力受到 GT 框约束，推理时没有 GT；若正则过强，模型可能依赖训练期形成的目标区域先验而降低泛化。
6. **注意力极性并非可识别目标**：后续卷积融合层可能部分抵消门控，故“极性翻正”不能单独作为成功判据。

### 7.3 损失如何相加、如何归一化

概念上应为：

```text
L_total = L_det + λ_attn · L_attn
```

建议 `L_attn` 先对每个 GT 框独立求平均，再对有效 GT 框求平均：

```text
l_i = ReLU(floor - mean(M inside box_i))
L_attn = mean_i(l_i)
```

这样不同大小的目标权重相近，不会让大框主导梯度。无 GT 的图像返回 0，并防止除零；分布式训练还要保证不同 rank 的有效框统计一致。

实现上，回调应返回未加权的 `loss_attn_reg`，然后像 SSCL 一样通过 `weight_dict["loss_attn_reg"] = λ_attn` 聚合。不要在回调和 `weight_dict` 中同时乘 λ，否则会重复加权。

需要特别检查手动优化路径：当前 `_compute_train_losses()` 会把 `weight_dict` 中的项按目标框归一化，而自动优化路径直接使用 criterion 返回值。新增损失必须明确自己返回的是“按框归一化的均值”还是“未归一化的分子”，并分别验证自动优化、手动优化、梯度累积和 DDP，不能机械照抄 SSCL 回调。

### 7.4 λ 与 floor 的起始策略

- `floor`：先试 `0.5`、`0.7`，不建议一开始就逼近 1.0。
- `λ_attn`：先从 `0.05` 或 `0.1` 起步，再试 `0.2/0.3`。
- 监控加权后的 `λ_attn L_attn` 占总损失的比例，建议早期约 5%～20%。
- 同时监控 SGM 参数上的辅助梯度与检测梯度范数，建议辅助梯度约为检测梯度的 10%～30%。
- 可在前 5～10 个 epoch 内 ramp-up，避免初始化阶段注意力读头被强行锁死。

### 7.5 更有判别力的实验矩阵

文档 §5 中的 `attn_reg` 变体若直接复用 `fixed_sga_lb`，会因为下界门控已经保留了至少一半 SPM 而弱化注意力的实际作用，难以证明“正确极性的门控”是否有增量。建议至少比较：

| 变体 | 目的 |
|---|---|
| `SPM-only + residual` | 确认空间先验分支的独立收益 |
| `lower_bound + residual` | 机械保底基线 |
| `product + residual + attn_reg` | 直接检验注意力正则能否带来增量 |
| `product + residual` | 判断无正则时的反向门控代价 |

所有变体使用相同 recipe 和 seed，先短训筛选，再用至少 2～3 个 seed 做全量确认。若 `product + residual + attn_reg` 仍不超过 `SPM-only`，应接受“SPM 有价值、语义门控没有额外收益”的结论，回落到 SPM-only 路线，而不是继续堆叠背景 BCE、多尺度和 CFE。

---

## 参考

- SKYDET 论文全文：`docs/参考论文/SKYDET.md`
- SGA 结构方案（Phase 1/2 设计）：`docs/改进方案-SGM-encoder/RF-DETR引入SGM混合编码器改进方案.md`
- 全量实验报告：`output/0805-SHWX-SGA-rfdetr/实验报告.md`（含 §六 猜想验证）
- P0 修复五变体报告：`output/0806-0807-SHWX-test_sga-*/实验报告.md`
- 实验框架：`src/scripts/test_sga/`（common.py / run.py / README.md）
