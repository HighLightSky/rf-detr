# 方法篇：多模态原型引导的遥感细粒度目标检测

> 本文档参照学术论文"方法（Methods）"部分的写作规范，完整描述本文算法：
> 以 RF-DETR 为基座，提出**多模态原型引导（Multimodal Prototype Guidance）**框架，
> 并集成语义加权监督对比学习（SSCL）与难负样本抑制，面向遥感图像小样本细粒度检测。
> 公式采用标准 Markdown 数学格式（KaTeX 兼容）；记号约定与完整公式索引见
> [0模型统一公式体系.md](0模型统一公式体系.md)，各模块实现细节见
> [1](1视觉原型构建详解.md)–[4](4SSCL与难负样本抑制详解.md) 号文档。

---

## 摘要式总览

遥感图像目标检测面临三类核心困难：（i）**小样本类别**——航母（HM）、两栖舰（LQS）等
类别训练实例极少，模型难以建立稳定类别表征；（ii）**细粒度混淆**——中型舰船（MS）、
轻型舰船（QHS）等外观相近，分类边界模糊；（iii）**背景虚警**——港口、码头、道路、
阴影等"像目标"的复杂背景产生大量误检。

本文方法的核心思想：**离线构建"视觉细节 + 文本语义"的多模态类别原型作为先验锚，
在 query 选择的两个关键位置注入先验**——位置引导改变"哪些 encoder token 进入 decoder"，
内容引导给"进入 decoder 的 query 注入类别语义偏置"；同时在训练侧叠加三级判别约束
（原型分类辅助损失、语义加权对比学习、难负样本抑制），并用**两阶段联合训练**使新增
模块与检测器共适应。消融实验表明：位置引导显著提升召回入口质量，SSCL 压缩易混类
虚警，难负样本抑制经目标类选择与相位化调度后可精准压制背景虚警。

---

## 1. 算法总体设计

### 1.1 问题定义

给定遥感训练集 $\{(\mathbf{x}_n, \{\mathbf{b}_{n,k}, y_{n,k}\}_k)\}$，其中 $\mathbf{x}_n$
为输入图像，$\mathbf{b}_{n,k} \in \mathbb{R}^4$ 为归一化边界框，$y_{n,k} \in \{1,\dots,C\}$
为类别标签（$C=25$，含舰船 4 类、飞机 20 类、地面车辆 1 类）。目标为学习检测器
$\mathcal{D}$，输出每个目标的位置与类别，在类别分布极端不平衡（最稀有类训练实例
个位数）且类间外观高度相似（舰船细粒度类）的条件下保持高召回与低虚警。

### 1.2 总体架构

本文方法在 RF-DETR 的两阶段检测框架上嵌入三个创新组件与两个判别模块，整体前向如下：

```text
输入图像 x
  ↓
① 骨干编码器：DINOv2 + MultiScaleProjector（P4）→ 特征图 F（40×40×256）
  ↓ 1×1 投影 → memory m（正弦位置编码在 P4 分辨率生成，但仅作用于 decoder，
  ↓ 两阶段选择路径不消费 PE）
  ↓
② 多模态原型库（创新 1）：视觉原型 + CLIP 文本原型 → 投影 → 融合 → P_mm
  ↓
③ 原型引导的 Query 选择（创新 2）：
   select_score = 线性 objectness + λ(t)·校准后的原型 margin → top-k
  ↓
④ 原型增强的内容 Query（创新 3）：
   q' = q + γ(t)·gate·槽位交叉注意力上下文（按原型关联类别）
  ↓
⑤ Decoder（4 层，自注意力 + 可变形交叉注意力）
  ↓
⑥ 分类头（sigmoid focal）+ 回归头（L1 + GIoU）
  ↓
训练期判别约束：原型辅助损失（创新 4）+ SSCL（创新 5）+ 难负样本抑制（创新 6）
```

其中 ①② 为 RF-DETR 基座（简述于 §2），③④⑤ 为本文创新组件（详述于 §3–§5），
训练侧判别约束详述于 §6–§7。

### 1.3 记号约定（本文使用的主要符号）

| 符号 | 含义 | 符号 | 含义 |
| --- | --- | --- | --- |
| $C, d$ | 类别数 25；隐藏维度 256 | $\mathbf{P}_v, \mathbf{P}_t, \mathbf{P}_{\mathrm{mm}}$ | 视觉/文本/融合原型 |
| $M$ | 每类槽位数 10 | $\mathrm{proj}_v, \mathrm{proj}_t, \mathrm{proj}_{\mathrm{tok}}$ | 投影层 |
| $K, N$ | query 数 300；token 数 1600 | $w_v, w_t$ | 融合权重 |
| $\mathbf{m}_i, \mathbf{z}_i$ | memory token 及其投影 | $\tau_p$ | 原型温度 0.2 |
| $s_i^{\mathrm{lin}}, s_i^{\mathrm{proto}}$ | 线性 objectness；原型 margin | $\lambda(t), \gamma(t)$ | 位置/内容注入权重（warmup） |
| $\mathcal{Q}, c_i^*, \kappa_i$ | 选中集；关联类别；原型置信度 | $\mathbf{q}_i, \mathbf{ctx}_i, g_i$ | 内容 query；上下文；gate |
| $\mathbf{h}_i, \mathbf{u}_i$ | SSCL 特征及其归一化 | $\mathbf{S}$ | CLIP 语义矩阵 |
| $\tau, \rho, \omega_{\max}$ | SSCL 温度/放大/上限 | $\sigma^{\star}$ | Hungarian 匹配 |
| $\lambda_{\mathrm{cls}}, \lambda_{\mathrm{bbox}}, \lambda_{\mathrm{giou}}$ | 检测损失权重 1.0/5.0/2.0 | $\lambda_{\mathrm{aux}}, \lambda_{\mathrm{sscl}}, \lambda_{\mathrm{hn}}$ | 辅助/SSCL/难例权重 |
| $m_{\mathrm{hn}}, T_{\mathrm{hn}}, \tau_{\mathrm{hn}}, k_{\mathrm{hn}}$ | 难例 margin/温度/阈值/top-k | $t, T_w$ | 当前 epoch；warmup epoch |

---

## 2. 基座模型：RF-DETR（简述）

RF-DETR 是基于 LW-DETR 架构的两阶段 DETR 检测器，本文仅简述其与本方法直接相关的
部分，完整介绍见 [0模型统一公式体系.md](0模型统一公式体系.md)。

**骨干编码器。** 骨干为 DINOv2-windowed-small（patch 16，输出 block 3/6/9/12 共 4 层
特征），经 MultiScaleProjector 在 P4 尺度（DINOv2 原生分辨率）上投影为
$d=256$ 通道特征图 $\mathbf{F} \in \mathbb{R}^{d \times H' \times W'}$（640 输入 →
40×40）。**该架构中不存在 transformer encoder 层**——骨干与投影即 encoder，特征经
1×1 卷积投影（$\mathrm{enc\_output}$）构成 memory $\mathbf{m} \in \mathbb{R}^{d \times N}$
（$N=1600$）。正弦位置编码在 P4 特征图分辨率上生成，但**不加入 memory**——仅作为
独立张量在 decoder 可变形交叉注意力内部加到 query/key 上，两阶段选择路径
（位置 proposal、线性打分、原型打分）均不消费 PE。token 级预测由两个头产生：
$\mathrm{enc\_out\_class\_embed}$（分类，$C+1$ 维）与 $\mathrm{enc\_out\_bbox\_embed}$
（回归）。

**两阶段 Query 选择。** 以线性 objectness $s_i^{\mathrm{lin}} = \max_c \hat{p}_i^{\mathrm{enc}}[c]$
对全部 token 排序取 top-$K$，选中的 token 提供 decoder 位置 query（refpoint），
其内容经 enc 头产生独立预测并参与监督（enc 分支损失）。

**Decoder 与检测头。** 4 层 decoder，每层含自注意力（8 头）与可变形交叉注意力
（16 头、每头 2 采样点），refpoint 逐层迭代更新。分类头为 sigmoid focal
（$\alpha=0.25$），回归头输出 cxcywh 增量并与 refpoint 相加得最终框。训练采用
Hungarian 匹配（代价权重 2.0/5.0/2.0），匹配后计算分类 focal 损失、$\ell_1$ 回归
损失与 GIoU 损失，并叠加 decoder 中间层与 enc 分支的同结构辅助损失。推理时 300 个
query 直接输出，按置信度阈值取检测。

**并行组解码。** 训练时按 $G=13$ 组并行解码（$K \cdot G$ 个预测共享同一组 GT 匹配），
推理时单组。

---

## 3. 创新模块一：多模态原型库（离线构建与在线融合）

### 3.1 动机

类别原型是"类是什么"的稳定表征。遥感小样本检测中，仅靠少量训练实例学出的类别
表征容易过拟合个别样本的视角与尺度；仅靠文本语义又缺乏视觉细节。本文将两者互补：
**视觉原型提供形态细节，文本原型提供语义泛化**，二者在可学习投影空间融合为统一的
多模态原型。

### 3.2 视觉原型离线构建

视觉原型由离线脚本在训练前一次性构建（与训练同分布的数据与同源骨干权重）：

1. **特征提取**：加载 120 轮全量微调后的 RF-DETR（与训练侧同源的 backbone 权重，
   保证特征空间一致），在骨干 P4 特征图（$d=256$）上按 GT 框区域做空间均值池化，
   每个 GT 框产生一个实例特征向量；
2. **聚类**：对每个类别 $c$ 的实例特征集合做**余弦 k-means**（$k=\min(M, N_c)$，
   20 轮迭代），得到 $M=10$ 个槽位子原型 $\mathbf{P}_v[c, \cdot] \in \mathbb{R}^{M \times d}$；
3. **有效性掩码**：实例数不足 $M$ 的类别，缺失槽位置 $\mathbf{V}_v[c,m]=0$（无效槽位
   不参与任何后续计算）。无实例的类别全部无效。

**设计要点**：① 余弦 k-means 与后续打分度量（余弦相似度）一致——聚类空间 =
打分空间；② 多槽位表达**类内多形态**（舰船不同角度、飞机不同机位），避免单原型
把多形态平均抹平；③ 均值池化无参数、无学习型 RoI 头，原型构建完全确定。

### 3.3 文本原型构建

对每个类别 $c$，用 CLIP 文本编码器（ViT-L）编码多条遥感提示词（含完整型号名、
外观特征、遥感视角描述），逐条取 EOS/pooler 表征、经文本投影后 L2 归一化，再对
多条提示词取平均，得 $\mathbf{P}_t[c] \in \mathbb{R}^{768}$。多提示词平均增强 CLIP
对类别语义的稳定理解。

### 3.4 在线投影与融合

视觉与文本原型维度和语义空间均不可比，因此各自投影到统一的 $d=256$ 维融合空间
后再融合：

$$ \mathbf{P}_{\mathrm{mm}}[c, m] = \mathrm{normalize}\Big( \mathrm{softplus}(w_v) \cdot \mathrm{proj}_v(\mathbf{P}_v)[c, m] + \mathrm{softplus}(w_t) \cdot \mathrm{proj}_t(\mathbf{P}_t)[c] \Big) $$

- $\mathrm{proj}_v$：Linear + LayerNorm（浅层适配——视觉原型与 token 特征同处
  backbone 特征空间）；**LayerNorm 在加权和之前、视觉分支内部执行**（逐槽位向量
  归一化）；
- $\mathrm{proj}_t$：FSemProjection（768→512→256，Linear→Tanh→Linear，**不含
  LayerNorm**）——文本空间与检测特征空间差异大，需两层非线性；
- **文本逐槽位广播**：每类唯一的文本方向复制到全部 $M$ 个槽位——每个融合槽位 =
  公共语义质心 + 各自视觉形态偏差；
- 融合权重经 softplus 保证正性，初始 $w_v=0.3, w_t=0.7$（参数值经 softplus 后实际
  约 44% 视觉 / 56% 文本，**初始偏向文本**：少样本类视觉原型噪声大，文本作稳定锚），
  训练中可学习；
- 融合输出（加权和**之后**）做逐槽位 **L2 归一化**（`F.normalize`，非 LayerNorm）——
  把向量拉到单位球面，使后续打分/聚类/注意力共用余弦度量；无效槽位置零；
  **每次前向重算**（权重可学习，不可缓存）。

**冻结设计**：$\mathbf{P}_v, \mathbf{P}_t$ 为冻结锚（buffer），训练期不更新；对齐
由可学习投影完成——"离线锚 + 在线投影"的解耦使原型不随特征漂移，同时投影提供适配
能力。产物缺失或形状不符时模块整体不挂载，模型退化为 RF-DETR 原版（优雅降级）。

### 3.5 防 no-op 设计

- 所有注入为残差形式且近恒等初始化（λ/γ 初始 0.05），开关关闭时数学恒等于原版；
- 融合权重初始偏文本（少样本稳定锚）；
- 无效槽位全程置零，不污染计算。

---

## 4. 创新模块二：原型引导的 Query 选择（位置增强）

### 4.1 动机

两阶段 DETR 中，decoder 只能处理固定数量的 query；若目标在 top-k 选择阶段未被选中，
后续任何判别机制都无法补救。RF-DETR 原始选择仅依赖线性 objectness，小样本类与
微小目标因分类器训练不充分，其 token 分数常被背景压制。**位置增强的本质是改变
"哪些 token 进入 decoder"，即提升位置 query 的来源质量**——不修改任何坐标或特征。

### 4.2 原型相似度打分

对每个 memory token $\mathbf{m}_i$：

$$ \mathbf{z}_i = \mathrm{normalize}\big(\mathrm{proj}_{\mathrm{tok}}(\mathbf{m}_i)\big), \qquad \mathrm{class\_sim}_{ic} = \max_m \; \langle \mathbf{z}_i, \; \mathbf{P}_{\mathrm{mm}}[c, m] \rangle $$

- $\mathrm{proj}_{\mathrm{tok}}$ 把 token 投影到融合空间并归一化（与原型同度量）；
- 槽位维取 max：token 只认"最像的那个形态"，类内多形态不互相拖累；
- 原型分类 logits：$\mathrm{proto\_logits}_{ic} = \mathrm{class\_sim}_{ic} / \tau_p$
  （$\tau_p = 0.2$ 把余弦放大到 $[-5,5]$ 区间）。

### 4.3 原型 margin 与分数校准

原型位置分数定义为**目标类与最强竞争类的相似度差距**：

$$ s_i^{\mathrm{proto}} = \max_{c \in C_{\mathrm{target}}} \mathrm{class\_sim}_{ic} - \max_{c \notin C_{\mathrm{target}}} \mathrm{class\_sim}_{ic} $$

$C_{\mathrm{target}}$ 为目标类集合（本项目舰船易混类 $[0,1,2,3]$）。margin 直接衡量
"这个 token 有多像关注类、有多不像干扰类"，天然排除"像任何东西"的模糊 token。

由于 margin 与线性 objectness 尺度不同，直接相加会让一方主导 top-k，故先校准：

$$ \tilde{s}_i = \frac{s_i^{\mathrm{proto}} - \mu_s}{\sigma_s} \cdot \sigma_l \cdot 0.1 $$

（$\mu_s, \sigma_s$ 为 margin 的均值与标准差，$\sigma_l$ 为线性分数的标准差）——
margin 居中归一后缩放至线性分数的 10% 量级，**给原始 objectness 留出稳定主干**，
防止纯相似度把背景纹理抬进 top-k。

### 4.4 残差合并与 warmup 调度

$$ \hat{s}_i = s_i^{\mathrm{lin}} + \lambda(t) \cdot \tilde{s}_i, \qquad \lambda(t) = \lambda_0 + (\lambda_{\max} - \lambda_0) \cdot \min(t / T_w, 1) $$

$$\mathcal{Q} = \arg\mathrm{top}\text{-}K_i \; \hat{s}_i$$

- **residual 而非替换**：保留预训练线性头（它隐含 objectness 与 proposal 质量），
  原型分数只做边际修正——与直接替换为相似度头相比，降低港口/道路纹理误选风险；
- **warmup 调度**：λ 从 0.05 随 epoch 线性升至上限（joint120：0.3/10ep；joint20：
  1.0/2ep）。训练初期投影未学成，原型分数是噪声，渐进接管；init=0.05 而非 0 是为了
  保留梯度通路（top-k 是离散选择，零注入即死分支）；
- 选中的 $\mathcal{Q}$ 同时决定 **refpoint（位置 query 来源）** 与 **关联类别
  （内容增强输入）**。

---

## 5. 创新模块三：原型增强的内容 Query（内容增强）

### 5.1 动机

top-k 决定 decoder "看哪里"，内容 query 决定"用什么语义方向解释该位置"。原始
RF-DETR 的内容 query 为共享可学习嵌入，不含任何类别先验。**内容增强在选中 query
进入 decoder 前，按"最可能的类别"注入该类形态细节**，使 query 携带类别偏置。

### 5.2 关联类别与原型置信度

$$ c_i^* = \arg\max_c \; \mathrm{class\_sim}_{ic}, \qquad \kappa_i = \frac{\max_c \, \mathrm{softmax}(\mathrm{proto\_logits}_i)_c - 1/C}{1 - 1/C} $$

- $c_i^*$ 来自原型相似度（而非分类头），与位置引导共用一套类别语义；
- $\kappa_i$ 为去均匀项的置信度归一化——原型分类不可信时内容注入被自动削弱。

### 5.3 槽位交叉注意力（提取上下文）

$$ \alpha_{im} = \mathrm{softmax}_m\Big( \langle \mathbf{q}_i, \mathbf{P}_{\mathrm{mm}}[c_i^*, m] \rangle / \sqrt{d} \Big), \qquad \mathbf{ctx}_i = \sum_m \alpha_{im} \cdot \mathbf{P}_{\mathrm{mm}}[c_i^*, m] $$

query 以自身内容为 query、以关联类别 10 个槽位为 key/value：**按自身外观选择最贴合
的子形态**——类内多形态通过注意力自适应聚合，不被平均抹平；全部槽位无效时退回
均匀注意力（防御）。

### 5.4 Gate 与残差注入

$$ g_i = \sigma\Big( \mathrm{MLP}\big([\mathbf{q}_i, \mathbf{ctx}_i]\big) \Big) \cdot \kappa_i, \qquad \mathbf{q}_i' = \mathbf{q}_i + \gamma(t) \cdot g_i \cdot \mathbf{ctx}_i $$

- gate 的 MLP bias 初始 $\mathrm{logit}(0.05) \approx -2.944$（初始 gate≈0.05）——近恒等
  初始化，不破坏预训练 query 分布且保留梯度；
- **组合语义**：交叉注意力回答"注入什么"（哪个形态的细节），gate 回答"注入多少"
  （要不要信这个先验）；gate 还乘原型置信度 $\kappa_i$ 作二级门控；
- $\gamma(t)$ 与 λ 同款 warmup 调度（joint120：0.3；joint20：0.3——内容持续压弱注入，
  待联合训练验证后决定去留）；
- 内容增强仅作用于选中段（$\mathcal{Q}$ 对应 query），尾部原始 query 不变；
- 实现细节：内容 query 初始为可学习嵌入 $\mathbf{q}^{\mathrm{feat}}$ 的前 $K$ 段，
  refpoint 来自选中 token 的 enc 回归头——**位置与内容解耦**，位置增强只换来源、
  内容增强只注先验。

### 5.5 与位置增强的关系

两个模块共用同一套原型与打分：位置增强改变"选谁"，内容增强改变"选中者带着什么
语义进 decoder"。二者共享 $\mathrm{proj}_{\mathrm{tok}}$、融合原型与 $\tau_p$，但
参数与注入路径完全独立（λ vs γ、top-k vs gate），可独立消融。

---

## 6. 创新模块四：原型分类辅助损失

### 6.1 动机

top-k 是离散选择——位置分支的梯度若只依赖"改变选择→影响检测损失"的间接路径，
极易训练成 no-op（分数永远改不动 top-k）。需要一条**直接监督打分分支**的梯度通路。

### 6.2 定义

对 enc 分支 Hungarian 匹配到的前景 token：

$$ \mathcal{L}_{\mathrm{proto}} = \frac{1}{|C_{\mathrm{act}}|} \sum_{c \in C_{\mathrm{act}}} \frac{1}{|\mathcal{M}_c|} \sum_{i \in \mathcal{M}_c} \mathrm{CE}\big( \mathrm{proto\_logits}_i / \tau_p, \; y_i \big) $$

- $\mathcal{M}_c$：GT 标签为 $c$ 的匹配 token 集合；$C_{\mathrm{act}}$：当前 batch 出现的
  类别集；
- **只监督 matched foreground**：未匹配 token 远多于前景，若全部参与会训练成"全部低分"
  的塌缩态；监督语义为"被选进 decoder 的 token，其原型分类必须分对类"；
- **类别均衡**：各类独立求均值再平均，防止 MS 等高频类吞掉 HM 等稀少类的梯度；
- **梯度路径**：$\mathcal{L}_{\mathrm{proto}} \to \mathrm{proto\_logits} \to
  \mathbf{P}_{\mathrm{mm}} \to (\mathrm{proj}_v, \mathrm{proj}_t, w_v, w_t,
  \mathrm{proj}_{\mathrm{tok}})$——**融合模块全部参数获得直接监督**，这正是
  "文本-视觉语义对齐"的监督来源：类别标签把视觉信息与文本信息在融合空间拧向同一
  语义方向。

### 6.3 权重

$\lambda_{\mathrm{aux}}$：联合训练阶段一 0.2、阶段二 0.3、独立微调（E 系列）1.0。

---

## 7. 创新模块五：语义加权监督对比学习（SSCL）

### 7.1 动机

位置增强提高召回的同时，可能把更多易混类候选带进 decoder。SSCL 在 decoder 特征
空间施加类别分离约束——**对语义相似（易混淆）的异类负样本给予更强分离压力**，
把混淆类的特征边界拉开。

### 7.2 特征来源

从 decoder 最后一层 hidden states 中提取 Hungarian 匹配到的 foreground query 特征
$\mathbf{h}_i$（及其 GT 标签 $y_i$）——只用 matched query：类别语义可信、数量少
（每图目标数），对比矩阵规模可控。复用 criterion 的匹配结果，不重复匹配。

### 7.3 损失定义

$$ \mathcal{L}_{\mathrm{SSCL}} = -\frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \log \frac{ \sum_{j \in \mathcal{P}(i)} \exp(\mathrm{sim}_{ij}) }{ \sum_{j \in \mathcal{P}(i)} \exp(\mathrm{sim}_{ij}) + \sum_{j \in \mathcal{N}(i)} \exp(w_{ij} \cdot \mathrm{sim}_{ij}) } $$

其中 $\mathbf{u}_i = \mathrm{normalize}(\mathbf{h}_i)$，$\mathrm{sim}_{ij} = \langle
\mathbf{u}_i, \mathbf{u}_j \rangle / \tau$，负样本语义权重：

$$ w_{ij} = \mathrm{clamp}\big( 1 + \rho \cdot S[y_i, y_j], \; 1, \; \omega_{\max} \big) $$

- $\mathbf{S}$：CLIP 类别语义相似度矩阵（minmax 归一化到 $[0,1]$），由类别提示词
  的文本向量两两余弦得到——**易混类的 $S$ 值高 → 权重 $w>1$ → 被推得更远**；飞机类
  与舰船类的 $S$ 值低 → 权重≈1，按普通负样本处理；
- $\mathcal{P}(i)/\mathcal{N}(i)$：同类/异类样本；$\mathcal{A}$：anchor 集
  （`anchor_classes=[0,1,2,3]` 舰船类 ∩ 存在同类正样本）；
- $\tau=0.1$（温度）、$\rho=0.3$（放大系数）、$\omega_{\max}=2.0$（权重上限防不稳定）；
- 数值稳定：logsumexp 形式；无有效 anchor 或前景数 <2 时返回零损失但保持计算图
  连接（DDP 下各参数仍收到梯度）。

### 7.4 冻结策略

- `conservative`：冻结 backbone/encoder/bbox 头/decoder 前几层，仅解冻 decoder 末尾
  N 层 + norm + class_embed + 附加模块——**SSCL 只通过 decoder 最后一层重塑 query
  特征空间，不扰动主干与定位能力**（E 系列与联合训练阶段二使用）；
- `none`：全量微调（联合训练阶段一必须——否则原型模块开启时 conservative 会连带
  冻结 backbone，破坏联合训练）。

### 7.5 与多模态原型的关系

SSCL 与 ProtoGuidance 使用**不同的特征空间**（decoder matched query 特征 vs encoder
token/离线原型），类别索引与语义统一，但向量不直接比较；跨空间一致性只能比较类别
关系矩阵（`prototype_relation_alignment`）。SSCL 的原型锚定模式（EMA 原型库 +
投影头）为可选扩展，当前实验统一使用 instance-to-instance 模式。

---

## 8. 创新模块六：难负样本抑制

### 8.1 动机

"像目标但不是目标"的背景（港口、码头、道路、阴影纹理）在推理时以高置信检测框的
形式出现（测试集 MS 类 FP 达 207 例）。标准分类损失对所有未匹配 query 施加的背景
监督是分散的、无几何信息的；本文提出**显式的难例挖掘 + logit 封顶**。

### 8.2 难例选择（逐图）

1. **排除 Hungarian 匹配的 query**（$i \notin \mathrm{matched}$）——真阳绝不碰；
2. **IoU 带过滤**：$\max_j \mathrm{IoU}(\hat{\mathbf{b}}_i, \mathbf{b}_j^{\mathrm{gt}})
   \in [\mathrm{iou}_{\mathrm{low}}, \mathrm{iou}_{\mathrm{high}}]$——下界纳入纯背景
   （对准"纯背景虚警"靶点），上界保护真实目标的重复检测（DETR 1-to-1 匹配下，
   同一 GT 的第二个检测天然未匹配，绝不能当负样本）；
3. **分数过滤**：$\max_{c \in C_{\mathrm{target}}} \hat{p}_i[c] \ge \tau_{\mathrm{hn}}$
   ——只挑高置信候选；
4. **限量**：按分数 stable 降序取 top-$k_{\mathrm{hn}}$——背景 query 洪流不淹没梯度。

### 8.3 抑制损失

$$ \mathcal{L}_{\mathrm{HN}} = \mathrm{softplus}\Big( \big( \max_{c \in C_{\mathrm{target}}} \hat{p}_i[c] - m_{\mathrm{hn}} \big) / T_{\mathrm{hn}} \Big) \cdot T_{\mathrm{hn}} $$

- 只在 logit 超过 margin $m_{\mathrm{hn}}$ 时惩罚（"封顶"语义），softplus 平滑恒正，
  梯度在 margin 附近最强；
- **为什么罚未匹配 query 有用**：训练时的"unmatched"只是 Hungarian 分配结果，推理
  时所有 query 都会发射——被压分的难例正是推理时的 FP 群体；压分更新的是共享
  class_embed 与特征，效果泛化到新图像上同类背景模式（经典 hard negative mining 的
  DETR 形式）；
- **与标准背景监督的三点区别**：难例挖掘（梯度聚焦 top-k 最危险者）、margin 语义
  （超过才罚，而非从 0 压）、GT 几何信息（IoU 带确认"确定没撞上任何 GT"）。

### 8.4 参数设定与监控

E4 实验（λ=0.3、margin=-1.5、从 epoch 0、目标类含 MS）把 MS 召回从 0.7568 打到
0.5340（-22pt）——教训是抑制机制误伤了"低 IoU 但确是真目标"的样本。v2 参数四重
缓冲：**λ 减半（0.15）、margin 放宽（-2.0）、目标类排除 MS（高频易误伤类不参与）、
晚启动（epoch 10 起，独立门控）**。监控指标（每图难例数、IoU 带填充率、分数/IoU
均值）用于判断选择是否工作、是否误伤真阳。

---

## 9. 训练策略：两阶段联合训练

### 9.1 设计动机

E 系列三阶段流水线（120ep 纯训练 → 3ep 冷启动 → 18ep 微调）表明位置引导有效而
内容引导收益小。一个结构性解释：**三阶段里新增模块始终在对抗已收敛的检测器**——
投影与 gate 的梯度穿过收敛的分类头与特征，学习信号微弱。本文提出联合训练：
让新增模块在特征可塑期与主干共适应，再微调收敛。

### 9.2 阶段一：120 epoch 联合训练（joint120）

- **弱注入共适应**：λ→0.3、γ→0.3、辅助损失权重 0.2，λ/γ 前 10 epoch 从 0.05 线性
  warmup——不扰动主训练，同时投影/gate 有梯度可学；
- **SSCL 弱塑形**：λ_sscl=0.01、从 epoch 20 起（等匹配质量成形）、
  `sscl_freeze_strategy: none`（**联合训练的关键**：conservative 会在原型模块开启时
  冻结 backbone，破坏共适应）；
- **难负样本抑制关闭**（早期匹配是噪声，任何"负样本"标签都不可信）；
- 调度与基线对齐：lr=1e-4 / lr_enc=1.5e-4、lr_drop=70（最后 50ep 低 LR 长退火尾）、
  EMA、eval 每 5ep；数据口径 `-redo`（与对照实验一致）。

### 9.3 阶段二：20 epoch 微调（joint20）

- 从阶段一 EMA best checkpoint 启动，λ→1.0（位置升满）、γ 维持 0.3（内容继续压着
  验证）、辅助权重 0.3；
- SSCL 全量（λ=0.02）自 epoch 0 延续，`sscl_freeze_strategy: conservative`（与 E 系列
  可比）；
- 难负样本 v2：epoch 10 起（`sscl_hard_neg_start_epoch` 独立门控，与 SSCL 解耦）、
  λ=0.15、margin=-2.0、目标类排除 MS；
- 调度修复：lr=1e-5、lr_drop=14（**最后 1/3 退火**——E 系列 lr_drop=18==epochs 的
  配置导致整个微调期 LR 从不衰减，是已修复的缺陷）、train warmup 1ep、eval 每 ep。

### 9.4 防 no-op 监控体系

训练全程监控（写入 tensorboard）：

| 指标 | 判定 |
| --- | --- |
| `topk_overlap`（合并与纯线性 top-k 重叠率） | <0.95 且下行 → 位置分支生效 |
| `lambda_effective`（实际注入强度） | 随 warmup 上升 |
| `proto_logits_pmax / entropy`、`proto_margin_mean` | 选中 token 原型分类变锐、类间分离 |
| `prototype_offdiag_cos` / `effective_rank` | 融合原型类间分离、无塌缩 |
| `gate_mean` | 不应长期≈0（内容分支 no-op 信号） |
| `hn_count` / `hn_fill_rate` / `hn_score_mean` | 难例选择在工作、未误伤真阳 |
| `selected_class_hist` | 选中 query 类别分布与 GT 近似 |

---

## 10. 方法小结

本文在 RF-DETR 基础上提出三创新组件（多模态原型库、位置引导、内容增强）与三判别
机制（原型辅助损失、SSCL、难负样本抑制），并通过两阶段联合训练使其共适应：

- **原型库**：离线"视觉细节 + 文本语义"锚，在线可学习投影融合，残差注入、近恒等
  初始化、全程可降级；
- **位置引导**：残差合并 + 校准 + warmup，改变 top-k 来源，由辅助损失直接监督打分
  分支（防 no-op 的关键）；
- **内容引导**：槽位交叉注意力提取形态细节，gate×置信度控制注入量；
- **SSCL**：语义权重把对比压力按易混程度重分配，anchor 类聚焦舰船易混类；
- **难负样本抑制**：显式难例挖掘 + logit 封顶，经目标类选择与相位化调度避免误伤
  真阳；
- **联合训练**：特征可塑期共适应 + 低 LR 退火微调，修复了恒 LR 不衰减等调度缺陷。
