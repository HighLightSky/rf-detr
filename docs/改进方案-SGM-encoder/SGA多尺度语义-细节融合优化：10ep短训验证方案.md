# SGA 多尺度语义-细节融合优化：10ep 短训验证方案

> 撰写日期：2026-08-07
> 目标：用统一的 10 epoch 短训逐步验证 SGA 的结构性改进方向。重点不是继续寻找“更强的注意力图”，而是验证 DINOv2 全局语义与 CNN 高频细节能否在 P3/P4 多尺度上形成稳定、可检测的互补特征。
>
> 前置文档：
> - `SGM注意力门控反向优化：实验复盘与attn_reg弱监督方案.md`
> - `RF-DETR引入SGM混合编码器改进方案.md`
> - `src/scripts/test_sga/common.py`

---

## 1. 问题重述

当前 SGA 短训实验使用 `projector_scale=["P4"]`。因此 SPM 虽然生成了 `c2/c3/c4`（stride 8/16/32），实际只让 stride-16 的 `c3` 与 P4 融合；真正可能改善极小目标采样密度的 stride-8 `c2` 没有进入 decoder。

当前 SGM 还存在两个结构限制：

1. 只读取 `raw_feats[-1]`，生成一张单通道 sigmoid 图；
2. 用 `det * M` 或其保底变体调制 CNN 特征。

这让“是否保留细节”被压缩成一个标量门控问题。已有实验表明，这个标量会稳定学成目标低、背景高，因而 `attn_reg` 是合理诊断手段；但即使将极性纠正，也不能解决 P3 缺失、通道选择不足、语义与细节对齐不足的问题。

本方案的核心判断是：**优先验证真实多尺度细节，再验证语义如何调制细节；不要把单通道注意力图当成 SGA 的核心能力。**

---

## 2. 假设与可证伪结论

| 编号 | 假设 | 支持该假设的结果 | 否定该假设的结果 |
|---|---|---|---|
| H1 | 真实 stride-8 CNN 特征是小目标增益的主要来源 | `spm_p3p4` 明显优于仅 DINO 多尺度 | `spm_p3p4` 不优于 `vit_p3p4` |
| H2 | DINO 语义应作为残差调制信号，而不是硬乘法开关 | `semantic_film` 优于 `spm_p3p4`，且 FDR 不恶化 | 无增益或细节分支梯度塌缩 |
| H3 | P3 细节位置查询 P4 全局语义，可改善复杂背景中的选择性 | `local_xattn` 优于 `semantic_film` | Recall/FP 均无改善，或显存与吞吐代价不合理 |
| H4 | 对 P3 特征施加中心度监督比框内 attention 监督更匹配检测目标 | `centerness_aux` 改善 ship/all Recall 或 AP_S | 仅训练损失下降、测试指标不变或 FDR 上升 |
| H5 | `attn_reg` 只在上述结构已经有效时才可能有额外收益 | 最佳结构加 `attn_reg` 后仍有稳定增量 | 不超过无 `attn_reg` 的同结构版本 |

短训的结论只用于筛方向，不声称统计显著。进入全量训练的候选必须在至少 2～3 个 seed 上复核。

---

## 3. 目标结构

### 3.1 P3/P4 的职责

```text
输入图像（640×640）
├─ SPM
│  ├─ C2：stride 8   → 保留小目标边界、局部纹理、形状细节
│  └─ C3：stride 16  → 提供与 P4 对齐的中尺度纹理
└─ DINOv2 + projector
   ├─ V3：stride 8   → 上采样语义参考，不包含新的原始高频信息
   └─ V4：stride 16  → 全局类别语义与上下文

F3 = Fuse(V3, C2)  → decoder level 0
F4 = Fuse(V4, C3)  → decoder level 1
```

P3 的关键不是“将 ViT 特征插值到更大尺寸”，而是让 `C2` 的真实 stride-8 信息能够被 decoder 的多尺度 cross-attention 访问。`V3` 只承担语义对齐和上下文补充。

### 3.2 推荐的基础融合：语义条件残差调制

对每个尺度 `s ∈ {3, 4}`，定义：

```text
D_s = GN(Conv1x1(C_s))
S_s = GN(Conv1x1(V_s))
G_s = 0.5 * tanh(Conv1x1(GELU(Conv3x3(S_s))))
U_s = D_s * (1 + G_s)
F_s = V_s + α_s * Conv3x3(GELU(U_s))
```

- `G_s` 是与通道数相同的空间-通道调制，不是单通道标量图；其乘法范围为 `[0.5, 1.5]`，不会关闭 CNN 分支。
- `α_s` 是每个尺度独立的可学习标量，初始化为 `1e-3`。预训练 DINO 主路径在训练开始时近似不变，新分支逐步介入。
- `GN` 只用于新增投影与融合层，避免小 batch 下 CNN BatchNorm 统计与 DINO 特征分布失配。
- 第一版不使用 `beta` 加性偏移，避免语义分支直接生成与真实细节无关的伪纹理；后续仅在该版本成立后再考虑 FiLM 的完整 `gamma + beta`。

这不是强制“目标处注意力高”，而是让语义决定哪些细节通道与空间位置可被有限幅度地增强或减弱，同时由残差路径保留预训练检测能力。

### 3.3 可选增强：P3 对 P4 的局部跨尺度交叉注意力

在基础融合已验证有效后，只对 P3 增加一层轻量局部 cross-attention：

```text
Q = Conv1x1(F3)
K, V = Conv1x1(V4)
F3' = F3 + α_x * DeformableCrossAttention(Q, K, V, n_points=4)
```

- Query 来自局部细节 P3，key/value 来自未被 CNN 混合破坏的 DINO P4 语义；
- 使用 4 个采样点的 deformable cross-attention，而不是 P3×P4 的全量 attention；
- `α_x` 初始化为 `1e-3`；
- 坐标使用 P3 的归一化参考点，并正确传入 P4 mask/valid ratio，不能把 padding 区域当作语义来源。

该模块回答的是“某个细节位置应从哪些全局语义位置取上下文”，比从 P4 直接生成一张静态门控图更具表达力。

---

## 4. 统一实验协议

所有实验复用 `src/scripts/test_sga/common.py` 的当前训练 recipe，除模型结构字段外不改超参：

| 项目 | 固定值 |
|---|---|
| 数据集 / 类别 | SHWX / 25 类 |
| 初始化 | 同一份 COCO 预训练 `RFDETRMedium` 权重；新增模块随机初始化 |
| 输入分辨率 | 640×640 |
| 训练轮数 | 10 epoch |
| seed | 0 |
| batch / 梯度累积 | 16 / 4（有效 batch 64） |
| 基础 LR / encoder LR | `1e-4` / `1.5e-4` |
| warmup / LR drop | 2 epoch / epoch 15 |
| 数据增强 | `AUG_AERIAL`，`mosaic_p=0.8` |
| EMA / 测试口径 | `ema_decay=0.993`；复用 `test.py` 固定阈值比赛指标 |

### 4.1 每个变体必须记录

1. `test_result.txt` 中 all / ship Recall、FP、FN、FDR、F1；
2. 验证集 mAP 曲线，仅用于观察收敛，不作为单独选择依据；
3. `grad_norm/sga/*`，确认新分支有稳定梯度；
4. 每尺度的 `α_s`、`α_x` 数值和梯度；
5. P3/P4 特征均值、标准差、L2 范数，排查某一分支恒零或幅值爆炸；
6. 仍启用 SGM 的变体保留框内/背景注意力统计，但不以该统计作为最终成功条件。

### 4.2 建议的方向判据

短训下优先使用相对比较：

- `ship Recall` 提升超过约 1pp（当前 615 个 ship GT，对应约 6 个目标）才视为有方向信号；
- `all Recall` 不应低于被比较基线；
- FDR 的上升需要与 Recall 增益一起判断，不能以召回换取明显大量 FP；
- 大目标类别显著退化时，不能仅因小目标略涨而进入全量训练；
- 若增益只出现在 val mAP、固定测试阈值下 Recall 不涨，按无效处理。

---

## 5. 10ep 实验矩阵

### E0：多尺度与真实细节的必要性

本组不使用可学习 SGM 门控，也不开 CFE。目的只是分清“多级 decoder 的影响”与“真实 CNN 细节的影响”。

| 名称 | `projector_scale` | SGA | 融合 | 回答的问题 |
|---|---|---|---|---|
| `baseline_p4` | `[P4]` | 关 | — | 当前基线 |
| `vit_p3p4` | `[P3, P4]` | 关 | DINO projector 多级 | 多级 decoder 本身是否带来收益 |
| `spm_p3p4` | `[P3, P4]` | 开 | `ones` + 残差 concat | 真实 C2/P3 细节是否有额外收益 |

实现要求：

- `vit_p3p4` 与 `spm_p3p4` 都使用两级 decoder，并确认 checkpoint 的 projector 和 MSDeformAttn level 权重 warm-start 正确；
- `spm_p3p4` 的 P3 对应 `C2`，P4 对应 `C3`；
- `sga_gate_mode="ones"`，使 SGM 即使被实例化也不参与特征计算，避免门控成为变量；
- `use_cfe=False`，避免 FPN/PAN/RGM 同时引入多个因素。

判读：

- `vit_p3p4 > baseline_p4`：decoder 多尺度本身有价值；
- `spm_p3p4 > vit_p3p4`：真实 CNN 细节有独立价值，H1 成立；
- `spm_p3p4 <= vit_p3p4`：先不要设计更复杂的语义门控，应检查 C2 质量、P3 mask、权重加载、训练分辨率和目标尺寸分布。

### E1：语义条件残差调制

以 `spm_p3p4` 为唯一对照，替换 `concat + conv` 融合为 §3.2 的 `semantic_film` 融合：

| 名称 | P3/P4 融合 | 门控 | 关键初值 |
|---|---|---|---|
| `spm_p3p4` | 现有 concat 残差融合 | `ones` | 固定 `gamma=0.1` |
| `semantic_film_p3p4` | GN + 通道空间调制 + 可学习残差 | 无单通道 SGM | `α_3=α_4=1e-3` |

这里的“FiLM”仅指有界 `gamma` 调制，不启用 `beta`。若 `semantic_film_p3p4` 优于对照，说明 DINO 全局语义能有效选择 CNN 细节；若不优于对照，不应直接跳到更重的 attention。

### E2：局部跨尺度语义检索

仅在 `semantic_film_p3p4` 不低于 `spm_p3p4` 时运行：

| 名称 | 基础结构 | 新增模块 | 目标 |
|---|---|---|---|
| `semantic_film_p3p4` | E1 最佳版本 | 无 | 语义调制对照 |
| `local_xattn_p3p4` | 同上 | P3 query → 原始 DINO P4 key/value 的 4 点 deformable cross-attention | 让细节按需读取全局语义 |

若该变体只提高 FDR 而无 Recall 改善，说明额外语义上下文放大了背景模式，应停止该方向；不要再增加全局 attention 层数。

### E3：P3 中心度辅助监督

中心度监督是对“P3 是否携带目标性细节”的训练期约束，不是对 SGM 门控图的约束。以 E1/E2 中测试指标最好的结构为基线：

| 名称 | 新增训练期模块 | 损失 | 初始权重 |
|---|---|---|---|
| `best_no_aux` | 无 | — | — |
| `best_centerness_aux` | P3 上 `3×3 + GELU + 1×1` 单通道头 | 高斯中心热图 focal loss | `λ_center=0.1` |

目标生成规则：每个 GT 框在 P3 坐标系以中心点生成高斯峰；半径按目标框在 P3 上的较小边确定，并限制最小半径为 1 cell。重叠目标取逐像素最大值。损失按有效 GT 数归一化，空图只计算负样本 focal 项。

该头仅训练期存在，推理和导出时不参与任何预测结果。若 `λ_center=0.1` 导致训练早期主检测损失被压制，依次回退到 `0.05`，而不是修改主检测 loss。

### E4：将 `attn_reg` 放在最后比较

只有 E1～E3 证明结构有效后，才在最佳结构上比较 `attn_reg`：

| 名称 | 结构 | 正则 |
|---|---|---|
| `best_no_attn_reg` | E1/E2/E3 选择出的最佳结构 | 无 |
| `best_attn_reg` | 完全相同 | 框内下界 hinge，先试 `floor=0.5, λ=0.1` |

这里的目的不是证明 attention 图“漂亮”，而是验证在正确的多尺度融合结构上，GT 框约束是否仍能提供额外检测收益。若没有超过 `best_no_attn_reg`，就将 `attn_reg` 保留为诊断开关，不纳入主架构。

---

## 6. 实现边界与配置建议

### 6.1 推荐的新配置字段

以下字段建议默认关闭，确保现有模型和 checkpoint 行为不变：

```python
sga_fusion_mode: Literal["concat", "semantic_film"] = "concat"
sga_residual_alpha_init: float = 1e-3
sga_local_xattn: bool = False
sga_local_xattn_points: int = 4
sga_centerness_lambda: float = 0.0
```

`sga_gate_mode` 在 E0/E1/E2 中优先使用 `"ones"` 或不实例化 SGM；不要同时启用旧式 product gate 与新的语义调制，否则无法归因。

### 6.2 模块归属

| 模块 | 建议位置 | 说明 |
|---|---|---|
| P3/P4 SPM-DINO 融合 | `src/rfdetr/models/backbone/sga.py` | 替换或扩展现有 `SGAEncoder` 的每尺度融合层 |
| 局部 cross-attention | `sga.py` 或独立轻量模块 | 不放入 decoder，避免改变检测 query 的既有预训练路径 |
| 中心度头与损失 | `src/rfdetr/training/` + criterion 回调 | 训练期读取 P3 融合特征，推理时不执行 |
| 配置透传 | `config.py`、`_namespace.py`、`_types.py` | 默认值必须复现当前行为 |
| 短训注册表 | `src/scripts/test_sga/common.py` | 每个变体显式记录 `projector_scale`、融合模式和辅助损失 |

### 6.3 实现前的冒烟检查

每个新结构在训练前必须完成：

1. `projector_scale=["P3", "P4"]` 下的构建和一次前向；
2. 从 COCO 单级 P4 checkpoint 加载，确认 P4 共享权重被加载、P3 新增部分保持随机初始化；
3. decoder 的 `num_feature_levels=2`、feature shape、mask shape、valid ratio 均正确；
4. 反向后 `spm`、P3 融合模块、`α_s`、局部 cross-attention 参数都有非零且有限梯度；
5. 导出前向与训练前向在无 padding 输入下输出尺度一致；
6. 中心度辅助头关闭时，训练总损失与无该模块版本一致。

---

## 7. 停止条件与后续决策

| 观察结果 | 决策 |
|---|---|
| `spm_p3p4` 不超过 `vit_p3p4` | 暂停语义融合开发，优先检查 P3 特征、目标像素尺寸、输入分辨率和数据增广 |
| `spm_p3p4` 有稳定增益，但 E1/E2 无增益 | 保留 P3/P4 SPM-only，说明细节有用但复杂语义调制无价值 |
| E1 有增益、E2 无增益 | 采用有界残差调制，不引入 cross-attention 的延迟与复杂度 |
| E3 提升 Recall 且 FDR 可控 | 在全量训练中保留中心度辅助头，并搜索较小的 `λ_center` |
| 任意变体只提升 val mAP、测试固定阈值 Recall 不涨 | 按无效处理，不进入全量训练 |
| `attn_reg` 未超过相同结构的无正则版本 | 不将其纳入主方案，保留为可选诊断工具 |

只有满足以下条件的方案进入 100 epoch、多 seed 验证：

1. 相对于紧邻对照，ship Recall 有超过噪声范围的提升；
2. all Recall 不退化，FDR 增加可解释且可接受；
3. 新模块梯度、残差系数和特征统计正常；
4. 改进能由唯一新增变量解释。

---

## 8. 暂不做的事项

- 不在 E0～E3 同时启用 CFE、RGM、RepNCSPELAN 和多级语义监督；这些模块会淹没 P3 细节与语义融合的归因。
- 不先上框内/框外 BCE、背景上限或高权重 `attn_reg`；它们容易把“发现真实细节是否有用”的问题变成损失权重问题。
- 不把 P3 上采样的 DINO 特征当作新增高频信息；只有 SPM `C2` 才提供这个能力。
- 不因单个 10ep、单 seed 的最好数值直接下结论；短训的职责仅是淘汰明显错误的方向。

完成 E0 后，应先根据 `vit_p3p4` 与 `spm_p3p4` 的差异决定是否继续 E1；这是本方案最重要的分叉点。
