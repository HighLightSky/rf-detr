# RF-DETR 引入 SGM 混合编码器改进方案

## 0. 方案摘要

本方案在 RF-DETR 现有 DINOv2 特征提取链路上，参考 SKYDET（*SKYDET: An End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Objects in Remote Sensing Images*，详见 `docs/参考论文/SKYDET.md`）的 SGA（Semantic Guiding Adapter）+ CFE（Cross Fused Encoder）设计，新增一条 **SGM 引导分支**：

- **SPM（Spatial Prior Module）**：对原始图像做轻量卷积下采样，提供 DINOv2（patch16）无法提供的**原生 stride-8 高频纹理**；
- **SGM（Semantic Guiding Module）**：取 DINOv2 最深语义特征生成空间注意力图，对 SPM 空间特征做语义门控，抑制遥感背景噪声；
- **RGM（Reciprocal Guidance Module）**：语义特征与空间纹理特征做**双向互导融合**（空间注意力取自语义、通道注意力取自纹理），得到信息更丰富、边界更清晰的编码器输出。

目标收益：缓解 DIOR 遥感场景中小目标（车辆、飞机、船舶等）在 stride-16 单尺度下"亚像素不可检"的结构性瓶颈，提升 AP_S 与 AP75；同时为后续 SSCL 细粒度分类提供更强的语义特征底座。

本方案支持两种接入深度：
- **Phase 1（保守）**：单级 fused P4，decoder 结构完全不变，权重可加载，快速验证 SGM 分支有效性；
- **Phase 2（主目标）**：补 stride-8 多级特征，decoder 升级为多级（n_levels≥2），真正获得小目标空间分辨率收益。

训练策略：从 **COCO 预训练发布权重 `rf-detr-medium.pth`** 初始化（不使用 SHWX/DIOR 微调后的 checkpoint），新分支随机初始化；**全量微调**为主方案，另设"冻结 DINOv2"对照变体。

---

## 1. 现在的 RF-DETR encoder 是什么样的，是否具备加入 SGM 分支的条件

### 1.1 现有架构

RF-DETR 是 **LW-DETR 风格**：Transformer 中**没有 encoder**（[transformer.py:178](src/rfdetr/models/transformer.py#L178) 的 `self.encoder = None`）。所谓"编码器"由 backbone + projector 承担：

```
输入图像 I (640×640)
  └─ DINOv2（dinov2_windowed_small, patch16, 2×2 窗口）
       ├─ stage3 → 384ch, stride-16
       ├─ stage6 → 384ch, stride-16
       ├─ stage9 → 384ch, stride-16
       └─ stage12 → 384ch, stride-16
  └─ MultiScaleProjector（[projector.py](src/rfdetr/models/backbone/projector.py)）
       └─ P4（256ch, stride-16）   ← medium 配置只输出单尺度
  └─ decoder（MSDeformAttn, n_levels=1, two_stage=True, dec_layers=4）
```

关键配置（`RFDETRMediumConfig`，[config.py:833](src/rfdetr/config.py#L833)）：
- backbone：`dinov2_windowed_small`，patch16，num_windows=2；
- `out_feature_indexes=[3,6,9,12]`（**全部 stride-16**，因为 patch16 下 ViT 不跨层下采样）；
- `projector_scale=["P4"]` → `num_feature_levels=1`（[lwdetr.py:820](src/rfdetr/models/lwdetr.py#L820)），decoder 只吃一级；
- 训练分辨率 640（train.py 覆盖默认 576）。

### 1.2 是否具备加入 SGM 分支的条件：**具备，且基础设施比想象完善**

| 条件 | 现状 | 结论 |
|---|---|---|
| **decoder 多级能力** | [transformer.py:306-335](src/rfdetr/models/transformer.py#L306-L335) 的 `spatial_shapes`/`level_start_index`/`valid_ratios`、[MSDeformAttn](src/rfdetr/models/ops/modules/ms_deform_attn.py) 的 `n_levels` 全部现成，默认支持多级 | ✅ 加多级是"改配置+重训"，不是重写 |
| **通用接口** | `transformer.forward` 接收 `srcs/masks/pos_embeds` 列表，天然容纳任意尺度；`Joiner` 按特征数量生成位置编码 | ✅ 加一路特征零接口改动 |
| **checkpoint 加载** | [weights.py](src/rfdetr/models/weights.py) 用 `load_state_dict(strict=False)`，新模块缺失键自动跳过（仅 warning） | ✅ 新增 SPM/SGM/RGM 不会阻塞加载 |
| **参数分组** | [param_groups.py:61-81](src/rfdetr/training/param_groups.py#L61-L81) 已按"encoder 分层 / decoder 低LR / 其余满LR"分组，新分支自动进满 LR 组 | ✅ 多 LR 训练零配置 |
| **导出** | SPM/SGM/RGM 均为普通卷积/注意力，无重参数化，`export()` 链直接兼容 | ✅ 无需 fuse 步骤 |

**唯一的硬约束**：DINOv2 patch16 原生只有 stride-16 特征，**stride-8 必须由新增的 CNN 分支（SPM）提供**。这正是本方案引入 SGM 分支的核心价值，也与 SKYDET 中 ViT+SPM 的验证路线一致。

---

## 2. SGM 分支的结构

"SGM 分支"由三部分组成（对应 SKYDET 的 SGA + 融合机制）：**SPM 空间先验分支 + SGM 语义引导模块 + 融合层**。

### 2.1 SPM：空间先验分支（轻量 CNN）

对原始图像 `I ∈ R^{3×H×W}` 做逐级下采样（照 SKYDET 的 `SpatialPriorModulev2`，全部普通 `Conv+BN+GELU`，无 CSP，极轻）：

```
stem:   Conv3×3 s2 (3→16) + BN + GELU + MaxPool3×3 s2      → stride4
conv2:  Conv3×3 s2 (16→32) + BN                             → c2, stride8,  32ch
conv3:  GELU + Conv3×3 s2 (32→64) + BN                      → c3, stride16, 64ch
conv4:  GELU + Conv3×3 s2 (64→64) + BN                      → c4, stride32, 64ch
```

返回 `c2/c3/c4`（stride 8/16/32）。参数量 ~0.5M，FLOPs 个位数 G，相对 medium 总量可忽略。

**价值**：提供 DINOv2 拿不到的原生 stride-8 高频纹理——在 stride-8 上，一个 16×16px 的目标从"1×1 像素"变成"2×2 像素"响应。

### 2.2 SGM：语义引导模块

取 **DINOv2 最深特征**（`raw_feats[-1]`，stage12，384ch，stride-16）作为**唯一引导源**，蒸馏出多尺度空间注意力图：

```
M_att^i = Upsample( σ( G_SGM(F_sem^L) ) )     # i ∈ {c2, c3, c4}
```

- `G_SGM`：轻量卷积（如 1×1 + 3×3 + 1×1），输出 1 通道；
- `σ`：Sigmoid；
- 上采样到与第 i 个空间特征相同分辨率。

然后对 SPM 空间特征做**逐元素门控**（预融合，抑制背景、突出目标）：

```
F_det'^i = F_det^i ⊙ M_att^i
```

SKYDET 消融显示该语义引导相比朴素 FPN 带来 AP50 +2.7%、AP75 +5.0% 的提升（Group2→Group3，[SKYDET.md:231](docs/参考论文/SKYDET.md#L231)）。

### 2.3 融合层

- **投影**：SPM 各尺度经 `1×1 conv + BN` 投影到 `hidden_dim=256`，与 DINOv2 侧对齐；
- **RGM 互导融合**：见第 4 节。

> 可选（Phase 3）：尺度内交互可引入 RepNCSPELAN（YOLOv9 的 CSP+RepConv 重参数块，SKYDET 每个融合点一个、共 4 个）。第一版建议跳过以控制工作量。

---

## 3. 在 RF-DETR 哪里加入 SGM 分支，起什么效果

### 3.1 插入位置

在 **`Backbone.forward`** 内部（[backbone.py:151-174](src/rfdetr/models/backbone/backbone.py#L151-L174)），位于 `DINOv2 → projector` 之后、NestedTensor 包装之前：

```
raw_feats = self.encoder(tensor_list.tensors)     # DINOv2 多深度特征
feats = self.projector(raw_feats)                 # [P4]（medium 单尺度）
# ---- 新增 ----
c2, c3, c4 = self.spm(tensor_list.tensors)        # 原图 → 空间先验
c3 = self.sgm_gate(c3, raw_feats[-1])             # 最深语义 → 门控
fused_p4 = self.rgm(feats[0], proj(c3))           # 同尺度 RGM 融合
feats[0] = fused_p4
# ---- 原有逻辑不动 ----
mask 插值 → NestedTensor → Joiner 位置编码 → decoder
```

这样 **decoder / mask / 位置编码全部零改动**。`forward_export` 同步加同样逻辑。

### 3.2 起什么效果

| 效果 | 机制 | 依据 |
|---|---|---|
| **小目标 AP_S** | stride-8 原生高频 → 16px 目标 2×2 像素 | 本方案主目标（Phase 2 完整实现） |
| **定位精度 AP75/AP50:95** | RGM 语义/纹理互导精修边界 | SKYDET RGM 消融 +1.4% AP50 / +3.5% AP75 |
| **背景噪声抑制 AP50** | SGM 语义注意力门控滤除背景 | SKYDET SGA 消融 +2.7% AP50 |
| **特征多样性** | 双路特征互补，通道更丰富 | — |

---

## 4. SGM 分支与 DINOv2 分支如何融合？如何与 decoder 衔接？

### 4.1 不是"只做尺度间融合"——共有三层融合

**① 分支内语义引导（gating，非尺度间）**
SGM 用 DINOv2 最深语义特征生成注意力图，乘到 SPM 空间特征上（`F_det' = F_det ⊙ M_att`）。发生在 SPM 分支内部，解决"遥感背景噪声污染纹理特征"的问题。

**② 同尺度 RGM 互导融合（同分辨率）**
在 stride-16 上：`x0 = DINOv2-P4`（语义，对应 Xh），`x1 = SPM-c3`（纹理，对应 Xl）。照 SKYDET RGM 公式（[SKYDET.md:115-146](docs/参考论文/SKYDET.md#L115-L146)）：

```
M_s = σ( f^7×7( [AvgPool_c(x0); MaxPool_c(x0)] ) )   # 空间注意力取自语义
M_c = σ( W2·δ(W1·GAP(x1)) )                          # 通道注意力取自纹理（reduction=16）
x̃0 = x0 + x0 ⊙ M_c
x̃1 = x1 + x1 ⊙ M_s
Y  = f^1×1( [x̃0; x̃1] )                               # 1×1 融合回 256ch
```

方向与论文验证一致：**语义→空间（定位）、纹理→通道（判别）**，不可调换。

**③ 尺度间融合（Phase 2，可选路径）**
若走多级路线，按 top-down + bottom-up 相邻尺度两两 RGM：

```
RGM(upsample(P4),   SPM-c2) → fused P3   (stride8)
RGM(P4,              SPM-c3) → fused P4   (stride16, 即②)
RGM(downsample(P4),  SPM-c4) → fused P5   (stride32)
```

### 4.2 与 decoder 的衔接（两种深度，对应两种成本）

| | Phase 1（先做，验证） | Phase 2（主目标，小目标收益） |
|---|---|---|
| 输出 | 单级 **fused P4**（stride16） | 多级 **fused P3 + fused P4**（或再加 P5） |
| decoder 结构 | 不变（n_levels=1） | 升级 n_levels=2（或 3） |
| decoder 权重 | **完整加载**，低 LR 微调 | `sampling_offsets` 需 warm-start（复制 level-0 切片）+ 重训 |
| 小目标收益 | 受限（stride-16 天花板） | ✅ 真 AP_S 收益 |
| 代价 | ~+2-4 GFLOPs，几乎零风险 | 多级索引零改动（现成），decoder 重训 + 延迟 +20-30% |

**注意**：即使 Phase 2 升级多级，RF-DETR 的 decoder 基础设施本就支持多级（第 1.2 节），`MSDeformAttn` 权重只依赖 `n_levels` 不依赖 level 尺寸，warm-start 是标准做法，不需要从零重训。

---

## 5. 整体训练策略

### 5.1 权重初始化（关键约束：不用 SHWX/DIOR 微调后的权重）

| 部分 | 初始化来源 |
|---|---|
| DINOv2（backbone） | **COCO 预训练发布权重 `rf-detr-medium.pth`** |
| MultiScaleProjector | 同上（COCO 预训练） |
| decoder + 检测头 + query/refpoint embedding | 同上（COCO 预训练） |
| **SPM / SGM / RGM / 投影层（新增）** | **随机初始化** |

> 明确**不使用** `output/0805-SHWX-*`、`output/0726-DIOR-*` 等 SHWX/DIOR 微调后的 checkpoint 作为起点。目的是保持实验纯净：SGM 分支带来的任何增益都归因于新架构，而不是此前多次域内微调的累积。加载时 `strict=False` 会自动跳过新模块缺失键（[weights.py](src/rfdetr/models/weights.py)），DINOv2/projector/decoder 权重照常生效。

### 5.2 冻结策略

**主方案：全量微调**（方案训练策略的一部分）。

- DINOv2 **不冻结**，以 `lr_encoder × layer_decay` 微调；
- SPM/SGM/RGM/投影：随机初始化，满 LR 训练；
- decoder：以 `lr × lr_component_decay`（低 LR）微调，**必须跟着动**——否则冻结的 decoder 会"抵抗" fused 特征分布，把新分支逼成近恒等（no-op）；
- 检测头：满 LR 微调。

**可选对照变体：冻结 DINOv2**（SKYDET 式）。SKYDET 消融证明冻结 VFM 在小数据集上更稳且更高（0.726±0.001 vs 0.719±0.005，5 个随机种子，[SKYDET.md:249](docs/参考论文/SKYDET.md#L249)）。本方案将"全量微调"与"冻结 DINOv2"作为**一对消融**，归因时保证两组的 DINOv2 策略一致。

### 5.3 学习率分组（repo 已自动支持）

`get_param_dict`（[param_groups.py:61-81](src/rfdetr/training/param_groups.py#L61-L81)）已按名称自动分组，新模块落在 `other_params`（满 LR）：

| 参数 | LR | 备注 |
|---|---|---|
| SPM/SGM/RGM/投影/projector/检测头/queries | `args.lr` | 新分支需满 LR 才能充分训练 |
| decoder | `lr × lr_component_decay` | 低 LR，协同适应 fused 特征 |
| DINOv2 | `lr_encoder × layer_decay` | 全量微调主方案；冻结变体则置 0 |

### 5.4 训练配置与流程

- 走 **train.py 标准路径**（非 train_sscl.py——其保守冻结会连新分支一起冻住），完整 schedule（100 epochs 量级，DIOR train 5,862 张）；
- 加 **线性 warmup**：SPM/RGM 随机初始化，前期注入噪声需平滑；
- 监控 `spm.*`/`rgm.*` 参数**梯度范数**，确认新分支真的在学（而非塌缩成近恒等）；
- **公平归因**：baseline（无 SGM 分支）与 Phase 1/2 用完全相同 recipe（epochs/LR/增广/schedule/DINOv2 策略）训练，只比增量。

### 5.5 分阶段推进

| 阶段 | 内容 | decoder | 验证指标 |
|---|---|---|---|
| **Phase 1** | SPM+SGM+RGM 同尺度融合 → 单级 fused P4 | 不动（加载+低LR微调） | AP50/AP75 是否有小幅提升 |
| **Phase 2** | 补 stride-8 → 多级（fused P3+P4） | 升级 n_levels=2，warm-start+重训 | **AP_S 是否有真提升**（核心判据） |
| **Phase 3（可选）** | RepNCSPELAN 尺度内交互、PAN 下采样路、VGGBlock 重参数导出 | — | 精度进一步打磨 |

---

## 6. 风险与对照

| 风险 | 说明 | 对策 |
|---|---|---|
| **新分支训练不充分** | 随机初始化 + LR 太低 + epoch 太少 | 满 LR + 完整 schedule + warmup + 梯度范数监控 |
| **frozen decoder 抵抗新特征** | 冻结 decoder 会把新分支逼成 no-op | decoder 必须低 LR 协同微调 |
| **decoder 从零重训退化** | RF-DETR 无 denoising queries，小数据从零训 DETR decoder 收敛难 | Phase 1 加载不重训；Phase 2 只 warm-start，不从零 |
| **归因不清** | 同时改架构和训练配方 | baseline 与实验组 recipe 完全一致 |
| **延迟上升** | 新增 SPM(~2G)+RGM(+1.7G) | Phase 1 约 +5-8%；Phase 2 +20-30%，接受 |
| **与 SSCL 的协同** | encoder 特征变化 → 原型库需重新累积 | SGM 分支跑通后再接入 SSCL，需重训 |

---

## 7. 实现文件清单

**新增**
- `src/rfdetr/models/backbone/sga.py`：`SpatialPriorModule`、`SemanticGuidingModule`（SGM）、`SpatialGate`、`ChannelGate`、`ReciprocalGuidanceModule`、投影层

**修改**
- `src/rfdetr/models/backbone/backbone.py`：`__init__` 实例化新模块；`forward`/`forward_export` 插入融合逻辑
- `src/rfdetr/models/backbone/__init__.py`：`build_backbone` 透传开关
- `src/rfdetr/models/lwdetr.py`：`build_model` 透传开关
- `src/rfdetr/config.py` + `_namespace.py`：新增 `use_sga` 等配置字段
- （Phase 2）训练配置：`projector_scale`/`num_feature_levels` 调整 + decoder warm-start 加载逻辑

---

*参考：SKYDET 论文全文见 `docs/参考论文/SKYDET.md`；本方案中 SPM/SGM/RGM 结构均照 SKYDET 已确认的实现配方，RGM 方向遵循论文消融结论（语义→空间、纹理→通道，不可调换）。*
