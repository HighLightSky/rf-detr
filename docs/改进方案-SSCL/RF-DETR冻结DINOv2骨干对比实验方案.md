# RF-DETR 冻结 DINOv2 骨干对比实验方案

## 1. 背景与动机

当前一阶段 baseline（`output/0805-SHWX-data-expand-rfdetr-baseline`）是用 `src/scripts/train.py`
从 RF-DETR 发布权重出发、**全量微调 100 epoch** 得到的（`lr_encoder=1.5e-4`，骨干与 encoder 一起训练）。

这引出一个问题：**SHWX 训练集规模有限，全量微调 100 轮后，DINOv2 骨干是否对训练集过拟合了？** 如果过拟合，
骨干虽然训练集上特征适配，但测试集泛化可能受损。

本实验验证一个经典假设：**冻结 DINOv2 原始预训练权重、只训练骨干之外的参数（encoder / decoder / 分类头 / 回归头）**，
通用 DINOv2 特征更稳健，在测试集上可能泛化更好。

**本实验不含 SSCL**——只对比"一阶段微调"在冻结骨干 vs 全量微调两种情况下的检测性能。

## 2. 实验设计

### 2.1 两组对比

| | 对照组（已有） | 实验组（本方案） |
|---|---|---|
| 起点 | RF-DETR 发布权重（DINOv2 原始预训练 + COCO decoder） | 同左 |
| **DINOv2 骨干** | 全量微调（`lr_encoder=1.5e-4`） | **冻结（不训练）** |
| encoder / decoder / 分类头 / 回归头 | 全量微调 | 全量微调 |
| SSCL | 无 | 无 |
| 超参 | epochs=100, batch=16×4, lr=1e-4, lr_drop=60, mosaic=0.8, AUG_AERIAL, EMA | **与对照组完全一致** |

唯一变量是**骨干是否冻结**，其余超参保持相同，保证归因干净。

### 2.2 输出

- 新输出目录，例如 `output/0808-SHWX-rfdetr-baseline_frozen_backbone`
- 训练完用 `src/scripts/test.py` 测同一测试集，得到 test_result.txt

## 3. 预期与判断标准

| 结果 | 结论 | 后续动作 |
|---|---|---|
| 冻结骨干 ≥ 全量微调（总指标/各大类指标持平或更好） | 假设成立：全量微调过拟合了训练集 | 用冻结骨干 baseline 作为二阶段 SSCL 的起点，重新跑投影头实验 |
| 冻结骨干 < 全量微调 | 假设不成立：全量微调特征更适合 SHWX | 保留 0805 baseline，SSCL 二阶段继续用它 |

判断用比赛口径：**总指标（三大类 macro 平均）与各大类指标（小类平均）的 precision / recall / F1**，不只看 val mAP。

## 4. 实现方案

### 4.1 config：新增冻结开关

`TrainConfig` 增加字段（`src/rfdetr/config.py`）：

```python
backbone_freeze: bool = False
"""是否冻结 DINOv2 骨干（仅训练骨干之外的 encoder/decoder/head）。"""
```

### 4.2 module_model：加载预训练权重后冻结骨干

在 `RFDETRModelModule.__init__` 中，`load_pretrain_weights` **之后**（保证加载到的权重被冻结、不被重新随机化）：

```python
if train_config.backbone_freeze:
    for param in self.model.backbone[0].parameters():  # model.backbone[0] = DINOv2（Joiner[0]）
        param.requires_grad = False
```

- `model.backbone` 是 `Joiner = nn.Sequential(DINOv2, position_embedding)`，`[0]` 即 DINOv2 ViT。
- position embedding（`[1]`）保持可训练（微小的位置编码参数，冻结收益有限）。
- 冻结后 backbone 参数被 `get_param_dict` 的 `requires_grad` 过滤自动移出优化器，`lr_encoder` 失效（可保留原值，无害）。

### 4.3 train.py：常量开关

在训练参数区加开关并传入 `model.train(...)`：

```python
FREEZE_DINOV2 = True  # 冻结 DINOv2 骨干（对比实验变量）

# 传入 train()
backbone_freeze=FREEZE_DINOV2,
```

建议复制 `train.py` 为 `train_frozen_backbone.py` 或直接在 `train.py` 里用常量切换（当前 baseline 已跑完，`OUTPUT_DIR` 指向新目录即可，不会覆盖）。

### 4.4 注意：超参是否需要调整

- **保持超参一致**（干净的对比），先不做任何调整。
- 若冻结骨干后欠拟合（训练损失不降 / 指标差），可选的补偿旋钮（非默认）：调大 `LR`（如 1e-4 → 3e-4）或增加 `EPOCHS`。先按一致超参跑，结果出来再决定。

## 5. 与 SSCL 实验的关系

本实验是**一阶段基线对比**，为二阶段 SSCL 提供起点选择：

```
一阶段：全量微调 baseline（0805）   ←── 对照组
一阶段：冻结骨干 baseline（本实验） ←── 实验组
                    ↓ 择优
二阶段：SSCL（原型 + 投影头 + 实例正样本）在选中的一阶段 checkpoint 上微调
```

若冻结骨干胜出，后续投影头实验（`docs/改进方案-SSCL/RF-DETR-SSCL投影头改进方案.md`）
应从冻结骨干 baseline 重新出发，而不是从 0805。

## 6. 风险与注意

1. **冻结骨干可能欠拟合**：可训练参数显著减少（DINOv2 占大头），若 SHWX 需要骨干适配才能学精，冻结版本会差。这正是本实验要回答的问题。
2. **显存与速度**：骨干冻结后无梯度流经 DINOv2，`gradient_checkpointing` 对其失效但无害；可保留原设置，不做改动。
3. **EMA 与冻结**：EMA 会照常复制所有权重（含冻结的骨干），不受影响。
4. **与 LoRA 的区别**：本实验是"完全冻结骨干 + 直接训练头部"，不是 LoRA 的低秩适配；如需参数高效微调再考虑 LoRA 变体。

## 7. 验证命令

```bash
# 训练（在 train.py 中设 FREEZE_DINOV2=True、OUTPUT_DIR 指向新目录）
python src/scripts/train.py

# 测试（与 baseline 同口径）
python src/scripts/test.py --weights output/0808-SHWX-rfdetr-baseline_frozen_backbone/checkpoint_best_total.pth ...
```
