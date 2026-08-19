# ProtoGuidance 对齐修复实验

这组实验固定使用 SHWX 重标注数据、1024 分辨率和当前最佳 25 类 checkpoint。先运行 00--02 做现有辅助损失对照，再按 10 -> 11/12 -> 20/21/22 顺序筛选；后续 20 epoch 确认实验只从通过前一阶段判据的 checkpoint 续训。

## 运行

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/00_baseline_1024.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/01_selected_ce_current.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/02_selected_ce_kmeanspp.yaml
```

Dense 对齐实验：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/10_dense_ce_token_only.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/11_dense_ce_unfrozen_projection.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/12_dense_ce_lse_slots.yaml
```

若 `10` 出现训练/验证 group 不一致，先运行 group 0 修复实验：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/10b_dense_ce_group0_lr1e4.yaml
```

以 `10b/last_ema.pth` 为共同起点的并行实验：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/11_dense_ce_unfrozen_projection.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/12_dense_ce_lse_slots.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/20b_selection_group0_lambda025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/21b_selection_group0_lambda050.yaml

## 前景性修复（group_detr=1）

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/13_dense_foreground_group1.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/14_foreground_selection_group1_lambda025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/15_foreground_selection_group1_lambda050.yaml
```

`13` 只训练 token 投影和独立 foreground head。仅当验证集记录的
`proto_dense_fg_positive_mean` 高于 `proto_dense_fg_background_mean`，且二者差距
稳定扩大时，才启动 `14/15` 的 selection 强度扫描。

MS 语义选择与 decoder 适应对照：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/16_foreground_selection_ratio025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/20_ms_semantic_selection_w025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/21_ms_semantic_selection_w050.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/22_decoder_adapt_baseline.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/23_ms_semantic_decoder_adapt.yaml
```

`22/23` 使用同一原型对齐 checkpoint 与相同的保守解冻范围。两者的检测差异因而可以归因于
MS 的 prototype selection，而非 decoder 微调本身。

若 `23` 提升 MS 但 FSC 回退，运行降低位置 residual 的保护实验：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/24_ms_semantic_decoder_adapt_lambda025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/25_ms_semantic_decoder_adapt_lambda0375.yaml
```

选择实验：

```bash
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/20_dense_selection.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/21_dense_selection_lambda025.yaml
uv run --no-sync python src/scripts/train.py -c configs/experiments/proto_guidance_alignment_repair/22_dense_selection_lambda050.yaml
```

## 判据

重点读取 `metrics.csv` 中的 `train/loss_proto_dense_enc`、`train/proto/*`、验证集 recall 和逐类 recall。dense 正样本准确率、margin 和正样本数量会记录在 criterion 诊断字段中；未达到 dense 对齐判据时，不运行 query selection 和 content enhancement 实验。

## 阶段五状态

`30_dense_hardneg_ms.yaml` 目前只启用 MS 采样，P3 多尺度原型尚未实现。`31_dense_hardneg_fsc_truck.yaml` 明确使用 26 类 truck 数据，但尚无 26 类 ProtoGuidance 原型产物，因此配置关闭 ProtoGuidance，仅作为数据/检测基线。`32_dense_hardneg_aircraft_pairs.yaml` 等待 pairwise margin loss 实现，不能作为最终结论实验。

## 配置约束

- `proto_guidance_trainable_scope: token` 只训练 `proj_token`，用于判断 encoder token 投影是否是主要瓶颈。
- `proto_guidance_slot_reduction: lse` 使用温度化 log-sum-exp，避免硬 `max` 只给一个 slot 梯度。
- 评估时 ProtoGuidance 的 `eval()` 自动使用 warmup 完成强度；训练 epoch hook 会覆盖为当前 epoch。
