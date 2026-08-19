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
