# RGB 分支模型：实现与复现

## 目标与定位

RGB 分支负责彩色影像中的飞机类别和发射车 `FSC`，即全局类别 ID `4` 至 `24`。它使用的权重为：

```text
output/proto_guidance_alignment_repair/00_baseline_1024/checkpoint_best_total.pth
```

该分支同样是全局 25 类的 `RFDETRMedium`，不是仅包含 21 类的局部分类空间模型。在双分支推理中，仅对路由到 RGB 的图像保留类别 `4` 至 `24` 的预测。PAN 类预测会被丢弃。

## 网络与训练方案

网络主体与 PAN 分支一致：`RFDETRMedium`，`dinov2_windowed_small` 编码器、4 层 two-stage decoder、300 个 query、隐藏维度 256，输入尺寸为 `1024 x 1024`，输出全局 25 个前景类别和 1 个背景类别。

它从与 PAN 分支相同的 ProtoGuidance E4-hardneg-FSC checkpoint 初始化：

```text
output/0816在120ep1024基础上续训多模态原型-效果最好/
0816-SHWX-ProtoGuidance-E4-hardneg-fsc-次好-当前最佳/checkpoint_best_total.pth
```

因此不能把 RGB 分支称为“RF-DETR 训练 140 轮”的直接结果。权重链为：官方 `RFDETRMedium` 权重 -> 120 个 epoch 的 1024 基线 -> 20 个 epoch 的 ProtoGuidance-1024（此处累计才是 140）-> 20 个 epoch 的 E4-hardneg-FSC -> 本实验计划的 5 个 epoch 普通检测微调。按训练阶段名义累计为 165 个 epoch；每个阶段都是以 `pretrain_weights` 加载模型参数并重新建立优化器，故该累计值用于描述权重来源，不能当作一次连续 `resume` 训练的 epoch 计数。

`00_baseline_1024` 实际运行了 5 个 epoch，但 `checkpoint_best_total.pth` 的 `global_step=63`，而该 run 结束时 `last.ckpt` 的 `global_step=315`。这说明用于双分支的最佳 EMA 权重是在首个微调 epoch 的验证后选出的。此实验有意关闭了 `proto_guidance_enabled` 与 `sscl_enabled`，也不启用 SSCL 原型库、难负样本抑制、类别均衡采样、Logit Adjustment 或补丁粘贴。它的作用是作为稳定的 RGB 检测分支：保留初始化权重已经获得的检测能力，再用较短的全参数检测微调适配 RGB 图像。

训练使用 SHWX 全局 25 类 YOLO 数据集，采用 `AERIAL` 增强：水平翻转、垂直翻转、90 度旋转及亮度对比度扰动。不使用 Mosaic 和多尺度训练。batch size 为 8，梯度累积为 8，有效 batch 为 64；AdamW 的 backbone 与其他参数学习率均为 `1e-5`，weight decay 为 `1e-4`。每个 epoch 验证一次，EMA 启用且衰减为 `0.993`；`checkpoint_best_total.pth` 实际选用 EMA 权重。

完整生效参数以 [训练配置](../../configs/experiments/proto_guidance_alignment_repair/00_baseline_1024.yaml) 与输出目录的 `training_config.json` 为准。

## 复现前提

1. 安装完整开发环境：

   ```bash
   uv sync --all-groups
   ```

2. 准备全局 25 类 SHWX YOLO 数据集，并在配置中设置 `train.dataset_dir`。目录需要包含 `images/train`、`images/val`、`labels/train`、`labels/val`；类别 ID 需与全局 SHWX 映射一致。

3. 准备上述 E4-hardneg-FSC 初始化 checkpoint。RGB 微调虽然关闭 ProtoGuidance 和 SSCL，但该 checkpoint 是权重初始化的一部分；替换为其他 checkpoint 将不再是本实验的可比复现。

4. 使用 CUDA GPU。原始实验使用单卡和 6 个数据加载 worker。若调整 batch size，需要同步调整梯度累积以尽量保持有效 batch 为 64。

## 训练与验证

首先检查配置展开结果：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/proto_guidance_alignment_repair/00_baseline_1024.yaml \
  --dump-kwargs
```

启动 RGB 分支训练：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/proto_guidance_alignment_repair/00_baseline_1024.yaml
```

单模型全局评测可使用专用测试配置，并覆盖 checkpoint：

```bash
uv run --no-sync python src/scripts/test.py \
  -c configs/experiments/test_shwx.yaml \
  output/proto_guidance_alignment_repair/00_baseline_1024/checkpoint_best_total.pth \
  --set test.dataset_dir=/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test \
  --set test.resolution=1024
```

原始单模型评测中，飞机 macro 指标为 Recall `0.9962`、FDR `0.0194`；`FSC` 为 Recall `0.8263`、FDR `0.2887`。这些是单模型全类别评测结果，不能直接当作双分支合并后的总体指标。

## 接入双分支评测

在 [双分支测试配置](../../configs/experiments/test_dual_shwx_medium.yaml) 中使用下列设置：

```yaml
rgb_checkpoint: output/proto_guidance_alignment_repair/00_baseline_1024/checkpoint_best_total.pth
rgb_checkpoint_class_space: global
```

运行双分支评测：

```bash
uv run --no-sync python src/scripts/test_dual_shwx.py \
  -c configs/experiments/test_dual_shwx_medium.yaml
```

路由逻辑位于 `src/scripts/dual_shwx.py`：根据图像三个颜色通道的平均绝对差异，低于校准阈值的图像走 PAN，其余图像走 RGB。测试集路由完成后，RGB 分支只保留类别 `4-24`；因此 PAN 与 RGB 两支没有对同一类别的预测做 NMS 或加权融合，而是按模态和类别空间直接拼接预测记录。


## 如何从头训练
要训练出功能等效的 RGB 分支，不要从官方 RF-DETR 直接训练 140 轮。应复用已训练好的 E4-hardneg-FSC 初始化权重，再执行 RGB 的 5 轮纯检测微调：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/proto_guidance_alignment_repair/00_baseline_1024.yaml \
  --set train.dataset_dir=/你的/SHWX-dataset-dict-redo \
  --set train.output_dir=output/repro-rgb-1024
```

配置中的 `model.pretrain_weights` 必须指向：

```text
output/0816在120ep1024基础上续训多模态原型-效果最好/
0816-SHWX-ProtoGuidance-E4-hardneg-fsc-次好-当前最佳/checkpoint_best_total.pth
```

训练完成后使用 `output/repro-rgb-1024/checkpoint_best_total.pth`，不要用 `last.ckpt`。这会保留原方案的关键条件：25 类 `RFDETRMedium`、1024 输入、EMA、5 epoch、`lr=1e-5`、有效 batch 64、AERIAL 增强，同时关闭 ProtoGuidance 与 SSCL。

若该初始化权重不存在，需按以下链路重建：

```text
官方 RFDETRMedium
  -> 120 epoch 1024 全量基线
  -> 20 epoch ProtoGuidance
  -> 20 epoch E4-hardneg-FSC
  -> 5 epoch RGB 纯检测微调
```

即名义累计 165 个 epoch，不是 140。前 3 阶段还需要原始 SHWX 数据划分、`data/proto_guidance_shwx.pt` 与 `data/semantic_matrix_shwx.pt`。由于原实验未固定随机种子，重新训练无法得到逐参数完全一致的 checkpoint；复用 E4 初始化权重后重跑最后 5 轮，才是成本最低且指标最接近的复现方式。