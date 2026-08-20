# PAN 分支模型：实现与复现

## 目标与定位

PAN 分支负责单波段全色影像中的舰船细粒度类别：`HM`、`LQS`、`QHS`、`MS`（全局类别 ID `0` 至 `3`）。它使用的权重为：

```text
output/sscl-macro-balanced-v3-ship-hardneg/checkpoint_best_total.pth
```

该权重并不是仅含四类的局部模型，而是一个全局 25 类检测器。在双分支推理中，路由到 PAN 的图像只保留该模型的 `0` 至 `3` 类预测，其他类别预测被丢弃。这样可以保留完整训练数据带来的共享表征，同时针对船舶类定向降低虚警。

## 网络与训练方案

基础网络是 `RFDETRMedium`：`dinov2_windowed_small` 编码器、4 层 two-stage decoder、300 个 query、隐藏维度 256，输入尺寸为 `1024 x 1024`，检测头输出 25 个前景类别和 1 个背景类别。

训练从下列 ProtoGuidance E4-hardneg-FSC 权重开始，而不是从官方权重重新训练：

```text
output/0816在120ep1024基础上续训多模态原型-效果最好/
0816-SHWX-ProtoGuidance-E4-hardneg-fsc-次好-当前最佳/checkpoint_best_total.pth
```

在此初始化之上，PAN 分支继续训练 18 个 epoch。普通检测损失之外，方案有三部分：

1. ProtoGuidance：加载 `data/proto_guidance_shwx.pt` 中的离线原型，启用 encoder two-stage query 位置引导、decoder 内容残差及 matched foreground 的原型辅助分类损失。原型槽数为 10，温度为 `0.2`；位置和内容残差从 `0.05` warm up 到 `1.0`，warm up 为前 2 个 epoch。
2. SSCL 多原型：仅面向舰船类别 `0` 至 `3`。在 decoder 最后一层 Hungarian matched query 的 128 维投影空间维护 EMA 原型库，动量 `0.99`；每类最多两个槽，类别对为 `[HM, LQS]` 与 `[QHS, MS]`，组损失权重为 `1.25`。SSCL 系数为 `0.04`，温度 `0.1`。采用 `conservative` 冻结策略，仅解冻 decoder 最后两层以减小检测能力漂移。
3. 难负样本抑制：从未匹配 query 中选择与 GT IoU 在 `[0, 0.15]` 的低 IoU 高分候选，目标类别为 `HM`、`LQS`、`QHS`（`0, 1, 2`）。每图按类均衡选择，每类最多一个候选、总数最多 5；从第 2 个 epoch 起施加。logit 抑制权重为 `0.12`，并以 ProtoGuidance 原型相似度补充抑制，权重为 `0.08`。

训练使用 SHWX 全局 25 类 YOLO 数据集，`AERIAL` 增强包含水平翻转、垂直翻转、90 度旋转和亮度对比度扰动；不使用 Mosaic 或多尺度训练。batch size 为 32，梯度累积为 2，有效 batch 为 64；AdamW 的 backbone 与其他参数学习率均为 `1e-5`，weight decay 为 `1e-4`。EMA 启用，衰减为 `0.993`。最终 `checkpoint_best_total.pth` 选择的是 EMA 权重。

完整且可执行的参数以 [训练配置](../../configs/experiments/train_sscl_multproto_hardneg_macro_balanced_v1.yaml) 和输出目录的 `training_config.json` 为准。

## 复现前提

1. 安装项目完整开发环境：

   ```bash
   uv sync --all-groups
   ```

2. 准备全局 25 类 SHWX YOLO 数据集，并将配置中的 `train.dataset_dir` 改为实际路径。训练集目录必须具有 `images/train`、`images/val`、`labels/train`、`labels/val` 布局，类别 ID 必须保持全局映射：`0-3` 为舰船，`4-23` 为飞机，`24` 为 `FSC`。

3. 准备初始化 checkpoint、`data/proto_guidance_shwx.pt` 和 `data/semantic_matrix_shwx.pt`。后两者是本方案的必要输入；缺少它们无法等价复现 ProtoGuidance 与 SSCL。若没有上述初始化 checkpoint，只能复现同一训练结构，不能复现该实验的权重轨迹和指标。

4. 使用 CUDA GPU。原始实验使用单卡 CUDA、6 个数据加载 worker；显存不足时可等比例降低 `batch_size` 并增大 `grad_accum_steps`，保持有效 batch 为 64。

## 训练与验证

先检查 YAML 展开后的实际参数：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/train_sscl_multproto_hardneg_macro_balanced_v1.yaml \
  --dump-kwargs
```

确认路径无误后启动训练：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/train_sscl_multproto_hardneg_macro_balanced_v1.yaml
```

训练完成后，使用全局模型进行单模型评测：

```bash
uv run --no-sync python src/scripts/test.py \
  -c configs/experiments/train_sscl_multproto_hardneg_macro_balanced_v1.yaml \
  --set test.dataset_dir=/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test \
  --set test.resolution=1024
```

原始单模型测试记录中，舰船 macro 指标为 Recall `0.8664`、FDR `0.0961`。这是该分支被选入双分支方案的主要原因。单模型评测仍会报告全部 25 类；它不等价于双分支合并指标。

## 接入双分支评测

将 PAN checkpoint 填入 [双分支测试配置](../../configs/experiments/test_dual_shwx_medium.yaml) 的 `pan_checkpoint`，并保持：

```yaml
pan_checkpoint_class_space: global
```

合并评测命令如下：

```bash
uv run --no-sync python src/scripts/test_dual_shwx.py \
  -c configs/experiments/test_dual_shwx_medium.yaml
```

脚本会先以训练/验证集非空标签校准 RGB 通道平均绝对差异阈值，再对测试图像路由。当前实验的阈值是 `0.015033`；路由到 PAN 的图像仅接受类别 `0-3` 的输出。最终报告写入 `output/dual-v3-pan-proto00-rgb-1024/test_dual/test_result.txt`。

## 固定阈值单阶段续训（当前推荐）

此前的 P1 诊断曾在 full_test 上搜索逐类阈值；该结果不代表模型能力，已从最终方案中废弃。当前统一使用
`conf=0.25`，不配置 `class_conf_thresholds`，并把 full_test 仅作为内部 checkpoint 选择集。

推荐配置为 [PAN从120ep单阶段复现-v2-背景抑制.yaml](../../configs/experiments/双分支/PAN续训实验/PAN从120ep单阶段复现-v2-背景抑制.yaml)。
它从 120 epoch 全类基线直接启动一次 14 epoch 续训，中途不更换初始化 checkpoint、不重建优化器，也不使用
教师蒸馏。训练始终读取全局 25 类数据，而不是只含四类标签的 PAN 路由缓存，因此保留 25 类检测头和共享表征；
非船类允许小幅下降，但不能因缺失监督而崩溃。

训练目标固定为三项：

1. ProtoGuidance 使用 `data/proto_guidance_shwx_1024_current.pt`，只打开船舶类 foreground-semantic 位置引导
   和 dense 前景监督，关闭 content residual 与辅助分类损失。位置系数在同一训练进程内从 `0` warm up 到
   `0.35`，避免全色灰度边缘域的强内容注入造成额外虚警。
2. SSCL 使用 128 维投影空间的两槽 EMA 原型，重点约束 HM/LQS、QHS/MS 两组易混船舶。原型库从训练开始积累，
   SSCL 损失从 epoch 3 开始参与优化；该门控只是同一 run 内的损失调度，不是拆分训练阶段。
3. 难例抑制从 epoch 6 参与优化，只选择 IoU `[0, 0.08]` 且预测类别不出现在当前图 GT 中的低 IoU 高分
   query。目标类仅为 HM/LQS/QHS，MS 永久排除；损失权重降为 `0.02`，避免密集小目标召回被误压。

选模先检查 HM/LQS 不低于当前 PAN 的召回，再比较 MS/QHS 的固定 `0.25` 召回与船舶 FDR；最后检查 21 个
非船类是否出现灾难性下降。自动 best 按全类 mAP 选择，不能代替该规则。当前选定的是 epoch 7 regular：
`checkpoint_pan_selected_fixed025.pth`。任何依赖逐类阈值的 checkpoint 都不进入最终候选。

该主线从不含旧 ProtoGuidance buffer 的 120 epoch 基线启动，因此可以明确加载
`data/proto_guidance_shwx_1024_current.pt`，不存在“YAML 指向新产物、checkpoint 又覆盖成旧原型”的溯源歧义。
从旧 PAN checkpoint 出发的 `PAN统一续训-v1/v2` 仅用于确认损失边界，不作为正式复现路径。

统一续训 v1 的失败消融进一步确认了这一边界：将 MS 加入难例集合后，首个 hard-negative epoch 中 MS 占
全部候选的约 95%，固定 `0.25` 的 MS TP 从 1185 降到 1156，虽然 FP 从 258 降到 211，但这是不可接受的
召回换虚警。v2 排除 MS 后，每图难例数从 `0.0536` 降到 `0.0025`，且仅出现 QHS 候选，属于轻量校正。

120 epoch 基线 checkpoint 缺少 `num_queries/group_detr` 元数据。默认 13 组加载只能执行 flat slice，存在
query 分组错位风险；稳定主线因此显式使用 `group_detr=1`。虽然载入时会报告额外 group 参数未消费，但推理
使用的第 0 组完整载入，且已通过 fixed `0.25` 的 full_test 实测验证。

同口径 fixed `0.25` 结果如下：120ep 基线为 `1424 TP / 320 FP`，旧 PAN 为 `1415 / 298`，当前选定权重为
`1427 / 310`。新权重相对基线同时增加 3 个 TP、减少 10 个 FP；相对旧 PAN 增加 12 个 TP，但仍增加 12 个
FP，因此它是召回优先的稳定复现结果，而不是对旧 PAN 的严格 Pareto 超越。继续训练到 epoch 13 会退化为
`1421 / 304`，所以不得用最后一轮覆盖选定权重。

这里必须区分两种报告：

- `full_test_selected_fixed025/test_result.txt` 是 PAN 单模型 25 类报告，直接统计选定 PAN 权重的输出；它反映
  PAN 分支自身能力，报告中船舶为 `1426/309`、全类为 `6823/459`。
- `full_test_selected_dual_fixed025/test_result.txt` 是 RGB+PAN 路由后的双分支报告；它反映实际部署合并结果，
  报告中船舶为 `1427/310`、全类为 `6802/469`。

两者使用相同权重、相同 full_test 和 `conf=0.25`，差异来自 RGB/PAN 模态路由以及双分支合并规则，不是训练
结果不同。讨论“PAN 分支最好”时应看前者；讨论最终双分支部署时应看后者。

## 下一次复现的实际步骤

可以直接启动训练：

```bash
uv run --no-sync python src/scripts/train.py \
  -c configs/experiments/双分支/PAN续训实验/PAN从120ep单阶段复现-v2-背景抑制.yaml
```

但不要直接把训练输出的 `checkpoint_best_total.pth` 当作 PAN 最终权重。训练回调的 `best_total` 只比较验证集
`mAP_50:95`（regular 与 EMA 二选一），不使用 full_test、固定 `0.25`、船舶 Recall 或船舶 FP。当前这次
运行中，`checkpoint_best_total.pth` 来自 EMA 的 epoch 2，而固定阈值选出的 PAN 权重来自 epoch 7 regular。

重做数据集后，建议保持一次训练不拆段，但执行以下选模流程：

1. 修改 `train.dataset_dir`，若训练图像/标签语义发生变化，同时重新生成匹配的 ProtoGuidance 原型和
   `semantic_matrix_shwx.pt`，不要沿用语义不一致的旧产物。
2. 使用新的 `train.output_dir` 启动一次完整训练，保留 `checkpoint_interval` 周期 checkpoint。
3. 对 epoch 5-9 以及 `checkpoint_best_regular.pth`、`checkpoint_best_ema.pth` 逐个用固定
   `conf=0.25` 评测；先看 PAN 单模型报告，再看 RGB+PAN 双分支报告。
4. 以 HM/LQS 召回不下降、MS 召回不发生明显下降为硬约束，在船舶 TP/FP 中选择最终权重，并复制为
   `checkpoint_pan_selected_fixed025.pth`。只有这个文件进入最终双分支配置。

因此，配置负责复现训练轨迹；`checkpoint_pan_selected_fixed025.pth` 才是面向 PAN 目标选出的最终模型。

## 探索过程中的训练实现
当前 PAN 分支不是一个“只训练 PAN 四类”的模型，而是全程保持全局 25 类 `RFDETRMedium`，最终在双分支推理时仅保留 `0-3`（HM/LQS/QHS/MS）输出。

训练路径如下：

```text
官方 RF-DETRMedium
  -> 阶段 0：120 epoch 全类基线
  -> 阶段 1：20 epoch ProtoGuidance + 船舶 SSCL/难负样本
  -> 阶段 2：20 epoch ProtoGuidance + FSC 定向保护
  -> 阶段 3：18 epoch PAN v3 多原型 + 船舶难负样本
  -> checkpoint_best_total.pth
```

1. **阶段 0，120 epoch 全类基线**

`output/0815-SHWX-rfdetr-medium-baseline-精细标注-1024/checkpoint_best_total.pth`

- 全局 25 类、1024 分辨率、RFDETRMedium。
- 从官方 `rf-detr-medium.pth` 初始化，全量微调。
- `lr=1e-4`，encoder `1.5e-4`，有效 batch 64。
- AERIAL 增强，`mosaic_p=0.1`，开启多尺度。
- 对 HM/LQS (`0,1`) 做类别均衡采样。
- 不启用 ProtoGuidance、SSCL 或难负样本。
- 完整跑至 epoch 119；最佳 EMA 权重出现在 step 7475。

2. **阶段 1，20 epoch 原型引导和船舶判别强化**

`output/.../0816-SHWX-ProtoGuidance-1024/checkpoint_best_total.pth`

从 120 epoch 基线以 `pretrain_weights` 初始化，重新建立优化器，不是 `resume`。

- 开启 ProtoGuidance：`data/proto_guidance_shwx.pt`，10 个槽位，位置引导、内容残差、原型辅助分类损失全部开启。
- SSCL 仅针对船舶 `0-3`：`lambda=0.02`，保守冻结策略，解冻 decoder 后两层。
- 难负样本抑制目标为 `0,1,2`，IoU 区间 `[0, 0.15]`。
- 学习率降至 `1e-5`，20 epoch，AERIAL 增强；关闭 Mosaic 和多尺度。
- 继续对 HM/LQS 做采样均衡。
- 实际跑完 20 epoch，但 `best_total` 的最佳 EMA 出现在 step 65，不是最后一轮。

3. **阶段 2，20 epoch E4-hardneg-FSC**

`output/.../0816-SHWX-ProtoGuidance-E4-hardneg-fsc-次好-当前最佳/checkpoint_best_total.pth`

从阶段 1 最佳权重初始化。

- 保留阶段 1 的 ProtoGuidance 和船舶 SSCL。
- 类别均衡采样扩展为 `0,1,24`，将 FSC 纳入采样补偿。
- 难负样本目标扩展为 `0,1,2,24`。
- 为避免压低 FSC 召回，FSC 使用更宽松的难负样本参数：margin `-2.5`，该类损失系数 `0.15`；其余目标类保持 margin `-1.5`、全局难例系数 `0.3`。
- 20 epoch，`lr=1e-5`，保守冻结 decoder 后两层。
- 实际跑完 20 epoch；最终采用的 EMA 最佳权重出现在 step 66。

4. **阶段 3，PAN v3 多原型船舶强化**

`output/sscl-macro-balanced-v3-ship-hardneg/checkpoint_best_total.pth`

这是双分支方案实际使用的 PAN 权重。

- 从阶段 2 最佳权重初始化。
- ProtoGuidance 保持全开，仍使用 10 个融合原型槽。
- SSCL 强度从 `0.02` 增至 `0.04`，只聚焦船舶 `0-3`。
- 开启 128 维投影空间的 EMA 多原型库：每个船舶类最多 2 个原型槽，类别组为 `[HM,LQS]`、`[QHS,MS]`，组内难负权重 `1.25`。
- 难负样本只作用于 `0,1,2`，不再压 FSC。采用按类均衡选择，每类最多 1 个、每图最多 5 个；IoU `[0,0.15]`，从 epoch 2 开始；难例系数降至 `0.12`。
- 训练 18 epoch，`lr=1e-5`，有效 batch 64；关闭类别均衡采样、Mosaic 和多尺度。
- 实际完成 epoch 17、step 1134，但最终最佳 EMA checkpoint 出现在 step 441。

所以，PAN 分支的核心演化是：

- 阶段 0 建立全类检测能力。
- 阶段 1/2 用 ProtoGuidance、SSCL 和难负样本让船舶类别更可分，同时阶段 2 曾兼顾 FSC。
- 阶段 3 去掉对 FSC 的抑制，把所有额外判别能力集中到船舶多原型与船舶低 IoU 虚警，得到船舶 macro FDR `0.0961` 的 PAN 候选。

对应最终阶段配置为 [train_sscl_multproto_hardneg_macro_balanced_v1.yaml](/home/liu/wzt/rf-detr/configs/experiments/train_sscl_multproto_hardneg_macro_balanced_v1.yaml:1)。
