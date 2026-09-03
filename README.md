# 面向不均衡小样本遥感目标检测的 RF-DETR

本项目面向光学遥感卫星影像中的陆上目标检测识别任务，以 RF-DETR 为检测基座，针对
小样本、类别不均衡、细粒度类别混淆和复杂背景虚警进行改进。项目提供从训练、评估、
单图推理到比赛 Docker 交付的完整代码。

## 1. 任务与目标

数据集包含 25 个目标类别，类别实例数差异较大，其中舰船类别样本稀缺且相互外观相似。
输入还可能是约 `10000 x 10000` 像素的大幅面影像，因此模型需要同时处理以下问题：

- 稀有类别样本不足，导致分类偏置和漏检；
- 舰船等细粒度类别之间边界模糊，容易互相误检；
- 港口、道路、阴影和复杂纹理容易产生高置信背景虚警；
- 大图不能直接缩放到单张小图，需要在精度、覆盖范围和推理时延之间折中。

项目的优化目标是提高稀有类别和易混类别的召回率，降低背景虚警，并在目标硬件上完成
大图快速推理。最终指标以实际数据集和评测环境中的结果为准。

## 2. 总体技术路线

### 2.1 RF-DETR 检测基座

RF-DETR 是基于 LW-DETR 思路的两阶段 Transformer 检测器。模型使用 DINOv2 视觉骨干
提取多尺度特征，通过两阶段候选查询选择得到固定数量的查询，再由 Transformer 解码器
逐层细化目标位置和类别，最后输出分类结果与边界框。

```text
输入图像
   |
   v
DINOv2 骨干 + 多尺度投影
   |
   v
两阶段候选查询选择（位置提议）
   |
   v
Transformer 解码器（自注意力 + 可变形交叉注意力）
   |
   v
分类头 + 边界框回归头
   |
   v
目标类别、置信度和边界框
```

训练时使用匈牙利匹配将预测查询与标注目标对应，并联合优化分类焦点损失、L1 框回归
损失和 GIoU 损失。

### 2.2 一个锚点、两处引导、三类约束

在 RF-DETR 基础上，本项目引入多模态原型增强。整体可以概括为“一个原型锚点、两处
在线引导、三类训练约束”。

**一个原型锚点：多模态类别原型库**

1. 使用训练数据和骨干特征，在目标框区域提取视觉特征，并按类别进行余弦聚类，得到
   能表示不同角度、尺度和外观的多个视觉子原型。
2. 使用 CLIP 文本编码器编码类别名称和遥感场景提示词，得到文本原型。
3. 将视觉原型与文本原型投影到统一空间并融合。原型库在训练和推理中作为冻结的类别
   先验锚点使用，不参与在线更新。

**两处在线引导**

- **位置引导**：在两阶段查询选择前，计算记忆特征与类别原型的相似度和类别间隔，
  对原有目标性分数做残差修正，使更可能属于目标类别的特征进入解码器，同时保留基座
  模型的目标性判断。
- **内容引导**：对已经选中的查询，根据其最相关的类别和原型子槽位提取上下文，通
  过置信度控制的门控残差注入查询内容，让解码器获得更明确的类别和形态先验。

**三类训练约束**

- **原型辅助分类损失**：只对匈牙利匹配到的前景特征进行类别监督，并按类别均衡，
  让原型打分分支真正学习类别信息。
- **语义加权监督对比学习（SSCL）**：使用解码器末层的匹配前景查询作为特征，在投影
  空间中拉近同类样本；对 CLIP 语义上更相近的异类施加更强的分离压力，重点处理细粒
  度混淆类别。
- **难负样本抑制**：从未匹配查询中筛选与标注框处于指定 IoU 范围、且前景得分较高的
  候选，将其作为“像目标但不是目标”的背景样本，直接抑制其前景响应，减少港口、阴
  影和局部结构造成的虚警。

三类约束共同形成总训练目标：检测损失负责基本定位和分类，原型损失负责语义锚定，
SSCL 负责易混类别分离，难负样本损失负责前景与背景边界建模。

### 2.3 大图推理路线

对大幅面影像，部署流水线先使用边界模型定位可能包含目标的区域，再将区域切分为带
重叠的小块送入主检测模型。各裁块的预测框会映射回原图坐标，并通过 NMS 等后处理合
并重叠结果。这样可以保留小目标细节，同时避免整幅图缩放造成的目标尺寸过小。

## 3. 环境安装

要求 Python `>=3.10`。推荐使用仓库声明的 `uv` 环境：

```bash
pip install uv
uv sync --all-groups
```

如果只使用已经发布的基础包，也可以安装：

```bash
pip install rfdetr
```

GPU 训练和部署需要可用的 CUDA、PyTorch 以及对应驱动。Albumentations/Kornia 等增强依
赖属于可选组件，按 `pyproject.toml` 中的 extra 安装。

## 4. 数据集准备

训练入口支持 YOLO 和 COCO 数据格式。项目实验主要使用 YOLO 目录布局，示例结构如下：

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/                 # 也可使用 val/
│   ├── images/
│   └── labels/
└── test/                  # 可选
    ├── images/
    └── labels/
```

标注文件使用归一化的 YOLO 格式：`class_id x_center y_center width height`。类别编号需
与训练配置和原型/语义矩阵中的类别顺序一致。数据集根目录通过 YAML 的
`train.dataset_dir` 或 `test.dataset_dir` 指定，建议使用绝对路径或相对项目根的稳定路径。

## 5. 训练

### 5.1 统一 YAML 入口

训练脚本会读取 YAML，构造模型并将 `train:` 段参数透传给 `model.train(**kwargs)`。
配置错误会在启动时被校验。一个完整的多原型、SSCL 和难负样本实验可以这样启动：

```bash
uv run python src/scripts/train.py \
  -c configs/experiments/0807-SSCL对比学习/train_sscl_multproto_hardneg_suppress_v1.yaml
```

基线或纯 SSCL 实验可使用对应配置：

```bash
# 纯 SSCL/原型微调
uv run python src/scripts/train.py \
  -c configs/experiments/0807-SSCL对比学习/train_sscl_0807.yaml

# 查看最终 kwargs，不启动训练
uv run python src/scripts/train.py \
  -c configs/experiments/0807-SSCL对比学习/train_sscl_0807.yaml \
  --dump-kwargs
```

不复制 YAML 也可以使用 `--set` 覆盖单个字段：

```bash
uv run python src/scripts/train.py \
  -c configs/experiments/0807-SSCL对比学习/train_sscl_hardneg_k3.yaml \
  --set train.sscl_hard_neg_topk=5 \
  --set train.output_dir=output/exp-hardneg-k5
```

### 5.2 推荐训练策略

项目建议采用两阶段策略：

1. **联合适配阶段**：以 RF-DETR 基线权重为起点，让检测器和原型投影/引导模块共同适
   配，位置和内容注入从较小权重开始逐步增强。
2. **判别性微调阶段**：冻结骨干和编码器，保留解码器末层、分类头以及原型/SSCL 模块
   的可训练性，重点修正易混类别边界和背景虚警。

每个实验的冻结范围、学习率、损失权重、启动轮次和难例筛选阈值都应以实际 YAML 为准，
不要仅凭实验目录名称推断配置。

### 5.3 训练产物

训练输出目录通常包含：

- `checkpoint_best_total.pth`：按验证指标选出的模型权重；
- `checkpoint*.pth`：周期性或 EMA 权重；
- `training_config.json`：本次运行的完整生效配置；
- `metrics.csv`、TensorBoard 日志和验证结果：用于分析损失、mAP、召回率及原型/难例
  监控指标。

## 6. 评估与推理

### 6.1 批量评估

统一评估入口会执行批量推理、指标计算、混淆矩阵以及 FP/FN 分析。使用通用 SHWX 配置：

```bash
uv run python src/scripts/test.py \
  -c configs/experiments/train_tests/test_shwx.yaml
```

也可以用位置参数覆盖 checkpoint，或用 `--set` 覆盖测试参数：

```bash
uv run python src/scripts/test.py \
  -c configs/experiments/train_tests/test_shwx.yaml \
  /path/to/checkpoint_best_total.pth \
  --set test.conf_threshold=0.25 \
  --set test.save_fp_fn=true
```

评估配置中的常用字段包括：

- `checkpoint`、`dataset_dir`、`output_dir`：权重、数据和结果目录；
- `resolution`、`batch_size`、`num_workers`、`device`：推理资源配置；
- `conf_threshold`、`class_conf_thresholds`：全局和逐类置信度阈值；
- `save_fp_fn`、`save_yolo_preds`：是否保存误检/漏检可视化和 YOLO 预测文件；
- `large_image_min_side`、`boundary_checkpoint` 等：是否启用大图切块分支。

评估输出包括总体和逐类指标、混淆矩阵、FP/FN 样本以及大图推理耗时统计。模型选择不
应只看总体 mAP，还应结合稀有类召回、易混类别精确率和虚警数量。

### 6.2 单图或目录推理

预测入口对单张图片或整个目录生成 YOLO 格式预测文件和可视化结果：

```bash
uv run python src/scripts/predict.py \
  -c configs/experiments/train_tests/predict_shwx.yaml
```

配置中的 `predict.image` 指向图片或目录，`predict.output_dir` 指定输出位置。默认输出：

```text
output_dir/
├── labels/                 # class_id cx cy w h confidence
├── visualization/          # 预测框、类别和置信度
└── label_comparison/       # 开启 label_comparison 后生成
```

Python API 适合快速验证单个 checkpoint：

```python
from rfdetr import RFDETR

model = RFDETR.from_checkpoint("/path/to/checkpoint_best_total.pth")
results = model.predict("/path/to/image.jpg", threshold=0.25)
model.export(output_dir="output/onnx", format="onnx")
```

## 7. 部署

### 7.1 导出主检测和边界模型

比赛大图流程需要主检测模型和边界检测模型两个输入分支。仓库提供批量导出脚本：

```bash
uv run python src/scripts/onnx/export_shwx_onnx.py \
  --detector-checkpoint /path/to/detector.pth \
  --boundary-checkpoint /path/to/boundary.pth \
  --output-dir deploy/models \
  --detector-resolution 1024 \
  --boundary-resolution 704
```

导出的 ONNX 支持动态 batch、固定空间分辨率，并分别生成主检测和边界检测模型。通用
模型也可以通过 `model.export(format="onnx")` 导出；TensorRT 需要额外安装对应依赖，
详见 [docs/learn/export.md](docs/learn/export.md)。

### 7.2 准备交付资产

交付目录位于 `deploy/`，资产准备脚本会检查并复制模型、原型工件和运行时文件：

```bash
# 默认准备 ONNX 交付资产
uv run python deploy/prepare_delivery_assets.py

# 使用 PyTorch 后端时，额外准备 checkpoint、原型工件和精简运行时
uv run python deploy/prepare_delivery_assets.py --include-pytorch-runtime
```

纯 ONNX 运行时不加载 `.pth`、`.pt` 或 ProtoGuidance 原型文件，原型融合和类别数已经固
化在 ONNX 图中。使用 ONNX 后端时，`deploy/app/competition/configs/submission.yaml` 的
`main` 和 `boundary` 角色都应设置为 `backend: onnx`，模型字段只填写
`deploy/models/` 内的文件名。

### 7.3 构建和运行 Docker

```bash
cd deploy
docker build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t rfdetr-shwx:local .
```

使用赛事规定的输入输出参数运行容器：

```bash
mkdir -p test-input test-output
cp /path/to/test-image.jpg test-input/

docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  rfdetr-shwx:local \
  --input /input \
  --output /output
```

程序必须在 `/output/result.json` 生成包含状态、图像尺寸、时间戳和目标列表的结果文件。
容器启动时会检查启用角色的 GPU 后端，ONNX Runtime 必须能发现
`CUDAExecutionProvider`，不能静默回退到 CPU。

### 7.4 交付前检查

宿主机可用以下脚本评估容器结果，并可选生成可视化：

```bash
uv run python src/scripts/eval_deploy_result.py \
  --result deploy/test-output/result.json \
  --labels /path/to/yolo/labels \
  --images /path/to/images \
  --visualize \
  --vis-dir deploy/test-output/viz
```

完整的 ONNX 环境、镜像推送和平台提交步骤见：

- [deploy/README.md](deploy/README.md)
- [deploy/ONNX镜像构建说明.md](deploy/ONNX镜像构建说明.md)
- [比赛镜像构建与提交流程](docs/比赛方案/比赛镜像构建与提交流程.md)

## 8. 代码与文档索引

```text
src/rfdetr/                  # RF-DETR 模型、训练和原型/SSCL 模块
src/scripts/train.py         # 统一训练入口
src/scripts/test.py          # 批量评估入口
src/scripts/predict.py       # 单图/目录推理入口
src/scripts/onnx/            # ONNX 导出工具
configs/experiments/         # 训练、测试和推理 YAML
deploy/                      # Docker 交付运行时和模型资产
docs/改进方案-dinov2-proto/  # 多模态原型设计与机制说明
docs/改进方案-SSCL/          # SSCL、语义头和难例抑制方案
docs/竞赛文档草稿/           # 研究报告方法部分草稿
```

进一步阅读：

- [实验配置说明](configs/experiments/README.md)
- [训练参数与数据格式](docs/learn/train/index.md)
- [多模态原型引导方案](docs/改进方案-dinov2-proto/RF-DETR-DINOv2多模态原型引导方案.md)
- [SSCL 难例原型方案](docs/改进方案-SSCL/RF-DETR-SSCL难例原型改进方案.md)
- [大图切分实现](docs/比赛方案/项目介绍.md)

## 9. 复现注意事项

- 训练、评估和推理的分辨率应与 checkpoint 和 YAML 保持一致，尤其是大图边界分支。
- 训练侧的类别均衡、Logit Adjustment、SSCL 和难例抑制开关，必须与评估侧阈值配置一
  起记录，避免不同实验口径混用。
- 相对路径默认以项目根目录解析；交付 YAML 中的模型路径只能引用镜像内的模型文件名。
- 新实验优先复制现有 YAML 并修改输出目录，保留每次运行生成的 `training_config.json`，
  以便追溯实际生效参数。
