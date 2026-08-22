# 0822 proto + SSCL + HN 消融实验

本目录是一组**增量可加的消融实验**，用于厘清 RF-DETR 在多模态原型引导（ProtoGuidance）、SSCL 语义加权对比学习、难负样本抑制（Hard Negative）三套机制下各自的增益，并给出「完整方案 + 后处理插件」的最终口径。

运行顺序见 [运行顺序](#运行顺序一个一个跑)。每个实验都从**同一份基线权重**、**独立的输出目录**启动，彼此不依赖，所以先后顺序不影响结果（先后跑只是为了便于逐个核对日志与评估）。

## 一、消融矩阵

所有实验共用：`medium` 变体、`num_classes=25`、`resolution=1024`、`epochs=10`、`batch_size=32`、`grad_accum_steps=2`、`lr=1e-5`、`aug_config=AERIAL`、`use_ema=true`、`sscl_freeze_strategy=none`（全量微调，避免与其他分支冻结范围混杂）。类别均衡过采样 `class_balanced_sampling=true`（`class_ids=[0,1,24]`）。

| 实验 | 文件 | 启用内容 | 目的 |
| ---- | ---- | -------- | ---- |
| **A** | `A_仅基座RFDETR.yaml` | 无（原型/SSCL/HN 全关） | 纯 RF-DETR 基线，作为参照。 |
| **B** | `B_原型特征选择增强.yaml` | 原型 **position**（encoder query selection） | 验证「原型只用做召回入口」的增益。 |
| **C** | `C_原型语义信息增强.yaml` | 原型 **content**（decoder query 内容残差） | 验证「原型只用做解码器内容注入」的增益。 |
| **D** | `D_原型前向注入无损失.yaml` | 原型 **position + content**，无任何原型/SSCL/HN 损失 | **共同前向底座**：后面的 E/F/G/H 都在它基础上各自加一个损失分量。 |
| **E** | `E_原型前向加稠密对齐.yaml` | D + 原型 **dense** 语义对齐（encoder 前景/背景头监督） | 验证稠密对齐损失的贡献。 |
| **F** | `F_原型前向加辅助分类损失.yaml` | D + 原型 **aux** 辅助分类损失 | 验证 matched-foreground 原型分类监督的贡献。 |
| **G** | `G_原型前向加SSCL.yaml` | D + **SSCL**（instance-to-instance 语义加权对比） | 验证语义加权（语义矩阵）对比学习的贡献。 |
| **H** | `H_原型前向加难负样本抑制.yaml` | D + **难负样本抑制**（SSCL 仅作装配入口，`sscl_lambda=0`，不算 SSCL 损失） | 验证难负样本直接抑制的贡献。 |
| **I** | `I_完整方案加后处理插件.yaml` | D + dense + aux + SSCL + 难负样本（全开），评估时启用 FSC 后处理插件 | 完整方案 + 训练后评价。 |

> 关系说明：A 是纯基线；B / C 是「原型只用一半」的两个对照；D 是「原型完整前向但一个损失都不加」的共同底座；E / F / G / H 各自在 D 上**只加一个**损失，用来孤立单个机制；I 是全部机制叠加的完整方案。I 的 `test:` 段额外带了 `reason_plugin`（FSC 后处理插件），训练完即可直接评估。

## 二、公共前置条件

下面已提前核对存在，缺失时训练/评估会报错：

- 起始权重：`output/0815-SHWX-rfdetr-medium-baseline-精细标注-1024/checkpoint_best_total.pth`
- 原型产物：`data/proto_guidance_shwx_1024_from120ep.pt`
- 语义矩阵：`data/semantic_matrix_shwx.pt`
- 训练集：`/home/liu/wzt/datasets/SHWX-dataset-dict-redo`（yolo 布局）
- 测试集：`/home/liu/wzt/datasets/SHWX-dataset-dict-redo-full_test`
- 后处理插件（仅 I 用到）：`output/0818reasoning-candidate/reason_plugin_fsc.pth`

各配置里的 `pretrain_weights`、`dataset_dir` 均为相对项目根或绝对路径，`/` 开头的绝对路径原样使用，其余相对路径以项目根解析。

## 三、运行顺序（一个一个跑）

### 实验 A：纯基座 RF-DETR

```bash
# 训练
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/A_仅基座RFDETR.yaml"

# 评估（比赛口径，SHWX 大图切分管线）
python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/A-baseline/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/A-baseline-eval
```

### 实验 B：原型特征选择增强（position 召回入口）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/B_原型特征选择增强.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/B-position/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/B-position-eval
```

### 实验 C：原型语义信息增强（content 内容注入）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/C_原型语义信息增强.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/C-content/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/C-content-eval
```

### 实验 D：原型前向注入（无损失，共同底座）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/D_原型前向注入无损失.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/D-forward-only/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/D-forward-only-eval
```

### 实验 E：原型前向 + 稠密对齐（dense）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/E_原型前向加稠密对齐.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/E-dense/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/E-dense-eval
```

### 实验 F：原型前向 + 辅助分类损失（aux）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/F_原型前向加辅助分类损失.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/F-proto-aux/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/F-proto-aux-eval
```

### 实验 G：原型前向 + SSCL

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/G_原型前向加SSCL.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/G-sscl/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/G-sscl-eval
```

### 实验 H：原型前向 + 难负样本抑制（HN）

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/H_原型前向加难负样本抑制.yaml"

python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml \
  --set test.checkpoint=output/0822-proto-sscl-hn-ablation/H-hard-neg/checkpoint_best_total.pth \
  --set test.output_dir=output/0822-proto-sscl-hn-ablation/H-hard-neg-eval
```

### 实验 I：完整方案 + 后处理插件

训练命令与其他实验一致；评估**直接用本文件自带的 `test:` 段**（已经写好了 `reason_plugin` 与输出目录），无需再覆盖：

```bash
# 训练
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/I_完整方案加后处理插件.yaml"

# 评估（自带 FSC 后处理插件）
python src/scripts/test.py -c "configs/experiments/0822proto+SSCL+HN消融/I_完整方案加后处理插件.yaml"
```

## 四、关于 best_metric（用 F1 选最佳轮）

这批配置默认走 `best_metric: map`（普通检测看 `val/mAP_50_95`，带 regular/EMA 双轨选权）。若希望**用 F1 挑最佳权重**（例如看重舰船细分类的混淆控制），给训练命令追加 `--set train.best_metric=f1` 即可，例如：

```bash
python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/D_原型前向注入无损失.yaml" \
  --set train.best_metric=f1
```

注意：`best_metric=f1` 由 regular 模型的 `val/F1` 选权，会关闭 EMA 最佳权重跟踪，且**不能**与 `eval_ema_only=true` 同时使用。所有实验共享同一份 `train_template.yaml` 语义，该字段也已在模板中注释。

## 五、实用小提醒

- **检查配置展开**：任何实验训练前可用 `--dump-kwargs` 只打印展开后的 model/train kwargs，不真正训练：
  ```bash
  python src/scripts/train.py -c "configs/experiments/0822proto+SSCL+HN消融/A_仅基座RFDETR.yaml" --dump-kwargs
  ```
- **输出目录**：训练产物与 `training_config.json` 都在 `output/0822-proto-sscl-hn-ablation/<实验名>/`，评估结果在加 `-eval` 后缀的目录。
- **最佳权重**：每次训练会写 `checkpoint_best_total.pth`（最终用于推理）/ `checkpoint_best_regular.pth`（普通模型）/ `checkpoint_best_ema.pth`（EMA 模型）。上面的评估命令统一取 `checkpoint_best_total.pth`。
- **逐类阈值**：如需逐类重标定，运行
  ```bash
  python src/scripts/evaluation/calibrate_thresholds.py output/0822-proto-sscl-hn-ablation/<实验名>/checkpoint_best_total.pth \
      --bias-json output/0822-proto-sscl-hn-ablation/<实验名>/class_counts.json --bias-k 1.0
  ```
  产物可贴回 `test_shwx.yaml` 的 `class_conf_thresholds`。
