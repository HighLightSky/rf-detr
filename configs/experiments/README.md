# 实验配置目录（configs/experiments）

统一实验入口：`src/scripts/` 下的 `train.py` / `test.py` / `predict.py` 三个模板
读取本目录的 yaml 配置运行实验，替代了原有的散落脚本（旧脚本已删除，
git 历史可回溯）。

## 快速开始

```bash
# 训练（等价旧的 train_sscl_class_balance.py E1 配方）
python src/scripts/train.py -c configs/experiments/train_sscl_class_balance_E1.yaml

# 测试评估（比赛口径）
python src/scripts/test.py -c configs/experiments/test_shwx.yaml

# 推理
python src/scripts/predict.py -c configs/experiments/predict_shwx.yaml --image /path/to/img.jpg
```

## 目录结构

```
configs/experiments/
├── templates/                     # 全注释模板（新实验从这里复制）
│   ├── train_template.yaml
│   ├── test_template.yaml
│   └── predict_template.yaml
├── train_baseline.yaml            # nano 基线（0813 重标注版，分辨率 704 防漏检小目标）
├── train_baseline_medium.yaml     # medium 基线（与 nano 其余配置一致，分辨率 640）
├── train_sscl_0807.yaml           # 等价旧 train_sscl.py（0807 纯原型）
├── train_sscl_all.yaml            # 等价旧 train_sscl_all.py（全类别）
├── train_sscl_strong_A/B/C.yaml   # 等价旧 train_sscl_strong.py（强作用力/判别矩阵/组合）
├── train_sscl_hardneg_k3.yaml     # 等价旧 train_sscl_hardneg.py（k=5 用 --set）
├── train_sscl_class_balance_E1.yaml  # 等价旧 train_sscl_class_balance.py（P0/P1）
├── train_sscl_multproto_v1.yaml   # 多 slot 原型 v1（HM/LQS/QHS/MS）
├── train_sscl_multproto_hardneg_suppress_v1.yaml  # 多 slot + 难负样本直接抑制
├── train_lora.yaml                # 等价旧 train_LoRA.py
├── test_shwx.yaml / test_dior.yaml
├── test_shwx_large.yaml           # SHWX-large 数据集（672 小图 + 100 张 concat 大图）评测：
│                                  #   大图滑窗切分（tile_overlap） + 按类别 NMS 去重
└── predict_shwx.yaml
```

## yaml Schema（自定义三段式）

```yaml
_template:                          # 模板专用键（train.py 消费，不会传给 model.train）
  class_counts: auto                # auto=训练前统计类别数并自动注入 P0 的 counts 路径

model:                              # 模型构造 kwargs（RFDETR 变体类）
  variant: medium                   # 专用键：nano/small/medium/large/large_deprecated
  num_classes: 25
  resolution: 640                   # 缺省用变体默认（medium=576，旧实验脚本显式 640）
  gradient_checkpointing: true
  pretrain_weights: output/xxx/checkpoint_best_total.pth   # 不写 = 官方发布权重

train:                              # ★ 100% 透传为 model.train(**kwargs)，零映射表
  dataset_dir: /home/liu/wzt/datasets/SHWX-dataset-dict
  dataset_file: yolo
  output_dir: output/xxxx-SHWX-实验名
  epochs: 6
  ...                               # 全部 TrainConfig 字段（src/rfdetr/config.py）

test:                               # test.py 消费
  dataset: shwx                     # eval_lib.DATASET_CONFIGS 内置名（shwx/dior）；省略默认 shwx
  dataset_dir: /home/liu/wzt/datasets/SHWX-dataset-dict   # 数据集根目录（同训练侧 dataset_dir 模式）
                                    # 省略=内置默认；重新标注后数据路径变化时在此覆盖，类别语义仍由 dataset 提供
  resolution: 704                   # 推理输入分辨率（须与训练分辨率一致，如 nano 704 训练时填 704）；
                                    # 省略=使用 checkpoint 记录的分辨率
  checkpoint: output/xxx/checkpoint_best_total.pth
  conf_threshold: 0.25
  class_conf_thresholds: {}         # 可整段贴入 calibrate_thresholds 产物
  device: cuda:0
  batch_size: 32
  num_workers: 12
  output_dir: output/xxx-eval       # 测试输出目录（报告/混淆矩阵/FP·FN）；省略=数据集内置 exp_output_dir
  la_bias:                          # 推理侧 Logit Adjustment；省略=关闭
    counts_path: output/xxx/class_counts.json
    k: 1.0
    tau: 0.1
    clip: 1.0
  save_fp_fn: true
  save_yolo_preds: false
  tile_overlap: 256                # 大图滑窗重叠像素；0=关闭切分（>分辨率的大图走整图缩放路径）
  tile_nms_iou: 0.5                # 切分合并后按类别 NMS 的 IoU 阈值
  tile_batch_size: 16              # 切分路径 tile 批量；省略=沿用 batch_size
  viz_large_count: 3               # 随机可视化 N 张大图切割结果（左 GT / 右 Predict，固定种子）
                                   # 输出到 exp_output_dir/large_viz/；0=关闭

predict:                            # predict.py 消费
  checkpoint: output/xxx/checkpoint_best_total.pth
  conf: 0.25
  output_dir: output/xxx/predict
  image: null                       # 通常由 --image 提供
  class_names: shwx                 # 内置名 shwx/dior，或 {label: 名称} 字典
```

### 透传规则

1. `train:` 段键名原封不动作为 `model.train(**kwargs)` 的参数；新增 TrainConfig
    字段（`src/rfdetr/config.py`）直接加键即可，**无需改任何代码**。
2. `TrainConfig` 的 `extra="forbid"` 天然校验拼写错误——yaml 键写错会在启动时报出
    具体字段名。
3. 相对路径一律以项目根解析（与旧脚本 `project_root / X` 行为一致），绝对路径原样。
4. `--set` 支持点路径标量覆盖，变体实验无需复制 yaml：
    ```bash
    # E2 变体（P0 增强）
    python src/scripts/train.py -c configs/experiments/train_sscl_class_balance_E1.yaml \
        --set train.class_balance_enabled=true --set train.class_balance_beta=0.5 \
        --set train.class_balance_max_weight=5.0 --set train.output_dir=output/xxxx-E2
    # hardneg k=5
    python src/scripts/train.py -c configs/experiments/train_sscl_hardneg_k3.yaml \
        --set train.sscl_hard_neg_topk=5 --set train.output_dir=output/xxxx-HardNeg-k5-iou03
    # 多 slot + 难负样本直接抑制
    python src/scripts/train.py -c configs/experiments/train_sscl_multproto_hardneg_suppress_v1.yaml
    ```

## 旧脚本 → yaml 迁移映射表

| 旧脚本（已删除）            | 等价配置                         | 差异说明                              |
| --------------------------- | -------------------------------- | ------------------------------------- |
| train.py                    | train_baseline.yaml              | MODEL=nano、freeze_encoder=false      |
| train_sscl.py               | train_sscl_0807.yaml             | 0807 纯原型（instance_pos=false）     |
| train_sscl_all.py           | train_sscl_all.yaml              | 全类别，start_epoch=30                |
| train_sscl_strong.py        | train_sscl_strong\_{A,B,C}.yaml  | 原 env 变量改 yaml 字段               |
| train_sscl_hardneg.py       | train_sscl_hardneg_k3.yaml       | k 用 --set 切换                       |
| train_sscl_class_balance.py | train_sscl_class_balance_E1.yaml | E2/E3 用 --set                        |
| —                           | train_sscl_multproto_v1.yaml     | 新增多 slot 原型 v1，统一入口直接运行 |
| train_LoRA.py               | train_lora.yaml                  | freeze_encoder+backbone_lora          |
| test.py                     | test_shwx.yaml / test_dior.yaml  | 含推理侧 LA bias 配置                 |

## 注意事项

0. **大图滑窗切分（里程碑 1 基线）**：`tile_overlap > 0` 时，max(w,h) 超过模型
    分辨率（1024）的大图按 `tile_size=模型分辨率` 滑窗切块推理，坐标映射回全图后
    按类别 NMS（`tile_nms_iou`）去重；小图仍走整图批量路径，两路结果合并评测。
    关闭切分（`tile_overlap=0`）时大图走旧整图缩放路径，用于回归对照
    （`--set test.tile_overlap=0`）。已知缺陷（后续里程碑优化）：跨块目标可能
    产生残缺框、NMS 可能误合并重叠区相邻真实目标。
1. **两套"类均衡"易混淆**：
    - `class_balanced_sampling/class_balanced_*`：平方根频率过采样（数据采样层）；
    - `class_balance_*`：P0 正样本类均衡 IA-BCE（损失层）。
2. **`large` 有两代**：`large` = 新版 RFDETRLarge（分辨率 704），
    `large_deprecated` = 旧版（560）。旧 train.py 用的是 deprecated 版。
3. **训练/推理侧 LA bias 一致性**：P1 的训练参数（`logit_adjustment_*` /
    `class_balance_*`）写在 train yaml，推理侧对应参数写在 test yaml 的
    `la_bias:` 段——改了训练配方记得同步测试配置（不写 la_bias 则推理侧不生效）。
4. **`aug_config` 只接受预设名**：`AERIAL`/`CONSERVATIVE`/`AGGRESSIVE`/
    `INDUSTRIAL`/`none`（AUG\_\* 是嵌套 dict，无法直接写进 yaml）。
5. **cron 清理**：旧 `experiments_tmp/monitor_train.sh` 的 crontab 引用已失效，
    需手动清理（`crontab -e`）。
6. **配置溯源**：每个训练输出目录的 `training_config.json` 记录全量生效配置
    （157 键），是复现实验的权威来源。

## 常用工具

```bash
# 只打印展开后的 kwargs 不训练（等价性验证/排查配置）
python src/scripts/train.py -c configs/experiments/xxx.yaml --dump-kwargs

# 逐类阈值重标定（产物可贴入 test yaml 的 class_conf_thresholds）
python src/scripts/calibrate_thresholds.py output/xxx/checkpoint_best_total.pth \
    --bias-json output/xxx/class_counts.json --bias-k 1.0
```
