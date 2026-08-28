# 比赛 Docker 交付目录

该目录是独立于训练 `uv` 环境的比赛运行时。根目录保留给赛事要求的
`Dockerfile`、`environment.yml`、`app/` 与 `models/`；当前不创建 Dockerfile，必须使用主办方提供的原文件。

默认 `app/competition/configs/submission.yaml` 使用实验二导出的两个 ONNX 模型：主检测为
`shwx_detector_1024.onnx`，大图边界检测为 `large_cut_boundary_704.onnx`。它同时开启民船 NMS 和
发射车一级候选 NMS。通过替换该 YAML 为 `configs/variants/` 下的变体，可以为 `main`、`boundary`
角色分别选择 `onnx` 或 `pytorch` 后端。

先装配默认 ONNX 资产：

```bash
python deploy/prepare_delivery_assets.py
```

PyTorch 变体额外需要 checkpoint、ProtoGuidance 工件和精简后的 `rfdetr` 运行时：

```bash
python deploy/prepare_delivery_assets.py --include-pytorch-runtime
```

脚本会复用内容一致的已有资产，但不会覆盖同名且内容不同的文件。切换版本前应在核实文件后
新建交付目录或手动处置旧的 `models/` 内容。

在实际 Linux x86_64、可用 NVIDIA GPU 的 Docker 运行环境验证后，从该环境导出依赖：

```bash
conda env export --no-builds | sed '/^prefix:/d' > deploy/environment.yml
```

运行环境至少需要 Python、PyTorch CUDA、`onnxruntime-gpu`、NumPy、OpenCV、Pillow 和 PyYAML。`main.py`
只接收赛事定义的 `--input`、`--output`，输出 `result.json`；运行开始时会强制检查每个启用角色的 GPU。
