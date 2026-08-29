# ONNX 镜像构建说明

本文用于将比赛交付目录切换为两个 ONNX Runtime GPU 模型：一个主检测模型和一个大图边界模型。
所有命令默认在仓库根目录执行；Docker 构建命令除外。

## 运行方式

ONNX 版本不加载 `.pth`、`.pt` 或 ProtoGuidance 原型工件。模型、类别数和原型融合结果都已经固化在导出的
`.onnx` 中。运行时仍强制使用 `CUDAExecutionProvider`，不会允许 CPU 回退。

当前可用的基础 ONNX 权重为：

| 角色 | 源文件 | 镜像内文件名 |
| --- | --- | --- |
| 主检测 | `output/0827-0825基线方案/2基线+多模态原型+SSCL+难例抑制/onnx/shwx_detector_1024.onnx` | `shwx_detector_1024.onnx` |
| 大图边界 | `output/0827-0825基线方案/2基线+多模态原型+SSCL+难例抑制/onnx/large_cut_boundary_704.onnx` | `large_cut_boundary_704.onnx` |

若要换用其他主检测 ONNX（例如 `output/0828models_for_submission/onnx/` 中的文件），只替换主模型即可，
但 YAML 中的 `main.model` 必须改为实际复制后的文件名。边界 ONNX 必须保持为与大图裁切流程匹配的边界模型。

## 一、更新 submission.yaml

先使用纯 ONNX 模板覆盖当前提交配置：

```bash
cp deploy/app/competition/configs/variants/onnx_main_onnx_boundary.yaml \
   deploy/app/competition/configs/submission.yaml
```

模板的关键部分如下。两个角色均为 `backend: onnx`，并且**不要**填写 `proto_guidance_artifact`：

```yaml
roles:
  main:
    backend: onnx
    model: shwx_detector_1024.onnx
    resolution: 1024
  boundary:
    backend: onnx
    model: large_cut_boundary_704.onnx
    resolution: 704
```

`model` 只能是文件名。程序将其严格解析到 `/app/models/<文件名>`；不能填写宿主机路径、相对目录或 URL。
如果主模型是 26 类版本，类别编号 25 的辅助类仍会被现有 25 类后处理过滤，不会出现在 `result.json`。

## 二、放置 ONNX 权重

`deploy/models/` 是 Docker 构建上下文中的模型目录。切换到纯 ONNX 前，移走旧的 PyTorch 模型和原型文件，
只保留本次 YAML 引用的两个 `.onnx` 文件：

```bash
mkdir -p /tmp/deploy-models-backup
find deploy/models -maxdepth 1 -type f \( -name '*.pth' -o -name '*.pt' \) \
  -exec mv -t /tmp/deploy-models-backup {} +
cp "output/0827-0825基线方案/2基线+多模态原型+SSCL+难例抑制/onnx/shwx_detector_1024.onnx" \
   deploy/models/shwx_detector_1024.onnx
cp "output/0827-0825基线方案/2基线+多模态原型+SSCL+难例抑制/onnx/large_cut_boundary_704.onnx" \
   deploy/models/large_cut_boundary_704.onnx
chmod a+r deploy/models/*.onnx
```

不要通过 `-v` 挂载宿主机模型覆盖 `/app/models`。Dockerfile 在镜像构建时会统一设置 `/app/models` 可读；
这个机制可避免平台以非 root 用户启动时出现 `PermissionError`。

构建前可用以下命令校验 YAML 引用的文件都存在：

```bash
PYTHONPATH=deploy/app python -c '
from pathlib import Path
from competition.config import load_submission_config
config = load_submission_config(
    Path("deploy/app/competition/configs/submission.yaml"),
    Path("deploy/models"),
)
print([(name, str(role.model_path)) for name, role in config.detector.roles.items()])
'
```

## 三、更新 ONNX 运行环境

ONNX 后端至少需要 `numpy`、`pillow`、`pyyaml`、`opencv-python-headless` 和 `onnxruntime-gpu`。其中
`onnxruntime-gpu` 必须使用 CUDA 12 构建，且版本不低于 1.21；当前代码会调用 `preload_dlls()` 并检查
`CUDAExecutionProvider`，以阻止静默 CPU 回退。纯 ONNX 环境还需通过该包的 `cuda,cudnn` extra 安装
CUDA/cuDNN 动态库，不能假设 PyTorch 会替它提供这些库。

不要直接手写或在现有 PyTorch `environment.yml` 上删包。应在 Linux x86_64 且有 NVIDIA GPU 的环境中建立
纯 ONNX 环境，安装并跑通后再导出。例如：

```bash
conda create -y -n shwx-onnx python=3.11 pip
conda activate shwx-onnx
pip install numpy pillow pyyaml opencv-python-headless
pip install "onnxruntime-gpu[cuda,cudnn]>=1.21" \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
conda env export --no-builds | sed '/^prefix:/d' > deploy/environment.yml
```

`onnxruntime`（CPU 包）与 `onnxruntime-gpu` 不应同时保留。导出前先确认 CUDA provider：

```bash
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'
```

若此命令没有 `CUDAExecutionProvider`，不要构建镜像。应先重新安装 CUDA 12 的 `onnxruntime-gpu`，并在需要时
重启 Python 进程。修改 `environment.yml` 后必须重新构建，因为依赖层已变化。

## 四、构建和验证

在 `deploy/` 目录构建。平台需要单一 `linux/amd64` manifest，因此保留三个 build 参数：

```bash
cd deploy
sudo docker build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t detector-onnx:baseline .

sudo docker image inspect detector-onnx:baseline \
  --format '{{.Architecture}}|{{.Os}}'
```

检查输出必须为 `amd64|linux`。接着先检查镜像内的模型权限和 ONNX CUDA provider：

```bash
sudo docker run --rm --gpus '"device=0"' --network none \
  --entrypoint python detector-onnx:baseline \
  -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()'

sudo docker run --rm --entrypoint sh detector-onnx:baseline \
  -c 'stat -c "%a %n" /app/models/*'
```

最后执行与赛事一致的禁网 GPU 推理。镜像名之后的 `--input` 和 `--output` 是必填参数：

```bash
mkdir -p test-input test-output
sudo docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  detector-onnx:baseline \
  --input /input \
  --output /output

cat test-output/result.json
```

确认每张测试图都出现在 `result.json`，且模型初始化日志没有 CPU provider 或依赖加载错误后，再推送到赛事指定仓库。
