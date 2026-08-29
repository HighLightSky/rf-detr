# PyTorch 两方案构建、测试与评测命令

所有命令均在仓库根目录 `/home/liu/wzt/Ruiyingshizong/rf-detr` 执行。两个方案的构建都使用临时 Docker
上下文：不会修改 `deploy/models/` 或当前 `submission.yaml`，也不会把另一个方案的权重混入镜像。

测试集图像与标签为：

```bash
TEST_IMAGES=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/images/test
TEST_LABELS=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/labels/test
```

评测时会忽略标签中的辅助类别 `25`（`truck`），因为比赛提交 JSON 只输出 0-24 类。

## 方案一：25 类多模态原型，裁切 800，MS/FSC NMS

### 构建镜像

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

VARIANT="output/最终提交五个版本/1-25类-多模态原型-裁切800以下-ms-fsc-NMS"
CONTEXT="$(mktemp -d /tmp/detector-pytorch25.XXXXXX)"

cp deploy/Dockerfile "$CONTEXT/Dockerfile"
cp deploy/environment.yml "$CONTEXT/environment.yml"
cp -a deploy/app "$CONTEXT/app"
mkdir -p "$CONTEXT/models"
cp "$VARIANT/main.pth" "$CONTEXT/models/main.pth"
cp "$VARIANT/boundary.pth" "$CONTEXT/models/boundary.pth"
cp "$VARIANT/proto_guidance_shwx_1024_from120ep.pt" \
   "$CONTEXT/models/proto_guidance_shwx_1024_from120ep.pt"
cp "$VARIANT/submission.yaml" "$CONTEXT/app/competition/configs/submission.yaml"

sudo docker build \
  --pull=false \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t detector-pytorch25:800-ms-fsc-NMS \
  "$CONTEXT"

rm -rf "$CONTEXT"
sudo docker image inspect detector-pytorch25:800-ms-fsc-NMS \
  --format '{{.Architecture}}|{{.Os}}'
```

最后一条输出必须为 `amd64|linux`。

### 镜像可用性测试

此命令在禁网状态加载主模型、边界模型和原型文件，并检查 GPU 可用性：

```bash
sudo docker run --rm \
  --gpus '"device=0"' \
  --network none \
  --entrypoint python \
  detector-pytorch25:800-ms-fsc-NMS \
  -c 'from competition.config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR, load_submission_config; from competition.pipeline import CompetitionPipeline; c = load_submission_config(DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR); p = CompetitionPipeline.from_config(c); p.check_gpu(); print("模型与 CUDA 检查通过")'
```

### 真实测试集预测并归档结果

容器内部仍生成 `result.json`，随后命令会将它改名为带时间戳的文件，避免覆盖之前的结果：

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

TEST_IMAGES=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/images/test
RUN_ID="pytorch25_$(date +%Y%m%d_%H%M%S_%N)"
TEMP_OUTPUT="deploy/test-output/.${RUN_ID}"
RESULT="deploy/test-output/result_${RUN_ID}.json"
mkdir -p "$TEMP_OUTPUT"

sudo docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$TEST_IMAGES:/input:ro" \
  -v "$(pwd)/$TEMP_OUTPUT:/output" \
  detector-pytorch25:800-ms-fsc-NMS \
  --input /input \
  --output /output

mv "$TEMP_OUTPUT/result.json" "$RESULT"
rmdir "$TEMP_OUTPUT"
echo "结果文件: $RESULT"
```

快速验证时，只需将 `TEST_IMAGES` 改为 `deploy/test-input`。

### 评测和可视化

将下方 `RESULT` 改为上一段命令最后输出的结果文件名。可视化保存至 `deploy/test-output/visualization_pytorch25_时间戳/`：

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

RESULT=deploy/test-output/result_pytorch25_替换为实际时间戳.json
TEST_IMAGES=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/images/test
TEST_LABELS=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/labels/test
VIS_DIR="deploy/test-output/visualization_$(basename "$RESULT" .json | sed 's/^result_//')"

UV_CACHE_DIR=/tmp/rfdetr-uv-cache MPLCONFIGDIR=/tmp/rfdetr-matplotlib \
uv run --no-sync python src/scripts/eval_deploy_result.py \
  --result "$RESULT" \
  --labels "$TEST_LABELS" \
  --images "$TEST_IMAGES" \
  --visualize \
  --vis-dir "$VIS_DIR"
```

## 方案二：26 类（truck 辅助类）多模态原型，裁切 800，MS/FSC NMS

### 构建镜像

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

VARIANT="output/最终提交五个版本/2-26类-多模态原型-裁切800以下-ms-fsc-NMS"
CONTEXT="$(mktemp -d /tmp/detector-pytorch26.XXXXXX)"

cp deploy/Dockerfile "$CONTEXT/Dockerfile"
cp deploy/environment.yml "$CONTEXT/environment.yml"
cp -a deploy/app "$CONTEXT/app"
mkdir -p "$CONTEXT/models"
cp "$VARIANT/shwx_proto_hardneg_26cls_truck_1024.pth" \
   "$CONTEXT/models/shwx_proto_hardneg_26cls_truck_1024.pth"
cp "$VARIANT/boundary.pth" "$CONTEXT/models/boundary.pth"
cp "$VARIANT/proto_guidance_shwx_truck_26.pt" \
   "$CONTEXT/models/proto_guidance_shwx_truck_26.pt"
cp "$VARIANT/submission.yaml" "$CONTEXT/app/competition/configs/submission.yaml"

sudo docker build \
  --pull=false \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t detector-pytorch26:800-ms-fsc-NMS \
  "$CONTEXT"

rm -rf "$CONTEXT"
sudo docker image inspect detector-pytorch26:800-ms-fsc-NMS \
  --format '{{.Architecture}}|{{.Os}}'
```

最后一条输出必须为 `amd64|linux`。

### 镜像可用性测试

```bash
sudo docker run --rm \
  --gpus '"device=0"' \
  --network none \
  --entrypoint python \
  detector-pytorch26:800-ms-fsc-NMS \
  -c 'from competition.config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR, load_submission_config; from competition.pipeline import CompetitionPipeline; c = load_submission_config(DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR); p = CompetitionPipeline.from_config(c); p.check_gpu(); print("模型与 CUDA 检查通过")'
```

### 真实测试集预测并归档结果

类别 25 只在模型内部作为辅助类使用，提交输出会自动过滤该类。

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

TEST_IMAGES=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/images/test
RUN_ID="pytorch26_$(date +%Y%m%d_%H%M%S_%N)"
TEMP_OUTPUT="deploy/test-output/.${RUN_ID}"
RESULT="deploy/test-output/result_${RUN_ID}.json"
mkdir -p "$TEMP_OUTPUT"

sudo docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$TEST_IMAGES:/input:ro" \
  -v "$(pwd)/$TEMP_OUTPUT:/output" \
  detector-pytorch26:800-ms-fsc-NMS \
  --input /input \
  --output /output

mv "$TEMP_OUTPUT/result.json" "$RESULT"
rmdir "$TEMP_OUTPUT"
echo "结果文件: $RESULT"
```

快速验证时，只需将 `TEST_IMAGES` 改为 `deploy/test-input`。

### 评测和可视化

将下方 `RESULT` 改为上一段命令最后输出的结果文件名。可视化保存至 `deploy/test-output/visualization_pytorch26_时间戳/`：

```bash
cd /home/liu/wzt/Ruiyingshizong/rf-detr

RESULT=deploy/test-output/result_pytorch26_替换为实际时间戳.json
TEST_IMAGES=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/images/test
TEST_LABELS=/home/liu/datasets/SHWX-FINAL-no-FSC-expand-truck/labels/test
VIS_DIR="deploy/test-output/visualization_$(basename "$RESULT" .json | sed 's/^result_//')"

UV_CACHE_DIR=/tmp/rfdetr-uv-cache MPLCONFIGDIR=/tmp/rfdetr-matplotlib \
uv run --no-sync python src/scripts/eval_deploy_result.py \
  --result "$RESULT" \
  --labels "$TEST_LABELS" \
  --images "$TEST_IMAGES" \
  --visualize \
  --vis-dir "$VIS_DIR"
```
