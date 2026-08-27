目标检测算法 Docker 封装说明
一、准备目录
detector-docker-delivery/
├── Dockerfile
├── environment.yml
├── app/
│   ├── main.py
│   ├── detector.py       # 可选：模型加载和推理
│   ├── preprocess.py     # 可选：预处理
│   ├── postprocess.py    # 可选：后处理
│   └── 其他代码文件      # 可继续增加
└── models/
    └── 你的模型文件

文件	要求
Dockerfile	直接使用提供的文件，一般不需要修改
environment.yml	由 Conda 官方命令从 Linux 环境自动生成
app/	放入口文件及全部推理、预处理、后处理代码
models/	放入 .pth、.pt、.onnx、.engine 等模型

Dockerfile 会把整个 app/ 复制到镜像的 /app/。因此，除 main.py 外的 Python 文件、配置文件和本地代码都放在 app/ 下；main.py 中按实际文件名导入，例如 from detector import Detector。模型统一放在 models/ 下，代码中使用 /app/models/访问。不要写 C:\、D:\ 等 Windows 路径。
二、自动生成 environment.yml
没有 Linux 环境怎么办
environment.yml 必须在 Linux x86_64 环境中生成，不能直接使用 Windows 导出的 Conda 环境。没有 Linux 电脑时，可先准备以下任一种环境：
推荐：在 Windows 中安装 WSL2 和 Ubuntu，并在 Docker Desktop 中启用 WSL2 后端。该方式可以使用 Linux 容器，也便于进行 GPU 测试。
虚拟机：使用 VMware 或 VirtualBox 安装 Ubuntu 22.04/24.04 LTS 的 x86_64 版本。普通虚拟机通常不能直接使用 NVIDIA GPU，只适合整理代码、安装依赖和构建镜像；GPU 推理还需要在支持 GPU 的 Linux、WSL2 或远程 GPU 主机上测试。
其他方式：使用实体 Linux x86_64 电脑、Linux GPU 服务器或云 GPU 主机。
无论选择哪种方式，都要在 Linux x86_64 环境中安装代码实际使用的依赖，再执行本节的 conda env export。只有在支持 GPU 的环境中完成 docker run --gpus 测试，才能确认镜像可以通过最终评测。
不要手写依赖列表。先在 Linux x86_64 环境中安装并测试本队伍的全部运行依赖，再在同一个环境中执行下面两条命令：
conda activate 你的环境名
conda env export --no-builds \
  | sed '/^prefix:/d' \
  > environment.yml

这是 Conda 官方导出命令，生成的 environment.yml 会包含当前环境中的 Conda 包和 pip 包，并删除本机路径。每次修改依赖后，都要重新激活正确环境并重新执行命令。
Windows 队伍必须注意  Windows Conda 环境不能直接生成可用于 Linux 的依赖文件。请队伍自行准备 Linux x86_64 构建环境，在该环境中装好并跑通代码后再执行脚本。若提示 conda 未找到，说明当前环境还不能生成 environment.yml。

生成后不要手工改包名、版本号或 Windows 路径。确认 environment.yml 已更新，再执行下一节的 docker build。
不要直接复制 Windows 环境  environment.yml 必须对应 Linux x86_64。不要写入 win-64、pywin32、C:\ 路径或 Windows 专用库。

三、填写 app/main.py 和其他代码
main.py 是容器启动入口。请保留参数、输入读取、逐图推理、时间戳和 result.json 写出逻辑；把本队伍的模型加载和推理接入 Detector。下面是入口文件的完整结构，模型文件和具体推理代码必须替换为本队伍代码：
import argparse
import json
import time
from pathlib import Path
 
from PIL import Image
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MODEL_DIR = Path("/app/models")
 
 
class Detector:
    def __init__(self):
        # 1. 在这里加载模型。
        # 2. 模型文件路径使用 /app/models/，不要使用 Windows 路径。
        # 3. 模型格式不限：pth、pt、onnx、engine 或其他格式均可。
        # 4. 必须让模型使用 GPU，例如 PyTorch 的 cuda:0。
        # 5. 预处理、标签表等固定资源也可以在这里加载。
        self.model = load_your_model(MODEL_DIR / "你的模型文件")
 
    def predict(self, image):
        # image 是一张已经读取到内存的 PIL RGB 图片。
        # 在这里完成：预处理 -> GPU 推理 -> 后处理。
        # 返回下面格式的列表；没有检测目标时返回 []。
        objects = run_your_gpu_inference(self.model, image)
        return objects
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    detector = Detector()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # 只读取 /input 第一层的图片，不递归读取子目录。
    paths = sorted([
        p for p in Path(args.input).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])
 
    # 先读入内存，避免推理过程中持续读取输入文件。
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append((path, image.convert("RGB").copy()))
 
    final_results = []
    for path, image in images:
        objects = detector.predict(image)
        # 必须紧接本张图片推理结束后记录 Unix 毫秒时间戳。
        run_end_timestamp = time.time_ns() // 1_000_000
        final_results.append({
            "image_id": path.stem,
            "file_name": path.name,
            "width": image.width,
            "height": image.height,
            "run_end_timestamp": run_end_timestamp,
            "objects": objects
        })
 
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"status": "success", "images": final_results},
                  f, ensure_ascii=False, indent=2)
 
 
if __name__ == "__main__":
    main()

项目模板中的 main.py 还包含启动时的 GPU 检查。基于模板修改时，请保留 check_gpu() 函数和 main() 中的 check_gpu() 调用；如果按自己的框架重写，也必须在启动时确认 GPU 可用，并让实际推理使用 GPU。
如果已有多个代码文件，不要全部塞进 main.py。将它们放到 app/，例如 app/detector.py、app/preprocess.py、app/postprocess.py 或 app/utils/，然后在 main.py 中导入。Dockerfile 会复制整个 app/目录，因此这些文件会随镜像一起封装。
Detector.__init__() 只执行一次，适合加载模型；predict(image) 对每张图片调用一次。predict() 返回的每个目标必须包含：
字段	格式
category_id	整数
category_name	字符串
score	0 到 1 的数字
bbox	[x1, y1, x2, y2]，原图像素坐标

bbox 必须是原图像素坐标 [x1, y1, x2, y2]；run_end_timestamp 必须是 Unix 毫秒时间戳，且放在每张图片的结果中。输入图片格式只支持 .jpg、.jpeg、.png、.bmp。
{
  "status": "success",
  "images": [
    {
      "image_id": "000001",
      "file_name": "000001.jpg",
      "width": 1920,
      "height": 1080,
      "run_end_timestamp": 1723968000123,
      "objects": [
        {
          "category_id": 1,
          "category_name": "person",
          "score": 0.9321,
          "bbox": [100.5, 200.0, 500.0, 700.0]
        }
      ]
    }
  ]
}

四、构建镜像
确认模型、app/ 下的全部代码和自动生成的 environment.yml 都已放好后，在 Linux x86_64/WSL 的 Bash 中执行：
cd /mnt/e/sqy/tzb/detector-docker-delivery
 
docker build \
  --platform linux/amd64 \
  -t detector-team001:1.0 .

注意：Bash 换行使用一个反斜杠 \ 。也可以把 docker build 命令写成一行。
docker image inspect detector-team001:1.0 \
  --format '{{.Architecture}}|{{.Os}}' 

看到 amd64|linux 才符合目标平台。构建成功后，Python、Conda/Pip 依赖、代码和模型已经在镜像中。
五、本地运行验证
mkdir -p test-input test-output
cp 你的测试图片.jpg test-input/

把至少一张真实测试图片复制到 test-input/。测试图片应使用赛事允许的 jpg、jpeg、png 或 bmp 格式。
docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  detector-team001:1.0 \
  --input /input \
  --output /output

查看结果：
cat test-output/result.json

必须生成 test-output/result.json，且检测结果应来自本队伍真实模型。重点检查 status、每张图片的 file_name/width/height/run_end_timestamp，以及 objects 中的类别、置信度和原图坐标框。
如果算法支持多卡，把 --gpus 改为例如：
--gpus '"device=0,1"'

六、下一步：提交镜像
封装完成后继续阅读  确认镜像已经构建成功并完成本地验证后，不要继续修改本说明。请打开下面的《赛事评测管理系统参赛队伍使用手册》，从“使用前准备”开始，完成登录、获取 ACR 命令、docker login、docker tag、docker push 和提交评测。

E:\sqy\tzb\外协\赛事评测管理系统参赛队伍使用手册.docx

提交时以评测系统页面显示的镜像地址和 tag 为准，不要自行猜测仓库地址或 tag。
七、提交前检查
Dockerfile、environment.yml、app/ 全部代码和模型文件都在项目目录中。
environment.yml 是在 Linux x86_64 环境中自动生成的。
docker build --platform linux/amd64 构建成功。
容器能使用至少一块 GPU，并生成 /output/result.json。
已完成本地验证，再进入赛事评测管理系统提交。