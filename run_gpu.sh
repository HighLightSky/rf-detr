#!/usr/bin/env bash
# 包装 `uv run`：为 onnxruntime-gpu 的 CUDA EP 提供 torch 打包的 CUDA 13 / cuDNN 9 运行库。
#
# 系统无独立 CUDA toolkit，onnxruntime-gpu 需要 libcublasLt.so.13 / libcudnn.so.9，
# 而 torch 把这些库放在 venv 的 nvidia/*/lib 下。不设置 LD_LIBRARY_PATH 时，
# onnxruntime 会静默回退到 CPU EP，导致 ONNX 推理远慢于 PyTorch GPU。
#
# 用法（等价于原来的 `uv run --no-sync ...`，自动补上 LD_LIBRARY_PATH）：
#   ./run_gpu.sh python src/scripts/test.py -c configs/experiments/train_tests/test_shwx.yaml
set -euo pipefail

# 仓库根目录（相对本脚本位置，可从任意 CWD 调用）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 收集 venv 内所有 nvidia 运行库目录（cu13/cudnn 等），覆盖不同 Python 版本
NV_LIBS="$(find "$ROOT/.venv" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: - 2>/dev/null || true)"

if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"
fi

exec uv run --no-sync "$@"
