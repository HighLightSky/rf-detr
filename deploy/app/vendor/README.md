# PyTorch 后端运行时

当 YAML 中任一角色设置为 `backend: pytorch` 时，使用 `prepare_delivery_assets.py`
从当前仓库的 `src/rfdetr` 复制运行时闭包到本目录中的 `rfdetr/`。

默认提交配置全部使用 ONNX，不需要本目录的 RF-DETR 源码。
