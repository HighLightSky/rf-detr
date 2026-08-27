# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX detector 运行时包装器的单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from rfdetr.export._onnx.inference import ONNXDetector


def _build_tiny_detector_onnx(path: str) -> None:
    """构造一个动态 batch、双 rank-3 输出（dets/labels）的最小 ONNX 检测图。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2, 4, 4])
    dets_info = helper.make_tensor_value_info("dets", TensorProto.FLOAT, ["batch", 8, 4])
    labels_info = helper.make_tensor_value_info("labels", TensorProto.FLOAT, ["batch", 8, 4])

    relu = helper.make_node("Relu", ["input"], ["relu_out"], name="relu")
    dets_shape = numpy_helper.from_array(np.array([0, 8, 4], dtype=np.int64), name="dets_shape")
    labels_shape = numpy_helper.from_array(np.array([0, 8, 4], dtype=np.int64), name="labels_shape")
    dets = helper.make_node("Reshape", ["relu_out", "dets_shape"], ["dets"], name="reshape_dets")
    labels = helper.make_node("Reshape", ["relu_out", "labels_shape"], ["labels"], name="reshape_labels")

    graph = helper.make_graph(
        [relu, dets, labels],
        "tiny_detector",
        [input_info],
        [dets_info, labels_info],
        initializer=[dets_shape, labels_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


class TestONNXDetector:
    """验证 ONNX 原始输出可直接复用 PyTorch 后处理，以及 CUDA/CPU 分支选择。"""

    def test_returns_rfdetr_output_dictionary(self) -> None:
        """输出名乱序时仍应按名称恢复 boxes 和 logits。"""
        session = mock.Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[1, 3, 32, 32])]
        session.get_outputs.return_value = [
            SimpleNamespace(name="labels", shape=["batch", 1, 2]),
            SimpleNamespace(name="dets", shape=["batch", 1, 4]),
        ]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        logits = np.array([[[8.0, -8.0]]], dtype=np.float32)
        boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
        session.run.return_value = [logits, boxes]

        with mock.patch("rfdetr.export._onnx.inference._create_onnx_session", return_value=session):
            detector = ONNXDetector("model.onnx", num_select=1)

        outputs = detector(torch.zeros((1, 3, 32, 32)))

        assert torch.equal(outputs["pred_boxes"], torch.from_numpy(boxes))
        assert torch.equal(outputs["pred_logits"], torch.from_numpy(logits))
        post = detector.postprocess(outputs, target_sizes=torch.tensor([[20, 40]]))
        assert post[0]["labels"].tolist() == [0]
        assert post[0]["boxes"].tolist() == [[10.0, 5.0, 30.0, 15.0]]

    def test_cuda_batch_routes_to_iobinding(self) -> None:
        """CUDA batch + CUDA session 应走 IOBinding 零拷贝路径而非 numpy 路径。"""
        session = mock.Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", shape=["batch", 3, 32, 32])]
        session.get_outputs.return_value = [
            SimpleNamespace(name="dets", shape=["batch", 1, 4]),
            SimpleNamespace(name="labels", shape=["batch", 1, 2]),
        ]
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        io_binding = mock.Mock()
        session.io_binding.return_value = io_binding

        with mock.patch("rfdetr.export._onnx.inference._create_onnx_session", return_value=session):
            detector = ONNXDetector("model.onnx", num_select=1)

        batch = torch.zeros((1, 3, 32, 32))
        with mock.patch.object(torch.Tensor, "is_cuda", True):
            outputs = detector(batch)

        session.io_binding.assert_called_once()
        io_binding.bind_input.assert_called_once()
        assert io_binding.bind_input.call_args.args[1] == "cuda"
        assert io_binding.bind_output.call_count == 2
        assert all(call.args[1] == "cuda" for call in io_binding.bind_output.call_args_list)
        session.run_with_iobinding.assert_called_once_with(io_binding)
        session.run.assert_not_called()
        assert outputs["pred_boxes"].shape == (1, 1, 4)
        assert outputs["pred_logits"].shape == (1, 1, 2)

    def test_cpu_batch_uses_numpy_with_cuda_session(self) -> None:
        """即使 session 注册了 CUDA EP，CPU batch 仍应走 numpy 路径。"""
        session = mock.Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", shape=["batch", 3, 32, 32])]
        session.get_outputs.return_value = [
            SimpleNamespace(name="dets", shape=["batch", 1, 4]),
            SimpleNamespace(name="labels", shape=["batch", 1, 2]),
        ]
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session.run.return_value = [
            np.zeros((1, 1, 4), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
        ]

        with mock.patch("rfdetr.export._onnx.inference._create_onnx_session", return_value=session):
            detector = ONNXDetector("model.onnx", num_select=1)

        detector(torch.zeros((1, 3, 32, 32)))

        session.run.assert_called_once()
        session.io_binding.assert_not_called()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for IOBinding parity test")
    def test_iobinding_matches_numpy_path(self, tmp_path) -> None:
        """IOBinding 路径的输出应与 numpy 路径一致，且结果留在 CUDA。"""
        pytest.importorskip("onnx")
        onnx_path = tmp_path / "tiny_detector.onnx"
        _build_tiny_detector_onnx(str(onnx_path))

        detector = ONNXDetector(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        x = torch.randn(2, 2, 4, 4, device="cuda")

        out = detector(x)
        assert out["pred_boxes"].is_cuda and out["pred_logits"].is_cuda

        ref = detector.session.run(None, {detector.input_name: x.cpu().numpy()})
        ref_boxes = ref[detector._boxes_idx]
        ref_logits = ref[detector._logits_idx]

        np.testing.assert_allclose(out["pred_boxes"].cpu().numpy(), ref_boxes, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(out["pred_logits"].cpu().numpy(), ref_logits, atol=1e-5, rtol=1e-4)
