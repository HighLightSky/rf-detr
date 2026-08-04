# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""原型模式 SSCL 回调的单元测试。

不依赖 GPU 与网络，验证 RFDETRModelModule._sscl_loss_callback 的行为：
- 训练态调用后原型库被更新（预热）。
- 验证态调用不更新原型库。
- 实例模式回调保持兼容（不创建原型库、返回 loss_sscl 键）。
"""

from __future__ import annotations

import torch

from rfdetr.sscl import SSCLLoss
from rfdetr.training.module_model import RFDETRModelModule

# 5 类测试用语义相似度矩阵（与 test_sscl.py 一致）
_SEMANTIC_MATRIX = torch.tensor(
    [
        [1.0, 0.7, 0.5, 0.3, 0.1],
        [0.7, 1.0, 0.5, 0.3, 0.1],
        [0.5, 0.5, 1.0, 0.3, 0.1],
        [0.3, 0.3, 0.3, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ]
)


class _FakeCriterion:
    """镜像 RF-DETR criterion._get_src_permutation_idx 的最小实现。"""

    @staticmethod
    def _get_src_permutation_idx(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (batch_idx, src_idx)，与 criterion.py 语义一致。"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def _build_module(training: bool, prototype_mode: bool = True) -> RFDETRModelModule:
    """用 __new__ 绕过重型 __init__，构造最小可测模块。

    ``RFDETRModelModule.__new__`` 不会调用 ``nn.Module.__init__``，
    因此给 ``sscl_loss``（nn.Module 子模块）赋值需用 ``object.__setattr__``
    绕过 ``nn.Module.__setattr__`` 的"未初始化检查"；本测试不依赖子模块注册。
    """
    module = RFDETRModelModule.__new__(RFDETRModelModule)
    module.criterion = _FakeCriterion()
    object.__setattr__(
        module,
        "sscl_loss",
        SSCLLoss(
            semantic_matrix=_SEMANTIC_MATRIX,
            prototype_mode=prototype_mode,
            hidden_dim=16,
        ),
    )
    module.training = training
    return module


class TestSSCLPrototypeCallback:
    """原型模式 SSCL 回调的行为。"""

    def test_callback_updates_bank_only_in_training(self) -> None:
        """训练态调用回调后原型库被更新；验证态调用则不更新。"""
        module = _build_module(training=True)
        outputs = {"hs": torch.randn(1, 6, 16)}
        # matched labels = targets[0]["labels"][J]，J=[0,2,4] 时取到 [3,1,2]
        targets = [{"labels": torch.tensor([3, 3, 1, 1, 2, 4])}]
        indices = [(torch.tensor([0, 2, 4]), torch.tensor([0, 2, 4]))]

        result = module._sscl_loss_callback(outputs, targets, indices)
        assert "loss_sscl" in result
        assert torch.isfinite(result["loss_sscl"])
        # 训练态：原型库被更新（matched 类别 1/2/3 各建立一次）
        num_updates = module.sscl_loss.prototype_bank.num_updates
        assert int(num_updates[1].item()) >= 1
        assert int(num_updates[2].item()) >= 1
        assert int(num_updates[3].item()) >= 1

        # 验证态：再调用一次，原型库不变化
        before = module.sscl_loss.prototype_bank.num_updates.clone()
        module.training = False
        module._sscl_loss_callback(outputs, targets, indices)
        assert torch.equal(module.sscl_loss.prototype_bank.num_updates, before)

    def test_callback_instance_mode_no_bank_update(self) -> None:
        """实例模式回调不创建原型库，返回 loss_sscl 键（兼容性）。"""
        module = _build_module(training=True, prototype_mode=False)
        outputs = {"hs": torch.randn(1, 6, 16)}
        targets = [{"labels": torch.tensor([3, 3, 1, 1, 2, 4])}]
        indices = [(torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2]))]
        result = module._sscl_loss_callback(outputs, targets, indices)
        assert "loss_sscl" in result
        assert not hasattr(module.sscl_loss, "prototype_bank")
