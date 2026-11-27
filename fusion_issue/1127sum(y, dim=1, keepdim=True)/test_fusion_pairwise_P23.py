#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pairwise 种子用例 P23：2×7 × size 1 广播（2.4 高优先 ②）

OP 组合：sum(y, dim=1, keepdim=True) → x + s（s 为 (N,1)）
压的交互：归约输出 size 1 广播 × 加法（广播 index 计算，#14 bias 同机制）
断言：allclose（fp16 容差，全量比对）；不钉 kernel 数
来源：_inductor_Triton融合规则测试设计1123.md 2.4 高优先种子 P23
"""

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


def _shapes(min_ndim):
    cand = [s for s in TestUtils._pointwise_demo_shapes if len(s) >= min_ndim]
    return cand or [s for s in TestUtils._pointwise_demo_shapes_2 if len(s) >= min_ndim]


class TestFusionPairwiseP23(TestUtils):

    @parametrize("shape", _shapes(2))
    @parametrize('dtype', ['float16'])
    def test_fusion_pairwise_P23(self, shape, dtype):
        print(f"P23 2x7 size-1 broadcast: shape={shape}, dtype={dtype}")

        def fusion(x):
            # (N,1,...) 的 size 1 维与 x 广播
            s = torch.ops.aten.sum.dim_IntList(x, [1], True)
            return torch.ops.aten.add.Tensor(x, s)

        def fusion_1(x):
            result = fusion(x)
            return result + result

        x = self._generate_tensor(shape, dtype)
        return self.execute_inductor_test(shape, dtype, fusion, fusion_1, x,
                                          op_name='aten.sum.dim_IntList', kernel_expect=-1)


instantiate_parametrized_tests(TestFusionPairwiseP23)

if __name__ == "__main__":
    run_tests()
