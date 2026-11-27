#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pairwise 种子用例 P24：2×7 × NaN/±inf（2.4 高优先 ③）

OP 组合：含 NaN/±inf 输入 → sum(dim=1) + max(dim=1)
压的交互：NaN 传播语义（max 的 NaN 优先、sum 的 NaN 扩散）在融合/模板下不丢失
断言：NaN 位置 isnan/isposinf bitwise + 有限位置 allclose
来源：_inductor_Triton融合规则测试设计1123.md 2.4 高优先种子 P24
"""

import copy

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


def _assert_nan_inf_close(eager, inductor):
    """NaN/±inf 位置 bitwise 一致，有限位置 allclose。"""
    for e, i in zip(eager, inductor):
        assert torch.isnan(e).equal(torch.isnan(i)), "NaN 位置不一致"
        assert torch.isposinf(e).equal(torch.isposinf(i)), "+inf 位置不一致"
        assert torch.isneginf(e).equal(torch.isneginf(i)), "-inf 位置不一致"
        mask = torch.isfinite(e)
        torch.testing.assert_close(e[mask], i[mask], rtol=1e-5, atol=1e-5)


def _shapes():
    cand = [s for s in TestUtils._pointwise_demo_shapes if len(s) >= 2 and s[0] >= 3]
    if cand:
        return cand
    cand = [s for s in TestUtils._pointwise_demo_shapes_2 if len(s) >= 2 and s[0] >= 3]
    return cand or [(4, 8)]


class TestFusionPairwiseP24(TestUtils):

    @parametrize("shape", _shapes())
    @parametrize('dtype', ['float32'])
    def test_fusion_pairwise_P24(self, shape, dtype):
        print(f"P24 2x7 NaN/inf: shape={shape}, dtype={dtype}")

        def fusion(x):
            s = torch.ops.aten.sum.dim_IntList(x, [1], False)
            m = torch.ops.aten.max.dim(x, 1, False)[0]
            return (s, m)

        def fusion_1(x):
            s, m = fusion(x)
            return (s + s, m + m)

        def validator(eager, inductor, eager_fn=None, args=None, **kwargs):
            try:
                _assert_nan_inf_close(eager, inductor)
                return
            except AssertionError as eager_err:
                # 三方基准：eager vs inductor 不一致时，再看 inductor 是否与 CPU 一致。
                # 一致说明是 NPU eager 离群，而非 inductor 的问题。
                if eager_fn is None or args is None:
                    raise
                try:
                    cpu = eager_fn(*self.move_tensors_to_cpu(copy.deepcopy(args)))
                except Exception as cpu_err:
                    raise AssertionError(
                        f"eager vs inductor 不一致，且 CPU 参考执行失败: "
                        f"{eager_err}; cpu: {cpu_err}"
                    ) from eager_err
                try:
                    _assert_nan_inf_close(cpu, self.move_tensors_to_cpu(inductor))
                except AssertionError as cpu_err:
                    raise AssertionError(
                        f"eager vs inductor 不一致，inductor vs cpu 也不一致: {cpu_err}"
                    ) from cpu_err
                print(
                    "P24: eager vs inductor 不一致，但 inductor 与 cpu 一致，"
                    f"判定为 NPU eager 离群: {eager_err}"
                )

        x = self._generate_tensor(shape, dtype).clone()
        # 分三行注入，避免 inf 与 -inf 在同一归约里相消
        x[0, 0] = float('nan')
        x[1, 0] = float('inf')
        x[2, 0] = float('-inf')
        return self.execute_inductor_test(shape, dtype, fusion, fusion_1, x,
                                          op_name='aten.sum.dim_IntList',
                                          result_validator=validator,
                                          kernel_expect=-1)


instantiate_parametrized_tests(TestFusionPairwiseP24)

if __name__ == "__main__":
    run_tests()
