#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用例 #11：正确性⚠️（+边界）——multi_slice_concat 模板禁融（PR 44887 场景）。

组合：cat(定宽列切片) → clamp → mm
验证：模板 kernel 源码不含 0.375/maximum（clamp 不被静默吞掉）+ vs eager 比对
运行：python test_case06_multi_slice_concat_no_fuse.py
"""

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from op_blocks import aten_cat, aten_clamp_min, aten_mm
from testutils import TestUtils
from op_dtype_support import SPEC_DTYPES, OUTPUT_DTYPE
import torch_npu


class TestFusionBase11CatClampMm(TestUtils):

    @parametrize("shape", TestUtils._pointwise_demo_shapes_mm)
    @parametrize('dtype', ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool'])
    def test_fusion_base_11_cat_clamp_mm(self, shape, dtype):
        if dtype not in SPEC_DTYPES[aten_cat.__name__]:
            return
        mid_dtype = OUTPUT_DTYPE[aten_cat.__name__][dtype]
        if mid_dtype not in SPEC_DTYPES[aten_clamp_min.__name__]:
            return
        out_dtype = OUTPUT_DTYPE[aten_clamp_min.__name__][mid_dtype]
        if out_dtype not in SPEC_DTYPES[aten_mm.__name__] or dtype not in SPEC_DTYPES[aten_mm.__name__]:
            return
        print(f"The selected atenop is {aten_cat.__name__}, "
              f"{aten_clamp_min.__name__} and {aten_mm.__name__}")

        def fusion(tensor,other):
            y=torch.ops.aten.cat([tensor[:1,:],tensor[0:,:]], 0)
            return torch.ops.aten.mm(torch.ops.aten.clamp(tensor,-750,750), other)

        def fusion_1(tensor,other):
            y=torch.ops.aten.cat([tensor[:1,:],tensor[0:,:]], 0)
            result = torch.ops.aten.mm(torch.ops.aten.clamp(tensor,-750,750), other)
            return result + result

        m,k,n=shape
        tensor = self._generate_tensor((m,k), dtype)
        other = self._generate_tensor((k,n), dtype)
        self.isMmType = True
        return self.execute_inductor_test(shape, dtype, fusion, fusion_1, tensor, other, 
                                          op_name=aten_cat.__name__)


instantiate_parametrized_tests(TestFusionBase11CatClampMm)

if __name__ == "__main__":
    run_tests()
