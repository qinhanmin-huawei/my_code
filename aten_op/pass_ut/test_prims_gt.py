import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils

class TestPrimsGt(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_gt(self, shape, dtype):
        def op_calc(x, y):
            res = torch.ops.prims.gt(x, y)
            return res
        def op_calc_plus(x, y):
            res = torch.ops.prims.gt(x, y)
            return res + res
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_plus, x, y)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_gt_default(self, shape, dtype):
        def op_calc(x, y):
            res = torch.ops.prims.gt.default(x, y)
            return res
        def op_calc_plus(x, y):
            res = torch.ops.prims.gt.default(x, y)
            return res + res
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_plus, x, y)


instantiate_parametrized_tests(TestPrimsGt)

if __name__ == "__main__":
    run_tests()