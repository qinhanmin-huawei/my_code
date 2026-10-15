import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils

class TestPrimsGe(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_ge(self, shape, dtype):
        def op_calc(x, y):
            res = torch.ops.prims.ge(x, y)
            return res
        def op_calc_plus(x, y):
            res = torch.ops.prims.ge(x, y)
            return res + res
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_plus, x, y)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_ge_default(self, shape, dtype):
        def op_calc(x, y):
            res = torch.ops.prims.ge.default(x, y)
            return res
        def op_calc_plus(x, y):
            res = torch.ops.prims.ge.default(x, y)
            return res + res
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_plus, x, y)


instantiate_parametrized_tests(TestPrimsGe)

if __name__ == "__main__":
    run_tests()