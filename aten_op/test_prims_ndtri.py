import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsNdtri(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_ndtri(self, shape, dtype):
        def op_calc(x):
            return torch.ops.prims.ndtri(x)

        def op_calc_1(x):
            y = torch.ops.prims.ndtri(x)
            return y + y

        x = self._generate_tensor(shape, dtype, min=0.1, max=0.9)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x,
                                   op_name="prims.ndtri")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_ndtri_default(self, shape, dtype):
        def op_calc(x):
            return torch.ops.prims.ndtri.default(x)

        def op_calc_1(x):
            y = torch.ops.prims.ndtri.default(x)
            return y + y

        x = self._generate_tensor(shape, dtype, min=0.1, max=0.9)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x,
                                   op_name="prims.ndtri")


instantiate_parametrized_tests(TestPrimsNdtri)

if __name__ == "__main__":
    run_tests()