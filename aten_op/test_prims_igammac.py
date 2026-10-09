import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsIgammac(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_igammac(self, shape, dtype):
        def op_calc(x, y):
            return torch.ops.prims.igammac(x, y)

        def op_calc_1(x, y):
            z = torch.ops.prims.igammac(x, y)
            return z + z

        x = self._generate_tensor(shape, dtype, min=0.5, max=10)
        y = self._generate_tensor(shape, dtype, min=0.5, max=10)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x, y,
                                   op_name="prims.igammac")


instantiate_parametrized_tests(TestPrimsIgammac)

if __name__ == "__main__":
    run_tests()