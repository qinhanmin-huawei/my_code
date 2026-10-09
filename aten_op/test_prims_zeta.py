import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsZeta(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_zeta(self, shape, dtype):
        def op_calc(first_element, other):
            return torch.ops.prims.zeta(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.prims.zeta(first_element, other)
            return y + y

        tensor1 = self._generate_tensor(shape, dtype, min=2, max=7)
        other = self._generate_tensor(shape, dtype, min=2, max=7)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other, op_name="prims.zeta")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_zeta_default(self, shape, dtype):
        def op_calc(first_element, other):
            return torch.ops.prims.zeta.default(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.prims.zeta.default(first_element, other)
            return y + y

        tensor1 = self._generate_tensor(shape, dtype, min=2, max=7)
        other = self._generate_tensor(shape, dtype, min=2, max=7)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other, op_name="prims.zeta")


instantiate_parametrized_tests(TestPrimsZeta)

if __name__ == "__main__":
    run_tests()