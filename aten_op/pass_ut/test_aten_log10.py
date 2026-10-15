import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestAtenlog10(TestUtils):

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_log10(self, shape, dtype):
        def op_calc(first_element):
            x = first_element
            y = torch.ops.prims.log10(x)
            result = y
            return result
        def op_calc_1(first_element):
            x = first_element + 1
            y = torch.ops.prims.log10(x)
            result = y + 1
            return result
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_log10_default(self, shape, dtype):
        def op_calc(first_element):
            x = first_element
            y = torch.ops.prims.log10.default(x)
            result = y
            return result
        def op_calc_1(first_element):
            x = first_element + 1
            y = torch.ops.prims.log10.default(x)
            result = y + 1
            return result
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
instantiate_parametrized_tests(TestAtenlog10)

if __name__ == "__main__":
    run_tests()
