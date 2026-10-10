import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestHigherOrderWithEffects(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_higher_order_with_effects(self, shape, dtype):
        def op_calc(token, x, y):
            _, result = torch.ops.higher_order.with_effects(
                token, torch.ops.aten.mul.Tensor, x, y)
            return result

        def op_calc_1(token, x, y):
            _, result = torch.ops.higher_order.with_effects(
                token, torch.ops.aten.mul.Tensor, x, y)
            return torch.logical_or(result, result) if result.dtype == torch.bool else result + result

        token = [torch.zeros((0,), dtype=torch.int64, device='npu')]
        x = self._generate_tensor(shape, dtype, min=-2, max=2)
        y = self._generate_tensor(shape, dtype, min=-2, max=2)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, token, x, y,
                                   op_name="higher_order.with_effects")


instantiate_parametrized_tests(TestHigherOrderWithEffects)

if __name__ == "__main__":
    run_tests()