import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestHigherOrderWithEffects(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_with_effects(self, shape):
        def op_calc(token, x, y):
            _, result = torch.ops.higher_order.with_effects(
                token, torch.ops.aten.mul.Tensor, x, y)
            return result

        def op_calc_1(token, x, y):
            _, result = torch.ops.higher_order.with_effects(
                token, torch.ops.aten.mul.Tensor, x, y)
            return result + result

        token = [torch.zeros((0,), dtype=torch.int64, device='npu')]
        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        y = self._generate_tensor(shape, 'float32', min=-2, max=2)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, token, x, y,
                                   op_name="higher_order.with_effects")


instantiate_parametrized_tests(TestHigherOrderWithEffects)

if __name__ == "__main__":
    run_tests()