import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestHigherOrderWhileLoop(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_while_loop(self, shape):
        def op_calc(x, num_iters):
            def cond_fn(*carry):
                return torch.ops.aten.lt.Tensor(carry[1], carry[2])

            def body_fn(*carry):
                return (torch.ops.aten.add.Tensor(carry[0], carry[0]),
                        torch.ops.aten.add.Tensor(carry[1], torch.ones_like(carry[1])),
                        carry[2])

            result, _, _ = torch.ops.higher_order.while_loop(
                cond_fn, body_fn,
                (x, torch.zeros((), dtype=torch.int64, device='npu'), num_iters),
                ())
            return result

        def op_calc_1(x, num_iters):
            return op_calc(x, num_iters) + op_calc(x, num_iters)

        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        num_iters = torch.tensor(4, dtype=torch.int64, device='npu')
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, num_iters,
                                   op_name="higher_order.while_loop")


instantiate_parametrized_tests(TestHigherOrderWhileLoop)

if __name__ == "__main__":
    run_tests()