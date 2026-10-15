import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsPrepareSoftmaxOnline(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_prepare_softmax_online_default(self, shape, dtype):
        def op_calc(x):
            xmax, _ = torch.ops.prims.prepare_softmax_online(x, -1)
            return xmax

        def op_calc_1(x):
            _, xsum = torch.ops.prims.prepare_softmax_online(x, -1)
            return xsum

        x = self._generate_tensor(shape, dtype, min=-5, max=5)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x,
                                   op_name="prims.prepare_softmax_online")


instantiate_parametrized_tests(TestPrimsPrepareSoftmaxOnline)

if __name__ == "__main__":
    run_tests()