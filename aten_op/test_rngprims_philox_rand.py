import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestRngprimsPhiloxRand(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_rngprims_philox_rand(self, shape, dtype):
        dtype_torch = getattr(torch, dtype)

        def op_calc(seed, offset):
            result, _ = torch.ops.rngprims.philox_rand(seed, offset, shape, dtype=dtype_torch)
            return result

        def op_calc_1(seed, offset):
            result, _ = torch.ops.rngprims.philox_rand(seed, offset, shape, dtype=dtype_torch)
            return result + result

        seed = torch.tensor(2026, dtype=torch.int64, device='npu')
        offset = 0
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, seed, offset,
                                   op_name="rngprims.philox_rand")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_rngprims_philox_rand_default(self, shape, dtype):
        dtype_torch = getattr(torch, dtype)

        def op_calc(seed, offset):
            result, _ = torch.ops.rngprims.philox_rand.default(seed, offset, shape, dtype=dtype_torch)
            return result

        def op_calc_1(seed, offset):
            result, _ = torch.ops.rngprims.philox_rand.default(seed, offset, shape, dtype=dtype_torch)
            return result + result

        seed = torch.tensor(2026, dtype=torch.int64, device='npu')
        offset = 0
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, seed, offset,
                                   op_name="rngprims.philox_rand")


instantiate_parametrized_tests(TestRngprimsPhiloxRand)

if __name__ == "__main__":
    run_tests()