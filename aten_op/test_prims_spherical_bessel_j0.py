import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsSphericalBesselJ0(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_spherical_bessel_j0(self, shape, dtype):
        def op_calc(x):
            return torch.ops.prims.spherical_bessel_j0(x)

        def op_calc_1(x):
            y = torch.ops.prims.spherical_bessel_j0(x)
            return y + y

        x = self._generate_tensor(shape, dtype, min=-10, max=10)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x,
                                   op_name="prims.spherical_bessel_j0")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_float_dtypes)
    def test_prims_spherical_bessel_j0_default(self, shape, dtype):
        def op_calc(x):
            return torch.ops.prims.spherical_bessel_j0.default(x)

        def op_calc_1(x):
            y = torch.ops.prims.spherical_bessel_j0.default(x)
            return y + y

        x = self._generate_tensor(shape, dtype, min=-10, max=10)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x,
                                   op_name="prims.spherical_bessel_j0")


instantiate_parametrized_tests(TestPrimsSphericalBesselJ0)

if __name__ == "__main__":
    run_tests()