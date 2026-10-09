import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestAtenAnd(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_ints_bool_dtypes)
    def test_aten___and__(self, shape, dtype):
        def op_calc(first_element, other):
            return torch.ops.aten.__and__(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.aten.__and__(first_element, other)
            return torch.logical_or(y, y) if y.dtype == torch.bool else y + y

        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        other = self._generate_tensor(shape, dtype, min=-2, max=2)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_ints_bool_dtypes)
    def test_aten___and___Scalar(self, shape, dtype):
        def op_calc(first_element, other):
            return torch.ops.aten.__and__.Scalar(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.aten.__and__.Scalar(first_element, other)
            return torch.logical_or(y, y) if y.dtype == torch.bool else y + y

        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        other = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_ints_bool_dtypes)
    def test_aten___and___Tensor(self, shape, dtype):
        def op_calc(first_element, other):
            return torch.ops.aten.__and__.Tensor(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.aten.__and__.Tensor(first_element, other)
            return torch.logical_or(y, y) if y.dtype == torch.bool else y + y

        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        other = self._generate_tensor(shape, dtype, min=-2, max=2)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other)

    def test_aten___and___bool(self):
        def op_calc(auxiliary):
            y = torch.ops.aten.__and__.bool(True, True)
            return auxiliary | y

        def op_calc_1(auxiliary):
            y = torch.ops.aten.__and__.bool(True, True)
            y = auxiliary | y
            return torch.logical_or(y, y)

        auxiliary = torch.zeros((64,), dtype=torch.bool, device='npu')
        self.execute_inductor_test((64,), 'bool', op_calc, op_calc_1, auxiliary)

    @parametrize('dtype', TestUtils._test_ints_dtypes)
    def test_aten___and___int(self, dtype):
        def op_calc(auxiliary, first_element, other):
            return auxiliary + torch.ops.aten.__and__.int(first_element, other)

        def op_calc_1(auxiliary, first_element, other):
            y = auxiliary + torch.ops.aten.__and__.int(first_element, other)
            return y + y

        first_element = torch.tensor(6, dtype=getattr(torch, dtype)).item()
        other = torch.tensor(2, dtype=getattr(torch, dtype)).item()
        auxiliary = torch.zeros((64,), dtype=torch.int64, device='npu')
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1, auxiliary, first_element, other)


instantiate_parametrized_tests(TestAtenAnd)

if __name__ == "__main__":
    run_tests()
