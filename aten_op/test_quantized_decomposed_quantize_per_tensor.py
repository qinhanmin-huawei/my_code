import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestQuantizedDecomposedQuantizePerTensor(TestUtils):
    @staticmethod
    def _quant_range(quant_dtype):
        return (-2147483648, 2147483647) if quant_dtype == 'int32' else (-128, 127)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('quant_dtype', ['int8', 'int32'])
    def test_quantized_decomposed_quantize_per_tensor_default(self, shape, quant_dtype):
        quant_min, quant_max = self._quant_range(quant_dtype)
        quant_type = getattr(torch, quant_dtype)

        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, scale, zero_point, quant_min, quant_max, quant_type)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x, scale, zero_point, quant_min, quant_max, quant_type),
                scale, zero_point, quant_min, quant_max, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        scale = self._generate_tensor((), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, quant_dtype, op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.quantize_per_tensor")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('quant_dtype', ['int8', 'int32'])
    def test_quantized_decomposed_quantize_per_tensor_tensor(self, shape, quant_dtype):
        quant_min, quant_max = self._quant_range(quant_dtype)
        quant_type = getattr(torch, quant_dtype)

        def op_calc(x, scale, zero_point, quant_min, quant_max):
            return torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
                x, scale, zero_point, quant_min, quant_max, quant_type)

        def op_calc_1(x, scale, zero_point, quant_min, quant_max):
            y = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
                    x, scale, zero_point, quant_min, quant_max, quant_type),
                scale, zero_point, quant_min, quant_max, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        scale = self._generate_tensor((), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((), 'int32', min=-10, max=10)
        quant_min = torch.tensor(quant_min, dtype=torch.int32, device='npu')
        quant_max = torch.tensor(quant_max, dtype=torch.int32, device='npu')
        self.execute_inductor_test(shape, quant_dtype, op_calc, op_calc_1, x, scale, zero_point,
                                   quant_min, quant_max,
                                   op_name="quantized_decomposed.quantize_per_tensor")


instantiate_parametrized_tests(TestQuantizedDecomposedQuantizePerTensor)

if __name__ == "__main__":
    run_tests()