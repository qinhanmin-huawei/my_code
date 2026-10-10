import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestQuantizedDecomposedDequantizePerTensor(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_quantized_decomposed_dequantize_per_tensor_default(self, shape):
        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, scale, zero_point, -128, 127, torch.float32)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, scale, zero_point, -128, 127, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'int8', min=-128, max=127)
        scale = self._generate_tensor((), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.dequantize_per_tensor")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_quantized_decomposed_dequantize_per_tensor_tensor(self, shape):
        def op_calc(x, scale, zero_point, quant_min, quant_max):
            return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                x, scale, zero_point, quant_min, quant_max, torch.float32)

        def op_calc_1(x, scale, zero_point, quant_min, quant_max):
            y = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                x, scale, zero_point, quant_min, quant_max, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'int8', min=-128, max=127)
        scale = self._generate_tensor((), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((), 'int32', min=-10, max=10)
        quant_min = torch.tensor(-128, dtype=torch.int32, device='npu')
        quant_max = torch.tensor(127, dtype=torch.int32, device='npu')
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   quant_min, quant_max,
                                   op_name="quantized_decomposed.dequantize_per_tensor")


instantiate_parametrized_tests(TestQuantizedDecomposedDequantizePerTensor)

if __name__ == "__main__":
    run_tests()