import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestQuantizedDecomposedDequantizePerChannel(TestUtils):
    @staticmethod
    def _quant_range(quant_dtype):
        return (-2147483648, 2147483647) if quant_dtype == 'int32' else (-128, 127)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('quant_dtype', ['int8', 'int32'])
    def test_quantized_decomposed_dequantize_per_channel(self, shape, quant_dtype):
        quant_min, quant_max = self._quant_range(quant_dtype)

        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                x, scale, zero_point, 0, quant_min, quant_max, torch.float32)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_channel(
                x, scale, zero_point, 0, quant_min, quant_max, torch.float32)
            return y + y

        x = self._generate_tensor(shape, quant_dtype, min=quant_min, max=quant_max)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.dequantize_per_channel")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('quant_dtype', ['int8', 'int32'])
    def test_quantized_decomposed_dequantize_per_channel_default(self, shape, quant_dtype):
        quant_min, quant_max = self._quant_range(quant_dtype)

        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                x, scale, zero_point, 0, quant_min, quant_max, torch.float32)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                x, scale, zero_point, 0, quant_min, quant_max, torch.float32)
            return y + y

        x = self._generate_tensor(shape, quant_dtype, min=quant_min, max=quant_max)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.dequantize_per_channel")


instantiate_parametrized_tests(TestQuantizedDecomposedDequantizePerChannel)

if __name__ == "__main__":
    run_tests()