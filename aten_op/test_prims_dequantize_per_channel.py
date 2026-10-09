import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsDequantizePerChannel(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_dequantize_per_channel(self, shape):
        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                x, scale, zero_point, 0, -128, 127, torch.float32)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_channel(
                x, scale, zero_point, 0, -128, 127, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'int8', min=-128, max=127)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.dequantize_per_channel")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_dequantize_per_channel_default(self, shape):
        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                x, scale, zero_point, 0, -128, 127, torch.float32)

        def op_calc_1(x, scale, zero_point):
            y = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                x, scale, zero_point, 0, -128, 127, torch.float32)
            return y + y

        x = self._generate_tensor(shape, 'int8', min=-128, max=127)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.dequantize_per_channel")


instantiate_parametrized_tests(TestPrimsDequantizePerChannel)

if __name__ == "__main__":
    run_tests()