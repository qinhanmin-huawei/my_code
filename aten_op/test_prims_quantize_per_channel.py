import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsQuantizePerChannel(TestUtils):
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_quantize_per_channel(self, shape):
        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.quantize_per_channel(
                x, scale, zero_point, 0, -128, 127, torch.int8)

        def op_calc_1(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                torch.ops.quantized_decomposed.quantize_per_channel(
                    x, scale, zero_point, 0, -128, 127, torch.int8),
                scale, zero_point, 0, -128, 127, torch.float32)

        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'int8', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.quantize_per_channel")

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_prims_quantize_per_channel_default(self, shape):
        def op_calc(x, scale, zero_point):
            return torch.ops.quantized_decomposed.quantize_per_channel.default(
                x, scale, zero_point, 0, -128, 127, torch.int8)

        def op_calc_1(x, scale, zero_point):
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                torch.ops.quantized_decomposed.quantize_per_channel.default(
                    x, scale, zero_point, 0, -128, 127, torch.int8),
                scale, zero_point, 0, -128, 127, torch.float32)

        x = self._generate_tensor(shape, 'float32', min=-2, max=2)
        scale = self._generate_tensor((shape[0],), 'float32', min=0.05, max=0.5)
        zero_point = self._generate_tensor((shape[0],), 'int32', min=-10, max=10)
        self.execute_inductor_test(shape, 'int8', op_calc, op_calc_1, x, scale, zero_point,
                                   op_name="quantized_decomposed.quantize_per_channel")


instantiate_parametrized_tests(TestPrimsQuantizePerChannel)

if __name__ == "__main__":
    run_tests()