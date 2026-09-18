# aten.mkldnn_rnn_layer.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.mkldnn_rnn_layer.default(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) -> (Tensor, Tensor, Tensor, Tensor)
```

### 1.2 算子功能

调用 oneDNN RNN 层，返回输出、最终隐藏态、细胞态及工作区。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `input` | `Tensor` | 输入张量。 |
| `weight0` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `weight1` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `weight2` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `weight3` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `hx_` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `cx_` | `Tensor` | 算子配置参数，取值见使用方法。 |
| `reverse` | `bool` | 算子配置参数，取值见使用方法。 |
| `batch_sizes` | `List[int]` | 算子配置参数，取值见使用方法。 |
| `mode` | `int` | 算子配置参数，取值见使用方法。 |
| `hidden_size` | `int` | 算子配置参数，取值见使用方法。 |
| `num_layers` | `int` | 算子配置参数，取值见使用方法。 |
| `has_biases` | `bool` | 算子配置参数，取值见使用方法。 |
| `bidirectional` | `bool` | 算子配置参数，取值见使用方法。 |
| `batch_first` | `bool` | 算子配置参数，取值见使用方法。 |
| `train` | `bool` | 训练模式开关。 |
| `返回值` | `Tensor, Tensor, Tensor, Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度固定为 `3`。
2. 测试 shape：`(4, 8, 16)`。

### 2.3 特殊限制说明

该接口为 CPU/MkldnnCPU 实现，GPU/NPU 均不支持。使用 LSTM mode=2 的参数布局。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @unittest.skip('mkldnn_rnn_layer 为 CPU/oneDNN 专用算子，GPU/NPU 不支持')
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mkldnn_rnn_layer_default(self, shape, dtype):

        def op_calc(first_element, weight0, weight1, weight2, weight3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train):
            y = torch.ops.aten.mkldnn_rnn_layer.default(first_element, weight0, weight1, weight2, weight3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train)
            return (y[0], y[1], y[2], y[3])

        def op_calc_1(first_element, weight0, weight1, weight2, weight3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train):
            y = torch.ops.aten.mkldnn_rnn_layer.default(first_element, weight0, weight1, weight2, weight3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train)
            y = (y[0], y[1], y[2], y[3])
            return (y[0] + y[0], *y[1:])
        hidden = 8
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        weight0 = self._generate_tensor((4 * hidden, shape[-1]), dtype, min=-2, max=2)
        weight1 = self._generate_tensor((4 * hidden, hidden), dtype, min=-2, max=2)
        weight2 = self._generate_tensor((4 * hidden,), dtype, min=-2, max=2)
        weight3 = self._generate_tensor((4 * hidden,), dtype, min=-2, max=2)
        hx = self._generate_tensor((shape[1], hidden), dtype, min=-2, max=2)
        cx = self._generate_tensor((shape[1], hidden), dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, weight0, weight1,
            weight2, weight3, hx, cx, False, (), 2, 8, 1, True, False,
            False, False,
        )
```
