# aten.native_dropout.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.native_dropout.default(Tensor input, float p, bool? train) -> (Tensor, Tensor)
```

### 1.2 算子功能

按概率丢弃元素，返回缩放后的输出与 bool 掩码。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `input` | `Tensor` | 输入张量。 |
| `p` | `float` | 丢弃概率。 |
| `train` | `Optional[bool]` | 训练模式开关。 |
| `返回值` | `Tensor, Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

覆盖训练 p=0、p=1 和推理 p=0.5；这些路径输出确定，适配框架逐元素精度比较，不覆盖随机训练路径。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']
    dropout_params = [(0.0, True), (1.0, True), (0.5, False)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('dropout_params', dropout_params)
    def test_aten_native_dropout_default(self, shape, dtype, dropout_params):

        def op_calc(first_element, probability, train):
            y = torch.ops.aten.native_dropout.default(first_element, probability, train)
            return (y[0], y[1])

        def op_calc_1(first_element, probability, train):
            y = torch.ops.aten.native_dropout.default(first_element, probability, train)
            y = (y[0], y[1])
            return (y[0] + y[0], *y[1:])
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        probability, train = dropout_params
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, probability, train,
        )
```
