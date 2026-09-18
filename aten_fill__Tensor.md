# aten.fill_.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
```

### 1.2 算子功能

将输入 Tensor 的所有元素原地设置为 value。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `value` | `Tensor` | 填充值。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

Tensor value 为零维张量，dtype 与输入一致。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_fill__Tensor(self, shape, dtype):

        def op_calc(first_element, value):
            return torch.ops.aten.fill_.Tensor(first_element, value)

        def op_calc_1(first_element, value):
            y = torch.ops.aten.fill_.Tensor(first_element, value)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        value = self._generate_tensor((), dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, value,
        )
```
