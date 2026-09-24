# aten._foreach_mul.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_mul.Tensor(self: Tensor[], other: Tensor) -> Tensor[]
```

### 1.2 算子功能

将输入列表中的每个张量与同一个张量逐元素相乘。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 输入张量列表。 |
| `other` | `Tensor` | 0 维（标量）张量，与列表中的每个张量相乘。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维；`other` 固定为 0 维 Tensor。

### 2.3 特殊限制说明

`other` 必须是 0 维（标量）张量，不支持与列表元素逐元素广播；其 dtype 与列表元素相同；bool 输入执行对应的逻辑与运算。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_mul_tensor(self, shape, dtype):
        def op_calc(tensors, other):
            return torch.ops.aten._foreach_mul.Tensor(tensors, other)

        def op_calc_1(tensors, other):
            y = torch.ops.aten._foreach_mul.Tensor(tensors, other)
            return [item + item for item in y]

        tensors = [self._generate_tensor(shape, dtype, min=-4, max=4) for _ in range(2)]
        other = self._generate_tensor((), dtype, min=-4, max=4)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, other)
```
