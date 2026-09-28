# aten._foreach_div.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_div.Tensor(self: Tensor[], other: Tensor) -> Tensor[]
```

### 1.2 算子功能

将输入列表中的每个张量除以同一个 Tensor。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 被除数张量列表。 |
| `other` | `Tensor` | 作用于列表中每个元素的除数 Tensor。 |
| `返回值` | `Tensor[]` | 逐项相除后的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

注：gpu，npu eager模式均不支持该算子。
#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维；`other` 固定为 0 维 Tensor。

### 2.3 特殊限制说明

除数必须非零；整数输入产生 float32 结果。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_div_tensor(self, shape, dtype):
        def op_calc(tensors, other):
            return torch.ops.aten._foreach_div.Tensor(tensors, other)

        def op_calc_1(tensors, other):
            result = torch.ops.aten._foreach_div.Tensor(tensors, other)
            return [item + item for item in result]

        tensors = [self._generate_tensor(shape, dtype, min=1, max=8) for _ in range(2)]
        other = self._generate_tensor(shape, dtype, min=1, max=8)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, other)
```
